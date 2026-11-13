"""Local full-text index of what is written inside files (SQLite FTS5).

Concept extraction gives the graph its structure; full-text search makes sure a
word that appears in a document is always findable, even before (or without)
the LLM content pass. Text comes from the same converters graphify uses (Office
→ markdown, PDF text, transcripts). Accent- and case-insensitive. Cached per
file by size + mtime, so a re-index only re-reads changed files.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path

from graphify.fs.metadata import ScanResult

MAX_TEXT = 400_000          # characters kept per file
MAX_READ_BYTES = 40 * 1024 * 1024
TEXT_KINDS = {"text", "code", "web", "email"}
CONVERTED_EXTS = {".docx", ".xlsx", ".pptx", ".pdf"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(path TEXT PRIMARY KEY, sig TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
    path UNINDEXED, name, body, tokenize = 'unicode61 remove_diacritics 2'
);
"""


def _file_text(path: Path, ext: str, kind: str) -> str:
    from graphify import detect

    try:
        if path.stat().st_size > MAX_READ_BYTES:
            return ""
        if ext == ".docx":
            return detect.docx_to_markdown(path)
        if ext == ".xlsx":
            return detect.xlsx_to_markdown(path)
        if ext == ".pptx":
            return detect.pptx_to_markdown(path)
        if ext == ".pdf":
            return detect.extract_pdf_text(path)
        if kind in TEXT_KINDS:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ext in (".html", ".htm"):
                text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S | re.I)
                text = re.sub(r"<[^>]+>", " ", text)
            return text
    except Exception:
        return ""
    return ""


def _fts_query(query: str) -> str | None:
    words = [w for w in re.findall(r"[\w]+", query, re.UNICODE) if len(w) > 1]
    if not words:
        return None
    # Each word as a prefix term; all words should match (AND), with an OR fallback in search().
    return " ".join(f'"{w}"*' for w in words[:12])


class FullTextIndex:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn = conn
        return conn

    def update(self, scan: ScanResult, media_text: dict[str, Path] | None = None, progress=None) -> int:
        """Bring the index in line with the scan; returns the number of files (re)read."""
        c = self._conn()
        known = dict(c.execute("SELECT path, sig FROM meta"))
        live: set[str] = set()
        changed = 0
        for rec in scan.files:
            if rec.offline or rec.is_package:
                continue
            transcript = (media_text or {}).get(rec.rel)
            if not transcript and rec.ext not in CONVERTED_EXTS and rec.kind not in TEXT_KINDS:
                continue
            live.add(rec.rel)
            sig = f"{rec.size}:{int(rec.mtime)}:{1 if transcript else 0}"
            if known.get(rec.rel) == sig:
                continue
            if transcript:
                try:
                    text = Path(transcript).read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    text = ""
            else:
                text = _file_text(scan.root / rec.rel, rec.ext, rec.kind)
            c.execute("DELETE FROM docs WHERE path = ?", (rec.rel,))
            c.execute("INSERT INTO docs(path, name, body) VALUES (?, ?, ?)", (rec.rel, rec.name, text[:MAX_TEXT]))
            c.execute("INSERT OR REPLACE INTO meta(path, sig) VALUES (?, ?)", (rec.rel, sig))
            changed += 1
            if progress and changed % 100 == 0:
                progress(changed)
                c.commit()
        for gone in set(known) - live:
            c.execute("DELETE FROM docs WHERE path = ?", (gone,))
            c.execute("DELETE FROM meta WHERE path = ?", (gone,))
        c.commit()
        return changed

    def search(self, query: str, limit: int = 30) -> list[dict]:
        """Ranked matches: ``[{path, score, snippet}]`` (score > 0, higher is better)."""
        q = _fts_query(query)
        if not q:
            return []
        c = self._conn()
        sql = ("SELECT path, bm25(docs, 0.0, 2.0, 1.0) AS rank, "
               "snippet(docs, 2, '«', '»', ' … ', 18) FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?")
        try:
            rows = c.execute(sql, (q, limit)).fetchall()
            if not rows and " " in q:
                rows = c.execute(sql, (q.replace('* "', '* OR "'), limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"path": p, "score": round(-r, 3), "snippet": " ".join(s.split())} for p, r, s in rows]

    def text(self, rel: str) -> str:
        row = self._conn().execute("SELECT body FROM docs WHERE path = ?", (rel,)).fetchone()
        return row[0] if row else ""

    def passages(self, rel: str, query: str | None = None, budget: int = 3500) -> list[str]:
        """The most relevant passages of a file for ``query`` (or its beginning)."""
        text = self.text(rel)
        if not text:
            return []
        if not query:
            return [text[:budget]]
        chunks = [text[i:i + 700] for i in range(0, len(text), 600)]
        terms = {w.casefold() for w in re.findall(r"\w+", query) if len(w) > 2}
        if not terms:
            return [text[:budget]]
        scored = sorted(((sum(ch.casefold().count(t) for t in terms), i) for i, ch in enumerate(chunks)), reverse=True)
        picked = sorted(i for s, i in scored[: max(1, budget // 700)] if s > 0) or [0]
        return [chunks[i] for i in picked]

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
