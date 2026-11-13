"""Query tools over an fs graph, shared by the MCP server, the chat agent and the UI.

Every tool returns small JSON-serialisable dicts, so a small local model can read
the results. File arguments accept a node id (``file:a/b.pdf``), a path relative
to the indexed root, or just a file name.
"""
from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx

from graphify.fs.config import FsIndexPaths, IndexState, load_state
from graphify.fs.metadata import KIND_LABELS, file_id, folder_id
from graphify.fs.tags import fold, tag_key


class ToolError(ValueError):
    pass


def _norm(text: str) -> str:
    return fold(str(text or "")).casefold()


_WORD = re.compile(r"[\w#+]+", re.UNICODE)
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "about", "is", "are", "what",
    "which", "where", "who", "how", "my", "me", "i", "files", "file", "documents", "document", "show",
    "find", "all", "any", "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "sur", "pour",
    "avec", "dans", "mes", "mon", "ma", "quels", "quelles", "quel", "quelle", "fichiers", "fichier", "trouve",
    "montre", "qui", "que", "est", "sont", "au", "aux", "en", "par",
}


def query_terms(query: str) -> list[str]:
    words = [w for w in _WORD.findall(_norm(query)) if len(w) > 1]
    content = [w for w in words if w not in _STOP]
    return content or words


@dataclass
class FileBrief:
    id: str
    name: str
    path: str
    kind: str
    modified: str | None
    tags: list[str]
    topic: str | None
    offline: bool

    def as_dict(self) -> dict:
        d = {"id": self.id, "name": self.name, "path": self.path, "kind": self.kind,
             "modified": (self.modified or "")[:10] or None, "tags": self.tags[:6]}
        if self.topic:
            d["topic"] = self.topic
        if self.offline:
            d["offline"] = True
        return d


class FsIndex:
    """A loaded fs graph plus lookup indexes."""

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.paths = FsIndexPaths(self.index_dir)
        state = load_state(self.index_dir)
        if state is None:
            raise FileNotFoundError(f"not a graphify fs index: {self.index_dir}")
        self.state: IndexState = state
        self.root = Path(state.root)
        self._mtime = 0.0
        self.lock = threading.RLock()
        self.reload()

    # ------------------------------------------------------------------ load
    def reload(self) -> None:
        from graphify.serve import _read_graph

        if not self.paths.graph.exists():
            raise FileNotFoundError(f"index not built yet: {self.paths.graph}")
        mtime = self.paths.graph.stat().st_mtime
        G = _read_graph(str(self.paths.graph))
        for key in [k for k in G.graph if str(k).startswith("_")]:
            G.graph.pop(key, None)
        # Build every lookup table first and swap them in at the end, so a request
        # served while a re-index lands never sees half-built state.
        files: list[str] = []
        folders: list[str] = []
        tags: list[str] = []
        topic_names: dict[int, str] = {}
        by_tag: dict[str, set[str]] = defaultdict(set)
        by_topic: dict[int, set[str]] = defaultdict(set)
        by_name: dict[str, list[str]] = defaultdict(list)
        text: dict[str, dict[str, str]] = {}
        for nid, d in G.nodes(data=True):
            kind = d.get("fs_kind")
            cid = d.get("community")
            if isinstance(cid, int) and d.get("community_name"):
                topic_names.setdefault(cid, d["community_name"])
            if kind == "file":
                files.append(nid)
                by_name[_norm(d.get("label"))].append(nid)
                topics = d.get("topics") or ([cid] if isinstance(cid, int) else [])
                for c in topics[:1]:
                    by_topic[c].add(nid)
                text[nid] = {
                    "name": _norm(d.get("label")),
                    "path": _norm(d.get("path")),
                    "title": _norm(d.get("title")),
                    "tags": " | ".join(_norm(t) for t in d.get("tags", [])),
                    "keywords": " | ".join(_norm(t) for t in d.get("keywords", [])),
                }
            elif kind == "folder":
                folders.append(nid)
            elif kind == "tag":
                tags.append(nid)
        concept_files: dict[str, int] = {}
        for u, v, d in G.edges(data=True):
            rel = d.get("relation")
            if rel == "tagged_with" and str(u).startswith("file:"):
                by_tag[v].add(u)
            elif rel == "mentions":
                concept_files[v] = concept_files.get(v, 0) + 1
        fulltext = None
        if self.paths.fulltext.exists():
            from graphify.fs.fulltext import FullTextIndex

            fulltext = FullTextIndex(self.paths.fulltext)
        self.__dict__.update(
            G=G, _mtime=mtime, state=load_state(self.index_dir) or self.state,
            files=files, folders=folders, tags=tags, topic_names=topic_names, by_tag=by_tag,
            by_topic=by_topic, by_name=by_name, _text=text, _concept_files=concept_files,
            _topic_edges=None, fulltext=fulltext,
        )

    def maybe_reload(self) -> bool:
        try:
            if self.paths.graph.stat().st_mtime != self._mtime:
                with self.lock:
                    self.reload()
                return True
        except OSError:
            pass
        return False

    def apply_tag_change(self, fid: str, name: str, action: str, source: str = "user") -> dict:
        """Mirror a tag decision in the loaded graph (persisted by the next rebuild)."""
        with self.lock:
            G = self.G
            tid = self.resolve_tag(name)
            if action in ("accept", "add"):
                if tid is None:
                    key = tag_key(name)
                    tid = f"tag:{key}"
                    G.add_node(tid, label=name.strip(), fs_kind="tag", kind="tag", file_type="concept",
                               source_file="", tag_key=key, tag_sources=[source], color=None, color_index=0,
                               file_count=0)
                    self.tags.append(tid)
                if G.has_edge(fid, tid):
                    e = G.edges[fid, tid]
                    e["status"] = "accepted"
                    e["sources"] = sorted(set(e.get("sources", [])) | {source})
                    e["confidence"] = "EXTRACTED"
                else:
                    G.add_edge(fid, tid, relation="tagged_with", confidence="EXTRACTED", confidence_score=1.0,
                               sources=[source], status="accepted")
                self.by_tag[tid].add(fid)
            elif action in ("reject", "remove") and tid is not None and G.has_edge(fid, tid):
                G.remove_edge(fid, tid)
                self.by_tag[tid].discard(fid)
            if tid is not None and tid in G:
                G.nodes[tid]["file_count"] = len(self.by_tag.get(tid, ()))
            d = G.nodes[fid]
            d["tags"] = sorted((G.nodes[t].get("label") for t in G.successors(fid)
                                if G.edges[fid, t].get("relation") == "tagged_with"), key=str.casefold)
            self._text[fid]["tags"] = " | ".join(_norm(t) for t in d["tags"])
            return {"tags": self._file_tags(fid)}

    # --------------------------------------------------------------- helpers
    def _mentions(self, fid: str) -> list[str]:
        G = self.G
        return [v for v in G.successors(fid) if G.edges[fid, v].get("relation") == "mentions"]

    def _file_tags(self, fid: str) -> list[dict]:
        G = self.G
        out = []
        for v in G.successors(fid):
            e = G.edges[fid, v]
            if e.get("relation") == "tagged_with":
                n = G.nodes[v]
                out.append({"name": n.get("label"), "id": v, "status": e.get("status", "accepted"),
                            "sources": e.get("sources", []), "color": n.get("color")})
        out.sort(key=lambda t: (t["status"] != "accepted", str(t["name"]).casefold()))
        return out

    def topic_name(self, cid) -> str | None:
        return self.topic_names.get(cid) if isinstance(cid, int) else None

    def _file_topic(self, d: dict):
        topics = d.get("topics") or ([d["community"]] if isinstance(d.get("community"), int) else [])
        return topics[0] if topics else None

    def brief(self, fid: str) -> dict:
        d = self.G.nodes[fid]
        tags = [t["name"] for t in self._file_tags(fid) if t["status"] == "accepted"] or list(d.get("tags", []))
        return FileBrief(fid, d.get("label", ""), d.get("path", ""), d.get("kind", "other"),
                         d.get("modified"), tags, self.topic_name(self._file_topic(d)),
                         bool(d.get("offline"))).as_dict()

    def resolve_file(self, ref: str) -> str:
        if not ref:
            raise ToolError("a file is required")
        ref = str(ref).strip()
        G = self.G
        if ref in G and G.nodes[ref].get("fs_kind") == "file":
            return ref
        rel = ref[len("file:"):] if ref.startswith("file:") else ref
        rel = rel.strip("/")
        for cand in (file_id(rel), file_id(_strip_root(rel, self.root))):
            if cand in G:
                return cand
        hits = self.by_name.get(_norm(Path(rel).name)) or []
        if len(hits) == 1:
            return hits[0]
        if hits:
            return sorted(hits, key=lambda f: -float(G.nodes[f].get("mtime") or 0))[0]
        # Last resort: a file whose name/path contains every word of the reference.
        # Strict on purpose: a loose guess would hand a small model the wrong file.
        terms = query_terms(Path(rel).stem.replace("_", " "))
        for _score, fid in self._rank_files(terms, limit=5):
            txt = self._text[fid]["name"] + " " + self._text[fid]["path"]
            if terms and all(re.search(rf"\b{re.escape(t)}", txt) for t in terms):
                return fid
        raise ToolError(f"no file matches {ref!r}")

    def resolve_tag(self, ref: str) -> str | None:
        if not ref:
            return None
        if ref.startswith("tag:") and ref in self.G:
            return ref
        key = tag_key(ref)
        if f"tag:{key}" in self.G:
            return f"tag:{key}"
        for t in self.tags:
            if tag_key(self.G.nodes[t].get("label", "")) == key:
                return t
        if key.endswith("s") and f"tag:{key[:-1]}" in self.G:
            return f"tag:{key[:-1]}"
        return None

    def resolve_topic(self, ref) -> int | None:
        if ref is None or ref == "":
            return None
        if isinstance(ref, int) or str(ref).lstrip("-").isdigit():
            cid = int(ref)
            return cid if cid in self.topic_names or cid in self.by_topic else None
        want = _norm(str(ref).removeprefix("topic:"))
        if want.lstrip("-").isdigit():
            return int(want)
        for cid, name in self.topic_names.items():
            if _norm(name) == want:
                return cid
        for cid, name in self.topic_names.items():
            if want in _norm(name):
                return cid
        return None

    def resolve_folder(self, ref: str | None) -> str:
        if not ref or ref in ("/", ".", "dir:"):
            return folder_id("")
        if ref in self.G and self.G.nodes[ref].get("fs_kind") == "folder":
            return ref
        rel = ref.removeprefix("dir:").strip("/")
        rel = _strip_root(rel, self.root)
        if folder_id(rel) in self.G:
            return folder_id(rel)
        want = _norm(Path(rel).name)
        for f in self.folders:
            if _norm(self.G.nodes[f].get("label")) == want:
                return f
        raise ToolError(f"no folder matches {ref!r}")

    # ------------------------------------------------------------- ranking
    def _rank_files(self, terms: list[str], limit: int = 20, kind: str | None = None,
                    concept_boost: dict[str, float] | None = None) -> list[tuple[float, str]]:
        if not terms and not concept_boost:
            return []
        weights = {"name": 3.0, "title": 2.5, "tags": 2.5, "keywords": 2.0, "path": 1.0}
        scores: dict[str, float] = defaultdict(float)
        n_files = max(len(self.files), 1)
        df = {t: sum(1 for f in self.files if any(t in txt for txt in self._text[f].values())) for t in terms}
        for fid in self.files:
            if kind and self.G.nodes[fid].get("kind") != kind:
                continue
            txt = self._text[fid]
            s = 0.0
            matched = 0
            for t in terms:
                idf = math.log(1 + n_files / (1 + df.get(t, 0)))
                best = max((w for field, w in weights.items() if t in txt[field]), default=0.0)
                if best:
                    matched += 1
                    s += best * idf
            if matched:
                s *= (matched / len(terms)) ** 1.5
                scores[fid] += s
        for fid, boost in (concept_boost or {}).items():
            if kind and self.G.nodes[fid].get("kind") != kind:
                continue
            scores[fid] += boost
        ranked = sorted(((s, f) for f, s in scores.items() if s > 0), key=lambda x: (-x[0], x[1]))
        return ranked[:limit]

    def _concept_hits(self, query: str, limit: int = 25) -> list[tuple[float, str]]:
        try:
            from graphify.serve import _query_terms, _score_nodes

            ranked = _score_nodes(self.G, _query_terms(query))
        except Exception:
            return []
        out = []
        for score, nid in ranked:
            if self.G.nodes[nid].get("fs_kind"):
                continue
            out.append((score, nid))
            if len(out) >= limit:
                break
        return out

    # ---------------------------------------------------------------- tools
    def overview(self) -> dict:
        kinds = Counter(self.G.nodes[f].get("kind", "other") for f in self.files)
        concepts = sum(1 for _n, d in self.G.nodes(data=True) if not d.get("fs_kind"))
        top_tags = sorted(self.tags, key=lambda t: -len(self.by_tag.get(t, ())))[:20]
        top_topics = sorted(self.by_topic, key=lambda c: -len(self.by_topic[c]))[:15]
        return {
            "root": str(self.root),
            "files": len(self.files),
            "folders": len(self.folders),
            "tags": len(self.tags),
            "concepts": concepts,
            "topics": len(self.topic_names),
            "kinds": {KIND_LABELS.get(k, k): n for k, n in kinds.most_common()},
            "offline_files": sum(1 for f in self.files if self.G.nodes[f].get("offline")),
            "top_tags": [{"name": self.G.nodes[t]["label"], "files": len(self.by_tag.get(t, ()))} for t in top_tags],
            "top_topics": [{"id": c, "name": self.topic_name(c), "files": len(self.by_topic[c])} for c in top_topics],
            "last_indexed": _fmt_ts(self.state.last_index),
        }

    def search(self, query: str, limit: int = 10, kind: str | None = None) -> dict:
        """Files, concepts and tags matching a free-text query (name, tags, content)."""
        terms = query_terms(query)
        concepts = self._concept_hits(query)
        boost: dict[str, float] = defaultdict(float)
        top = concepts[0][0] if concepts else 1.0
        for score, nid in concepts:
            share = 1.0 / math.sqrt(max(self._concept_files.get(nid, 1), 1))
            for fid in self.G.predecessors(nid):
                if self.G.edges[fid, nid].get("relation") == "mentions":
                    boost[fid] += 4.0 * (score / top) * share
        tag_hits = []
        for t in self.tags:
            label = _norm(self.G.nodes[t].get("label"))
            if terms and all(term in label for term in terms) or _norm(query) == label:
                tag_hits.append(t)
                for fid in self.by_tag.get(t, ()):
                    boost[fid] += 3.0
        snippets: dict[str, str] = {}
        if self.fulltext is not None:
            hits = self.fulltext.search(query, limit=60)
            top_ft = hits[0]["score"] if hits else 1.0
            for h in hits:
                fid = file_id(h["path"])
                if fid in self.G:
                    boost[fid] += 3.0 * (h["score"] / top_ft if top_ft > 0 else 1.0)
                    snippets[fid] = h["snippet"]
        ranked = self._rank_files(terms, limit=limit, kind=kind, concept_boost=boost)
        files = []
        for _s, f in ranked:
            b = self.brief(f)
            if f in snippets:
                b["snippet"] = snippets[f][:240]
            files.append(b)
        return {
            "query": query,
            "files": files,
            "concepts": [
                {"id": nid, "label": self.G.nodes[nid].get("label", nid),
                 "files": self._concept_files.get(nid, 0),
                 "topic": self.topic_name(self.G.nodes[nid].get("community"))}
                for _s, nid in concepts[:8]
            ],
            "tags": [{"name": self.G.nodes[t]["label"], "files": len(self.by_tag.get(t, ()))} for t in tag_hits[:8]],
        }

    def find_files(self, tag: str | None = None, kind: str | None = None, folder: str | None = None,
                   topic=None, name: str | None = None, modified_after: str | None = None,
                   modified_before: str | None = None, sort: str = "modified", limit: int = 25,
                   offset: int = 0) -> dict:
        """Filter files by tag, kind, folder (recursive), topic, name and modification date."""
        candidates: set[str] | None = None
        if tag:
            tid = self.resolve_tag(tag)
            if tid is None:
                return {"files": [], "total": 0, "note": f"no tag named {tag!r}"}
            candidates = set(self.by_tag.get(tid, ()))
        if topic not in (None, ""):
            cid = self.resolve_topic(topic)
            if cid is None:
                return {"files": [], "total": 0, "note": f"no topic named {topic!r}"}
            pool = self.by_topic.get(cid, set())
            candidates = pool if candidates is None else candidates & pool
        pool_list = candidates if candidates is not None else self.files
        prefix = None
        if folder:
            fold_id = self.resolve_folder(folder)
            rel = self.G.nodes[fold_id].get("path", "")
            prefix = rel + "/" if rel else ""
        after = _parse_date(modified_after)
        before = _parse_date(modified_before)
        want_name = _norm(name) if name else None
        kinds = _kind_aliases(kind)
        out = []
        for fid in pool_list:
            d = self.G.nodes[fid]
            if kinds and d.get("kind") not in kinds:
                continue
            if prefix is not None and not str(d.get("path", "")).startswith(prefix):
                continue
            mt = float(d.get("mtime") or 0)
            if after and mt < after:
                continue
            if before and mt > before:
                continue
            if want_name and want_name not in self._text[fid]["name"]:
                continue
            out.append(fid)
        if sort == "name":
            out.sort(key=lambda f: _norm(self.G.nodes[f].get("label")))
        elif sort == "size":
            out.sort(key=lambda f: -int(self.G.nodes[f].get("size") or 0))
        else:
            out.sort(key=lambda f: -float(self.G.nodes[f].get("mtime") or 0))
        limit = max(1, min(int(limit or 25), 500))
        return {"total": len(out), "files": [self.brief(f) for f in out[offset:offset + limit]]}

    def list_folder(self, path: str | None = None, limit: int = 100) -> dict:
        """Sub-folders and files directly inside a folder ('' = indexed root)."""
        fid = self.resolve_folder(path)
        G = self.G
        subs, files = [], []
        for v in G.successors(fid):
            if G.edges[fid, v].get("relation") != "contains":
                continue
            d = G.nodes[v]
            if d.get("fs_kind") == "folder":
                subs.append({"id": v, "name": d.get("label"), "path": d.get("path"),
                             "files": d.get("file_count", 0), "tags": d.get("tags", [])})
            elif d.get("fs_kind") == "file":
                files.append(v)
        subs.sort(key=lambda s: _norm(s["name"]))
        files.sort(key=lambda f: _norm(G.nodes[f].get("label")))
        d = G.nodes[fid]
        parent = None
        rel = d.get("path", "")
        if rel:
            parent = folder_id(rel.rsplit("/", 1)[0] if "/" in rel else "")
        return {"id": fid, "name": d.get("label"), "path": rel, "parent": parent,
                "folders": subs, "files": [self.brief(f) for f in files[:limit]], "total_files": len(files)}

    def list_tags(self, query: str | None = None, limit: int = 50, include_suggested: bool = True) -> dict:
        """Tags with file counts, sources and their most related tags."""
        G = self.G
        want = _norm(query) if query else None
        rows = []
        for t in self.tags:
            d = G.nodes[t]
            if want and want not in _norm(d.get("label")):
                continue
            files = self.by_tag.get(t, set())
            accepted = sum(1 for f in files if G.edges[f, t].get("status") == "accepted")
            if not include_suggested and not accepted:
                continue
            rows.append((len(files), t, accepted))
        rows.sort(key=lambda r: (-r[0], _norm(G.nodes[r[1]].get("label"))))
        out = []
        for n, t, accepted in rows[:limit]:
            d = G.nodes[t]
            out.append({"id": t, "name": d.get("label"), "files": n, "accepted_files": accepted,
                        "sources": d.get("tag_sources", []), "color": d.get("color"),
                        "topic": self.topic_name((d.get("topics") or [None])[0]),
                        "related": self._related_tags(t, 4)})
        return {"total": len(rows), "tags": out}

    def _related_tags(self, tid: str, n: int = 4) -> list[str]:
        G = self.G
        related = sorted(
            ((G.edges[a, b].get("weight", 0), b if a == tid else a)
             for a, b in list(G.out_edges(tid)) + list(G.in_edges(tid))
             if G.edges[a, b].get("relation") == "co_occurs_with"),
            reverse=True,
        )[:n]
        return [G.nodes[r].get("label") for _w, r in related]

    def list_topics(self, limit: int = 40) -> dict:
        """Topics (communities of related concepts/files) with sizes and top tags."""
        G = self.G
        rows = []
        for cid, name in self.topic_names.items():
            files = self.by_topic.get(cid, set())
            tag_counts = Counter()
            for f in files:
                for t in G.successors(f):
                    if G.edges[f, t].get("relation") == "tagged_with":
                        tag_counts[G.nodes[t].get("label")] += 1
            rows.append({"id": cid, "name": name, "files": len(files),
                         "top_tags": [t for t, _n in tag_counts.most_common(5)]})
        rows.sort(key=lambda r: (-r["files"], r["name"] or ""))
        return {"total": len(rows), "topics": rows[:limit]}

    def get_file(self, file: str) -> dict:
        """Everything known about one file: metadata, tags, topic, concepts, related files."""
        fid = self.resolve_file(file)
        G = self.G
        d = G.nodes[fid]
        concepts = sorted(self._mentions(fid), key=lambda n: (-G.degree(n), str(n)))
        topics = d.get("topics") or ([d["community"]] if isinstance(d.get("community"), int) else [])
        info = {
            "id": fid, "name": d.get("label"), "path": d.get("path"), "kind": d.get("kind"),
            "size": d.get("size"), "modified": d.get("modified"), "created": d.get("created"),
            "title": d.get("title"), "author": d.get("author"), "pages": d.get("pages"),
            "offline": bool(d.get("offline")), "folder": folder_id(d.get("path", "").rsplit("/", 1)[0] if "/" in d.get("path", "") else ""),
            "tags": self._file_tags(fid),
            "topics": [{"id": c, "name": self.topic_name(c)} for c in topics],
            "concepts": [{"id": c, "label": G.nodes[c].get("label", c)} for c in concepts[:20]],
            "concept_count": len(concepts),
            "related": self.related_files(fid, limit=6)["files"],
        }
        return {k: v for k, v in info.items() if v not in (None, [], "")}

    def read_file(self, file: str, query: str | None = None) -> dict:
        """The text inside a file (extracted), or its passages most relevant to ``query``."""
        fid = self.resolve_file(file)
        d = self.G.nodes[fid]
        if d.get("offline"):
            return {"id": fid, "path": d.get("path"), "note": "file is only in iCloud (not downloaded)"}
        passages = self.fulltext.passages(d.get("path", ""), query) if self.fulltext else []
        if not passages:
            return {"id": fid, "path": d.get("path"), "note": "no readable text for this file",
                    "concepts": [self.G.nodes[c].get("label", c) for c in self._mentions(fid)[:20]]}
        return {"id": fid, "path": d.get("path"), "name": d.get("label"), "passages": passages}

    def related_files(self, file: str, limit: int = 10) -> dict:
        """Files related to a file through shared concepts, tags, topic and folder."""
        fid = self.resolve_file(file)
        G = self.G
        d = G.nodes[fid]
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, Counter] = defaultdict(Counter)
        for c in self._mentions(fid):
            n = self._concept_files.get(c, 1)
            if n > 200:
                continue
            w = 1.0 / math.log(1 + n)
            for other in G.predecessors(c):
                if other != fid and G.edges[other, c].get("relation") == "mentions":
                    scores[other] += w
                    reasons[other]["concepts"] += 1
        n_files = max(len(self.files), 2)
        for t in G.successors(fid):
            if G.edges[fid, t].get("relation") != "tagged_with":
                continue
            members = self.by_tag.get(t, set())
            if len(members) > n_files * 0.3:
                continue
            w = 1.5 * math.log(n_files / max(len(members), 1)) / math.log(n_files)
            for other in members:
                if other != fid:
                    scores[other] += w
                    reasons[other]["tags"] += 1
        topic = self._file_topic(d)
        folder = d.get("path", "").rsplit("/", 1)[0] if "/" in d.get("path", "") else ""
        for other in list(scores):
            od = G.nodes[other]
            if topic is not None and self._file_topic(od) == topic:
                scores[other] += 0.4
            if (od.get("path", "").rsplit("/", 1)[0] if "/" in od.get("path", "") else "") == folder:
                scores[other] += 0.3
                reasons[other]["same folder"] += 1
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        out = []
        for other, s in ranked:
            b = self.brief(other)
            r = reasons[other]
            why = []
            if r["concepts"]:
                why.append(f"{r['concepts']} shared concept(s)")
            if r["tags"]:
                why.append(f"{r['tags']} shared tag(s)")
            if r["same folder"]:
                why.append("same folder")
            b["why"] = ", ".join(why)
            b["score"] = round(s, 2)
            out.append(b)
        return {"file": fid, "files": out}

    def explain(self, name: str) -> dict:
        """A concept, tag or topic: what it connects to and which files mention it."""
        G = self.G
        tid = self.resolve_tag(name)
        if tid:
            d = G.nodes[tid]
            files = sorted(self.by_tag.get(tid, ()), key=lambda f: -float(G.nodes[f].get("mtime") or 0))
            return {"type": "tag", "id": tid, "name": d.get("label"), "files": len(files),
                    "sources": d.get("tag_sources", []),
                    "related_tags": self._related_tags(tid, 6),
                    "examples": [self.brief(f) for f in files[:10]]}
        cid = self.resolve_topic(name) if not str(name).strip().isdigit() else None
        if cid is not None and _norm(self.topic_names.get(cid)) == _norm(name):
            topic = next((t for t in self.list_topics(limit=10_000)["topics"] if t["id"] == cid), None)
            files = sorted(self.by_topic.get(cid, ()), key=lambda f: -float(G.nodes[f].get("mtime") or 0))
            return {"type": "topic", **(topic or {"id": cid, "name": self.topic_names.get(cid)}),
                    "examples": [self.brief(f) for f in files[:10]]}
        hits = self._concept_hits(name, limit=1)
        if not hits:
            raise ToolError(f"nothing in the graph matches {name!r}")
        nid = hits[0][1]
        d = G.nodes[nid]
        neigh = []
        for v in nx.all_neighbors(G, nid):
            vd = G.nodes[v]
            if vd.get("fs_kind"):
                continue
            e = G.edges[nid, v] if G.has_edge(nid, v) else G.edges[v, nid]
            neigh.append((G.degree(v), vd.get("label", v), e.get("relation", "related")))
        neigh.sort(reverse=True)
        files = [f for f in G.predecessors(nid) if G.edges[f, nid].get("relation") == "mentions"]
        return {"type": "concept", "id": nid, "name": d.get("label", nid),
                "topic": self.topic_name(d.get("community")),
                "description": d.get("rationale") or d.get("description"),
                "connections": [{"label": lbl, "relation": rel} for _deg, lbl, rel in neigh[:15]],
                "files": [self.brief(f) for f in files[:10]], "file_count": len(files)}

    def connection(self, a: str, b: str) -> dict:
        """How two things (files, tags, concepts) are connected: the shortest path."""
        src, dst = self._resolve_any(a), self._resolve_any(b)
        G = self.G
        H = nx.Graph()
        for u, v, d in G.edges(data=True):
            if d.get("relation") == "contains":
                continue
            H.add_edge(u, v, relation=d.get("relation", "related"))
        if src not in H or dst not in H:
            return {"found": False, "note": "one of them has no connections"}
        try:
            path = nx.shortest_path(H, src, dst)
        except nx.NetworkXNoPath:
            return {"found": False, "note": "no connection found"}
        hops = []
        for x, y in zip(path, path[1:]):
            hops.append({"from": G.nodes[x].get("label", x), "relation": H.edges[x, y]["relation"],
                         "to": G.nodes[y].get("label", y)})
        return {"found": True, "length": len(hops), "path": [G.nodes[n].get("label", n) for n in path],
                "ids": path, "hops": hops}

    def _resolve_any(self, ref: str) -> str:
        if ref in self.G:
            return ref
        tid = self.resolve_tag(ref)
        if tid:
            return tid
        try:
            return self.resolve_file(ref) if ("." in ref or "/" in ref) else self._concept_or_file(ref)
        except ToolError:
            return self._concept_or_file(ref)

    def _concept_or_file(self, ref: str) -> str:
        hits = self._concept_hits(ref, limit=1)
        if hits:
            return hits[0][1]
        return self.resolve_file(ref)

    # ------------------------------------------------------------ UI views
    def neighborhood(self, node: str, limit: int = 120) -> dict:
        """Nodes and edges around one node, sized for an interactive graph view."""
        G = self.G
        if node.startswith("topic:"):
            return self._topic_neighborhood(int(node.split(":", 1)[1]), limit)
        if node not in G:
            node = self._resolve_any(node)
        center = G.nodes[node]
        picked: list[str] = [node]
        kind = center.get("fs_kind")
        if kind == "file":
            picked += [v for v in G.successors(node) if G.edges[node, v].get("relation") == "tagged_with"]
            picked += sorted(self._mentions(node), key=lambda n: -G.degree(n))[:25]
            picked += [r["id"] for r in self.related_files(node, limit=12)["files"]]
            picked += list(G.predecessors(node))[:1]
        elif kind == "tag":
            files = sorted(self.by_tag.get(node, ()), key=lambda f: -float(G.nodes[f].get("mtime") or 0))
            picked += files[:60]
            picked += [b if a == node else a for a, b in list(G.out_edges(node)) + list(G.in_edges(node))
                       if G.edges[a, b].get("relation") == "co_occurs_with"][:20]
        elif kind == "folder":
            picked += [v for v in G.successors(node) if G.edges[node, v].get("relation") == "contains"][:limit]
            picked += list(G.predecessors(node))[:1]
        else:
            neigh = sorted(nx.all_neighbors(G, node), key=lambda n: -G.degree(n))
            picked += [n for n in neigh if not G.nodes[n].get("fs_kind")][:40]
            picked += [n for n in neigh if G.nodes[n].get("fs_kind") == "file"][:30]
        seen: list[str] = []
        for n in picked:
            if n in G and n not in seen:
                seen.append(n)
        return self._view(seen[:limit], center=node)

    def _topic_neighborhood(self, cid: int, limit: int) -> dict:
        G = self.G
        files = sorted(self.by_topic.get(cid, ()), key=lambda f: -float(G.nodes[f].get("mtime") or 0))[:50]
        concepts = sorted((n for n, d in G.nodes(data=True) if not d.get("fs_kind") and d.get("community") == cid),
                          key=lambda n: -G.degree(n))[:30]
        tags = [t for t in self.tags if (G.nodes[t].get("topics") or [None])[0] == cid][:20]
        view = self._view(concepts + tags + files, center=None)
        view["nodes"].insert(0, {"id": f"topic:{cid}", "label": self.topic_names.get(cid, f"Topic {cid}"),
                                 "group": "topic", "value": len(files) + 5})
        for n in concepts[:8] + tags[:8]:
            view["edges"].append({"from": f"topic:{cid}", "to": n, "relation": "in_topic"})
        view["center"] = f"topic:{cid}"
        return view

    def topic_map(self, limit: int = 60) -> dict:
        """Topics as nodes, linked when their concepts or tags connect."""
        G = self.G
        if self._topic_edges is None:
            links: dict[int, Counter] = defaultdict(Counter)
            for u, v, d in G.edges(data=True):
                if d.get("relation") in ("contains", "mentions"):
                    continue
                cu, cv = G.nodes[u].get("community"), G.nodes[v].get("community")
                if isinstance(cu, int) and isinstance(cv, int) and cu != cv:
                    links[cu][cv] += 1
                    links[cv][cu] += 1
            self._topic_edges = links
        top = sorted(self.topic_names, key=lambda c: -len(self.by_topic.get(c, ())))[:limit]
        keep = set(top)
        nodes = [{"id": f"topic:{c}", "label": self.topic_names[c], "group": "topic",
                  "value": len(self.by_topic.get(c, ())) + 1, "files": len(self.by_topic.get(c, ()))} for c in top]
        edges = []
        seen = set()
        for c in top:
            for other, n in self._topic_edges.get(c, Counter()).most_common(4):
                if other in keep and (other, c) not in seen:
                    seen.add((c, other))
                    edges.append({"from": f"topic:{c}", "to": f"topic:{other}", "value": n, "relation": "linked"})
        return {"nodes": nodes, "edges": edges, "center": None}

    def _view(self, nodes: list[str], center: str | None) -> dict:
        G = self.G
        keep = set(nodes)
        out_nodes = []
        for n in nodes:
            d = G.nodes[n]
            kind = d.get("fs_kind") or "concept"
            out_nodes.append({
                "id": n, "label": _short(d.get("label", n)), "title": d.get("path") or d.get("label", n),
                "group": d.get("kind") if kind == "file" else kind, "fs_kind": kind,
                "value": max(1, G.degree(n)) if kind == "concept" else (len(self.by_tag.get(n, ())) if kind == "tag" else 3),
                "color": d.get("color") if kind == "tag" else None,
            })
        out_edges = []
        for u, v, d in G.subgraph(keep).edges(data=True):
            out_edges.append({"from": u, "to": v, "relation": d.get("relation", ""),
                              "dashes": d.get("status") == "suggested"})
        return {"nodes": out_nodes, "edges": out_edges, "center": center}

    def facets(self) -> dict:
        G = self.G
        kinds = Counter(G.nodes[f].get("kind", "other") for f in self.files)
        years = Counter((G.nodes[f].get("modified") or "")[:4] for f in self.files)
        return {
            "total_files": len(self.files),
            "kinds": [{"id": k, "label": KIND_LABELS.get(k, k), "files": n} for k, n in kinds.most_common()],
            "tags": self.list_tags(limit=300)["tags"],
            "topics": self.list_topics(limit=200)["topics"],
            "years": [{"year": y, "files": n} for y, n in sorted(years.items(), reverse=True) if y],
            "root": self.list_folder("", limit=0),
        }


def _short(text: str, n: int = 38) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _strip_root(rel: str, root: Path) -> str:
    r = root.as_posix().strip("/")
    rel = rel.strip("/")
    if rel.startswith(r + "/"):
        return rel[len(r) + 1:]
    return rel


def _fmt_ts(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts).isoformat(timespec="minutes")


def _parse_date(value: str | None) -> float | None:
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y", "%Y-%m-%dT%H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    raise ToolError(f"unrecognised date {value!r} (use YYYY-MM-DD)")


_KIND_ALIASES = {
    "pdf": {"pdf"}, "pdfs": {"pdf"}, "slides": {"slides"}, "slide": {"slides"}, "presentation": {"slides"},
    "presentations": {"slides"}, "ppt": {"slides"}, "pptx": {"slides"}, "deck": {"slides"},
    "document": {"document", "text", "pdf"}, "documents": {"document", "text", "pdf"}, "doc": {"document"},
    "docx": {"document"}, "word": {"document"}, "spreadsheet": {"spreadsheet"}, "excel": {"spreadsheet"},
    "xlsx": {"spreadsheet"}, "sheet": {"spreadsheet"}, "image": {"image"}, "images": {"image"},
    "photo": {"image"}, "photos": {"image"}, "picture": {"image"}, "video": {"video"}, "videos": {"video"},
    "audio": {"audio"}, "recording": {"audio", "video"}, "text": {"text"}, "note": {"text"}, "notes": {"text"},
    "markdown": {"text"}, "code": {"code"}, "archive": {"archive"}, "zip": {"archive"}, "email": {"email"},
    "web": {"web"}, "other": {"other"},
}


def _kind_aliases(kind: str | None) -> set[str] | None:
    if not kind:
        return None
    k = _norm(kind).strip()
    return _KIND_ALIASES.get(k, {k})


# ------------------------------------------------------------------ registry
TOOL_SPECS: list[dict] = [
    {"name": "overview", "description": "Summary of the indexed folder: file counts by kind, top tags and top topics. Call first when you know nothing about the collection.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "search", "description": "Search files by meaning and name: matches file names, titles, tags and the concepts extracted from file contents. Returns ranked files plus matching concepts and tags.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "words to look for, e.g. 'pricing model 2025'"},
         "kind": {"type": "string", "description": "optional file kind: pdf, slides, document, spreadsheet, image, video, audio, text"},
         "limit": {"type": "integer", "description": "max files (default 10)"}}, "required": ["query"]}},
    {"name": "find_files", "description": "List files matching exact filters: tag, kind, folder (recursive), topic, part of the name, and modification dates (YYYY-MM-DD). Sorted newest first.",
     "parameters": {"type": "object", "properties": {
         "tag": {"type": "string"}, "kind": {"type": "string"}, "folder": {"type": "string"},
         "topic": {"type": "string"}, "name": {"type": "string"},
         "modified_after": {"type": "string"}, "modified_before": {"type": "string"},
         "limit": {"type": "integer"}}}},
    {"name": "get_file", "description": "Details of one file: metadata, tags, topic, the concepts it contains and related files.",
     "parameters": {"type": "object", "properties": {"file": {"type": "string", "description": "file id, path or name"}}, "required": ["file"]}},
    {"name": "read_file", "description": "Read the text inside a file (documents, slides, PDFs, transcripts). With a query, returns only the most relevant passages. Use it to answer questions about what a file says.",
     "parameters": {"type": "object", "properties": {"file": {"type": "string", "description": "file id, path or name"},
                                                     "query": {"type": "string", "description": "optional: what you are looking for in the file"}}, "required": ["file"]}},
    {"name": "related_files", "description": "Files related to a given file through shared concepts, tags, topic or folder.",
     "parameters": {"type": "object", "properties": {"file": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["file"]}},
    {"name": "list_tags", "description": "Tags in the collection with file counts and related tags; optionally filtered by text.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "list_topics", "description": "Topics (clusters of related concepts and files) with their sizes and top tags.",
     "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "list_folder", "description": "Sub-folders and files directly inside a folder ('' for the top folder).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "explain", "description": "Explain a concept, tag or topic: what it is linked to and which files mention it.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "connection", "description": "How two things (files, tags or concepts) are connected in the graph: the shortest chain between them.",
     "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}},
]

TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


def call_tool(index: FsIndex, name: str, args: dict | None) -> dict:
    """Dispatch a tool by name with defensive argument handling."""
    if name not in TOOL_NAMES:
        raise ToolError(f"unknown tool {name!r}")
    args = {k: v for k, v in (args or {}).items() if v not in (None, "")}
    fn = getattr(index, name)
    import inspect

    params = inspect.signature(fn).parameters
    clean = {}
    for k, v in args.items():
        if k not in params:
            continue
        if k == "limit":
            try:
                v = max(1, min(int(v), 100))
            except (TypeError, ValueError):
                continue
        elif k == "offset":
            try:
                v = max(0, int(v))
            except (TypeError, ValueError):
                continue
        clean[k] = v
    return fn(**clean)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")
