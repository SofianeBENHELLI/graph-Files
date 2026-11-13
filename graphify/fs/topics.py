"""Topic tags suggested by a local LLM.

For each file that has something to go on (extracted concepts, a title or
keywords), the model sees its path, kind, title, existing tags, the concepts
graphify extracted from it and its topic, and returns 2–6 short topic tags. It
is told to reuse the existing vocabulary, so tags converge instead of
fragmenting. Results are cached per file by a hash of that context: a rerun only
asks about files whose content changed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import networkx as nx

from graphify.fs.config import read_json, write_json_atomic
from graphify.fs.graph import file_concepts
from graphify.fs.tags import clean_tag

SYSTEM_PROMPT = """You organise a personal knowledge base of files (documents, slides, PDFs, images, videos).
For each file you receive, choose 2 to 6 topic tags that describe what the file is ABOUT, so related files can be found together.

Rules:
- A tag is 1 to 3 words, a noun phrase (e.g. "machine learning", "budget 2025", "customer onboarding").
- Prefer tags from the EXISTING TAGS list whenever one fits; only invent a tag when none fits.
- Use the main language of the file's content; keep product, company and people names as written.
- No generic tags such as "document", "file", "pdf", "presentation", "image", "misc", "notes".
- Do not repeat the file name or its folder path as a tag.
Return JSON only: {"files": [{"id": <id>, "tags": ["tag", ...]}]}"""

GENERIC = {"document", "documents", "file", "files", "pdf", "presentation", "slides", "image", "images",
           "photo", "misc", "miscellaneous", "notes", "other", "untitled", "draft", "video", "audio",
           "spreadsheet", "report", "fichier", "divers", "autre"}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "tags"],
            },
        }
    },
    "required": ["files"],
}


def file_context(G: nx.DiGraph, fid: str, community_names: dict[int, str] | None = None) -> dict | None:
    data = G.nodes[fid]
    concepts = [G.nodes[c].get("label", c) for c in file_concepts(G, fid, limit=15)]
    if not concepts and not data.get("title") and not data.get("keywords"):
        return None
    ctx = {
        "path": data.get("path"),
        "kind": data.get("kind"),
        "title": data.get("title") or None,
        "tags": data.get("finder_tags", []) + data.get("keywords", []),
        "concepts": concepts,
    }
    topics = data.get("topics") or []
    if topics and community_names:
        ctx["topic"] = community_names.get(topics[0])
    return {k: v for k, v in ctx.items() if v}


def context_signature(ctx: dict) -> str:
    return hashlib.sha1(json.dumps(ctx, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _clean(tags, ctx: dict) -> list[str]:
    out: list[str] = []
    name_stem = Path(ctx.get("path", "")).stem.casefold()
    for t in tags if isinstance(tags, list) else []:
        tag = clean_tag(str(t))
        if not tag:
            continue
        low = tag.casefold()
        if low in GENERIC or low == name_stem or len(tag.split()) > 4:
            continue
        if low not in {o.casefold() for o in out}:
            out.append(tag)
    return out[:6]


class TopicTagger:
    def __init__(self, client, model: str, cache_path: Path, *, batch_size: int = 6):
        self.client = client
        self.model = model
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.cache: dict = read_json(self.cache_path, {})

    def _ask(self, batch: list[tuple[str, dict]], vocabulary: list[str]) -> dict[int, list[str]]:
        items = [{"id": i, **ctx} for i, (_rel, ctx) in enumerate(batch, 1)]
        user = (
            "EXISTING TAGS: " + (", ".join(vocabulary[:150]) if vocabulary else "(none yet)")
            + "\n\nFILES:\n" + json.dumps(items, ensure_ascii=False, indent=0)
        )
        result = self.client.chat_json(
            self.model,
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            schema=RESPONSE_SCHEMA,
            options={"temperature": 0.1, "num_ctx": 8192},
        )
        out: dict[int, list[str]] = {}
        for entry in result.get("files", []) if isinstance(result, dict) else []:
            try:
                out[int(entry.get("id"))] = entry.get("tags", [])
            except (TypeError, ValueError, AttributeError):
                continue
        return out

    def suggest(self, G: nx.DiGraph, community_names: dict[int, str], vocabulary: list[str],
                *, limit: int | None = None, progress=None, use_llm: bool = True) -> dict[str, list]:
        """Return ``{rel: [[tag, score], ...]}`` for every file with context (cached + fresh).

        Without the LLM (``use_llm=False``) a file whose context changed keeps its
        previous suggestions until the next LLM run refreshes them.
        """
        todo: list[tuple[str, dict, str]] = []
        result: dict[str, list] = {}
        for fid, data in G.nodes(data=True):
            if data.get("fs_kind") != "file":
                continue
            ctx = file_context(G, fid, community_names)
            if ctx is None:
                continue
            rel = data["path"]
            sig = context_signature(ctx)
            cached = self.cache.get(rel)
            if cached and (cached.get("sig") == sig or not use_llm):
                result[rel] = cached.get("tags", [])
            elif use_llm:
                todo.append((rel, ctx, sig))
        if not use_llm or self.client is None:
            return result
        if limit is not None:
            todo = todo[:limit]
        vocab = list(vocabulary)
        done = 0
        failures = 0
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start:start + self.batch_size]
            try:
                answers = self._ask([(rel, ctx) for rel, ctx, _sig in batch], vocab)
            except Exception as exc:  # one bad batch must not stop the index
                failures += 1
                print(f"[graphify fs] topic tagging batch failed: {exc}", file=sys.stderr)
                if failures >= 3 and done == 0:
                    print("[graphify fs] topic tagging disabled for this run (model unavailable)", file=sys.stderr)
                    break
                continue
            for i, (rel, ctx, sig) in enumerate(batch, 1):
                tags = _clean(answers.get(i, []), ctx)
                scored = [[t, round(0.9 - 0.08 * j, 2)] for j, t in enumerate(tags)]
                result[rel] = scored
                self.cache[rel] = {"sig": sig, "tags": scored}
                for t in tags:
                    if t.casefold() not in {v.casefold() for v in vocab}:
                        vocab.append(t)
            done += len(batch)
            if progress:
                progress(done, len(todo))
            if done % (self.batch_size * 10) == 0:
                write_json_atomic(self.cache_path, self.cache)
        write_json_atomic(self.cache_path, self.cache)
        return result

    def cached(self) -> dict[str, list]:
        return {rel: v.get("tags", []) for rel, v in self.cache.items()}


TOPIC_PROMPT = """You name clusters ("topics") of a personal knowledge graph of files.
Each topic lists its most important concepts, tags and example file names.
Give each topic a short, specific name of 2 to 5 words that a person would use as a folder name
(e.g. "Atlas product roadmap", "2025 budget and invoices"). Use the main language of the content.
Return JSON only: {"topics": [{"id": <id>, "name": "..."}]}"""

TOPIC_SCHEMA = {
    "type": "object",
    "properties": {"topics": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
        "required": ["id", "name"]}}},
    "required": ["topics"],
}


def topic_contexts(G: nx.DiGraph, communities: dict[int, list[str]]) -> dict[int, dict]:
    """What each topic is made of: top concepts, tags and file names."""
    out: dict[int, dict] = {}
    for cid, members in communities.items():
        concepts, tags, files = [], [], []
        for n in members:
            d = G.nodes[n] if n in G else {}
            kind = d.get("fs_kind")
            if kind == "tag":
                tags.append(n)
            elif kind == "file":
                files.append(n)
            elif not kind:
                concepts.append(n)
        concepts.sort(key=lambda n: (-G.degree(n), str(n)))
        tags.sort(key=lambda n: (-G.degree(n), str(n)))
        files.sort(key=lambda n: (-G.degree(n), str(n)))
        ctx = {
            "concepts": [G.nodes[n].get("label", n) for n in concepts[:12]],
            "tags": [G.nodes[n].get("label", n) for n in tags[:6]],
            "files": [G.nodes[n].get("label", n) for n in files[:5]],
        }
        if any(ctx.values()):
            out[cid] = ctx
    return out


class TopicNamer:
    """Human-readable topic names from the local model, cached by topic content."""

    def __init__(self, client, model: str, cache_path: Path, batch_size: int = 10):
        self.client = client
        self.model = model
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.cache: dict = read_json(self.cache_path, {})

    def names(self, G: nx.DiGraph, communities: dict[int, list[str]], fallback: dict[int, str],
              *, use_llm: bool = True) -> dict[int, str]:
        contexts = topic_contexts(G, communities)
        names = dict(fallback)
        todo: list[tuple[int, dict, str]] = []
        for cid, ctx in contexts.items():
            sig = context_signature(ctx)
            if sig in self.cache:
                names[cid] = self.cache[sig]
            else:
                todo.append((cid, ctx, sig))
        if not use_llm or self.client is None:
            return names
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start:start + self.batch_size]
            items = [{"id": cid, **ctx} for cid, ctx, _sig in batch]
            try:
                result = self.client.chat_json(
                    self.model,
                    [{"role": "system", "content": TOPIC_PROMPT},
                     {"role": "user", "content": "TOPICS:\n" + json.dumps(items, ensure_ascii=False, indent=0)}],
                    schema=TOPIC_SCHEMA, options={"temperature": 0.1, "num_ctx": 8192},
                )
            except Exception as exc:
                print(f"[graphify fs] topic naming failed: {exc}", file=sys.stderr)
                break
            by_id = {}
            for entry in result.get("topics", []) if isinstance(result, dict) else []:
                try:
                    by_id[int(entry.get("id"))] = str(entry.get("name", "")).strip()
                except (TypeError, ValueError, AttributeError):
                    continue
            for cid, _ctx, sig in batch:
                name = clean_tag(by_id.get(cid, "")) if by_id.get(cid) else None
                if name:
                    names[cid] = name
                    self.cache[sig] = name
        write_json_atomic(self.cache_path, self.cache)
        return names
