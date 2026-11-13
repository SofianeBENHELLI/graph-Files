"""Video and audio: transcribe locally (faster-whisper), then extract concepts.

graphify's headless ``extract`` skips media, so the fs indexer handles it:
each media file gets a transcript under ``fs/media/<key>/`` (key = hash of path,
size and mtime, so an edited file is re-transcribed) and the transcript goes
through graphify's semantic extraction with the local model. Results are cached
per transcript.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from graphify.fs.config import read_json, write_json_atomic
from graphify.fs.metadata import ScanResult

MEDIA_KINDS = {"video", "audio"}
MAX_MEDIA_BYTES = 8 * 1024 * 1024 * 1024


def media_key(rel: str, size: int, mtime: float) -> str:
    return hashlib.sha1(f"{rel}|{size}|{int(mtime)}".encode()).hexdigest()[:12]


def process_media(scan: ScanResult, media_dir: Path, *, extract_model: str, whisper_model: str,
                  extract: bool = True, progress=None) -> tuple[dict, dict[str, str]]:
    """Transcribe + extract every online media file.

    Returns ``(extraction, source_map)`` where extraction is ``{nodes, edges}``
    and source_map maps each transcript's ``source_file`` spelling to the media
    file's path relative to the root.
    """
    media = [r for r in scan.files if r.kind in MEDIA_KINDS and not r.offline and r.size <= MAX_MEDIA_BYTES]
    nodes: list[dict] = []
    edges: list[dict] = []
    source_map: dict[str, str] = {}
    if not media:
        return {"nodes": nodes, "edges": edges}, source_map
    try:
        from graphify import transcribe as tr
    except ImportError as exc:
        print(f"[graphify fs] media skipped: {exc}", file=sys.stderr)
        return {"nodes": nodes, "edges": edges}, source_map
    try:
        tr._get_whisper()
    except ImportError as exc:
        print(f"[graphify fs] {len(media)} media file(s) skipped: {exc}", file=sys.stderr)
        return {"nodes": nodes, "edges": edges}, source_map
    os.environ.setdefault("GRAPHIFY_WHISPER_MODEL", whisper_model)
    for i, rec in enumerate(media, 1):
        key = media_key(rec.rel, rec.size, rec.mtime)
        out_dir = media_dir / key
        try:
            transcript = tr.transcribe(scan.root / rec.rel, out_dir)
        except Exception as exc:
            print(f"[graphify fs] could not transcribe {rec.rel}: {exc}", file=sys.stderr)
            continue
        rel_t = transcript.relative_to(media_dir).as_posix()
        source_map[rel_t] = rec.rel
        if progress:
            progress(i, len(media), rec.rel)
        cache_file = out_dir / "extraction.json"
        cached = read_json(cache_file, None)
        if cached is None and (not extract or not transcript.read_text(encoding="utf-8", errors="ignore").strip()):
            continue
        if cached is None:
            try:
                from graphify.llm import extract_files_direct

                cached = extract_files_direct([transcript], backend="ollama", model=extract_model, root=media_dir)
                cached = {"nodes": cached.get("nodes", []), "edges": cached.get("edges", [])}
                write_json_atomic(cache_file, cached)
            except Exception as exc:
                print(f"[graphify fs] concept extraction failed for {rec.rel}: {exc}", file=sys.stderr)
                continue
        nodes.extend(cached.get("nodes", []))
        edges.extend(cached.get("edges", []))
    return {"nodes": nodes, "edges": edges}, source_map


def merge_extraction(G, extraction: dict) -> int:
    """Add extracted concept nodes/edges into the content graph (existing ids win)."""
    added = 0
    for node in extraction.get("nodes", []):
        nid = node.get("id")
        if not nid:
            continue
        if nid not in G:
            attrs = {k: v for k, v in node.items() if k != "id"}
            attrs.setdefault("label", nid)
            G.add_node(nid, **attrs)
            added += 1
    for edge in extraction.get("edges", []):
        u, v = edge.get("source"), edge.get("target")
        if u in G and v in G and not G.has_edge(u, v):
            G.add_edge(u, v, **{k: val for k, val in edge.items() if k not in ("source", "target")})
    return added
