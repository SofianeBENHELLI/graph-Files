"""``graphify fs index``: build or refresh the fs knowledge graph of a folder.

Stages (each one is resumable and cached, so an interrupted run picks up where
it stopped):

1. scan        folders, files, sizes, dates, Finder tags, document keywords (seconds)
2. content     graphify's own ``extract`` with the local Ollama model, writing the
               content graph to ``<index>/graphify-out/graph.json`` (slow; cached per file)
3. media       local Whisper transcripts of video/audio + concept extraction
4. topics      local-LLM topic tags per file (cached per file)
5. assemble    content + fs layer + tag relations + topics → ``fs/graph.json``

``metadata_only`` skips every LLM stage but still reuses whatever content and
topic results earlier runs cached, so it doubles as a fast refresh.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from graphify.fs.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EXTRACT_MODEL,
    DEFAULT_WHISPER_MODEL,
    FsIndexPaths,
    IndexState,
    default_index_dir,
    load_state,
    save_state,
)
from graphify.fs.graph import (
    SourceResolver,
    apply_tags,
    assemble_base,
    assign_topics,
    load_content_graph,
    tag_relations,
    write_fs_graph,
)
from graphify.fs.metadata import scan
from graphify.fs.tags import TagDecisions


@dataclass
class IndexOptions:
    index: Path | None = None
    metadata_only: bool = False
    content: bool = True
    media: bool = True
    topics: bool = True
    extract_model: str | None = None
    chat_model: str | None = None
    whisper_model: str | None = None
    vision: bool | None = None
    topic_limit: int | None = None
    extra_extract_args: list[str] = field(default_factory=list)


def _log_default(msg: str) -> None:
    print(msg, flush=True)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60} min"
    return f"{seconds / 3600:.1f} h"


def run_content_pass(root: Path, index: Path, model: str, vision: bool, log=_log_default,
                     extra_args: list[str] | None = None) -> bool:
    """Run ``graphify extract`` on the folder with the local model; True on success."""
    extra = list(extra_args or [])
    if "--token-budget" not in extra:
        # graphify sizes Ollama's num_ctx as chunk tokens + 16k output; 60k-token
        # chunks (its default) would need a ~78k context, too heavy for a 12B
        # model on a laptop. 12k keeps the context near 30k.
        extra += ["--token-budget", os.environ.get("GRAPHIFY_FS_TOKEN_BUDGET", "12000")]
    cmd = [sys.executable, "-m", "graphify", "extract", str(root), "--backend", "ollama",
           "--model", model, "--out", str(index), *extra]
    env = os.environ.copy()
    env.setdefault("OLLAMA_API_KEY", "ollama")
    env["GRAPHIFY_OLLAMA_VISION"] = "1" if vision else "0"
    log(f"[graphify fs] content pass: graphify extract with {model} (this is the slow part)")
    start = time.time()
    first_chunk_at: float | None = None
    proc = subprocess.Popen(cmd, cwd=str(index), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        m = re.search(r"chunk (\d+)/(\d+) done", line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            now = time.time()
            if first_chunk_at is None:
                first_chunk_at = now
            rate = (now - start) / max(done, 1)
            eta = rate * (total - done)
            log(f"{line}   (~{_fmt_duration(eta)} left)")
        elif line:
            log(line)
    rc = proc.wait()
    if rc != 0:
        log(f"[graphify fs] content pass exited with code {rc}; continuing with what was extracted")
    log(f"[graphify fs] content pass took {_fmt_duration(time.time() - start)}")
    return rc == 0


def run_index(root: Path, options: IndexOptions | None = None, log=_log_default) -> dict:
    opts = options or IndexOptions()
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    index = Path(opts.index).expanduser().resolve() if opts.index else default_index_dir(root)
    index.mkdir(parents=True, exist_ok=True)
    paths = FsIndexPaths(index)
    paths.fs_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(index) or IndexState(root=str(root))
    state.root = str(root)
    state.extract_model = opts.extract_model or state.extract_model or DEFAULT_EXTRACT_MODEL
    state.chat_model = opts.chat_model or state.chat_model or DEFAULT_CHAT_MODEL
    state.whisper_model = opts.whisper_model or state.whisper_model or DEFAULT_WHISPER_MODEL
    if opts.vision is not None:
        state.vision = opts.vision
    save_state(index, state)
    t0 = time.time()
    log(f"[graphify fs] indexing {root}")
    log(f"[graphify fs] index: {index}")

    # 1. scan --------------------------------------------------------------
    result = scan(root, cache_path=paths.meta_cache,
                  progress=lambda n: log(f"[graphify fs] scanned {n} files..."))
    offline = sum(1 for f in result.files if f.offline)
    log(f"[graphify fs] {len(result.files)} files in {len(result.folders)} folders"
        + (f" ({offline} iCloud placeholders, not downloaded)" if offline else ""))

    use_llm = not opts.metadata_only
    client = None
    if use_llm:
        from graphify.fs.ollama import Ollama

        client = Ollama()
        if not client.available():
            log(f"[graphify fs] Ollama is not running at {client.url}: skipping LLM stages "
                "(start Ollama, then run `graphify fs setup`)")
            use_llm = False
            client = None
        else:
            missing = [m for m in {state.extract_model, state.chat_model} if not client.has_model(m)]
            if missing:
                log(f"[graphify fs] missing Ollama model(s): {', '.join(missing)} - run `graphify fs setup`")
                if state.extract_model in missing:
                    opts.content = False
                if state.chat_model in missing:
                    opts.topics = False

    # 2. content -----------------------------------------------------------
    if use_llm and opts.content:
        run_content_pass(root, index, state.extract_model, state.vision, log, opts.extra_extract_args)
        state.last_content_pass = time.time()
        save_state(index, state)
    content = load_content_graph(paths.content_graph)
    if content is not None:
        log(f"[graphify fs] content graph: {content.number_of_nodes()} concepts, {content.number_of_edges()} links")

    # 3. media -------------------------------------------------------------
    media_map: dict[str, str] = {}
    if opts.media:
        from graphify.fs.media import merge_extraction, process_media

        media_count = sum(1 for f in result.files if f.kind in ("video", "audio") and not f.offline)
        if media_count:
            log(f"[graphify fs] media: {media_count} video/audio file(s)")
            try:
                extraction, media_map = process_media(
                    result, paths.media_dir, extract_model=state.extract_model,
                    whisper_model=state.whisper_model, extract=bool(use_llm and opts.content),
                    progress=lambda i, n, rel: log(f"[graphify fs] media {i}/{n}: {rel}"),
                )
                if extraction["nodes"]:
                    import networkx as nx

                    if content is None:
                        content = nx.DiGraph()
                    added = merge_extraction(content, extraction)
                    log(f"[graphify fs] media concepts added: {added}")
            except Exception as exc:  # media must never break indexing
                log(f"[graphify fs] media stage failed: {exc}")
    resolver = SourceResolver.build(result, extra=media_map)

    # Full-text index of what is written inside files (no LLM; incremental).
    from graphify.fs.fulltext import FullTextIndex

    media_text = {file_rel: paths.media_dir / t_rel for t_rel, file_rel in media_map.items()}
    ft = FullTextIndex(paths.fulltext)
    try:
        read = ft.update(result, media_text, progress=lambda n: log(f"[graphify fs] full-text: {n} files read..."))
        if read:
            log(f"[graphify fs] full-text index: {read} file(s) (re)read")
    finally:
        ft.close()

    # 4. topics ------------------------------------------------------------
    decisions = TagDecisions.load(paths.tags_state)
    from graphify.fs.topics import TopicTagger

    tagger = TopicTagger(client, state.chat_model, paths.topics_cache)
    suggestions = tagger.cached()
    G = assemble_base(result, content, resolver)
    vocab = apply_tags(G, result, decisions, suggestions)
    communities, names = assign_topics(G)
    llm_topics = bool(use_llm and opts.topics and client is not None)
    if llm_topics:
        log(f"[graphify fs] topic tags with {state.chat_model}...")
    suggestions = tagger.suggest(
        G, names, vocab.top(200), limit=opts.topic_limit, use_llm=llm_topics,
        progress=lambda d, n: log(f"[graphify fs] topic tags {d}/{n}") if d % 30 == 0 or d == n else None,
    )
    # 5. assemble with the final suggestions
    G = assemble_base(result, content, resolver)
    apply_tags(G, result, decisions, suggestions)
    relations = tag_relations(G)
    communities, names = assign_topics(G)
    from graphify.fs.topics import TopicNamer

    namer = TopicNamer(client, state.chat_model, paths.fs_dir / "topic-names.json")
    names = namer.names(G, communities, names, use_llm=bool(use_llm and opts.topics and client is not None))
    write_fs_graph(G, communities, names, paths.graph, root=root)

    tags = [n for n, d in G.nodes(data=True) if d.get("fs_kind") == "tag"]
    suggested = sum(1 for _u, _v, d in G.edges(data=True)
                    if d.get("relation") == "tagged_with" and d.get("status") == "suggested")
    concepts = sum(1 for _n, d in G.nodes(data=True) if not d.get("fs_kind"))
    mentions = sum(1 for _u, _v, d in G.edges(data=True) if d.get("relation") == "mentions")
    stats = {
        "files": len(result.files), "folders": len(result.folders), "offline": offline,
        "concepts": concepts, "mentions": mentions, "tags": len(tags), "suggested_tags": suggested,
        "tag_relations": relations, "topics": len(names), "seconds": round(time.time() - t0, 1),
        "scan_errors": len(result.errors),
    }
    state.stats = stats
    state.last_index = time.time()
    save_state(index, state)
    log(f"[graphify fs] done in {_fmt_duration(stats['seconds'])}: {stats['files']} files, "
        f"{stats['concepts']} concepts, {stats['tags']} tags ({suggested} suggested), "
        f"{stats['topics']} topics")
    log(f"[graphify fs] graph: {paths.graph}")
    return {"index": str(index), **stats}
