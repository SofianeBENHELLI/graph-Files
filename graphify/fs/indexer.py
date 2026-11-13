"""``graphify fs index``: build or refresh the fs knowledge graph of a folder.

Stages (each one is resumable and cached, so an interrupted run picks up where
it stopped):

1. scan        folders, files, sizes, dates, Finder tags, document keywords (seconds)
2. content     graphify's own ``extract`` with the local Ollama model, writing the
               content graph to ``<index>/graphify-out/graph.json`` (slow; cached per file)
3. media       local Whisper transcripts of video/audio + concept extraction
4. topics      local-LLM topic tags per file (cached per file)
5. assemble    content + fs layer + tag relations + topics → ``fs/graph.json``

``metadata_only`` skips every LLM stage (and Whisper) but still reuses whatever
content, transcripts and topic results earlier runs cached, so it doubles as a
fast refresh.

Only one run per index at a time: runs take ``fs/index.lock`` (see ``lock.py``).
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
from graphify.fs.lock import IndexLock
from graphify.fs.metadata import scan
from graphify.fs.tags import TagDecisions

# Extensions whose content graphify's extract pass reads (documents, PDFs,
# Office files, images, text, code). Used to tell which changed files still
# need a content pass. Audio/video go through the media stage instead.
def content_extensions(vision: bool = True) -> frozenset[str]:
    from graphify.detect import (
        CODE_EXTENSIONS,
        DOC_EXTENSIONS,
        IMAGE_EXTENSIONS,
        OFFICE_EXTENSIONS,
        PAPER_EXTENSIONS,
        VIDEO_EXTENSIONS,
    )

    images = IMAGE_EXTENSIONS if vision else {".svg"}
    return frozenset(e.lower() for e in (CODE_EXTENSIONS | DOC_EXTENSIONS | images
                                         | OFFICE_EXTENSIONS | PAPER_EXTENSIONS | VIDEO_EXTENSIONS))


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
    # Wait for another run on the same index to finish (False: raise IndexBusy).
    wait_for_lock: bool = True


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


@dataclass
class _LlmRun:
    """What the slow (model) phase did, for the assemble phase."""

    client: object | None = None
    topics: bool = False
    content_started: float | None = None


def _settings(index: Path, root: Path, opts: IndexOptions) -> IndexState:
    state = load_state(index) or IndexState(root=str(root))
    state.root = str(root)
    state.extract_model = opts.extract_model or state.extract_model or DEFAULT_EXTRACT_MODEL
    state.chat_model = opts.chat_model or state.chat_model or DEFAULT_CHAT_MODEL
    state.whisper_model = opts.whisper_model or state.whisper_model or DEFAULT_WHISPER_MODEL
    if opts.vision is not None:
        state.vision = opts.vision
    return state


def _scan(root: Path, paths: FsIndexPaths, log):
    return scan(root, cache_path=paths.meta_cache,
                progress=lambda n: log(f"[graphify fs] scanned {n} files..."))


def run_index(root: Path, options: IndexOptions | None = None, log=_log_default) -> dict:
    """Index ``root``. Two phases so the slow one never blocks quick refreshes:

    1. models (content pass, transcription), under ``content.lock``: writes only
       caches (graphify's content graph, transcripts), each atomically;
    2. assemble (scan, full text, topic tags, graph), under ``index.lock``.

    ``opts.wait_for_lock=False`` raises ``IndexBusy`` instead of waiting when the
    first lock it needs is taken.
    """
    opts = options or IndexOptions()
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    index = Path(opts.index).expanduser().resolve() if opts.index else default_index_dir(root)
    index.mkdir(parents=True, exist_ok=True)
    paths = FsIndexPaths(index)
    paths.fs_dir.mkdir(parents=True, exist_ok=True)

    def waiting(who: str) -> None:
        log(f"[graphify fs] waiting for another index run to finish ({who})...")

    t0 = time.time()
    log(f"[graphify fs] indexing {root}")
    log(f"[graphify fs] index: {index}")
    llm = _LlmRun()
    first_wait = opts.wait_for_lock
    if not opts.metadata_only:
        content_lock = IndexLock(paths.content_lock)
        content_lock.acquire(wait=first_wait, on_wait=waiting)
        first_wait = True  # the assemble phase is short: always wait for it
        try:
            llm = _model_phase(root, index, paths, opts, log)
        finally:
            content_lock.release()
    lock = IndexLock(paths.lock)
    lock.acquire(wait=first_wait, on_wait=waiting)
    try:
        return _assemble_phase(root, index, paths, opts, llm, log, t0)
    finally:
        lock.release()


def _model_phase(root: Path, index: Path, paths: FsIndexPaths, opts: IndexOptions, log) -> _LlmRun:
    from graphify.fs.ollama import Ollama

    state = _settings(index, root, opts)
    run = _LlmRun()
    client = Ollama()
    if not client.available():
        log(f"[graphify fs] Ollama is not running at {client.url}: skipping LLM stages "
            "(start Ollama, then run `graphify fs setup`)")
        return run
    content, topics = opts.content, opts.topics
    missing = [m for m in {state.extract_model, state.chat_model} if not client.has_model(m)]
    if missing:
        log(f"[graphify fs] missing Ollama model(s): {', '.join(missing)} - run `graphify fs setup`")
        if state.extract_model in missing:
            content = False
        if state.chat_model in missing:
            topics = False
    run.client = client
    run.topics = topics

    # content: graphify extract writes the content graph atomically when done
    if content:
        run.content_started = time.time()  # files changed during the pass stay "unread"
        run_content_pass(root, index, state.extract_model, state.vision, log, opts.extra_extract_args)

    # media: transcripts and their concepts are cached per file under fs/media/
    if opts.media:
        result = _scan(root, paths, log)
        media_count = sum(1 for f in result.files if f.kind in ("video", "audio") and not f.offline)
        if media_count:
            from graphify.fs.media import process_media

            log(f"[graphify fs] media: {media_count} video/audio file(s)")
            try:
                process_media(
                    result, paths.media_dir, extract_model=state.extract_model,
                    whisper_model=state.whisper_model, extract=content,
                    progress=lambda i, n, rel: log(f"[graphify fs] media {i}/{n}: {rel}"),
                )
            except Exception as exc:  # media must never break indexing
                log(f"[graphify fs] media stage failed: {exc}")
    return run


def _assemble_phase(root: Path, index: Path, paths: FsIndexPaths, opts: IndexOptions, llm: _LlmRun,
                    log, t0: float) -> dict:
    # Re-read the state under the lock: a quick refresh may have saved it meanwhile.
    state = _settings(index, root, opts)
    if llm.content_started:
        state.last_content_pass = max(state.last_content_pass or 0, llm.content_started)
    save_state(index, state)

    # 1. scan --------------------------------------------------------------
    result = _scan(root, paths, log)
    offline = sum(1 for f in result.files if f.offline)
    log(f"[graphify fs] {len(result.files)} files in {len(result.folders)} folders"
        + (f" ({offline} iCloud placeholders, not downloaded)" if offline else ""))

    # 2. content graph (from the model phase of this or an earlier run) ------
    content = load_content_graph(paths.content_graph)
    if content is not None:
        log(f"[graphify fs] content graph: {content.number_of_nodes()} concepts, {content.number_of_edges()} links")

    # 3. media: cached transcripts and concepts only (made by the model phase)
    media_map: dict[str, str] = {}
    if opts.media:
        from graphify.fs.media import merge_extraction, process_media

        try:
            extraction, media_map = process_media(
                result, paths.media_dir, extract_model=state.extract_model,
                whisper_model=state.whisper_model, extract=False, transcribe_new=False,
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
    from graphify.fs.topics import TopicNamer, TopicTagger

    client = llm.client
    llm_topics = bool(client is not None and llm.topics)
    tagger = TopicTagger(client, state.chat_model, paths.topics_cache)
    suggestions = tagger.cached()
    G = assemble_base(result, content, resolver)
    vocab = apply_tags(G, result, decisions, suggestions)
    communities, names = assign_topics(G)
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
    namer = TopicNamer(client, state.chat_model, paths.fs_dir / "topic-names.json")
    names = namer.names(G, communities, names, use_llm=llm_topics)
    write_fs_graph(G, communities, names, paths.graph, root=root)

    tags = [n for n, d in G.nodes(data=True) if d.get("fs_kind") == "tag"]
    suggested = sum(1 for _u, _v, d in G.edges(data=True)
                    if d.get("relation") == "tagged_with" and d.get("status") == "suggested")
    concepts = sum(1 for _n, d in G.nodes(data=True) if not d.get("fs_kind"))
    mentions = sum(1 for _u, _v, d in G.edges(data=True) if d.get("relation") == "mentions")
    # Files no completed content pass has read yet (new or edited since). The
    # folder watcher uses this to schedule a background content pass.
    exts = content_extensions(state.vision)
    since = state.last_content_pass or 0
    unread = sum(1 for f in result.files if not f.offline and f.ext in exts and f.mtime > since)
    stats = {
        "files": len(result.files), "folders": len(result.folders), "offline": offline,
        "unread": unread,
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
