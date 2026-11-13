"""``graphify fs`` command line.

    graphify fs setup                  check Ollama and download the local models
    graphify fs index <folder>         build/refresh the knowledge graph of a folder
    graphify fs explore [folder]       open the explorer web app (local LLM chat)
    graphify fs mcp [folder]           MCP stdio server for Claude / other LLM clients
    graphify fs ask "<question>"       ask the local navigator from the terminal
    graphify fs status                 list indexes and their stats
    graphify fs undo-tags [-n N]       revert the last Finder tag writes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graphify.fs.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EXTRACT_MODEL,
    list_indexes,
    load_state,
    resolve_index,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graphify fs", description="Knowledge graph explorer for your files (local LLM).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="check Ollama and pull the local models")
    s.add_argument("--extract-model", default=None, help=f"content extraction model (default {DEFAULT_EXTRACT_MODEL})")
    s.add_argument("--chat-model", default=None, help=f"chat/navigation model (default {DEFAULT_CHAT_MODEL})")
    s.add_argument("--index", default=None, help="also store these models in an existing index")

    i = sub.add_parser("index", help="build or refresh the graph of a folder")
    i.add_argument("folder")
    i.add_argument("--index", default=None, help="index directory (default ~/.graphify-fs/<name>-<hash>)")
    i.add_argument("--metadata-only", action="store_true",
                   help="fast refresh: folders, files, tags only; reuse cached content/topics, no LLM")
    i.add_argument("--no-content", action="store_true", help="skip graphify's content extraction")
    i.add_argument("--no-media", action="store_true", help="skip video/audio transcription")
    i.add_argument("--no-topics", action="store_true", help="skip LLM topic tags")
    i.add_argument("--extract-model", default=None)
    i.add_argument("--chat-model", default=None)
    i.add_argument("--whisper-model", default=None, help="faster-whisper model (tiny/base/small/medium/large-v3)")
    i.add_argument("--no-vision", action="store_true", help="do not send images to the model")
    i.add_argument("--topic-limit", type=int, default=None, help="tag at most N new files this run")

    e = sub.add_parser("explore", help="open the explorer web app")
    e.add_argument("target", nargs="?", default=None, help="indexed folder or index dir (default: latest)")
    e.add_argument("--port", type=int, default=8765)
    e.add_argument("--no-browser", action="store_true")
    e.add_argument("--no-finder-write", action="store_true", help="never write Finder tags")
    e.add_argument("--chat-model", default=None)

    m = sub.add_parser("mcp", help="MCP stdio server over the fs graph")
    m.add_argument("target", nargs="?", default=None)

    a = sub.add_parser("ask", help="ask the local navigator a question")
    a.add_argument("question")
    a.add_argument("target", nargs="?", default=None)
    a.add_argument("--chat-model", default=None)
    a.add_argument("--json", action="store_true", help="print the full trace as JSON")

    st = sub.add_parser("status", help="list indexes")
    st.add_argument("target", nargs="?", default=None)

    u = sub.add_parser("undo-tags", help="revert the last Finder tag writes")
    u.add_argument("target", nargs="?", default=None)
    u.add_argument("-n", "--count", type=int, default=1)
    return p


def cmd_setup(args) -> int:
    from graphify.fs.ollama import Ollama, OllamaError

    extract_model = args.extract_model or DEFAULT_EXTRACT_MODEL
    chat_model = args.chat_model or DEFAULT_CHAT_MODEL
    if args.index:
        from graphify.fs.config import save_state

        idx = resolve_index(args.index)
        state = load_state(idx)
        extract_model = args.extract_model or state.extract_model
        chat_model = args.chat_model or state.chat_model
        state.extract_model, state.chat_model = extract_model, chat_model
        save_state(idx, state)
    client = Ollama()
    if not client.available():
        print(f"Ollama is not running at {client.url}.")
        print("Install it from https://ollama.com/download (macOS app), open it once, then rerun:")
        print("    graphify fs setup")
        return 2
    print(f"Ollama is running at {client.url}")
    for model in dict.fromkeys([extract_model, chat_model]):
        if client.has_model(model):
            print(f"  ✓ {model} already downloaded")
            continue
        print(f"  ↓ pulling {model} (several GB, one time)...")
        last = [None]

        def progress(msg):
            status = msg.get("status", "")
            total, done = msg.get("total"), msg.get("completed")
            line = f"    {status}" + (f" {done * 100 // total}%" if total and done else "")
            if line != last[0]:
                print(line, end="\r" if total else "\n", flush=True)
                last[0] = line
        try:
            client.pull(model, on_progress=progress)
            print(f"\n  ✓ {model} ready")
        except OllamaError as exc:
            print(f"\n  ✗ could not pull {model}: {exc}")
            return 1
    print("\nNext: graphify fs index <folder>")
    return 0


def cmd_index(args) -> int:
    from graphify.fs.indexer import IndexOptions, run_index

    opts = IndexOptions(
        index=Path(args.index) if args.index else None,
        metadata_only=args.metadata_only,
        content=not args.no_content,
        media=not args.no_media,
        topics=not args.no_topics,
        extract_model=args.extract_model,
        chat_model=args.chat_model,
        whisper_model=args.whisper_model,
        vision=False if args.no_vision else None,
        topic_limit=args.topic_limit,
    )
    run_index(Path(args.folder), opts)
    print("\nOpen it with: graphify fs explore " + json.dumps(str(Path(args.folder).expanduser().resolve())))
    return 0


def _load(target):
    from graphify.fs.tools import FsIndex

    return FsIndex(resolve_index(target))


def cmd_explore(args) -> int:
    from graphify.fs.explorer.app import serve

    serve(_load(args.target), port=args.port, open_browser=not args.no_browser,
          finder_write=not args.no_finder_write, chat_model=args.chat_model)
    return 0


def cmd_mcp(args) -> int:
    from graphify.fs.mcp import run_stdio

    run_stdio(resolve_index(args.target))
    return 0


def cmd_ask(args) -> int:
    from graphify.fs.agent import Navigator

    index = _load(args.target)
    nav = Navigator(index, model=args.chat_model)
    if args.json:
        print(json.dumps(nav.answer(args.question), ensure_ascii=False, indent=1))
        return 0
    for ev in nav.run(args.question):
        if ev["type"] == "tool_call":
            print(f"  · {ev['name']} {json.dumps(ev['args'], ensure_ascii=False)}", file=sys.stderr)
        elif ev["type"] == "tool_result":
            print(f"    → {ev['summary']}", file=sys.stderr)
        elif ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "error":
            print(f"error: {ev['message']}", file=sys.stderr)
            return 1
    print()
    return 0


def cmd_status(args) -> int:
    rows = [(resolve_index(args.target), None)] if args.target else list_indexes()
    if not rows:
        print("No index yet. Build one with: graphify fs index <folder>")
        return 0
    for idx, state in rows:
        state = state or load_state(idx)
        s = state.stats or {}
        print(f"{state.root}\n  index: {idx}\n  models: extract={state.extract_model} chat={state.chat_model}")
        if s:
            print(f"  {s.get('files', 0)} files · {s.get('concepts', 0)} concepts · {s.get('tags', 0)} tags "
                  f"({s.get('suggested_tags', 0)} suggested) · {s.get('topics', 0)} topics · "
                  f"{s.get('offline', 0)} iCloud-only")
    return 0


def cmd_undo(args) -> int:
    from graphify.fs.config import FsIndexPaths
    from graphify.fs.finder import FinderTagWriter

    idx = resolve_index(args.target)
    state = load_state(idx)
    writer = FinderTagWriter(Path(state.root), FsIndexPaths(idx).tag_writes)
    for line in writer.undo(args.count) or ["nothing to undo"]:
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"setup": cmd_setup, "index": cmd_index, "explore": cmd_explore, "mcp": cmd_mcp,
                "ask": cmd_ask, "status": cmd_status, "undo-tags": cmd_undo}
    try:
        return handlers[args.command](args) or 0
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
