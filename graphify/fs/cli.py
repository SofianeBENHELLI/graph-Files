"""``graphify fs`` command line.

    graphify fs setup                  check Ollama and download the local models
    graphify fs index <folder>         build/refresh the knowledge graph of a folder
    graphify fs explore [folder]       open the explorer web app (local LLM chat)
    graphify fs open [folder]          open the explorer, starting it in the background
    graphify fs serve [folder]         run the explorer without a browser (keeps the index updated)
    graphify fs stop [folder]          stop the background explorer
    graphify fs app install [folder]   macOS launcher app (+ --login to start at login)
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
    i.add_argument("--watch", action="store_true",
                   help="then keep watching the folder and update the index as files change")

    e = sub.add_parser("explore", help="open the explorer web app")
    e.add_argument("target", nargs="?", default=None, help="indexed folder or index dir (default: latest)")
    e.add_argument("--port", type=int, default=8765)
    e.add_argument("--no-browser", action="store_true")
    e.add_argument("--no-finder-write", action="store_true", help="never write Finder tags")
    e.add_argument("--chat-model", default=None)
    _watch_flags(e)

    sv = sub.add_parser("serve", help="run the explorer in the background, keeping the index updated")
    sv.add_argument("target", nargs="?", default=None, help="indexed folder or index dir (default: latest)")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--no-finder-write", action="store_true", help="never write Finder tags")
    sv.add_argument("--chat-model", default=None)
    _watch_flags(sv)

    o = sub.add_parser("open", help="open the explorer (starts it in the background if needed)")
    o.add_argument("target", nargs="?", default=None)
    o.add_argument("--no-browser", action="store_true", help="only print the address")

    sp = sub.add_parser("stop", help="stop the background explorer")
    sp.add_argument("target", nargs="?", default=None)

    ap = sub.add_parser("app", help="macOS launcher app and login item")
    ap.add_argument("action", choices=["install", "uninstall"])
    ap.add_argument("target", nargs="?", default=None)
    ap.add_argument("--login", action="store_true",
                    help="also start the explorer at login so the index stays up to date")
    ap.add_argument("--name", default=None, help="app name (default \"Graph Files\")")
    ap.add_argument("--dir", default=None, help="where to put the app (default ~/Applications)")

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


def _watch_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-watch", action="store_true", help="do not follow changes in the folder")
    p.add_argument("--no-content-updates", action="store_true",
                   help="follow changes, but never start the local model in the background")


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
    result = run_index(Path(args.folder), opts)
    folder = json.dumps(str(Path(args.folder).expanduser().resolve()))
    if args.watch:
        return _watch_foreground(Path(args.folder), Path(result["index"]))
    print(f"\nOpen it with: graphify fs open {folder}")
    return 0


def _watch_foreground(folder: Path, index_dir: Path) -> int:
    import time

    from graphify.fs.launcher import running_server
    from graphify.fs.watch import Updater

    info = running_server(index_dir)
    if info and info.get("watching"):
        print(f"\nThe explorer server (pid {info.get('pid')}) already keeps this index up to date.")
        return 0
    updater = Updater(folder, index_dir, log=lambda m: print(m, flush=True)).start()
    if not updater.watching:
        return 1
    print("[graphify fs] press Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(3600)
    finally:
        updater.stop()


def _load(target):
    from graphify.fs.tools import FsIndex

    return FsIndex(resolve_index(target))


def cmd_explore(args) -> int:
    from graphify.fs.explorer.app import serve

    return serve(_load(args.target), port=args.port, open_browser=not args.no_browser,
                 finder_write=not args.no_finder_write, chat_model=args.chat_model,
                 watch=not args.no_watch, content_updates=not args.no_content_updates)


def cmd_serve(args) -> int:
    from graphify.fs.explorer.app import serve

    return serve(_load(args.target), port=args.port, open_browser=False,
                 finder_write=not args.no_finder_write, chat_model=args.chat_model,
                 watch=not args.no_watch, content_updates=not args.no_content_updates)


def cmd_open(args) -> int:
    from graphify.fs.launcher import ensure_server, open_url

    idx = resolve_index(args.target)
    try:
        info = ensure_server(idx)
    except (RuntimeError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(info["url"])
    if not args.no_browser:
        open_url(info["url"])
    return 0


def cmd_stop(args) -> int:
    from graphify.fs.launcher import login_item_path, stop_server

    idx = resolve_index(args.target)
    stopped = stop_server(idx)
    print("explorer stopped" if stopped else "the explorer was not running")
    if stopped and login_item_path(idx):
        print("(it starts again at your next login; `graphify fs app uninstall` removes the login item)")
    return 0


def cmd_app(args) -> int:
    from graphify.fs import launcher

    idx = resolve_index(args.target)
    if args.action == "uninstall":
        removed = launcher.uninstall(idx, apps_dir=args.dir)
        print("\n".join(f"removed {r}" for r in removed) or "nothing to remove")
        return 0
    try:
        app = launcher.install_app(idx, name=args.name, apps_dir=args.dir)
    except (RuntimeError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"launcher app: {app}")
    print("  open it from Spotlight, Launchpad or the Dock (drag it there to keep it)")
    for note in launcher.install_notes(idx):
        print(f"note: {note}")
    if args.login:
        try:
            agent = launcher.install_login_item(idx, app)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"login item: {agent}")
        print("  the explorer now runs in the background and keeps the index up to date")
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
                  f"{s.get('offline', 0)} iCloud-only · {s.get('unread', 0)} not read by the model yet")
        from graphify.fs.launcher import login_item_path, running_server

        info = running_server(idx)
        if info:
            watching = " · watching for changes" if info.get("watching") else ""
            print(f"  explorer running (pid {info.get('pid')}){watching}: graphify fs open")
        if login_item_path(idx):
            print("  starts at login")
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
    handlers = {"setup": cmd_setup, "index": cmd_index, "explore": cmd_explore, "serve": cmd_serve,
                "open": cmd_open, "stop": cmd_stop, "app": cmd_app, "mcp": cmd_mcp,
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
