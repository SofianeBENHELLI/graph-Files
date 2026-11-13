"""The fs explorer: a local web app to browse and chat with the knowledge graph.

Security model (it serves your files):

- binds to 127.0.0.1 only;
- every API call needs the per-launch token (set as an HttpOnly, SameSite=Strict
  cookie when you open the launch URL), so other web pages cannot call it;
- the Host header must be the local address (blocks DNS-rebinding);
- raw files, thumbnails and open/reveal actions are limited to files that are in
  the index, under the indexed root.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from graphify.fs.finder import FinderTagWriter, finder_tags_supported
from graphify.fs.tags import TagDecisions
from graphify.fs.tools import FsIndex, ToolError

STATIC = Path(__file__).parent / "static"
COOKIE = "gfs_token"
PREVIEW_KINDS = {"pdf", "image", "video", "audio", "text", "web", "code"}
TEXT_PREVIEW_MAX = 2 * 1024 * 1024


class ExplorerState:
    def __init__(self, index: FsIndex, token: str, *, port: int, finder_write: bool = True,
                 chat_model: str | None = None, extra_hosts: set[str] | None = None):
        self.index = index
        self.token = token
        self.port = port
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"} | set(extra_hosts or ())
        self.finder_write = finder_write and finder_tags_supported()
        self.chat_model = chat_model or index.state.chat_model
        self.refresh_lock = threading.Lock()
        self.refresh_status: dict = {"running": False, "last": None, "log": []}
        self.writer = FinderTagWriter(index.root, index.paths.tag_writes)

    # -- path safety ------------------------------------------------------
    def file_path(self, fid: str) -> Path:
        G = self.index.G
        if fid not in G or G.nodes[fid].get("fs_kind") != "file":
            raise ToolError("unknown file")
        rel = G.nodes[fid].get("path", "")
        path = (self.index.root / rel).resolve()
        if not path.is_relative_to(self.index.root.resolve()):
            raise ToolError("file outside the indexed folder")
        return path


def _json(data, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status)


def _err(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def create_app(index: FsIndex, *, token: str | None = None, port: int = 8765,
               finder_write: bool = True, chat_model: str | None = None,
               extra_hosts: set[str] | None = None) -> Starlette:
    """Build the ASGI app. ``extra_hosts`` is for tests (e.g. Starlette's "testserver")."""
    st = ExplorerState(index, token or secrets.token_urlsafe(24), port=port,
                       finder_write=finder_write, chat_model=chat_model, extra_hosts=extra_hosts)

    def q(request: Request, name: str, default=None):
        value = request.query_params.get(name)
        return value if value not in (None, "") else default

    def guarded(fn):
        async def wrapper(request: Request):
            try:
                return await fn(request)
            except ToolError as exc:
                return _err(str(exc), 404)
            except (ValueError, KeyError) as exc:
                return _err(f"bad request: {exc}", 400)
        return wrapper

    # -- pages ------------------------------------------------------------
    async def home(request: Request):
        t = request.query_params.get("t")
        if t is not None:
            if not secrets.compare_digest(t, st.token):
                return PlainTextResponse("invalid token", status_code=403)
            resp = RedirectResponse("/", status_code=303)
            resp.set_cookie(COOKIE, st.token, httponly=True, samesite="strict", max_age=7 * 86400)
            return resp
        if not secrets.compare_digest(request.cookies.get(COOKIE, ""), st.token):
            return HTMLResponse("<h3>graphify fs</h3><p>Open the link printed by <code>graphify fs explore</code>.</p>", status_code=403)
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    # -- read API ---------------------------------------------------------
    async def run_sync(fn, *args, **kwargs):
        from starlette.concurrency import run_in_threadpool

        st.index.maybe_reload()
        return await run_in_threadpool(fn, *args, **kwargs)

    @guarded
    async def api_config(request: Request):
        from graphify.fs.ollama import Ollama

        def probe():
            client = Ollama()
            try:
                ok = client.available()
                models = client.models() if ok else []
            except Exception:
                ok, models = False, []
            finally:
                client.close()
            return ok, models
        ok, models = await run_sync(probe)
        s = st.index.state
        return _json({
            "root": str(st.index.root), "root_name": st.index.root.name,
            "chat_model": st.chat_model, "extract_model": s.extract_model,
            "ollama": ok, "models": models,
            "chat_ready": ok and any(m == st.chat_model or m == st.chat_model + ":latest" for m in models),
            "finder_write": st.finder_write, "platform": sys.platform,
            "stats": s.stats, "last_index": s.last_index,
        })

    @guarded
    async def api_overview(request: Request):
        return _json(await run_sync(st.index.overview))

    @guarded
    async def api_facets(request: Request):
        return _json(await run_sync(st.index.facets))

    @guarded
    async def api_search(request: Request):
        query = q(request, "q", "")
        if not query.strip():
            return _json({"query": "", "files": [], "concepts": [], "tags": []})
        return _json(await run_sync(st.index.search, query, int(q(request, "limit", 40)), q(request, "kind")))

    @guarded
    async def api_files(request: Request):
        year = q(request, "year")
        after = q(request, "after") or (f"{year}-01-01" if year else None)
        before = q(request, "before") or (f"{int(year) + 1}-01-01" if year else None)
        return _json(await run_sync(
            st.index.find_files, tag=q(request, "tag"), kind=q(request, "kind"), folder=q(request, "folder"),
            topic=q(request, "topic"), name=q(request, "name"), modified_after=after, modified_before=before,
            sort=q(request, "sort", "modified"), limit=int(q(request, "limit", 200)), offset=int(q(request, "offset", 0)),
        ))

    @guarded
    async def api_folder(request: Request):
        return _json(await run_sync(st.index.list_folder, q(request, "path", ""), int(q(request, "limit", 500))))

    @guarded
    async def api_file(request: Request):
        return _json(await run_sync(st.index.get_file, q(request, "id")))

    @guarded
    async def api_related(request: Request):
        return _json(await run_sync(st.index.related_files, q(request, "id"), int(q(request, "limit", 12))))

    @guarded
    async def api_neighborhood(request: Request):
        return _json(await run_sync(st.index.neighborhood, q(request, "id")))

    @guarded
    async def api_topicmap(request: Request):
        return _json(await run_sync(st.index.topic_map))

    @guarded
    async def api_explain(request: Request):
        return _json(await run_sync(st.index.explain, q(request, "name")))

    # -- files ------------------------------------------------------------
    @guarded
    async def api_raw(request: Request):
        fid = q(request, "id")
        path = st.file_path(fid)
        node = st.index.G.nodes[fid]
        if node.get("offline"):
            return _err("this file is in iCloud only (not downloaded)", 409)
        kind = node.get("kind")
        if kind not in PREVIEW_KINDS:
            return _err("no inline preview for this kind", 415)
        if node.get("ext") in (".heic", ".heif"):
            jpg = await run_sync(_heic_preview, path, st.index.paths.thumbs)
            if jpg:
                return FileResponse(jpg, media_type="image/jpeg")
            return _err("cannot convert HEIC", 415)
        if kind in ("text", "code", "web"):
            if path.stat().st_size > TEXT_PREVIEW_MAX:
                return _err("file too large to preview", 413)
            text = path.read_text(encoding="utf-8", errors="replace")
            return PlainTextResponse(text, headers={"Content-Security-Policy": "default-src 'none'"})
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, headers={
            "Content-Disposition": "inline", "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"})

    @guarded
    async def api_thumb(request: Request):
        fid = q(request, "id")
        path = st.file_path(fid)
        node = st.index.G.nodes[fid]
        if node.get("offline"):
            return Response(status_code=404)
        size = min(max(int(q(request, "s", 320)), 64), 1024)
        png = await run_sync(_thumbnail, path, st.index.paths.thumbs, size, float(node.get("mtime") or 0))
        if png is None:
            return Response(status_code=404)
        return FileResponse(png, media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @guarded
    async def api_open(request: Request):
        body = await request.json()
        path = st.file_path(body.get("id"))
        action = body.get("action", "open")
        _open_path(path, action)
        return _json({"ok": True})

    # -- tags -------------------------------------------------------------
    @guarded
    async def api_tag(request: Request):
        body = await request.json()
        fid = body.get("id")
        name = str(body.get("name", "")).strip()
        action = body.get("action")
        if action not in ("accept", "reject", "add", "remove") or not name:
            return _err("action must be accept/reject/add/remove with a tag name")
        path = st.file_path(fid)
        rel = st.index.G.nodes[fid]["path"]
        decisions = TagDecisions.load(st.index.paths.tags_state)
        getattr(decisions, action)(rel, name)
        decisions.save(st.index.paths.tags_state)
        finder_note = None
        write = body.get("finder", True) and st.finder_write
        if write and action in ("accept", "add"):
            try:
                st.writer.add(path, [name])
                finder_note = "saved to Finder tags"
            except Exception as exc:
                finder_note = f"Finder tag not written: {exc}"
        elif write and action == "remove":
            try:
                st.writer.remove(path, [name])
                finder_note = "removed from Finder tags"
            except Exception as exc:
                finder_note = f"Finder tag not removed: {exc}"
        result = st.index.apply_tag_change(fid, name, action)
        return _json({**result, "finder": finder_note})

    @guarded
    async def api_undo_tags(request: Request):
        from starlette.concurrency import run_in_threadpool

        done = await run_in_threadpool(st.writer.undo, 1)
        return _json({"undone": done})

    # -- refresh (fast, metadata-only rebuild) ------------------------------
    async def api_refresh(request: Request):
        if st.refresh_status["running"]:
            return _json(st.refresh_status)

        def job():
            from graphify.fs.indexer import IndexOptions, run_index

            log: list[str] = []
            st.refresh_status.update(running=True, log=log, started=time.time())
            try:
                run_index(st.index.root, IndexOptions(index=st.index.index_dir, metadata_only=True),
                          log=lambda m: log.append(m))
                st.index.maybe_reload()
                st.refresh_status["last"] = time.time()
            except Exception as exc:
                log.append(f"refresh failed: {exc}")
            finally:
                st.refresh_status["running"] = False

        threading.Thread(target=job, daemon=True).start()
        return _json({"running": True})

    async def api_refresh_status(request: Request):
        s = dict(st.refresh_status)
        s["log"] = s.get("log", [])[-8:]
        return _json(s)

    # -- chat ---------------------------------------------------------------
    async def api_chat(request: Request):
        body = await request.json()
        question = str(body.get("question", "")).strip()
        if not question:
            return _err("empty question")
        history = body.get("history") or []
        from graphify.fs.agent import Navigator
        from graphify.fs.ollama import Ollama

        def events():
            client = Ollama()
            try:
                nav = Navigator(st.index, client, model=st.chat_model)
                for ev in nav.run(question, history):
                    yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            finally:
                client.close()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    routes = [
        Route("/", home),
        Route("/api/config", api_config),
        Route("/api/overview", api_overview),
        Route("/api/facets", api_facets),
        Route("/api/search", api_search),
        Route("/api/files", api_files),
        Route("/api/folder", api_folder),
        Route("/api/file", api_file),
        Route("/api/related", api_related),
        Route("/api/neighborhood", api_neighborhood),
        Route("/api/topicmap", api_topicmap),
        Route("/api/explain", api_explain),
        Route("/api/raw", api_raw),
        Route("/api/thumb", api_thumb),
        Route("/api/open", api_open, methods=["POST"]),
        Route("/api/tag", api_tag, methods=["POST"]),
        Route("/api/undo-tags", api_undo_tags, methods=["POST"]),
        Route("/api/refresh", api_refresh, methods=["POST"]),
        Route("/api/refresh", api_refresh_status, methods=["GET"]),
        Route("/api/chat", api_chat, methods=["POST"]),
        Mount("/static", app=StaticFiles(directory=str(STATIC)), name="static"),
    ]
    app = Starlette(routes=routes)
    app.state.explorer = st
    return _Guard(app, st)


class _Guard:
    """ASGI middleware: Host check for everything, token check for /api."""

    def __init__(self, app, st: ExplorerState):
        self.app = app
        self.st = st
        self.state = app.state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host = headers.get("host", "")
        allowed = self.st.allowed_hosts
        if host not in allowed:
            return await PlainTextResponse("forbidden host", status_code=403)(scope, receive, send)
        path = scope.get("path", "")
        if path.startswith("/api/"):
            cookie = ""
            for part in headers.get("cookie", "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE:
                    cookie = v
            token = headers.get("x-gfs-token", "") or cookie
            if not token or not secrets.compare_digest(token, self.st.token):
                return await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
            origin = headers.get("origin")
            if scope.get("method") not in ("GET", "HEAD") and origin and origin.split("//", 1)[-1] not in allowed:
                return await JSONResponse({"error": "cross-origin request refused"}, status_code=403)(scope, receive, send)
        return await self.app(scope, receive, send)


# ---------------------------------------------------------------------- helpers
def _cache_name(path: Path, size: int, mtime: float, suffix: str) -> str:
    return hashlib.sha1(f"{path}|{size}|{int(mtime)}".encode()).hexdigest()[:20] + suffix


def _thumbnail(path: Path, cache_dir: Path, size: int, mtime: float) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / _cache_name(path, size, mtime, ".png")
    if out.exists():
        return out
    ql = shutil.which("qlmanage")
    if ql:
        tmp = cache_dir / f"tmp-{os.getpid()}-{threading.get_ident()}"
        tmp.mkdir(exist_ok=True)
        try:
            subprocess.run([ql, "-t", "-s", str(size), "-o", str(tmp), str(path)],
                           capture_output=True, timeout=25, check=False)
            made = list(tmp.glob("*.png"))
            if made:
                os.replace(made[0], out)
                return out
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    try:  # non-macOS fallback for plain images
        from PIL import Image

        with Image.open(path) as im:
            im.thumbnail((size, size))
            im.save(out, "PNG")
        return out
    except Exception:
        return None


def _heic_preview(path: Path, cache_dir: Path) -> Path | None:
    st = path.stat()
    out = cache_dir / _cache_name(path, st.st_size, st.st_mtime, ".jpg")
    if out.exists():
        return out
    from graphify.llm import _heic_to_jpeg

    data = _heic_to_jpeg(path)
    if not data:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def _open_path(path: Path, action: str) -> None:
    if sys.platform == "darwin":
        if action == "reveal":
            cmd = ["open", "-R", str(path)]
        elif action == "quicklook":
            cmd = ["qlmanage", "-p", str(path)]
        else:
            cmd = ["open", str(path)]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    elif sys.platform == "win32":
        if action == "reveal":
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        target = path.parent if action == "reveal" else path
        subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def serve(index: FsIndex, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          finder_write: bool = True, chat_model: str | None = None) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the explorer only binds to localhost")
    port = _free_port(port)
    token = secrets.token_urlsafe(24)
    app = create_app(index, token=token, port=port, finder_write=finder_write, chat_model=chat_model)
    url = f"http://127.0.0.1:{port}/?t={token}"
    print(f"[graphify fs] explorer for {index.root}", flush=True)
    print(f"[graphify fs] open: {url}", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: _open_url(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _open_url(url: str) -> None:
    import webbrowser

    webbrowser.open(url)


def _free_port(preferred: int) -> int:
    import socket

    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError("no free port")
