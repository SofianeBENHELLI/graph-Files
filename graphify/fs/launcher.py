"""Start, find and stop the explorer server, and the macOS launcher app.

- ``graphify fs serve`` runs the explorer in the background (no browser) and
  records its address in ``fs/explorer.json`` (owner-only file: it holds the
  access token). One server per index: it holds ``fs/explorer.lock``.
- ``graphify fs open`` reuses the running server or starts one, then opens the
  browser. The launcher app and the login item call it.
- ``graphify fs app install`` writes ``~/Applications/Graph Files.app`` (Dock,
  Spotlight, Launchpad) and, with ``--login``, a LaunchAgent that starts the
  server at login so the index is kept up to date even with no window open.

macOS privacy protection (TCC) guards iCloud Drive, Documents, Desktop and
Downloads per app. Processes started from Terminal use Terminal's permission;
the launcher app has its own. So the login item starts the server *through the
app* (``open -a``), and the user grants "Graph Files" access once.
"""
from __future__ import annotations

import json
import os
import plistlib
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from graphify.fs.config import FsIndexPaths, load_state

APP_NAME = "Graph Files"
APP_INDEX_KEY = "GraphifyFsIndex"   # Info.plist key marking apps this module made
AGENT_PREFIX = "io.graphify.fs."
EXECUTABLE = "graph-files"


# --------------------------------------------------------------------------- runtime file
def write_runtime(index_dir: Path, info: dict) -> Path:
    path = FsIndexPaths(index_dir).runtime
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    os.replace(tmp, path)
    return path


def read_runtime(index_dir: Path) -> dict | None:
    try:
        return json.loads(FsIndexPaths(index_dir).runtime.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear_runtime(index_dir: Path, pid: int | None = None) -> None:
    info = read_runtime(index_dir)
    if info is not None and (pid is None or info.get("pid") == pid):
        try:
            FsIndexPaths(index_dir).runtime.unlink()
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def server_alive(info: dict | None, timeout: float = 2.0) -> bool:
    """True when the server in ``info`` runs and answers with its token."""
    if not info or not info.get("port") or not info.get("token"):
        return False
    if info.get("pid") and not _pid_alive(int(info["pid"])):
        return False
    port = int(info["port"])
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/refresh",
                                 headers={"X-Gfs-Token": info["token"]})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def running_server(index_dir: Path) -> dict | None:
    info = read_runtime(index_dir)
    return info if server_alive(info) else None


def server_command(index_dir: Path, *, port: int | None = None, extra: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "graphify", "fs", "serve", str(index_dir)]
    if port:
        cmd += ["--port", str(port)]
    return cmd + list(extra or [])


def ensure_server(index_dir: Path, *, port: int | None = None, timeout: float = 40.0,
                  extra: list[str] | None = None) -> dict:
    """Return the running server of ``index_dir``, starting one in the background."""
    info = running_server(index_dir)
    if info:
        return info
    paths = FsIndexPaths(index_dir)
    paths.fs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(paths.server_log, "ab") as log:
        proc = subprocess.Popen(server_command(index_dir, port=port, extra=extra),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True, close_fds=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = read_runtime(index_dir)
        if server_alive(info, timeout=1.0):
            return info  # ours, or one another `open` started at the same time
        rc = proc.poll()
        if rc not in (None, 0):
            raise RuntimeError(f"the explorer server failed to start (exit {rc}); see {paths.server_log}\n"
                               + _tail(paths.server_log))
        time.sleep(0.25)
    raise TimeoutError(f"the explorer server did not start within {int(timeout)}s; see {paths.server_log}")


def stop_server(index_dir: Path, timeout: float = 10.0) -> bool:
    """Stop the background server of ``index_dir``; True if one was running."""
    info = read_runtime(index_dir)
    pid = int(info.get("pid") or 0) if info else 0
    if not pid or not _pid_alive(pid):
        clear_runtime(index_dir)
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.2)
    if _pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
    clear_runtime(index_dir, pid)
    return True


def open_url(url: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        import webbrowser

        webbrowser.open(url)


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# --------------------------------------------------------------------------- macOS app
def _apps_dir(apps_dir: Path | None) -> Path:
    return Path(apps_dir).expanduser() if apps_dir else Path.home() / "Applications"


def _agents_dir(agents_dir: Path | None) -> Path:
    return Path(agents_dir).expanduser() if agents_dir else Path.home() / "Library" / "LaunchAgents"


def agent_label(index_dir: Path) -> str:
    return AGENT_PREFIX + Path(index_dir).name


def _app_index(app: Path) -> str | None:
    try:
        with open(app / "Contents" / "Info.plist", "rb") as fh:
            return plistlib.load(fh).get(APP_INDEX_KEY)
    except (OSError, plistlib.InvalidFileException, ValueError):
        return None


def find_apps(index_dir: Path, apps_dir: Path | None = None) -> list[Path]:
    d = _apps_dir(apps_dir)
    if not d.is_dir():
        return []
    want = str(Path(index_dir).resolve())
    return [a for a in sorted(d.glob("*.app")) if _app_index(a) == want]


_WAITING_NOTE = ("Graph Files is waiting for macOS permission to read your files. "
                 "Look for a dialog, or allow it in System Settings > Privacy & Security.")


def _launcher_script(index_dir: Path) -> str:
    cmd = " ".join(shlex.quote(c) for c in [sys.executable, "-m", "graphify", "fs", "open", str(index_dir)])
    note = _WAITING_NOTE.replace('"', "'")
    return ("#!/bin/sh\n"
            "# Opens the graphify fs explorer. Generated by `graphify fs app install`.\n"
            "# If macOS holds the launch for a privacy permission, say so instead of doing nothing.\n"
            f"( sleep 20; kill -0 $$ 2>/dev/null && /usr/bin/osascript -e "
            f"'display notification \"{note}\" with title \"Graph Files\"' ) >/dev/null 2>&1 &\n"
            "NB=\"\"\n"
            "for a in \"$@\"; do [ \"$a\" = \"--no-browser\" ] && NB=\"--no-browser\"; done\n"
            f"exec {cmd} $NB\n")


def protected_locations(*paths: Path) -> list[str]:
    """Human names of the macOS privacy-protected places among ``paths``."""
    home = Path.home()
    places = [
        (home / "Library" / "Mobile Documents", "iCloud Drive"),
        (home / "Library" / "CloudStorage", "cloud storage (Dropbox, OneDrive, Google Drive...)"),
        (home / "Documents", "Documents"),
        (home / "Desktop", "Desktop"),
        (home / "Downloads", "Downloads"),
        (Path("/Volumes"), "external drives"),
    ]
    found: list[str] = []
    for p in paths:
        try:
            rp = Path(p).expanduser().resolve()
        except OSError:
            continue
        for base, name in places:
            if (rp == base or rp.is_relative_to(base)) and name not in found:
                found.append(name)
    return found


def install_notes(index_dir: Path) -> list[str]:
    """What the user should know before the first launch (privacy prompts)."""
    import graphify

    state = load_state(index_dir)
    notes: list[str] = []
    code = Path(graphify.__file__).resolve().parent
    code_places = protected_locations(code, Path(sys.executable))
    if code_places:
        notes.append(f"graphify itself runs from {code.parent} ({', '.join(code_places)}), which macOS "
                     "protects: the app needs permission to read it. For everyday use, install a regular "
                     f"copy outside it: uv tool install --python 3.12 \"{code.parent}[fs,video]\" "
                     "(no --editable), then run graphify fs app install again.")
    folder_places = protected_locations(Path(state.root)) if state else []
    if folder_places:
        notes.append(f"The first time it opens, macOS asks to let \"{APP_NAME}\" access "
                     f"{', '.join(folder_places)}: click Allow. If no dialog appears, allow it in "
                     "System Settings > Privacy & Security > Files and Folders.")
    return notes


def _make_icns(dest: Path) -> bool:
    """Render the explorer icon into an .icns (macOS tools only; best effort)."""
    tools = [shutil.which(t) for t in ("qlmanage", "sips", "iconutil")]
    if not all(tools):
        return False
    ql, sips, iconutil = tools
    svg = (Path(__file__).parent / "explorer" / "static" / "icon.svg").read_text(encoding="utf-8")
    # Apple's icon grid leaves a margin around the rounded square.
    svg = svg.replace('viewBox="0 0 64 64"', 'viewBox="-7.75 -7.75 79.5 79.5"', 1)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "AppIcon.svg").write_text(svg, encoding="utf-8")
            subprocess.run([ql, "-t", "-s", "1024", "-o", tmp, str(t / "AppIcon.svg")],
                           capture_output=True, timeout=30, check=False)
            png = t / "AppIcon.svg.png"
            if not png.exists():
                return False
            iconset = t / "AppIcon.iconset"
            iconset.mkdir()
            for size in (16, 32, 128, 256, 512):
                for scale, suffix in ((1, ""), (2, "@2x")):
                    px = size * scale
                    subprocess.run([sips, "-z", str(px), str(px), str(png), "--out",
                                    str(iconset / f"icon_{size}x{size}{suffix}.png")],
                                   capture_output=True, timeout=30, check=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([iconutil, "-c", "icns", str(iconset), "-o", str(dest)],
                           capture_output=True, timeout=30, check=True)
            return dest.exists()
    except (OSError, subprocess.SubprocessError):
        return False


def install_app(index_dir: Path, *, name: str | None = None, apps_dir: Path | None = None,
                icon: bool = True) -> Path:
    """Write the launcher app for ``index_dir`` and return its path."""
    if sys.platform != "darwin":
        raise RuntimeError("the launcher app is macOS only; use `graphify fs open` instead")
    index_dir = Path(index_dir).resolve()
    state = load_state(index_dir)
    if state is None:
        raise FileNotFoundError(f"no graphify fs index at {index_dir}")
    root = Path(state.root)
    d = _apps_dir(apps_dir)
    d.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = APP_NAME
        current = _app_index(d / f"{name}.app")
        if current and current != str(index_dir):  # another folder already has the plain name
            name = f"{APP_NAME} - {root.name}"
    app = d / f"{name}.app"
    if app.exists():
        if _app_index(app) is None:
            raise FileExistsError(f"{app} exists and was not made by graphify fs; pick another --name")
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    exe = macos / EXECUTABLE
    exe.write_text(_launcher_script(index_dir), encoding="utf-8")
    exe.chmod(0o755)
    info = {
        "CFBundleName": name[:15],
        "CFBundleDisplayName": name,
        "CFBundleIdentifier": f"{AGENT_PREFIX}app.{index_dir.name}",
        "CFBundleExecutable": EXECUTABLE,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "11.0",
        "LSUIElement": True,  # no Dock icon of its own: it only opens the browser
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": f"Knowledge-graph explorer for {root}",
        APP_INDEX_KEY: str(index_dir),
    }
    if icon and _make_icns(app / "Contents" / "Resources" / "AppIcon.icns"):
        info["CFBundleIconFile"] = "AppIcon"
    with open(app / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump(info, fh)
    lsregister = Path("/System/Library/Frameworks/CoreServices.framework/Frameworks/"
                      "LaunchServices.framework/Support/lsregister")
    if lsregister.exists():  # so Spotlight and Launchpad see it right away
        subprocess.run([str(lsregister), "-f", str(app)], capture_output=True, timeout=30, check=False)
    return app


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=30, check=False)


LOGIN_CHECK_SECONDS = 900  # the login item re-checks the server this often


def install_login_item(index_dir: Path, app: Path, *, agents_dir: Path | None = None,
                       launchctl=_launchctl, start: bool = True) -> Path:
    """LaunchAgent that starts the explorer server through ``app`` at login.

    It runs ``open -g -a <app> --args --no-browser``: the server then runs as the
    app, under the privacy permissions granted to it. It runs again every
    ``LOGIN_CHECK_SECONDS`` and restarts the server if it stopped (cheap no-op
    otherwise).
    """
    if sys.platform != "darwin":
        raise RuntimeError("login items are macOS only")
    index_dir = Path(index_dir).resolve()
    paths = FsIndexPaths(index_dir)
    label = agent_label(index_dir)
    plist = {
        "Label": label,
        "ProgramArguments": ["/usr/bin/open", "-g", "-a", str(Path(app).resolve()), "--args", "--no-browser"],
        "RunAtLoad": True,
        "StartInterval": LOGIN_CHECK_SECONDS,
        "StandardOutPath": str(paths.server_log),
        "StandardErrorPath": str(paths.server_log),
    }
    d = _agents_dir(agents_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{label}.plist"
    with open(path, "wb") as fh:
        plistlib.dump(plist, fh)
    if start:  # runs now too (RunAtLoad): a server that already runs is simply kept
        domain = f"gui/{os.getuid()}"
        launchctl("bootout", f"{domain}/{label}")
        res = launchctl("bootstrap", domain, str(path))
        if res.returncode != 0:
            raise RuntimeError(f"launchctl bootstrap failed: {(res.stderr or res.stdout).strip()}")
    return path


def uninstall(index_dir: Path, *, apps_dir: Path | None = None, agents_dir: Path | None = None,
              launchctl=_launchctl) -> list[str]:
    """Remove the login item and launcher app(s) of ``index_dir``; return what was removed."""
    index_dir = Path(index_dir).resolve()
    removed: list[str] = []
    label = agent_label(index_dir)
    plist = _agents_dir(agents_dir) / f"{label}.plist"
    if plist.exists():
        if sys.platform == "darwin":
            launchctl("bootout", f"gui/{os.getuid()}/{label}")
        plist.unlink()
        removed.append(str(plist))
    for app in find_apps(index_dir, apps_dir):
        shutil.rmtree(app)
        removed.append(str(app))
    return removed


def login_item_path(index_dir: Path, agents_dir: Path | None = None) -> Path | None:
    path = _agents_dir(agents_dir) / f"{agent_label(Path(index_dir).resolve())}.plist"
    return path if path.exists() else None
