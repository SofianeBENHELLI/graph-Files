"""graphify fs: folder watching, index locks, the background server and the launcher app."""
from __future__ import annotations

import os
import plistlib
import stat
import sys
import time
import unicodedata
from pathlib import Path

import pytest

from graphify.fs.config import FsIndexPaths, default_index_dir, load_state
from graphify.fs.indexer import IndexOptions, run_index
from graphify.fs.lock import IndexBusy, IndexLock
from graphify.fs.watch import BUSY_RETRY, MODEL_RETRY, ChangeBatch, Updater, relevant_rel
from tests.test_fs import make_corpus, write_content_graph

macos_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_FS_HOME", str(tmp_path / "home"))
    root = make_corpus(tmp_path / "docs").resolve()
    index = default_index_dir(root)
    write_content_graph(root, index)
    run_index(root, IndexOptions(metadata_only=True), log=lambda m: None)
    return root, index


# ---------------------------------------------------------------- event filtering
def test_relevant_rel_mirrors_scan_rules(tmp_path):
    root = tmp_path.resolve()
    assert relevant_rel(root, str(root)) == ""
    assert relevant_rel(root, f"{root}/Notes/meeting.txt") == "Notes/meeting.txt"
    assert relevant_rel(root, f"{root}/Notes/.hidden.txt") is None
    assert relevant_rel(root, f"{root}/.DS_Store") is None
    assert relevant_rel(root, f"{root}/node_modules/x/index.js") is None
    assert relevant_rel(root, f"{root}/a/.git/HEAD") is None
    assert relevant_rel(root, f"{root}/Finance/~$budget.docx") is None
    # a change inside a macOS package is a change to the package
    assert relevant_rel(root, f"{root}/Deck.key/Data/image.png") == "Deck.key"
    assert relevant_rel(root, f"{tmp_path.parent}/elsewhere.txt") is None
    assert relevant_rel(root, f"{root}-sibling/x.txt") is None
    # FSEvents may report decomposed (NFD) names
    nfd = unicodedata.normalize("NFD", f"{root}/Résumé.pdf")
    assert relevant_rel(root, nfd) == "Résumé.pdf"
    assert relevant_rel(root, f"{root}/idx/fs/graph.json", ignore=(f"{root}/idx",)) is None


def test_relevant_rel_skips_library_only_for_home(monkeypatch, tmp_path):
    home = (tmp_path / "home").resolve()
    (home / "Library").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert relevant_rel(home, f"{home}/Library/Caches/x") is None
    assert relevant_rel(home, f"{home}/Documents/Library/x.txt") == "Documents/Library/x.txt"
    other = (tmp_path / "docs").resolve()
    assert relevant_rel(other, f"{other}/Library/x.txt") == "Library/x.txt"


def test_change_batch_debounces():
    b = ChangeBatch()
    assert not b.due(0, 3, 60)
    b.add("a", 0.0)
    b.add("b", 2.0)
    assert not b.due(4.0, 3, 60)       # still changing
    assert b.due(5.0, 3, 60)           # calm for 3 s
    b2 = ChangeBatch()
    for t in range(0, 70, 2):          # never calm: refresh anyway after max_wait
        b2.add(f"f{t}", float(t))
    assert b2.due(68.0, 3, 60)
    assert b.take() == {"a", "b"} and len(b) == 0 and not b.due(100, 3, 60)
    b.put_back({"c"}, 10.0)
    assert len(b) == 1 and b.due(13.0, 3, 60)


# ---------------------------------------------------------------- scheduling
class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _updater(tmp_path, results, *, ready=(True, ""), content=True):
    clock = FakeClock()
    calls: list[bool] = []

    def fake_run(root, opts, log=None):
        calls.append(opts.metadata_only)
        assert opts.wait_for_lock is False
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    (tmp_path / "idx").mkdir(exist_ok=True)
    u = Updater(tmp_path, tmp_path / "idx", watch=False, content=content, quiet=3, max_wait=60,
                content_delay=300, clock=clock, run_index=fake_run, models_ready=lambda m: ready)
    return u, clock, calls


def test_updater_quick_then_content_refresh(tmp_path):
    u, clock, calls = _updater(tmp_path, [{"unread": 2}, {"unread": 0}])
    u.notify("Notes/new.md")
    assert u.tick() is None and u.state == "waiting"     # not calm yet
    clock.t += 3
    assert u.tick() == "quick" and calls == [True]
    assert u.unread == 2 and u.last_changes == 1
    assert u.status()["content_in"] == 300
    clock.t += 299
    assert u.tick() is None and calls == [True]
    clock.t += 1
    assert u.tick() == "content" and calls == [True, False]
    assert u.unread == 0 and u.status()["content_in"] is None
    clock.t += 1000
    assert u.tick() is None and u.state == "idle"


def test_updater_new_changes_push_the_content_pass_back(tmp_path):
    u, clock, calls = _updater(tmp_path, [{"unread": 1}, {"unread": 2}, {"unread": 0}])
    u.request()
    assert u.tick() == "quick"
    clock.t += 200
    u.notify("a.md")
    clock.t += 3
    assert u.tick() == "quick"                            # content pass reset to +300 s
    clock.t += 200
    assert u.tick() is None
    clock.t += 100
    assert u.tick() == "content" and calls == [True, True, False]


def test_updater_waits_for_the_model_and_retries_when_busy(tmp_path):
    u, clock, calls = _updater(tmp_path, [{"unread": 3}], ready=(False, "local model not running"))
    u.request()
    assert u.tick() == "quick"
    clock.t += 300
    assert u.tick() == "no-model" and u.status()["note"] == "local model not running"
    assert u.status()["content_in"] == MODEL_RETRY and calls == [True]

    u2, clock2, calls2 = _updater(tmp_path, [IndexBusy("pid 1"), {"unread": 0}])
    u2.notify("x.md")
    clock2.t += 3
    assert u2.tick() == "busy" and len(u2.batch) == 1     # changes kept for the retry
    clock2.t += BUSY_RETRY - 1
    assert u2.tick() is None
    clock2.t += 1
    assert u2.tick() == "quick" and calls2 == [True, True] and len(u2.batch) == 0


def test_updater_manual_content_request_and_disabled_content(tmp_path):
    u, clock, calls = _updater(tmp_path, [{"unread": 0}])
    u.request(content=True)
    assert u.tick() == "content" and calls == [False]
    u2, clock2, calls2 = _updater(tmp_path, [{"unread": 5}], content=False)
    u2.request()
    assert u2.tick() == "quick"
    clock2.t += 10_000
    assert u2.tick() is None and calls2 == [True]         # never starts the model itself


# ---------------------------------------------------------------- locks and the indexer
def test_index_lock_is_exclusive_and_released(tmp_path):
    a, b = IndexLock(tmp_path / "x.lock"), IndexLock(tmp_path / "x.lock")
    a.acquire(wait=False)
    with pytest.raises(IndexBusy):
        b.acquire(wait=False)
    assert "pid" in a.holder()
    a.release()
    b.acquire(wait=False)
    b.release()


def test_quick_refresh_runs_during_a_content_pass_but_not_during_assembly(indexed):
    root, index = indexed
    paths = FsIndexPaths(index)
    content = IndexLock(paths.content_lock)
    content.acquire(wait=False)          # a long content pass is running
    try:
        stats = run_index(root, IndexOptions(index=index, metadata_only=True, wait_for_lock=False),
                          log=lambda m: None)
        assert stats["files"] == 4
        with pytest.raises(IndexBusy):   # a second content pass must wait its turn
            run_index(root, IndexOptions(index=index, wait_for_lock=False), log=lambda m: None)
    finally:
        content.release()
    held = IndexLock(paths.lock)
    held.acquire(wait=False)
    try:
        with pytest.raises(IndexBusy):
            run_index(root, IndexOptions(index=index, metadata_only=True, wait_for_lock=False),
                      log=lambda m: None)
    finally:
        held.release()


def test_unread_counts_files_newer_than_the_last_content_pass(indexed):
    root, index = indexed
    stats = load_state(index).stats
    assert stats["unread"] == 3          # no content pass yet: md, docx, txt (not the .key package)
    from graphify.fs.config import save_state

    state = load_state(index)
    state.last_content_pass = time.time() + 5
    save_state(index, state)
    assert run_index(root, IndexOptions(index=index, metadata_only=True), log=lambda m: None)["unread"] == 0


def test_metadata_refresh_never_starts_whisper(indexed, monkeypatch):
    root, index = indexed
    (root / "memo.m4a").write_bytes(b"not really audio")
    import graphify.transcribe as tr

    def boom(*a, **k):
        raise AssertionError("whisper must not run on a quick refresh")
    monkeypatch.setattr(tr, "transcribe", boom)
    monkeypatch.setattr(tr, "_get_whisper", boom)
    stats = run_index(root, IndexOptions(index=index, metadata_only=True), log=lambda m: None)
    assert stats["files"] == 5 and stats["unread"] == 4


# ---------------------------------------------------------------- real folder watching
def test_watcher_picks_up_new_files(indexed):
    pytest.importorskip("watchdog")
    root, index = indexed
    from graphify.fs.tools import FsIndex

    idx = FsIndex(index)
    u = Updater(root, index, watch=True, content=False, quiet=0.3, max_wait=5).start()
    try:
        assert u.watching
        deadline = time.time() + 20
        while time.time() < deadline and (u.state != "idle" or u.last_update is None):
            time.sleep(0.1)  # startup catch-up refresh
        (root / "Notes/kickoff.md").write_text("# Kickoff\nThe Orion launch plan.\n")
        found = None
        while time.time() < deadline:
            idx.maybe_reload()
            found = idx.search("Orion")["files"]
            if found:
                break
            time.sleep(0.2)
        assert found and found[0]["path"] == "Notes/kickoff.md"
        assert u.status()["watching"] and u.status()["graph_version"]
    finally:
        u.stop()
    assert not u.watching


# ---------------------------------------------------------------- explorer API
def test_explorer_refresh_goes_through_the_updater(indexed):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    from graphify.fs.explorer.app import create_app
    from graphify.fs.tools import FsIndex

    root, index = indexed
    app = create_app(FsIndex(index), token="tok", port=8765, finder_write=False, extra_hosts={"testserver"})
    c = TestClient(app)
    h = {"X-Gfs-Token": "tok"}
    s = c.get("/api/refresh", headers=h).json()
    assert {"state", "watching", "pending", "unread", "graph_version", "log"} <= set(s)
    assert s["watching"] is False
    before = s["graph_version"]
    (root / "Notes/extra.md").write_text("# Extra\n")
    time.sleep(0.05)
    assert c.post("/api/refresh", json={}, headers=h).status_code == 200
    deadline = time.time() + 20
    while time.time() < deadline:
        s = c.get("/api/refresh", headers=h).json()
        if s["last_changes"] is not None and s["state"] == "idle" and s["graph_version"] != before:
            break
        time.sleep(0.1)
    assert s["graph_version"] != before
    assert c.get("/api/files?name=extra", headers=h).json()["files"]


# ---------------------------------------------------------------- background server
def test_runtime_file_is_private_and_dead_servers_are_ignored(tmp_path):
    from graphify.fs import launcher

    idx = tmp_path / "idx"
    p = launcher.write_runtime(idx, {"pid": 999999, "port": 9, "token": "t", "url": "u"})
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert launcher.read_runtime(idx)["token"] == "t"
    assert not launcher.server_alive(launcher.read_runtime(idx))
    assert launcher.running_server(idx) is None
    assert launcher.stop_server(idx) is False and launcher.read_runtime(idx) is None


def test_open_starts_one_background_server_and_stop_ends_it(indexed):
    pytest.importorskip("uvicorn")
    from graphify.fs import launcher

    root, index = indexed
    info = launcher.ensure_server(index, port=18765, extra=["--no-watch"])
    try:
        assert info["url"].startswith("http://127.0.0.1:") and launcher.server_alive(info)
        assert info["index"] == str(index) and info["watching"] is False
        again = launcher.ensure_server(index)
        assert again["pid"] == info["pid"]          # reused, not a second server
        import subprocess

        dup = subprocess.run(launcher.server_command(index), capture_output=True, text=True, timeout=60)
        assert dup.returncode == 0 and "already running" in dup.stdout
        import http.cookiejar
        import urllib.request

        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        with opener.open(info["url"], timeout=5) as resp:   # token link sets the cookie and redirects
            assert resp.status == 200 and b"<html" in resp.read().lower()
    finally:
        assert launcher.stop_server(index)
    assert launcher.running_server(index) is None and launcher.read_runtime(index) is None


# ---------------------------------------------------------------- launcher app
@macos_only
def test_install_app_and_login_item(indexed, tmp_path):
    from graphify.fs import launcher

    root, index = indexed
    apps, agents = tmp_path / "Apps", tmp_path / "Agents"
    app = launcher.install_app(index, apps_dir=apps, icon=False)
    assert app == apps / "Graph Files.app"
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert info[launcher.APP_INDEX_KEY] == str(index.resolve()) and info["LSUIElement"] is True
    exe = app / "Contents/MacOS" / info["CFBundleExecutable"]
    assert os.access(exe, os.X_OK)
    script = exe.read_text()
    assert " fs open " in script and str(index) in script and sys.executable in script
    assert "osascript" in script and '"--no-browser"' in script
    assert launcher.install_app(index, apps_dir=apps, icon=False) == app      # reinstall replaces
    # another folder gets its own name instead of taking over the first app
    other_root = make_corpus(tmp_path / "other")
    other = default_index_dir(other_root.resolve())
    run_index(other_root, IndexOptions(metadata_only=True), log=lambda m: None)
    assert launcher.install_app(other, apps_dir=apps, icon=False).name == "Graph Files - other.app"
    foreign = apps / "Mine.app"
    foreign.mkdir()
    with pytest.raises(FileExistsError):
        launcher.install_app(index, name="Mine", apps_dir=apps, icon=False)

    calls = []
    plist = launcher.install_login_item(index, app, agents_dir=agents,
                                        launchctl=lambda *a: calls.append(a) or _ok())
    agent = plistlib.loads(plist.read_bytes())
    assert agent["Label"] == launcher.agent_label(index) and agent["RunAtLoad"] is True
    # started through the app, so it runs under the app's privacy permissions
    assert agent["ProgramArguments"] == ["/usr/bin/open", "-g", "-a", str(app.resolve()), "--args", "--no-browser"]
    assert agent["StartInterval"] == launcher.LOGIN_CHECK_SECONDS
    assert [c[0] for c in calls] == ["bootout", "bootstrap"]
    assert launcher.login_item_path(index, agents) == plist

    removed = launcher.uninstall(index, apps_dir=apps, agents_dir=agents, launchctl=lambda *a: _ok())
    assert str(plist) in removed and str(app) in removed
    assert not app.exists() and not plist.exists() and (apps / "Graph Files - other.app").exists()


@macos_only
def test_app_icon_renders():
    import shutil

    if not all(shutil.which(t) for t in ("qlmanage", "sips", "iconutil")):
        pytest.skip("macOS icon tools missing")
    from graphify.fs.launcher import _make_icns

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "AppIcon.icns"
        assert _make_icns(dest) and dest.stat().st_size > 10_000


@macos_only
def test_launcher_script_runs_and_filters_arguments(indexed, tmp_path, monkeypatch):
    """The generated script runs `fs open` and only forwards --no-browser."""
    import subprocess

    from graphify.fs import launcher

    root, index = indexed
    fake = tmp_path / "fakepy"
    fake.write_text("#!/bin/sh\necho \"$@\" > \"$(dirname \"$0\")/args.txt\"\n")
    fake.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(fake))
    app = launcher.install_app(index, apps_dir=tmp_path / "Apps", icon=False)
    exe = app / "Contents/MacOS" / launcher.EXECUTABLE
    subprocess.run([str(exe), "-psn_0_12345", "--no-browser"], check=True, timeout=20)
    assert (tmp_path / "args.txt").read_text().split() == ["-m", "graphify", "fs", "open", str(index.resolve()),
                                                          "--no-browser"]
    subprocess.run([str(exe)], check=True, timeout=20)
    assert (tmp_path / "args.txt").read_text().split()[-1] == str(index.resolve())


def test_protected_locations(monkeypatch, tmp_path):
    from graphify.fs import launcher

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    icloud = tmp_path / "Library/Mobile Documents/com~apple~CloudDocs/Work"
    icloud.mkdir(parents=True)
    (tmp_path / "Documents").mkdir()
    assert launcher.protected_locations(icloud, tmp_path / "Documents/x", tmp_path / "Other") == [
        "iCloud Drive", "Documents"]
    assert launcher.protected_locations(tmp_path / "Other") == []


def _ok():
    import subprocess

    return subprocess.CompletedProcess([], 0, "", "")


# ---------------------------------------------------------------- CLI
def test_cli_status_open_and_app(indexed, capsys, tmp_path):
    from graphify.fs.cli import main

    root, index = indexed
    assert main(["status", str(root)]) == 0
    out = capsys.readouterr().out
    assert "not read by the model yet" in out and "explorer running" not in out
    if sys.platform == "darwin":
        assert main(["app", "install", str(root), "--dir", str(tmp_path / "Apps")]) == 0
        assert "launcher app:" in capsys.readouterr().out
        assert main(["app", "uninstall", str(root), "--dir", str(tmp_path / "Apps")]) == 0
        assert "removed" in capsys.readouterr().out
