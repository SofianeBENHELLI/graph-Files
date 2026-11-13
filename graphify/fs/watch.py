"""Keep an fs index up to date while files change.

``Updater`` watches the indexed folder (watchdog: FSEvents on macOS) and runs
every refresh of one index, one at a time:

- **quick refresh** a few seconds after changes stop (names, folders, Finder
  tags, keywords, full text; cached content and topics). New and edited files
  show up and are searchable by their text within seconds.
- **content refresh** once the folder has been calm for ``content_delay``
  seconds and some files have not been read by the local model yet (new or
  edited since the last content pass). It runs the full index (content pass,
  transcripts, topic tags) in the background, only while Ollama has the models.
  Results are cached per file, so this only re-reads what changed.

The explorer's Refresh button goes through the same queue.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from graphify.fs.config import FsIndexPaths, load_state
from graphify.fs.lock import IndexBusy
from graphify.fs.metadata import PACKAGE_SUFFIXES, SKIP_DIR_NAMES, _skip_entry, nfc, skip_dir

QUIET_SECONDS = 3.0        # refresh once changes have stopped for this long
MAX_WAIT_SECONDS = 60.0    # ...or this long after the first change, if they never stop
CONTENT_DELAY = 300.0      # read new contents after 5 calm minutes
BUSY_RETRY = 15.0          # another run holds the index: try again later
MODEL_RETRY = 600.0        # Ollama or its models are missing: check again later
LOG_LINES = 12


def relevant_rel(root: Path, path: str, ignore: tuple[str, ...] = ()) -> str | None:
    """Root-relative path for an event at ``path``; None when the index ignores it.

    Mirrors ``scan``: hidden entries, noise folders and ``~/Library`` (when the
    root is the home folder) are ignored. A change inside a macOS package
    (``.key``, ``.app`` ...) is reported as a change to the package itself.
    """
    root_s = nfc(str(root)).rstrip("/")
    p = nfc(str(path))
    if p == root_s:
        return ""
    if not p.startswith(root_s + "/"):
        return None
    if any(p == i or p.startswith(i + "/") for i in ignore):
        return None
    parts = p[len(root_s) + 1:].split("/")
    parent = root
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if not part:
            return None
        if last:
            if _skip_entry(part) or part in SKIP_DIR_NAMES:
                return None
        else:
            if skip_dir(root, parent, part):
                return None
            if Path(part).suffix.lower() in PACKAGE_SUFFIXES:
                return "/".join(parts[: i + 1])
        parent = parent / part
    return "/".join(parts)


class ChangeBatch:
    """Changed paths collected between two refreshes (thread-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.paths: set[str] = set()
        self.first: float | None = None
        self.last: float | None = None

    def add(self, rel: str, now: float) -> None:
        with self._lock:
            self.paths.add(rel)
            self.first = self.first if self.first is not None else now
            self.last = now

    def due(self, now: float, quiet: float, max_wait: float) -> bool:
        with self._lock:
            if self.first is None or self.last is None:
                return False
            return now - self.last >= quiet or now - self.first >= max_wait

    def take(self) -> set[str]:
        with self._lock:
            out, self.paths = self.paths, set()
            self.first = self.last = None
            return out

    def put_back(self, paths: set[str], now: float) -> None:
        with self._lock:
            self.paths |= paths
            self.first = self.first if self.first is not None else now
            self.last = self.last if self.last is not None else now

    def __len__(self) -> int:
        with self._lock:
            return len(self.paths)


def models_ready(extract_model: str) -> tuple[bool, str]:
    """Is Ollama running with the extraction model? (ok, reason)"""
    from graphify.fs.ollama import Ollama

    client = Ollama(timeout=5)
    try:
        if not client.available():
            return False, "local model not running (open Ollama)"
        if not client.has_model(extract_model):
            return False, f"model {extract_model} not downloaded (run graphify fs setup)"
        return True, ""
    except Exception as exc:
        return False, f"local model unreachable: {exc}"
    finally:
        client.close()


class Updater:
    """Serialises refreshes of one index and (optionally) watches its folder."""

    def __init__(self, root: Path, index_dir: Path, *, watch: bool = True, content: bool = True,
                 quiet: float = QUIET_SECONDS, max_wait: float = MAX_WAIT_SECONDS,
                 content_delay: float = CONTENT_DELAY, log=None, on_update=None,
                 clock=time.monotonic, run_index=None, models_ready=models_ready):
        self.root = Path(root).expanduser().resolve()
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.paths = FsIndexPaths(self.index_dir)
        self.watch = watch
        self.content = content
        self.quiet, self.max_wait, self.content_delay = quiet, max_wait, content_delay
        self._log_fn = log
        self._on_update = on_update
        self._clock = clock
        if run_index is None:
            from graphify.fs.indexer import run_index
        self._run_index = run_index
        self._models_ready = models_ready
        self.batch = ChangeBatch()
        self._manual: str | None = None      # "quick" / "content" requested by the user
        self._retry_at = 0.0
        self._content_at: float | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer = None
        self.log_lines: list[str] = []
        self.state = "idle"                    # idle / waiting / updating / reading
        self.watching = False
        self.unread = 0
        self.last_update: float | None = None  # wall clock
        self.last_changes = 0
        self.note = ""
        self.error: str | None = None

    # -- logging / status ---------------------------------------------------
    def log(self, msg: str) -> None:
        self.log_lines = (self.log_lines + [msg])[-LOG_LINES:]
        if self._log_fn:
            self._log_fn(msg)

    def status(self) -> dict:
        try:
            version = self.paths.graph.stat().st_mtime
        except OSError:
            version = None
        content_in = None
        if self._content_at is not None:
            content_in = max(0, round(self._content_at - self._clock()))
        return {
            "running": self.state in ("updating", "reading"), "state": self.state,
            "watching": self.watching, "pending": len(self.batch), "unread": self.unread,
            "content_updates": self.content, "content_in": content_in, "note": self.note,
            "last": self.last_update, "last_changes": self.last_changes, "error": self.error,
            "graph_version": version, "log": self.log_lines[-8:],
        }

    # -- inputs -----------------------------------------------------------
    def notify(self, rel: str) -> None:
        self.batch.add(rel, self._clock())
        if self.state == "idle":
            self.state = "waiting"
        self._wake.set()

    def request(self, content: bool = False) -> None:
        """Refresh now (the explorer's Refresh button). ``content`` also reads contents."""
        self._manual = "content" if content else (self._manual or "quick")
        self._retry_at = 0.0
        if self.state == "idle":
            self.state = "waiting"
        self._wake.set()

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> Updater:
        state = load_state(self.index_dir)
        if state and state.stats:
            self.unread = int(state.stats.get("unread", 0) or 0)
            self.last_update = state.last_index
        if self.watch:
            self._start_observer()
            if self.watching:  # catch up with changes made while nobody watched
                self._manual = self._manual or "quick"
        self.ensure_running()
        return self

    def ensure_running(self) -> None:
        """Start the worker thread (without watching) if it is not running yet."""
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._loop, name="graphify-fs-updater", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception:
                pass
            self._observer = None
        self.watching = False
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def _start_observer(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            self.log("[graphify fs] folder watching needs watchdog: pip install watchdog")
            return
        updater = self
        ignore = (nfc(str(self.index_dir)),)

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.event_type in ("opened", "closed", "closed_no_write"):
                    return
                for p in (getattr(event, "src_path", None), getattr(event, "dest_path", None)):
                    if not p:
                        continue
                    p = p.decode() if isinstance(p, bytes) else p
                    rel = relevant_rel(updater.root, p, ignore)
                    if rel is not None:
                        updater.notify(rel)

        try:
            observer = Observer()
            observer.schedule(Handler(), str(self.root), recursive=True)
            observer.daemon = True
            observer.start()
        except Exception as exc:
            self.log(f"[graphify fs] cannot watch {self.root}: {exc}")
            return
        self._observer = observer
        self.watching = True
        self.log(f"[graphify fs] watching {self.root} for changes")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # never let the updater thread die
                self.error = str(exc)
                self.log(f"[graphify fs] update failed: {exc}")
                self.state = "idle"
            self._wake.wait(0.5)
            self._wake.clear()

    # -- the scheduler (called every ~0.5 s; tests call it directly) -------
    def tick(self) -> str | None:
        now = self._clock()
        if now < self._retry_at:
            return None
        manual, self._manual = self._manual, None
        if manual == "content":
            self.batch.take()
            return self._content_refresh(now, forced=True)
        if manual or self.batch.due(now, self.quiet, self.max_wait):
            return self._quick_refresh(now)
        if len(self.batch):
            self.state = "waiting"
            return None
        if (self.content and self._content_at is not None and now >= self._content_at):
            return self._content_refresh(now)
        self.state = "idle"
        return None

    def _run(self, *, metadata_only: bool) -> dict | None:
        from graphify.fs.indexer import IndexOptions

        opts = IndexOptions(index=self.index_dir, metadata_only=metadata_only, wait_for_lock=False)
        try:
            result = self._run_index(self.root, opts, log=self.log)
        except IndexBusy as exc:
            self.note = f"another index run is in progress ({exc}); retrying"
            return None
        self.error = None
        self.last_update = time.time()
        self.unread = int(result.get("unread", 0) or 0)
        if self._on_update:
            self._on_update()
        return result

    def _quick_refresh(self, now: float) -> str | None:
        changed = self.batch.take()
        self.state = "updating"
        try:
            result = self._run(metadata_only=True)
        finally:
            self.state = "idle"
        if result is None:
            self.batch.put_back(changed, now)
            self._retry_at = now + BUSY_RETRY
            return "busy"
        self.last_changes = len(changed)
        self.note = ""
        self._schedule_content(now)
        return "quick"

    def _schedule_content(self, now: float) -> None:
        if self.content and self.unread:
            self._content_at = now + self.content_delay
        elif not self.unread:
            self._content_at = None

    def _content_refresh(self, now: float, forced: bool = False) -> str | None:
        state = load_state(self.index_dir)
        model = state.extract_model if state else ""
        ok, reason = self._models_ready(model)
        if not ok:
            self.note = reason
            self._content_at = now + MODEL_RETRY
            if forced:  # still pick up names and tags
                return self._quick_refresh(now)
            return "no-model"
        self.state = "reading"
        self.note = f"reading {self.unread} file(s) with {model}" if self.unread else f"reading with {model}"
        try:
            result = self._run(metadata_only=False)
        finally:
            self.state = "idle"
        if result is None:
            self._content_at = now + BUSY_RETRY
            return "busy"
        self.note = ""
        self._content_at = None
        self._schedule_content(self._clock())
        return "content"
