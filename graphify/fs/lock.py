"""One index run at a time per index.

``graphify fs index``, the explorer's Refresh button and the folder watcher all
rebuild the same files (graph, caches, full-text database). An OS file lock on
``fs/index.lock`` serialises them across threads and processes. The lock is
released by the OS if the holder dies, so a crash never leaves it stuck.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt


class IndexBusy(RuntimeError):
    """Another run holds the index lock."""


class IndexLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def holder(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _try(self, fd: int) -> bool:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - Windows
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def acquire(self, wait: bool = True, on_wait=None, poll: float = 1.0) -> None:
        """Take the lock; raise ``IndexBusy`` if ``wait`` is False and it is held."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        notified = False
        while not self._try(fd):
            if not wait:
                os.close(fd)
                raise IndexBusy(self.holder() or "another index run")
            if on_wait and not notified:
                on_wait(self.holder())
                notified = True
            time.sleep(poll)
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()}".encode())
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            else:  # pragma: no cover - Windows
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)

    def __enter__(self) -> IndexLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
