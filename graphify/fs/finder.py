"""Read and write macOS Finder tags (the ``_kMDItemUserTags`` extended attribute).

Finder stores a file's tags as a binary plist array of strings, each either
``"Name"`` or ``"Name\\n<colour index>"``. Reading never touches file content, so
it is safe on iCloud placeholders. Writing only happens on an explicit user
action, is limited to paths inside the indexed root, and every change is logged
(before/after) so it can be undone.
"""
from __future__ import annotations

import json
import plistlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

TAG_XATTR = "com.apple.metadata:_kMDItemUserTags"

# Finder's colour indexes for tags.
FINDER_COLORS = {
    0: None,
    1: "gray",
    2: "green",
    3: "purple",
    4: "blue",
    5: "yellow",
    6: "red",
    7: "orange",
}


@dataclass(frozen=True)
class FinderTag:
    name: str
    color: int = 0

    @property
    def color_name(self) -> str | None:
        return FINDER_COLORS.get(self.color)

    def encode(self) -> str:
        return f"{self.name}\n{self.color}" if self.color else self.name


def _xattr():
    if sys.platform != "darwin":
        return None
    try:
        import xattr  # type: ignore
    except ImportError:
        return None
    return xattr


def finder_tags_supported() -> bool:
    return _xattr() is not None


def parse_tag_plist(raw: bytes) -> list[FinderTag]:
    try:
        items = plistlib.loads(raw)
    except Exception:
        return []
    tags: list[FinderTag] = []
    if not isinstance(items, list):
        return tags
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        name, _, color = item.partition("\n")
        try:
            idx = int(color) if color else 0
        except ValueError:
            idx = 0
        tags.append(FinderTag(name.strip(), idx if idx in FINDER_COLORS else 0))
    return tags


def encode_tag_plist(tags: list[FinderTag]) -> bytes:
    return plistlib.dumps([t.encode() for t in tags], fmt=plistlib.FMT_BINARY)


def read_finder_tags(path: Path) -> list[FinderTag]:
    """Tags Finder shows for ``path`` ([] when none, or off macOS)."""
    xattr = _xattr()
    if xattr is None:
        return []
    try:
        raw = xattr.getxattr(str(path), TAG_XATTR)
    except OSError:
        return []
    return parse_tag_plist(raw)


def _write_raw(path: Path, tags: list[FinderTag]) -> None:
    xattr = _xattr()
    if xattr is None:
        raise RuntimeError("Finder tags can only be written on macOS (pip install xattr)")
    if tags:
        xattr.setxattr(str(path), TAG_XATTR, encode_tag_plist(tags))
    else:
        try:
            xattr.removexattr(str(path), TAG_XATTR)
        except OSError:
            pass


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


class FinderTagWriter:
    """Adds/removes Finder tags under ``root`` and logs every change to ``log_path``."""

    def __init__(self, root: Path, log_path: Path):
        self.root = Path(root)
        self.log_path = Path(log_path)

    def _check(self, path: Path) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        if not _inside(path, self.root):
            raise PermissionError(f"refusing to tag a file outside the indexed folder: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _log(self, path: Path, before: list[FinderTag], after: list[FinderTag]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "path": str(path),
            "before": [t.encode() for t in before],
            "after": [t.encode() for t in after],
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def set_tags(self, path: Path, tags: list[FinderTag]) -> list[FinderTag]:
        path = self._check(path)
        before = read_finder_tags(path)
        if [t.encode() for t in before] == [t.encode() for t in tags]:
            return before
        _write_raw(path, tags)
        self._log(path, before, tags)
        return tags

    def add(self, path: Path, names: list[str], color: int = 0) -> list[FinderTag]:
        path = self._check(path)
        current = read_finder_tags(path)
        have = {t.name.casefold() for t in current}
        new = list(current) + [FinderTag(n.strip(), color) for n in names if n.strip() and n.strip().casefold() not in have]
        return self.set_tags(path, new)

    def remove(self, path: Path, names: list[str]) -> list[FinderTag]:
        path = self._check(path)
        drop = {n.strip().casefold() for n in names}
        current = read_finder_tags(path)
        return self.set_tags(path, [t for t in current if t.name.casefold() not in drop])

    def history(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def undo(self, count: int = 1) -> list[str]:
        """Revert the last ``count`` logged writes (newest first).

        A file whose tags changed again since (outside graphify) is skipped, so
        undo never clobbers a newer edit.
        """
        entries = self.history()
        undone: list[str] = []
        kept = list(entries)
        for entry in reversed(entries[-count:] if count > 0 else []):
            path = Path(entry["path"])
            current = [t.encode() for t in read_finder_tags(path)] if path.exists() else None
            if current != entry["after"]:
                undone.append(f"skipped {path} (changed since)")
                continue
            before = parse_tag_plist(plistlib.dumps(entry["before"]))
            _write_raw(path, before)
            kept.remove(entry)
            undone.append(f"restored {path}")
        self.log_path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept), encoding="utf-8")
        return undone
