"""Where a graphify fs index lives and which local models it uses.

An index is kept outside the indexed folder (default ``~/.graphify-fs/<name>-<hash>``)
so indexing never writes into your documents, and never into iCloud Drive.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_EXTRACT_MODEL = "gemma3:12b"
DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_WHISPER_MODEL = "small"
STATE_FILE = "state.json"


def fs_home() -> Path:
    """Root directory holding every fs index (``GRAPHIFY_FS_HOME`` overrides)."""
    override = os.environ.get("GRAPHIFY_FS_HOME")
    return Path(override).expanduser() if override else Path.home() / ".graphify-fs"


def _slug(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "-", folded).strip("-").lower()[:40] or "root"


def default_index_dir(root: Path) -> Path:
    root = root.expanduser().resolve()
    digest = hashlib.sha1(unicodedata.normalize("NFC", str(root)).encode()).hexdigest()[:8]
    return fs_home() / f"{_slug(root.name)}-{digest}"


@dataclass
class IndexState:
    """Persisted settings of one index (``<index>/state.json``)."""

    root: str
    extract_model: str = DEFAULT_EXTRACT_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL
    whisper_model: str = DEFAULT_WHISPER_MODEL
    vision: bool = True
    created: float = field(default_factory=time.time)
    last_index: float | None = None
    last_content_pass: float | None = None
    stats: dict = field(default_factory=dict)

    @property
    def root_path(self) -> Path:
        return Path(self.root)


@dataclass
class FsIndexPaths:
    """All paths of one index directory."""

    index: Path

    @property
    def out(self) -> Path:  # graphify's own output dir (content graph, caches)
        return self.index / "graphify-out"

    @property
    def content_graph(self) -> Path:
        return self.out / "graph.json"

    @property
    def fs_dir(self) -> Path:
        return self.out / "fs"

    @property
    def graph(self) -> Path:
        return self.fs_dir / "graph.json"

    @property
    def tags_state(self) -> Path:
        return self.fs_dir / "tags.json"

    @property
    def tag_writes(self) -> Path:
        return self.fs_dir / "tag-writes.jsonl"

    @property
    def topics_cache(self) -> Path:
        return self.fs_dir / "topics-cache.json"

    @property
    def meta_cache(self) -> Path:
        return self.fs_dir / "meta-cache.json"

    @property
    def fulltext(self) -> Path:
        return self.fs_dir / "fulltext.db"

    @property
    def media_dir(self) -> Path:
        return self.fs_dir / "media"

    @property
    def thumbs(self) -> Path:
        return self.fs_dir / "thumbs"

    @property
    def lock(self) -> Path:  # held while an index run rebuilds the files above
        return self.fs_dir / "index.lock"

    @property
    def content_lock(self) -> Path:  # held during the slow model phase
        return self.fs_dir / "content.lock"

    @property
    def runtime(self) -> Path:  # the running explorer server (port, token, pid)
        return self.fs_dir / "explorer.json"

    @property
    def server_log(self) -> Path:
        return self.fs_dir / "explorer.log"

    @property
    def state(self) -> Path:
        return self.index / STATE_FILE


def load_state(index: Path) -> IndexState | None:
    path = FsIndexPaths(index).state
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {k: v for k, v in data.items() if k in IndexState.__dataclass_fields__}
    return IndexState(**known)


def save_state(index: Path, state: IndexState) -> None:
    path = FsIndexPaths(index).state
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, asdict(state))


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def resolve_index(target: str | Path | None) -> Path:
    """Find an index from an index dir, an indexed folder, or the only/latest index.

    Accepts the index directory itself, the indexed root folder, or nothing (then
    the most recently used index under ``fs_home()`` is picked).
    """
    if target:
        p = Path(target).expanduser().resolve()
        if (p / STATE_FILE).exists():
            return p
        candidate = default_index_dir(p)
        if (candidate / STATE_FILE).exists():
            return candidate
        raise FileNotFoundError(
            f"No graphify fs index for {p}. Build one with: graphify fs index \"{p}\""
        )
    home = fs_home()
    indexes = sorted(
        (d for d in home.glob("*") if (d / STATE_FILE).exists()),
        key=lambda d: (d / STATE_FILE).stat().st_mtime,
        reverse=True,
    ) if home.exists() else []
    if not indexes:
        raise FileNotFoundError("No graphify fs index yet. Build one with: graphify fs index <folder>")
    return indexes[0]


def list_indexes() -> list[tuple[Path, IndexState]]:
    home = fs_home()
    out: list[tuple[Path, IndexState]] = []
    if not home.exists():
        return out
    for d in sorted(home.iterdir()):
        st = load_state(d) if d.is_dir() else None
        if st:
            out.append((d, st))
    return out
