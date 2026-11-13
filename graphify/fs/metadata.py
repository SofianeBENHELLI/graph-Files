"""Walk a folder and describe every file: kind, size, dates, Finder tags, keywords.

This layer is cheap (no LLM, no content parsing beyond Office/PDF document
properties) and runs on every index, so the explorer is useful within seconds
even before the slow content pass has finished. iCloud placeholders are listed
with ``offline: true`` and are never read.
"""
from __future__ import annotations

import html
import os
import re
import stat
import sys
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from graphify.fs.config import read_json, write_json_atomic
from graphify.fs.finder import FinderTag, read_finder_tags
from graphify.fs.tags import split_keywords

# Directory names that are tooling noise in a document library.
SKIP_DIR_NAMES = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv",
    "graphify-out", ".Trash", ".Trashes", "$RECYCLE.BIN", "System Volume Information",
    ".fseventsd", ".Spotlight-V100", ".DocumentRevisions-V100", ".TemporaryItems",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
}
SKIP_FILE_NAMES = {".DS_Store", "Icon\r", ".localized", "desktop.ini", "Thumbs.db", ".com.apple.timemachine.supported"}

# macOS packages: directories Finder shows as a single file. Listed as one file
# and never descended into.
PACKAGE_SUFFIXES = {
    ".app", ".key", ".pages", ".numbers", ".rtfd", ".photoslibrary", ".bundle",
    ".framework", ".band", ".logicx", ".fcpbundle", ".imovielibrary", ".xcodeproj",
    ".xcworkspace", ".playground", ".sparsebundle", ".pkg", ".mpkg", ".plugin", ".kext",
    ".musiclibrary", ".tvlibrary", ".scriv", ".sketch", ".lpdf",
}

KIND_BY_EXT: dict[str, str] = {}
for _kind, _exts in {
    "pdf": {".pdf"},
    "slides": {".pptx", ".ppt", ".key", ".odp", ".pps", ".ppsx"},
    "document": {".docx", ".doc", ".pages", ".odt", ".rtf", ".rtfd", ".tex", ".epub"},
    "spreadsheet": {".xlsx", ".xls", ".numbers", ".ods", ".csv", ".tsv"},
    "image": {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".svg", ".raw", ".cr2", ".nef", ".psd", ".ai"},
    "video": {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv"},
    "audio": {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".aiff", ".aif"},
    "text": {".md", ".mdx", ".txt", ".rst", ".org", ".qmd"},
    "web": {".html", ".htm", ".webarchive", ".url", ".webloc"},
    "email": {".eml", ".msg", ".emlx", ".mbox"},
    "archive": {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".dmg", ".iso", ".bz2", ".xz"},
    "code": {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".cs", ".sh", ".sql", ".json", ".yaml", ".yml", ".toml", ".xml", ".ipynb", ".r", ".m"},
    "app": {".app", ".pkg", ".mpkg", ".exe", ".msi"},
}.items():
    for _e in _exts:
        KIND_BY_EXT.setdefault(_e, _kind)

KIND_LABELS = {
    "folder": "Folder", "pdf": "PDF", "slides": "Slides", "document": "Document",
    "spreadsheet": "Spreadsheet", "image": "Image", "video": "Video", "audio": "Audio",
    "text": "Text", "web": "Web", "email": "Email", "archive": "Archive", "code": "Code",
    "app": "App", "other": "Other",
}

SF_DATALESS = 0x40000000
_OFFICE_CORE_FIELDS = {
    "keywords": r"cp:keywords",
    "subject": r"dc:subject",
    "title": r"dc:title",
    "author": r"dc:creator",
    "category": r"cp:category",
}
_META_MAX_BYTES = 300 * 1024 * 1024


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def kind_for(path: Path, is_dir: bool = False) -> str:
    ext = path.suffix.lower()
    if is_dir and ext not in PACKAGE_SUFFIXES:
        return "folder"
    return KIND_BY_EXT.get(ext, "other")


def file_id(rel: str) -> str:
    return f"file:{rel}"


def folder_id(rel: str) -> str:
    return f"dir:{rel}"


def tag_id(key: str) -> str:
    return f"tag:{key}"


def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="minutes")
    except (OverflowError, OSError, ValueError):
        return None


@dataclass
class FileRecord:
    rel: str
    name: str
    kind: str
    ext: str
    size: int
    mtime: float
    created: float | None
    offline: bool
    finder_tags: list[FinderTag] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    title: str | None = None
    author: str | None = None
    pages: int | None = None
    is_package: bool = False

    def attrs(self) -> dict:
        return {
            "label": self.name,
            "fs_kind": "file",
            "kind": self.kind,
            "ext": self.ext,
            "path": self.rel,
            "source_file": self.rel,
            "file_type": _graphify_file_type(self.kind),
            "size": self.size,
            "mtime": self.mtime,
            "modified": _iso(self.mtime),
            "created": _iso(self.created),
            "offline": self.offline,
            "finder_tags": [t.name for t in self.finder_tags],
            "keywords": self.keywords,
            "title": self.title,
            "author": self.author,
            "pages": self.pages,
            "package": self.is_package,
        }


@dataclass
class FolderRecord:
    rel: str
    name: str
    mtime: float
    finder_tags: list[FinderTag] = field(default_factory=list)
    file_count: int = 0
    total_size: int = 0

    def attrs(self) -> dict:
        return {
            "label": self.name,
            "fs_kind": "folder",
            "kind": "folder",
            "path": self.rel,
            "source_file": self.rel,
            "file_type": "concept",
            "mtime": self.mtime,
            "modified": _iso(self.mtime),
            "finder_tags": [t.name for t in self.finder_tags],
            "file_count": self.file_count,
            "size": self.total_size,
        }


def _graphify_file_type(kind: str) -> str:
    return {"pdf": "paper", "image": "image", "code": "code"}.get(kind, "document")


@dataclass
class ScanResult:
    root: Path
    files: list[FileRecord]
    folders: list[FolderRecord]
    errors: list[str] = field(default_factory=list)


def _office_properties(path: Path) -> dict:
    """Read docProps/core.xml of an Office file without a full parse (regex, size-capped)."""
    try:
        with zipfile.ZipFile(path) as zf:
            try:
                info = zf.getinfo("docProps/core.xml")
            except KeyError:
                return {}
            if info.file_size > 2 * 1024 * 1024:
                return {}
            xml = zf.read(info).decode("utf-8", "replace")
    except (zipfile.BadZipFile, OSError, ValueError, RuntimeError):
        return {}
    out: dict = {}
    for key, tag in _OFFICE_CORE_FIELDS.items():
        m = re.search(rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", xml, re.S)
        if m:
            value = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
            if value:
                out[key] = value
    return out


def _pdf_properties(path: Path) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}
    try:
        reader = PdfReader(str(path), strict=False)
        meta = reader.metadata or {}
        out: dict = {}
        for key, field_name in (("keywords", "/Keywords"), ("subject", "/Subject"), ("title", "/Title"), ("author", "/Author")):
            value = meta.get(field_name)
            if value:
                value = str(value).strip()
                if value and value.lower() not in ("untitled", "none"):
                    out[key] = value
        try:
            out["pages"] = len(reader.pages)
        except Exception:
            pass
        return out
    except Exception:
        return {}


def document_properties(path: Path) -> dict:
    """Keywords/subject/title/author (and PDF page count) from document properties."""
    ext = path.suffix.lower()
    try:
        if path.stat().st_size > _META_MAX_BYTES:
            return {}
    except OSError:
        return {}
    if ext in (".docx", ".xlsx", ".pptx", ".docm", ".xlsm", ".pptm"):
        return _office_properties(path)
    if ext == ".pdf":
        return _pdf_properties(path)
    return {}


def _is_offline(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_flags", 0) & SF_DATALESS)


def _skip_entry(name: str) -> bool:
    return name in SKIP_FILE_NAMES or name.startswith(".") or name.startswith("~$")


def scan(root: Path, *, cache_path: Path | None = None, read_properties: bool = True,
         progress=None) -> ScanResult:
    """Walk ``root`` and return every folder and file with its metadata.

    ``cache_path`` keeps parsed document properties keyed by size+mtime so a
    rescan of thousands of files only re-reads what changed.
    """
    root = Path(root).expanduser().resolve()
    cache: dict = read_json(cache_path, {}) if cache_path else {}
    new_cache: dict = {}
    files: list[FileRecord] = []
    folders: dict[str, FolderRecord] = {}
    errors: list[str] = []

    try:
        root_st = os.stat(root)
    except OSError as exc:
        raise FileNotFoundError(f"cannot read {root}: {exc}") from exc
    folders[""] = FolderRecord("", nfc(root.name) or str(root), root_st.st_mtime, read_finder_tags(root))

    count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)
        rel_dir = nfc(dp.relative_to(root).as_posix()) if dp != root else ""
        keep_dirs = []
        package_dirs = []
        for d in sorted(dirnames):
            if d in SKIP_DIR_NAMES or _skip_entry(d):
                continue
            if os.path.islink(dp / d):
                continue
            if Path(d).suffix.lower() in PACKAGE_SUFFIXES:
                package_dirs.append(d)
            else:
                keep_dirs.append(d)
        dirnames[:] = keep_dirs
        for d in keep_dirs:
            full = dp / d
            rel = nfc(full.relative_to(root).as_posix())
            try:
                st = os.stat(full)
            except OSError as exc:
                errors.append(f"{rel}: {exc}")
                continue
            folders[rel] = FolderRecord(rel, nfc(d), st.st_mtime, read_finder_tags(full))

        entries = [(f, False) for f in sorted(filenames) if not _skip_entry(f)] + [(d, True) for d in package_dirs]
        for name, is_package in entries:
            full = dp / name
            rel = nfc(full.relative_to(root).as_posix())
            try:
                lst = os.lstat(full)
                if stat.S_ISLNK(lst.st_mode):
                    continue
                if not is_package and not stat.S_ISREG(lst.st_mode):
                    continue
                st = lst
            except OSError as exc:
                errors.append(f"{rel}: {exc}")
                continue
            size = st.st_size
            if is_package:
                size = _dir_size(full)
            offline = _is_offline(st)
            rec = FileRecord(
                rel=rel,
                name=nfc(name),
                kind=kind_for(full, is_dir=False),
                ext=full.suffix.lower(),
                size=size,
                mtime=st.st_mtime,
                created=getattr(st, "st_birthtime", None),
                offline=offline,
                finder_tags=read_finder_tags(full),
                is_package=is_package,
            )
            if read_properties and not offline and not is_package:
                sig = [size, st.st_mtime]
                cached = cache.get(rel)
                if cached and cached.get("sig") == sig:
                    props = cached.get("props", {})
                else:
                    props = document_properties(full)
                new_cache[rel] = {"sig": sig, "props": props}
                kws = split_keywords(props.get("keywords"))
                for extra in ("subject", "category"):
                    for kw in split_keywords(props.get(extra)):
                        if len(kw) <= 32 and kw not in kws:
                            kws.append(kw)
                rec.keywords = kws[:20]
                rec.title = props.get("title")
                rec.author = props.get("author")
                rec.pages = props.get("pages")
            files.append(rec)
            parent = folders.get(rel_dir)
            if parent is not None:
                parent.file_count += 1
            count += 1
            if progress and count % 500 == 0:
                progress(count)

    # Roll sizes and counts up the tree.
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for rec in files:
        parts = rec.rel.split("/")[:-1]
        for i in range(len(parts) + 1):
            key = "/".join(parts[:i])
            totals[key] = totals.get(key, 0) + rec.size
            counts[key] = counts.get(key, 0) + 1
    for key, folder in folders.items():
        folder.total_size = totals.get(key, 0)
        folder.file_count = counts.get(key, 0)

    if cache_path:
        write_json_atomic(cache_path, new_cache)
    return ScanResult(root=root, files=files, folders=list(folders.values()), errors=errors)


def _dir_size(path: Path, limit: int = 20000) -> int:
    total = 0
    seen = 0
    for dirpath, _dirs, fnames in os.walk(path):
        for f in fnames:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
            seen += 1
            if seen >= limit:
                return total
    return total


def parent_rel(rel: str) -> str:
    return rel.rsplit("/", 1)[0] if "/" in rel else ""


def is_macos() -> bool:
    return sys.platform == "darwin"
