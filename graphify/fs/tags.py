"""Tag normalization, the tag vocabulary, and the user's tag decisions.

Tags come from three sources: Finder tags, document keywords (Office core
properties, PDF /Keywords) and topic tags suggested by a local LLM. They are
merged into one vocabulary keyed by a normalized form, so ``Machine-Learning``,
``machine learning`` and ``Machine Learnings`` become one tag node.

User decisions (accept / reject / add / remove / rename) live in
``fs/tags.json`` and always win over a rerun's suggestions.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from graphify.fs.config import read_json, write_json_atomic

SOURCE_FINDER = "finder"
SOURCE_KEYWORD = "keyword"
SOURCE_LLM = "llm"
SOURCE_USER = "user"

_SPLIT_KEYWORDS = re.compile(r"[;,\n|]+")
_SEPARATORS = re.compile(r"[\s_\-/.:+]+")
_EDGE_PUNCT = re.compile(r"^[^\w#]+|[^\w+#]+$")
MAX_TAG_LEN = 48


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tag_key(name: str) -> str:
    """Normalized identity of a tag: accent/case/separator-insensitive."""
    key = _SEPARATORS.sub(" ", fold(name).casefold()).strip()
    key = _EDGE_PUNCT.sub("", key).strip()
    return key


def _singular(key: str) -> str:
    words = key.split()
    if words and len(words[-1]) > 4 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def clean_tag(name: str) -> str | None:
    name = " ".join(str(name).split()).strip().strip("#").strip()
    if not name or len(name) > MAX_TAG_LEN:
        return None
    if not any(ch.isalnum() for ch in name):
        return None
    return name


def split_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for part in _SPLIT_KEYWORDS.split(str(raw)):
        tag = clean_tag(part)
        if tag and tag_key(tag) not in {tag_key(t) for t in out}:
            out.append(tag)
    return out[:20]


class TagVocabulary:
    """Canonical tags: maps every spelling to one key and one display name."""

    def __init__(self, renamed: dict[str, str] | None = None):
        self._display: dict[str, Counter] = {}
        self._alias: dict[str, str] = {}
        self._renamed = {tag_key(k): v for k, v in (renamed or {}).items()}

    def _match(self, key: str) -> str | None:
        if key in self._display:
            return key
        if key in self._alias:
            return self._alias[key]
        sing = _singular(key)
        for cand in (sing,):
            if cand in self._display:
                return cand
            if cand in self._alias:
                return self._alias[cand]
        if len(key) >= 6 and self._display:
            try:
                from rapidfuzz import fuzz, process

                hit = process.extractOne(key, list(self._display), scorer=fuzz.ratio, score_cutoff=92)
            except ImportError:
                hit = None
            if hit:
                return hit[0]
        return None

    def add(self, name: str) -> str | None:
        """Register a spelling and return its canonical key."""
        tag = clean_tag(name)
        if not tag:
            return None
        key = tag_key(tag)
        if not key:
            return None
        canon = self._match(key)
        if canon is None:
            canon = key
            self._display[canon] = Counter()
        elif canon != key:
            self._alias[key] = canon
        self._display[canon][tag] += 1
        return canon

    def key_for(self, name: str) -> str | None:
        tag = clean_tag(name)
        if not tag:
            return None
        return self._match(tag_key(tag))

    def display(self, key: str) -> str:
        if key in self._renamed:
            return self._renamed[key]
        counts = self._display.get(key)
        if not counts:
            return key
        # Most frequent spelling; prefer one with capitals on ties (proper nouns).
        return max(counts.items(), key=lambda kv: (kv[1], kv[0] != kv[0].lower(), kv[0]))[0]

    def keys(self) -> list[str]:
        return list(self._display)

    def top(self, n: int = 200) -> list[str]:
        ranked = sorted(self._display, key=lambda k: -sum(self._display[k].values()))
        return [self.display(k) for k in ranked[:n]]


@dataclass
class TagDecisions:
    """The user's explicit tag decisions, keyed by file path relative to the root."""

    accepted: dict[str, list[str]] = field(default_factory=dict)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    added: dict[str, list[str]] = field(default_factory=dict)
    removed: dict[str, list[str]] = field(default_factory=dict)
    renamed: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "TagDecisions":
        data = read_json(path, {})
        return cls(
            accepted=data.get("accepted", {}),
            rejected=data.get("rejected", {}),
            added=data.get("added", {}),
            removed=data.get("removed", {}),
            renamed=data.get("renamed", {}),
        )

    def save(self, path: Path) -> None:
        write_json_atomic(path, {
            "version": 1,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "added": self.added,
            "removed": self.removed,
            "renamed": self.renamed,
        })

    @staticmethod
    def _keys(values: list[str]) -> set[str]:
        return {tag_key(v) for v in values}

    def _put(self, bucket: dict[str, list[str]], rel: str, name: str) -> None:
        items = bucket.setdefault(rel, [])
        if tag_key(name) not in self._keys(items):
            items.append(name)

    def _drop(self, bucket: dict[str, list[str]], rel: str, name: str) -> None:
        if rel in bucket:
            bucket[rel] = [v for v in bucket[rel] if tag_key(v) != tag_key(name)]
            if not bucket[rel]:
                del bucket[rel]

    def accept(self, rel: str, name: str) -> None:
        self._put(self.accepted, rel, name)
        self._drop(self.rejected, rel, name)
        self._drop(self.removed, rel, name)

    def reject(self, rel: str, name: str) -> None:
        self._put(self.rejected, rel, name)
        self._drop(self.accepted, rel, name)

    def add(self, rel: str, name: str) -> None:
        self._put(self.added, rel, name)
        self._drop(self.removed, rel, name)
        self._drop(self.rejected, rel, name)

    def remove(self, rel: str, name: str) -> None:
        self._put(self.removed, rel, name)
        self._drop(self.added, rel, name)
        self._drop(self.accepted, rel, name)

    def is_rejected(self, rel: str, name: str) -> bool:
        return tag_key(name) in self._keys(self.rejected.get(rel, []))

    def is_removed(self, rel: str, name: str) -> bool:
        return tag_key(name) in self._keys(self.removed.get(rel, []))

    def is_accepted(self, rel: str, name: str) -> bool:
        return tag_key(name) in self._keys(self.accepted.get(rel, []))
