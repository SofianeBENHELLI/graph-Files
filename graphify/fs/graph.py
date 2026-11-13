"""Assemble the fs graph: content graph + folders + files + tags + topics.

Node id scheme (stable, readable by small LLMs):

- ``dir:<relative/path>``   folders (``dir:`` is the indexed root)
- ``file:<relative/path>``  files
- ``tag:<normalized key>``  tags
- anything else             concept nodes from graphify's content graph

Edges added here: ``contains`` (folder → folder/file), ``mentions`` (file →
concept extracted from it), ``tagged_with`` (file/folder → tag) and
``co_occurs_with`` (tag ↔ tag, weighted by the Jaccard overlap of their files).
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import networkx as nx

from graphify.fs.metadata import (
    FileRecord,
    ScanResult,
    file_id,
    folder_id,
    nfc,
    parent_rel,
    tag_id,
)
from graphify.fs.tags import (
    SOURCE_FINDER,
    SOURCE_KEYWORD,
    SOURCE_LLM,
    SOURCE_USER,
    TagDecisions,
    TagVocabulary,
)

OFFICE_EXTS = {".docx", ".xlsx", ".pptx"}
FS_KINDS = {"file", "folder", "tag"}


def is_fs_node(data: dict) -> bool:
    return data.get("fs_kind") in FS_KINDS


def load_content_graph(path: Path) -> nx.DiGraph | None:
    if not Path(path).exists():
        return None
    from graphify.serve import _read_graph

    G = _read_graph(str(path))
    if not G.is_directed():
        G = G.to_directed()
    for key in [k for k in G.graph if str(k).startswith("_")]:
        G.graph.pop(key, None)
    return G


@dataclass
class SourceResolver:
    """Map a content node's ``source_file`` spelling back to an indexed file."""

    root: Path
    direct: dict[str, str] = field(default_factory=dict)
    by_name: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, scan: ScanResult, extra: dict[str, str] | None = None) -> "SourceResolver":
        from graphify.detect import office_sidecar_name

        r = cls(scan.root)
        for rec in scan.files:
            r.direct[rec.rel] = rec.rel
            if rec.ext in OFFICE_EXTS:
                try:
                    r.by_name[office_sidecar_name(scan.root / rec.rel, scan.root)] = rec.rel
                except OSError:
                    pass
        for k, v in (extra or {}).items():
            r.direct[nfc(k)] = v
        return r

    def resolve(self, source) -> str | None:
        if not source or not isinstance(source, str):
            return None
        sf = nfc(source.replace("\\", "/"))
        while sf.startswith("./"):
            sf = sf[2:]
        if sf in self.direct:
            return self.direct[sf]
        p = PurePosixPath(sf)
        if p.is_absolute():
            try:
                rel = p.relative_to(PurePosixPath(self.root.as_posix())).as_posix()
                if rel in self.direct:
                    return self.direct[rel]
            except ValueError:
                pass
        return self.by_name.get(p.name)


def _sources(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def assemble_base(scan: ScanResult, content: nx.DiGraph | None,
                  resolver: SourceResolver | None = None) -> nx.DiGraph:
    """Content graph + folder/file nodes + ``contains`` and ``mentions`` edges."""
    G = content.copy() if content is not None else nx.DiGraph()
    resolver = resolver or SourceResolver.build(scan)

    for folder in scan.folders:
        G.add_node(folder_id(folder.rel), **folder.attrs())
    for folder in scan.folders:
        if folder.rel:
            G.add_edge(folder_id(parent_rel(folder.rel)), folder_id(folder.rel),
                       relation="contains", confidence="EXTRACTED", confidence_score=1.0)
    for rec in scan.files:
        fid = file_id(rec.rel)
        G.add_node(fid, **rec.attrs())
        G.add_edge(folder_id(parent_rel(rec.rel)), fid, relation="contains",
                   confidence="EXTRACTED", confidence_score=1.0)

    if content is not None:
        mentions: dict[str, set[str]] = defaultdict(set)
        for nid, data in content.nodes(data=True):
            for sf in _sources(data.get("source_file")):
                rel = resolver.resolve(sf)
                if rel:
                    mentions[rel].add(nid)
        for u, v, data in content.edges(data=True):
            for sf in _sources(data.get("source_file")):
                rel = resolver.resolve(sf)
                if rel:
                    mentions[rel].update((u, v))
        for rel, nodes in mentions.items():
            fid = file_id(rel)
            if fid not in G:
                continue
            G.nodes[fid]["concepts"] = len(nodes)
            for nid in nodes:
                if nid in G and not is_fs_node(G.nodes[nid]):
                    G.add_edge(fid, nid, relation="mentions", confidence="EXTRACTED",
                               confidence_score=1.0)
    return G


def file_concepts(G: nx.DiGraph, fid: str, limit: int | None = None) -> list[str]:
    """Concept nodes a file mentions, most connected first."""
    nodes = [v for v in G.successors(fid) if G.edges[fid, v].get("relation") == "mentions"]
    nodes.sort(key=lambda n: (-G.degree(n), str(n)))
    return nodes[:limit] if limit else nodes


@dataclass
class TagAssignment:
    name: str
    source: str
    status: str  # "accepted" | "suggested"
    score: float = 1.0
    color: int = 0


def gather_tags(scan: ScanResult, decisions: TagDecisions,
                suggestions: dict[str, list] | None) -> dict[str, list[TagAssignment]]:
    """Every tag claim per path (files and folders), before vocabulary merging."""
    out: dict[str, list[TagAssignment]] = defaultdict(list)
    for folder in scan.folders:
        for t in folder.finder_tags:
            out["dir:" + folder.rel].append(TagAssignment(t.name, SOURCE_FINDER, "accepted", 1.0, t.color))
    for rec in scan.files:
        key = "file:" + rec.rel
        for t in rec.finder_tags:
            out[key].append(TagAssignment(t.name, SOURCE_FINDER, "accepted", 1.0, t.color))
        for kw in rec.keywords:
            out[key].append(TagAssignment(kw, SOURCE_KEYWORD, "accepted", 1.0))
        for item in (suggestions or {}).get(rec.rel, []):
            name, score = (item[0], float(item[1])) if isinstance(item, (list, tuple)) else (str(item), 0.6)
            if decisions.is_rejected(rec.rel, name):
                continue
            status = "accepted" if decisions.is_accepted(rec.rel, name) else "suggested"
            out[key].append(TagAssignment(name, SOURCE_LLM, status, score))
        for name in decisions.added.get(rec.rel, []):
            out[key].append(TagAssignment(name, SOURCE_USER, "accepted", 1.0))
        out[key] = [a for a in out[key] if not decisions.is_removed(rec.rel, a.name)]
    return out


def apply_tags(G: nx.DiGraph, scan: ScanResult, decisions: TagDecisions,
               suggestions: dict[str, list] | None = None) -> TagVocabulary:
    claims = gather_tags(scan, decisions, suggestions)
    vocab = TagVocabulary(decisions.renamed)
    # Register human-made spellings first so they win the display name.
    for source in (SOURCE_USER, SOURCE_FINDER, SOURCE_KEYWORD, SOURCE_LLM):
        for items in claims.values():
            for a in items:
                if a.source == source:
                    vocab.add(a.name)

    tag_files: dict[str, set[str]] = defaultdict(set)
    tag_sources: dict[str, set[str]] = defaultdict(set)
    tag_colors: dict[str, Counter] = defaultdict(Counter)
    edges: dict[tuple[str, str], dict] = {}
    for owner, items in claims.items():
        owner_id = owner if owner.startswith("dir:") else owner
        if owner_id not in G:
            continue
        for a in items:
            key = vocab.key_for(a.name)
            if not key:
                continue
            tid = tag_id(key)
            e = edges.setdefault((owner_id, tid), {"sources": set(), "status": "suggested", "score": 0.0})
            e["sources"].add(a.source)
            if a.status == "accepted":
                e["status"] = "accepted"
            e["score"] = max(e["score"], a.score)
            if a.color:
                tag_colors[key][a.color] += 1
            tag_sources[key].add(a.source)
            tag_files[key].add(owner_id)

    from graphify.fs.finder import FINDER_COLORS

    for key, owners in tag_files.items():
        tid = tag_id(key)
        color = tag_colors[key].most_common(1)[0][0] if tag_colors[key] else 0
        G.add_node(tid, label=vocab.display(key), fs_kind="tag", kind="tag", file_type="concept",
                   source_file="", tag_key=key, tag_sources=sorted(tag_sources[key]),
                   color=FINDER_COLORS.get(color), color_index=color,
                   file_count=sum(1 for o in owners if o.startswith("file:")))
    for (owner_id, tid), e in edges.items():
        human = e["sources"] - {SOURCE_LLM}
        G.add_edge(owner_id, tid, relation="tagged_with",
                   confidence="EXTRACTED" if human else "INFERRED",
                   confidence_score=1.0 if human else round(e["score"], 2),
                   sources=sorted(e["sources"]), status="accepted" if human else e["status"])
    # Per-file tag lists for fast display.
    for owner_id in {o for (o, _t) in edges}:
        tags = []
        for _o, tid in G.out_edges(owner_id):
            ed = G.edges[owner_id, tid]
            if ed.get("relation") == "tagged_with":
                tags.append(G.nodes[tid]["label"])
        G.nodes[owner_id]["tags"] = sorted(tags, key=str.casefold)
    return vocab


def tag_relations(G: nx.DiGraph, *, min_shared: int = 2, min_jaccard: float = 0.08, per_tag: int = 12) -> int:
    """Add ``co_occurs_with`` edges between tags that share files."""
    files_of: dict[str, set[str]] = defaultdict(set)
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "tagged_with" and str(u).startswith("file:"):
            files_of[v].add(u)
    tags_of: dict[str, set[str]] = defaultdict(set)
    for t, fs in files_of.items():
        for f in fs:
            tags_of[f].add(t)
    shared: Counter = Counter()
    for tags in tags_of.values():
        ordered = sorted(tags)
        if len(ordered) > 40:
            ordered = ordered[:40]
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                shared[(a, b)] += 1
    best: dict[str, list[tuple[float, str, str, int]]] = defaultdict(list)
    for (a, b), n in shared.items():
        if n < min_shared:
            continue
        jac = n / len(files_of[a] | files_of[b])
        if jac < min_jaccard:
            continue
        best[a].append((jac, a, b, n))
        best[b].append((jac, a, b, n))
    keep: set[tuple[str, str]] = set()
    for _t, items in best.items():
        for jac, a, b, n in sorted(items, reverse=True)[:per_tag]:
            keep.add((a, b))
    added = 0
    for a, b in keep:
        n = shared[(a, b)]
        jac = n / len(files_of[a] | files_of[b])
        G.add_edge(a, b, relation="co_occurs_with", confidence="INFERRED",
                   confidence_score=round(jac, 3), weight=round(jac, 3), shared_files=n)
        added += 1
    return added


def assign_topics(G: nx.DiGraph) -> tuple[dict[int, list[str]], dict[int, str]]:
    """Communities ("topics") for the whole fs graph, plus their names.

    Concept nodes keep graphify's communities. A file's topic is the majority
    community of the concepts it mentions; a tag's is the majority over its
    files. Without a content graph (metadata-only index) topics are found by
    clustering the file–tag graph instead.
    """
    communities: dict[int, list[str]] = defaultdict(list)
    names: dict[int, str] = {}
    for nid, data in G.nodes(data=True):
        if is_fs_node(data):
            continue
        cid = data.get("community")
        if isinstance(cid, int):
            communities[cid].append(nid)
            if data.get("community_name") and cid not in names:
                names[cid] = data["community_name"]

    if not communities:
        return _cluster_tags(G)

    # Concepts without a community (e.g. from the media pass) follow their neighbours.
    for nid, data in G.nodes(data=True):
        if is_fs_node(data) or isinstance(data.get("community"), int):
            continue
        votes = Counter(
            G.nodes[n].get("community") for n in nx.all_neighbors(G, nid)
            if not is_fs_node(G.nodes[n]) and isinstance(G.nodes[n].get("community"), int)
        )
        if votes:
            cid = votes.most_common(1)[0][0]
            G.nodes[nid]["community"] = cid
            communities[cid].append(nid)

    tag_votes: dict[str, Counter] = defaultdict(Counter)
    for nid, data in G.nodes(data=True):
        if data.get("fs_kind") != "file":
            continue
        votes = Counter()
        for v in G.successors(nid):
            if G.edges[nid, v].get("relation") == "mentions":
                cid = G.nodes[v].get("community")
                if isinstance(cid, int):
                    votes[cid] += 1
        if not votes:
            continue
        top = [cid for cid, _n in votes.most_common(3)]
        data["topics"] = top
        communities[top[0]].append(nid)
        for _f, t in G.out_edges(nid):
            if G.edges[nid, t].get("relation") == "tagged_with":
                tag_votes[t][top[0]] += 1
    for tid, votes in tag_votes.items():
        top = [cid for cid, _n in votes.most_common(3)]
        G.nodes[tid]["topics"] = top
        communities[top[0]].append(tid)

    if len(names) < len(communities):
        from graphify.cluster import label_communities_by_hub

        content_only = G.subgraph([n for n, d in G.nodes(data=True) if not is_fs_node(d)])
        hub = label_communities_by_hub(content_only, {c: m for c, m in communities.items() if c not in names})
        names.update(hub)
    return dict(communities), names


def _cluster_tags(G: nx.DiGraph) -> tuple[dict[int, list[str]], dict[int, str]]:
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d.get("relation") in ("tagged_with", "co_occurs_with") and not str(u).startswith("dir:"):
            H.add_edge(u, v, weight=float(d.get("weight", 1.0)))
    if H.number_of_edges() == 0:
        return {}, {}
    from graphify.cluster import cluster

    raw = cluster(H)
    communities: dict[int, list[str]] = {}
    names: dict[int, str] = {}
    for cid, members in raw.items():
        tags = [m for m in members if str(m).startswith("tag:")]
        if not tags:
            continue
        communities[cid] = list(members)
        hub = max(tags, key=lambda t: (H.degree(t), str(t)))
        names[cid] = G.nodes[hub].get("label", hub)
        for m in members:
            if m in G and G.nodes[m].get("fs_kind") in ("file", "tag"):
                G.nodes[m]["topics"] = [cid]
    return communities, names


def write_fs_graph(G: nx.DiGraph, communities: dict[int, list[str]], names: dict[int, str],
                   path: Path, *, root: Path) -> None:
    from graphify.export import to_json

    path.parent.mkdir(parents=True, exist_ok=True)
    for key in [k for k in G.graph if str(k).startswith("_")]:
        G.graph.pop(key, None)
    G.graph["fs_root"] = str(root)
    G.graph["fs_built"] = time.time()
    ok = to_json(G, communities, str(path), force=True, community_labels=names)
    if not ok:
        raise RuntimeError(f"could not write {path}")


def file_summary_record(rec: FileRecord) -> dict:
    return {"path": rec.rel, "kind": rec.kind, "size": rec.size}
