"""graphify fs: file-system knowledge graph, tags, tools, agent, explorer API, MCP."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import networkx as nx
import pytest

httpx = pytest.importorskip("httpx")

from graphify.fs import graph as fsgraph  # noqa: E402
from graphify.fs.config import FsIndexPaths, default_index_dir, load_state  # noqa: E402
from graphify.fs.finder import FinderTag, FinderTagWriter, encode_tag_plist, finder_tags_supported, parse_tag_plist, read_finder_tags  # noqa: E402
from graphify.fs.indexer import IndexOptions, run_index  # noqa: E402
from graphify.fs.metadata import scan  # noqa: E402
from graphify.fs.ollama import Ollama as RealOllama  # noqa: E402
from graphify.fs.tags import TagDecisions, TagVocabulary, split_keywords, tag_key  # noqa: E402
from graphify.fs.tools import TOOL_NAMES, FsIndex, ToolError, call_tool  # noqa: E402
from tests import fs_fake_ollama  # noqa: E402

macos_only = pytest.mark.skipif(not finder_tags_supported(), reason="Finder tags need macOS + xattr")


# ---------------------------------------------------------------- fixtures
def _docx(path: Path, text: str, keywords: str | None = None) -> None:
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph(text)
    if keywords:
        d.core_properties.keywords = keywords
    d.save(path)


def make_corpus(root: Path) -> Path:
    (root / "Projects/Atlas").mkdir(parents=True)
    (root / "Finance").mkdir()
    (root / "Notes").mkdir()
    (root / "Projects/Atlas/roadmap.md").write_text("# Atlas roadmap\nSemantic search and GPU budget.\n")
    _docx(root / "Finance/budget.docx", "Budget for the GPU cluster.", keywords="budget; Finance")
    (root / "Notes/meeting.txt").write_text("Atlas steering committee: GPU budget approved.\n")
    (root / "Notes/.hidden.txt").write_text("secret")
    (root / ".DS_Store").write_bytes(b"\0")
    (root / "Deck.key").mkdir()  # macOS package: listed as one file
    (root / "Deck.key/Index.zip").write_bytes(b"x" * 10)
    (root / "node_modules/x").mkdir(parents=True)
    (root / "node_modules/x/index.js").write_text("ignored")
    return root


def write_content_graph(root: Path, index: Path) -> None:
    """A small graphify content graph whose source_files use every spelling the resolver handles."""
    from graphify.detect import office_sidecar_name
    from graphify.export import to_json

    sidecar = f"{index}/graphify-out/converted/" + office_sidecar_name(root / "Finance/budget.docx", root)
    G = nx.DiGraph()
    G.add_node("atlas", label="Atlas", file_type="concept", source_file="Projects/Atlas/roadmap.md")
    G.add_node("semantic_search", label="Semantic search", file_type="concept", source_file="Projects/Atlas/roadmap.md")
    G.add_node("gpu_budget", label="GPU budget", file_type="concept", source_file=sidecar)
    G.add_node("steering", label="Steering committee", file_type="concept", source_file=str(root / "Notes/meeting.txt"))
    G.add_edge("atlas", "semantic_search", relation="uses", confidence="EXTRACTED", source_file="Projects/Atlas/roadmap.md")
    G.add_edge("atlas", "gpu_budget", relation="references", confidence="INFERRED", source_file="Notes/meeting.txt")
    G.add_edge("steering", "atlas", relation="references", confidence="INFERRED", source_file="Notes/meeting.txt")
    out = FsIndexPaths(index).content_graph
    out.parent.mkdir(parents=True, exist_ok=True)
    assert to_json(G, {0: ["atlas", "semantic_search", "steering"], 1: ["gpu_budget"]}, str(out), force=True,
                   community_labels={0: "Atlas project", 1: "Money"})


class FakeOllama:
    """graphify.fs.ollama.Ollama wired to the scripted fake model."""

    def __new__(cls, *args, **kwargs):
        return RealOllama("http://fake-ollama", transport=httpx.MockTransport(fs_fake_ollama.handler))


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_FS_HOME", str(tmp_path / "home"))
    root = make_corpus(tmp_path / "docs")
    index = default_index_dir(root)
    write_content_graph(root, index)
    run_index(root, IndexOptions(metadata_only=True), log=lambda m: None)
    return root, index, FsIndex(index)


# ---------------------------------------------------------------- tags
def test_tag_key_folds_case_accents_separators():
    assert tag_key("Machine-Learning") == tag_key("machine learning") == tag_key("  MACHINE_learning ")
    assert tag_key("Étude") == tag_key("etude")


def test_vocabulary_merges_spellings_and_plurals():
    v = TagVocabulary()
    a = v.add("Knowledge Graph")
    assert v.add("knowledge-graphs") == a
    assert v.add("knowledge graph") == a
    assert v.display(a) in ("Knowledge Graph", "knowledge graph")
    assert v.add("budget") != a


def test_vocabulary_rename_wins():
    v = TagVocabulary(renamed={"ml": "Machine learning"})
    key = v.add("ML")
    assert v.display(key) == "Machine learning"


def test_split_keywords():
    assert split_keywords("budget; Finance, GPU|budget") == ["budget", "Finance", "GPU"]
    assert split_keywords(None) == []


def test_decisions_accept_reject_roundtrip(tmp_path):
    d = TagDecisions()
    d.accept("a.pdf", "Atlas")
    d.reject("a.pdf", "misc")
    d.add("a.pdf", "Q3")
    d.remove("a.pdf", "Q3")
    d.save(tmp_path / "t.json")
    d2 = TagDecisions.load(tmp_path / "t.json")
    assert d2.is_accepted("a.pdf", "atlas")
    assert d2.is_rejected("a.pdf", "Misc")
    assert d2.is_removed("a.pdf", "q3") and "a.pdf" not in d2.added


# ---------------------------------------------------------------- finder
def test_tag_plist_roundtrip():
    tags = [FinderTag("Important", 6), FinderTag("Atlas")]
    assert parse_tag_plist(encode_tag_plist(tags)) == tags
    assert parse_tag_plist(b"garbage") == []


@macos_only
def test_finder_writer_add_remove_undo(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("x")
    w = FinderTagWriter(tmp_path, tmp_path / "log.jsonl")
    w.add(f, ["Atlas"], color=4)
    w.add(f, ["atlas", "Q3"])  # case-insensitive duplicate ignored
    assert [t.name for t in read_finder_tags(f)] == ["Atlas", "Q3"]
    w.remove(f, ["q3"])
    assert [t.name for t in read_finder_tags(f)] == ["Atlas"]
    assert w.undo(1) == [f"restored {f}"]
    assert [t.name for t in read_finder_tags(f)] == ["Atlas", "Q3"]


@macos_only
def test_finder_writer_refuses_outside_root(tmp_path):
    inside = tmp_path / "root"
    inside.mkdir()
    outside = tmp_path / "other.txt"
    outside.write_text("x")
    with pytest.raises(PermissionError):
        FinderTagWriter(inside, tmp_path / "log.jsonl").add(outside, ["x"])


# ---------------------------------------------------------------- scan
def test_scan_lists_files_folders_packages_and_skips_noise(tmp_path):
    root = make_corpus(tmp_path / "docs")
    result = scan(root)
    rels = {f.rel for f in result.files}
    assert {"Projects/Atlas/roadmap.md", "Finance/budget.docx", "Notes/meeting.txt", "Deck.key"} <= rels
    assert not any("hidden" in r or "node_modules" in r or "DS_Store" in r for r in rels)
    deck = next(f for f in result.files if f.rel == "Deck.key")
    assert deck.is_package and deck.kind == "slides"
    budget = next(f for f in result.files if f.rel == "Finance/budget.docx")
    assert budget.keywords == ["budget", "Finance"] and budget.kind == "document"
    folders = {f.rel: f for f in result.folders}
    assert folders[""].file_count == 4 and folders["Projects"].file_count == 1


def test_scan_marks_icloud_placeholders_offline(tmp_path, monkeypatch):
    from graphify.fs import metadata

    root = make_corpus(tmp_path / "docs")
    monkeypatch.setattr(metadata, "_is_offline", lambda st: st.st_size == os.stat(root / "Notes/meeting.txt").st_size)
    called = []
    monkeypatch.setattr(metadata, "document_properties", lambda p: called.append(p) or {})
    result = scan(root)
    meeting = next(f for f in result.files if f.rel == "Notes/meeting.txt")
    assert meeting.offline
    assert root / "Notes/meeting.txt" not in called  # offline files are never read


@macos_only
def test_scan_reads_finder_tags(tmp_path):
    root = make_corpus(tmp_path / "docs")
    FinderTagWriter(root, tmp_path / "log").add(root / "Notes/meeting.txt", ["Atlas"], color=4)
    result = scan(root)
    meeting = next(f for f in result.files if f.rel == "Notes/meeting.txt")
    assert meeting.finder_tags == [FinderTag("Atlas", 4)]


# ---------------------------------------------------------------- graph assembly
def test_assemble_links_files_to_concepts_through_every_source_spelling(built):
    root, index, idx = built
    G = idx.G
    def mentions(fid):
        return {v for v in G.successors(fid) if G.edges[fid, v].get("relation") == "mentions"}
    assert mentions("file:Projects/Atlas/roadmap.md") >= {"atlas", "semantic_search"}
    assert "gpu_budget" in mentions("file:Finance/budget.docx")  # via the Office sidecar name
    assert {"steering", "atlas", "gpu_budget"} <= mentions("file:Notes/meeting.txt")  # abs path + edge source
    assert G.has_edge("dir:", "dir:Projects") and G.has_edge("dir:Finance", "file:Finance/budget.docx")


def test_tags_merge_sources_and_files_get_topics(built):
    _root, _index, idx = built
    G = idx.G
    assert G.nodes["tag:budget"]["tag_sources"] == ["keyword"]
    roadmap = G.nodes["file:Projects/Atlas/roadmap.md"]
    assert idx.topic_name(roadmap["topics"][0]) == "Atlas project"


def test_tag_relations_and_topic_fallback_without_content(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    for i in range(3):
        (root / f"f{i}.txt").write_text("x")
    result = scan(root)
    G = fsgraph.assemble_base(result, None)
    suggestions = {"f0.txt": [["alpha", 0.9], ["beta", 0.8]], "f1.txt": [["alpha", 0.9], ["beta", 0.8]],
                   "f2.txt": [["gamma", 0.9]]}
    fsgraph.apply_tags(G, result, TagDecisions(), suggestions)
    assert fsgraph.tag_relations(G) == 1
    assert G.edges["tag:alpha", "tag:beta"]["relation"] == "co_occurs_with"
    communities, names = fsgraph.assign_topics(G)
    assert communities and set(names.values()) & {"alpha", "beta", "gamma"}


def test_rejected_and_removed_tags_are_dropped(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("x")
    result = scan(root)
    G = fsgraph.assemble_base(result, None)
    d = TagDecisions()
    d.reject("a.txt", "noise")
    d.accept("a.txt", "keep")
    fsgraph.apply_tags(G, result, d, {"a.txt": [["noise", 0.9], ["keep", 0.8], ["maybe", 0.7]]})
    edges = {v: G.edges["file:a.txt", v] for v in G.successors("file:a.txt") if v.startswith("tag:")}
    assert "tag:noise" not in edges
    assert edges["tag:keep"]["status"] == "accepted" and edges["tag:maybe"]["status"] == "suggested"


# ---------------------------------------------------------------- tools
def test_overview_and_facets(built):
    _root, _index, idx = built
    ov = idx.overview()
    assert ov["files"] == 4 and ov["concepts"] == 4 and ov["topics"] == 2
    facets = idx.facets()
    assert facets["total_files"] == 4 and any(t["name"] == "budget" for t in facets["tags"])


def test_search_ranks_by_name_tags_and_concepts(built):
    _root, _index, idx = built
    res = idx.search("GPU budget")
    paths = [f["path"] for f in res["files"]]
    assert paths[0] == "Finance/budget.docx"
    assert "Notes/meeting.txt" in paths  # found through the concept, not the name
    assert any(c["label"] == "GPU budget" for c in res["concepts"])


def test_find_files_filters(built):
    _root, _index, idx = built
    assert [f["path"] for f in idx.find_files(tag="finance")["files"]] == ["Finance/budget.docx"]
    assert idx.find_files(kind="text")["total"] == 2
    assert idx.find_files(folder="Projects")["total"] == 1
    assert idx.find_files(topic="Atlas project")["total"] >= 1
    assert idx.find_files(modified_after="2999-01-01")["total"] == 0
    assert idx.find_files(tag="nope")["total"] == 0
    with pytest.raises(ToolError):
        idx.find_files(modified_after="not a date")


def test_get_file_related_and_resolution(built):
    _root, _index, idx = built
    info = idx.get_file("roadmap.md")  # by bare name
    assert info["id"] == "file:Projects/Atlas/roadmap.md"
    assert {c["label"] for c in info["concepts"]} >= {"Atlas", "Semantic search"}
    related = idx.related_files("Projects/Atlas/roadmap.md")["files"]
    assert related and related[0]["path"] == "Notes/meeting.txt" and "concept" in related[0]["why"]
    with pytest.raises(ToolError):
        idx.get_file("does-not-exist.pdf")


def test_list_folder_explain_connection_and_views(built):
    _root, _index, idx = built
    top = idx.list_folder("")
    assert {f["name"] for f in top["folders"]} == {"Projects", "Finance", "Notes"}
    assert idx.explain("budget")["type"] == "tag"
    concept = idx.explain("semantic search")
    assert concept["type"] == "concept" and concept["file_count"] == 1
    path = idx.connection("roadmap.md", "budget.docx")
    assert path["found"] and path["ids"][0] == "file:Projects/Atlas/roadmap.md"
    view = idx.neighborhood("file:Projects/Atlas/roadmap.md")
    assert view["center"] == "file:Projects/Atlas/roadmap.md" and len(view["nodes"]) > 2
    assert idx.topic_map()["nodes"]
    assert idx.neighborhood("topic:0")["center"] == "topic:0"


def test_call_tool_validates(built):
    _root, _index, idx = built
    assert set(TOOL_NAMES) >= {"search", "find_files", "get_file"}
    assert len(call_tool(idx, "find_files", {"limit": "1", "bogus": 3})["files"]) == 1
    with pytest.raises(ToolError):
        call_tool(idx, "rm_rf", {})


def test_apply_tag_change_in_memory(built):
    _root, _index, idx = built
    fid = "file:Notes/meeting.txt"
    idx.apply_tag_change(fid, "Steering", "add")
    assert idx.find_files(tag="steering")["total"] == 1
    idx.apply_tag_change(fid, "Steering", "remove")
    assert idx.find_files(tag="steering")["total"] == 0


# ---------------------------------------------------------------- LLM stages (fake model)
def test_index_with_local_model_tags_and_names_topics(tmp_path, monkeypatch):
    import graphify.fs.ollama as ollama_mod

    monkeypatch.setenv("GRAPHIFY_FS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(ollama_mod, "Ollama", FakeOllama)
    root = make_corpus(tmp_path / "docs")
    index = default_index_dir(root)
    write_content_graph(root, index)
    stats = run_index(root, IndexOptions(content=False, media=False), log=lambda m: None)
    assert stats["suggested_tags"] > 0
    idx = FsIndex(index)
    roadmap = idx.get_file("roadmap.md")
    suggested = {t["name"] for t in roadmap["tags"] if t["status"] == "suggested"}
    assert {"Atlas", "Semantic search"} & suggested
    names = {t["name"] for t in idx.list_topics()["topics"]}
    assert any("&" in n for n in names)  # named by the (fake) model
    # A second run is fully cached: no new model calls needed.
    calls = []
    real = FakeOllama.__new__

    def counting(cls, *a, **k):
        client = real(cls, *a, **k)
        orig = client.chat_json
        client.chat_json = lambda *x, **y: calls.append(1) or orig(*x, **y)
        return client
    monkeypatch.setattr(FakeOllama, "__new__", counting)
    run_index(root, IndexOptions(content=False, media=False), log=lambda m: None)
    assert calls == []


def test_navigator_uses_tools_and_cites_files(built):
    from graphify.fs.agent import Navigator
    from graphify.fs.ollama import Ollama

    _root, _index, idx = built
    client = Ollama("http://fake", transport=httpx.MockTransport(fs_fake_ollama.handler))
    events = list(Navigator(idx, client, model="qwen3:8b").run("GPU budget"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "tool_call" and "tool_result" in kinds and "token" in kinds and kinds[-1] == "done"
    done = events[-1]
    assert "Finance/budget.docx" in done["answer"]
    assert done["files"][0] == "file:Finance/budget.docx"


def test_navigator_reports_unreachable_model(built):
    from graphify.fs.agent import Navigator
    from graphify.fs.ollama import Ollama

    _root, _index, idx = built

    def down(request):
        raise httpx.ConnectError("refused")
    events = list(Navigator(idx, Ollama("http://fake", transport=httpx.MockTransport(down))).run("hi"))
    assert events[-1]["type"] == "error"


# ---------------------------------------------------------------- explorer API
@pytest.fixture
def client(built, monkeypatch):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    import graphify.fs.ollama as ollama_mod
    from graphify.fs.explorer.app import create_app

    monkeypatch.setattr(ollama_mod, "Ollama", FakeOllama)
    _root, _index, idx = built
    app = create_app(idx, token="tok", port=8765, finder_write=False, extra_hosts={"testserver"})
    return TestClient(app)


def test_explorer_requires_token_and_local_host(client):
    assert client.get("/api/overview").status_code == 401
    assert client.get("/api/overview", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/?t=wrong").status_code == 403
    r = client.get("/?t=tok", follow_redirects=False)
    assert r.status_code == 303 and "gfs_token" in r.headers["set-cookie"]
    assert client.get("/api/overview").status_code == 200  # cookie now set
    assert "graphify fs" in client.get("/").text


def test_explorer_read_endpoints(client):
    client.get("/?t=tok")
    assert client.get("/api/search", params={"q": "budget"}).json()["files"]
    assert client.get("/api/files", params={"tag": "finance"}).json()["total"] == 1
    assert client.get("/api/folder", params={"path": "Notes"}).json()["total_files"] == 1
    info = client.get("/api/file", params={"id": "file:Notes/meeting.txt"}).json()
    assert info["path"] == "Notes/meeting.txt"
    assert client.get("/api/neighborhood", params={"id": "file:Notes/meeting.txt"}).json()["nodes"]
    assert client.get("/api/raw", params={"id": "file:Notes/meeting.txt"}).text.startswith("Atlas steering")
    assert client.get("/api/file", params={"id": "file:../../etc/passwd"}).status_code == 404
    assert client.get("/api/config").json()["chat_ready"] is True


def test_explorer_tag_endpoint_persists_decision(client, built):
    _root, index, _idx = built
    client.get("/?t=tok")
    r = client.post("/api/tag", json={"id": "file:Notes/meeting.txt", "name": "Steering", "action": "add"})
    assert r.status_code == 200 and any(t["name"] == "Steering" for t in r.json()["tags"])
    assert TagDecisions.load(FsIndexPaths(index).tags_state).added == {"Notes/meeting.txt": ["Steering"]}
    bad = client.post("/api/tag", json={"id": "file:Notes/meeting.txt", "name": "x", "action": "explode"})
    assert bad.status_code == 400
    cross = client.post("/api/tag", json={"id": "file:Notes/meeting.txt", "name": "x", "action": "add"},
                        headers={"Origin": "http://evil.example"})
    assert cross.status_code == 403


def test_explorer_chat_streams_events(client):
    client.get("/?t=tok")
    with client.stream("POST", "/api/chat", json={"question": "GPU budget"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[0]["type"] == "tool_call" and events[-1]["type"] == "done"


# ---------------------------------------------------------------- MCP
def test_mcp_server_exposes_tools(built):
    pytest.importorskip("mcp")
    from graphify.fs.mcp import build_server

    _root, _index, idx = built
    srv = build_server(idx)

    async def go():
        tools = await srv.list_tools()
        res = await srv.call_tool("find_files", {"tag": "finance"})
        return tools, res
    tools, res = asyncio.run(go())
    assert {t.name for t in tools} == set(TOOL_NAMES)
    assert "Finance/budget.docx" in str(res)


# ---------------------------------------------------------------- CLI
def test_cli_status_and_ask_without_model(built, capsys, monkeypatch):
    from graphify.fs.cli import main

    root, _index, _idx = built
    assert main(["status"]) == 0
    assert "4 files" in capsys.readouterr().out
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")  # nothing listens there
    assert main(["ask", "hello", str(root)]) == 1


def test_state_records_stats(built):
    _root, index, _idx = built
    state = load_state(index)
    assert state.stats["files"] == 4 and state.last_index


# ---------------------------------------------------------------- full text
def test_fulltext_search_finds_words_inside_files(built):
    _root, _index, idx = built
    res = idx.search("approved")  # only written inside Notes/meeting.txt
    assert res["files"][0]["path"] == "Notes/meeting.txt"
    assert "«approved»" in res["files"][0]["snippet"]
    docx_hit = idx.search("cluster")  # inside the .docx body only
    assert docx_hit["files"][0]["path"] == "Finance/budget.docx"


def test_fulltext_is_accent_insensitive_and_incremental(tmp_path):
    from graphify.fs.fulltext import FullTextIndex

    root = tmp_path / "docs"
    root.mkdir()
    (root / "fr.md").write_text("Étude sémantique du budget prévisionnel")
    ft = FullTextIndex(tmp_path / "ft.db")
    assert ft.update(scan(root)) == 1
    assert ft.update(scan(root)) == 0  # unchanged files are not re-read
    assert ft.search("etude semantique")[0]["path"] == "fr.md"
    (root / "fr.md").unlink()
    ft.update(scan(root))
    assert ft.search("etude") == []


def test_read_file_returns_relevant_passages(built):
    _root, _index, idx = built
    out = idx.read_file("meeting.txt", query="budget")
    assert "GPU budget approved" in out["passages"][0]
    assert call_tool(idx, "read_file", {"file": "budget.docx"})["passages"]
