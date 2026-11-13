"""``graphify fs mcp``: expose an fs index to any MCP client (Claude Code, Claude Desktop, ...).

Stdio server with the same tools the explorer's chat agent uses (descriptions
come from ``graphify.fs.tools.TOOL_SPECS``). Works with the mcp SDK 1.x
(FastMCP) and 2.x (MCPServer). The graph is reloaded automatically after a
re-index.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphify.fs.tools import TOOL_SPECS, FsIndex, ToolError, call_tool

INSTRUCTIONS = (
    "Knowledge graph of a personal folder of files (documents, slides, PDFs, images, videos). "
    "Nodes are folders, files, tags (Finder tags, document keywords, AI topic tags) and concepts "
    "extracted from file contents; files are grouped into topics. Start with `overview` or `search`, "
    "then use `get_file`, `related_files`, `find_files` and `explain` to navigate. Cite files by path."
)

_DESCRIPTIONS = {spec["name"]: spec["description"] for spec in TOOL_SPECS}


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2

        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x

        return FastMCP


def tool_functions(index: FsIndex) -> dict:
    """Plain functions (typed signatures → JSON schemas) for every fs tool."""

    def run(tool: str, /, **args) -> str:
        index.maybe_reload()
        try:
            result = call_tool(index, tool, args)
        except ToolError as exc:
            result = {"error": str(exc)}
        return json.dumps(result, ensure_ascii=False, default=str)

    def overview() -> str:
        return run("overview")

    def search(query: str, kind: str | None = None, limit: int | None = None) -> str:
        return run("search", query=query, kind=kind, limit=limit)

    def find_files(tag: str | None = None, kind: str | None = None, folder: str | None = None,
                   topic: str | None = None, name: str | None = None,
                   modified_after: str | None = None, modified_before: str | None = None,
                   limit: int | None = None) -> str:
        return run("find_files", tag=tag, kind=kind, folder=folder, topic=topic, name=name,
                   modified_after=modified_after, modified_before=modified_before, limit=limit)

    def get_file(file: str) -> str:
        return run("get_file", file=file)

    def read_file(file: str, query: str | None = None) -> str:
        return run("read_file", file=file, query=query)

    def related_files(file: str, limit: int | None = None) -> str:
        return run("related_files", file=file, limit=limit)

    def list_tags(query: str | None = None, limit: int | None = None) -> str:
        return run("list_tags", query=query, limit=limit)

    def list_topics(limit: int | None = None) -> str:
        return run("list_topics", limit=limit)

    def list_folder(path: str | None = None) -> str:
        return run("list_folder", path=path)

    def explain(name: str) -> str:
        return run("explain", name=name)

    def connection(a: str, b: str) -> str:
        return run("connection", a=a, b=b)

    fns = {f.__name__: f for f in (overview, search, find_files, get_file, read_file, related_files, list_tags,
                                     list_topics, list_folder, explain, connection)}
    assert set(fns) == set(_DESCRIPTIONS), "MCP tools out of sync with TOOL_SPECS"
    for name, fn in fns.items():
        fn.__doc__ = _DESCRIPTIONS[name]
    return fns


def build_server(index: FsIndex):
    server = _server_class()(name="graphify-fs", instructions=INSTRUCTIONS)
    for name, fn in tool_functions(index).items():
        server.add_tool(fn, name=name, description=_DESCRIPTIONS[name])
    return server


def run_stdio(index_dir: Path) -> None:
    build_server(FsIndex(index_dir)).run("stdio")
