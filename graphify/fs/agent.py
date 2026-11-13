"""The explorer's chat agent: a small local model that navigates the graph with tools.

The model gets the shared tools (``graphify.fs.tools``) through Ollama's native
tool calling, runs at most ``max_steps`` tool rounds per question, then answers.
Every step is emitted as an event so the UI can show which files the agent
looked at and highlight them:

- ``{"type": "tool_call", "name", "args"}``
- ``{"type": "tool_result", "name", "files": [ids], "nodes": [ids], "summary"}``
- ``{"type": "token", "text"}``       streamed answer text
- ``{"type": "done", "answer", "files": [ids]}``
- ``{"type": "error", "message"}``
"""
from __future__ import annotations

import json
import re
from typing import Iterator

from graphify.fs.ollama import Ollama, OllamaError
from graphify.fs.tools import TOOL_SPECS, FsIndex, ToolError, call_tool

SYSTEM_PROMPT = """You are the navigator of the user's personal knowledge graph of files: documents, slides, PDFs, spreadsheets, images, audio and video in the folder "{root}".
The graph links folders, files, tags (Finder tags, document keywords, AI topic tags), topics, and concepts extracted from file contents.

How to work:
- ALWAYS use tools to look things up before answering; never invent files, paths or contents.
- Start with `search` for free-text questions, `find_files` for filters (tag, kind, folder, date), `get_file` for one file's details, `read_file` to read what a file says (pass a query to get the relevant passages), `related_files` to explore around a file, `explain` for a concept/tag/topic, `overview` when the user asks what is in the collection.
- Use at most a few tool calls, then answer.
- Answer in the user's language, briefly. Mention files by their exact path in backticks, like `Projects/plan.pptx`, so the user can click them.
- If nothing relevant is found, say so and suggest a better search.
Today is {today}. The collection has {files} files, {tags} tags and {topics} topics."""

_PATH_RE = re.compile(r"`([^`]+)`")


def ollama_tools() -> list[dict]:
    return [{"type": "function", "function": spec} for spec in TOOL_SPECS]


def _ids_in(result) -> tuple[list[str], list[str]]:
    files: list[str] = []
    nodes: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            ident = obj.get("id")
            if isinstance(ident, str):
                (files if ident.startswith("file:") else nodes).append(ident)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(result)
    dedup = lambda xs: list(dict.fromkeys(xs))  # noqa: E731
    return dedup(files)[:60], dedup(nodes)[:60]


def _summary(name: str, result: dict) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    if "files" in result and isinstance(result["files"], list):
        n = result.get("total", len(result["files"]))
        return f"{n} file(s)"
    if "tags" in result and isinstance(result["tags"], list):
        return f"{result.get('total', len(result['tags']))} tag(s)"
    if "topics" in result and isinstance(result["topics"], list):
        return f"{result.get('total', len(result['topics']))} topic(s)"
    if name == "get_file":
        return result.get("path", "file")
    return "ok"


def _compact(result: dict, budget: int = 6000) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= budget:
        return text
    # Trim long lists until it fits.
    trimmed = json.loads(text)
    for key in ("files", "examples", "concepts", "connections", "tags", "topics", "related", "folders"):
        if isinstance(trimmed.get(key), list) and len(trimmed[key]) > 5:
            trimmed[key] = trimmed[key][:5]
            trimmed[f"{key}_truncated"] = True
            text = json.dumps(trimmed, ensure_ascii=False, default=str)
            if len(text) <= budget:
                return text
    return text[:budget]


class Navigator:
    def __init__(self, index: FsIndex, client: Ollama | None = None, model: str | None = None,
                 max_steps: int = 6):
        self.index = index
        self.client = client or Ollama()
        self.model = model or index.state.chat_model
        self.max_steps = max_steps

    def system_message(self) -> dict:
        from datetime import date

        ov = self.index
        return {"role": "system", "content": SYSTEM_PROMPT.format(
            root=ov.root.name, today=date.today().isoformat(), files=len(ov.files),
            tags=len(ov.tags), topics=len(ov.topic_names))}

    def run(self, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        """Answer ``question``; yields events (see module docstring)."""
        self.index.maybe_reload()
        messages: list[dict] = [self.system_message()]
        for turn in (history or [])[-8:]:
            if turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
                messages.append({"role": turn["role"], "content": turn["content"][:4000]})
        messages.append({"role": "user", "content": question})
        tools = ollama_tools()
        seen_files: list[str] = []
        try:
            for step in range(self.max_steps + 1):
                allow_tools = step < self.max_steps
                content_parts: list[str] = []
                tool_calls: list[dict] = []
                for chunk in self.client.chat_stream(self.model, messages, tools=tools if allow_tools else None,
                                                     options={"num_ctx": 16384, "temperature": 0.2}):
                    msg = chunk.get("message") or {}
                    if msg.get("tool_calls"):
                        tool_calls.extend(msg["tool_calls"])
                    text = msg.get("content") or ""
                    if text:
                        content_parts.append(text)
                        if not tool_calls:
                            yield {"type": "token", "text": text}
                content = "".join(content_parts)
                if not tool_calls:
                    answer = _strip_think(content).strip()
                    files = self._cited_files(answer) or seen_files[:10]
                    yield {"type": "done", "answer": answer, "files": files}
                    return
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                for call in tool_calls:
                    fn = call.get("function") or {}
                    name = fn.get("name", "")
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    yield {"type": "tool_call", "name": name, "args": args}
                    try:
                        result = call_tool(self.index, name, args)
                    except ToolError as exc:
                        result = {"error": str(exc)}
                    except Exception as exc:  # a tool bug must not kill the chat
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                    files, nodes = _ids_in(result)
                    for f in files:
                        if f not in seen_files:
                            seen_files.append(f)
                    yield {"type": "tool_result", "name": name, "files": files, "nodes": nodes,
                           "summary": _summary(name, result)}
                    messages.append({"role": "tool", "tool_name": name, "content": _compact(result)})
        except OllamaError as exc:
            yield {"type": "error", "message": str(exc)}

    def _cited_files(self, answer: str) -> list[str]:
        out: list[str] = []
        for ref in _PATH_RE.findall(answer):
            try:
                fid = self.index.resolve_file(ref) if ("." in ref or "/" in ref) else None
            except ToolError:
                fid = None
            if fid and fid not in out:
                out.append(fid)
        return out

    def answer(self, question: str) -> dict:
        """Blocking helper (CLI / tests): returns the final ``done`` event plus the trace."""
        trace = []
        final = {"type": "done", "answer": "", "files": []}
        for ev in self.run(question):
            if ev["type"] in ("tool_call", "tool_result", "error"):
                trace.append(ev)
            if ev["type"] in ("done", "error"):
                final = ev
        return {**final, "trace": trace}


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)
