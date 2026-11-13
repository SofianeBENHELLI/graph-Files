"""A scripted stand-in for Ollama's native API, for graphify fs tests.

Behaves like a tiny tool-calling model:
- chat with tools, last message from the user  -> calls ``search`` with the question
- chat after a tool result                       -> answers citing the first file path found
- chat with ``format`` (topic tagging)           -> tags made from the files' concepts/keywords

Use ``handler`` with ``httpx.MockTransport`` in unit tests, or run it as a
server for end-to-end checks: ``python tests/fs_fake_ollama.py 11500``.
"""
from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["qwen3:8b", "gemma3:12b"]


def _answer_from_tool(content: str) -> str:
    try:
        data = json.loads(content)
    except ValueError:
        data = {}
    files = data.get("files") or data.get("examples") or []
    if files:
        paths = ", ".join(f"`{f['path']}`" for f in files[:3] if isinstance(f, dict) and f.get("path"))
        return f"I found {len(files)} relevant file(s). The best matches are {paths}."
    return "I could not find matching files; try other words."


def _topic_tags(user: str) -> dict:
    files = []
    for block in re.findall(r"\{[^{}]*\}", user.split("FILES:", 1)[-1], re.S):
        try:
            item = json.loads(block)
        except ValueError:
            continue
        words = (item.get("concepts") or []) + (item.get("tags") or [])
        tags = [w for w in words if isinstance(w, str)][:3] or ["general knowledge"]
        files.append({"id": item.get("id"), "tags": tags})
    return {"files": files}


_SOURCE = re.compile(r'<untrusted_source path="([^"]+)"[^>]*>(.*?)</untrusted_source>', re.S)
_STOP = {"The", "This", "That", "And", "For", "With", "From", "Next", "Present", "Owner", "Action", "Attendees",
         "Slide", "Speaker", "Voice", "All", "Risks", "Scope", "Milestones", "Item", "Cost", "Privacy"}


def _extraction(user: str) -> dict:
    """graphify-style extraction: capitalised words become concepts, linked within a file."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for path, body in _SOURCE.findall(user):
        words = []
        for w in re.findall(r"\b[A-Z][A-Za-z]{2,}\b|\bGPU\b|\bCRM\b", body):
            if w not in _STOP and w not in words:
                words.append(w)
        ids = []
        for w in words[:8]:
            nid = w.lower()
            nodes.setdefault(nid, {"id": nid, "label": w, "file_type": "concept", "source_file": path,
                                   "source_location": None})
            ids.append(nid)
        for a, b in zip(ids, ids[1:]):
            edges.append({"source": a, "target": b, "relation": "conceptually_related_to",
                          "confidence": "INFERRED", "confidence_score": 0.7, "source_file": path, "weight": 1.0})
    return {"nodes": list(nodes.values()), "edges": edges, "hyperedges": []}


def reply(payload: dict) -> list[dict]:
    """Return the list of chunks Ollama would send for this /api/chat payload."""
    messages = payload.get("messages", [])
    last = messages[-1] if messages else {}
    if "<untrusted_source" in str(last.get("content", "")) and not payload.get("tools"):
        content = json.dumps(_extraction(last["content"]))
        return [{"message": {"role": "assistant", "content": content}, "done": True,
                 "prompt_eval_count": 100, "eval_count": 200}]
    if payload.get("format") is not None and "TOPICS:" in str(last.get("content", "")):
        items = json.loads(last["content"].split("TOPICS:", 1)[1])
        named = [{"id": t["id"], "name": " & ".join((t.get("concepts") or t.get("tags") or ["Misc"])[:2])} for t in items]
        return [{"message": {"role": "assistant", "content": json.dumps({"topics": named})}, "done": True}]
    if payload.get("format") is not None:
        content = json.dumps(_topic_tags(last.get("content", "")))
        return [{"message": {"role": "assistant", "content": content}, "done": True}]
    if payload.get("tools") and last.get("role") == "user":
        call = {"function": {"name": "search", "arguments": {"query": last.get("content", "")}}}
        return [{"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True}]
    if last.get("role") == "tool":
        text = _answer_from_tool(last.get("content", ""))
        words = text.split(" ")
        chunks = [{"message": {"role": "assistant", "content": w + (" " if i < len(words) - 1 else "")}, "done": False}
                  for i, w in enumerate(words)]
        return chunks + [{"message": {"role": "assistant", "content": ""}, "done": True}]
    return [{"message": {"role": "assistant", "content": "Hello."}, "done": True}]


def handler(request):
    """httpx.MockTransport handler."""
    import httpx

    path = request.url.path
    if path == "/api/version":
        return httpx.Response(200, json={"version": "0.0-fake"})
    if path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": m} for m in MODELS]})
    if path == "/api/chat":
        payload = json.loads(request.content or b"{}")
        chunks = reply(payload)
        if payload.get("stream"):
            body = "".join(json.dumps(c) + "\n" for c in chunks)
            return httpx.Response(200, text=body)
        merged = {"role": "assistant", "content": "".join(c["message"].get("content", "") for c in chunks)}
        for c in chunks:
            if c["message"].get("tool_calls"):
                merged["tool_calls"] = c["message"]["tool_calls"]
        return httpx.Response(200, json={"message": merged, "done": True})
    return httpx.Response(404, json={"error": "not found"})


class _Server(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, status: int, body: bytes, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send(200, json.dumps({"version": "0.0-fake"}).encode())
        if self.path == "/api/tags":
            return self._send(200, json.dumps({"models": [{"name": m} for m in MODELS]}).encode())
        return self._send(404, b"{}")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/api/chat":
            return self._send(404, b"{}")
        chunks = reply(payload)
        if payload.get("stream"):
            body = "".join(json.dumps(c) + "\n" for c in chunks).encode()
            return self._send(200, body, "application/x-ndjson")
        merged = {"role": "assistant", "content": "".join(c["message"].get("content", "") for c in chunks)}
        for c in chunks:
            if c["message"].get("tool_calls"):
                merged["tool_calls"] = c["message"]["tool_calls"]
        return self._send(200, json.dumps({"message": merged, "done": True, "done_reason": "stop",
                                           "prompt_eval_count": 100, "eval_count": 200}).encode())


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11500
    ThreadingHTTPServer(("127.0.0.1", port), _Server).serve_forever()
