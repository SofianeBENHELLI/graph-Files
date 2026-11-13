"""Minimal client for a local Ollama server (native ``/api`` endpoints).

Used for topic tagging (JSON output) and for the explorer's chat agent (tool
calling, streamed). The base URL follows graphify's own rules: ``OLLAMA_BASE_URL``
(an OpenAI-style ``.../v1`` URL is accepted) or ``OLLAMA_HOST``, else
``http://localhost:11434``.
"""
from __future__ import annotations

import json
import os
from typing import Iterator

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=5.0)


class OllamaError(RuntimeError):
    pass


def base_url() -> str:
    from graphify.llm import _resolve_ollama_base_url

    url = _resolve_ollama_base_url("http://localhost:11434/v1").rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


class Ollama:
    def __init__(self, url: str | None = None, timeout: httpx.Timeout = DEFAULT_TIMEOUT, transport=None):
        self.url = (url or base_url()).rstrip("/")
        self._client = httpx.Client(base_url=self.url, timeout=timeout, transport=transport)
        self._no_think: set[str] = set()

    def close(self) -> None:
        self._client.close()

    # ---- discovery -------------------------------------------------------
    def available(self) -> bool:
        try:
            return self._client.get("/api/version", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def models(self) -> list[str]:
        try:
            resp = self._client.get("/api/tags", timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama is not reachable at {self.url}: {exc}") from exc
        return [m.get("name") or m.get("model") for m in resp.json().get("models", [])]

    def has_model(self, model: str) -> bool:
        names = self.models()
        want = model if ":" in model else model + ":latest"
        return any(n == model or n == want for n in names)

    def pull(self, model: str, on_progress=None) -> None:
        with self._client.stream("POST", "/api/pull", json={"model": model, "stream": True}, timeout=None) as resp:
            if resp.status_code != 200:
                raise OllamaError(f"pull {model} failed: HTTP {resp.status_code}")
            for line in resp.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise OllamaError(msg["error"])
                if on_progress:
                    on_progress(msg)

    # ---- chat ------------------------------------------------------------
    def _payload(self, model, messages, tools, fmt, stream, options, think) -> dict:
        payload: dict = {"model": model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        if fmt is not None:
            payload["format"] = fmt
        payload["options"] = {"temperature": 0.2, **(options or {})}
        if think is not None and model not in self._no_think:
            payload["think"] = think
        payload["keep_alive"] = os.environ.get("GRAPHIFY_FS_KEEP_ALIVE", "15m")
        return payload

    def _maybe_retry_without_think(self, model: str, resp: httpx.Response) -> bool:
        if resp.status_code == 400 and "think" in resp.text.lower() and model not in self._no_think:
            self._no_think.add(model)
            return True
        return False

    def chat(self, model: str, messages: list[dict], *, tools: list[dict] | None = None,
             fmt=None, options: dict | None = None, think: bool | None = False) -> dict:
        """One non-streamed chat turn; returns the ``message`` dict."""
        for _ in range(2):
            payload = self._payload(model, messages, tools, fmt, False, options, think)
            try:
                resp = self._client.post("/api/chat", json=payload)
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama request failed ({self.url}): {exc}") from exc
            if self._maybe_retry_without_think(model, resp):
                continue
            if resp.status_code != 200:
                raise OllamaError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
            return resp.json().get("message", {})
        raise OllamaError("Ollama chat failed")

    def chat_stream(self, model: str, messages: list[dict], *, tools: list[dict] | None = None,
                    options: dict | None = None, think: bool | None = False) -> Iterator[dict]:
        """Streamed chat turn; yields Ollama's per-chunk JSON objects."""
        for attempt in range(2):
            payload = self._payload(model, messages, tools, None, True, options, think)
            try:
                with self._client.stream("POST", "/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = resp.read().decode("utf-8", "replace")
                        fake = httpx.Response(resp.status_code, text=body)
                        if attempt == 0 and self._maybe_retry_without_think(model, fake):
                            continue
                        raise OllamaError(f"Ollama error {resp.status_code}: {body[:300]}")
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if "error" in chunk:
                            raise OllamaError(chunk["error"])
                        yield chunk
                    return
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama request failed ({self.url}): {exc}") from exc

    def chat_json(self, model: str, messages: list[dict], schema: dict | None = None,
                  options: dict | None = None) -> dict:
        """Chat turn constrained to JSON (a JSON schema when given); returns parsed JSON."""
        msg = self.chat(model, messages, fmt=schema or "json", options=options)
        content = (msg.get("content") or "").strip()
        try:
            return json.loads(content)
        except ValueError:
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end + 1])
                except ValueError:
                    pass
        raise OllamaError(f"model returned invalid JSON: {content[:200]}")
