"""
Model clients for schema-constrained generation. Every agent role asks for JSON that must match
a schema, so a malformed answer is a detectable failure instead of free text downstream.

  OllamaClient      native /api/chat with `format=<json schema>` (llama.cpp grammar-constrained)
  OpenAICompatible  /v1/chat/completions with response_format=json_schema (OpenAI, vLLM, gateway)
  ScriptedLLM       deterministic responses for tests
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx


@dataclass
class LLMResult:
    data: dict
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str | None = None, base_url: str | None = None, timeout_s: float = 300):
        self.model = model or os.environ.get("AGENT_MODEL", "llama3.2")
        self.base_url = (base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout_s = timeout_s
        self.name = f"ollama:{self.model}"

    def generate(self, system: str, user: str, schema: dict, max_tokens: int = 400) -> LLMResult:
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{self.base_url}/api/chat", timeout=self.timeout_s, json={
                "model": self.model, "stream": False, "format": schema,
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
            body = r.json()
            data = json.loads(body["message"]["content"])
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        return LLMResult(data, body.get("prompt_eval_count", 0), body.get("eval_count", 0), time.perf_counter() - t0)


class OpenAICompatible:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", timeout_s: float = 60):
        self.base_url, self.model, self.api_key, self.timeout_s = base_url.rstrip("/"), model, api_key, timeout_s
        self.name = f"openai-compatible:{model}"

    def generate(self, system: str, user: str, schema: dict, max_tokens: int = 400) -> LLMResult:
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", timeout=self.timeout_s,
                           headers={"Authorization": f"Bearer {self.api_key}"}, json={
                               "model": self.model, "temperature": 0, "max_tokens": max_tokens,
                               "response_format": {"type": "json_schema",
                                                   "json_schema": {"name": "out", "schema": schema}},
                               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
            body = r.json()
            data = json.loads(body["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, ValueError, IndexError) as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        u = body.get("usage", {})
        return LLMResult(data, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.perf_counter() - t0)


@dataclass
class ScriptedLLM:
    """Test double: `script` maps a role ("You are the <role>" in the system prompt) to a function of the user prompt."""
    script: dict[str, Callable[[str], dict]]
    name: str = "scripted"
    calls: list[str] = field(default_factory=list)

    def generate(self, system: str, user: str, schema: dict, max_tokens: int = 400) -> LLMResult:
        for role, fn in self.script.items():
            if system.startswith(f"You are the {role}"):
                self.calls.append(role)
                return LLMResult(fn(user), len(user) // 4, 50, 0.0)
        raise LLMError("no scripted response")
