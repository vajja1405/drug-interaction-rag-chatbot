"""
inference/router.py
───────────────────
Routes chat completions to a self-hosted OpenAI-compatible server first (vLLM, Ollama,
TGI and llama.cpp all expose /v1/chat/completions) and falls back to an external API when
the primary fails, times out, returns an empty answer, or returns invalid JSON when JSON
was requested. A circuit breaker skips the primary for a cooldown after repeated failures,
so an outage costs one timeout, not one per request.

    request ─▶ primary (self-hosted) ──ok──▶ response
                  │ error / timeout / empty / bad JSON / breaker open
                  ▼
               fallback (external API) ─────▶ response  (+ llm_fallback_total{reason})
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from observability import LLM_FALLBACK, LLM_LATENCY


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str = "not-needed"
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls, prefix: str, name: str) -> "Provider | None":
        base = os.environ.get(f"{prefix}_BASE_URL")
        if not base:
            return None
        return cls(name=name, base_url=base.rstrip("/"), model=os.environ.get(f"{prefix}_MODEL", ""),
                   api_key=os.environ.get(f"{prefix}_API_KEY", "not-needed"),
                   timeout_s=float(os.environ.get(f"{prefix}_TIMEOUT_S", "30")))


@dataclass
class CircuitBreaker:
    threshold: int = 3
    cooldown_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            if self.clock() - self.opened_at >= self.cooldown_s:  # half-open: let one request probe
                self.opened_at = None
                self.failures = self.threshold - 1
                return True
            return False

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self.failures, self.opened_at = 0, None
            else:
                self.failures += 1
                if self.failures >= self.threshold:
                    self.opened_at = self.clock()


class RouteError(RuntimeError):
    pass


class ModelRouter:
    def __init__(self, primary: Provider | None, fallback: Provider | None,
                 breaker: CircuitBreaker | None = None, transport: httpx.BaseTransport | None = None):
        if not (primary or fallback):
            raise ValueError("configure at least one provider")
        self.primary, self.fallback = primary, fallback
        self.breaker = breaker or CircuitBreaker()
        self._client = httpx.Client(transport=transport)

    @classmethod
    def from_env(cls) -> "ModelRouter":
        return cls(Provider.from_env("PRIMARY", "self_hosted"), Provider.from_env("FALLBACK", "external"),
                   CircuitBreaker(int(os.environ.get("BREAKER_THRESHOLD", "3")),
                                  float(os.environ.get("BREAKER_COOLDOWN_S", "60"))))

    def _call(self, p: Provider, body: dict) -> dict:
        t0 = time.perf_counter()
        try:
            r = self._client.post(f"{p.base_url}/chat/completions", json=body | {"model": p.model or body.get("model")},
                                  headers={"Authorization": f"Bearer {p.api_key}"}, timeout=p.timeout_s)
            r.raise_for_status()
            return r.json()
        finally:
            LLM_LATENCY.labels(p.name).observe(time.perf_counter() - t0)

    @staticmethod
    def _problem(resp: dict, wants_json: bool) -> str | None:
        try:
            text = resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return "malformed"
        if not text.strip():
            return "empty"
        if wants_json:
            try:
                json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
            except ValueError:
                return "invalid_json"
        return None

    def complete(self, body: dict) -> dict:
        wants_json = (body.get("response_format") or {}).get("type") == "json_object"
        reason = "no_primary"
        if self.primary is not None:
            if self.breaker.allow():
                try:
                    resp = self._call(self.primary, body)
                    reason = self._problem(resp, wants_json)
                    if reason is None:
                        self.breaker.record(True)
                        return resp | {"x_provider": self.primary.name}
                except httpx.TimeoutException:
                    reason = "timeout"
                except httpx.HTTPError:
                    reason = "error"
                self.breaker.record(False)
            else:
                reason = "breaker_open"
        if self.fallback is None:
            raise RouteError(f"primary unavailable ({reason}) and no fallback configured")
        LLM_FALLBACK.labels(reason).inc()
        try:
            resp = self._call(self.fallback, body)
        except httpx.HTTPError as exc:
            raise RouteError(f"primary unavailable ({reason}) and fallback failed ({type(exc).__name__})") from exc
        return resp | {"x_provider": self.fallback.name, "x_fallback_reason": reason}
