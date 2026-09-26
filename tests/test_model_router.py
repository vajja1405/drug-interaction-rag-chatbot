"""Model router: primary success, each fallback reason, and circuit-breaker behaviour (no network)."""
import json

import httpx
import pytest

from inference.router import CircuitBreaker, ModelRouter, Provider, RouteError

PRIMARY = Provider("self_hosted", "http://primary/v1", "local-model", timeout_s=1)
FALLBACK = Provider("external", "http://fallback/v1", "gpt-4o-mini", timeout_s=1)


def reply(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def make_transport(primary_behaviour, calls):
    def handler(request: httpx.Request):
        host = request.url.host
        calls.append(host)
        if host == "fallback":
            return httpx.Response(200, json=reply('{"ok": true}'))
        return primary_behaviour(request)
    return httpx.MockTransport(handler)


def router_with(primary_behaviour, breaker=None):
    calls = []
    return ModelRouter(PRIMARY, FALLBACK, breaker, make_transport(primary_behaviour, calls)), calls


def test_primary_answers_when_healthy():
    r, calls = router_with(lambda req: httpx.Response(200, json=reply("hello")))
    out = r.complete({"messages": []})
    assert out["x_provider"] == "self_hosted" and calls == ["primary"]


@pytest.mark.parametrize("behaviour, reason", [
    (lambda req: httpx.Response(500), "error"),
    (lambda req: httpx.Response(200, json=reply("")), "empty"),
    (lambda req: httpx.Response(200, json={"weird": 1}), "malformed"),
])
def test_fallback_reasons(behaviour, reason):
    r, calls = router_with(behaviour)
    out = r.complete({"messages": []})
    assert out["x_provider"] == "external" and out["x_fallback_reason"] == reason


def test_invalid_json_triggers_fallback_only_when_json_requested():
    r, _ = router_with(lambda req: httpx.Response(200, json=reply("not json")))
    assert r.complete({"messages": []})["x_provider"] == "self_hosted"
    out = r.complete({"messages": [], "response_format": {"type": "json_object"}})
    assert out["x_fallback_reason"] == "invalid_json"


def test_timeout_falls_back():
    def slow(req):
        raise httpx.ReadTimeout("slow", request=req)
    r, _ = router_with(slow)
    assert r.complete({"messages": []})["x_fallback_reason"] == "timeout"


def test_breaker_opens_then_half_opens():
    now = [0.0]
    breaker = CircuitBreaker(threshold=2, cooldown_s=30, clock=lambda: now[0])
    r, calls = router_with(lambda req: httpx.Response(503), breaker)
    r.complete({"messages": []}); r.complete({"messages": []})      # two failures open the breaker
    calls.clear()
    out = r.complete({"messages": []})
    assert out["x_fallback_reason"] == "breaker_open" and calls == ["fallback"]  # primary skipped
    now[0] = 31.0
    calls.clear()
    r.complete({"messages": []})
    assert calls[0] == "primary"                                    # half-open probe


def test_no_fallback_raises():
    r = ModelRouter(PRIMARY, None, transport=httpx.MockTransport(lambda req: httpx.Response(500)))
    with pytest.raises(RouteError):
        r.complete({"messages": []})


def test_both_providers_down_raises_route_error_not_500():
    def down(req):
        raise httpx.ConnectError("refused", request=req)
    r = ModelRouter(PRIMARY, FALLBACK, transport=httpx.MockTransport(down))
    with pytest.raises(RouteError, match="fallback failed"):
        r.complete({"messages": []})
