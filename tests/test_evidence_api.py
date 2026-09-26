"""Evidence search endpoint, its cache, and Prometheus metrics — with an injected fake searcher."""
from api.server import app
from search.hybrid import HybridSearcher
from tests.test_hybrid_search import PASSAGES, HashEncoder, OverlapReranker


def _install_fake():
    app.state.evidence_searcher = HybridSearcher(PASSAGES, HashEncoder(), OverlapReranker(),
                                                 candidates=4, rerank_depth=4)
    app.state.evidence_cache = None


def test_search_returns_attributed_hits_and_timings(client):
    _install_fake()
    r = client.get("/api/v3/evidence/search", params={"q": "warfarin aspirin bleeding", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["id"] == "warfarin#0" and body["results"][0]["label_drug"] == "warfarin"
    assert set(body["timings_ms"]) == {"bm25", "dense", "rerank", "total"} and body["cached"] is False


def test_repeat_query_is_served_from_cache(client):
    _install_fake()
    params = {"q": "spironolactone hyperkalemia", "mode": "hybrid"}
    assert client.get("/api/v3/evidence/search", params=params).json()["cached"] is False
    assert client.get("/api/v3/evidence/search", params=params).json()["cached"] is True


def test_validation(client):
    _install_fake()
    assert client.get("/api/v3/evidence/search", params={"q": "ab"}).status_code == 422
    assert client.get("/api/v3/evidence/search", params={"q": "abc", "mode": "magic"}).status_code == 422
    assert client.get("/api/v3/evidence/search", params={"q": "abc", "k": 99}).status_code == 422


def test_metrics_expose_latency_and_cache_series(client):
    _install_fake()
    client.get("/api/v3/evidence/search", params={"q": "warfarin monitor INR"})
    text = client.get("/metrics").text
    assert 'http_request_duration_seconds_bucket{le="0.005",method="GET",route="/api/v3/evidence/search"}' in text
    assert "retrieval_stage_seconds_count" in text and 'evidence_cache_events_total{result="miss"}' in text
