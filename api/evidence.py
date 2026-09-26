"""
Evidence search over FDA label interaction passages (hybrid BM25 + dense + cross-encoder).

GET /api/v3/evidence/search?q=...&mode=hybrid_rerank&k=5
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from observability import CACHE_EVENTS, RETRIEVAL_STAGE, span

router = APIRouter(prefix="/api/v3/evidence", tags=["Evidence search"])
ROOT = Path(__file__).resolve().parents[1]
EMB_CACHE = ROOT / "data" / "benchmark" / "passage_embeddings.npy"
_lock = threading.Lock()


class LRU:
    def __init__(self, size: int = 512) -> None:
        self.size, self._d, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


def get_searcher(request: Request):
    """Lazily build the production searcher once per process (models load on first use)."""
    searcher = getattr(request.app.state, "evidence_searcher", None)
    if searcher is not None:
        return searcher
    with _lock:
        searcher = getattr(request.app.state, "evidence_searcher", None)
        if searcher is None:
            from search.corpus import load
            from search.hybrid import default_searcher
            passages = load()
            emb = np.load(EMB_CACHE) if EMB_CACHE.exists() else None
            if emb is not None and len(emb) != len(passages):
                emb = None
            searcher = default_searcher(passages, embeddings=emb,
                                        with_reranker=os.environ.get("EVIDENCE_RERANK", "1") == "1")
            if emb is None:
                np.save(EMB_CACHE, searcher.embeddings)
            request.app.state.evidence_searcher = searcher
    return searcher


def get_cache(request: Request) -> LRU:
    cache = getattr(request.app.state, "evidence_cache", None)
    if cache is None:
        cache = request.app.state.evidence_cache = LRU(int(os.environ.get("EVIDENCE_CACHE_SIZE", "512")))
    return cache


@router.get("/search")
async def evidence_search(
    request: Request,
    q: Annotated[str, Query(min_length=3, max_length=200)],
    mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
    k: Annotated[int, Query(ge=1, le=20)] = 5,
):
    key = (q.strip().lower(), mode, k)
    cache = get_cache(request)
    cached = cache.get(key)
    if cached is not None:
        CACHE_EVENTS.labels("hit").inc()
        return cached | {"cached": True}
    CACHE_EVENTS.labels("miss").inc()
    searcher = await run_in_threadpool(get_searcher, request)

    def _run():
        with span("evidence.search", mode=mode, k=k):
            return searcher.search(q, k=k, mode=mode), searcher.last_timings

    try:
        hits, t = await run_in_threadpool(_run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for stage, ms in (("bm25", t.bm25_ms), ("dense", t.dense_ms), ("rerank", t.rerank_ms)):
        if ms:
            RETRIEVAL_STAGE.labels(stage).observe(ms / 1000)
    body = {
        "query": q, "mode": mode,
        "results": [{"id": h.id, "label_drug": h.drug, "set_id": h.set_id, "score": h.score, "text": h.text}
                    for h in hits],
        "timings_ms": {"bm25": round(t.bm25_ms, 2), "dense": round(t.dense_ms, 2),
                       "rerank": round(t.rerank_ms, 2), "total": round(t.total_ms, 2)},
        "source": "openFDA drug label `drug_interactions` sections; retrieval only, not clinical advice",
        "cached": False,
    }
    cache.put(key, body)
    return body
