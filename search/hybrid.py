"""
search/hybrid.py
────────────────
Four retrieval modes over the same passage list, so they can be compared on one benchmark:

  bm25           lexical BM25 (exact drug names, dosing terms)
  dense          sentence-embedding cosine search (paraphrases, typos, synonyms)
  hybrid         Reciprocal Rank Fusion of bm25 + dense (no score calibration needed)
  hybrid_rerank  hybrid candidates re-scored by a cross-encoder

Query → BM25 top-N ┐
                   ├─ RRF fuse → top-M candidates → cross-encoder → top-k
Query → dense top-N┘
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by can could do does for from how i in is it its may me of on or "
    "should that the their them these this to was were what when which will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d)), ranks starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Encoder(Protocol):
    def encode(self, texts: list[str], **kw) -> np.ndarray: ...


class Scorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kw) -> np.ndarray: ...


@dataclass
class Hit:
    id: str
    drug: str
    text: str
    score: float
    set_id: str | None = None


@dataclass
class SearchTimings:
    bm25_ms: float = 0.0
    dense_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.bm25_ms + self.dense_ms + self.rerank_ms


@dataclass
class HybridSearcher:
    passages: list[dict]
    encoder: Encoder
    reranker: Scorer | None = None
    candidates: int = 50
    rerank_depth: int = 30
    rrf_k: int = 60
    embeddings: np.ndarray | None = None
    last_timings: SearchTimings = field(default_factory=SearchTimings)

    def __post_init__(self) -> None:
        self._bm25 = BM25Okapi([tokenize(p["text"]) for p in self.passages])
        if self.embeddings is None:
            self.embeddings = self._embed([p["text"] for p in self.passages])

    # ── building blocks ────────────────────────────────────────────────────
    def _embed(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(self.encoder.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

    def bm25_rank(self, query: str, n: int) -> list[int]:
        t0 = time.perf_counter()
        scores = self._bm25.get_scores(tokenize(query))
        order = [int(i) for i in np.argsort(-scores)[:n] if scores[i] > 0]
        self.last_timings.bm25_ms = (time.perf_counter() - t0) * 1000
        return order

    def dense_rank(self, query: str, n: int) -> list[int]:
        t0 = time.perf_counter()
        q = self._embed([query])[0]
        order = [int(i) for i in np.argsort(-(self.embeddings @ q))[:n]]
        self.last_timings.dense_ms = (time.perf_counter() - t0) * 1000
        return order

    # ── public API ─────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 10, mode: str = "hybrid_rerank") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.last_timings = SearchTimings()
        if mode == "bm25":
            ranked = [(i, 1.0 / r) for r, i in enumerate(self.bm25_rank(query, k), start=1)]
        elif mode == "dense":
            ranked = [(i, 1.0 / r) for r, i in enumerate(self.dense_rank(query, k), start=1)]
        else:
            fused = rrf_fuse([self.bm25_rank(query, self.candidates),
                              self.dense_rank(query, self.candidates)], k=self.rrf_k)
            if mode == "hybrid" or self.reranker is None:
                ranked = fused[:k]
            else:
                pool = [i for i, _ in fused[: self.rerank_depth]]
                t0 = time.perf_counter()
                scores = self.reranker.predict([(query, self.passages[i]["text"]) for i in pool])
                self.last_timings.rerank_ms = (time.perf_counter() - t0) * 1000
                ranked = sorted(zip(pool, map(float, scores)), key=lambda kv: -kv[1])[:k]
        return [Hit(id=self.passages[i]["id"], drug=self.passages[i]["drug"], text=self.passages[i]["text"],
                    score=round(float(s), 6), set_id=self.passages[i].get("set_id")) for i, s in ranked]


def default_searcher(passages: list[dict], embeddings: np.ndarray | None = None,
                     with_reranker: bool = True) -> HybridSearcher:
    """Production wiring: MiniLM bi-encoder + MS MARCO MiniLM cross-encoder (CPU-friendly)."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = os.environ.get("SEARCH_DEVICE", "cpu")  # CPU is deterministic and fits small containers
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    rr = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device) if with_reranker else None
    # Depth 10 is the measured operating point: ~3.6x faster than 30 for most of the quality gain
    # (docs/benchmarks/rerank-depth-*.json).
    depth = int(os.environ.get("EVIDENCE_RERANK_DEPTH", "10"))
    return HybridSearcher(passages, enc, rr, embeddings=embeddings, rerank_depth=depth)
