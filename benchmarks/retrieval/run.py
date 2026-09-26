"""
Runs every retrieval mode on the benchmark and writes JSON + Markdown results.

    python -m benchmarks.retrieval.run
"""
from __future__ import annotations

import json
import platform
import statistics
from datetime import date
from pathlib import Path

import numpy as np

from benchmarks.retrieval.metrics import hit_at, mrr_at, ndcg_at, recall_at
from search.corpus import load
from search.hybrid import MODES, default_searcher

ROOT = Path(__file__).resolve().parents[2]
QUERIES = ROOT / "data" / "benchmark" / "queries.jsonl"
DOCS = ROOT / "docs" / "benchmarks"


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


def main() -> dict:
    passages = load()
    queries = [json.loads(l) for l in QUERIES.read_text().splitlines() if l.strip()]
    searcher = default_searcher(passages)
    searcher.rerank_depth = 30  # the published table uses depth 30; rerank_depth.py sweeps the trade-off
    searcher.search("warmup query warfarin aspirin", k=10, mode="hybrid_rerank")
    results: dict = {}
    for mode in MODES:
        per_slice: dict[str, dict[str, list[float]]] = {}
        latencies = []
        for q in queries:
            hits = searcher.search(q["query"], k=10, mode=mode)
            latencies.append(searcher.last_timings.total_ms)
            ranked, rel = [h.id for h in hits], set(q["relevant"])
            for s in (q["slice"], "all"):
                m = per_slice.setdefault(s, {"hit@1": [], "hit@5": [], "recall@5": [], "recall@10": [],
                                             "mrr@10": [], "ndcg@10": []})
                m["hit@1"].append(hit_at(ranked, rel, 1))
                m["hit@5"].append(hit_at(ranked, rel, 5))
                m["recall@5"].append(recall_at(ranked, rel, 5))
                m["recall@10"].append(recall_at(ranked, rel, 10))
                m["mrr@10"].append(mrr_at(ranked, rel, 10))
                m["ndcg@10"].append(ndcg_at(ranked, rel, 10))
        results[mode] = {
            "slices": {s: {k: round(statistics.fmean(v), 4) for k, v in m.items()} | {"n": len(m["hit@1"])}
                       for s, m in per_slice.items()},
            "latency_ms": {"p50": round(pct(latencies, 50), 2), "p95": round(pct(latencies, 95), 2),
                           "p99": round(pct(latencies, 99), 2)},
        }
    report = {
        "date": str(date.today()),
        "corpus": {"passages": len(passages), "labels": len({p["drug"] for p in passages}),
                   "source": "openFDA drug label `drug_interactions` sections"},
        "queries": {"total": len(queries), "clean": sum(q["slice"] == "clean" for q in queries),
                    "typo": sum(q["slice"] == "typo" for q in queries)},
        "relevance": "mention-based proxy: label A passages naming B, plus label B passages naming A",
        "models": {"dense": "sentence-transformers/all-MiniLM-L6-v2",
                   "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "bm25": "rank_bm25 Okapi",
                   "fusion": "Reciprocal Rank Fusion, k=60, 50 candidates per retriever, rerank depth 30"},
        "hardware": f"{platform.machine()} CPU, {platform.system()}; single query at a time",
        "results": results,
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    stem = DOCS / f"retrieval-{report['date']}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
    lines = [f"# Retrieval benchmark — {report['date']}", "",
             f"{report['corpus']['passages']} passages from {report['corpus']['labels']} FDA labels; "
             f"{report['queries']['total']} queries ({report['queries']['clean']} clean, "
             f"{report['queries']['typo']} with a misspelled drug). Relevance: {report['relevance']}.", "",
             "| Mode | Slice | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mode, r in results.items():
        for s in ("clean", "typo", "all"):
            m = r["slices"][s]
            lines.append(f"| {mode} | {s} | {m['hit@1']:.3f} | {m['hit@5']:.3f} | {m['recall@10']:.3f} | "
                         f"{m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | {r['latency_ms']['p50']} | {r['latency_ms']['p95']} |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    rep = main()
    for mode, r in rep["results"].items():
        a = r["slices"]["all"]; c = r["slices"]["clean"]; t = r["slices"]["typo"]
        print(f"{mode:14s} all: hit@5={a['hit@5']:.3f} mrr={a['mrr@10']:.3f} ndcg={a['ndcg@10']:.3f} | "
              f"clean mrr={c['mrr@10']:.3f} | typo mrr={t['mrr@10']:.3f} | p50={r['latency_ms']['p50']}ms "
              f"p95={r['latency_ms']['p95']}ms")
