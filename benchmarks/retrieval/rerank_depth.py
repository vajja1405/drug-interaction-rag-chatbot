"""Quality vs latency as the cross-encoder rerank depth changes (hybrid_rerank mode only)."""
from __future__ import annotations

import json
import statistics
from datetime import date
from pathlib import Path

import numpy as np

from benchmarks.retrieval.metrics import hit_at, mrr_at
from search.corpus import load
from search.hybrid import default_searcher

ROOT = Path(__file__).resolve().parents[2]


def main(depths=(5, 10, 15, 20, 30)):
    queries = [json.loads(l) for l in (ROOT / "data/benchmark/queries.jsonl").read_text().splitlines() if l]
    s = default_searcher(load())
    s.search("warmup warfarin aspirin", mode="hybrid_rerank")
    rows = []
    for d in depths:
        s.rerank_depth = d
        mrr, hit5, lat = [], [], []
        for q in queries:
            hits = s.search(q["query"], k=10, mode="hybrid_rerank")
            lat.append(s.last_timings.total_ms)
            ids, rel = [h.id for h in hits], set(q["relevant"])
            mrr.append(mrr_at(ids, rel)); hit5.append(hit_at(ids, rel, 5))
        rows.append({"depth": d, "mrr@10": round(statistics.fmean(mrr), 4), "hit@5": round(statistics.fmean(hit5), 4),
                     "p50_ms": round(float(np.percentile(lat, 50)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1)})
        print(rows[-1], flush=True)
    out = ROOT / "docs/benchmarks" / f"rerank-depth-{date.today()}.json"
    out.write_text(json.dumps({"queries": len(queries), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
