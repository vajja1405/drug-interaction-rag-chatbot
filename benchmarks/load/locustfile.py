"""
Load test for evidence search.

    locust -f benchmarks/load/locustfile.py --headless -H http://localhost:8000 -u 50 -r 25 -t 60s --csv out
Queries are drawn from the retrieval benchmark; REPEAT_SHARE controls how many repeat a small hot set
(cache hits) versus fresh queries (full BM25 + dense + rerank path).
"""
import json
import os
import random
from pathlib import Path

from locust import FastHttpUser, between, task

QUERIES = [json.loads(l)["query"] for l in
           (Path(__file__).resolve().parents[2] / "data/benchmark/queries.jsonl").read_text().splitlines() if l]
HOT = QUERIES[:20]
REPEAT_SHARE = float(os.environ.get("REPEAT_SHARE", "0.0"))
MODE = os.environ.get("SEARCH_MODE", "hybrid_rerank")


class EvidenceUser(FastHttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def search(self):
        q = random.choice(HOT) if random.random() < REPEAT_SHARE else random.choice(QUERIES) + f" #{random.randrange(10**6)}"
        self.client.get("/api/v3/evidence/search", params={"q": q, "mode": MODE, "k": 5}, name="evidence_search")
