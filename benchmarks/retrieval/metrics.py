"""Ranking metrics for a single query; `relevant` is a set of passage ids."""
from __future__ import annotations

import math


def hit_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(any(r in relevant for r in ranked[:k]))


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len([r for r in ranked[:k] if r in relevant]) / min(len(relevant), k)


def mrr_at(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for i, r in enumerate(ranked[:k], start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
