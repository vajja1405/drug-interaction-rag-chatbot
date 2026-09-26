"""
Builds the retrieval benchmark from the label corpus.

For every drug pair (A, B) where label A's interaction section names drug B, the
relevant set is: passages of label A that mention B, plus passages of label B that
mention A. Relevance is therefore mention-based (a proxy), not pharmacist-judged.

Two slices per pair:
  clean – natural question with correct spelling
  typo  – same question with one character dropped from drug B (patients misspell)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from search.corpus import load

OUT = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "queries.jsonl"
TEMPLATES = [
    "Can I take {a} with {b}?",
    "{a} and {b} interaction",
    "What happens if {b} is combined with {a}?",
    "Is {b} safe while on {a}?",
    "risk of using {a} together with {b}",
]


def _typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:]


def build(n_pairs: int = 300, seed: int = 7) -> list[dict]:
    passages = load()
    vocab = sorted({p["drug"] for p in passages})
    pat = {d: re.compile(rf"\b{re.escape(d)}\b", re.I) for d in vocab}
    mentions: dict[tuple[str, str], set[str]] = {}
    for p in passages:
        for d in vocab:
            if d != p["drug"] and pat[d].search(p["text"]):
                mentions.setdefault((p["drug"], d), set()).add(p["id"])
    pairs = sorted({tuple(sorted(k)) for k in mentions})
    rng = random.Random(seed)
    rng.shuffle(pairs)
    queries = []
    for a, b in pairs[:n_pairs]:
        if rng.random() < 0.5:
            a, b = b, a
        rel = sorted(mentions.get((a, b), set()) | mentions.get((b, a), set()))
        t = rng.choice(TEMPLATES)
        queries.append({"qid": f"{a}|{b}|clean", "slice": "clean", "query": t.format(a=a, b=b), "relevant": rel})
        if len(b.replace(" ", "")) >= 6:
            queries.append({"qid": f"{a}|{b}|typo", "slice": "typo",
                            "query": t.format(a=a, b=_typo(b, rng)), "relevant": rel})
    OUT.write_text("".join(json.dumps(q) + "\n" for q in queries))
    return queries


if __name__ == "__main__":
    qs = build()
    print(f"{len(qs)} queries ({sum(q['slice']=='clean' for q in qs)} clean, "
          f"{sum(q['slice']=='typo' for q in qs)} typo) -> {OUT}")
