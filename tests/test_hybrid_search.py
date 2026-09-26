"""Hybrid evidence search: tokenizer, RRF math, mode behaviour, metrics. No model downloads."""
import numpy as np
import pytest

from benchmarks.retrieval.metrics import hit_at, mrr_at, ndcg_at, recall_at
from search.corpus import chunk
from search.hybrid import HybridSearcher, rrf_fuse, tokenize


class HashEncoder:
    """Deterministic bag-of-trigrams encoder: robust to one dropped character, like a real embedder."""
    dim = 256

    def encode(self, texts, **_):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = f"  {t.lower()}  "
            for j in range(len(s) - 2):
                out[i, hash(s[j:j + 3]) % self.dim] += 1.0
        return out


class OverlapReranker:
    def predict(self, pairs, **_):
        return np.array([len(set(tokenize(q)) & set(tokenize(p))) for q, p in pairs], dtype=float)


PASSAGES = [
    {"id": "warfarin#0", "drug": "warfarin", "text": "Aspirin increases the risk of bleeding when used with warfarin."},
    {"id": "warfarin#1", "drug": "warfarin", "text": "Monitor INR closely when starting amiodarone."},
    {"id": "metformin#0", "drug": "metformin", "text": "Carbonic anhydrase inhibitors may increase lactic acidosis risk."},
    {"id": "lisinopril#0", "drug": "lisinopril", "text": "Potassium-sparing diuretics such as spironolactone can cause hyperkalemia."},
]


@pytest.fixture(scope="module")
def searcher():
    import random
    random.seed(0)
    return HybridSearcher(PASSAGES, HashEncoder(), OverlapReranker(), candidates=4, rerank_depth=4)


def test_tokenize_drops_stopwords_and_keeps_hyphenated_terms():
    assert tokenize("Can I take the potassium-sparing drug?") == ["take", "potassium-sparing", "drug"]


def test_rrf_rewards_agreement_between_rankers():
    fused = dict(rrf_fuse([[1, 2, 3], [2, 1, 3]], k=60))
    assert fused[1] == fused[2] > fused[3]
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)


def test_bm25_finds_exact_drug_names(searcher):
    assert searcher.search("spironolactone hyperkalemia", k=1, mode="bm25")[0].id == "lisinopril#0"


def test_dense_survives_a_misspelled_drug(searcher):
    hits = searcher.search("spironolactne", k=1, mode="dense")
    assert hits[0].id == "lisinopril#0"
    assert searcher.search("spironolactne", k=1, mode="bm25") == []  # lexical miss


def test_hybrid_rerank_returns_ranked_hits_with_timings(searcher):
    hits = searcher.search("warfarin aspirin bleeding", k=2, mode="hybrid_rerank")
    assert hits[0].id == "warfarin#0" and len(hits) == 2
    assert searcher.last_timings.rerank_ms >= 0 and searcher.last_timings.total_ms > 0


def test_unknown_mode_rejected(searcher):
    with pytest.raises(ValueError):
        searcher.search("x", mode="magic")


def test_chunk_respects_word_budget():
    text = " ".join(f"Sentence number {i} mentions warfarin and aspirin together." for i in range(40))
    chunks = chunk(text, max_words=40)
    assert len(chunks) > 1 and all(len(c.split()) <= 40 for c in chunks)


def test_ranking_metrics():
    ranked, rel = ["a", "b", "c"], {"b", "z"}
    assert hit_at(ranked, rel, 1) == 0 and hit_at(ranked, rel, 2) == 1
    assert mrr_at(ranked, rel) == 0.5
    assert recall_at(ranked, rel, 3) == 0.5
    assert 0 < ndcg_at(ranked, rel) < 1
