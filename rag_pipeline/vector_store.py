"""
rag_pipeline/vector_store.py
──────────────────────────────
FAISS-backed vector store for drug interaction documents.

Index types
───────────
  IndexFlatIP   – exact cosine search (unit vectors).
                  Recommended for up to ~500 k documents.
  IndexIVFFlat  – approximate search with inverted file index.
                  Recommended for > 500 k documents; requires training step.

Architecture
────────────
  ┌──────────────────────────────────────────────────────────┐
  │  DrugVectorStore                                          │
  │                                                           │
  │  faiss.Index         ←→  parallel lists                  │
  │  (unit-normalised)       doc_texts[i]   metadata[i]      │
  │                                                           │
  │  _pair_index:  frozenset({drug_a, drug_b}) → list[int]   │
  │    O(1) exact lookup – built on load / build              │
  └──────────────────────────────────────────────────────────┘

Persistence
───────────
  data/vectorstore/drug_interactions.faiss  – FAISS binary index
  data/vectorstore/metadata.pkl             – doc_texts + metadata

New in this version vs embedding_model-era vector_store
────────────────────────────────────────────────────────
• add_documents()   – incremental adds without full rebuild
• filter_search()   – post-filter results by severity level
• mmr_search()      – Maximal Marginal Relevance (reduces redundancy)
• Large-scale mode  – IndexIVFFlat with configurable nlist / nprobe
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

_INDEX_FILE = "drug_interactions.faiss"
_META_FILE = "metadata.pkl"

IndexType = Literal["flat", "ivf"]


class DrugVectorStore:
    """
    Manages a FAISS index alongside parallel metadata and text lists.

    The integer ID at position i in the FAISS index corresponds to
    doc_texts[i] and metadata[i], making lookups O(1) by position.
    """

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.doc_texts: list[str] = []
        self.metadata: list[dict] = []
        # Fast pair lookup: frozenset({drug_a, drug_b}) → list of FAISS IDs
        self._pair_index: dict[frozenset, list[int]] = {}

    # ══════════════════════════════════════════════════════════════════════════
    # Build
    # ══════════════════════════════════════════════════════════════════════════

    def build(
        self,
        documents: list[dict],
        embeddings: np.ndarray,
        index_type: IndexType = "flat",
        ivf_nlist: int = 100,
    ) -> None:
        """
        Construct the FAISS index from precomputed, unit-normalised embeddings.

        Parameters
        ----------
        documents : list[dict]
            Each dict must have 'document_text' (str) and 'metadata' (dict).
        embeddings : np.ndarray
            float32 array shape (len(documents), dim). Must be unit-normalised.
        index_type : "flat" | "ivf"
            "flat" → IndexFlatIP (exact, default).
            "ivf"  → IndexIVFFlat (approximate, faster for large corpora).
        ivf_nlist : int
            Number of IVF clusters (only used when index_type="ivf").
            Typical rule: nlist ≈ sqrt(N) where N = number of vectors.
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(documents)} documents vs "
                f"{embeddings.shape[0]} embeddings"
            )

        dim = embeddings.shape[1]
        emb_f32 = embeddings.astype(np.float32)

        if index_type == "ivf":
            # IVFFlat requires a training step on a representative sample
            quantiser = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantiser, dim, ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info(
                "Training IndexIVFFlat (nlist=%d) on %d vectors …",
                ivf_nlist,
                len(emb_f32),
            )
            self.index.train(emb_f32)
            self.index.nprobe = max(1, ivf_nlist // 10)   # search 10 % of clusters
        else:
            # Exact inner product search (cosine for unit vectors)
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(emb_f32)

        self.doc_texts = [doc["document_text"] for doc in documents]
        self.metadata = [doc["metadata"] for doc in documents]
        self._build_pair_index()

        logger.info(
            "FAISS %s index built: %d vectors (dim=%d)",
            index_type,
            self.index.ntotal,
            dim,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Incremental add
    # ══════════════════════════════════════════════════════════════════════════

    def add_documents(
        self,
        documents: list[dict],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add new documents to an existing index without a full rebuild.

        This is useful for streaming updates – e.g., adding newly fetched
        drug interaction records to a running server.

        Returns the number of documents successfully added.
        """
        if self.index is None:
            raise RuntimeError(
                "Index not initialised. Call build() or load() first."
            )
        if len(documents) != embeddings.shape[0]:
            raise ValueError("documents and embeddings must have equal length")

        # Detect duplicate pairs before inserting
        new_docs, new_embs = [], []
        for doc, emb in zip(documents, embeddings):
            meta = doc.get("metadata", {})
            pair_key = frozenset([
                meta.get("drug_a", ""), meta.get("drug_b", "")
            ])
            if pair_key in self._pair_index:
                logger.debug("Skipping duplicate pair: %s", pair_key)
                continue
            new_docs.append(doc)
            new_embs.append(emb)

        if not new_docs:
            logger.info("add_documents: all %d documents are duplicates, skipped", len(documents))
            return 0

        start_id = len(self.doc_texts)
        new_embs_arr = np.array(new_embs, dtype=np.float32)
        self.index.add(new_embs_arr)

        for i, doc in enumerate(new_docs):
            self.doc_texts.append(doc["document_text"])
            meta = doc.get("metadata", {})
            self.metadata.append(meta)
            # Update pair index for the newly added document
            drug_a = meta.get("drug_a", "")
            drug_b = meta.get("drug_b", "")
            if drug_a and drug_b:
                key = frozenset([drug_a, drug_b])
                self._pair_index.setdefault(key, []).append(start_id + i)

        added = len(new_docs)
        logger.info(
            "add_documents: added %d new documents (index total: %d)",
            added,
            self.index.ntotal,
        )
        return added

    # ══════════════════════════════════════════════════════════════════════════
    # Search
    # ══════════════════════════════════════════════════════════════════════════

    def search(
        self,
        query_embedding: np.ndarray,
        k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Standard k-nearest-neighbour semantic search.

        Parameters
        ----------
        query_embedding : 1-D float32 array of shape (dim,), unit-normalised.
        k               : Number of results to return (default: settings.rag_top_k).
        threshold       : Minimum cosine similarity [0, 1].

        Returns
        -------
        list[dict]  Each dict: { 'text', 'metadata', 'similarity_score' }
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        k = k or settings.rag_top_k
        threshold = threshold if threshold is not None else settings.similarity_threshold

        query_2d = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_2d, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < threshold:
                break
            results.append({
                "text": self.doc_texts[idx],
                "metadata": self.metadata[idx],
                "similarity_score": round(float(score), 4),
            })
        return results

    def filter_search(
        self,
        query_embedding: np.ndarray,
        severity_filter: list[str] | None = None,
        k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search with optional post-retrieval metadata filter.

        Retrieves 4× the requested k to ensure enough results remain
        after the severity filter is applied.

        Parameters
        ----------
        severity_filter : list[str] | None
            E.g. ["Severe", "Moderate"] – only return results whose
            metadata severity matches one of these values.
            None means no filtering.
        """
        oversample_k = (k or settings.rag_top_k) * 4
        raw_results = self.search(query_embedding, k=oversample_k, threshold=threshold)

        if severity_filter is None:
            return raw_results[: k or settings.rag_top_k]

        # Normalise to title case for comparison
        allowed = {s.strip().title() for s in severity_filter}
        filtered = [
            r for r in raw_results
            if r["metadata"].get("severity", "").strip().title() in allowed
        ]
        return filtered[: k or settings.rag_top_k]

    def mmr_search(
        self,
        query_embedding: np.ndarray,
        k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float = 0.5,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Maximal Marginal Relevance (MMR) search.

        Balances relevance (similarity to query) with diversity (dissimilarity
        to already-selected results).  Reduces the number of near-duplicate
        documents when multiple records describe the same interaction.

        Parameters
        ----------
        k         : Final number of documents to return.
        fetch_k   : Initial candidate pool size before MMR re-ranking
                    (default: 3 × k).
        lambda_mult : Trade-off parameter [0, 1].
                      1.0 = pure relevance, 0.0 = pure diversity.
        """
        k = k or settings.rag_top_k
        fetch_k = fetch_k or k * 3

        candidates = self.search(query_embedding, k=fetch_k, threshold=threshold)
        if not candidates:
            return []
        if len(candidates) <= k:
            return candidates

        # Retrieve the raw embedding vectors for MMR re-ranking
        candidate_indices = self._texts_to_indices(
            [c["text"] for c in candidates]
        )
        candidate_vecs = self._get_vectors(candidate_indices)

        # Run MMR selection
        selected_positions: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_positions) < k and remaining:
            if not selected_positions:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: candidates[i]["similarity_score"])
            else:
                # Subsequent picks: balance relevance and diversity
                selected_vecs = candidate_vecs[selected_positions]
                best = max(
                    remaining,
                    key=lambda i: (
                        lambda_mult * candidates[i]["similarity_score"]
                        - (1 - lambda_mult) * float(
                            np.max(selected_vecs @ candidate_vecs[i])
                        )
                    ),
                )
            selected_positions.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected_positions]

    # ══════════════════════════════════════════════════════════════════════════
    # Exact pair lookup
    # ══════════════════════════════════════════════════════════════════════════

    def exact_pair_lookup(self, drug_a: str, drug_b: str) -> list[dict[str, Any]]:
        """
        O(1) lookup by drug pair key.
        Returns all stored documents for the pair with similarity_score=1.0.
        """
        key = frozenset([drug_a.lower(), drug_b.lower()])
        ids = self._pair_index.get(key, [])
        return [
            {
                "text": self.doc_texts[i],
                "metadata": self.metadata[i],
                "similarity_score": 1.0,
            }
            for i in ids
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, directory: str | None = None) -> None:
        """Save FAISS index and metadata to disk."""
        directory = directory or settings.vector_store_path
        Path(directory).mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(Path(directory) / _INDEX_FILE))
        with open(Path(directory) / _META_FILE, "wb") as fh:
            pickle.dump(
                {"doc_texts": self.doc_texts, "metadata": self.metadata},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(
            "Vector store saved → %s  (%d vectors)", directory, self.index.ntotal
        )

    @classmethod
    def load(cls, directory: str | None = None) -> "DrugVectorStore":
        """
        Load a saved vector store from disk.

        Raises FileNotFoundError if the index files are absent.
        Run `python build_index.py` to create them.
        """
        directory = directory or settings.vector_store_path
        faiss_path = Path(directory) / _INDEX_FILE
        meta_path = Path(directory) / _META_FILE

        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Vector store not found in '{directory}'. "
                "Run:  python build_index.py"
            )

        store = cls()
        store.index = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as fh:
            payload = pickle.load(fh)
        store.doc_texts = payload["doc_texts"]
        store.metadata = payload["metadata"]
        store._build_pair_index()

        logger.info(
            "Vector store loaded ← %s  (%d vectors)", directory, store.index.ntotal
        )
        return store

    # ══════════════════════════════════════════════════════════════════════════
    # Introspection
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    def stats(self) -> dict[str, Any]:
        return {
            "total_documents": self.total_vectors,
            "unique_pairs": len(self._pair_index),
            "index_type": type(self.index).__name__ if self.index else "None",
        }

    def get_all_known_drugs(self) -> list[str]:
        """Return a sorted, deduplicated list of all drug names in the store."""
        drugs: set[str] = set()
        for meta in self.metadata:
            if meta.get("drug_a"):
                drugs.add(meta["drug_a"])
            if meta.get("drug_b"):
                drugs.add(meta["drug_b"])
        return sorted(drugs)

    # ══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _build_pair_index(self) -> None:
        self._pair_index = {}
        for i, meta in enumerate(self.metadata):
            a, b = meta.get("drug_a", ""), meta.get("drug_b", "")
            if a and b:
                self._pair_index.setdefault(frozenset([a, b]), []).append(i)

    def _texts_to_indices(self, texts: list[str]) -> list[int]:
        """Map text strings back to their FAISS index positions."""
        text_to_idx = {t: i for i, t in enumerate(self.doc_texts)}
        return [text_to_idx.get(t, 0) for t in texts]

    def _get_vectors(self, indices: list[int]) -> np.ndarray:
        """Reconstruct stored embedding vectors by FAISS ID (for MMR)."""
        if not hasattr(self.index, "reconstruct"):
            # IVFFlat may not support reconstruct; fall back to zeros
            return np.zeros((len(indices), self.index.d), dtype=np.float32)
        vecs = np.empty((len(indices), self.index.d), dtype=np.float32)
        for i, idx in enumerate(indices):
            try:
                self.index.reconstruct(idx, vecs[i])
            except Exception:
                vecs[i] = 0.0
        return vecs
