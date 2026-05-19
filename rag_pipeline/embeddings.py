"""
rag_pipeline/embeddings.py
───────────────────────────
Full embedding pipeline for drug interaction documents.

Responsibilities:
  1. Document chunking  – split long clinical text into fixed-size,
                          overlapping chunks so no context is cut off
  2. Embedding          – batch-encode chunks with sentence-transformers
  3. Disk caching       – avoid re-computing embeddings for unchanged docs
  4. Statistics         – track throughput, model metadata, cache hit rates

Pipeline flow
─────────────
  raw documents (list[dict])
        │
        ▼
  DocumentChunker          ← splits long texts into overlapping windows
        │
        ▼
  EmbeddingCache           ← returns precomputed vectors for seen texts
        │ (cache miss)
        ▼
  SentenceTransformer      ← encodes new texts in batches
        │
        ▼
  EmbeddingPipeline.run()  ← returns (documents_with_chunks, embeddings_array)

Key design choices
──────────────────
• Chunking strategy: character-level sliding window with 20 % overlap.
  Drug interaction records are short (150–400 tokens) so most pass through
  as a single chunk.  Long FDA label excerpts are split gracefully.
• Cache key: SHA-256 of (model_name + chunk_text).  Changing the model
  automatically invalidates all cached embeddings.
• Embeddings are always L2-normalised so dot-product == cosine similarity
  (compatible with FAISS IndexFlatIP).
"""

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import settings

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 512        # characters per chunk
DEFAULT_CHUNK_OVERLAP = 102     # ~20 % overlap
DEFAULT_BATCH_SIZE = 64
CACHE_DIR = "data/embedding_cache"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Document Chunker
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """One text chunk produced from a source document."""
    text: str
    metadata: dict           # copied from source document
    chunk_index: int         # position within the source document
    total_chunks: int        # total chunks from this source document
    char_start: int
    char_end: int


class DocumentChunker:
    """
    Splits document texts into overlapping fixed-size character windows.

    Why character-level and not token-level?
    ─────────────────────────────────────────
    Drug interaction records are short enough that character-level chunking
    is accurate and avoids a tokeniser dependency.  For the typical record
    (200–400 chars), chunk_size=512 means zero splitting.

    Parameters
    ----------
    chunk_size : int
        Maximum characters per chunk.
    overlap : int
        Characters shared between consecutive chunks (prevents context loss
        at chunk boundaries).
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._step = chunk_size - overlap

    def chunk_document(self, document: dict) -> list[Chunk]:
        """
        Split a single processed document into one or more Chunk objects.

        Parameters
        ----------
        document : dict
            Must contain 'document_text' (str) and 'metadata' (dict).
        """
        text: str = document.get("document_text", "").strip()
        metadata: dict = document.get("metadata", {})

        if not text:
            return []

        # Short document → single chunk (the common case for DDI records)
        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    text=text,
                    metadata=metadata,
                    chunk_index=0,
                    total_chunks=1,
                    char_start=0,
                    char_end=len(text),
                )
            ]

        # Long document → sliding window
        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=metadata,
                        chunk_index=len(chunks),
                        total_chunks=0,         # filled in below
                        char_start=start,
                        char_end=end,
                    )
                )
            if end == len(text):
                break
            start += self._step

        # Back-fill total_chunks now that we know the count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[Chunk]:
        """Chunk all documents and return a flat list of Chunk objects."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.debug(
            "Chunked %d documents → %d chunks (chunk_size=%d, overlap=%d)",
            len(documents),
            len(all_chunks),
            self.chunk_size,
            self.overlap,
        )
        return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Embedding Cache
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    """
    Disk-backed cache: text + model_name  →  embedding vector.

    Cache key: SHA-256(model_name + text)
    Storage format: pickle file mapping key → np.ndarray

    The cache is loaded into memory on construction and flushed to disk
    on demand (or when used as a context manager).
    """

    def __init__(
        self,
        cache_dir: str = CACHE_DIR,
        model_name: str = "",
    ) -> None:
        self._model_name = model_name
        self._cache_path = Path(cache_dir) / "embeddings.pkl"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, np.ndarray] = self._load()
        self._hits = 0
        self._misses = 0

    # ── Interface ─────────────────────────────────────────────────────────────

    def get(self, text: str) -> np.ndarray | None:
        """Return cached embedding or None if not cached."""
        key = self._make_key(text)
        vector = self._store.get(key)
        if vector is not None:
            self._hits += 1
        else:
            self._misses += 1
        return vector

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding in the in-memory cache."""
        self._store[self._make_key(text)] = embedding

    def put_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        """Store a batch of embeddings."""
        for text, emb in zip(texts, embeddings):
            self.put(text, emb)

    def flush(self) -> None:
        """Persist the in-memory cache to disk."""
        with open(self._cache_path, "wb") as fh:
            pickle.dump(self._store, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Embedding cache flushed (%d entries)", len(self._store))

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_key(self, text: str) -> str:
        raw = f"{self._model_name}::{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self) -> dict:
        if self._cache_path.exists():
            with open(self._cache_path, "rb") as fh:
                data = pickle.load(fh)
            logger.info(
                "Embedding cache loaded: %d entries from %s",
                len(data),
                self._cache_path,
            )
            return data
        return {}

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Embedding Statistics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingStats:
    """Tracks embedding pipeline performance."""
    model_name: str
    embedding_dim: int
    total_documents: int = 0
    total_chunks: int = 0
    cache_hits: int = 0
    newly_encoded: int = 0
    encoding_time_s: float = 0.0
    embeddings_shape: tuple = field(default_factory=tuple)

    @property
    def throughput(self) -> float:
        """Chunks encoded per second (excluding cache hits)."""
        if self.encoding_time_s == 0 or self.newly_encoded == 0:
            return 0.0
        return round(self.newly_encoded / self.encoding_time_s, 1)

    def report(self) -> str:
        return (
            f"EmbeddingStats:\n"
            f"  Model          : {self.model_name}\n"
            f"  Dimension      : {self.embedding_dim}\n"
            f"  Documents      : {self.total_documents}\n"
            f"  Chunks         : {self.total_chunks}\n"
            f"  Cache hits     : {self.cache_hits}\n"
            f"  Newly encoded  : {self.newly_encoded}\n"
            f"  Encoding time  : {self.encoding_time_s:.2f}s\n"
            f"  Throughput     : {self.throughput} chunks/s\n"
            f"  Output shape   : {self.embeddings_shape}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Main Embedding Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingPipeline:
    """
    End-to-end pipeline: processed documents → embedding array.

    Usage
    -----
    ::

        pipeline = EmbeddingPipeline()
        chunks, embeddings, stats = pipeline.run(documents)

        # chunks : list[Chunk]          one per row in embeddings
        # embeddings : np.ndarray       shape (N, 384), float32, L2-normalised
        # stats : EmbeddingStats        throughput & cache metrics
    """

    def __init__(
        self,
        model_name: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_cache: bool = True,
        cache_dir: str = CACHE_DIR,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size
        self.use_cache = use_cache

        # Load sentence-transformer model
        logger.info("EmbeddingPipeline: loading model %s …", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("  └─ embedding_dim = %d", self.embedding_dim)

        # Sub-components
        self._chunker = DocumentChunker(chunk_size, chunk_overlap)
        self._cache = (
            EmbeddingCache(cache_dir, self.model_name)
            if use_cache
            else None
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        documents: list[dict],
        show_progress: bool = True,
    ) -> tuple[list[Chunk], np.ndarray, EmbeddingStats]:
        """
        Full pipeline: chunk all documents, embed, return results.

        Returns
        -------
        chunks : list[Chunk]
            One chunk per row in `embeddings`.
        embeddings : np.ndarray
            Shape (len(chunks), embedding_dim), float32, unit-normalised.
        stats : EmbeddingStats
            Pipeline performance metrics.
        """
        stats = EmbeddingStats(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            total_documents=len(documents),
        )

        # ── Step 1: Chunk ─────────────────────────────────────────────────────
        chunks = self._chunker.chunk_documents(documents)
        stats.total_chunks = len(chunks)

        if not chunks:
            logger.warning("No chunks produced from %d documents", len(documents))
            return [], np.empty((0, self.embedding_dim), dtype=np.float32), stats

        # ── Step 2: Resolve cache vs. new texts ───────────────────────────────
        texts = [c.text for c in chunks]
        cached_vectors: list[np.ndarray | None] = []
        texts_to_encode: list[str] = []
        encode_indices: list[int] = []          # which positions need encoding

        for i, text in enumerate(texts):
            if self._cache:
                vec = self._cache.get(text)
            else:
                vec = None

            cached_vectors.append(vec)
            if vec is None:
                texts_to_encode.append(text)
                encode_indices.append(i)

        stats.cache_hits = stats.total_chunks - len(texts_to_encode)
        stats.newly_encoded = len(texts_to_encode)

        # ── Step 3: Encode new texts in batches ───────────────────────────────
        new_embeddings: np.ndarray | None = None
        if texts_to_encode:
            t0 = time.perf_counter()
            new_embeddings = self._encode_batches(texts_to_encode, show_progress)
            stats.encoding_time_s = round(time.perf_counter() - t0, 3)

            # Write new embeddings back to cache
            if self._cache:
                self._cache.put_batch(texts_to_encode, new_embeddings)
                self._cache.flush()

        # ── Step 4: Assemble final embedding array ────────────────────────────
        final_embeddings = np.empty(
            (len(chunks), self.embedding_dim), dtype=np.float32
        )
        new_idx = 0
        for i, cached_vec in enumerate(cached_vectors):
            if cached_vec is not None:
                final_embeddings[i] = cached_vec
            else:
                final_embeddings[i] = new_embeddings[new_idx]
                new_idx += 1

        stats.embeddings_shape = final_embeddings.shape

        logger.info(stats.report())
        return chunks, final_embeddings, stats

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single free-text query string.

        Returns a 1-D float32 array of shape (embedding_dim,), unit-normalised.
        Useful for semantic search queries like:
          "Can warfarin and ibuprofen be taken together?"
        """
        vector = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector[0].astype(np.float32)

    def embed_documents_raw(
        self, texts: list[str], show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a list of raw strings without chunking.
        Useful for re-indexing or incremental additions.
        """
        return self._encode_batches(texts, show_progress)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _encode_batches(
        self, texts: list[str], show_progress: bool
    ) -> np.ndarray:
        """Encode texts in mini-batches, return unit-normalised float32 array."""
        all_embeddings: list[np.ndarray] = []
        iterator = range(0, len(texts), self.batch_size)

        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Encoding ({self.model_name.split('/')[-1]})",
                unit="batch",
            )

        for start in iterator:
            batch = texts[start : start + self.batch_size]
            batch_embeddings = self._model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            all_embeddings.append(batch_embeddings.astype(np.float32))

        return np.vstack(all_embeddings) if all_embeddings else np.empty(
            (0, self.embedding_dim), dtype=np.float32
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Convenience factory
# ═══════════════════════════════════════════════════════════════════════════════

_pipeline_singleton: EmbeddingPipeline | None = None


def get_pipeline(
    model_name: str | None = None,
    use_cache: bool = True,
) -> EmbeddingPipeline:
    """
    Return a module-level singleton EmbeddingPipeline.

    The model is loaded only on first call; subsequent calls return the
    same object, avoiding repeated model loading.
    """
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = EmbeddingPipeline(
            model_name=model_name, use_cache=use_cache
        )
    return _pipeline_singleton
