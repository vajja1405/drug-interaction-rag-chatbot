"""
rag_pipeline/embedding_model.py
─────────────────────────────────
Wraps a Sentence Transformers model to produce dense vector embeddings
for both documents (at index-build time) and user queries (at inference time).

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional vectors
  - Good balance of speed and semantic quality
  - normalize_embeddings=True enables cosine similarity via dot product,
    which is compatible with FAISS IndexFlatIP (Inner Product index)

Alternative models that can be swapped in via EMBEDDING_MODEL env var:
  - BAAI/bge-small-en-v1.5         (higher MTEB accuracy, similar size)
  - pritamdeka/S-PubMedBert-MS-MARCO (biomedical domain fine-tuning)
  - multi-qa-MiniLM-L6-cos-v1      (trained on QA pairs – good for retrieval)
"""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer providing document and query
    embedding methods.

    Note: constructing this object downloads the model weights on first use
    (~90 MB for all-MiniLM-L6-v2). Subsequent runs use the local cache.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        logger.info("Loading embedding model: %s", self.model_name)
        # SentenceTransformer automatically uses GPU if available
        self._model = SentenceTransformer(self.model_name)
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded – dimension: %d", self.embedding_dim
        )

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of document strings into a 2-D float32 array
        of shape (len(texts), embedding_dim).

        Parameters
        ----------
        texts : list[str]
            Document strings to embed.
        batch_size : int
            Number of texts to process in one forward pass.
        show_progress : bool
            Whether to display a tqdm progress bar.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # unit vectors → cosine ≡ dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string and return a 1-D float32 array
        of shape (embedding_dim,).
        """
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)

    def embed_pair_query(self, drug_a: str, drug_b: str) -> np.ndarray:
        """
        Convenience method: build a natural-language interaction query
        for a drug pair and embed it.

        The query string is structured to maximise recall of relevant
        interaction documents in the FAISS index.
        """
        query = (
            f"drug interaction between {drug_a} and {drug_b} "
            f"effects mechanism severity clinical management"
        )
        return self.embed_query(query)


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """
    Module-level singleton factory.

    Uses lru_cache so the model is loaded only once per process, regardless
    of how many modules import this function. The FastAPI lifespan hook
    also calls this to pre-warm the model on startup.
    """
    return EmbeddingModel()
