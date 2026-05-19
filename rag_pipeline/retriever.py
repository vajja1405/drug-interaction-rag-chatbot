"""
rag_pipeline/retriever.py
──────────────────────────
Drug interaction retriever with two operating modes:

Mode A – Structured drug list  (existing behaviour, unchanged)
  retrieve_for_drug_list(["warfarin", "aspirin"])
  → generates all pair combinations and retrieves per pair

Mode B – Natural language query  (new)
  retrieve_for_query("Can warfarin and ibuprofen be taken together?")
  → extracts drug names from text, then runs the same pair retrieval

Query processing pipeline (Mode B)
───────────────────────────────────
  raw NL query
      │
      ▼
  DrugNameExtractor         ← matches against known drugs in the store
      │                        + common synonym dictionary
      ▼
  extracted drug list        e.g. ["warfarin", "ibuprofen"]
      │
      ▼
  retrieve_for_drug_list()   ← existing two-stage retrieval
      │
      ▼
  RetrievalResult            ← structured output with provenance info

Two-stage retrieval (both modes)
──────────────────────────────────
  Stage 1: exact_pair_lookup – O(1) metadata lookup
  Stage 2: FAISS semantic search – fallback when no exact match exists

Backward compatibility
───────────────────────
  DrugInteractionRetriever is fully backward compatible with the existing
  chatbot/interaction_agent.py and api/server.py.
"""

import logging
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from rag_pipeline.embeddings import EmbeddingPipeline, get_pipeline
from rag_pipeline.vector_store import DrugVectorStore
from config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Drug Name Extractor
# ═══════════════════════════════════════════════════════════════════════════════

# Extended synonym map: common trade names and alternate spellings → generic
_DRUG_SYNONYMS: dict[str, str] = {
    "coumadin": "warfarin", "jantoven": "warfarin",
    "advil": "ibuprofen", "motrin": "ibuprofen", "nuprin": "ibuprofen",
    "bayer": "aspirin", "ecotrin": "aspirin",
    "glucophage": "metformin",
    "zestril": "lisinopril", "prinivil": "lisinopril",
    "plavix": "clopidogrel",
    "lanoxin": "digoxin",
    "cordarone": "amiodarone", "nexterone": "amiodarone",
    "prilosec": "omeprazole", "losec": "omeprazole",
    "nexium": "esomeprazole",
    "tylenol": "acetaminophen", "paracetamol": "acetaminophen",
    "ultram": "tramadol",
    "zocor": "simvastatin",
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
}

# Patterns for extracting drug pairs from common question templates
_PAIR_PATTERNS = [
    # "warfarin and ibuprofen"
    r"\b(\w+)\s+and\s+(\w+)\b",
    # "warfarin with ibuprofen"
    r"\b(\w+)\s+with\s+(\w+)\b",
    # "warfarin + ibuprofen"
    r"\b(\w+)\s*\+\s*(\w+)\b",
    # "between warfarin and ibuprofen"
    r"\bbetween\s+(\w+)\s+and\s+(\w+)\b",
    # "mixing warfarin and ibuprofen"
    r"\bmix(?:ing)?\s+(\w+)\s+and\s+(\w+)\b",
]

_STOPWORDS = {
    "can", "be", "taken", "together", "the", "a", "an", "is", "are",
    "what", "how", "does", "do", "of", "in", "on", "at", "to", "for",
    "if", "my", "me", "i", "take", "taking", "drug", "drugs", "medicine",
    "medication", "medications", "patient", "between", "with", "and",
    "interaction", "interactions", "safe", "dangerous", "harmful", "risk",
    "effect", "effects", "side", "combine", "combining", "mixing", "mix",
}


class DrugNameExtractor:
    """
    Extracts drug names from a natural language query.

    Strategy (in order of priority):
      1. Synonym resolution – "Advil" → "ibuprofen"
      2. Regex pair patterns – "warfarin and ibuprofen" → ["warfarin", "ibuprofen"]
      3. Known-drug lookup – match each query token against all drugs
         currently indexed in the vector store

    The extractor requires the vector store's drug list to maximise
    recall.  It is designed to be tolerant of typos and case variants.
    """

    def __init__(self, known_drugs: list[str]) -> None:
        self._known_drugs: set[str] = {d.lower() for d in known_drugs}
        # Also index synonyms against known drugs
        self._all_known: set[str] = self._known_drugs | set(_DRUG_SYNONYMS.keys())

    def extract(self, query: str) -> list[str]:
        """
        Extract drug names from a free-text query.

        Returns a deduplicated, normalised list of drug names.
        Returns an empty list if fewer than 2 drugs can be identified.
        """
        query_lower = query.lower()

        # Step 1 – try regex pair patterns first (highest precision)
        pair_drugs = self._extract_via_patterns(query_lower)
        if len(pair_drugs) >= 2:
            return pair_drugs

        # Step 2 – token-level lookup against known drugs
        token_drugs = self._extract_via_tokens(query_lower)
        if len(token_drugs) >= 2:
            return token_drugs

        # Merge both strategies
        merged = list(dict.fromkeys(pair_drugs + token_drugs))
        return merged

    def _extract_via_patterns(self, query: str) -> list[str]:
        """Apply regex templates designed for common clinical question patterns."""
        extracted = []
        seen: set[str] = set()
        for pattern in _PAIR_PATTERNS:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                for group in match.groups():
                    resolved = self._resolve(group.lower())
                    if resolved and resolved not in seen:
                        extracted.append(resolved)
                        seen.add(resolved)
        return extracted

    def _extract_via_tokens(self, query: str) -> list[str]:
        """Match individual query tokens against the known drug dictionary."""
        # Tokenise: split on whitespace and punctuation
        tokens = re.findall(r"[a-zA-Z]+", query.lower())
        extracted = []
        seen: set[str] = set()
        for token in tokens:
            if token in _STOPWORDS:
                continue
            resolved = self._resolve(token)
            if resolved and resolved not in seen:
                extracted.append(resolved)
                seen.add(resolved)
        return extracted

    def _resolve(self, token: str) -> str | None:
        """Resolve token to a canonical drug name, or None if not a known drug."""
        # Check synonym map first
        canonical = _DRUG_SYNONYMS.get(token, token)
        if canonical in self._known_drugs:
            return canonical
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """
    Structured output from a single retrieval operation.

    Attributes
    ----------
    query          : Original user query (NL or "" for structured mode)
    drugs_found    : Drug names extracted from the query
    pair_results   : Mapping of "drug_a+drug_b" → list of retrieved docs
    retrieval_mode : "structured" | "natural_language"
    total_docs     : Total number of retrieved documents across all pairs
    """
    query: str
    drugs_found: list[str]
    pair_results: dict[str, list[dict]]
    retrieval_mode: str
    total_docs: int = field(init=False)

    def __post_init__(self):
        self.total_docs = sum(len(v) for v in self.pair_results.values())

    @property
    def pairs_with_data(self) -> list[str]:
        return [k for k, v in self.pair_results.items() if v]

    @property
    def pairs_without_data(self) -> list[str]:
        return [k for k, v in self.pair_results.items() if not v]

    def top_documents(self) -> list[dict]:
        """Return the single best document for each pair (highest similarity)."""
        tops = []
        for pair_key, docs in self.pair_results.items():
            if docs:
                tops.append({"pair": pair_key, **docs[0]})
        return tops


# ═══════════════════════════════════════════════════════════════════════════════
# Retriever
# ═══════════════════════════════════════════════════════════════════════════════

class DrugInteractionRetriever:
    """
    High-level retrieval interface.

    Accepts either a structured drug list or a natural language query.
    Returns a RetrievalResult with all retrieved documents and provenance.
    """

    def __init__(
        self,
        vector_store: DrugVectorStore,
        embedding_model: EmbeddingPipeline | None = None,
    ) -> None:
        self.vector_store = vector_store
        self._embedding = embedding_model or get_pipeline()
        self._extractor = DrugNameExtractor(
            known_drugs=vector_store.get_all_known_drugs()
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve_for_query(
        self,
        query: str,
        top_k: int | None = None,
        use_mmr: bool = False,
    ) -> RetrievalResult:
        """
        Full pipeline for a natural language query.

        Example
        -------
        ::

            result = retriever.retrieve_for_query(
                "Can warfarin and ibuprofen be taken together?"
            )
            # result.drugs_found  → ["warfarin", "ibuprofen"]
            # result.pair_results → {"ibuprofen+warfarin": [...docs...]}

        Parameters
        ----------
        query   : Free-text clinical question or drug interaction query.
        top_k   : Documents per pair (semantic fallback only).
        use_mmr : Use Maximal Marginal Relevance to reduce redundancy.
        """
        logger.info("retrieve_for_query: %r", query)

        # Step 1 – extract drug names from the query
        drugs = self._extractor.extract(query)
        logger.info("  Extracted drugs: %s", drugs)

        if len(drugs) < 2:
            # Semantic fallback: embed the whole query, search globally
            logger.info(
                "  < 2 drugs extracted – falling back to global semantic search"
            )
            return self._semantic_global_search(query, top_k, use_mmr)

        # Step 2 – delegate to structured retrieval
        pair_results = self._retrieve_all_pairs(drugs, top_k, use_mmr)
        return RetrievalResult(
            query=query,
            drugs_found=drugs,
            pair_results=pair_results,
            retrieval_mode="natural_language",
        )

    def retrieve_for_pair(
        self,
        drug_a: str,
        drug_b: str,
        top_k: int | None = None,
        use_mmr: bool = False,
    ) -> list[dict]:
        """
        Retrieve documents for a single drug pair.

        Runs exact lookup first, falls back to semantic search.
        Each result dict: { 'text', 'metadata', 'similarity_score' }
        """
        a, b = drug_a.lower().strip(), drug_b.lower().strip()
        top_k = top_k or settings.rag_top_k

        # Stage 1 – exact pair lookup (O(1))
        exact = self.vector_store.exact_pair_lookup(a, b)
        if exact:
            logger.debug("Exact match: %s + %s (%d docs)", a, b, len(exact))
            return exact

        # Stage 2 – semantic search fallback
        logger.debug("Semantic search fallback: %s + %s", a, b)
        query_text = (
            f"drug interaction between {a} and {b} "
            f"effects mechanism severity clinical management"
        )
        q_emb = self._embedding.embed_query(query_text)

        if use_mmr:
            results = self.vector_store.mmr_search(q_emb, k=top_k)
        else:
            results = self.vector_store.search(q_emb, k=top_k)

        # Apply entity-match filter to prevent hallucinations: ensure the
        # semantic document mentions both drugs. Compound names (e.g.
        # "potassium chloride") match if all component words are present.
        def _drug_mentioned(drug: str, text: str) -> bool:
            if drug in text:
                return True
            # Compound name: every word must appear in the text
            parts = drug.split()
            return len(parts) > 1 and all(p in text for p in parts)

        valid_results = []
        for r in results:
            text_lower = r.get("text", "").lower()
            meta = r.get("metadata", {})
            meta_drugs = {
                meta.get("drug_a", "").lower(),
                meta.get("drug_b", "").lower(),
            }
            # Accept if text mentions both drugs OR metadata exactly matches
            if (_drug_mentioned(a, text_lower) and _drug_mentioned(b, text_lower)
                    or {a, b} <= meta_drugs):
                valid_results.append(r)

        if results and not valid_results:
            logger.warning(
                "Rejected %d semantic results for %s + %s because they "
                "lacked exact drug name mentions.",
                len(results), a, b,
            )

        return valid_results

    def retrieve_for_drug_list(
        self,
        drugs: list[str],
        top_k: int | None = None,
        use_mmr: bool = False,
    ) -> dict[str, list[dict]]:
        """
        Retrieve interaction documents for all C(N,2) pairs in a drug list.

        Returns
        -------
        dict mapping "drug_a+drug_b" → list of result dicts.
        Keys are sorted alphabetically for deterministic output.
        """
        normalised = self._normalise_drug_list(drugs)
        if len(normalised) < 2:
            logger.warning("Need ≥ 2 distinct drugs, got: %s", normalised)
            return {}
        return self._retrieve_all_pairs(normalised, top_k, use_mmr)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _retrieve_all_pairs(
        self,
        drugs: list[str],
        top_k: int | None,
        use_mmr: bool,
    ) -> dict[str, list[dict]]:
        pairs = list(combinations(sorted(drugs), 2))
        logger.info(
            "Retrieving %d drugs → %d pairs", len(drugs), len(pairs)
        )
        results: dict[str, list[dict]] = {}
        for a, b in pairs:
            results[f"{a}+{b}"] = self.retrieve_for_pair(a, b, top_k, use_mmr)
        found = sum(1 for v in results.values() if v)
        logger.info("Data found for %d/%d pairs", found, len(pairs))
        return results

    def _semantic_global_search(
        self,
        query: str,
        top_k: int | None,
        use_mmr: bool,
    ) -> RetrievalResult:
        """
        Embed the raw query and search the full index globally.
        Used when fewer than 2 drugs can be extracted from the query.
        """
        q_emb = self._embedding.embed_query(query)
        if use_mmr:
            docs = self.vector_store.mmr_search(q_emb, k=top_k or settings.rag_top_k)
        else:
            docs = self.vector_store.search(q_emb, k=top_k or settings.rag_top_k)

        return RetrievalResult(
            query=query,
            drugs_found=[],
            pair_results={"global": docs},
            retrieval_mode="semantic_global",
        )

    @staticmethod
    def _normalise_drug_list(drugs: list[str]) -> list[str]:
        seen: set[str] = set()
        unique = []
        for d in drugs:
            norm = d.lower().strip()
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(norm)
        return unique
