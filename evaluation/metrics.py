"""
evaluation/metrics.py
──────────────────────
RAG evaluation metrics inspired by RAGAS, Amazon, Anthropic, and LangChain
best practices.

Metrics (all scored 0.0 – 1.0):
  1. Faithfulness      — % of LLM claims grounded in retrieved context
  2. Answer Relevancy  — overlap between queried drugs and response fields
  3. Context Precision  — do retrieved docs mention the queried drug pair?
  4. Citation Accuracy   — does the source match expected source?
  5. Response Consistency — output variance across repeated runs

References:
  - RAGAS: https://docs.ragas.io/en/latest/concepts/metrics/
  - Amazon: https://aws.amazon.com/blogs/machine-learning/evaluate-rag-applications-with-amazon-bedrock/
  - Anthropic: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/
  - LangChain: https://python.langchain.com/docs/guides/evaluation/
"""

import re
from difflib import SequenceMatcher


def faithfulness(
    mechanism: str,
    clinical_effects: str,
    management: str,
    retrieved_context: str,
) -> float:
    """
    Measures factual grounding: what fraction of key claims in the LLM
    output appear (even partially) in the retrieved evidence context.

    Approach: extract meaningful phrases (3+ word n-grams) from the LLM
    output and check how many have a fuzzy match in the context.
    Inspired by RAGAS faithfulness metric.
    """
    if not retrieved_context or not retrieved_context.strip():
        return 0.0

    llm_text = f"{mechanism} {clinical_effects} {management}".lower().strip()
    context_lower = retrieved_context.lower()

    if not llm_text.strip():
        return 0.0

    # Extract 3-word sliding windows as "claims"
    words = re.findall(r"\b[a-z]{2,}\b", llm_text)
    if len(words) < 3:
        # For very short outputs, do direct substring check
        return 1.0 if llm_text in context_lower else 0.0

    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0

    supported = 0
    for tri in trigrams:
        # Check exact substring or fuzzy match
        if tri in context_lower:
            supported += 1
        else:
            # Fuzzy: does any 3-word window in context match ≥ 75%?
            ctx_words = re.findall(r"\b[a-z]{2,}\b", context_lower)
            ctx_trigrams = [
                " ".join(ctx_words[i : i + 3]) for i in range(len(ctx_words) - 2)
            ]
            for ct in ctx_trigrams:
                if SequenceMatcher(None, tri, ct).ratio() >= 0.75:
                    supported += 1
                    break

    return round(supported / len(trigrams), 4)


def answer_relevancy(
    queried_drugs: list[str],
    response_mechanism: str,
    response_clinical_effects: str,
    response_management: str,
) -> float:
    """
    Measures how relevant the answer is to the actual query (drug names).
    Checks whether both queried drug names appear in the combined response.
    Inspired by RAGAS Answer Relevancy.
    """
    combined = (
        f"{response_mechanism} {response_clinical_effects} {response_management}"
    ).lower()

    if not combined.strip() or combined.strip() in (
        "no data available",
        "unknown",
        "see retrieved evidence.",
    ):
        # No answer was generated — relevancy is N/A, score 0
        return 0.0

    score = 0.0
    for drug in queried_drugs:
        drug_lower = drug.strip().lower()
        # Check direct mention or drug class mention
        if drug_lower in combined:
            score += 1.0
        elif any(
            alias in combined
            for alias in _drug_class_aliases(drug_lower)
        ):
            score += 0.5

    return round(min(score / max(len(queried_drugs), 1), 1.0), 4)


def context_precision(
    queried_drugs: list[str],
    retrieved_documents: list[dict],
) -> float:
    """
    Measures what fraction of retrieved documents are actually about
    the queried drug pair. Inspired by RAGAS Context Precision.

    A document is considered relevant if it mentions both drug names.
    """
    if not retrieved_documents:
        return 0.0

    drugs_lower = {d.strip().lower() for d in queried_drugs}
    relevant = 0
    for doc in retrieved_documents:
        text = doc.get("text", "").lower()
        meta = doc.get("metadata", {})
        doc_drugs = {
            meta.get("drug_a", "").lower(),
            meta.get("drug_b", "").lower(),
        }
        # Check metadata match or text mention
        if drugs_lower <= doc_drugs or all(d in text for d in drugs_lower):
            relevant += 1

    return round(relevant / len(retrieved_documents), 4)


def citation_accuracy(
    expected_source: str,
    actual_source_name: str,
) -> float:
    """
    Binary metric: does the cited source match the expected source?
    Inspired by Amazon's citation grounding evaluation.
    """
    if not expected_source:
        # If no expected source, any citation is fine
        return 1.0 if actual_source_name else 0.0

    return 1.0 if expected_source.lower() == actual_source_name.lower() else 0.0


def severity_accuracy(
    expected_severity: str,
    actual_severity: str,
) -> float:
    """
    Binary metric: does the predicted severity match the expected?
    This is the most critical clinical correctness metric.
    """
    # Normalise
    expected = expected_severity.strip().title()
    actual = actual_severity.strip().title()
    return 1.0 if expected == actual else 0.0


def keyword_coverage(
    expected_keywords: list[str],
    actual_text: str,
) -> float:
    """
    Measures what fraction of expected keywords appear in the actual text.
    Used for mechanism and management field validation.
    """
    if not expected_keywords:
        return 1.0  # No keywords to check

    text_lower = actual_text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return round(found / len(expected_keywords), 4)


def _drug_class_aliases(drug: str) -> list[str]:
    """Return common drug class aliases for a given drug name."""
    aliases = {
        "warfarin": ["anticoagulant", "coumadin"],
        "aspirin": ["nsaid", "antiplatelet", "salicylate"],
        "ibuprofen": ["nsaid", "anti-inflammatory"],
        "metformin": ["biguanide", "antidiabetic", "glucophage"],
        "simvastatin": ["statin", "hmg-coa"],
        "atorvastatin": ["statin", "hmg-coa", "lipitor"],
        "amiodarone": ["antiarrhythmic", "cordarone"],
        "lisinopril": ["ace inhibitor", "acei"],
        "digoxin": ["cardiac glycoside", "lanoxin"],
        "clopidogrel": ["antiplatelet", "plavix"],
        "omeprazole": ["ppi", "proton pump inhibitor"],
        "fluoxetine": ["ssri", "serotonin"],
        "tramadol": ["opioid", "analgesic"],
        "lithium": ["mood stabiliser", "mood stabilizer"],
        "methotrexate": ["dmard", "antimetabolite", "mtx"],
    }
    return aliases.get(drug, [])
