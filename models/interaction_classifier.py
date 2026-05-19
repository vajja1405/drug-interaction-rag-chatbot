"""
models/interaction_classifier.py
──────────────────────────────────
Four-tier severity classifier for drug-drug interactions.

Tier 1 – Structured source (highest priority / highest confidence)
  If the retrieved document's metadata contains a severity value from a
  trusted source (e.g., "rxnorm", "curated"), return it directly with
  confidence 0.95.

Tier 2 – Rule-based keyword classifier
  Scans the interaction text for clinical signal words associated with
  each severity level. Returns the highest-signal category with a
  confidence score proportional to keyword matches.

Tier 3 – ML-based classifier (RandomForestSeverityClassifier)
  Invoked when Tiers 1-2 yield Unknown or low confidence (< 0.65).
  Uses TF-IDF + drug-class features from severity_classifier.py.
  Bridges label mapping: Mild (ML) ↔ Low (this module).

Tier 4 – "Unknown" fallback
  Returned when no tier produces a confident classification.
  The LLM in the chatbot agent will attempt to reason about severity.

Severity levels
───────────────
  Severe   – Contraindicated or potentially life-threatening
  Moderate – Requires clinical monitoring or dose adjustment
  Low      – Minor or theoretical; routine care sufficient
  Unknown  – Insufficient data to classify
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ML classifier is imported lazily to avoid loading sklearn/torch on every
# import of this module.  Set to None until first use.
_ml_clf = None

def _get_ml_classifier():
    """Lazy-load the RandomForestSeverityClassifier singleton."""
    global _ml_clf
    if _ml_clf is None:
        try:
            from models.severity_classifier import get_trained_classifier
            _ml_clf = get_trained_classifier()
        except Exception as exc:
            logger.warning("ML classifier unavailable: %s", exc)
    return _ml_clf


# ── Severity definitions ──────────────────────────────────────────────────────

SEVERITY_LEVELS = ["Severe", "Moderate", "Low", "Unknown"]

# Trusted severity sources from the data pipeline
TRUSTED_SOURCES = {"rxnorm", "curated", "fda", "drugbank"}


# ── Keyword sets ──────────────────────────────────────────────────────────────
# Each set maps to a severity level.  Words/phrases are matched
# case-insensitively against the full interaction text.

SEVERE_KEYWORDS: set[str] = {
    "contraindicated",
    "do not use",
    "do not co-administer",
    "avoid",
    "life-threatening",
    "fatal",
    "serious",
    "severe",
    "rhabdomyolysis",
    "serotonin syndrome",
    "anaphylaxis",
    "torsades de pointes",
    "qt prolongation",
    "hemorrhage",
    "haemorrhage",
    "bleeding risk",
    "myopathy",
    "myelosuppression",
    "agranulocytosis",
    "respiratory depression",
    "coma",
    "arrhythmia",
    "toxicity",
    "overdose",
    "lactic acidosis",
    "intracranial",
    "significantly increased risk",
}

MODERATE_KEYWORDS: set[str] = {
    "caution",
    "monitor closely",
    "reduce dose",
    "dose reduction",
    "adjust dose",
    "clinical significance",
    "significant interaction",
    "may increase",
    "may decrease",
    "elevated risk",
    "impaired",
    "inhibits",
    "induces",
    "potentiates",
    "competitive inhibition",
    "protein-binding displacement",
    "renal clearance",
    "hepatic clearance",
    "cyp inhibitor",
    "p-glycoprotein",
    "narrow therapeutic",
    "close monitoring",
}

LOW_KEYWORDS: set[str] = {
    "minor",
    "minimal",
    "unlikely",
    "limited clinical",
    "theoretical",
    "no significant",
    "not clinically significant",
    "routine monitoring",
    "low risk",
    "generally safe",
    "well tolerated",
    "no dose adjustment",
}


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    severity: str          # "Severe" | "Moderate" | "Low" | "Unknown"
    confidence: float      # 0.0 – 1.0
    method: str            # "structured" | "keyword" | "default"
    matched_keywords: list[str]  # keywords that triggered the classification


# ── Classifier ────────────────────────────────────────────────────────────────

class InteractionClassifier:
    """
    Stateless severity classifier.  Call classify(result_dict) with a
    single retrieval result dict produced by DrugInteractionRetriever.
    """

    def classify(self, retrieval_result: dict) -> ClassificationResult:
        """
        Classify the severity of a single retrieved interaction document.

        Parameters
        ----------
        retrieval_result : dict
            A result dict from the retriever:
            { 'text': str, 'metadata': dict, 'similarity_score': float }

        Returns
        -------
        ClassificationResult
        """
        metadata = retrieval_result.get("metadata", {})
        text = retrieval_result.get("text", "")

        # Tier 1 – structured severity from metadata
        structured = self._from_structured(metadata)
        if structured is not None:
            return structured

        # Tier 2 – keyword-based classification on document text
        keyword_result = self._from_keywords(text)
        if keyword_result.severity != "Unknown" and keyword_result.confidence >= 0.65:
            return keyword_result

        # Tier 3 – ML classifier fallback (RandomForest + TF-IDF)
        drug_a = metadata.get("drug_a", "")
        drug_b = metadata.get("drug_b", "")
        if drug_a and drug_b and text:
            ml_result = self._from_ml(drug_a, drug_b, text)
            if ml_result is not None:
                # Prefer ML over a low-confidence keyword result
                if keyword_result.severity == "Unknown" or ml_result.confidence > keyword_result.confidence:
                    return ml_result
                return keyword_result

        # Tier 4 – return keyword result even if low-confidence, else Unknown
        if keyword_result.severity != "Unknown":
            return keyword_result

        return ClassificationResult(
            severity="Unknown",
            confidence=0.0,
            method="default",
            matched_keywords=[],
        )

    def classify_best(
        self, retrieval_results: list[dict]
    ) -> ClassificationResult:
        """
        Classify severity using the best (highest-scoring) document from
        a list of retrieval results for a drug pair.

        Uses the structured severity of the top result if available,
        otherwise aggregates keyword signals across all results and
        returns the most frequently/confidently signalled level.
        """
        if not retrieval_results:
            return ClassificationResult(
                severity="Unknown",
                confidence=0.0,
                method="default",
                matched_keywords=[],
            )

        # Try structured first on the highest-similarity document
        top = retrieval_results[0]
        structured = self._from_structured(top.get("metadata", {}))
        if structured is not None:
            return structured

        # Aggregate keyword signals across all retrieved docs
        combined_text = " ".join(r.get("text", "") for r in retrieval_results)
        return self._from_keywords(combined_text)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _from_ml(drug_a: str, drug_b: str, text: str) -> ClassificationResult | None:
        """
        Tier 3: ML-based classification using the RandomForestSeverityClassifier.

        Bridges the label mapping between the two modules:
          ML label "Mild" → InteractionClassifier severity "Low"
        """
        clf = _get_ml_classifier()
        if clf is None:
            return None
        try:
            pred = clf.predict(drug_a, drug_b, text)
            # Map "Mild" → "Low" for consistency with this module
            severity = "Low" if pred.label == "Mild" else pred.label
            return ClassificationResult(
                severity=severity,
                confidence=pred.confidence,
                method="ml_random_forest",
                matched_keywords=[],
            )
        except Exception as exc:
            logger.warning("ML classifier predict failed: %s", exc)
            return None

    @staticmethod
    def _from_structured(metadata: dict) -> ClassificationResult | None:
        """
        Return a ClassificationResult if the metadata contains a severity
        value from a trusted source, otherwise return None.
        """
        severity = metadata.get("severity", "").strip()
        source = metadata.get("severity_source", "").lower()

        if severity in SEVERITY_LEVELS and source in TRUSTED_SOURCES:
            return ClassificationResult(
                severity=severity,
                confidence=0.95,
                method="structured",
                matched_keywords=[],
            )
        return None

    @staticmethod
    def _from_keywords(text: str) -> ClassificationResult:
        """
        Scan `text` for severity keywords and return the most prominent level.

        Confidence is calculated as:
          base_confidence + 0.05 * min(keyword_count, 5)
        clipped to [0.0, 0.95].
        """
        text_lower = text.lower()

        severe_hits = [kw for kw in SEVERE_KEYWORDS if kw in text_lower]
        moderate_hits = [kw for kw in MODERATE_KEYWORDS if kw in text_lower]
        low_hits = [kw for kw in LOW_KEYWORDS if kw in text_lower]

        if severe_hits:
            conf = min(0.95, 0.70 + 0.05 * len(severe_hits))
            return ClassificationResult(
                severity="Severe",
                confidence=round(conf, 2),
                method="keyword",
                matched_keywords=severe_hits,
            )
        if moderate_hits:
            conf = min(0.90, 0.60 + 0.05 * len(moderate_hits))
            return ClassificationResult(
                severity="Moderate",
                confidence=round(conf, 2),
                method="keyword",
                matched_keywords=moderate_hits,
            )
        if low_hits:
            conf = min(0.85, 0.60 + 0.05 * len(low_hits))
            return ClassificationResult(
                severity="Low",
                confidence=round(conf, 2),
                method="keyword",
                matched_keywords=low_hits,
            )

        return ClassificationResult(
            severity="Unknown",
            confidence=0.0,
            method="keyword",
            matched_keywords=[],
        )


def get_overall_risk(severities: list[str]) -> str:
    """
    Return the highest severity across a list of pair severities.
    Useful for computing an overall polypharmacy risk level.
    """
    priority = {"Severe": 3, "Moderate": 2, "Low": 1, "Unknown": 0}
    if not severities:
        return "Unknown"
    return max(severities, key=lambda s: priority.get(s, 0))
