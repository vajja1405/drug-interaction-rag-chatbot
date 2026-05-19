"""
chatbot/interaction_agent.py
─────────────────────────────
Drug interaction reasoning agent.

Pipeline (6 explicit stages)
──────────────────────────────
  Stage 1 – Combination generation
    Produce all C(N,2) drug pair combinations from the input list.

  Stage 2 – RAG retrieval
    Query the FAISS vector store for each pair (exact lookup → semantic
    fallback).  Returns a RetrievedEvidence object per pair.

  Stage 3 – Dual severity classification
    For every pair run:
      a) InteractionClassifier  (Tier 1-4: structured → keyword → ML → unknown)
      b) RandomForestSeverityClassifier  (direct ML path)
    The two predictions are reconciled into a single SeverityAssessment.

  Stage 4 – LLM structured explanation
    A JSON-mode prompt instructs the LLM to return a machine-readable
    breakdown per pair: mechanism / clinical_effects / management /
    monitoring_points.  The agent parses this into PairExplanation objects.

  Stage 5 – Overall risk + monitoring synthesis
    Compute the highest severity across all pairs and aggregate the
    monitoring points into a deduplicated list.

  Stage 6 – Response assembly
    Package everything into a typed InteractionReport.

Key design choices
───────────────────
• Typed output  – all intermediate and final objects are dataclasses /
                  Pydantic-compatible; no raw dicts leak out.
• Dual classifier – the ML RandomForest prediction is always captured
                    alongside the rule-based result; both are visible in
                    the report for auditability.
• Reasoning trace – every decision is appended to a ReasoningTrace list
                    so callers can inspect exactly what the agent did.
• JSON LLM output – the primary LLM prompt asks for structured JSON;
                    a fallback plain-text prompt is tried if JSON parsing
                    fails, so the agent never silently drops results.
• Streaming       – analyze_stream() yields markdown chunks in real time.
• Chat history    – DrugInteractionAgent is stateful; conversation turns
                    accumulate and can be cleared with reset_history().
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import settings
from models.interaction_classifier import (
    ClassificationResult,
    InteractionClassifier,
    get_overall_risk,
)
from models.severity_classifier import (
    RandomForestSeverityClassifier,
    SeverityPrediction,
    get_trained_classifier,
)
from rag_pipeline.retriever import DrugInteractionRetriever

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "⚠️  This analysis is for informational purposes only. "
    "Consult a qualified healthcare provider before making any clinical decisions."
)

# ══════════════════════════════════════════════════════════════════════════════
# Typed intermediate and output models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DrugCombination:
    """
    One pair generated during Stage 1.
    pair_key is always alphabetically ordered so downstream lookups are
    deterministic regardless of which drug was listed first.
    """
    drug_a: str
    drug_b: str
    pair_key: str = field(init=False)

    def __post_init__(self) -> None:
        a, b = sorted([self.drug_a, self.drug_b])
        self.pair_key = f"{a}+{b}"

    def __str__(self) -> str:
        return f"{self.drug_a.title()} + {self.drug_b.title()}"


@dataclass
class RetrievedEvidence:
    """Output of Stage 2 for a single drug pair."""
    combination: DrugCombination
    documents: list[dict]           # raw dicts from the retriever
    top_score: float                # highest similarity_score, or 1.0 for exact
    retrieval_method: str           # "exact_match" | "semantic" | "no_results"
    document_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.document_count = len(self.documents)

    @property
    def top_text(self) -> str:
        if self.documents:
            return self.documents[0].get("text", "")
        return ""

    @property
    def top_metadata(self) -> dict:
        if self.documents:
            return self.documents[0].get("metadata", {})
        return {}


@dataclass
class SeverityAssessment:
    """
    Output of Stage 3 — dual classification result.

    Both the rule/structured path (via InteractionClassifier) and the
    ML RandomForest path are always computed and stored here for
    full auditability.
    """
    combination: DrugCombination
    # Rule-based / structured classifier result
    rule_based: ClassificationResult
    # Direct ML classifier result (may be None if model unavailable)
    ml_prediction: SeverityPrediction | None
    # Final reconciled label (see _reconcile below)
    final_severity: str
    final_confidence: float
    final_method: str
    evidence_score: float           # retrieval similarity score

    @staticmethod
    def reconcile(
        rule: ClassificationResult,
        ml: SeverityPrediction | None,
        evidence_score: float,
    ) -> tuple[str, float, str]:
        """
        Merge rule-based and ML results into a single verdict.

        Priority:
          1. Structured metadata (rule.method == "structured")  → trust fully
          2. Both models agree                                   → boost confidence
          3. Rule-based confidence ≥ 0.80                       → use rule-based
          4. ML confidence ≥ 0.75                               → use ML
          5. Majority vote between the two                      → lower confidence
          6. Rule-based fallback
        """
        if rule.method == "structured":
            return rule.severity, rule.confidence, "structured"

        if ml is None:
            return rule.severity, rule.confidence, rule.method

        ml_label = "Low" if ml.label == "Mild" else ml.label

        # Both agree → boost confidence slightly
        if rule.severity == ml_label:
            boosted = min(0.97, (rule.confidence + ml.confidence) / 2 + 0.05)
            return rule.severity, round(boosted, 4), f"consensus ({rule.method}+ml)"

        # Rule-based is high confidence
        if rule.confidence >= 0.80:
            return rule.severity, rule.confidence, rule.method

        # ML is high confidence
        if ml.confidence >= 0.75:
            return ml_label, ml.confidence, "ml_random_forest"

        # Majority vote (here 2 classifiers so first wins on tie)
        method = f"rule_based_over_ml ({rule.method})"
        return rule.severity, rule.confidence, method


@dataclass
class PairExplanation:
    """
    Output of Stage 4 — structured clinical explanation for one drug pair.
    Extracted from the LLM JSON response.
    """
    combination: DrugCombination
    severity: str
    confidence: float
    mechanism: str
    clinical_effects: str
    management: str
    monitoring_points: list[str]
    interaction_type: str           # "pharmacokinetic" | "pharmacodynamic" | "mixed" | "unknown"
    evidence_quality: str           # "high" | "moderate" | "low"
    source_text: str                # snippet from retrieved document
    # Citation fields
    source_name: str = ""           # e.g. "RxNorm", "OpenFDA", "Curated"
    source_url: str = ""            # deeplink to the source
    source_page: str = ""           # description of the source page


@dataclass
class ReasoningTrace:
    """
    Human-readable log of every decision the agent made.
    Attached to InteractionReport for transparency and debugging.
    """
    steps: list[str] = field(default_factory=list)

    def add(self, step: str) -> None:
        self.steps.append(step)
        logger.debug("[trace] %s", step)

    def summary(self) -> str:
        return "\n".join(f"  [{i+1:02d}] {s}" for i, s in enumerate(self.steps))


@dataclass
class InteractionReport:
    """
    Final structured output of DrugInteractionAgent.analyze().

    All fields are typed and serialisable to JSON via to_dict().
    """
    # Inputs
    drugs_analyzed: list[str]
    combinations: list[DrugCombination]

    # Stage outputs
    evidences: dict[str, RetrievedEvidence]          # pair_key → evidence
    assessments: dict[str, SeverityAssessment]       # pair_key → assessment
    explanations: list[PairExplanation]               # one per pair

    # Synthesis
    overall_risk: str
    monitoring_priorities: list[str]
    narrative: str                                    # LLM summary paragraph
    reasoning_trace: ReasoningTrace

    # Meta
    processing_time_ms: float
    retrieved_docs_total: int
    disclaimer: str = DISCLAIMER

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def severe_pairs(self) -> list[PairExplanation]:
        return [e for e in self.explanations if e.severity == "Severe"]

    @property
    def moderate_pairs(self) -> list[PairExplanation]:
        return [e for e in self.explanations if e.severity == "Moderate"]

    @property
    def low_pairs(self) -> list[PairExplanation]:
        return [e for e in self.explanations if e.severity in ("Low", "Mild")]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON output / API response."""
        return {
            "drugs_analyzed": self.drugs_analyzed,
            "pairs_checked": [c.pair_key for c in self.combinations],
            "overall_risk": self.overall_risk,
            "interactions": [
                {
                    "drug_pair": [e.combination.drug_a, e.combination.drug_b],
                    "severity": e.severity,
                    "confidence": e.confidence,
                    "interaction_type": e.interaction_type,
                    "mechanism": e.mechanism,
                    "clinical_effects": e.clinical_effects,
                    "management": e.management,
                    "monitoring_points": e.monitoring_points,
                    "evidence_quality": e.evidence_quality,
                    "source_name": e.source_name,
                    "source_url": e.source_url,
                    "source_page": e.source_page,
                    "classification_method": self.assessments[
                        e.combination.pair_key
                    ].final_method,
                    "ml_prediction": (
                        {
                            "label": self.assessments[e.combination.pair_key]
                            .ml_prediction.label,
                            "confidence": self.assessments[e.combination.pair_key]
                            .ml_prediction.confidence,
                            "probabilities": self.assessments[
                                e.combination.pair_key
                            ].ml_prediction.probabilities,
                        }
                        if self.assessments[e.combination.pair_key].ml_prediction
                        else None
                    ),
                }
                for e in self.explanations
            ],
            "monitoring_priorities": self.monitoring_priorities,
            "narrative": self.narrative,
            "retrieved_docs_total": self.retrieved_docs_total,
            "processing_time_ms": self.processing_time_ms,
            "reasoning_trace": self.reasoning_trace.steps,
            "disclaimer": self.disclaimer,
        }

    def print_report(self) -> None:
        """Pretty-print the full report to stdout."""
        border = "═" * 70
        print(f"\n{border}")
        print(f"  Drug Interaction Analysis Report")
        print(border)
        print(f"  Drugs  : {', '.join(self.drugs_analyzed)}")
        print(f"  Pairs  : {len(self.combinations)}")
        print(f"  Risk   : {self.overall_risk}")
        print(f"  Time   : {self.processing_time_ms:.0f} ms")
        print(border)

        # Interaction table
        print("\n── Pair-Level Results ──")
        print(f"  {'Pair':<35} {'Severity':<10} {'Conf':>6}  {'Method'}")
        print("  " + "─" * 65)
        for expl in sorted(
            self.explanations,
            key=lambda e: {"Severe": 0, "Moderate": 1, "Low": 2, "Unknown": 3}.get(
                e.severity, 4
            ),
        ):
            a = self.assessments[expl.combination.pair_key]
            pair_str = str(expl.combination)
            print(
                f"  {pair_str:<35} {expl.severity:<10} "
                f"{expl.confidence:>5.1%}  {a.final_method}"
            )

        # Per-pair detail
        print("\n── Clinical Details ──")
        for expl in self.explanations:
            print(f"\n  ▸ {expl.combination}  [{expl.severity}]")
            print(f"    Mechanism       : {expl.mechanism}")
            print(f"    Clinical effects: {expl.clinical_effects}")
            print(f"    Management      : {expl.management}")
            if expl.monitoring_points:
                for pt in expl.monitoring_points:
                    print(f"    Monitor         : {pt}")

            # ML prediction alongside rule-based
            assess = self.assessments[expl.combination.pair_key]
            if assess.ml_prediction:
                ml = assess.ml_prediction
                print(
                    f"    ML prediction   : {ml.label} "
                    f"({ml.confidence:.1%})  "
                    f"[Mild {ml.probabilities.get('Mild',0):.0%} | "
                    f"Mod {ml.probabilities.get('Moderate',0):.0%} | "
                    f"Sev {ml.probabilities.get('Severe',0):.0%}]"
                )

        # Monitoring priorities
        if self.monitoring_priorities:
            print("\n── Monitoring Priorities ──")
            for pt in self.monitoring_priorities:
                print(f"  • {pt}")

        # Narrative
        if self.narrative:
            print(f"\n── Clinical Narrative ──\n{self.narrative}")

        print(f"\n{self.disclaimer}")
        print(border + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# LLM Prompt Templates
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a clinical pharmacist specialist in drug-drug interactions (DDI).
You provide precise, evidence-based analysis drawn strictly from the retrieved \
evidence provided in the prompt.

Output rules:
- Respond ONLY with valid JSON — no markdown fences, no prose outside the JSON.
- Write complete sentences for every field. Never truncate mid-word or mid-sentence.
- If no evidence is available for a pair, output "No data available" for text fields \
  and an empty list for monitoring_points.
- Severity must be one of: "Severe", "Moderate", "Low", "Unknown".
- interaction_type must be one of: "pharmacokinetic", "pharmacodynamic", \
  "mixed", "unknown".
- evidence_quality must be one of: "high", "moderate", "low".
- mechanism: 1-3 sentences explaining the pharmacological mechanism.
- clinical_effects: 1-2 sentences on patient-facing risks.
- management: 1-3 sentences with specific, actionable clinical recommendations.
- monitoring_points: list of 1-4 specific parameters to monitor.
"""

_JSON_PROMPT = """\
Patient medication list: {drug_list}

For each drug pair below, use ONLY the evidence provided to fill in the \
JSON fields.  Do not invent information. Write complete, detailed sentences.

=== RETRIEVED EVIDENCE ===
{evidence_block}

=== SEVERITY PRE-ASSESSMENT ===
{severity_block}

Respond with this exact JSON schema (one entry per pair):

{{
  "pairs": [
    {{
      "drug_a": "string",
      "drug_b": "string",
      "mechanism": "string — 1-3 sentences on the pharmacological mechanism",
      "clinical_effects": "string — 1-2 sentences on patient-facing risks",
      "management": "string — 1-3 sentences with specific clinical recommendations",
      "monitoring_points": ["string", ...],
      "interaction_type": "pharmacokinetic|pharmacodynamic|mixed|unknown",
      "evidence_quality": "high|moderate|low",
      "source_document": "string — name of the source (e.g. RxNorm, OpenFDA, Curated)"
    }}
  ],
  "overall_summary": "string — 2-3 sentence regimen risk summary",
  "monitoring_priorities": ["string", ...]
}}

Example output for one pair:
{{
  "pairs": [
    {{
      "drug_a": "warfarin",
      "drug_b": "aspirin",
      "mechanism": "Aspirin inhibits COX-1-mediated platelet aggregation and displaces warfarin from plasma protein binding sites, raising free warfarin concentration.",
      "clinical_effects": "Significantly increased risk of serious bleeding, including gastrointestinal and intracranial haemorrhage.",
      "management": "Avoid concurrent use if possible. If medically necessary, monitor INR weekly and use the lowest effective aspirin dose (≤100 mg/day).",
      "monitoring_points": ["INR", "Signs of bleeding", "Hemoglobin"],
      "interaction_type": "pharmacodynamic",
      "evidence_quality": "high",
      "source_document": "Curated"
    }}
  ],
  "overall_summary": "Concurrent warfarin and aspirin significantly increases bleeding risk. Close INR monitoring is essential.",
  "monitoring_priorities": ["INR", "Signs of bleeding"]
}}
"""

# Plain-text fallback used when JSON parsing fails
_FALLBACK_PROMPT = """\
Patient medication list: {drug_list}

Analyse all drug-drug interactions using the evidence below.
For each pair, provide the following in clearly labelled sections:
- **Mechanism**: 1-3 sentences on the pharmacological mechanism.
- **Clinical Effects**: 1-2 sentences on patient-facing risks.
- **Management**: 1-3 sentences with specific recommendations.
- **Monitoring**: List specific parameters to monitor.

End with an overall risk summary and key monitoring priorities.
Write complete sentences throughout — never truncate mid-word.

=== RETRIEVED EVIDENCE ===
{evidence_block}

=== SEVERITY PRE-ASSESSMENT ===
{severity_block}

{disclaimer}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════

class DrugInteractionAgent:
    """
    Full-pipeline drug interaction reasoning agent.

    Usage
    ─────
    ::

        agent = DrugInteractionAgent(retriever, classifier)

        # Structured report
        report = agent.analyze(["warfarin", "aspirin", "metformin"])
        report.print_report()
        data = report.to_dict()

        # Streaming (yields markdown tokens)
        for chunk in agent.analyze_stream(["warfarin", "ibuprofen"]):
            print(chunk, end="", flush=True)

        # Multi-turn chat
        reply = agent.chat("Is it safe to add ibuprofen to my warfarin regimen?")
        reply2 = agent.chat("What about naproxen instead?")
        agent.reset_history()
    """

    def __init__(
        self,
        retriever: DrugInteractionRetriever,
        classifier: InteractionClassifier | None = None,
        ml_classifier: RandomForestSeverityClassifier | None = None,
    ) -> None:
        self.retriever = retriever
        self.classifier = classifier or InteractionClassifier()

        # ML classifier — load from disk if available, train on first use otherwise
        if ml_classifier is not None:
            self._ml_clf = ml_classifier
        else:
            try:
                self._ml_clf = get_trained_classifier()
            except Exception as exc:
                logger.warning("ML classifier unavailable: %s", exc)
                self._ml_clf = None

        # LLM chain setup
        self._llm = ChatOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        self._json_chain = (
            ChatPromptTemplate.from_messages([
                ("system", _SYSTEM_PROMPT),
                ("human", _JSON_PROMPT),
            ])
            | self._llm
            | StrOutputParser()
        )
        self._fallback_chain = (
            ChatPromptTemplate.from_messages([
                ("system", _SYSTEM_PROMPT.replace("ONLY with valid JSON", "in clear clinical prose")),
                ("human", _FALLBACK_PROMPT),
            ])
            | self._llm
            | StrOutputParser()
        )

        # Multi-turn chat history
        self._history: list[dict[str, str]] = []

    # ── Public interface ───────────────────────────────────────────────────────

    def analyze(self, drugs: list[str]) -> InteractionReport:
        """
        Run the full 6-stage pipeline and return a typed InteractionReport.

        Parameters
        ----------
        drugs : list[str]
            2–20 drug names (generic names preferred).

        Returns
        -------
        InteractionReport — fully typed, serialisable via .to_dict()
        """
        t_start = time.perf_counter()
        trace = ReasoningTrace()

        # ── Stage 1: Combination generation ───────────────────────────────────
        combos = self._stage1_combinations(drugs, trace)
        if not combos:
            return self._empty_report(drugs, trace, t_start)

        # ── Stage 2: RAG retrieval ─────────────────────────────────────────────
        evidences = self._stage2_retrieval(combos, trace)

        # ── Stage 3: Dual classification ───────────────────────────────────────
        assessments = self._stage3_classification(combos, evidences, trace)

        # ── Stage 4: LLM structured explanation ───────────────────────────────
        explanations, narrative = self._stage4_explanation(
            drugs, combos, evidences, assessments, trace
        )

        # ── Stage 5: Overall risk + monitoring ────────────────────────────────
        overall_risk, monitoring = self._stage5_synthesis(assessments, explanations, trace)

        # ── Stage 6: Assemble report ───────────────────────────────────────────
        elapsed = (time.perf_counter() - t_start) * 1000
        trace.add(f"Report assembled in {elapsed:.0f} ms")

        return InteractionReport(
            drugs_analyzed=drugs,
            combinations=combos,
            evidences=evidences,
            assessments=assessments,
            explanations=explanations,
            overall_risk=overall_risk,
            monitoring_priorities=monitoring,
            narrative=narrative,
            reasoning_trace=trace,
            processing_time_ms=round(elapsed, 1),
            retrieved_docs_total=sum(e.document_count for e in evidences.values()),
        )

    def analyze_stream(
        self,
        drugs: list[str],
        include_trace: bool = False,
    ) -> Iterator[str]:
        """
        Stream the clinical explanation as markdown chunks.

        Yields string tokens as they arrive from the LLM.  The final token
        is always the disclaimer.

        Parameters
        ----------
        drugs         : Drug names to analyse.
        include_trace : If True, prepend the reasoning trace before streaming.
        """
        trace = ReasoningTrace()
        combos = self._stage1_combinations(drugs, trace)
        if not combos:
            yield "No drug pairs could be generated from the provided drug list.\n"
            return

        evidences = self._stage2_retrieval(combos, trace)
        assessments = self._stage3_classification(combos, evidences, trace)

        evidence_block = self._build_evidence_block(combos, evidences)
        severity_block = self._build_severity_block(assessments)
        drug_list_str = ", ".join(drugs)

        if include_trace:
            yield "### Reasoning Trace\n"
            yield trace.summary() + "\n\n"

        yield f"### Drug Interaction Analysis: {drug_list_str}\n\n"

        prompt_inputs = {
            "drug_list": drug_list_str,
            "evidence_block": evidence_block,
            "severity_block": severity_block,
            "disclaimer": DISCLAIMER,
        }
        try:
            for chunk in self._fallback_chain.stream(prompt_inputs):
                yield chunk
        except Exception as exc:
            yield f"\n[LLM stream error: {exc}]\n"

        yield f"\n\n{DISCLAIMER}\n"

    def chat(self, message: str) -> str:
        """
        Multi-turn conversational interface.

        Maintains conversation history so follow-up questions like
        "What about naproxen instead?" are contextualised.

        Parameters
        ----------
        message : str  – User message (may reference a drug list or ask
                         follow-up questions about a previous analysis).

        Returns
        -------
        str  – LLM response with clinical context from history.
        """
        # Build messages including history
        messages: list[tuple[str, str]] = [("system", _SYSTEM_PROMPT)]
        for turn in self._history:
            messages.append((turn["role"], turn["content"]))
        messages.append(("human", message))

        chat_prompt = ChatPromptTemplate.from_messages(messages)
        chain = chat_prompt | self._llm | StrOutputParser()

        try:
            response = chain.invoke({})
        except Exception as exc:
            response = f"LLM unavailable: {exc}"

        # Persist history
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": response})
        return response

    def reset_history(self) -> None:
        """Clear the multi-turn conversation history."""
        self._history.clear()
        logger.info("Chat history cleared.")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — Combination generation
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _stage1_combinations(
        drugs: list[str], trace: ReasoningTrace
    ) -> list[DrugCombination]:
        """
        Generate all C(N,2) drug pair combinations from the input list.

        Normalises names to lowercase, strips whitespace, and deduplicates
        before generating pairs.
        """
        # Normalise and deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for d in drugs:
            norm = d.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(norm)

        if len(unique) < 2:
            trace.add(
                f"Stage 1 FAILED: need ≥ 2 distinct drugs, got {unique}"
            )
            return []

        combos = [
            DrugCombination(drug_a=a, drug_b=b)
            for a, b in combinations(sorted(unique), 2)
        ]
        n = len(unique)
        trace.add(
            f"Stage 1 — {n} drugs → {len(combos)} pairs: "
            + ", ".join(c.pair_key for c in combos)
        )
        return combos

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 — RAG retrieval
    # ══════════════════════════════════════════════════════════════════════════

    def _stage2_retrieval(
        self,
        combos: list[DrugCombination],
        trace: ReasoningTrace,
    ) -> dict[str, RetrievedEvidence]:
        """
        Query the RAG system for each drug pair.

        Uses the retriever's two-stage strategy:
          1. Exact pair lookup in the pair_index (O(1))
          2. Semantic FAISS search if no exact match exists
        """
        evidences: dict[str, RetrievedEvidence] = {}
        hits, misses = 0, 0

        for combo in combos:
            docs = self.retriever.retrieve_for_pair(
                combo.drug_a, combo.drug_b
            )

            if not docs:
                method = "no_results"
                top_score = 0.0
                misses += 1
            else:
                top_score = docs[0].get("similarity_score", 0.0)
                method = "exact_match" if top_score == 1.0 else "semantic"
                hits += 1

            evidences[combo.pair_key] = RetrievedEvidence(
                combination=combo,
                documents=docs,
                top_score=top_score,
                retrieval_method=method,
            )
            trace.add(
                f"Stage 2 — {combo}: {len(docs)} doc(s) "
                f"[{method}, score={top_score:.3f}]"
            )

        trace.add(
            f"Stage 2 complete — {hits}/{len(combos)} pairs have evidence "
            f"({misses} with no data)"
        )
        return evidences

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3 — Dual severity classification
    # ══════════════════════════════════════════════════════════════════════════

    def _stage3_classification(
        self,
        combos: list[DrugCombination],
        evidences: dict[str, RetrievedEvidence],
        trace: ReasoningTrace,
    ) -> dict[str, SeverityAssessment]:
        """
        Run both classifiers for every pair and reconcile.

        Rule-based path: InteractionClassifier (structured → keyword → ML-tier3 → unknown)
        Direct ML path:  RandomForestSeverityClassifier.predict()
        """
        assessments: dict[str, SeverityAssessment] = {}

        for combo in combos:
            ev = evidences[combo.pair_key]

            # ── Rule-based / 4-tier classifier ────────────────────────────────
            rule_result: ClassificationResult = self.classifier.classify_best(
                ev.documents
            )

            # ── Direct ML classifier (RandomForest) ────────────────────────────
            ml_pred: SeverityPrediction | None = None
            if self._ml_clf is not None and ev.top_text:
                try:
                    ml_pred = self._ml_clf.predict(
                        combo.drug_a, combo.drug_b, ev.top_text
                    )
                except Exception as exc:
                    logger.warning(
                        "ML classifier failed for %s: %s", combo.pair_key, exc
                    )

            # ── Reconcile ─────────────────────────────────────────────────────
            final_sev, final_conf, final_method = SeverityAssessment.reconcile(
                rule_result, ml_pred, ev.top_score
            )

            assessments[combo.pair_key] = SeverityAssessment(
                combination=combo,
                rule_based=rule_result,
                ml_prediction=ml_pred,
                final_severity=final_sev,
                final_confidence=final_conf,
                final_method=final_method,
                evidence_score=ev.top_score,
            )

            ml_str = (
                f"ML={ml_pred.label}({ml_pred.confidence:.0%})"
                if ml_pred else "ML=N/A"
            )
            trace.add(
                f"Stage 3 — {combo}: "
                f"rule={rule_result.severity}({rule_result.confidence:.0%},"
                f"{rule_result.method}) | {ml_str} "
                f"→ final={final_sev}({final_conf:.0%}) [{final_method}]"
            )

        return assessments

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4 — LLM structured explanation
    # ══════════════════════════════════════════════════════════════════════════

    def _stage4_explanation(
        self,
        drugs: list[str],
        combos: list[DrugCombination],
        evidences: dict[str, RetrievedEvidence],
        assessments: dict[str, SeverityAssessment],
        trace: ReasoningTrace,
    ) -> tuple[list[PairExplanation], str]:
        """
        Call the LLM with a JSON-mode prompt and parse the response.

        Falls back to plain-text mode and extracts fields heuristically
        if JSON parsing fails.
        """
        evidence_block = self._build_evidence_block(combos, evidences)
        severity_block = self._build_severity_block(assessments)
        drug_list_str = ", ".join(drugs)

        prompt_inputs = {
            "drug_list": drug_list_str,
            "evidence_block": evidence_block,
            "severity_block": severity_block,
        }

        trace.add(
            f"Stage 4 — calling LLM ({settings.llm_model}, "
            f"temp={settings.llm_temperature}) with {len(combos)} pairs"
        )

        llm_raw = ""
        parsed_json: dict | None = None

        # ── Try JSON mode first ────────────────────────────────────────────────
        try:
            llm_raw = self._json_chain.invoke(prompt_inputs)
            parsed_json = _parse_json_response(llm_raw)
            trace.add("Stage 4 — JSON response parsed successfully")
        except Exception as exc:
            trace.add(f"Stage 4 — JSON call/parse failed ({exc}); trying fallback")
            parsed_json = None

        # ── Plain-text fallback ────────────────────────────────────────────────
        if parsed_json is None:
            try:
                llm_raw = self._fallback_chain.invoke(
                    {**prompt_inputs, "disclaimer": DISCLAIMER}
                )
                trace.add("Stage 4 — fallback plain-text response received")
            except Exception as exc2:
                llm_raw = (
                    f"LLM unavailable: {exc2}. "
                    "Severity classifications are based on structured/ML data."
                )
                trace.add(f"Stage 4 — LLM completely unavailable: {exc2}")

        # ── Build PairExplanation objects ──────────────────────────────────────
        explanations: list[PairExplanation] = []
        narrative = ""

        if parsed_json:
            narrative = parsed_json.get("overall_summary", "")
            pair_data = {
                f"{p.get('drug_a', '').lower()}+{p.get('drug_b', '').lower()}": p
                for p in parsed_json.get("pairs", [])
            }
            # Also try reversed pair key (LLM may swap order)
            pair_data.update({
                f"{p.get('drug_b', '').lower()}+{p.get('drug_a', '').lower()}": p
                for p in parsed_json.get("pairs", [])
            })

            for combo in combos:
                assess = assessments[combo.pair_key]
                ev = evidences[combo.pair_key]
                # Try both key orderings
                pd = pair_data.get(combo.pair_key) or pair_data.get(
                    f"{combo.drug_b}+{combo.drug_a}", {}
                )
                explanations.append(PairExplanation(
                    combination=combo,
                    severity=assess.final_severity,
                    confidence=assess.final_confidence,
                    mechanism=pd.get("mechanism", ev.top_text if ev.top_text else "See retrieved evidence."),
                    clinical_effects=pd.get("clinical_effects", ""),
                    management=pd.get("management", "Consult healthcare provider."),
                    monitoring_points=pd.get("monitoring_points", []),
                    interaction_type=pd.get("interaction_type", "unknown"),
                    evidence_quality=_score_to_quality(ev.top_score),
                    source_text=ev.top_text,
                    source_name=ev.top_metadata.get("source", "").title() or "Unknown",
                    source_url=ev.top_metadata.get("source_url", ""),
                    source_page=ev.top_metadata.get("source_page", ""),
                ))
        else:
            # Heuristic extraction from plain-text LLM response
            narrative = llm_raw
            for combo in combos:
                assess = assessments[combo.pair_key]
                ev = evidences[combo.pair_key]
                doc_text = ev.top_text
                explanations.append(PairExplanation(
                    combination=combo,
                    severity=assess.final_severity,
                    confidence=assess.final_confidence,
                    mechanism=_extract_field(doc_text, "Mechanism") or "",
                    clinical_effects=_extract_field(doc_text, "Clinical Effects") or "",
                    management=_extract_field(doc_text, "Management") or "Consult healthcare provider.",
                    monitoring_points=[],
                    interaction_type="unknown",
                    evidence_quality=_score_to_quality(ev.top_score),
                    source_text=doc_text,
                    source_name=ev.top_metadata.get("source", "").title() or "Unknown",
                    source_url=ev.top_metadata.get("source_url", ""),
                    source_page=ev.top_metadata.get("source_page", ""),
                ))

        trace.add(
            f"Stage 4 — {len(explanations)} pair explanation(s) assembled"
        )
        return explanations, narrative

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 5 — Overall risk + monitoring synthesis
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _stage5_synthesis(
        assessments: dict[str, SeverityAssessment],
        explanations: list[PairExplanation],
        trace: ReasoningTrace,
    ) -> tuple[str, list[str]]:
        """
        Compute overall regimen risk and deduplicated monitoring list.

        Overall risk = highest single-pair severity (Severe > Moderate > Low).
        Monitoring list = all monitoring_points across pairs, deduplicated
        and sorted by severity of the pair they came from.
        """
        severities = [a.final_severity for a in assessments.values()]
        overall = get_overall_risk(severities)

        # Aggregate monitoring points; deduplicate case-insensitively
        seen_pts: set[str] = set()
        ordered_pts: list[str] = []
        priority = {"Severe": 0, "Moderate": 1, "Low": 2, "Unknown": 3}
        sorted_expls = sorted(
            explanations,
            key=lambda e: priority.get(e.severity, 3),
        )
        for expl in sorted_expls:
            for pt in expl.monitoring_points:
                key = pt.lower().strip()
                if key and key not in seen_pts:
                    seen_pts.add(key)
                    ordered_pts.append(pt)

        trace.add(
            f"Stage 5 — overall_risk={overall}, "
            f"{len(ordered_pts)} unique monitoring point(s)"
        )
        return overall, ordered_pts

    # ══════════════════════════════════════════════════════════════════════════
    # Prompt context builders
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_evidence_block(
        combos: list[DrugCombination],
        evidences: dict[str, RetrievedEvidence],
    ) -> str:
        """Format retrieved documents into the LLM evidence block."""
        sections: list[str] = []
        for combo in combos:
            ev = evidences[combo.pair_key]
            header = (
                f"[{combo}]  "
                f"source={ev.top_metadata.get('source','?')}  "
                f"score={ev.top_score:.3f}"
            )
            if not ev.top_text:
                sections.append(f"{header}\nNo evidence in knowledge base.\n")
            else:
                sections.append(f"{header}\n{ev.top_text.strip()[:4000]}\n")
        return "\n---\n".join(sections)

    @staticmethod
    def _build_severity_block(
        assessments: dict[str, SeverityAssessment],
    ) -> str:
        """Compact severity table for the LLM prompt."""
        lines = []
        for a in assessments.values():
            ml_str = (
                f"  ML={a.ml_prediction.label}({a.ml_prediction.confidence:.0%})"
                if a.ml_prediction else ""
            )
            lines.append(
                f"  {a.combination}: {a.final_severity} "
                f"({a.final_confidence:.0%}, {a.final_method}){ml_str}"
            )
        return "\n".join(lines) if lines else "No classifications."

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _empty_report(
        drugs: list[str],
        trace: ReasoningTrace,
        t_start: float,
    ) -> InteractionReport:
        elapsed = (time.perf_counter() - t_start) * 1000
        return InteractionReport(
            drugs_analyzed=drugs,
            combinations=[],
            evidences={},
            assessments={},
            explanations=[],
            overall_risk="Unknown",
            monitoring_priorities=[],
            narrative="Could not generate pairs from the drug list provided.",
            reasoning_trace=trace,
            processing_time_ms=round(elapsed, 1),
            retrieved_docs_total=0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module-level utilities
# ══════════════════════════════════════════════════════════════════════════════

def _parse_json_response(raw: str) -> dict | None:
    """
    Extract and parse the first valid JSON object from an LLM response.

    Handles common LLM quirks:
      - JSON wrapped in markdown code fences
      - Leading/trailing prose before or after the JSON object
      - Truncated JSON caused by max_token limits
    """
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    def _try_parse(s: str) -> dict | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    # Step 1: Direct parse
    parsed = _try_parse(raw)
    if parsed:
        return parsed

    # Step 2: Extract {...}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        extracted = match.group()
        parsed = _try_parse(extracted)
        if parsed:
            return parsed
            
        # Step 3: Heuristic healing for truncated JSON
        # If it's cut off, usually it's missing ] or } at the end.
        heal_attempts = [
            extracted + '"}]}',
            extracted + '"]}',
            extracted + '}]}',
            extracted + ']}',
            extracted + '}',
            extracted + '"}',
            extracted + '""}]}',
        ]
        for attempt in heal_attempts:
            parsed = _try_parse(attempt)
            if parsed:
                return parsed

    return None


def _extract_field(text: str, field_name: str) -> str | None:
    """Pull a named field value from document text (e.g., 'Mechanism: ...')."""
    for line in text.splitlines():
        if line.lower().startswith(field_name.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return None


def _score_to_quality(score: float) -> str:
    """Convert a retrieval similarity score to a human-readable quality label."""
    if score >= 0.90:
        return "high"
    if score >= 0.70:
        return "moderate"
    return "low"
