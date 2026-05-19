"""
rag_pipeline/rag_engine.py
───────────────────────────
End-to-end RAG engine: natural language query → clinical explanation.

This is the main user-facing interface of the RAG pipeline.  It connects:

  EmbeddingPipeline
       ↓
  DrugVectorStore (FAISS)
       ↓
  DrugInteractionRetriever
       ↓
  InteractionClassifier
       ↓
  LangChain LCEL chain (ChatOpenAI)
       ↓
  RAGResponse (structured Pydantic model)

Two entry points
─────────────────
  engine.query(question)          – accepts a natural language question
  engine.analyze_drugs(drug_list) – accepts an explicit list of drug names

Example (natural language)
──────────────────────────
  engine = RAGEngine.from_disk()
  result = engine.query("Can warfarin and ibuprofen be taken together?")
  print(result.answer)

Example (drug list)
───────────────────
  result = engine.analyze_drugs(["warfarin", "aspirin", "metformin"])
  for interaction in result.interactions:
      print(interaction)

Design notes
─────────────
• LLM temperature = 0.1  →  deterministic clinical prose
• Severity comes from structured data (classifier), never the LLM
• The LLM only generates the natural-language explanation
• Supports streaming via engine.stream_query() using LangChain streaming
• Optional chat history for multi-turn conversations
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import settings
from models.interaction_classifier import (
    ClassificationResult,
    InteractionClassifier,
    get_overall_risk,
)
from rag_pipeline.embeddings import get_pipeline
from rag_pipeline.retriever import DrugInteractionRetriever, RetrievalResult
from rag_pipeline.vector_store import DrugVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are a clinical pharmacist assistant specialised in drug-drug interactions.
You provide evidence-based, medically grounded analysis drawn from FDA-approved drug labelling
and peer-reviewed pharmacology literature.

Rules:
- Rely only on the retrieved evidence provided in the prompt.
- State the interaction MECHANISM (pharmacokinetic or pharmacodynamic).
- State the clinical EFFECTS and associated risks.
- Provide a specific MANAGEMENT recommendation.
- State the severity classification: Severe / Moderate / Low / Unknown.
- If no evidence was retrieved for a pair, say so explicitly.
- Keep the tone precise and professional.
- Conclude with a disclaimer that this is for informational purposes only."""

_QA_TEMPLATE = """\
The user has asked: "{question}"

Identified drug pair(s): {drug_list}

=== RETRIEVED EVIDENCE FROM KNOWLEDGE BASE ===
{retrieved_context}

=== SEVERITY CLASSIFICATIONS ===
{severity_table}

Based solely on the evidence above, answer the user's question.
For each drug pair, explain:
  1. Mechanism of interaction
  2. Clinical effects / risks
  3. Management recommendation
  4. Severity: {overall_risk}

End your answer with this exact line:
⚠️  Disclaimer: This analysis is for informational purposes only. Consult a qualified healthcare provider before making any clinical decisions.
"""

_MULTI_DRUG_TEMPLATE = """\
A patient is prescribed: {drug_list}

Analyse all drug-drug interactions in this regimen.

=== RETRIEVED EVIDENCE ===
{retrieved_context}

=== SEVERITY CLASSIFICATIONS ===
{severity_table}

For each drug pair:
  1. Mechanism
  2. Clinical effects and risks
  3. Management recommendation
  4. Severity classification

Overall regimen risk level: {overall_risk}

End with:
⚠️  Disclaimer: This analysis is for informational purposes only. Consult a qualified healthcare provider before making any clinical decisions.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Response models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairSummary:
    drug_pair: tuple[str, str]
    severity: str
    confidence: float
    method: str
    description: str

    def __str__(self) -> str:
        a, b = self.drug_pair
        return (
            f"{a.title()} + {b.title()}: {self.severity} "
            f"(confidence {self.confidence:.0%}, {self.method})\n"
            f"  {self.description}"
        )


@dataclass
class RAGResponse:
    """
    Structured response from the RAG engine.

    Attributes
    ----------
    question      : Original user question (empty for structured mode)
    drugs         : Drug names identified/provided
    interactions  : Per-pair severity summaries
    overall_risk  : Highest severity across all pairs
    answer        : LLM-generated clinical explanation
    sources       : Number of retrieved documents used
    latency_ms    : End-to-end latency
    """
    question: str
    drugs: list[str]
    interactions: list[PairSummary]
    overall_risk: str
    answer: str
    sources: int
    latency_ms: float

    def print_report(self) -> None:
        """Pretty-print the full analysis report to stdout."""
        border = "═" * 64
        print(f"\n{border}")
        print(f"  Drug Interaction Analysis Report")
        print(border)
        if self.question:
            print(f"  Query  : {self.question}")
        print(f"  Drugs  : {', '.join(self.drugs)}")
        print(f"  Risk   : {self.overall_risk}")
        print(f"  Pairs  : {len(self.interactions)}")
        print(f"  Sources: {self.sources} retrieved document(s)")
        print(f"  Time   : {self.latency_ms:.0f} ms")
        print(border)

        print("\n── Per-Pair Severity Summary ──")
        for pair in self.interactions:
            print(f"  {pair}")

        print("\n── Clinical Explanation (LLM) ──")
        print(self.answer)
        print(border + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Engine
# ═══════════════════════════════════════════════════════════════════════════════

class RAGEngine:
    """
    End-to-end RAG engine for drug interaction analysis.

    Instantiate via RAGEngine.from_disk() (recommended) or the constructor.

    Parameters
    ----------
    vector_store : DrugVectorStore
        Pre-loaded FAISS index.
    retriever    : DrugInteractionRetriever
        Configured retriever (uses the same vector_store).
    classifier   : InteractionClassifier
        Severity classifier.
    chat_history : list[dict], optional
        Multi-turn conversation history injected into the LLM context.
        Each entry: {"role": "user"|"assistant", "content": str}
    """

    def __init__(
        self,
        vector_store: DrugVectorStore,
        retriever: DrugInteractionRetriever,
        classifier: InteractionClassifier | None = None,
        chat_history: list[dict] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.retriever = retriever
        self.classifier = classifier or InteractionClassifier()
        self.chat_history: list[dict] = chat_history or []

        # Build LangChain chain
        self._llm = ChatOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        self._qa_prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _QA_TEMPLATE),
        ])
        self._multi_prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _MULTI_DRUG_TEMPLATE),
        ])
        self._chain_qa = self._qa_prompt | self._llm | StrOutputParser()
        self._chain_multi = self._multi_prompt | self._llm | StrOutputParser()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_disk(
        cls,
        vector_store_path: str | None = None,
        model_name: str | None = None,
    ) -> "RAGEngine":
        """
        Convenience factory: load vector store from disk and wire everything.

        Requires that `python build_index.py` has been run first.
        """
        logger.info("RAGEngine: loading vector store …")
        store = DrugVectorStore.load(vector_store_path)
        pipeline = get_pipeline(model_name)
        retriever = DrugInteractionRetriever(
            vector_store=store,
            embedding_model=pipeline,
        )
        return cls(vector_store=store, retriever=retriever)

    # ── Natural language query ─────────────────────────────────────────────────

    def query(
        self,
        question: str,
        use_mmr: bool = False,
        top_k: int | None = None,
    ) -> RAGResponse:
        """
        Answer a free-text drug interaction question end-to-end.

        Example
        -------
        ::

            result = engine.query(
                "Can warfarin and ibuprofen be taken together?"
            )
            result.print_report()

        Parameters
        ----------
        question : str
            Any natural language question about drug interactions.
        use_mmr  : bool
            Use Maximal Marginal Relevance to reduce redundant results.
        top_k    : int, optional
            Documents to retrieve per pair.
        """
        t_start = time.perf_counter()
        logger.info("RAGEngine.query: %r", question)

        # Step 1 – retrieve
        retrieval: RetrievalResult = self.retriever.retrieve_for_query(
            question, top_k=top_k, use_mmr=use_mmr
        )

        # Step 2 – classify
        classified = self._classify_all(retrieval.pair_results)
        overall_risk = get_overall_risk([r.severity for r in classified.values()])

        # Step 3 – build context
        retrieved_context = _format_context(retrieval.pair_results)
        severity_table = _format_severity_table(classified)

        # Step 4 – LLM call
        answer = self._call_llm(
            chain=self._chain_qa,
            inputs={
                "question": question,
                "drug_list": ", ".join(retrieval.drugs_found) or "unknown",
                "retrieved_context": retrieved_context,
                "severity_table": severity_table,
                "overall_risk": overall_risk,
            },
        )

        # Step 5 – build response
        interactions = _build_pair_summaries(retrieval.pair_results, classified)
        latency = (time.perf_counter() - t_start) * 1000

        response = RAGResponse(
            question=question,
            drugs=retrieval.drugs_found,
            interactions=interactions,
            overall_risk=overall_risk,
            answer=answer,
            sources=retrieval.total_docs,
            latency_ms=round(latency, 1),
        )

        # Store in chat history for multi-turn use
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return response

    # ── Structured drug list ───────────────────────────────────────────────────

    def analyze_drugs(
        self,
        drugs: list[str],
        use_mmr: bool = False,
        top_k: int | None = None,
    ) -> RAGResponse:
        """
        Analyse drug-drug interactions for an explicit list of drugs.

        Example
        -------
        ::

            result = engine.analyze_drugs(["warfarin", "aspirin", "metformin"])
            result.print_report()
        """
        t_start = time.perf_counter()
        logger.info("RAGEngine.analyze_drugs: %s", drugs)

        pair_results = self.retriever.retrieve_for_drug_list(
            drugs, top_k=top_k, use_mmr=use_mmr
        )
        classified = self._classify_all(pair_results)
        overall_risk = get_overall_risk([r.severity for r in classified.values()])

        retrieved_context = _format_context(pair_results)
        severity_table = _format_severity_table(classified)

        answer = self._call_llm(
            chain=self._chain_multi,
            inputs={
                "drug_list": ", ".join(drugs),
                "retrieved_context": retrieved_context,
                "severity_table": severity_table,
                "overall_risk": overall_risk,
            },
        )

        interactions = _build_pair_summaries(pair_results, classified)
        latency = (time.perf_counter() - t_start) * 1000

        return RAGResponse(
            question="",
            drugs=drugs,
            interactions=interactions,
            overall_risk=overall_risk,
            answer=answer,
            sources=sum(len(v) for v in pair_results.values()),
            latency_ms=round(latency, 1),
        )

    # ── Streaming variant ──────────────────────────────────────────────────────

    def stream_query(self, question: str) -> Iterator[str]:
        """
        Stream the LLM answer token-by-token for the given question.

        Usage::

            for token in engine.stream_query("Can warfarin and ibuprofen be taken together?"):
                print(token, end="", flush=True)

        Yields
        ------
        str : Individual text chunks/tokens as they are generated.
        """
        retrieval = self.retriever.retrieve_for_query(question)
        classified = self._classify_all(retrieval.pair_results)
        overall_risk = get_overall_risk([r.severity for r in classified.values()])

        prompt_inputs = {
            "question": question,
            "drug_list": ", ".join(retrieval.drugs_found) or "unknown",
            "retrieved_context": _format_context(retrieval.pair_results),
            "severity_table": _format_severity_table(classified),
            "overall_risk": overall_risk,
        }
        # LangChain streaming: use .stream() on the chain
        for chunk in self._chain_qa.stream(prompt_inputs):
            yield chunk

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify_all(
        self, pair_results: dict[str, list[dict]]
    ) -> dict[str, ClassificationResult]:
        classified: dict[str, ClassificationResult] = {}
        for pair_key, docs in pair_results.items():
            classified[pair_key] = self.classifier.classify_best(docs)
        return classified

    def _call_llm(self, chain, inputs: dict) -> str:
        try:
            return chain.invoke(inputs)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return (
                f"LLM explanation unavailable: {exc}\n"
                "Severity classifications above are based on structured data."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Context formatters (module-level helpers)
# ═══════════════════════════════════════════════════════════════════════════════

def _format_context(pair_results: dict[str, list[dict]]) -> str:
    """
    Build the retrieved-evidence block injected into the LLM prompt.

    Each section shows the top retrieved document for one drug pair,
    along with its similarity score and source severity.
    """
    if not any(pair_results.values()):
        return "No interaction evidence found in the knowledge base."

    sections: list[str] = []
    for pair_key, docs in pair_results.items():
        if pair_key == "global":
            header = "[GLOBAL SEARCH RESULTS]"
        else:
            a, b = pair_key.split("+", 1)
            header = f"[{a.title()} + {b.title()}]"

        if not docs:
            sections.append(f"{header}\nNo evidence retrieved.\n")
            continue

        top = docs[0]
        meta = top.get("metadata", {})
        sections.append(
            f"{header} | severity={meta.get('severity','?')} "
            f"| score={top.get('similarity_score', 0):.3f} "
            f"| source={meta.get('source','?')}\n"
            f"{top['text'].strip()}\n"
        )

    return "\n---\n".join(sections)


def _format_severity_table(
    classified: dict[str, ClassificationResult]
) -> str:
    """Compact one-line-per-pair severity summary for the prompt."""
    if not classified:
        return "No pairs classified."
    lines = []
    for pair_key, res in classified.items():
        if pair_key == "global":
            continue
        a, b = pair_key.split("+", 1)
        lines.append(
            f"  {a.title()} + {b.title():30s} → {res.severity:8s} "
            f"(conf {res.confidence:.0%}, {res.method})"
        )
    return "\n".join(lines)


def _build_pair_summaries(
    pair_results: dict[str, list[dict]],
    classified: dict[str, ClassificationResult],
) -> list[PairSummary]:
    summaries: list[PairSummary] = []
    for pair_key, res in classified.items():
        if pair_key == "global":
            continue
        a, b = pair_key.split("+", 1)
        docs = pair_results.get(pair_key, [])
        # Extract Clinical Effects line or fall back to first 200 chars
        if docs:
            raw = docs[0].get("text", "")
            desc = _extract_field(raw, "Clinical Effects") or raw[:200]
        else:
            desc = "No interaction data found in knowledge base."
        summaries.append(PairSummary(
            drug_pair=(a, b),
            severity=res.severity,
            confidence=res.confidence,
            method=res.method,
            description=desc.strip(),
        ))
    return summaries


def _extract_field(text: str, field_name: str) -> str | None:
    for line in text.splitlines():
        if line.lower().startswith(field_name.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone demo
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo() -> None:
    """
    Interactive demo that runs the full RAG pipeline.

    Run with:
        python -m rag_pipeline.rag_engine

    The demo works even WITHOUT an LLM key – it shows the retrieval and
    classification steps and marks the LLM step as unavailable.
    """
    import json

    print("\n" + "═" * 64)
    print("  Drug Interaction RAG Engine – Demo")
    print("═" * 64)

    # ── Load vector store ─────────────────────────────────────────────────────
    try:
        engine = RAGEngine.from_disk()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("Run first:  python build_index.py --all-seed")
        return

    print(f"\nKnowledge base: {engine.vector_store.total_vectors} documents")
    known = engine.vector_store.get_all_known_drugs()
    print(f"Known drugs   : {', '.join(known[:10])} … ({len(known)} total)")

    # ═════════════════════════════════════════════════════════════════════════
    # DEMO QUERY 1 – Natural language question
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 64)
    print("DEMO 1 — Natural Language Query")
    print("─" * 64)

    query = "Can warfarin and ibuprofen be taken together?"
    print(f"\nQuery: {query!r}\n")

    # Show the retrieval step explicitly
    retrieval = engine.retriever.retrieve_for_query(query)
    print(f"  Drugs extracted : {retrieval.drugs_found}")
    print(f"  Pairs checked   : {list(retrieval.pair_results.keys())}")
    print(f"  Retrieval mode  : {retrieval.retrieval_mode}")

    for pair_key, docs in retrieval.pair_results.items():
        if docs:
            top = docs[0]
            print(f"\n  [{pair_key}]")
            print(f"    similarity_score : {top['similarity_score']:.4f}")
            print(f"    severity         : {top['metadata'].get('severity')}")
            print(f"    source           : {top['metadata'].get('source')}")
            print("    text snippet     :", top["text"][:120].replace("\n", " "), "…")
        else:
            print(f"\n  [{pair_key}] → No data found")

    # Full end-to-end response
    print("\n  Calling RAGEngine.query() …")
    result = engine.query(query)
    result.print_report()

    # ═════════════════════════════════════════════════════════════════════════
    # DEMO QUERY 2 – Polypharmacy (explicit drug list)
    # ═════════════════════════════════════════════════════════════════════════
    print("─" * 64)
    print("DEMO 2 — Polypharmacy Drug List")
    print("─" * 64)

    drugs = ["warfarin", "aspirin", "metformin"]
    print(f"\nDrugs: {drugs}\n")

    result2 = engine.analyze_drugs(drugs)
    result2.print_report()

    # ═════════════════════════════════════════════════════════════════════════
    # DEMO QUERY 3 – Streaming tokens
    # ═════════════════════════════════════════════════════════════════════════
    print("─" * 64)
    print("DEMO 3 — Streaming Output")
    print("─" * 64)
    stream_query = "What happens when you take simvastatin with amiodarone?"
    print(f"\nQuery: {stream_query!r}")
    print("\nStreamed answer:\n")
    try:
        for token in engine.stream_query(stream_query):
            print(token, end="", flush=True)
        print("\n")
    except Exception as exc:
        print(f"[Streaming unavailable: {exc}]\n")

    # ═════════════════════════════════════════════════════════════════════════
    # Show JSON output structure
    # ═════════════════════════════════════════════════════════════════════════
    print("─" * 64)
    print("JSON Output Structure (result from DEMO 1)")
    print("─" * 64)
    output = {
        "question": result.question,
        "drugs_found": result.drugs,
        "overall_risk": result.overall_risk,
        "interactions": [
            {
                "drug_pair": list(p.drug_pair),
                "severity": p.severity,
                "confidence": p.confidence,
                "classification_method": p.method,
                "description": p.description,
            }
            for p in result.interactions
        ],
        "sources_retrieved": result.sources,
        "latency_ms": result.latency_ms,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run_demo()
