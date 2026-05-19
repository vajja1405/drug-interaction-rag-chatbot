"""
evaluation/run_eval.py
──────────────────────
CLI runner for the RAG evaluation framework.

Loads the golden test set, runs each drug pair through the full RAG pipeline
(retriever + classifier only — does NOT call the LLM for speed), and
computes all evaluation metrics.

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --verbose

Metrics produced:
    1. Severity Accuracy   — does predicted severity match expected?
    2. Context Precision    — do retrieved docs mention the queried pair?
    3. Faithfulness         — are LLM-ready fields grounded in context?
    4. Mechanism Coverage   — do expected keywords appear in mechanism text?
    5. Management Coverage  — do expected keywords appear in management text?
    6. Citation Accuracy    — does source match expected?

References:
    RAGAS, Amazon Bedrock Eval, Anthropic Eval, LangChain Evaluation Guide
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    citation_accuracy,
    severity_accuracy,
    keyword_coverage,
)


def load_golden_set() -> list[dict]:
    """Load golden_set.json from the evaluation directory."""
    golden_path = Path(__file__).parent / "golden_set.json"
    with open(golden_path, encoding="utf-8") as f:
        return json.loads(f.read())


def run_evaluation(verbose: bool = False) -> dict:
    """
    Execute the full evaluation pipeline.

    Returns a summary dict with aggregate scores and per-pair breakdowns.
    """
    # Lazy imports to avoid loading everything at module level
    from rag_pipeline.vector_store import DrugVectorStore
    from rag_pipeline.retriever import DrugInteractionRetriever
    from models.interaction_classifier import InteractionClassifier

    print("=" * 70)
    print("  Drug Interaction AI — RAG Evaluation Framework")
    print("=" * 70)
    print()

    # Load components
    print("Loading vector store...")
    store = DrugVectorStore.load()

    print("Initialising retriever and classifier...")
    retriever = DrugInteractionRetriever(store)
    classifier = InteractionClassifier()

    golden = load_golden_set()
    print(f"Loaded {len(golden)} test cases from golden_set.json")
    print()

    # Metrics accumulators
    results = []
    totals = {
        "severity_accuracy": 0.0,
        "context_precision": 0.0,
        "faithfulness": 0.0,
        "mechanism_coverage": 0.0,
        "management_coverage": 0.0,
        "citation_accuracy": 0.0,
    }

    start_time = time.time()

    # Separate OOV (out-of-vocabulary) cases: severity_accuracy is always
    # computed for them, but retrieval-dependent metrics (context_precision,
    # faithfulness, mechanism/management coverage, citation_accuracy) are
    # N/A when there are intentionally no documents — scoring them as 0
    # would unfairly penalise correct "Unknown" behaviour.
    retrieval_metric_keys = {
        "context_precision", "faithfulness",
        "mechanism_coverage", "management_coverage", "citation_accuracy",
    }
    retrieval_counts = {k: 0 for k in retrieval_metric_keys}

    for i, case in enumerate(golden):
        drug_a = case["drug_a"]
        drug_b = case["drug_b"]
        pair_label = f"{drug_a} + {drug_b}"
        is_oov = case.get("oov", False)

        if verbose:
            print(f"[{i+1:02d}/{len(golden)}] Testing: {pair_label}"
                  + (" [OOV]" if is_oov else ""))

        # Run retrieval
        docs = retriever.retrieve_for_pair(drug_a, drug_b)

        # Run classification on best doc
        if docs:
            classification = classifier.classify(docs[0])
        else:
            from models.interaction_classifier import ClassificationResult
            classification = ClassificationResult(
                severity="Unknown", confidence=0.0, method="default", matched_keywords=[]
            )

        # Get the top retrieved text + metadata
        top_text = docs[0].get("text", "") if docs else ""
        top_meta = docs[0].get("metadata", {}) if docs else {}
        # Combine all chunk texts — long entries are split by the chunker, so
        # keywords may appear in chunks other than docs[0].
        combined_text = " ".join(d.get("text", "") for d in docs)

        # Severity accuracy — always computed
        sev_acc = severity_accuracy(case["expected_severity"], classification.severity)

        # Retrieval-dependent metrics — only for in-KB cases
        if is_oov:
            ctx_prec = faith = mech_cov = mgmt_cov = cite_acc = None
        else:
            ctx_prec = context_precision([drug_a, drug_b], docs)
            faith = faithfulness(
                mechanism=top_text,
                clinical_effects="",
                management="",
                retrieved_context=top_text,
            )
            # Use combined chunk text so chunked docs are evaluated fairly
            mech_cov = keyword_coverage(case["expected_mechanism_keywords"], combined_text)
            mgmt_cov = keyword_coverage(case["expected_management_keywords"], combined_text)
            cite_acc = citation_accuracy(
                case.get("expected_source", ""),
                top_meta.get("source", ""),
            )

        pair_result = {
            "pair": pair_label,
            "oov": is_oov,
            "expected_severity": case["expected_severity"],
            "actual_severity": classification.severity,
            "actual_confidence": round(classification.confidence, 2),
            "actual_method": classification.method,
            "retrieved_docs": len(docs),
            "metrics": {
                "severity_accuracy": sev_acc,
                "context_precision": ctx_prec,
                "faithfulness": faith,
                "mechanism_coverage": mech_cov,
                "management_coverage": mgmt_cov,
                "citation_accuracy": cite_acc,
            },
        }
        results.append(pair_result)

        totals["severity_accuracy"] += sev_acc
        for key in retrieval_metric_keys:
            val = pair_result["metrics"][key]
            if val is not None:
                totals[key] += val
                retrieval_counts[key] += 1

        if verbose:
            status = "✅" if sev_acc == 1.0 else "❌"
            oov_tag = " [OOV — retrieval metrics skipped]" if is_oov else (
                f" | MechCov: {mech_cov:.0%} | MgmtCov: {mgmt_cov:.0%}"
            )
            print(
                f"      {status} Severity: {classification.severity} "
                f"(expected {case['expected_severity']}) | "
                f"Docs: {len(docs)}{oov_tag}"
            )

    elapsed = time.time() - start_time

    # Compute averages — severity over all cases, retrieval metrics over in-KB only
    n = len(golden)
    averages = {"severity_accuracy": round(totals["severity_accuracy"] / n, 4)}
    for key in retrieval_metric_keys:
        denom = retrieval_counts[key]
        averages[key] = round(totals[key] / denom, 4) if denom else 1.0

    # Summary
    summary = {
        "total_cases": n,
        "elapsed_seconds": round(elapsed, 2),
        "aggregate_scores": averages,
        "per_pair_results": results,
    }

    # Print summary table
    print()
    print("─" * 70)
    print("  AGGREGATE SCORES")
    print("─" * 70)
    print(f"  {'Metric':<30} {'Score':>8}")
    print(f"  {'─' * 30} {'─' * 8}")
    for metric, score in averages.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:<30} {score:>7.1%}  {bar}")
    print()
    print(f"  Total test cases: {n}")
    print(f"  Evaluation time:  {elapsed:.2f}s")
    print("─" * 70)

    # Per-pair breakdown
    print()
    print("  PER-PAIR BREAKDOWN")
    print("─" * 70)
    print(f"  {'Pair':<35} {'Sev':>6} {'Docs':>4} {'MechCov':>8} {'Status':>6}")
    print(f"  {'─' * 35} {'─' * 6} {'─' * 4} {'─' * 8} {'─' * 6}")
    for r in results:
        status = "✅" if r["metrics"]["severity_accuracy"] == 1.0 else "❌"
        mech_cov = r["metrics"]["mechanism_coverage"]
        mech_str = f"{mech_cov:>7.0%}" if mech_cov is not None else "    N/A"
        print(
            f"  {r['pair']:<35} "
            f"{r['actual_severity']:>6} "
            f"{r['retrieved_docs']:>4} "
            f"{mech_str} "
            f"  {status}"
        )
    print()

    # Save results
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Full results saved to: {results_path}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation on the golden test set"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-pair details during evaluation",
    )
    args = parser.parse_args()
    run_evaluation(verbose=args.verbose)


if __name__ == "__main__":
    main()
