"""
data_pipeline/preprocess_data.py
──────────────────────────────────
Transforms raw drug interaction JSONL records into clean, embeddable
document objects ready for FAISS indexing.

Processing steps:
  1. Load raw JSONL records from data/raw/
  2. Normalise drug names (lowercase, strip whitespace)
  3. Deduplicate symmetric pairs (A+B == B+A)
  4. Build a rich document string combining all interaction fields
  5. Filter low-quality records missing critical fields
  6. Output processed JSONL to data/processed/

Document format written to processed output:
  {
    "document_text": str,      # full text to embed
    "metadata": {
        "drug_a": str,
        "drug_b": str,
        "severity": str,
        "severity_source": str,
        "source": str,
        "rxcui_a": str | None,
        "rxcui_b": str | None,
    }
  }

Usage:
  python -m data_pipeline.preprocess_data \
      --in  data/raw/interactions_raw.jsonl \
      --out data/processed/interactions.jsonl
"""

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)


# ── Normalisation helpers ─────────────────────────────────────────────────────

# Common drug name synonyms – map trade/alternate names to generic names
SYNONYM_MAP: dict[str, str] = {
    "coumadin": "warfarin",
    "jantoven": "warfarin",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "nuprin": "ibuprofen",
    "bayer": "aspirin",
    "ecotrin": "aspirin",
    "glucophage": "metformin",
    "zestril": "lisinopril",
    "prinivil": "lisinopril",
    "plavix": "clopidogrel",
    "lanoxin": "digoxin",
    "cordarone": "amiodarone",
    "nexium": "esomeprazole",
    "prilosec": "omeprazole",
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "ultram": "tramadol",
    "zocor": "simvastatin",
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
}


def normalise_drug_name(name: str) -> str:
    """Lowercase, strip whitespace, and resolve common synonyms."""
    name = name.strip().lower()
    # Remove dosage information e.g. "aspirin 100mg" → "aspirin"
    name = re.sub(r"\s+\d+\s*(mg|mcg|g|ml|iu).*$", "", name)
    return SYNONYM_MAP.get(name, name)


# ── Document builder ──────────────────────────────────────────────────────────

def build_document_text(record: dict) -> str:
    """
    Construct a single, embedding-friendly text block from a raw record.

    Each document captures the full clinical context of one drug pair so
    that semantic search on phrases like "warfarin bleeding risk" reliably
    surfaces the correct document.
    """
    drug_a = record.get("drug_a", "").title()
    drug_b = record.get("drug_b", "").title()
    severity = record.get("severity", "Unknown")
    mechanism = record.get("mechanism", "").strip()
    clinical_effects = record.get("clinical_effects", "").strip()
    management = record.get("management", "").strip()
    raw_text = record.get("raw_text", "").strip()

    parts = [
        f"Drug Interaction: {drug_a} + {drug_b}",
        f"Severity: {severity}",
    ]
    if mechanism:
        parts.append(f"Mechanism: {mechanism}")
    if clinical_effects and clinical_effects != mechanism:
        parts.append(f"Clinical Effects: {clinical_effects}")
    if management:
        parts.append(f"Management: {management}")
    # Append raw text only if it adds new information
    if raw_text and raw_text not in mechanism and raw_text not in clinical_effects:
        # Truncate to avoid bloating the embedding with redundant content
        parts.append(f"Context: {raw_text[:400]}")

    return "\n".join(parts)


# ── Quality filtering ─────────────────────────────────────────────────────────

def is_valid_record(record: dict) -> bool:
    """
    Return True if the record has sufficient information for useful retrieval.
    Records missing both drug names or any interaction content are discarded.
    """
    has_both_drugs = bool(record.get("drug_a")) and bool(record.get("drug_b"))
    has_content = any([
        record.get("mechanism"),
        record.get("clinical_effects"),
        record.get("raw_text"),
    ])
    return has_both_drugs and has_content


# ── Main processing pipeline ──────────────────────────────────────────────────

def process_records(records: list[dict]) -> list[dict]:
    """
    Full preprocessing pipeline: normalise → deduplicate → filter → build docs.

    Returns a list of processed document dicts ready for embedding.
    """
    processed: list[dict] = []
    seen_pairs: set[frozenset] = set()
    skipped_invalid = 0
    skipped_duplicate = 0

    for raw in records:
        # Step 1 – normalise drug names
        drug_a = normalise_drug_name(raw.get("drug_a", ""))
        drug_b = normalise_drug_name(raw.get("drug_b", ""))

        if not drug_a or not drug_b:
            skipped_invalid += 1
            continue

        # Step 2 – deduplicate symmetric pairs
        pair_key = frozenset([drug_a, drug_b])
        if pair_key in seen_pairs:
            skipped_duplicate += 1
            continue
        seen_pairs.add(pair_key)

        # Step 3 – enrich record with normalised names
        enriched = {**raw, "drug_a": drug_a, "drug_b": drug_b}

        # Step 4 – quality filter
        if not is_valid_record(enriched):
            skipped_invalid += 1
            continue

        # Step 5 – build embeddable document text
        doc_text = build_document_text(enriched)

        # Step 6 – assemble output object
        processed.append({
            "document_text": doc_text,
            "metadata": {
                "drug_a": drug_a,
                "drug_b": drug_b,
                "severity": enriched.get("severity", "Unknown"),
                "severity_source": enriched.get("severity_source", "unknown"),
                "source": enriched.get("source", "unknown"),
                "source_url": enriched.get("source_url", ""),
                "source_page": enriched.get("source_page", ""),
                "rxcui_a": enriched.get("rxcui_a"),
                "rxcui_b": enriched.get("rxcui_b"),
            },
        })

    logger.info(
        "Preprocessing complete: %d accepted, %d duplicates removed, "
        "%d invalid records skipped",
        len(processed),
        skipped_duplicate,
        skipped_invalid,
    )
    return processed


def load_jsonl(path: str) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON line: %s", exc)
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    """Persist records as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    logger.info("Saved %d processed documents to %s", len(records), path)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess drug interaction data")
    parser.add_argument(
        "--in",
        dest="input_files",
        nargs="+",
        default=["data/raw/interactions_raw.jsonl"],
        help="Input raw JSONL files",
    )
    parser.add_argument(
        "--out",
        default="data/processed/interactions.jsonl",
        help="Output processed JSONL file",
    )
    args = parser.parse_args()

    raw_records = []
    for filepath in args.input_files:
        if Path(filepath).exists():
            records = load_jsonl(filepath)
            raw_records.extend(records)
            logger.info("Loaded %d raw records from %s", len(records), filepath)
        else:
            logger.warning("Input file not found: %s", filepath)
            
    if not raw_records:
        logger.error("No input records found.")
        return

    processed = process_records(raw_records)
    save_jsonl(processed, args.out)

    print(f"Preprocessing complete: {len(processed)} documents → {args.out}")


if __name__ == "__main__":
    main()
