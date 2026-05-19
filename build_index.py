"""
build_index.py
───────────────
One-shot script that runs the complete data pipeline and builds the
FAISS vector store.

Steps:
  1. Fetch drug interaction data  (data_pipeline/fetch_drug_data.py)
  2. Preprocess & clean records   (data_pipeline/preprocess_data.py)
  3. Embed documents              (rag_pipeline/embedding_model.py)
  4. Build & save FAISS index     (rag_pipeline/vector_store.py)

Usage:
  python build_index.py                            # uses seed data (offline)
  python build_index.py --drugs warfarin aspirin   # + live RxNorm fetch
  python build_index.py --all-seed                 # seed only, no API calls
  python build_index.py --processed-only           # skip fetch, reuse existing raw data

After running this script, start the API server:
  uvicorn api.server:app --reload
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from data_pipeline.fetch_drug_data import fetch_all_interactions, save_records
from data_pipeline.preprocess_data import process_records, load_jsonl, save_jsonl
from rag_pipeline.embeddings import EmbeddingPipeline
from rag_pipeline.vector_store import DrugVectorStore
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# Default list of drugs to fetch live interaction data for
DEFAULT_DRUGS = [
    "warfarin", "aspirin", "ibuprofen", "metformin",
    "lisinopril", "simvastatin", "amiodarone", "digoxin",
    "clopidogrel", "omeprazole", "methotrexate", "tramadol",
    "atorvastatin", "amlodipine", "losartan", "furosemide",
    "metoprolol", "atenolol", "prednisone", "ciprofloxacin",
]


def main():
    parser = argparse.ArgumentParser(
        description="Build the drug interaction FAISS index"
    )
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=DEFAULT_DRUGS,
        help="Drug names to fetch live interaction data for",
    )
    parser.add_argument(
        "--all-seed",
        action="store_true",
        help="Use only curated seed data (no live API calls)",
    )
    parser.add_argument(
        "--processed-only",
        action="store_true",
        help="Skip data fetching; use existing data/processed/interactions.jsonl",
    )
    parser.add_argument(
        "--raw-file",
        default=settings.raw_data_path + "/interactions_raw.jsonl",
        help="Path to raw JSONL file",
    )
    parser.add_argument(
        "--processed-file",
        default=settings.processed_data_path + "/interactions.jsonl",
        help="Path to processed JSONL file",
    )
    args = parser.parse_args()

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    if not args.processed_only:
        logger.info("=== Step 1: Fetching drug interaction data ===")
        drugs_to_fetch = [] if args.all_seed else args.drugs
        records = asyncio.run(
            fetch_all_interactions(
                drugs=drugs_to_fetch,
                include_seed=True,
            )
        )
        save_records(records, args.raw_file)
        logger.info("Raw data saved: %d records → %s", len(records), args.raw_file)
    else:
        logger.info("Skipping data fetch (--processed-only mode)")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    if not args.processed_only:
        logger.info("=== Step 2: Preprocessing records ===")
        raw_records = load_jsonl(args.raw_file)
        processed = process_records(raw_records)
        save_jsonl(processed, args.processed_file)
        logger.info(
            "Processed data saved: %d documents → %s",
            len(processed),
            args.processed_file,
        )
    else:
        logger.info("=== Step 2: Loading existing processed data ===")
        if not Path(args.processed_file).exists():
            logger.error(
                "Processed file not found: %s\n"
                "Run without --processed-only first.",
                args.processed_file,
            )
            return
        processed = load_jsonl(args.processed_file)
        logger.info(
            "Loaded %d processed documents from %s",
            len(processed),
            args.processed_file,
        )

    if not processed:
        logger.error("No processed documents available. Aborting.")
        return

    # ── Step 3: Embed documents via the new EmbeddingPipeline ────────────────
    logger.info("=== Step 3: Embedding %d documents ===", len(processed))
    pipeline = EmbeddingPipeline(use_cache=True)
    chunks, embeddings, embed_stats = pipeline.run(processed, show_progress=True)
    logger.info(embed_stats.report())

    # When chunk_size > max document length (common for DDI records),
    # each document produces exactly one chunk so len(chunks) == len(processed).
    # Re-map chunks back to the document metadata expected by vector_store.build().
    if len(chunks) != len(processed):
        logger.info(
            "Chunking produced %d chunks from %d documents – "
            "rebuilding document list from chunks",
            len(chunks), len(processed),
        )
        # Each chunk carries its source document metadata; rebuild the doc list
        processed = [
            {"document_text": c.text, "metadata": c.metadata}
            for c in chunks
        ]

    logger.info(
        "Embedding complete: shape=%s, dtype=%s",
        embeddings.shape,
        embeddings.dtype,
    )

    # ── Step 4: Build and save FAISS index ────────────────────────────────────
    logger.info("=== Step 4: Building FAISS index ===")
    vector_store = DrugVectorStore()
    vector_store.build(documents=processed, embeddings=embeddings)
    vector_store.save(settings.vector_store_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    stats = vector_store.stats()
    print("\n" + "=" * 60)
    print("  Drug Interaction Index Build Complete")
    print("=" * 60)
    print(f"  Documents indexed : {stats['total_documents']}")
    print(f"  Unique drug pairs : {stats['unique_pairs']}")
    print(f"  Index type        : {stats['index_type']}")
    print(f"  Vector store      : {settings.vector_store_path}/")
    print(f"  Embedding model   : {settings.embedding_model}")
    print("=" * 60)
    print("\nNext step: start the API server")
    print("  uvicorn api.server:app --reload --port 8000\n")


if __name__ == "__main__":
    main()
