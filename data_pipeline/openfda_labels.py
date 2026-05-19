"""
data_pipeline/openfda_labels.py
───────────────────────────────
Ingests drug interaction and warning text from the OpenFDA Drug Label API.

Extracts the 'drug_interactions' and 'warnings' sections from the label,
identifies the primary drug from the 'openfda' metadata, and attempts to
extract interacting drug pairs by matching against a known drug dictionary.

Outputs to data/raw/openfda_raw.jsonl in the canonical ingestion format.

Usage:
  python -m data_pipeline.openfda_labels --limit 100
"""

import aiohttp
import asyncio
import argparse
import json
import logging
import re
from pathlib import Path

from config import settings
from data_pipeline.preprocess_data import normalise_drug_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logger = logging.getLogger(__name__)

# Basic dictionary to extract pairs from unstructured FDA text.
KNOWN_DRUGS = {
    "warfarin", "aspirin", "ibuprofen", "metformin", "lisinopril",
    "clopidogrel", "digoxin", "amiodarone", "omeprazole", "esomeprazole",
    "acetaminophen", "paracetamol", "tramadol", "simvastatin", 
    "atorvastatin", "rosuvastatin", "naproxen", "celecoxib", "diclofenac",
}

async def fetch_openfda_labels(limit: int) -> list[dict]:
    """Fetch label data containing 'drug_interactions' from OpenFDA."""
    
    # We query labels that specifically have the drug_interactions section
    # and require them to have openfda.generic_name populated.
    url = f"{settings.openfda_base_url}?search=_exists_:drug_interactions+AND+_exists_:openfda.generic_name&limit={limit}"
    
    logger.info("Fetching OpenFDA labels (limit=%d)...", limit)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error("OpenFDA API error: %d", response.status)
                return []
            
            data = await response.json()
            results = data.get("results", [])
            logger.info("Retrieved %d labels from OpenFDA.", len(results))
            return results

def extract_pairs_from_text(primary_drug: str, text: str) -> list[tuple[str, str]]:
    """Scan text for known drugs to form drug-drug pairs with the primary drug."""
    text_lower = text.lower()
    pairs = []
    
    # Tokenize simply
    tokens = set(re.findall(r"[a-z]+", text_lower))
    
    for drug in KNOWN_DRUGS:
        if drug != primary_drug and drug in tokens:
            pairs.append((primary_drug, drug))
            
    return pairs

def process_fda_record(record: dict) -> list[dict]:
    """Convert an OpenFDA label record into our canonical document schema."""
    openfda = record.get("openfda", {})
    generic_names = openfda.get("generic_name", [])
    if not generic_names:
        return []
        
    # Take the first generic name as primary
    primary_drug = normalise_drug_name(generic_names[0])
    
    interactions = record.get("drug_interactions", [])
    warnings = record.get("warnings", [])
    
    # Combine texts
    raw_texts = interactions + warnings
    if not raw_texts:
        return []
        
    combined_text = " ".join(raw_texts)
    
    # Attempt to extract known drug pairs
    pairs = extract_pairs_from_text(primary_drug, combined_text)
    
    canonical_docs = []
    
    if pairs:
        # Create a document for each identified pair
        for drug_a, drug_b in pairs:
            doc = {
                "drug_a": drug_a,
                "drug_b": drug_b,
                "rxcui_a": None,
                "rxcui_b": None,
                "severity": "Unknown",  # Will be classified by pipeline
                "severity_source": "fda",
                "mechanism": "FDA Label Interaction Section",
                "clinical_effects": "See full label text.",
                "management": "Consult prescribing information.",
                "source": "openfda",
                "raw_text": f"OpenFDA Label for {primary_drug.title()}.\n\n{combined_text}",
            }
            canonical_docs.append(doc)
    else:
        # Fallback: create a single-drug document centered on the primary drug
        # This allows semantic search to find it even if we didn't extract a specific pair
        doc = {
            "drug_a": primary_drug,
            "drug_b": "general_interaction",
            "rxcui_a": None,
            "rxcui_b": None,
            "severity": "Unknown",
            "severity_source": "fda",
            "mechanism": "FDA Label Interaction Section",
            "clinical_effects": "See full label text.",
            "management": "Consult prescribing information.",
            "source": "openfda",
            "raw_text": f"OpenFDA Label for {primary_drug.title()}.\n\n{combined_text}",
        }
        canonical_docs.append(doc)
        
    return canonical_docs

async def run(limit: int, output_file: str):
    labels = await fetch_openfda_labels(limit)
    
    all_docs = []
    for label in labels:
        docs = process_fda_record(label)
        all_docs.extend(docs)
        
    if not all_docs:
        logger.warning("No interactions extracted from OpenFDA.")
        return
        
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc) + "\n")
            
    logger.info("Saved %d extracted documents to %s", len(all_docs), output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OpenFDA labels")
    parser.add_argument("--limit", type=int, default=100, help="Number of labels to fetch")
    parser.add_argument(
        "--out", 
        type=str, 
        default="data/raw/openfda_raw.jsonl", 
        help="Output JSONL file path"
    )
    args = parser.parse_args()
    
    asyncio.run(run(args.limit, args.out))
