"""
search/corpus.py
────────────────
Builds a retrieval corpus from the `drug_interactions` section of FDA drug labels
(openFDA Drug Label API, public, no key required for this volume).

Each label section is split into sentence windows of at most ~120 words. The
passage keeps the label owner (`drug`), the openFDA `set_id` and its text, so
every search hit can be traced back to its source label.

Usage:
    python -m search.corpus            # fetch + write data/benchmark/label_passages.jsonl
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

OPENFDA = "https://api.fda.gov/drug/label.json"
ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "data" / "benchmark"
RAW_DIR = BENCH_DIR / "raw"
PASSAGES = BENCH_DIR / "label_passages.jsonl"

# Frequently prescribed, single-ingredient generics with well-documented interactions.
DRUGS = [
    "warfarin", "apixaban", "rivaroxaban", "dabigatran", "clopidogrel", "prasugrel", "ticagrelor",
    "aspirin", "ibuprofen", "naproxen", "celecoxib", "diclofenac", "meloxicam", "ketorolac",
    "acetaminophen", "tramadol", "oxycodone", "hydrocodone", "morphine", "fentanyl", "methadone",
    "buprenorphine", "gabapentin", "pregabalin", "carbamazepine", "phenytoin", "valproic acid",
    "lamotrigine", "levetiracetam", "topiramate", "lithium", "sertraline", "fluoxetine",
    "paroxetine", "citalopram", "escitalopram", "venlafaxine", "duloxetine", "bupropion",
    "mirtazapine", "trazodone", "amitriptyline", "nortriptyline", "quetiapine", "olanzapine",
    "risperidone", "aripiprazole", "haloperidol", "clozapine", "alprazolam", "lorazepam",
    "diazepam", "clonazepam", "zolpidem", "metformin", "glipizide", "glyburide", "glimepiride",
    "pioglitazone", "sitagliptin", "empagliflozin", "dapagliflozin", "insulin glargine",
    "lisinopril", "enalapril", "ramipril", "losartan", "valsartan", "irbesartan", "amlodipine",
    "diltiazem", "verapamil", "nifedipine", "metoprolol", "atenolol", "carvedilol", "propranolol",
    "hydrochlorothiazide", "furosemide", "spironolactone", "digoxin", "amiodarone", "dronedarone",
    "sotalol", "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin",
    "ezetimibe", "gemfibrozil", "fenofibrate", "omeprazole", "esomeprazole", "pantoprazole",
    "lansoprazole", "famotidine", "ondansetron", "metoclopramide", "levothyroxine", "prednisone",
    "methylprednisolone", "dexamethasone", "allopurinol", "colchicine", "methotrexate",
    "tacrolimus", "cyclosporine", "sirolimus", "azathioprine", "mycophenolate mofetil",
    "hydroxychloroquine", "ciprofloxacin", "levofloxacin", "moxifloxacin", "azithromycin",
    "clarithromycin", "erythromycin", "doxycycline", "amoxicillin", "sulfamethoxazole",
    "trimethoprim", "metronidazole", "rifampin", "isoniazid", "fluconazole", "itraconazole",
    "ketoconazole", "voriconazole", "terbinafine", "acyclovir", "ritonavir", "sildenafil",
    "tadalafil", "tamsulosin", "finasteride", "montelukast", "theophylline", "albuterol",
    "cetirizine", "loratadine", "diphenhydramine", "hydroxyzine", "cyclobenzaprine", "baclofen",
    "tizanidine", "sumatriptan", "donepezil", "memantine", "levodopa", "ropinirole",
    "pramipexole", "selegiline", "linezolid", "estradiol", "medroxyprogesterone", "tamoxifen",
]

_SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(])")


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"^\s*\d+(\.\d+)*\s+DRUG INTERACTIONS\s*", "", text, flags=re.I).strip()


def chunk(text: str, max_words: int = 120) -> list[str]:
    """Greedy sentence windows of at most `max_words` words (a long sentence stands alone)."""
    out, cur, n = [], [], 0
    for sent in _SENT.split(_clean(text)):
        w = len(sent.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sent)
        n += w
    if cur:
        out.append(" ".join(cur))
    return [c for c in out if len(c.split()) >= 8]


def fetch_label(drug: str, session: requests.Session) -> dict | None:
    cache = RAW_DIR / f"{drug.replace(' ', '_')}.json"
    if cache.exists():
        return json.loads(cache.read_text()) or None
    q = f'openfda.generic_name:"{drug}" AND _exists_:drug_interactions'
    r = session.get(OPENFDA, params={"search": q, "limit": 5}, timeout=30)
    rec = None
    if r.status_code == 200:
        results = r.json().get("results", [])
        # Prefer a single-ingredient label (generic name equals the query).
        results.sort(key=lambda x: [g.lower() for g in x.get("openfda", {}).get("generic_name", [])] != [drug])
        rec = results[0] if results else None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec or {}))
    time.sleep(0.3)  # stay well inside openFDA's anonymous rate limit
    return rec


def build() -> list[dict]:
    session = requests.Session()
    passages = []
    for drug in DRUGS:
        rec = fetch_label(drug, session)
        if not rec:
            continue
        text = " ".join(rec.get("drug_interactions", []))
        for i, c in enumerate(chunk(text)):
            passages.append({"id": f"{drug.replace(' ', '_')}#{i}", "drug": drug,
                             "set_id": rec.get("set_id"), "text": c})
    PASSAGES.parent.mkdir(parents=True, exist_ok=True)
    with PASSAGES.open("w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    return passages


def load(path: Path = PASSAGES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    ps = build()
    print(f"{len(ps)} passages from {len({p['drug'] for p in ps})} labels -> {PASSAGES}")
