"""
tests/test_spark_preprocess.py
──────────────────────────────
Validates the drug-interaction preprocessing logic that the Spark and batch
pipelines share.

Two layers:
  1. Pure-Python tests of the shared transform functions (no Spark/Java needed) —
     these guarantee the Spark UDFs produce identical results to the batch path.
  2. A Spark integration test that runs the real distributed pipeline end-to-end,
     auto-skipped when PySpark (or its Java runtime) is unavailable.
"""
import json

import pytest

from data_pipeline.preprocess_data import (
    build_document_text,
    is_valid_record,
    normalise_drug_name,
)

# A record shaped exactly like the raw JSONL shards in data/raw/.
SAMPLE = {
    "drug_a": "Coumadin 5mg",
    "drug_b": "Aspirin",
    "severity": "Severe",
    "severity_source": "curated",
    "mechanism": "Displacement from plasma protein binding increases free warfarin.",
    "clinical_effects": "Increased bleeding risk.",
    "management": "Avoid concurrent use; monitor INR.",
    "source": "curated",
    "raw_text": "Warfarin-aspirin interaction, high bleeding risk.",
}


# ── 1. Shared transform logic (drives both pipelines) ─────────────────────────

def test_normalise_resolves_synonym_and_strips_dosage():
    # "Coumadin 5mg" → synonym(coumadin)=warfarin, dosage stripped
    assert normalise_drug_name("Coumadin 5mg") == "warfarin"
    assert normalise_drug_name("  ASPIRIN ") == "aspirin"
    assert normalise_drug_name("advil 200 mg tablets") == "ibuprofen"


def test_symmetric_pair_key_is_order_independent():
    # This is the exact key the Spark job dedups on (array_sort of the pair).
    a, b = normalise_drug_name("Coumadin"), normalise_drug_name("Aspirin")
    key_ab = tuple(sorted([a, b]))
    key_ba = tuple(sorted([b, a]))
    assert key_ab == key_ba


def test_quality_filter_rejects_incomplete_records():
    assert is_valid_record(SAMPLE) is True
    assert is_valid_record({"drug_a": "warfarin", "drug_b": ""}) is False
    assert is_valid_record({"drug_a": "warfarin", "drug_b": "aspirin"}) is False  # no content


def test_document_text_contains_core_clinical_fields():
    doc = build_document_text({**SAMPLE, "drug_a": "warfarin", "drug_b": "aspirin"})
    assert "Warfarin" in doc and "Aspirin" in doc
    assert "Severity: Severe" in doc
    assert "Mechanism:" in doc


# ── 2. Spark integration (auto-skips without PySpark/Java) ─────────────────────

@pytest.fixture(scope="module")
def spark():
    pyspark = pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.appName("test-drug-preprocess")
            .master("local[1]")
            .getOrCreate()
        )
    except Exception as exc:  # e.g. no Java runtime on the box
        pytest.skip(f"Spark session unavailable: {exc}")
    yield session
    session.stop()


def test_spark_pipeline_dedups_and_filters(spark, tmp_path):
    from data_pipeline.spark_preprocess import preprocess

    # Two symmetric duplicates + one invalid (no content) → expect 1 clean doc.
    rows = [
        {"drug_a": "Coumadin", "drug_b": "Aspirin"},  # invalid same pair must not erase valid row
        SAMPLE,
        {**SAMPLE, "drug_a": "Aspirin", "drug_b": "Coumadin"},  # same pair, swapped
        {"drug_a": "metformin", "drug_b": "lisinopril"},         # no content → dropped
    ]
    shard = tmp_path / "raw.jsonl"
    shard.write_text("\n".join(json.dumps(r) for r in rows))

    result = preprocess(spark, str(shard)).collect()

    assert len(result) == 1
    md = result[0]["metadata"]
    assert {md["drug_a"], md["drug_b"]} == {"warfarin", "aspirin"}
    assert md["severity"] == "Severe"
