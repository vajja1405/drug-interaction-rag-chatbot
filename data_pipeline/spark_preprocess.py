"""
data_pipeline/spark_preprocess.py
──────────────────────────────────
Distributed (PySpark) implementation of the drug-interaction preprocessing
stage. It performs the *same* transformation as data_pipeline/preprocess_data.py
— drug-name normalisation, symmetric-pair deduplication, quality filtering, and
embeddable-document assembly — but on Spark DataFrames, so the identical logic
scales from the curated pairs bundled here to the larger authorized, reviewed corpora without any code change.

Why a Spark path exists alongside the pandas path:
  * `preprocess_data.py` holds everything in memory — fine for the curated seed
    set, but it does not scale once the raw corpus is sharded across many files.
  * `spark_preprocess.py` reads a whole directory of JSONL shards in parallel,
    applies the same pure-Python transform functions as UDFs, and writes
    partitioned Parquet (plus a single consolidated JSONL for the FAISS builder).

The transform functions (`normalise_drug_name`, `build_document_text`,
`is_valid_record`) are imported from `preprocess_data`, so the batch and Spark
paths share their transformations and stay unit-testable without a Spark session.

Run locally:
  spark-submit data_pipeline/spark_preprocess.py \
      --in  data/raw \
      --out data/processed/spark

  # or, without a cluster, using Spark's bundled local[*] master:
  python -m data_pipeline.spark_preprocess --in data/raw --out data/processed/spark
"""

from __future__ import annotations

import argparse
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Reuse the exact same transform logic as the single-node pipeline so the two
# code paths share those transformations.
from data_pipeline.preprocess_data import (
    build_document_text,
    is_valid_record,
    normalise_drug_name,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)


# ── UDFs (thin wrappers around the shared pure-Python functions) ──────────────

normalise_udf = F.udf(normalise_drug_name, T.StringType())


@F.udf(T.BooleanType())
def is_valid_udf(
    drug_a: str | None,
    drug_b: str | None,
    mechanism: str | None,
    clinical_effects: str | None,
    raw_text: str | None,
) -> bool:
    return is_valid_record(
        {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "mechanism": mechanism,
            "clinical_effects": clinical_effects,
            "raw_text": raw_text,
        }
    )


@F.udf(T.StringType())
def build_document_udf(
    drug_a: str | None,
    drug_b: str | None,
    severity: str | None,
    mechanism: str | None,
    clinical_effects: str | None,
    management: str | None,
    raw_text: str | None,
) -> str:
    return build_document_text(
        {
            "drug_a": drug_a or "",
            "drug_b": drug_b or "",
            "severity": severity or "Unknown",
            "mechanism": mechanism or "",
            "clinical_effects": clinical_effects or "",
            "management": management or "",
            "raw_text": raw_text or "",
        }
    )


def _col(df: DataFrame, name: str):
    """Return the column if present, else a typed null literal — raw shards do
    not always carry every optional field."""
    return F.col(name) if name in df.columns else F.lit(None).cast(T.StringType())


# ── Pipeline ──────────────────────────────────────────────────────────────────

def preprocess(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Distributed equivalent of preprocess_data.process_records().

    Steps:
      1. Read raw JSONL shard(s) in parallel.
      2. Normalise drug_a / drug_b (lowercase, strip dosage, resolve synonyms).
      3. Drop rows missing either drug name.
      4. Deduplicate symmetric pairs (A+B == B+A) via an order-independent key.
      5. Quality-filter rows lacking interaction content.
      6. Build the embeddable document_text + nested metadata struct.
    """
    raw = spark.read.json(input_path)
    total_in = raw.count()

    # 2 – normalise names
    norm = raw.withColumn("drug_a", normalise_udf(_col(raw, "drug_a"))).withColumn(
        "drug_b", normalise_udf(_col(raw, "drug_b"))
    )

    # 3 – drop rows missing either drug
    norm = norm.filter(
        (F.col("drug_a").isNotNull())
        & (F.col("drug_b").isNotNull())
        & (F.length(F.trim(F.col("drug_a"))) > 0)
        & (F.length(F.trim(F.col("drug_b"))) > 0)
    )

    # 4 – symmetric-pair dedup: sort the two names so {A,B} == {B,A}, then drop
    #     duplicates on the resulting order-independent key.
    norm = norm.withColumn(
        "pair_key", F.array_sort(F.array(F.col("drug_a"), F.col("drug_b")))
    )

    # 5 – quality filter
    valid = norm.filter(
        is_valid_udf(
            F.col("drug_a"),
            F.col("drug_b"),
            _col(norm, "mechanism"),
            _col(norm, "clinical_effects"),
            _col(norm, "raw_text"),
        )
    )

    valid = valid.dropDuplicates(["pair_key"])

    # 6 – build document_text + metadata struct (mirrors the JSONL schema the
    #     FAISS index builder consumes)
    out = valid.select(
        build_document_udf(
            F.col("drug_a"),
            F.col("drug_b"),
            _col(valid, "severity"),
            _col(valid, "mechanism"),
            _col(valid, "clinical_effects"),
            _col(valid, "management"),
            _col(valid, "raw_text"),
        ).alias("document_text"),
        F.struct(
            F.col("drug_a").alias("drug_a"),
            F.col("drug_b").alias("drug_b"),
            F.coalesce(_col(valid, "severity"), F.lit("Unknown")).alias("severity"),
            F.coalesce(_col(valid, "severity_source"), F.lit("unknown")).alias("severity_source"),
            F.coalesce(_col(valid, "source"), F.lit("unknown")).alias("source"),
            F.coalesce(_col(valid, "evidence_status"), F.lit("unverified")).alias("evidence_status"),
            F.coalesce(_col(valid, "clinical_validation"), F.lit("not_established")).alias("clinical_validation"),
            F.coalesce(_col(valid, "source_url"), F.lit("")).alias("source_url"),
            F.coalesce(_col(valid, "source_page"), F.lit("")).alias("source_page"),
            _col(valid, "rxcui_a").alias("rxcui_a"),
            _col(valid, "rxcui_b").alias("rxcui_b"),
        ).alias("metadata"),
    )

    total_out = out.count()
    logger.info(
        "Spark preprocessing complete: %d raw rows → %d clean documents "
        "(%d removed as duplicate/invalid)",
        total_in,
        total_out,
        total_in - total_out,
    )
    return out


def write_outputs(df: DataFrame, output_path: str) -> None:
    """Write partitioned Parquet (analytics-friendly) plus a single consolidated
    JSONL that the existing FAISS index builder can consume unchanged."""
    df.write.mode("overwrite").parquet(f"{output_path}/parquet")
    # coalesce(1) → one JSONL part so build_index.py needs no changes
    df.coalesce(1).write.mode("overwrite").json(f"{output_path}/jsonl")
    logger.info("Wrote Parquet + JSONL outputs under %s", output_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed (PySpark) drug-interaction preprocessing")
    parser.add_argument(
        "--in",
        dest="input_path",
        default="data/raw",
        help="Input directory or JSONL glob of raw interaction shards",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/processed/spark",
        help="Output directory for Parquet + JSONL",
    )
    parser.add_argument(
        "--master",
        default="local[*]",
        help="Spark master URL (default: local[*])",
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("drug-interaction-preprocess")
        .master(args.master)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        processed = preprocess(spark, args.input_path)
        write_outputs(processed, args.output_path)
        print(f"Spark preprocessing complete → {args.output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
