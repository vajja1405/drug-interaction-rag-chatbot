"""Content-addressed cache identities. No external services or secrets."""
import hashlib
import json

PIPELINE_VERSION = "2026-09-11-calibration-v2"

def evidence_version(texts, metadata):
    payload = json.dumps([texts, metadata], sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

def pair_cache_key(drugs, evidence, model, temperature, max_tokens, include_low=True):
    payload = [PIPELINE_VERSION, sorted(d.strip().lower() for d in drugs),
               evidence, model, temperature, max_tokens, include_low]
    return "ddi_v2:" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()
