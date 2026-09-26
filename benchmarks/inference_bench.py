"""
Self-hosted inference benchmark and gateway outage drill (Ollama on a laptop; vLLM exposes the same API).

    python -m benchmarks.inference_bench --model llama3.2
1. Direct: time-to-first-token and decode tokens/s from Ollama's own timing fields.
2. Gateway: end-to-end latency through inference.gateway with the self-hosted model as primary.
3. Outage drill: primary pointed at a dead port, self-hosted model as fallback; shows fallback reasons
   and the circuit breaker opening.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

PROMPTS = [
    "In two sentences, why can warfarin and aspirin together increase bleeding risk?",
    "List three questions a patient should ask a pharmacist before combining two new medications.",
    "Explain in one sentence what a drug-drug interaction is.",
    "Summarize: rifampin induces CYP3A4, lowering concentrations of many co-administered drugs.",
    "Give a one-sentence caution about combining sertraline and tramadol.",
]


def direct(base: str, model: str, max_tokens: int) -> dict:
    rows = []
    for p in PROMPTS * 2:
        r = httpx.post(f"{base}/api/generate", json={"model": model, "prompt": p, "stream": False,
                                                     "options": {"num_predict": max_tokens, "temperature": 0}},
                       timeout=120).json()
        rows.append({"ttft_ms": (r["load_duration"] + r["prompt_eval_duration"]) / 1e6,
                     "tok_s": r["eval_count"] / (r["eval_duration"] / 1e9), "tokens": r["eval_count"]})
    rows = rows[1:]  # first call includes model load
    return {"requests": len(rows), "ttft_ms_p50": round(statistics.median(r["ttft_ms"] for r in rows), 1),
            "decode_tokens_per_s": round(statistics.fmean(r["tok_s"] for r in rows), 1),
            "mean_output_tokens": round(statistics.fmean(r["tokens"] for r in rows), 1)}


def through_gateway(env: dict, model: str, max_tokens: int, n: int) -> dict:
    import os
    os.environ.update(env)
    from inference import gateway
    gateway.app.state.router = None
    lat, providers, reasons = [], [], []
    with TestClient(gateway.app) as c:
        for i in range(n):
            t0 = time.perf_counter()
            r = c.post("/v1/chat/completions", json={"model": model, "max_tokens": max_tokens, "temperature": 0,
                                                     "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]})
            lat.append((time.perf_counter() - t0) * 1000)
            body = r.json()
            providers.append(body.get("x_provider"))
            reasons.append(body.get("x_fallback_reason"))
    return {"requests": n, "p50_ms": round(statistics.median(lat), 1), "max_ms": round(max(lat), 1),
            "providers": {p: providers.count(p) for p in set(providers)},
            "fallback_reasons": {r: reasons.count(r) for r in set(reasons) if r}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--base", default="http://localhost:11434")
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    report = {"date": str(date.today()), "model": f"{a.model} (Ollama, Q4 GGUF)", "hardware": "Apple M3, 8 GB",
              "direct": direct(a.base, a.model, a.max_tokens)}
    report["gateway_primary_self_hosted"] = through_gateway(
        {"PRIMARY_BASE_URL": f"{a.base}/v1", "PRIMARY_MODEL": a.model, "PRIMARY_TIMEOUT_S": "120",
         "FALLBACK_BASE_URL": ""}, a.model, a.max_tokens, 6)
    report["outage_drill"] = through_gateway(
        {"PRIMARY_BASE_URL": "http://127.0.0.1:9/v1", "PRIMARY_MODEL": "dead", "PRIMARY_TIMEOUT_S": "2",
         "FALLBACK_BASE_URL": f"{a.base}/v1", "FALLBACK_MODEL": a.model, "FALLBACK_TIMEOUT_S": "120", "BREAKER_THRESHOLD": "3",
         "BREAKER_COOLDOWN_S": "600"}, a.model, a.max_tokens, 6)
    out = Path(__file__).resolve().parents[1] / "docs/benchmarks" / f"inference-{report['date']}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
