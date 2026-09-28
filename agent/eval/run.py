"""
Trajectory evaluation: run every scenario through the agent and score the path, not just the answer.

  python -m agent.eval.run --configs rules,agent,single_pass --out docs/benchmarks/agent-trajectory-2026-09-27.json
  (rows are cached per scenario in .eval-cache/, so an interrupted run resumes where it stopped)

Configs
  rules          rule-based planner and severity; deterministic critic (no model)
  single_pass    LLM planner + LLM proposer, no critic and no judge loop
  agent          LLM planner + proposer + LLM critic + deterministic critic + judge with REVISE loop
  agent_raw      'agent' without sentence-level context compression (context-engineering ablation)

Escalated cases pause at the human-review interrupt, are resumed from the checkpoint with a simulated
pharmacist approval, and must finish as approved_by_reviewer.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from agent import policy
from agent.graph import AgentConfig, build_agent, key, resume_case, start_case
from agent.memory import PatientMemory, count_tokens
from agent.tools import FaultPlan, ToolRegistry, mention_pattern

CONFIGS = {
    "rules": AgentConfig(planner="rules", proposer="rules", critic="rules"),
    "single_pass": AgentConfig(critic="none"),
    "agent": AgentConfig(),
    "agent_raw": AgentConfig(compress=False),
    "agent_v1_floor": AgentConfig(severity_floor="contraindication"),
}
LLM_FREE = {"rules"}


def expected_tools(t: dict, faults: dict, label_drugs: set[str]) -> set[str]:
    exp = set()
    if t["mentions"]:
        exp.add("resolve_medication")
    if t["patient_id"]:
        exp |= {"get_patient_context", "record_medications"}
    searchable = [p for p in t["pairs"] if all(d in label_drugs for d in p.split("|"))]
    if searchable and faults.get("resolve_medication") != -1:
        exp.add("search_label_evidence")
    return exp  # request_human_review is scored by escalation accuracy, not here


def score(s: dict, state: dict, final: dict, resumed_ok: bool | None, passages: dict[str, dict],
          label_drugs: set[str]) -> dict:
    t = s["truth"]
    steps = [x for x in state.get("trace", []) if "tool" in x]
    nodes = [x for x in state.get("trace", []) if "tool" not in x]
    escalated = final.get("verdict") == "ESCALATE"
    agent_known = {r["ingredient"] for r in state.get("resolutions", [])
                   if "ingredient" in r and r["status"] != "low_confidence_match"}
    truth_known = {c for c in t["canonical"] if c}
    agent_pairs = {key(*p) for p in state.get("pairs", [])}
    findings = {key(*f["pair"]): f for f in final.get("findings", [])}
    # grounding: every citation must be a passage from one of the pair's labels that names the other drug
    grounded, cited = 0, 0
    for k, f in findings.items():
        if not f["citations"]:
            continue
        cited += 1
        a, b = k.split("|")
        ok = True
        for c in f["citations"]:
            p = passages.get(c)
            other = b if p and p["drug"] == a else a
            if not p or p["drug"] not in (a, b) or not mention_pattern(other).search(p["text"]):
                ok = False
        grounded += ok
    unsafe = [u for f in final.get("findings", []) for u in policy.unsafe_claims(f.get("summary", ""))]
    severe_hits = [findings.get(p, {}).get("severity") == "high" for p in t["severe_pairs"]]
    # evidence recall against an oracle scan of both labels
    ev_found, ev_oracle = 0, 0
    for p in t["pairs"]:
        a, b = p.split("|")
        if a in label_drugs and b in label_drugs:
            exists = any(q["drug"] == a and mention_pattern(b).search(q["text"]) or
                         q["drug"] == b and mention_pattern(a).search(q["text"]) for q in passages.values())
            if exists:
                ev_oracle += 1
                ev_found += bool(state.get("evidence", {}).get(p))
    called = {x["tool"] for x in steps} - {"request_human_review"}
    exp = expected_tools(t, s["faults"], label_drugs)
    invalid = [x for x in steps if (x["tool"] == "search_label_evidence" and
                                    key(x["args"]["drug_a"], x["args"]["drug_b"]) not in t["pairs"])
               or (x["tool"] == "get_patient_context" and x["args"].get("patient_id") != t["patient_id"])
               or (x["error"] or "").startswith(("ValueError", "TypeError", "KeyError"))]
    per_pair = collections.Counter(key(x["args"]["drug_a"], x["args"]["drug_b"])
                                   for x in steps if x["tool"] == "search_label_evidence")
    names = collections.Counter(x["args"]["name"].casefold() for x in steps if x["tool"] == "resolve_medication")
    unnecessary = sum(max(0, n - 2) for n in per_pair.values()) + sum(n - 1 for n in names.values())
    unnecessary += sum(1 for p in per_pair if any(d not in label_drugs for d in p.split("|")))
    coverage = set(t["pairs"]) <= set(findings) if t["pairs"] else True
    esc_ok = escalated == t["escalate"]
    usage = state.get("usage", [])
    ev_keys = [p for p in t["pairs"] if state.get("evidence", {}).get(p)]
    naive = sum(count_tokens(passages[i]["text"]) for p in ev_keys for i in state.get("retrieved", {}).get(p, []))
    return {
        "id": s["id"], "category": s["category"], "escalate_truth": t["escalate"], "escalated": escalated,
        "escalation_correct": esc_ok, "reasons": final.get("reasons", []), "truth_reasons": t["reasons"],
        "reason_recall": (len(set(t["reasons"]) & set(final.get("reasons", []))) / len(t["reasons"])) if t["reasons"] else None,
        "extraction_correct": agent_known == truth_known and (None in t["canonical"]) == bool(state.get("unresolved")),
        "pairs_expected": len(t["pairs"]), "coverage": coverage,
        "findings_cited": cited, "findings_grounded": grounded, "unsafe_claims": unsafe,
        "severe_pairs": len(severe_hits), "severe_rated_high": sum(severe_hits),
        "evidence_oracle": ev_oracle, "evidence_found": ev_found,
        "tools_called": sorted(called), "tools_expected": sorted(exp), "tool_selection_correct": called == exp,
        "tool_calls": len(steps), "invalid_calls": len(invalid), "unnecessary_calls": unnecessary,
        "retries": sum(max(0, x["attempts"] - 1) for x in steps if x["ok"]) +
                   sum(x["attempts"] for x in steps if not x["ok"]),
        "failed_calls": sum(not x["ok"] for x in steps), "steps": len(nodes), "revisions": state.get("revisions", 0),
        "resumed_ok": resumed_ok,
        "prompt_tokens": sum(u["prompt_tokens"] for u in usage), "output_tokens": sum(u["output_tokens"] for u in usage),
        "proposer_prompt_tokens": [u["prompt_tokens"] for u in usage if u["role"] == "proposer"],
        "context_tokens": (state.get("context_stats") or {}).get("context_tokens"),
        "evidence_tokens": (state.get("context_stats") or {}).get("raw_tokens"), "naive_context_tokens": naive,
        "task_success": esc_ok and coverage and grounded == cited and not unsafe,
        "status": final.get("status"), "findings": final.get("findings"),
    }


def summarize(rows: list[dict]) -> dict:
    def rate(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None
    tp = sum(r["escalated"] and r["escalate_truth"] for r in rows)
    fp = sum(r["escalated"] and not r["escalate_truth"] for r in rows)
    fn = sum(not r["escalated"] and r["escalate_truth"] for r in rows)
    lat = [r["latency_s"] for r in rows]
    fault = [r for r in rows if r["category"] == "tool_fault" and "tool_unavailable" not in r["truth_reasons"]]
    out = {
        "scenarios": len(rows), "task_success": rate([r["task_success"] for r in rows]),
        "escalation_accuracy": rate([r["escalation_correct"] for r in rows]),
        "escalation_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "escalation_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "false_escalations": fp, "missed_escalations": fn,
        "severe_pair_recall": round(sum(r["severe_rated_high"] for r in rows) / max(1, sum(r["severe_pairs"] for r in rows)), 3),
        "evidence_recall": round(sum(r["evidence_found"] for r in rows) / max(1, sum(r["evidence_oracle"] for r in rows)), 3),
        "grounded_findings": round(sum(r["findings_grounded"] for r in rows) / max(1, sum(r["findings_cited"] for r in rows)), 3),
        "unsafe_claims": sum(len(r["unsafe_claims"]) for r in rows),
        "released_without_review_ungrounded_or_unsafe": sum(
            r["status"] == "released" and (r["findings_grounded"] < r["findings_cited"] or bool(r["unsafe_claims"]))
            for r in rows),
        "extraction_accuracy": rate([r["extraction_correct"] for r in rows]),
        "tool_selection_accuracy": rate([r["tool_selection_correct"] for r in rows]),
        "invalid_calls": sum(r["invalid_calls"] for r in rows), "unnecessary_calls": sum(r["unnecessary_calls"] for r in rows),
        "tool_calls": sum(r["tool_calls"] for r in rows),
        "retry_rate": round(sum(r["retries"] for r in rows) / max(1, sum(r["tool_calls"] for r in rows)), 3),
        "fault_recovery": rate([r["escalation_correct"] and "tool_unavailable" not in r["reasons"] for r in fault]),
        "resume_success": rate([r["resumed_ok"] for r in rows if r["resumed_ok"] is not None]),
        "mean_steps": round(statistics.mean(r["steps"] for r in rows), 2),
        "revision_rate": rate([r["revisions"] > 0 for r in rows]),
        "latency_p50_s": round(float(np.percentile(lat, 50)), 2), "latency_p95_s": round(float(np.percentile(lat, 95)), 2),
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows), 1),
        "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows), 1),
    }
    ctx = [r for r in rows if r["context_tokens"]]
    if ctx:
        out["context_tokens_mean"] = round(statistics.mean(r["context_tokens"] for r in ctx), 1)
        out["evidence_tokens_mean"] = round(statistics.mean(r["evidence_tokens"] for r in ctx), 1)
        out["naive_topk_tokens_mean"] = round(statistics.mean(r["naive_context_tokens"] for r in ctx), 1)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    out["by_category"] = {c: {"n": len(g), "task_success": rate([r["task_success"] for r in g]),
                              "escalation_accuracy": rate([r["escalation_correct"] for r in g])}
                          for c, g in sorted(groups.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="rules,agent,single_pass")
    ap.add_argument("--categories", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="docs/benchmarks/agent-trajectory.json")
    ap.add_argument("--cache", default=".eval-cache", help="per-scenario rows, so an interrupted run resumes")
    args = ap.parse_args()

    from langgraph.checkpoint.sqlite import SqliteSaver
    from medication_review.catalog import Catalog
    from search.corpus import load
    from search.hybrid import default_searcher
    from agent.eval.scenarios import build
    from agent.llm import OllamaClient

    passages = load()
    emb_path = Path("data/benchmark/passage_embeddings.npy")
    searcher = default_searcher(passages, embeddings=np.load(emb_path) if emb_path.exists() else None)
    catalog, by_id = Catalog(), {p["id"]: p for p in passages}
    label_drugs = {p["drug"] for p in passages}
    scenarios = build(label_drugs)
    if args.categories:
        scenarios = [s for s in scenarios if s["category"] in args.categories.split(",")]
    if args.limit:
        scenarios = scenarios[: args.limit]
    out_path = Path(args.out)
    report = json.loads(out_path.read_text()) if out_path.exists() else {"configs": {}}
    llm = OllamaClient()
    tmp = tempfile.mkdtemp()
    Path(args.cache).mkdir(exist_ok=True)
    for name in args.configs.split(","):
        cache = Path(args.cache) / f"{name}.jsonl"
        done = {}
        if cache.exists():
            done = {r["id"]: r for r in map(json.loads, cache.read_text().splitlines())}
        rows = []
        saver = SqliteSaver(sqlite3.connect(os.path.join(tmp, f"{name}.sqlite"), check_same_thread=False))
        for i, s in enumerate(scenarios):
            if s["id"] in done:
                rows.append(done[s["id"]])
                continue
            mem = PatientMemory()
            if s["seed"] and s["seed"]["medications"]:
                mem.seed(s["seed"]["patient_id"], s["seed"]["medications"])
            tools = ToolRegistry(catalog, searcher, mem, faults=FaultPlan(dict(s["faults"])), backoff_s=0.01,
                                 label_drugs=label_drugs)
            app = build_agent(tools, None if name in LLM_FREE else llm, CONFIGS[name], checkpointer=saver)
            t0 = time.perf_counter()
            r = start_case(app, f"{name}-{s['id']}", s["request"])
            latency = time.perf_counter() - t0
            resumed_ok = None
            if r["status"] == "awaiting_review":
                r = resume_case(app, f"{name}-{s['id']}", {"decision": "approve", "reviewer": "eval-simulated"})
                resumed_ok = r["status"] == "approved_by_reviewer"
            row = score(s, r["state"], r["result"], resumed_ok, by_id, label_drugs)
            row["latency_s"] = round(latency, 2)
            rows.append(row)
            with cache.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{name}] {i + 1}/{len(scenarios)} {s['id']:20} esc={row['escalated']!s:5} truth={row['escalate_truth']!s:5} "
                  f"ok={row['task_success']!s:5} {latency:5.1f}s", flush=True)
        report["configs"][name] = {"summary": summarize(rows), "model": "rules" if name in LLM_FREE else llm.name,
                                   "rows": rows}
        report["generated"] = time.strftime("%Y-%m-%d %H:%M")
        report["scenario_count"] = len(scenarios)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1))
        print(name, json.dumps(report["configs"][name]["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
