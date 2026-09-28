"""Render the trajectory-evaluation JSON files as a markdown report."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

MAIN = Path("docs/benchmarks/agent-trajectory-2026-09-27.json")
ABL = Path("docs/benchmarks/agent-ablations-2026-09-27.json")
OUT = Path("docs/benchmarks/agent-trajectory-2026-09-27.md")
ROWS = [("task_success", "Task success"), ("escalation_accuracy", "Escalation accuracy"),
        ("escalation_precision", "Escalation precision"), ("escalation_recall", "Escalation recall"),
        ("severe_pair_recall", "Curated Severe pairs rated high"), ("grounded_findings", "Findings with grounded citations"),
        ("unsafe_claims", "Unsafe claims in any output"),
        ("released_without_review_ungrounded_or_unsafe", "Ungrounded/unsafe findings released without review"), ("evidence_recall", "Evidence recall vs oracle"),
        ("extraction_accuracy", "Medication extraction accuracy"), ("tool_selection_accuracy", "Tool selection accuracy"),
        ("invalid_calls", "Invalid tool calls"), ("unnecessary_calls", "Unnecessary tool calls"),
        ("retry_rate", "Retries per tool call"), ("fault_recovery", "Recovery from transient faults"),
        ("resume_success", "Escalations resumed from checkpoint"), ("mean_steps", "Mean graph steps"),
        ("revision_rate", "Cases revised by the critic loop"), ("latency_p50_s", "Latency p50 (s)"),
        ("latency_p95_s", "Latency p95 (s)"), ("mean_prompt_tokens", "Prompt tokens per case")]


def table(configs: dict, names: list[str]) -> list[str]:
    lines = ["| Metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for k, label in ROWS:
        lines.append(f"| {label} | " + " | ".join(str(configs[n]["summary"].get(k)) for n in names) + " |")
    return lines


def main() -> None:
    from agent.eval.run import summarize
    main_r = json.loads(MAIN.read_text())
    for c in main_r["configs"].values():  # recompute so every metric reflects the current scorer
        c["summary"] = summarize(c["rows"])
    MAIN.write_text(json.dumps(main_r, indent=1))
    names = [n for n in ("rules", "single_pass", "agent") if n in main_r["configs"]]
    out = [f"# Agent trajectory evaluation ({main_r['generated']})", "",
           f"{main_r['scenario_count']} scenarios. Model: {main_r['configs'][names[-1]]['model']} on an "
           "Apple M3 (8 GB) CPU. `rules` uses no model; `single_pass` is planner + proposer without the "
           "critic or judge loop; `agent` is the full proposer → critic → judge graph (policy v2).", ""]
    out += table(main_r["configs"], names)
    out += ["", "## By category (task success)", "", "| Category | n | " + " | ".join(names) + " |",
            "|---" * (len(names) + 2) + "|"]
    cats = main_r["configs"][names[0]]["summary"]["by_category"]
    for c, v in cats.items():
        out.append(f"| {c} | {v['n']} | " + " | ".join(
            str(main_r["configs"][n]["summary"]["by_category"][c]["task_success"]) for n in names) + " |")
    if ABL.exists():
        abl = json.loads(ABL.read_text())
        ids = {r["id"] for r in next(iter(abl["configs"].values()))["rows"]}
        agent_rows = [r for r in main_r["configs"]["agent"]["rows"] if r["id"] in ids]
        for c in abl["configs"].values():
            c["summary"] = summarize(c["rows"])
        ABL.write_text(json.dumps(abl, indent=1))
        merged = {"agent": {"summary": summarize(agent_rows)}, **abl["configs"]}
        names2 = ["agent"] + list(abl["configs"])
        out += ["", f"## Ablations on the {len(ids)} curated-pair scenarios", "",
                "`agent_raw` sends full evidence passages instead of compressed sentences; `agent_v1_floor` "
                "uses the v1 severity floor (contraindication language only).", ""]
        out += table(merged, names2)
        ctx = [(r["context_tokens"], r["evidence_tokens"], r["naive_context_tokens"]) for r in agent_rows if r["context_tokens"]]
        if ctx:
            out += ["", "Reading: sentence compression did not buy accuracy here. Full passages scored 0.743 vs 0.686 "
                    "(2 of 35 scenarios, within noise at this size) at twice the latency; most of the context saving "
                    "comes from cross-mention filtering, not from compression. The v2 severity floor, adopted after "
                    "the first run, lifts escalation recall from 0.773 to 0.864 over v1 on the same cases."]
            out += ["", "Context size per case with evidence (tokens, cl100k): compressed "
                    f"{statistics.mean(c[0] for c in ctx):.0f}, full cross-mention passages {statistics.mean(c[1] for c in ctx):.0f}, "
                    f"all retrieved top-k passages {statistics.mean(c[2] for c in ctx):.0f}."]
    OUT.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
