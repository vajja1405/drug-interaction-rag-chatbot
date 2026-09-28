# Agent trajectory evaluation (2026-09-28 00:44)

127 scenarios. Model: ollama:llama3.2 on an Apple M3 (8 GB) CPU. `rules` uses no model; `single_pass` is planner + proposer without the critic or judge loop; `agent` is the full proposer → critic → judge graph (policy v2).

| Metric | rules | single_pass | agent |
|---|---|---|---|
| Task success | 0.827 | 0.74 | 0.772 |
| Escalation accuracy | 0.827 | 0.772 | 0.787 |
| Escalation precision | 0.855 | 0.783 | 0.788 |
| Escalation recall | 0.855 | 0.855 | 0.882 |
| Curated Severe pairs rated high | 0.673 | 0.636 | 0.673 |
| Findings with grounded citations | 1.0 | 0.955 | 0.978 |
| Unsafe claims in any output | 0 | 0 | 0 |
| Ungrounded/unsafe findings released without review | 0 | 1 | 0 |
| Evidence recall vs oracle | 0.968 | 0.968 | 0.968 |
| Medication extraction accuracy | 0.906 | 0.984 | 0.984 |
| Tool selection accuracy | 0.992 | 0.992 | 0.992 |
| Invalid tool calls | 0 | 0 | 0 |
| Unnecessary tool calls | 0 | 0 | 0 |
| Retries per tool call | 0.071 | 0.069 | 0.069 |
| Recovery from transient faults | 0.889 | 0.778 | 0.778 |
| Escalations resumed from checkpoint | 1.0 | 1.0 | 1.0 |
| Mean graph steps | 9.06 | 8.59 | 10.01 |
| Cases revised by the critic loop | 0.0 | 0.0 | 0.181 |
| Latency p50 (s) | 0.16 | 23.18 | 25.3 |
| Latency p95 (s) | 1.06 | 60.37 | 212.96 |
| Prompt tokens per case | 0 | 431.6 | 1166.4 |

## By category (task success)

| Category | n | rules | single_pass | agent |
|---|---|---|---|---|
| brand_name | 8 | 0.625 | 0.5 | 0.5 |
| control_pair | 15 | 1.0 | 1.0 | 1.0 |
| curated_pair | 35 | 0.743 | 0.629 | 0.686 |
| memory | 11 | 0.727 | 0.455 | 0.545 |
| misspelling | 15 | 0.667 | 0.6 | 0.667 |
| multi_drug | 10 | 0.9 | 0.8 | 0.8 |
| prompt_injection | 10 | 1.0 | 1.0 | 1.0 |
| tool_fault | 15 | 0.933 | 0.867 | 0.867 |
| unresolvable | 8 | 1.0 | 1.0 | 1.0 |
