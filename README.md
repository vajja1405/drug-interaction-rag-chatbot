# Drug Interaction AI

A medication evidence workspace with a React interface, FastAPI backend, RxNorm terminology, DailyMed label retrieval, and a separate retrieval/classification/LLM research demonstration.

[Open the application](https://astra6-drug-interaction-ai.hf.space) · [Hugging Face Space](https://huggingface.co/spaces/astra6/drug-interaction-ai) · [Offline-style browser demo](https://vajja1405.github.io/drug-interaction-rag-chatbot/)

## Medication label review

Search **23,651 RxNorm terms** across ingredients, combinations and brands. Select **2–20 distinct concepts** and inspect all **1–190 unordered pairs**. The catalog is terminology, not a list of 23,651 clinically verified medicines or complete interaction coverage.

The review retrieves a bounded sample of human prescription/OTC labels linked to each selected RxCUI, verifies the active-ingredient names against the selected concept, and looks for direct names in the dedicated interaction section. Labels with additional or missing active ingredients are excluded. Strength, route, manufacturer and exact formulation still require confirmation against the original package.

Results distinguish:

| Result | Meaning |
|---|---|
| Possible ingredient overlap | Selected concepts share an ingredient identifier; confirm exact products and intended use. |
| Direct label mention | An interaction-section passage names a selected drug or ingredient. It may describe no effect, a study or a conditional finding. |
| No direct mention | No exact name match was found in the sampled sections. **This does not establish no interaction or safety.** |
| Incomplete sources | At least one necessary interaction section could not be retrieved or matched. |

The interface includes quoted context, original label links, source versions, per-medicine coverage, filters, pagination, a pair matrix and a downloadable discussion report. Label review does **not** use an LLM, predict severity, recommend a dose, or tell users to start or stop medicines. It is intended to help prepare a conversation with a pharmacist or prescriber.

**Not assessed:** class-based interactions, dose/route/timing effects, patient-specific risks, food interactions, or three-drug/higher-order effects. Reviewing 190 pairs does not establish the safety of a 20-medicine regimen. Clinical validation has not been established.

## System design

```text
Medication label review
  RxNorm catalog search → explicit concept selection (RxCUI)
  → bounded DailyMed human-label lookup → active-ingredient-set check
  → original interaction-section excerpts + possible ingredient overlap
  → all pair categories + source coverage + discussion report

RAG research mode
  Medication names → pair expansion → FAISS retrieval
  → structured / keyword / calibrated-model evidence assessment
  → remote LLM explanation → versioned response cache
```

`medication_review/` implements terminology, bounded HTTP retrieval, XML identity checks and conservative matching. `api/review.py` exposes `/api/v2/catalog`, `/api/v2/medications` and `/api/v2/review`. Catalog selection and label review do not depend on the explanation provider.

Source requests use fixed NLM origins, bounded response sizes, four retrieval workers, a shared request pacer, per-medication deadlines and an overall review deadline. A bounded in-memory source cache expires after 12 hours; HTTP failures are not cached as successful empty results. Review admission allows two concurrent requests and 1,000 requests per process per UTC day. The public endpoint allows five reviews per minute per observed client IP. Shared reverse proxies can make an IP limit apply to multiple visitors. These limits are not a production-capacity guarantee.

## RAG research mode

The separate **RAG research demo** retains the full Python retrieval/classification/explanation path, with a five-medication hosted limit. Its small curated corpus and severity output are unvalidated research fixtures. Ordinary uncached API requests can call the explanation model even when structured metadata supplied severity. The classifier module, API and component evaluator have different call paths.

The random-forest model uses isotonic calibration with `cv=3` around the entire TF-IDF/classifier pipeline. Calibration-fold data does not fit the training vocabulary. This is a fitting procedure, not evidence of clinical reliability.

The 24-pair component regression measures agreement with authored severity labels and skips the final LLM. Its custom checks are **not an official RAGAS evaluation**; generated-answer faithfulness is not evaluated. See the [component record](docs/validation-2026-09-11.json) and [original hosted smoke check](docs/hosting-smoke-2026-09-11.json) for their precise scopes. Cache keys include source content, pipeline version, model settings and filtering; failed generations are not cached as successful responses. Cache-hit latency is not total-system cost reduction.

RxNav's retired interaction endpoint is disabled. RxNorm is used for terminology, not presented as a current interaction API. Curated fixtures, medication labels and generated explanations have different provenance.

## Run and test

Use Python 3.11 or 3.12 and Node 22. Java 17 is needed for the Spark integration test.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python build_index.py --all-seed
uvicorn api.server:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Label review uses public NLM services; no explanation-model key is required for that path. Set `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `LLM_MODEL` privately for the RAG mode. Never put secrets in frontend variables or source files.

```bash
python -m pytest tests/ -v
python -m evaluation.run_eval --verbose
cd frontend
npm test
npm run build
```

A [live-source verification record](docs/label-review-verification.json) documents one 20-concept / 190-pair smoke check and its incomplete sources. It is not a clinical evaluation.

Tests cover concept identity and input bounds, all 190 pair combinations, missing/failed sources, negated excerpts, substring rejection, ingredient overlap, rejection of combination labels for single ingredients, XML identity/version, truncation, cache expiry, and the existing API/classifier/cache/Spark behavior. Mocked tests do not establish medical-answer correctness.

Refresh the terminology snapshot explicitly, inspect the diff, and redeploy:

```bash
python -m medication_review.refresh_catalog
```

The snapshot includes its source URL, source version, retrieval timestamp and hash. Label versions and freshness information are retained for traceability. Updating the catalog does not independently validate its interaction coverage.

## Hosting and privacy

A Docker Space on CPU Basic serves React and FastAPI from the same origin. The image includes the terminology snapshot, fixture index, embedding weights and calibrated research classifier, and runs as a non-root user. The RAG mode uses `Qwen/Qwen3-4B-Instruct-2507:nscale` through Hugging Face Inference Providers, with credentials stored as a runtime secret.

Label review sends selected concept identifiers to the server and NLM services. It does not persist a user's medication list or send it to a language-model provider. Source responses are cached by resource, not by patient. The UI has no patient-record field and no browser-storage persistence for the list. Hosting and upstream services may retain operational logs; this is not a clinical privacy/compliance certification. RAG mode separately sends medication names and evidence to its model provider.

The hosted RAG configuration bounds inference at one concurrent uncached analysis, 50 uncached analyses/process/UTC day, 1,800 output tokens and 40 seconds per attempt. A JSON attempt can fall back to one text attempt. Process-local counters reset on restart and are not billing ceilings. Provider availability/credits and Space startup can affect availability. The browser-only backup uses 35 local research records and makes no server or LLM calls.

## Agentic track: stateful clinical evidence agent (September 27, 2026)

`agent/` adds a **LangGraph** agent on top of the evidence search: it reads a free-text request, resolves
medications to RxNorm, retrieves label evidence for every pair, drafts cited findings, critiques them, and
decides under an explicit policy whether it may release the result or must hand the case to a pharmacist.
Architecture, state, policy and design decisions: **[docs/AGENT.md](docs/AGENT.md)**.

```
intake (planner LLM, injection screen) → patient memory → RxNorm resolve → label-scoped hybrid retrieval
  → evidence validation (query-expansion retry) → proposer → critic (deterministic + LLM) → judge
judge: REVISE → proposer (≤2)   ESCALATE → review ticket → human_review [interrupt, SQLite checkpoint]   PASS → finalize
```

- **Proposer → critic → judge.** Citations must be passages retrieved for that pair; risk concepts in a
  summary must appear in the cited text; no "safe to take together" claims; serious-harm label language
  forces severity *high*. Failed checks go back to the proposer with the critic's feedback.
- **Human in the loop.** Escalations open an idempotent ticket and pause at `interrupt()`; the case
  resumes from its checkpoint when a pharmacist approves, overrides or rejects, even after a restart.
- **Memory and context.** SQLite patient memory (written only after release or approval) and a context
  builder that keeps partner-drug and risk sentences under a token budget, tagged with passage ids.
- **Tools and MCP.** Five tools with validation, retries, backoff and fault injection; an MCP server
  (FastMCP) exposes the read tools and the review workflow. API: `/api/v4/agent/reviews`.
- **Escalation routing (n8n).** With `AGENT_ESCALATION_WEBHOOK` set, each review ticket is posted to an
  n8n workflow that pages the on-call pharmacist for urgent cases and queues routine ones
  ([integrations/n8n](integrations/n8n)); verified end to end on n8n 2.40.7. Only ticket metadata leaves the agent.

**Trajectory evaluation**: 127 scenarios with ground truth the agent never sees (curated pairs, controls,
misspellings, brand names, unidentifiable drugs, injected tool faults, prompt injections, patient-memory
follow-ups, multi-drug reconciliations). Llama 3.2 3B via Ollama on an Apple M3 CPU.

| | Rules only | Single pass (LLM) | Proposer → critic → judge |
|---|---|---|---|
| Task success | **0.827** | 0.740 | 0.772 |
| Escalation recall / precision | 0.855 / **0.855** | 0.855 / 0.783 | **0.882** / 0.788 |
| Curated Severe pairs rated high | 0.673 | 0.636 | 0.673 |
| Medication extraction from free text | 0.906 | **0.984** | **0.984** |
| Findings with grounded citations | 1.000 | 0.955 | 0.978 |
| Ungrounded or unsafe findings released without review | 0 | **1** | 0 |
| Invalid / unnecessary tool calls | 0 / 0 | 0 / 0 | 0 / 0 |
| Escalations resumed from checkpoint | 100% | 100% | 100% |
| Cases revised by the critic loop | – | – | 18% |
| Latency p50 / p95 (CPU) | 0.2 s / 1.1 s | 23 s / 60 s | 25 s / 213 s |

The critic loop fixed 4 scenarios that single pass got wrong and broke none. Evidence in the prompt: all
retrieved top-k passages would average 1,727 tokens per case; cross-mention filtering leaves 495 and
sentence compression 362 (−79% in total, most of it from filtering). Ablations on the 35 curated pairs:
full passages instead of compressed sentences scored 0.743 vs 0.686 task success (2 scenarios, within
noise) at twice the latency, so compression buys speed and tokens rather than accuracy; the v1 severity
floor scored 0.629 with escalation recall 0.773 vs 0.864 for v2. Full tables:
[docs/benchmarks/agent-trajectory-2026-09-27.md](docs/benchmarks/agent-trajectory-2026-09-27.md).

What the numbers say:
- **The model helps where language matters.** The planner extracts medications from free text better
  than dictionary rules (0.984 vs 0.906; unknown and misspelled names), and the critic loop raises
  escalation recall and grounding over a single pass.
- **Deterministic guardrails still carry safety.** Rules alone score highest on task success here: the 3B
  model over-escalates relative to the curated labels (e.g. NSAID + ACE-inhibitor renal warnings rated
  high) and, like the rules, misses Severe pairs whose labels use mild wording (fluconazole + warfarin).
  The single pass released one ungrounded finding; the full graph released none.
- **Honest limits.** Severity ground truth is a small set of curated research fixtures, not clinical
  adjudication; one policy change (v1 → v2 severity floor) was made after the first run and both are
  reported in [the results](docs/benchmarks/agent-trajectory-2026-09-27.md).

Reproduce: `python -m agent.eval.run --configs rules,agent,single_pass` then `python -m agent.eval.report`.
Tests: `tests/test_clinical_agent.py` (18, scripted model, in CI).

## Production AI track (September 26, 2026)

The label-review app and research chatbot stay as they were. This track adds the pieces a team would need to
run retrieval and generation as a service: evidence search with measured retrieval quality, a model gateway
with self-hosted inference and fallback, metrics and traces, load testing, and deployment manifests.

### 1. Hybrid evidence search with a cross-encoder reranker

`search/` indexes the `drug_interactions` sections of **154 FDA drug labels (openFDA)**, split into **1,011 passages**,
and serves them at `GET /api/v3/evidence/search?q=...&mode=hybrid_rerank`.

```
query ─┬─▶ BM25 (exact drug names) ──────┐
       └─▶ MiniLM dense (paraphrase, typos)┴─▶ reciprocal rank fusion ─▶ cross-encoder rerank ─▶ top-k passages
```

**Benchmark:** 600 queries built from drug pairs the labels actually mention (300 clean, 300 with a misspelled drug name,
as patients type). Relevance is mention-based (label A passages naming drug B, and vice versa), a proxy rather than
pharmacist judgment. `python -m benchmarks.retrieval.run` reproduces it.

| Mode | Hit@5 | MRR@10 | nDCG@10 | MRR (clean) | MRR (misspelled) | p50 | p95 |
|---|---|---|---|---|---|---|---|
| BM25 | 0.618 | 0.490 | 0.513 | 0.782 | 0.197 | 0.3 ms | 0.4 ms |
| Dense (MiniLM) | 0.682 | 0.496 | 0.505 | 0.542 | 0.450 | 3.12 ms | 4.57 ms |
| Hybrid (RRF) | 0.700 | 0.575 | 0.584 | 0.773 | 0.377 | 3.7 ms | 5.19 ms |
| Hybrid + cross-encoder (depth 30) | 0.793 | 0.652 | 0.651 | 0.725 | 0.579 | 548.03 ms | 1034.49 ms |

Hybrid + rerank lifts MRR@10 from **0.496 (dense only, the chatbot's original approach) to 0.652 (+31%)** and Hit@5 from
0.682 to 0.793, and nearly triples BM25 on misspelled drugs (MRR 0.197 → 0.579). The cost is the reranker, so rerank depth
was swept (`benchmarks/retrieval/rerank_depth.py`):

| Rerank depth | MRR@10 | Hit@5 | p50 | p95 |
|---|---|---|---|---|
| 5 | 0.583 | 0.700 | 69.6 ms | 91.8 ms |
| 10 | 0.619 | 0.752 | 151.7 ms | 231.8 ms |
| 15 | 0.638 | 0.777 | 240.6 ms | 359.5 ms |
| 20 | 0.645 | 0.782 | 387.1 ms | 689.8 ms |
| 30 | 0.652 | 0.793 | 584.9 ms | 1081.4 ms |

The service defaults to depth 10 (`EVIDENCE_RERANK_DEPTH`): most of the quality gain at a quarter of the depth-30 latency.

### 2. Model gateway: self-hosted first, external fallback

`inference/gateway.py` is an OpenAI-compatible `/v1/chat/completions` proxy. The API only changes `OPENAI_BASE_URL`.
The router sends each request to a self-hosted server (vLLM in the EKS manifests, Ollama on a laptop) and falls back to
the external API on a timeout, an HTTP error, an empty answer, or invalid JSON when JSON was requested. A circuit breaker
skips the primary for a cooldown after repeated failures, then lets one request probe it (half-open). Fallbacks are
counted by reason in `llm_fallback_total`.

Measured on the laptop (Apple M3, 8 GB; Llama 3.2 3B Q4 via Ollama, which ran on CPU under memory pressure):

| | |
|---|---|
| Time to first token (p50) | 1.67 s |
| Decode throughput | 9.1 tokens/s |
| Gateway end-to-end, self-hosted primary (p50, ~70-token answers) | 9.2 s, 6/6 served by the self-hosted model |
| Outage drill: primary on a dead port | 6/6 answered by the fallback; 3 fell back on `error`, then the breaker opened and 3 skipped the primary (`breaker_open`) |

The drill surfaced a real bug that is now fixed and tested: when the fallback also failed, the gateway returned a raw 500
instead of a clean 503. A GPU node running vLLM (continuous batching, prefix caching) is the production target in
`k8s/overlays/production/vllm.yaml`; its throughput has not been measured in this update.

### 3. Observability

- `GET /metrics` (Prometheus): request latency histograms per route template, retrieval stage latency (bm25 / dense / rerank),
  evidence-cache hit/miss, LLM latency per provider, fallbacks by reason.
- OpenTelemetry tracing turns on when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (FastAPI spans plus an `evidence.search` span).
- `docker compose -f docker-compose.observability.yml up -d` starts Prometheus, Grafana (dashboard provisioned from
  `observability/grafana/dashboards/api.json`: P50/P95/P99, throughput, error rate, stage latency, cache hit rate, LLM latency) and Jaeger.

### 4. Load testing

`benchmarks/load/run_scenarios.sh` runs Locust against the evidence endpoint (fresh queries unless noted; 45 s per scenario;
one uvicorn worker, 4 CPU threads, Locust on the same laptop):

| Scenario | Users | Req/s | p50 | p95 | p99 | Failures |
|---|---|---|---|---|---|---|
| Rerank depth 30 (before) | 10 | 1.8 | 4.90 s | 7.40 s | 10.00 s | 0 |
| Rerank depth 30 (before) | 25 | 0.5 | 34.00 s | 41.00 s | 41.00 s | 0 |
| Rerank depth 10 (after) | 10 | 8.9 | 0.96 s | 1.20 s | 1.50 s | 0 |
| Rerank depth 10 (after) | 25 | 7.6 | 3.00 s | 3.90 s | 4.30 s | 0 |
| Depth 10, 50% repeated queries (cache) | 25 | 13.1 | 2.60 s | 4.00 s | 5.60 s | 0 |

**Before → change → after.** At 10 users, cutting rerank depth from 30 to 10 raised throughput **4.9x (1.8 → 8.9 req/s)** and cut
p95 **84% (7.4 s → 1.2 s)**; at 25 users depth 30 collapsed while depth 10 held 7.6 req/s. With half the queries repeating,
the content-keyed cache lifts throughput another 72% (7.6 → 13.1 req/s). A Prometheus scrape of a 10-user run confirmed the
bottleneck by stage: p95 BM25 5 ms, dense 238 ms, rerank 993 ms. Next step: batch reranking across concurrent requests or run
the cross-encoder on a GPU / as an INT8 ONNX model.

### 5. Deployment

- `k8s/base`: API, model gateway and Redis Deployments and Services, startup/readiness/liveness probes, CPU and memory
  requests/limits, non-root containers, ConfigMaps, optional Secret references, an HPA (70% CPU) and an Ingress.
  `k8s/overlays/local` targets kind; `k8s/overlays/production` pulls from ECR, runs 2–10 API replicas and adds a vLLM
  Deployment on a GPU node pool.
- `infra/terraform`: VPC (2 AZs, single NAT for cost), EKS with a general node group and a GPU node group that defaults
  to zero nodes, and an immutable, scanned ECR repository.

Validated in this update: `kustomize build` for both overlays passes `kubeconform -strict` against the Kubernetes 1.31
schemas (11 and 13 resources), and `terraform validate` passes. Both checks now run in CI (`infrastructure` job).
Nothing was applied to a cloud account, so no EKS cost was incurred; the GPU node group defaults to 0 nodes.

### Honest scope

Retrieval relevance is mention-based, not pharmacist-judged. Latencies are CPU numbers from one Apple M3 laptop.
The EKS/vLLM path is written and validated but was not applied to a paid AWS account in this update.

## Sources and intended use

- [RxNorm API and non-proprietary terminology terms](https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html)
- [DailyMed API](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm) and [label query parameters](https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm)
- [FDA information about drug interactions](https://www.fda.gov/drugs/resources-drugs/drug-interactions-what-you-should-know)
- [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.

Public access to a working website does not establish clinical effectiveness, comprehensive coverage, regulatory compliance or medical-device authorization. Actual treatment decisions require qualified professional review and appropriate validation of the intended workflow.
