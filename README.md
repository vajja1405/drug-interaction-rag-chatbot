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

## Sources and intended use

- [RxNorm API and non-proprietary terminology terms](https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html)
- [DailyMed API](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm) and [label query parameters](https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm)
- [FDA information about drug interactions](https://www.fda.gov/drugs/resources-drugs/drug-interactions-what-you-should-know)
- [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.

Public access to a working website does not establish clinical effectiveness, comprehensive coverage, regulatory compliance or medical-device authorization. Actual treatment decisions require qualified professional review and appropriate validation of the intended workflow.
