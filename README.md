# Drug Interaction AI

A React/TypeScript + FastAPI research prototype for drug-pair retrieval, severity classification, and LLM explanations.

**Full public application:** https://astra6-drug-interaction-ai.hf.space

[Hugging Face Space](https://huggingface.co/spaces/astra6/drug-interaction-ai) runs React + FastAPI, embeddings, FAISS retrieval and the classifier on CPU Basic. Explanations use `Qwen/Qwen3-4B-Instruct-2507:nscale` through Hugging Face Inference Providers, with the API key stored as a server-side Space secret. Visitors need no account. Medication names are sent to the server and model provider; use demonstration inputs only.

**Browser-only backup:** https://vajja1405.github.io/drug-interaction-rag-chatbot/

The deployed GitHub Pages site performs deterministic lookup of 35 bundled research fixtures entirely in the browser. It requires no account or API key. It does **not** host FastAPI, run the trained classifier, or call an LLM. Unsupported or conflicting pairs return **Unknown**, never an assurance of safety. Public fixtures are incomplete and not clinically adjudicated; do not use this prototype for treatment decisions. It supplies no personalized management or dosing recommendations.

## Full Python system

The backend retains the original retrieval/classification/explanation architecture:

```text
Names → pair expansion → FAISS retrieval → evidence/classification stages
      → LLM explanation on ordinary uncached API requests → response
      ↕ versioned, expiring pair cache (Redis or local LRU)
```

The classifier module, 24-pair evaluator, and API have different execution paths. A severity cascade does not imply that the ordinary API avoids an explanation-model call. The random-forest severity model uses `CalibratedClassifierCV(method="isotonic", cv=3)` around the **entire** feature-fitting pipeline. This avoids fitting TF-IDF on calibration-fold examples. Calibration still requires independent reliability evaluation; it is not a clinical guarantee.

RxNav retired interaction features on January 2, 2024. The old interaction helper now returns no records without calling that endpoint. RxNorm remains a terminology service; optional OpenFDA label helpers and curated fixtures have distinct provenance. A generic literature-search URL is not clinician adjudication of a fixture.

## Verification, September 11, 2026

- **24/24 severity labels matched** in the saved component regression. This is agreement with a small curated answer key, not population accuracy or held-out clinical validation.
- Generated-answer faithfulness is **not evaluated**. The old self-comparison has been removed and its denominator is zero. Custom keyword/citation checks are not official RAGAS metrics.
- Saved results, scope, denominators, and individual pairs: [validation artifact](docs/validation-2026-09-11.json).
- Regression tests cover JSON-body parsing, distinct cache-hit request IDs, evidence/settings/filter cache invalidation, TTL/eviction, no retired API calls, and feature fitting inside calibration folds.
- The public demo has six Node tests for supported pairs, normalization, missing/conflicting evidence, bounds, and source preservation.
- Local Python testing requires model artifacts/downloads for integration tests. LLM calls are mocked; this does not test a live model's faithfulness. Spark requires PySpark and Java; CI provisions both.

Cache hits have their own request ID and elapsed time. Cache keys include evidence content, pipeline version, model settings and the low-severity filter; entries expire after one hour. Restart the API after replacing its knowledge base. Faster hits do not establish proportional total-cost savings.

## Run locally

Use Python 3.11 and Node 22. Install Java 17 if running Spark.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY privately in .env for the full explanation path.
python build_index.py --all-seed
uvicorn api.server:app --reload --port 8000
```

Index creation uses curated fixtures. The embedding model may download on first use. In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

The development proxy connects React to port 8000. API docs: http://localhost:8000/docs . The public Pages build is a separate mode:

```bash
cd frontend
npm test
VITE_DEMO_MODE=true npm run build -- --base=/drug-interaction-rag-chatbot/
```

```bash
python -m pytest tests/ -v
python -m evaluation.run_eval --verbose
```

For Docker, run `docker compose up --build`; the image build creates its own index. The image includes the React build, fixture index, embedding weights, calibrated classifier, and cache modules. It runs as a non-root user and does not need host data mounts. It has not been certified for clinical production. GitHub Pages publishes only the static demo; the Hugging Face Space hosts the full Python backend. Never put API keys in Vite variables or browser bundles.

## Sources and limits

- [NLM RxNav FAQ](https://lhncbc.nlm.nih.gov/RxNav/information/FAQs.html)
- [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)
- [GitHub Pages hosting scope](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

This maintenance release improves engineering correctness. It does not retroactively establish historical deployment dates, avoided inference costs, independent clinical validation, or production usage.


## Full backend deployment

The Docker image serves React and FastAPI on the same origin (`PORT`, default 7860). Build with `docker build -t drug-interaction-ai .`. Supply a real compatible model service through runtime secrets/settings: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`. A local Ollama URL pointing at localhost does not become a cloud model service when the image is uploaded.

`deployment/HF_README.md` provides Hugging Face Docker metadata. `deployment/render.yaml` is a reviewable Render blueprint with a paid plan; neither file creates an account or starts billing. See the provider's current prices before applying. Do not upload `.env` or reuse a broad GitHub token as a model credential.

Docker defaults: five medications, five requests/minute per connection IP, one concurrent analysis, 200 uncached analyses per process per UTC day, 1,800 output tokens and a 25-second timeout per model attempt with no SDK retries. The ordinary JSON call can fall back to one text call. The daily counter resets on process restart and is **not** a provider billing cap. Use provider-side budgets as well. Proxy headers are not trusted by default; an upstream shared proxy may make the IP limit apply globally. Configure trusted proxies only after verifying the host topology.

`/api/v1/ready` returns 503 when resources or model-key configuration are missing. A ready response does not claim live model verification. Responses explicitly distinguish generated JSON, generated text and model-unavailable evidence fallback. Failed generations are not cached. Unknown-evidence pairs remain visible even when Low pairs are filtered out. Integration tests cover those behaviors. Clinical validation and generated-claim adjudication remain outstanding.


### Hosted verification and operating limits

On September 11, the public Space returned generated JSON for one supported pair, served its repeat from cache with a distinct request ID, retained an unsupported pair as Unknown with Low results filtered out, and served readiness while inference was running. React example loading, analysis and expanded evidence were also checked in the public browser. [Smoke-check record](docs/hosting-smoke-2026-09-11.json). These are deployment checks, not adjudicated medical-answer evaluation or a latency benchmark. GitHub integration CI passed 28 tests, including Spark.

The Space overrides the daily allowance to **50 uncached analyses per process per UTC day** and the timeout to **40 seconds per model attempt**. Other bounds remain five medications, five requests per minute per observed client IP, one concurrent uncached analysis and 1,800 output tokens. CPU Basic was selected; no paid GPU or storage upgrade was enabled. Inference uses the account's provider credits and can fail when credit or provider availability is exhausted. A process-local throttle does not guarantee a spending cap. An idle Space can require startup time; keep the browser-only backup available.
