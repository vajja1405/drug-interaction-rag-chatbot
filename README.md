# Drug Interaction AI

A React/TypeScript + FastAPI research prototype for drug-pair retrieval, severity classification, and LLM explanations.

**Public demo:** https://vajja1405.github.io/drug-interaction-rag-chatbot/

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

For Docker, build the index first, then `docker compose up --build`. The image includes the cache modules; it has not been certified for clinical production. GitHub Pages publishes only the static demo; Python hosting is still needed for a publicly accessible full backend. Never put API keys in Vite variables or browser bundles.

## Sources and limits

- [NLM RxNav FAQ](https://lhncbc.nlm.nih.gov/RxNav/information/FAQs.html)
- [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)
- [GitHub Pages hosting scope](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

This maintenance release improves engineering correctness. It does not retroactively establish historical deployment dates, avoided inference costs, independent clinical validation, or production usage.
