# Drug Interaction AI
### RAG-Based Clinical Decision Support System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-TypeScript-61DAFB?logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/FAISS-vector%20store-orange" />
  <img src="https://img.shields.io/badge/LangChain-LCEL-green" />
  <img src="https://img.shields.io/badge/Severity%20Accuracy-100%25-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A production-grade AI assistant that analyses drug-drug interactions (DDI) using Retrieval-Augmented Generation (RAG). The system retrieves evidence from a FAISS vector knowledge base, classifies interaction severity across a 4-tier hierarchy, and generates medically grounded explanations via an LLM — with a React/TypeScript frontend, FastAPI backend, Redis caching, and a full RAGAS-inspired evaluation framework.

---

## Evaluation Results (RAGAS-Inspired, 24 Drug Pairs)

> Evaluated against a golden test set of known severe, moderate, low, and out-of-distribution drug pairs.

| Metric | Score | Description |
|--------|-------|-------------|
| **Severity Accuracy** | **1.000 (100%)** | Predicted severity matches known clinical classification |
| **Mechanism Coverage** | **1.000 (100%)** | Expected pharmacokinetic keywords present in mechanism |
| **Management Coverage** | 0.979 (98%) | Expected management instructions covered |
| **Faithfulness** | 0.917 (92%) | LLM output grounded in retrieved context |
| **Context Precision** | 0.917 (92%) | Retrieved docs relevant to queried drug pair |
| **Citation Accuracy** | 0.917 (92%) | Cited source matches expected reference |

Run evaluation: `python -m evaluation.run_eval --verbose`

---

## Why This Project Stands Out

| Signal | What it demonstrates |
|--------|----------------------|
| **Full-stack delivery** | React/TypeScript frontend + FastAPI backend + FAISS vector store + Redis caching |
| **Production RAG** | Polypharmacy-aware retrieval — all C(N,2) pairs; 5 drugs → 10 pairs in ~10ms |
| **4-tier classifier** | Structured (RxNorm) → keyword → RandomForest (TF-IDF) → LLM fallback; prevents hallucinated severity |
| **100x cache speedup** | Benchmarked pair-level LRU / Redis cache; `python benchmark_cache.py` to reproduce |
| **RAGAS evaluation** | Custom eval framework with 6 metrics, 24 golden pairs — 100% severity accuracy |
| **LLM-agnostic** | OpenAI, local Llama3 via Ollama, any OpenAI-compatible endpoint — swap with one env var |
| **Dockerised** | `docker compose up --build` — API live at localhost:8000 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Drug Interaction AI                           │
│                                                                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│   │ Data Pipeline│    │  RAG Pipeline│    │    Chatbot Agent     │  │
│   │              │    │              │    │                      │  │
│   │ fetch_drug_  │───▶│ embedding_   │    │  DrugInteraction     │  │
│   │ data.py      │    │ model.py     │    │  Agent               │  │
│   │              │    │              │    │  ┌────────────────┐  │  │
│   │ preprocess_  │───▶│ vector_      │───▶│  │  LangChain     │  │  │
│   │ data.py      │    │ store.py     │    │  │  LCEL Chain    │  │  │
│   └──────────────┘    │              │    │  │                │  │  │
│                       │ retriever.py │    │  │  ChatOpenAI    │  │  │
│   ┌──────────────┐    └──────────────┘    │  └────────────────┘  │  │
│   │   Models     │           │            └──────────────────────┘  │
│   │              │           │                       │               │
│   │ interaction_ │◀──────────┘                       │               │
│   │ classifier   │                                   │               │
│   └──────────────┘                                   ▼               │
│                                         ┌──────────────────────┐    │
│                                         │     FastAPI Server   │    │
│                                         │  POST /api/v1/analyze│    │
│                                         └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Modules

| Module | File | Responsibility |
|--------|------|----------------|
| Data Pipeline | `data_pipeline/fetch_drug_data.py` | Fetch DDI data from RxNorm + OpenFDA APIs + curated seed |
| Data Pipeline | `data_pipeline/preprocess_data.py` | Normalise, deduplicate, build embeddable documents |
| RAG | `rag_pipeline/embedding_model.py` | Sentence-transformer embeddings (all-MiniLM-L6-v2) |
| RAG | `rag_pipeline/vector_store.py` | FAISS IndexFlatIP with O(1) pair lookup |
| RAG | `rag_pipeline/retriever.py` | Polypharmacy-aware retrieval (all C(N,2) pairs) |
| Models | `models/interaction_classifier.py` | 4-tier severity classification (structured → keyword → RandomForest → LLM fallback) |
| Chatbot | `chatbot/interaction_agent.py` | LangChain LCEL orchestration + LLM explanation |
| API | `api/server.py` | FastAPI server with rate limiting and health checks |
| CLI | `build_index.py` | One-shot pipeline to build the FAISS index |

---

## Usage & Testing

### 1. Clone and install dependencies

```bash
git clone https://github.com/vajja1405/drug-interaction-rag-chatbot.git
cd drug-interaction-rag-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Automated Test Suite

The comprehensive test suite uses `pytest` and runs entirely offline by mocking external LLM calls. It validates the agent pipeline, classifier logic, and vector retrieval.

```bash
pip install pytest httpx pytest-asyncio
pytest -q
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY (or configure a local LLM – see below)
# Optional: Set REDIS_URL for caching (e.g., redis://localhost:6379)
```

### 4. Build the knowledge base index

```bash
# Uses curated seed data only (no API calls required, works offline)
python build_index.py --all-seed

# OR fetch additional live data from RxNorm for specific drugs
python build_index.py --drugs warfarin aspirin metformin ibuprofen lisinopril
```

This pulls from curated data. You can expand the knowledge base by grabbing raw narrative labels directly from the OpenFDA dataset and merging them with the API data:

```bash
# Fetch interaction warnings from OpenFDA
python -m data_pipeline.openfda_labels --limit 100

# Preprocess and merge datasets
python -m data_pipeline.preprocess_data --in data/raw/interactions_raw.jsonl data/raw/openfda_raw.jsonl

# Rebuild index
python build_index.py --processed-only
```

This creates:
```
data/
  raw/interactions_raw.jsonl          ← fetched records
  processed/interactions.jsonl        ← cleaned documents
  vectorstore/
    drug_interactions.faiss           ← FAISS binary index
    metadata.pkl                      ← parallel metadata
```

### 5. Start the API server

```bash
uvicorn api.server:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### 6. Caching Benchmark

Pair-level LLM results and FAISS retrieval outputs are cached. To test the >100x latency acceleration locally (using either Redis or in-memory LRU):

```bash
python benchmark_cache.py
```

---

## Docker Quick Start

Run the entire backend with one command using Docker Compose.

### 1. Prepare

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# Build the FAISS index first (on the host)
python build_index.py --all-seed
```

### 2. Start

```bash
# Backend only (recommended)
docker compose up --build

# With Redis (for future caching)
docker compose --profile with-redis up --build
```

The API will be available at [http://localhost:8000](http://localhost:8000).

### 3. Test

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Analyze drug interactions
curl -X POST http://localhost:8000/analyze_interaction \
  -H "Content-Type: application/json" \
  -d '{"drugs": ["warfarin", "ibuprofen", "aspirin"]}'
```

### 4. Stop

```bash
docker compose down
```

> **Note:** The `data/` directory is bind-mounted into the container. Any changes
> to the FAISS index on the host are immediately available inside the container.

---

## API Usage

### POST `/api/v1/analyze`

Analyse drug-drug interactions for a list of drugs.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"drugs": ["warfarin", "aspirin", "metformin"]}'
```

**Response:**
```json
{
  "request_id": "a3f7c2d1-...",
  "drugs_analyzed": ["warfarin", "aspirin", "metformin"],
  "pairs_checked": ["aspirin+warfarin", "aspirin+metformin", "metformin+warfarin"],
  "interactions": [
    {
      "drug_pair": ["aspirin", "warfarin"],
      "severity": "Severe",
      "confidence": 0.95,
      "classification_method": "structured",
      "description": "Significantly increased risk of serious or fatal bleeding...",
      "matched_keywords": []
    },
    {
      "drug_pair": ["aspirin", "metformin"],
      "severity": "Low",
      "confidence": 0.95,
      "classification_method": "structured",
      "description": "No clinically significant interaction at standard aspirin doses.",
      "matched_keywords": []
    },
    {
      "drug_pair": ["metformin", "warfarin"],
      "severity": "Low",
      "confidence": 0.95,
      "classification_method": "structured",
      "description": "No direct pharmacokinetic interaction.",
      "matched_keywords": []
    }
  ],
  "overall_risk": "Severe",
  "explanation": "## Aspirin + Warfarin\n**Mechanism:** ...\n**Clinical Effects:** ...",
  "retrieved_docs": 3,
  "processing_time_ms": 843.2,
  "disclaimer": "This analysis is for informational purposes only..."
}
```

### GET `/api/v1/health`
```bash
curl http://localhost:8000/api/v1/health
```

### GET `/api/v1/stats`
```bash
curl http://localhost:8000/api/v1/stats
```

### GET `/api/v1/drugs/search?q=warfa`
```bash
curl "http://localhost:8000/api/v1/drugs/search?q=warfa"
```

---

## Using a Local LLM (Ollama)

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3

# In .env:
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=llama3
```

No other code changes are needed – the system uses OpenAI's client interface.

---

## Drug Pair Analysis Examples

| Drug Pair | Severity | Mechanism |
|-----------|----------|-----------|
| Warfarin + Aspirin | **Severe** | Additive anticoagulant/antiplatelet + protein-binding displacement |
| Warfarin + Ibuprofen | **Severe** | COX-1 inhibition + protein-binding displacement |
| Simvastatin + Amiodarone | **Severe** | CYP3A4 inhibition → rhabdomyolysis risk |
| Digoxin + Amiodarone | **Severe** | P-gp inhibition → digoxin toxicity |
| SSRI + Tramadol | **Severe** | Serotonin syndrome risk |
| Methotrexate + Ibuprofen | **Severe** | Reduced renal clearance of methotrexate |
| Aspirin + Ibuprofen | **Moderate** | Competitive COX-1 binding → reduced aspirin antiplatelet effect |
| Lisinopril + Ibuprofen | **Moderate** | Reduced antihypertensive effect + AKI risk |
| Clopidogrel + Omeprazole | **Moderate** | CYP2C19 inhibition → reduced clopidogrel activation |
| Ibuprofen + Metformin | **Moderate** | Reduced renal clearance → lactic acidosis risk |
| Aspirin + Metformin | **Low** | No significant interaction at standard doses |
| Warfarin + Metformin | **Low** | No significant interaction |

---

## Project Structure

```
DrugInteractionAI/
├── data_pipeline/
│   ├── __init__.py
│   ├── fetch_drug_data.py      # Fetch from RxNorm, OpenFDA, curated seed
│   └── preprocess_data.py      # Normalise, deduplicate, build documents
├── rag_pipeline/
│   ├── __init__.py
│   ├── embedding_model.py      # Sentence-transformer wrapper
│   ├── vector_store.py         # FAISS index + O(1) pair lookup
│   └── retriever.py            # Polypharmacy-aware retrieval
├── models/
│   ├── __init__.py
│   └── interaction_classifier.py   # 4-tier severity classification
├── chatbot/
│   ├── __init__.py
│   └── interaction_agent.py    # LangChain LCEL orchestration
├── api/
│   ├── __init__.py
│   └── server.py               # FastAPI server
├── data/
│   ├── raw/                    # Fetched JSONL records
│   ├── processed/              # Cleaned documents
│   └── vectorstore/            # FAISS index files
├── frontend/                   # React + Vite UI (see frontend/README.md)
├── build_index.py              # One-shot index builder
├── config.py                   # Pydantic settings
├── requirements.txt
├── Dockerfile                  # Backend container image
├── docker-compose.yml          # Multi-service orchestration
├── .dockerignore
├── .env.example
└── README.md
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | – | API key for OpenAI or compatible endpoint |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint (OpenAI-compatible) |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RAG_TOP_K` | `5` | Documents to retrieve per pair |
| `SIMILARITY_THRESHOLD` | `0.55` | Minimum cosine similarity for results |
| `MAX_DRUGS_PER_REQUEST` | `20` | Maximum drugs per API request |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests per minute per IP |

---

## Technical Notes

**Severity Classification Hierarchy**
1. Structured data (RxNorm/curated) → confidence 0.95
2. Keyword-based rules on document text → confidence 0.60–0.95
3. ML classifier — `RandomForestSeverityClassifier` (TF-IDF + engineered features,
   wrapped in `CalibratedClassifierCV` with Platt scaling and validated via
   `StratifiedKFold` out-of-fold predictions, so the emitted confidence is calibrated
   rather than a raw vote share)
4. LLM-based reasoning (in explanation) → narrative only, not structured

**Why structured > LLM for severity?**
LLMs can hallucinate severity classifications. Using structured sources (RxNorm severity codes, curated clinical data) as primary classifiers prevents hallucinated life-threatening misclassifications.

**Polypharmacy scaling**
- 5 drugs → 10 pairs, ~10 ms retrieval
- 10 drugs → 45 pairs, ~45 ms retrieval
- 20 drugs → 190 pairs, ~200 ms retrieval (within rate limit per request)

---

## RAG Evaluation Framework

A comprehensive evaluation module measures retrieval and generation quality against a golden test set of 24 drug pairs. Metrics are inspired by [RAGAS](https://docs.ragas.io/), Amazon Bedrock Eval, and Anthropic's evaluation practices.

### Running the evaluation

```bash
python -m evaluation.run_eval --verbose
```

### Metrics

| Metric | Description | Inspired By |
|--------|-------------|-------------|
| **Severity Accuracy** | Does predicted severity match expected? | Custom |
| **Context Precision** | Do retrieved docs mention the queried pair? | RAGAS |
| **Faithfulness** | Are LLM-ready fields grounded in context? | RAGAS |
| **Mechanism Coverage** | Do expected keywords appear in mechanism? | Amazon |
| **Management Coverage** | Do expected keywords appear in management? | Amazon |
| **Citation Accuracy** | Does cited source match expected? | Amazon |

### Golden test set

Located at `evaluation/golden_set.json` — 24 drug pairs covering:
- Known severe interactions (warfarin + aspirin, digoxin + amiodarone)
- Known moderate interactions (ibuprofen + metformin, clopidogrel + omeprazole)
- Known low interactions (warfarin + metformin)
- Out-of-distribution pairs (atorvastatin + warfarin — expected Unknown)

Results are saved to `evaluation/results.json`.

---

## Architecture Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for a comprehensive breakdown of:
- System overview with Mermaid data flow diagram
- Module-by-module reference (every file, class, function)
- Vector store design and FAISS index details
- Confidence score calculation formulas for all 4 classification tiers
- Dual-classifier consensus reconciliation logic

---

## Disclaimer

> **This system is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider before making any clinical decisions.**
