"""
api/server.py
──────────────
FastAPI server for the Drug Interaction Analysis system.

Endpoints
─────────
  POST /analyze_interaction   – Main drug interaction analysis  ← primary
  POST /api/v1/analyze        – Same endpoint, versioned alias
  GET  /api/v1/health         – Health check with vector-store status
  GET  /api/v1/drugs/search   – Drug name autocomplete (RxNorm)
  GET  /api/v1/stats          – Index and model statistics
  GET  /                      – Welcome / documentation redirect

Startup
───────
  On server start the lifespan context manager loads once:
    - EmbeddingPipeline  (sentence-transformers, all-MiniLM-L6-v2)
    - DrugVectorStore    (FAISS index from disk)
    - DrugInteractionRetriever
    - InteractionClassifier
    - DrugInteractionAgent

  If the vector-store files are missing the health endpoint reports
  "unhealthy" and all analysis endpoints return 503.
  Run  python build_index.py  first.

Rate limiting
─────────────
  30 requests / minute per IP (set RATE_LIMIT_PER_MINUTE in .env).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from chatbot.interaction_agent import DrugInteractionAgent
from models.interaction_classifier import InteractionClassifier
from rag_pipeline.embeddings import get_pipeline
from rag_pipeline.retriever import DrugInteractionRetriever
from rag_pipeline.vector_store import DrugVectorStore
from config import settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan: load all heavy resources once at startup ────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indexes on startup; release on shutdown."""
    logger.info("Starting up Drug Interaction AI server…")
    app.state.healthy = False
    app.state.startup_error = None

    try:
        logger.info("Loading embedding pipeline…")
        app.state.embedding_pipeline = get_pipeline()

        logger.info("Loading vector store…")
        app.state.vector_store = DrugVectorStore.load(settings.vector_store_path)

        app.state.retriever = DrugInteractionRetriever(
            vector_store=app.state.vector_store,
            embedding_model=app.state.embedding_pipeline,
        )

        app.state.classifier = InteractionClassifier()

        app.state.agent = DrugInteractionAgent(
            retriever=app.state.retriever,
            classifier=app.state.classifier,
        )

        app.state.healthy = True
        logger.info(
            "Server ready – %d interaction documents loaded",
            app.state.vector_store.total_vectors,
        )

    except FileNotFoundError as exc:
        app.state.startup_error = str(exc)
        logger.error("Vector store not found – run: python build_index.py\n%s", exc)
    except Exception as exc:
        app.state.startup_error = str(exc)
        logger.exception("Startup failed: %s", exc)

    yield

    logger.info("Shutting down Drug Interaction AI server")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Drug Interaction Analysis API",
    description=(
        "RAG-based clinical decision support for drug-drug interaction analysis.\n\n"
        "**Quick start:** `POST /analyze_interaction` with a JSON body containing a `drugs` list.\n\n"
        "Provides evidence-based severity classification (Severe / Moderate / Low) and "
        "LLM-generated clinical explanations for all drug pair combinations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Pydantic models ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request body for drug interaction analysis."""

    drugs: list[str] = Field(
        min_length=2,
        description="List of drug names to analyse (2–20 drugs).",
        examples=[["warfarin", "ibuprofen", "aspirin"]],
    )
    include_low_severity: bool = Field(
        default=True,
        description="Set to false to suppress Low/Unknown pairs in the response.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"drugs": ["warfarin", "ibuprofen", "aspirin"]}
        }
    }


class DrugInteraction(BaseModel):
    """Result for a single drug pair."""

    pair: list[str] = Field(description="The two drug names in this pair.")
    severity: str = Field(description="Severe | Moderate | Low | Unknown")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier confidence (0–1).")
    classification_method: str = Field(
        description="How severity was determined: structured | keyword | ml_random_forest | default"
    )
    mechanism: str = Field(description="Pharmacological mechanism of the interaction.")
    clinical_effects: str = Field(description="Expected clinical consequences.")
    management: str = Field(description="Recommended clinical management.")
    monitoring: list[str] = Field(description="Specific parameters to monitor.")
    interaction_type: str = Field(
        default="",
        description="pharmacodynamic | pharmacokinetic | unknown",
    )
    source_name: str = Field(default="", description="Data source name (e.g. RxNorm, OpenFDA, Curated).")
    source_url: str = Field(default="", description="Direct URL to the source document.")
    source_page: str = Field(default="", description="Source page or section description.")


class AnalyzeResponse(BaseModel):
    """Full response from /analyze_interaction."""

    request_id: str = Field(description="Unique ID for this request (for support/logging).")
    drugs: list[str] = Field(description="Normalised list of drugs that were analysed.")
    pairs_checked: list[str] = Field(description="All pair combinations that were evaluated.")
    overall_risk: str = Field(description="Highest severity across all pairs.")
    interactions: list[DrugInteraction] = Field(
        description="Per-pair interaction details, sorted Severe → Moderate → Low."
    )
    explanation: str = Field(description="LLM-generated clinical narrative summary.")
    monitoring_priorities: list[str] = Field(
        description="Deduplicated list of the most important monitoring actions."
    )
    retrieved_docs: int = Field(description="Total evidence documents retrieved from the knowledge base.")
    processing_time_ms: float = Field(description="Server-side processing time in milliseconds.")
    disclaimer: str = Field(
        default=(
            "This analysis is for informational purposes only. "
            "Consult a qualified healthcare provider before making any clinical decisions."
        )
    )


class HealthResponse(BaseModel):
    status: str
    vector_store_documents: int
    model: str
    error: str | None


class StatsResponse(BaseModel):
    total_documents: int
    unique_pairs: int
    index_type: str
    embedding_model: str
    llm_model: str
    llm_base_url: str


# ── Shared helpers ────────────────────────────────────────────────────────────

_SEVERITY_ORDER = {"Severe": 0, "Moderate": 1, "Low": 2, "Unknown": 3}


def _require_healthy(request: Request) -> None:
    if not request.app.state.healthy:
        err = request.app.state.startup_error or "Server not ready"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Service unavailable: {err}. "
                "Build the vector store first (python build_index.py) and restart."
            ),
        )


from cache import cache

async def _run_analysis(request: Request, drugs_raw: list[str], include_low: bool) -> AnalyzeResponse:
    """
    Core analysis logic shared by both /analyze_interaction and /api/v1/analyze.
    Validates input, calls the agent, and maps the InteractionReport to the
    response model.
    """
    _require_healthy(request)

    if len(drugs_raw) > settings.max_drugs_per_request:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Too many drugs: {len(drugs_raw)}. "
                f"Maximum allowed: {settings.max_drugs_per_request}."
            ),
        )

    drugs = [d.strip().lower() for d in drugs_raw if d.strip()]
    if len(set(drugs)) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least 2 distinct drug names are required.",
        )

    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()
    logger.info("[%s] Analyzing: %s", request_id, drugs)
    
    # Caching path for pair-level evaluations (exactly 2 drugs)
    cache_key = None
    if len(drugs) == 2:
        sorted_pair = "+".join(sorted(drugs))
        index_docs = request.app.state.vector_store.stats()["total_documents"]
        cache_key = f"ddi_v1:{settings.llm_model}:{index_docs}:{sorted_pair}"
        
        cached_data = await cache.get(cache_key)
        if cached_data:
            logger.info("[%s] Cache HIT: %s", request_id, sorted_pair)
            return AnalyzeResponse(**cached_data)

    try:
        report = request.app.state.agent.analyze(drugs)
    except Exception as exc:
        logger.exception("[%s] Analysis failed: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        ) from exc

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    raw = report.to_dict()

    # Build per-pair interaction objects
    interactions: list[DrugInteraction] = []
    for item in raw["interactions"]:
        sev = item["severity"]
        if not include_low and sev in ("Low", "Unknown"):
            continue
        interactions.append(
            DrugInteraction(
                pair=item["drug_pair"],
                severity=sev,
                confidence=item["confidence"],
                classification_method=item["classification_method"],
                mechanism=item.get("mechanism") or "",
                clinical_effects=item.get("clinical_effects") or "",
                management=item.get("management") or "Consult a healthcare provider.",
                monitoring=item.get("monitoring_points") or [],
                interaction_type=item.get("interaction_type") or "",
                source_name=item.get("source_name") or "",
                source_url=item.get("source_url") or "",
                source_page=item.get("source_page") or "",
            )
        )

    # Sort by severity
    interactions.sort(key=lambda i: _SEVERITY_ORDER.get(i.severity, 99))

    response = AnalyzeResponse(
        request_id=request_id,
        drugs=raw["drugs_analyzed"],
        pairs_checked=raw["pairs_checked"],
        overall_risk=raw["overall_risk"],
        interactions=interactions,
        explanation=raw["narrative"],
        monitoring_priorities=raw.get("monitoring_priorities", []),
        retrieved_docs=raw["retrieved_docs_total"],
        processing_time_ms=round(elapsed_ms, 1),
    )

    logger.info(
        "[%s] Done in %.0f ms – %d pairs, risk=%s",
        request_id, elapsed_ms, len(raw["pairs_checked"]), raw["overall_risk"],
    )
    
    if cache_key:
        await cache.set(cache_key, response.model_dump(mode="json"))
        
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Drug Interaction Analysis API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "analyze": "POST /analyze_interaction",
    }


# ── Primary endpoint ──────────────────────────────────────────────────────────

@app.post(
    "/analyze_interaction",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyse drug-drug interactions",
    description=(
        "Accepts a list of drug names and returns severity-classified interactions "
        "for every pair combination, backed by a FAISS knowledge base and an LLM "
        "clinical explanation.\n\n"
        "**Example request:**\n"
        "```json\n"
        '{"drugs": ["warfarin", "ibuprofen", "aspirin"]}\n'
        "```"
    ),
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def analyze_interaction(request: Request, body: AnalyzeRequest):
    return await _run_analysis(request, body.drugs, body.include_low_severity)


# ── Versioned alias (keeps backward-compat with any existing integrations) ────

@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyse drug-drug interactions (versioned alias)",
    include_in_schema=True,
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def analyze_interaction_v1(request: Request, body: AnalyzeRequest):
    return await _run_analysis(request, body.drugs, body.include_low_severity)


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request):
    """Returns server health and vector-store document count."""
    if request.app.state.healthy:
        return HealthResponse(
            status="healthy",
            vector_store_documents=request.app.state.vector_store.total_vectors,
            model=settings.llm_model,
            error=None,
        )
    return HealthResponse(
        status="unhealthy",
        vector_store_documents=0,
        model=settings.llm_model,
        error=request.app.state.startup_error,
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["System"])
async def stats(request: Request):
    """Returns index and model statistics."""
    _require_healthy(request)
    store_stats = request.app.state.vector_store.stats()
    return StatsResponse(
        total_documents=store_stats["total_documents"],
        unique_pairs=store_stats["unique_pairs"],
        index_type=store_stats["index_type"],
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        llm_base_url=settings.openai_base_url,
    )


@app.get("/api/v1/drugs/search", tags=["Drugs"])
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def search_drugs(request: Request, q: str):
    """
    Autocomplete drug names via the public RxNorm API.
    Returns up to 10 matching drug names.  Minimum query length: 2 characters.
    """
    if len(q) < 2:
        return {"suggestions": []}
    url = f"{settings.rxnorm_base_url}/spellingsuggestions.json"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params={"name": q})
            resp.raise_for_status()
            data = resp.json()
            suggestions = (
                data.get("suggestionGroup", {})
                    .get("suggestionList", {})
                    .get("suggestion", [])
            )
            return {"suggestions": suggestions[:10]}
    except Exception as exc:
        logger.warning("Drug search failed for %r: %s", q, exc)
        return {"suggestions": [], "error": "Drug name lookup service unavailable"}


# ── Direct run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
