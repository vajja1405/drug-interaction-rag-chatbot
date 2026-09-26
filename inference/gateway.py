"""
OpenAI-compatible gateway. Point the API at it with OPENAI_BASE_URL=http://gateway:9000/v1.

    PRIMARY_BASE_URL=http://vllm:8000/v1   PRIMARY_MODEL=Qwen/Qwen2.5-3B-Instruct
    FALLBACK_BASE_URL=https://api.openai.com/v1 FALLBACK_MODEL=gpt-4o-mini FALLBACK_API_KEY=...
    uvicorn inference.gateway:app --port 9000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from inference.router import ModelRouter, RouteError
from observability import metrics_endpoint, metrics_middleware

app = FastAPI(title="Model gateway", version="1.0.0")
app.middleware("http")(metrics_middleware)
app.add_route("/metrics", metrics_endpoint, include_in_schema=False)


def _router(request: Request) -> ModelRouter:
    r = getattr(request.app.state, "router", None)
    if r is None:
        r = request.app.state.router = ModelRouter.from_env()
    return r


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    if body.get("stream"):
        raise HTTPException(400, "streaming is not proxied by this gateway; call the provider directly")
    try:
        return await run_in_threadpool(_router(request).complete, body)
    except RouteError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
