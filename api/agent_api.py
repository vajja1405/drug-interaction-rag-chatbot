"""
Clinical agent API: start a medication review, then record the pharmacist's decision on escalations.

POST /api/v4/agent/reviews                     run the graph; PASS → released, ESCALATE → awaiting_review
POST /api/v4/agent/reviews/{case_id}/decision  resume an escalated case from its checkpoint
GET  /api/v4/agent/reviews/{case_id}           current status of a case
GET  /api/v4/agent/audit/verify                check the hash chain of the decision log
"""
from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/api/v4/agent", tags=["Clinical agent"])
_lock = threading.Lock()


class ReviewRequest(BaseModel):
    request: str = Field("", max_length=2000, description="Free-text clinician request")
    medications: list[str] = Field(default_factory=list, max_length=12)
    patient_id: str | None = Field(None, pattern=r"^P-\d{3,6}$")


class Decision(BaseModel):
    decision: Literal["approve", "override", "reject"]
    reviewer: str = Field(min_length=1, max_length=80)
    note: str = Field("", max_length=1000)


def get_agent(request: Request):
    """Build the graph once per process: tools, memory, audit log, SQLite checkpoints, model client."""
    st = request.app.state
    if getattr(st, "agent", None) is not None:
        return st.agent
    with _lock:
        if getattr(st, "agent", None) is None:
            from langgraph.checkpoint.sqlite import SqliteSaver
            from medication_review.catalog import Catalog
            from agent.audit import DecisionLog
            from agent.graph import AgentConfig, build_agent
            from agent.llm import OllamaClient, OpenAICompatible
            from agent.memory import PatientMemory
            from agent.tools import ToolRegistry
            from api.evidence import get_searcher

            data = Path(os.environ.get("AGENT_DATA_DIR", "data/agent"))
            data.mkdir(parents=True, exist_ok=True)
            backend = os.environ.get("AGENT_LLM", "ollama")
            if backend == "rules":
                llm, cfg = None, AgentConfig(planner="rules", proposer="rules", critic="rules")
            elif backend == "openai":  # any OpenAI-compatible endpoint, e.g. the model gateway or vLLM
                llm, cfg = OpenAICompatible(os.environ["AGENT_LLM_BASE_URL"], os.environ.get("AGENT_MODEL", "llama3.2"),
                                            os.environ.get("AGENT_LLM_API_KEY", "not-needed")), AgentConfig()
            else:
                llm, cfg = OllamaClient(), AgentConfig()
            st.agent_audit = DecisionLog(str(data / "audit.sqlite"))
            st.agent_tools = ToolRegistry(Catalog(), get_searcher(request), PatientMemory(str(data / "memory.sqlite")))
            saver = SqliteSaver(sqlite3.connect(str(data / "checkpoints.sqlite"), check_same_thread=False))
            st.agent = build_agent(st.agent_tools, llm, cfg, checkpointer=saver, audit=st.agent_audit)
    return st.agent


def _public(r: dict) -> dict:
    out = {"case_id": r.get("case_id") or r["result"]["case_id"], "status": r["status"]}
    if r["status"] == "awaiting_review":
        out["review_request"] = r["review_request"]
    else:
        out["result"] = r["result"]
    return out


@router.post("/reviews")
async def create_review(body: ReviewRequest, request: Request) -> dict:
    from agent.graph import start_case
    if not body.request and len(body.medications) < 1:
        raise HTTPException(422, "provide a request or a medication list")
    app = get_agent(request)
    case_id = f"case-{uuid.uuid4().hex[:12]}"
    r = await run_in_threadpool(start_case, app, case_id, body.request, body.medications, body.patient_id)
    return _public({**r, "case_id": case_id})


@router.post("/reviews/{case_id}/decision")
async def decide(case_id: str, body: Decision, request: Request) -> dict:
    from agent.graph import resume_case
    app = get_agent(request)
    snap = app.get_state({"configurable": {"thread_id": case_id}})
    if not snap.values:
        raise HTTPException(404, "unknown case")
    if "human_review" not in (snap.next or ()):
        raise HTTPException(409, "case is not awaiting review")
    r = await run_in_threadpool(resume_case, app, case_id, body.model_dump())
    return _public({**r, "case_id": case_id})


@router.get("/reviews/{case_id}")
async def get_review(case_id: str, request: Request) -> dict:
    app = get_agent(request)
    snap = app.get_state({"configurable": {"thread_id": case_id}})
    if not snap.values:
        raise HTTPException(404, "unknown case")
    if snap.next:
        return {"case_id": case_id, "status": "awaiting_review", "reasons": snap.values.get("reasons", []),
                "ticket": snap.values.get("ticket")}
    return {"case_id": case_id, "status": snap.values["final"]["status"], "result": snap.values["final"]}


@router.get("/audit/verify")
async def verify_audit(request: Request) -> dict:
    get_agent(request)
    log = request.app.state.agent_audit
    return {"entries": len(log.entries()), "valid": log.verify()}
