"""
MCP server exposing the agent's tools to any MCP client (Claude Desktop, IDE agents, other services).

    python -m agent.mcp_server            # stdio transport

Read tools and the review workflow are exposed; the patient-memory write (record_medications) is not.
Memory only changes inside the graph, after the judge passes a case or a pharmacist approves it.
"""
from __future__ import annotations

import uuid

from mcp.server.fastmcp import FastMCP

from agent import policy
from agent.graph import start_case
from agent.tools import ToolRegistry

POLICY_TEXT = policy.__doc__ or ""


def create_server(tools: ToolRegistry, app=None) -> FastMCP:
    mcp = FastMCP("clinical-evidence")

    def run(tool: str, **args) -> dict:
        out, step = tools.call(tool, node="mcp", **args)
        if out is None:
            raise ValueError(step["error"] or f"{tool} failed")
        return out

    @mcp.tool()
    def resolve_medication(name: str) -> dict:
        """Map a medication name, brand, or misspelling to an RxNorm ingredient (RxCUI)."""
        return run("resolve_medication", name=name)

    @mcp.tool()
    def search_label_evidence(drug_a: str, drug_b: str, expand: bool = False) -> dict:
        """FDA label passages from either drug's label that name the other drug or its class."""
        return run("search_label_evidence", drug_a=drug_a, drug_b=drug_b, expand=expand)

    @mcp.tool()
    def get_patient_context(patient_id: str) -> dict:
        """Current medication list and recent reviews for a patient (synthetic IDs like P-201)."""
        return run("get_patient_context", patient_id=patient_id)

    @mcp.tool()
    def request_human_review(case_id: str, reasons: list[str], priority: str = "routine") -> dict:
        """Open (or return the existing) pharmacist review ticket for a case."""
        return run("request_human_review", case_id=case_id, reasons=reasons, priority=priority)

    if app is not None:
        @mcp.tool()
        def review_medications(request: str, case_id: str = "") -> dict:
            """Run the full review graph. Escalated cases return status awaiting_review and a ticket."""
            r = start_case(app, case_id or f"mcp-{uuid.uuid4().hex[:8]}", request)
            if r["status"] == "awaiting_review":
                return {"status": r["status"], "case_id": r["case_id"], "review_request": r["review_request"]}
            return {"status": r["status"], "result": r["result"]}

    @mcp.resource("policy://escalation")
    def escalation_policy() -> str:
        """When the agent acts on its own and when it must hand off to a pharmacist."""
        return POLICY_TEXT

    return mcp


def default_server() -> FastMCP:
    import numpy as np
    from pathlib import Path
    from medication_review.catalog import Catalog
    from search.corpus import load
    from search.hybrid import default_searcher
    from agent.graph import build_agent
    from agent.llm import OllamaClient
    from agent.memory import PatientMemory

    passages = load()
    emb = Path("data/benchmark/passage_embeddings.npy")
    searcher = default_searcher(passages, embeddings=np.load(emb) if emb.exists() else None)
    tools = ToolRegistry(Catalog(), searcher, PatientMemory("data/agent/memory.sqlite"))
    return create_server(tools, build_agent(tools, OllamaClient()))


if __name__ == "__main__":
    default_server().run()
