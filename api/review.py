"""Public medication label review API, independent of the research LLM path."""
import asyncio
from typing import Annotated
from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from hosting_limits import AnalysisBudget, AdmissionDenied

router = APIRouter(prefix="/api/v2", tags=["Medication label review"])
limiter = Limiter(key_func=get_remote_address)
budget = AnalysisBudget(2, 1000)


class ReviewRequest(BaseModel):
    rxcuis: list[Annotated[str, Field(pattern=r"^\d{1,12}$")]] = Field(min_length=2, max_length=20)
    model_config = {"extra": "forbid"}


@router.get("/catalog")
async def catalog_info(request: Request):
    catalog = request.app.state.review_service.catalog
    return {"terms": len(catalog.items), "metadata": catalog.metadata, "max_medications": 20,
            "max_pairs": 190, "source": "RxNorm terminology; not interaction coverage"}


@router.get("/medications")
@limiter.limit("90/minute")
async def search(request: Request, q: Annotated[str, Query(min_length=2, max_length=80)]):
    return {"results": request.app.state.review_service.catalog.search(q)}


@router.post("/review")
@limiter.limit("5/minute")
async def review(request: Request, body: ReviewRequest):
    ids = list(dict.fromkeys(body.rxcuis))
    service = request.app.state.review_service
    if len(ids) < 2:
        raise HTTPException(422, "Select at least two distinct catalog entries.")
    if any(i not in service.catalog.by_id for i in ids):
        raise HTTPException(422, "A medication is not in this catalog. Search and select it again.")
    try:
        with budget.reserve():
            return await service.review(ids)
    except AdmissionDenied as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "30"}) from exc
    except (TimeoutError, asyncio.CancelledError) as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise HTTPException(503, "Sources took too long to respond. Try fewer medicines or retry later.") from exc
