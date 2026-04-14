from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1)
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[Dict[str, Any]]


def get_retriever(request: Request):
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return retriever


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, retriever=Depends(get_retriever)):
    results = retriever.search(payload.query)

    if payload.min_score is not None:
        results = [r for r in results if r.get("final_score", 0.0) >= payload.min_score]

    if payload.top_k is not None:
        results = results[: payload.top_k]

    return {"query": payload.query, "count": len(results), "results": results}


@router.post("/clear-cache")
def clear_cache(retriever=Depends(get_retriever)):
    retriever.clear_entity_cache()
    return {"status": "ok"}
