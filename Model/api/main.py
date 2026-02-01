"""FastAPI service for the retrieval system (outside harvester, inside Model)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup: allow importing from harvester/admin/retrieval
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = MODEL_DIR / "harvester" / "admin" / "retrieval"

if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from simple_retriever import MinimalModelManager, SimpleRetriever  # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="VeriFact Retrieval API", version="1.0.0")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override MIN_SCORE if provided")


class SearchResult(BaseModel):
    final_score: float
    scores: Dict[str, float]
    passage: Dict[str, Any]
    faiss_score: Optional[float] = None
    faiss_rank: Optional[int] = None
    es_score: Optional[float] = None
    es_rank: Optional[int] = None
    rrf_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[SearchResult]


model_manager: Optional[MinimalModelManager] = None
retriever: Optional[SimpleRetriever] = None


def _init_retriever() -> None:
    global model_manager, retriever
    if retriever is not None:
        return

    index_dir = MODEL_DIR / "harvester" / "storage" / "outputs" / "combined"
    model_manager = MinimalModelManager(index_dir)
    retriever = SimpleRetriever(model_manager, index_dir)


@app.on_event("startup")
def startup_event() -> None:
    _init_retriever()


@app.get("/health")
def health() -> Dict[str, Any]:
    if retriever is None:
        return {"status": "initializing"}
    return {
        "status": "ok",
        "index_dir": str(MODEL_DIR / "harvester" / "storage" / "outputs" / "combined"),
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    results = retriever.search(request.query)

    # Apply optional min_score override
    if request.min_score is not None:
        results = [r for r in results if r.get("final_score", 0.0) >= request.min_score]

    # Apply top_k slicing
    results = results[: request.top_k]

    return SearchResponse(
        query=request.query,
        total=len(results),
        results=results,
    )


@app.post("/cache/clear")
def clear_cache() -> Dict[str, str]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    retriever.clear_entity_cache()
    return {"status": "cleared"}
