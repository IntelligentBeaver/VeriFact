from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from config import load_verifier_api_config
from verifier.verifier_service import PickleVerifier

verifier_api_config = load_verifier_api_config()

router = APIRouter(prefix="/verifier", tags=["verifier"])


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="Claim to verify")
    top_k: Optional[int] = Field(
        None,
        ge=verifier_api_config.min_top_k,
        le=verifier_api_config.max_top_k,
        description="Top-k evidence passages",
    )
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum evidence score")


class EvidenceInfo(BaseModel):
    title: Optional[str]
    url: Optional[str]
    text: Optional[str]
    score: Optional[float]
    passage_id: Optional[str]


class VerifyResponse(BaseModel):
    claim: str
    verdict: str
    confidence: Optional[float]
    scores: Dict[str, float]
    evidence: Optional[List[EvidenceInfo]]
    retriever_candidates: int


def get_retriever(request: Request):
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return retriever


def get_verifier(request: Request) -> PickleVerifier:
    verifier = getattr(request.app.state, "verifier", None)
    if verifier is None:
        raise HTTPException(status_code=503, detail="Verifier not initialized")
    return verifier


@router.get("/health")
def health(request: Request) -> Dict[str, Any]:
    verifier = getattr(request.app.state, "verifier", None)
    return {"status": "ok", "verifier_loaded": verifier is not None}


@router.post("/verify", response_model=VerifyResponse)
def verify_claim(
    payload: VerifyRequest,
    retriever=Depends(get_retriever),
    verifier: PickleVerifier = Depends(get_verifier),
) -> VerifyResponse:
    results = retriever.search(payload.claim, deduplicate=True)

    min_score = payload.min_score if payload.min_score is not None else verifier_api_config.default_min_score
    results = [r for r in results if r.get("final_score", 0.0) >= min_score]

    top_k = payload.top_k if payload.top_k is not None else verifier_api_config.default_top_k
    results = results[:top_k]

    if not results:
        return VerifyResponse(
            claim=payload.claim,
            verdict=verifier_api_config.no_evidence_verdict,
            confidence=None,
            scores={},
            evidence=None,
            retriever_candidates=0,
        )

    best = results[0]
    passage = best.get("passage", {})
    evidence_text = passage.get("text") or ""

    if not evidence_text:
        return VerifyResponse(
            claim=payload.claim,
            verdict=verifier_api_config.no_evidence_verdict,
            confidence=None,
            scores={},
            evidence=None,
            retriever_candidates=len(results),
        )

    try:
        result = verifier.verify(payload.claim, evidence_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Verifier inference error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Verifier inference failed") from exc
    # Build list of evidences from the top-k retriever results
    evidence_list: List[EvidenceInfo] = []
    for r in results:
        p = r.get("passage", {})
        evidence_list.append(
            EvidenceInfo(
                title=p.get("title"),
                url=p.get("url"),
                text=p.get("text"),
                score=r.get("final_score"),
                passage_id=p.get("passage_id"),
            )
        )

    # Apply configurable confidence threshold: map low-confidence predictions
    # to the no-evidence verdict while preserving scores and reported confidence.
    threshold = getattr(verifier_api_config, "confidence_threshold", None)
    verdict = result.label
    if threshold is not None and result.confidence is not None:
        try:
            if float(result.confidence) < float(threshold):
                verdict = verifier_api_config.no_evidence_verdict
        except Exception:
            # If conversion fails, ignore threshold and return original verdict
            pass

    return VerifyResponse(
        claim=payload.claim,
        verdict=verdict,
        confidence=result.confidence,
        scores=result.scores,
        evidence=evidence_list,
        retriever_candidates=len(results),
    )
