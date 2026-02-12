"""FastAPI routes for QA using the in-process retriever and Ollama."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from qa.config import QAConfigDefaults, load_env_float, load_env_int, load_env_str


@dataclass(frozen=True)
class QAServiceConfig:
    ollama_url: str
    ollama_model: str
    top_k: int
    min_score: float
    max_context_chars: int


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override top_k")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override min_score")


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]


class OllamaClient:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.HTTPError as exc:
            details = ""
            try:
                details = response.text.strip()
            except Exception:
                details = ""
            detail_msg = f" Details: {details}" if details else ""
            raise RuntimeError(
                "Ollama returned an HTTP error. Check the model tag and server logs."
                + detail_msg
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not reachable at the configured URL. "
                "Check the OLLAMA_URL env var and server status."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Ollama request timed out. Try a smaller context or check server load."
            ) from exc


router = APIRouter(prefix="/qa", tags=["qa"])


def load_config() -> QAServiceConfig:
    return QAServiceConfig(
        ollama_url=load_env_str("OLLAMA_URL", QAConfigDefaults.ollama_url),
        ollama_model=load_env_str("OLLAMA_MODEL", QAConfigDefaults.ollama_model),
        top_k=load_env_int("TOP_K", QAConfigDefaults.top_k),
        min_score=load_env_float("MIN_SCORE", QAConfigDefaults.min_score),
        max_context_chars=load_env_int("MAX_CONTEXT_CHARS", QAConfigDefaults.max_context_chars),
    )


def get_retriever(request: Request):
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return retriever


def get_ollama(request: Request) -> OllamaClient:
    client = getattr(request.app.state, "ollama", None)
    if client is None:
        cfg = load_config()
        client = OllamaClient(cfg.ollama_url, cfg.ollama_model)
        request.app.state.ollama = client
    return client


def _build_context(results: List[Dict[str, Any]], max_chars: int) -> (str, List[Dict[str, Any]]):
    context_blocks = []
    sources = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        passage = result.get("passage", {})
        title = passage.get("title", "")
        url = passage.get("url", "")
        text = passage.get("text", "")
        score = result.get("final_score", 0)

        block = (
            f"[Source {i}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Score: {score:.3f}\n"
            f"Text: {text}\n"
        )

        if total_chars + len(block) > max_chars:
            break

        context_blocks.append(block)
        total_chars += len(block)
        sources.append(
            {
                "id": i,
                "title": title,
                "url": url,
                "score": score,
            }
        )

    return "\n".join(context_blocks), sources


def _build_prompt(question: str, context: str) -> str:
    return (
        "You are a careful medical QA assistant. Answer using only the provided sources. "
        "If the sources do not support an answer, say you do not have enough evidence. "
        "If at least one source indicates an association, risk, or link, answer with that "
        "association even if causation is not proven. "
        "If sources explicitly state transmission or causation, you may say it causes or transmits. "
        "If sources show association or risk but not causation, say 'associated with increased risk' "
        "and avoid claiming it causes the outcome. "
        "Use this format: 'Conclusion: ...' then 'Evidence: ...'. "
        "Limit to 2 sentences total. "
        "Only cite sources that explicitly mention diabetes or prediabetes risk/link. "
        "Cite sources like [Source 1], [Source 2].\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n{context}\n\n"
        "Answer:"
    )


@router.get("/health")
def health() -> Dict[str, Any]:
    cfg = load_config()
    return {
        "status": "ok",
        "ollama_url": cfg.ollama_url,
        "ollama_model": cfg.ollama_model,
    }


@router.post("/answer", response_model=AnswerResponse)
def answer(
    request: AnswerRequest,
    retriever=Depends(get_retriever),
    ollama: OllamaClient = Depends(get_ollama),
) -> AnswerResponse:
    cfg = load_config()
    top_k = request.top_k or cfg.top_k
    min_score = request.min_score if request.min_score is not None else cfg.min_score

    results = retriever.search(request.question)
    results = [r for r in results if r.get("final_score", 0.0) >= min_score]
    results = results[:top_k]

    if not results:
        return AnswerResponse(
            question=request.question,
            answer="I do not have enough evidence to answer from the index.",
            sources=[],
        )

    context, sources = _build_context(results, cfg.max_context_chars)
    prompt = _build_prompt(request.question, context)

    try:
        answer_text = ollama.generate(prompt)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    return AnswerResponse(question=request.question, answer=answer_text, sources=sources)
