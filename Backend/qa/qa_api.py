"""FastAPI service for QA built on top of the retriever API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import load_qa_config


@dataclass
class QAServiceConfig:
    retriever_url: str
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


def load_config() -> QAServiceConfig:
    defaults = load_qa_config()
    return QAServiceConfig(
        retriever_url=defaults.retriever_url,
        ollama_url=defaults.ollama_url,
        ollama_model=defaults.ollama_model,
        top_k=defaults.top_k,
        min_score=defaults.min_score,
        max_context_chars=defaults.max_context_chars,
    )


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
                "Check the OLLAMA_URL env var and container status."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Ollama request timed out. Try a smaller context or check server load."
            ) from exc


app = FastAPI(title="VeriFact QA API", version="1.0.0")
config = load_config()
ollama = OllamaClient(config.ollama_url, config.ollama_model)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "retriever_url": config.retriever_url,
        "ollama_url": config.ollama_url,
        "ollama_model": config.ollama_model,
    }


def _build_context(results: List[Dict[str, Any]], max_chars: int) -> tuple[str, List[Dict[str, Any]]]:
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
        "Cite sources like [Source 1], [Source 2].\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n{context}\n\n"
        "Answer:"
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    top_k = request.top_k or config.top_k
    min_score = request.min_score if request.min_score is not None else config.min_score

    try:
        retriever_resp = requests.post(
            f"{config.retriever_url}/search",
            json={"query": request.question, "top_k": top_k, "min_score": min_score},
            timeout=120,
        )
        retriever_resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Retriever error: {exc}") from exc

    payload = retriever_resp.json()
    results = payload.get("results", [])

    if not results:
        return AnswerResponse(
            question=request.question,
            answer="I do not have enough evidence to answer from the index.",
            sources=[],
        )

    context, sources = _build_context(results, config.max_context_chars)
    prompt = _build_prompt(request.question, context)

    try:
        answer_text = ollama.generate(prompt)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    return AnswerResponse(question=request.question, answer=answer_text, sources=sources)
