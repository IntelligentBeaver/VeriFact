"""FastAPI service for QA built on top of the retriever API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import QAConfigDefaults, load_env_float, load_env_int, load_env_str, PROMPT_TEMPLATE
from qa_system import QASystem


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
    return QAServiceConfig(
        retriever_url=load_env_str("RETRIEVER_URL", QAConfigDefaults.retriever_url),
        ollama_url=load_env_str("OLLAMA_URL", QAConfigDefaults.ollama_url),
        ollama_model=load_env_str("OLLAMA_MODEL", QAConfigDefaults.ollama_model),
        top_k=load_env_int("TOP_K", QAConfigDefaults.top_k),
        min_score=load_env_float("MIN_SCORE", QAConfigDefaults.min_score),
        max_context_chars=load_env_int("MAX_CONTEXT_CHARS", QAConfigDefaults.max_context_chars),
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
    return PROMPT_TEMPLATE.format(question=question, context=context)


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
