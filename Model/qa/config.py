"""Shared configuration utilities for QA CLI/API."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class QAConfigDefaults:
    top_k: int = 10
    min_score: float = 0.35
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2:3b"
    max_context_chars: int = 5000
    retriever_url: str = "http://retriever:8000"

# Prompt template uses {question} and {context} placeholders
PROMPT_TEMPLATE = (
    "You are a careful medical QA assistant. Answer using only the provided sources. "
    "If the sources do not support an answer, say you do not have enough evidence. "
    "If at least one source indicates an association, risk, or link, answer with that "
    "association even if causation is not proven. "
    "If sources explicitly state transmission or causation, you may say it causes or transmits. "
    "If sources show association or risk but not causation, say 'associated with increased risk' "
    "and avoid claiming it causes the outcome. "
    "Use this format: 'Conclusion: ...' then 'Evidence: ...'. Limit to 2 sentences total. "
    "Cite only sources that explicitly support the Conclusion.\n\n"
    "Question: {question}\n\n"
    "Sources:\n{context}\n\n"
    "Answer:"
)


def load_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_env_str(key: str, default: str) -> str:
    return os.getenv(key, default)
