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
