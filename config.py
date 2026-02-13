"""Centralized configuration for the Verifact backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import os


@dataclass(frozen=True)
class AppConfigDefaults:
    app_title: str = "Verifact Retrieval API"
    index_dir: str = "storage"
    verifier_model_path: str = "verifier/verifier_model.pkl"


@dataclass(frozen=True)
class VerifierModelConfigDefaults:
    labels_csv: str = "neutral,refutes,supports"
    input_template: str = "{claim} [SEP] {evidence}"
    # Prefer the tokenizer *directory* (contains tokenizer.json + tokenizer_config.json)
    tokenizer_path: str = "verifier"
    base_model: str = "dmis-lab/biobert-v1.1"
    tokenizer_max_length: int = 512
    tokenizer_padding: str = "max_length"
    tokenizer_local_files_only: bool = True


@dataclass(frozen=True)
class VerifierApiConfigDefaults:
    min_top_k: int = 1
    max_top_k: int = 10
    default_top_k: int = 3
    default_min_score: float = 0.35
    no_evidence_verdict: str = "NOT_ENOUGH_EVIDENCE"
    # Threshold below which predictions are considered insufficiently confident
    confidence_threshold: float = 0.68


@dataclass(frozen=True)
class AppConfig:
    app_title: str
    index_dir: str
    verifier_model_path: str


@dataclass(frozen=True)
class VerifierModelConfig:
    labels_csv: str
    input_template: str
    tokenizer_path: str
    base_model: str
    tokenizer_max_length: int
    tokenizer_padding: str
    tokenizer_local_files_only: bool


@dataclass(frozen=True)
class VerifierApiConfig:
    min_top_k: int
    max_top_k: int
    default_top_k: int
    default_min_score: float
    no_evidence_verdict: str
    confidence_threshold: float


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


def load_env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def load_app_config() -> AppConfig:
    defaults = AppConfigDefaults()
    return AppConfig(
        app_title=load_env_str("APP_TITLE", defaults.app_title),
        index_dir=load_env_str("INDEX_DIR", defaults.index_dir),
        verifier_model_path=load_env_str("VERIFIER_MODEL_PATH", defaults.verifier_model_path),
    )


def load_verifier_model_config() -> VerifierModelConfig:
    defaults = VerifierModelConfigDefaults()
    return VerifierModelConfig(
        labels_csv=load_env_str("VERIFIER_LABELS", defaults.labels_csv),
        input_template=load_env_str("VERIFIER_INPUT_TEMPLATE", defaults.input_template),
        tokenizer_path=load_env_str("VERIFIER_TOKENIZER_PATH", defaults.tokenizer_path),
        base_model=load_env_str("VERIFIER_BASE_MODEL", defaults.base_model),
        tokenizer_max_length=load_env_int("VERIFIER_TOKENIZER_MAX_LENGTH", defaults.tokenizer_max_length),
        tokenizer_padding=load_env_str("VERIFIER_TOKENIZER_PADDING", defaults.tokenizer_padding),
        tokenizer_local_files_only=load_env_bool(
            "VERIFIER_TOKENIZER_LOCAL_FILES_ONLY", defaults.tokenizer_local_files_only
        ),
    )


def load_verifier_api_config() -> VerifierApiConfig:
    defaults = VerifierApiConfigDefaults()
    return VerifierApiConfig(
        min_top_k=load_env_int("VERIFIER_MIN_TOP_K", defaults.min_top_k),
        max_top_k=load_env_int("VERIFIER_MAX_TOP_K", defaults.max_top_k),
        default_top_k=load_env_int("VERIFIER_DEFAULT_TOP_K", defaults.default_top_k),
        default_min_score=load_env_float("VERIFIER_DEFAULT_MIN_SCORE", defaults.default_min_score),
        no_evidence_verdict=load_env_str("VERIFIER_NO_EVIDENCE_VERDICT", defaults.no_evidence_verdict),
        confidence_threshold=load_env_float("VERIFIER_CONFIDENCE_THRESHOLD", defaults.confidence_threshold),
    )


# ============================================================================
# QA & RETRIEVER DEFAULTS - centralized configurable values used across services
# ============================================================================


@dataclass(frozen=True)
class QAConfigDefaults:
    top_k: int = 10
    min_score: float = 0.35
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2:3b"
    max_context_chars: int = 5000
    retriever_url: str = "http://retriever:8000"


@dataclass(frozen=True)
class RetrieverConfigDefaults:
    es_host: str = "127.0.0.1"
    es_port: int = 9200
    es_index: str = "medical_passages"
    faiss_topk: int = 50
    es_topk: int = 50
    final_topk: int = 5
    rrf_k: int = 60
    rrf_weight_faiss: float = 0.5
    rrf_weight_es: float = 0.5
    weight_faiss: float = 0.25
    weight_cross_encoder: float = 0.40
    weight_entity_match: float = 0.10
    weight_lexical: float = 0.10
    weight_domain: float = 0.10
    weight_freshness: float = 0.05
    min_score: float = 0.4
    medical_review_bonus: float = 0.10
    author_bonus: float = 0.02
    domain_scores: Optional[Dict[str, float]] = None
    freshness_recent: int = 365
    freshness_moderate: int = 1095
    freshness_old: int = 1825


def load_qa_config() -> QAConfigDefaults:
    defaults = QAConfigDefaults()
    return QAConfigDefaults(
        top_k=load_env_int("QA_TOP_K", defaults.top_k),
        min_score=load_env_float("QA_MIN_SCORE", defaults.min_score),
        ollama_url=load_env_str("OLLAMA_URL", defaults.ollama_url),
        ollama_model=load_env_str("OLLAMA_MODEL", defaults.ollama_model),
        max_context_chars=load_env_int("QA_MAX_CONTEXT_CHARS", defaults.max_context_chars),
        retriever_url=load_env_str("RETRIEVER_URL", defaults.retriever_url),
    )


def load_retriever_config() -> RetrieverConfigDefaults:
    defaults = RetrieverConfigDefaults()
    domain_scores = {
        'who.int': 1.0,
        'cdc.gov': 1.0,
        'nih.gov': 1.0,
        'fda.gov': 1.0,
        'mayoclinic.org': 0.95,
        'hopkinsmedicine.org': 0.95,
        'health.harvard.edu': 0.95,
        'clevelandclinic.org': 0.90,
        'jamanetwork.com': 0.90,
        'webmd.com': 0.85,
        'healthline.com': 0.80,
        'medicalnewstoday.com': 0.80,
        'medlineplus.gov': 0.85,
        'default': 0.60,
    }

    return RetrieverConfigDefaults(
        es_host=load_env_str("ES_HOST", defaults.es_host),
        es_port=load_env_int("ES_PORT", defaults.es_port),
        es_index=load_env_str("ES_INDEX", defaults.es_index),
        faiss_topk=load_env_int("FAISS_TOPK", defaults.faiss_topk),
        es_topk=load_env_int("ES_TOPK", defaults.es_topk),
        final_topk=load_env_int("FINAL_TOPK", defaults.final_topk),
        rrf_k=load_env_int("RRF_K", defaults.rrf_k),
        rrf_weight_faiss=load_env_float("RRF_WEIGHT_FAISS", defaults.rrf_weight_faiss),
        rrf_weight_es=load_env_float("RRF_WEIGHT_ES", defaults.rrf_weight_es),
        weight_faiss=load_env_float("WEIGHT_FAISS", defaults.weight_faiss),
        weight_cross_encoder=load_env_float("WEIGHT_CROSS_ENCODER", defaults.weight_cross_encoder),
        weight_entity_match=load_env_float("WEIGHT_ENTITY_MATCH", defaults.weight_entity_match),
        weight_lexical=load_env_float("WEIGHT_LEXICAL", defaults.weight_lexical),
        weight_domain=load_env_float("WEIGHT_DOMAIN", defaults.weight_domain),
        weight_freshness=load_env_float("WEIGHT_FRESHNESS", defaults.weight_freshness),
        min_score=load_env_float("RETRIEVER_MIN_SCORE", defaults.min_score),
        medical_review_bonus=load_env_float("MEDICAL_REVIEW_BONUS", defaults.medical_review_bonus),
        author_bonus=load_env_float("AUTHOR_BONUS", defaults.author_bonus),
        domain_scores=domain_scores,
        freshness_recent=load_env_int("FRESHNESS_RECENT", defaults.freshness_recent),
        freshness_moderate=load_env_int("FRESHNESS_MODERATE", defaults.freshness_moderate),
        freshness_old=load_env_int("FRESHNESS_OLD", defaults.freshness_old),
    )
