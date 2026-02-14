# Configurable Environment Variables

This project centralizes runtime defaults in `config.py`. You can override them with environment variables.

## QA / LLM
- `QA_TOP_K` — number of sources to return (int). Default: 10
- `QA_MIN_SCORE` — minimum retriever score to include a source (float). Default: 0.35
- `QA_MAX_CONTEXT_CHARS` — max characters of combined context passed to LLM (int). Default: 5000
- `OLLAMA_URL` — Ollama API URL. Default: `http://localhost:11434/api/generate`
- `OLLAMA_MODEL` — Ollama model tag. Default: `llama3.2:3b`
- `RETRIEVER_URL` — URL for the retriever service (used by `qa/qa_api.py`). Default: `http://retriever:8000`

## Retriever / ElasticSearch / FAISS
- `ES_HOST` — ElasticSearch host. Default: `127.0.0.1`
- `ES_PORT` — ElasticSearch port. Default: `9200`
- `ES_INDEX` — ElasticSearch index name. Default: `medical_passages`
- `FAISS_TOPK` — candidates to retrieve from FAISS (int). Default: 50
- `ES_TOPK` — candidates to retrieve from ElasticSearch (int). Default: 50
- `FINAL_TOPK` — final returned results (int). Default: 5
- `RRF_K` — RRF fusion k parameter (int). Default: 60
- `RRF_WEIGHT_FAISS` — RRF weight for FAISS (float). Default: 0.5
- `RRF_WEIGHT_ES` — RRF weight for ES (float). Default: 0.5
- `WEIGHT_FAISS` — scoring weight for FAISS (float). Default: 0.25
- `WEIGHT_CROSS_ENCODER` — scoring weight for cross-encoder (float). Default: 0.40
- `WEIGHT_ENTITY_MATCH` — scoring weight for entity match (float). Default: 0.10
- `WEIGHT_LEXICAL` — scoring weight for lexical match (float). Default: 0.10
- `WEIGHT_DOMAIN` — scoring weight for domain authority (float). Default: 0.10
- `WEIGHT_FRESHNESS` — scoring weight for freshness (float). Default: 0.05
- `RETRIEVER_MIN_SCORE` — retrieval minimum score used by retriever (float). Default: 0.4
- `MEDICAL_REVIEW_BONUS` — bonus added if medically reviewed (float). Default: 0.10
- `AUTHOR_BONUS` — small author bonus (float). Default: 0.02
- `FRESHNESS_RECENT` — days threshold for "recent" (int). Default: 365
- `FRESHNESS_MODERATE` — days threshold for "moderate" (int). Default: 1095
- `FRESHNESS_OLD` — days threshold for "old" (int). Default: 1825

## Example — PowerShell
```powershell
$env:OLLAMA_URL = "http://localhost:11434/api/generate"
$env:OLLAMA_MODEL = "llama3.2:3b"
$env:QA_TOP_K = "5"
$env:ES_HOST = "127.0.0.1"
$env:ES_PORT = "9200"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Example — Bash
```bash
export OLLAMA_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="llama3.2:3b"
export QA_TOP_K=5
export ES_HOST=127.0.0.1
export ES_PORT=9200
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Notes
- The loader functions in `config.py` already prefer env vars when present, so you can set either environment variables or change the defaults in `config.py`.
- After changing envs, restart the FastAPI process to pick up new values on startup.
