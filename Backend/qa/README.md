# QA System

This directory provides a QA layer on top of the existing retriever.

## Requirements

- ElasticSearch running (per retrieval docs)
- Ollama running (local or Docker)
- Llama 3.1 8B model pulled in Ollama

## Ollama (Docker) Setup

```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull llama3.1:8b
```

## Quick Start

From the repo root:

```bash
cd qa
python run_qa.py
```

Configuration is at the top of `run_qa.py`.

## Configuration

Edit the variables at the top of `run_qa.py`:

- `INDEX_DIR`
- `OLLAMA_URL`
- `OLLAMA_MODEL`
- `TOP_K`
- `MIN_SCORE`
- `MAX_CONTEXT_CHARS`
