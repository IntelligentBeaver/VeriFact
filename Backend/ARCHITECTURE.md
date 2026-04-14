# Verifact Backend Architecture

## Overview

Verifact Backend is a FastAPI application that exposes three related systems:

1. Retrieval - finds relevant medical passages for a query.
2. QA - uses retrieved passages plus an LLM to answer a question.
3. Verifier - checks whether a claim is supported by evidence.

All three systems share the same retrieval layer, and the main application loads the retriever once at startup so the API can serve requests without reloading the index each time.

## Runtime Entry Point

The main entry point is [main.py](main.py). On startup it does two things:

- Creates the retriever from the configured index directory.
- Loads the verifier model from a pickle file if it exists.

It then registers the three route groups:

- `/retrieval`
- `/qa`
- `/verifier`

The root endpoint `/` returns a simple health status, and `/health` reports whether the retriever is loaded and whether ElasticSearch is connected.

## Configuration

Runtime settings are centralized in [config.py](config.py). The application reads environment variables first and falls back to defaults.

The main configuration groups are:

- App settings such as the app title, index directory, and verifier model path.
- QA settings such as `QA_TOP_K`, `QA_MIN_SCORE`, `OLLAMA_URL`, and `OLLAMA_MODEL`.
- Retriever settings such as ElasticSearch host/port, ranking weights, and score thresholds.
- Verifier settings such as label names, tokenizer settings, and confidence thresholds.

This design keeps operational tuning outside the code while still giving each subsystem sensible defaults.

## Retrieval System

The retrieval system is implemented in [retrieval/simple_retriever.py](retrieval/simple_retriever.py).

### What it does

It turns a user query into ranked evidence passages. The retriever combines multiple signals instead of relying on a single search strategy.

### How it works

The search flow is:

1. FAISS semantic search finds passages that are close in embedding space.
2. ElasticSearch lexical search finds passages that match the query terms.
3. Reciprocal Rank Fusion combines the two ranked lists.
4. A cross-encoder reranks the merged candidates.
5. Additional scoring signals are blended in, including lexical overlap, domain authority, freshness, and medical review signals.
6. Duplicate passages are removed if deduplication is enabled.
7. The retriever returns the top results above the minimum score.

### Supporting data

The retriever reads from the `storage/` directory, which contains the FAISS index, embeddings, metadata, and MeSH concept data used for medical entity matching.

### API routes

The retrieval router in [routes/retrieval.py](routes/retrieval.py) exposes:

- `POST /retrieval/search` - search for passages.
- `POST /retrieval/clear-cache` - clear the retriever entity cache.

## QA System

The QA system is implemented in [qa/qa_api.py](qa/qa_api.py) for the API version and [qa/qa_system.py](qa/qa_system.py) for the local CLI/system version.

### What it does

It answers a natural-language question by:

- searching the retriever for relevant passages,
- filtering those passages by score,
- building a compact evidence context,
- sending that context to Ollama,
- returning a grounded answer with source metadata.

### How it works

The QA layer does not answer from the LLM alone. It first retrieves evidence and then builds a prompt that instructs the model to:

- answer only from the provided sources,
- state when there is not enough evidence,
- prefer careful medical wording,
- cite sources using source tags.

The prompt is intentionally strict so the answer stays tied to retrieved evidence.

### API routes

The QA router in [routes/qa.py](routes/qa.py) exposes:

- `GET /qa/health` - report the configured Ollama connection.
- `POST /qa/answer` - retrieve evidence and generate an answer.

### Local CLI

The interactive CLI in [qa/run_qa.py](qa/run_qa.py) runs the same overall flow from the terminal and can optionally save the full JSON response.

## Verifier System

The verifier is implemented in [verifier/verifier_service.py](verifier/verifier_service.py) and exposed through [routes/verifier.py](routes/verifier.py).

### What it does

It checks whether a claim is supported by evidence returned from the retriever.

### How it works

The request flow is:

1. The claim is sent through the retriever.
2. The top evidence passages are filtered by score.
3. The best passage text is passed into the verifier model.
4. The verifier returns a label, confidence value, and per-class scores.
5. If confidence is below the configured threshold, the verdict is downgraded to the configured no-evidence verdict.

### Model loading

The verifier loader supports multiple model styles:

- objects that implement `verify_claim()`
- models with `predict_proba()`
- models with `decision_function()`
- models with `predict()`
- callable Torch models

It can load the model from a pickle, joblib artifact, or Torch-compatible file, and it can fall back to a tokenizer from the configured tokenizer directory or base model name.

### API routes

The verifier router exposes:

- `GET /verifier/health` - report whether the verifier is loaded.
- `POST /verifier/verify` - retrieve evidence and verify a claim.

## Request Flow Summary

A typical request uses the systems in this order:

- Retrieval only: query -> retriever -> ranked passages
- QA: question -> retriever -> evidence context -> Ollama -> answer
- Verification: claim -> retriever -> top evidence -> verifier model -> verdict

The retriever is the shared foundation for both QA and verification.

## Useful Supporting Files

- [CONFIG_ENV.md](CONFIG_ENV.md) documents the main environment variables.
- [qa/README.md](qa/README.md) explains the QA helper workflow.
- [retrieval/SIMPLE_README.md](retrieval/SIMPLE_README.md) explains the single-file retriever.
- [verifier/readme.md](verifier/readme.md) explains verifier model placement.

## Local Run

Start the API with:

```bash
uvicorn main:app --reload
```

For the QA CLI and retrieval helper scripts, use the commands documented in their own README files.
