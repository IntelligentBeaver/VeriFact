# VeriFact Backend

This directory contains the production-oriented FastAPI backend that powers retrieval, QA, and claim verification APIs for VeriFact clients.

It is designed as a single process with shared in-memory retriever state, so evidence retrieval remains consistent across all downstream services.

## 1) Mission and Design Intent

The backend exists to solve three related problems through one API surface:

- **Retrieval**: find relevant medical passages for a query.
- **QA**: answer questions using retrieval-grounded evidence and an LLM endpoint.
- **Verifier**: score claims against retrieved evidence and return verdicts.

Core design principle: retrieval is a shared foundational subsystem used by QA and verifier routes to minimize evidence drift.

## 2) Runtime Architecture

Primary runtime entrypoint: `main.py`

### Startup Lifecycle

At startup, backend performs:

1. loads app config via `load_app_config()` from `config.py`,
2. initializes `MinimalModelManager` and `SimpleRetriever` from `retrieval/simple_retriever.py`,
3. stores retriever on `app.state.retriever`,
4. attempts to load verifier artifact from configured model path,
5. stores verifier instance on `app.state.verifier` (or `None` if unavailable),
6. mounts route groups.

### Shutdown Lifecycle

On shutdown, it clears retriever entity cache to release memory.

### Mounted Route Groups

- `/retrieval` -> `routes/retrieval.py`
- `/qa` -> `routes/qa.py`
- `/verifier` -> `routes/verifier.py`

### Health Endpoints

- `GET /` returns simple status payload.
- `GET /health` includes retriever loaded state and Elasticsearch connectivity indicator.

## 3) Configuration System (`config.py`)

Backend configuration is centralized and environment-variable-first.

Config domains:

1. **App config**
- app title
- index directory
- verifier model path

2. **Retriever config**
- Elasticsearch connection and index name
- FAISS/ES candidate depths
- RRF parameters
- scoring weights and thresholds
- domain authority map
- freshness thresholds
- dedup toggle

3. **QA config**
- top_k, min_score
- Ollama URL/model
- max context length

4. **Verifier config**
- label map
- tokenizer/base model settings
- API defaults and confidence thresholding

This centralization ensures route modules remain focused on orchestration rather than environment parsing.

## 4) Retrieval Subsystem

Implementation: `retrieval/simple_retriever.py`

### 4.1 Core Pipeline

For each search query, the retriever executes:

1. FAISS semantic retrieval
2. Elasticsearch lexical retrieval
3. Reciprocal Rank Fusion (RRF)
4. Cross-encoder reranking
5. Multi-signal score computation
6. Optional deduplication
7. min-score filter + final top-k clipping

### 4.2 Score Composition

Final score blends multiple components:

- FAISS similarity
- cross-encoder relevance (sigmoid-normalized)
- optional entity match (MeSH/SapBERT)
- lexical overlap
- domain authority
- freshness
- metadata bonuses (medical review / author)

This balances semantic quality, source quality, and recency.

### 4.3 Entity and Domain Awareness

Retriever model manager can load:

- MeSH concept dictionary (`storage/mesh_concepts.json`)
- optional SapBERT embeddings

Domain tiering is derived from configured domain score map, enabling source-trust weighting.

### 4.4 Elasticsearch Utilities

The module also provides utility functions for index creation and bulk document indexing used by setup workflows.

## 5) Retrieval API

Router: `routes/retrieval.py`

### Endpoints

1. `POST /retrieval/search`
- request:
	- `query` (required)
	- `top_k` (optional)
	- `min_score` (optional)
- behavior:
	- executes retriever search,
	- applies optional score threshold,
	- applies optional top-k override,
	- returns `{query, count, results}`.

2. `POST /retrieval/clear-cache`
- clears retriever entity cache and returns status.

## 6) QA Subsystem

Router: `routes/qa.py`

### 6.1 Request Flow

1. validate `question` and optional overrides,
2. run retrieval (`deduplicate=False` in current QA route flow),
3. filter by score and clip by top_k,
4. build bounded context blocks from passages,
5. construct strict prompt template,
6. call Ollama generate API,
7. return answer with source metadata.

### 6.2 Ollama Client

QA route includes robust error mapping for:

- HTTP errors,
- connection failures,
- request timeouts.

This helps clients surface actionable diagnostics.

### 6.3 QA API Endpoints

- `GET /qa/health`
- `POST /qa/answer`

## 7) Verifier Subsystem

Router: `routes/verifier.py`

### 7.1 Request Flow

1. validate claim and override parameters,
2. retrieve candidate evidence passages,
3. score-filter and top-k clip,
4. select best evidence for model inference,
5. run verifier model,
6. apply confidence thresholding,
7. return verdict, confidence, class scores, and evidence list.

### 7.2 Verifier Service Compatibility Layer

Implementation: `verifier/verifier_service.py`

The verifier loader/inference wrapper supports multiple model styles:

- native `verify_claim()` objects,
- `predict_proba` models,
- `decision_function` models,
- plain `predict` models,
- callable torch/transformers models.

It also supports loading from joblib/torch/pickle pathways and tokenizer resolution from local path or base model fallback.

### 7.3 Confidence-Gated Verdicting

If model confidence is below configured threshold, verdict is downgraded to configured no-evidence verdict while preserving score information.

This is important for safety-sensitive claim-checking use cases.

### 7.4 Verifier API Endpoints

- `GET /verifier/health`
- `POST /verifier/verify`

## 8) Data and Artifacts Expected by Backend

Typical runtime artifacts:

- retrieval index directory (default `storage`) containing:
	- `index.faiss`
	- `metadata.json`
	- optional embedding side artifacts
- verifier model file (default `verifier/verifier_model.pkl`)
- tokenizer assets (default verifier dir includes `tokenizer.json`)

## 9) Running the Backend

From `Backend/`:

```bash
uvicorn main:app --reload
```

Optional host/port:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 10) Supporting Scripts and Docs

- Architecture overview: `ARCHITECTURE.md`
- Env variable guide: `CONFIG_ENV.md`
- Retrieval helper scripts: `retrieval/run_retriever.py`
- QA local subsystem docs: `qa/README.md`
- Verifier placement note: `verifier/readme.md`

## 11) Operational Guidance

### 11.1 First-Time Bring-Up

1. Prepare index artifacts under configured `INDEX_DIR`.
2. Start Elasticsearch if lexical retrieval is desired.
3. Place verifier model artifact in configured path.
4. Start FastAPI app.
5. Validate:
	 - `/health`
	 - `/retrieval/search`
	 - `/qa/health`
	 - `/verifier/health`

### 11.2 If Retrieval Returns Weak/No Results

- verify Elasticsearch reachability,
- verify FAISS index path and metadata alignment,
- inspect min-score thresholds,
- inspect domain/freshness scoring bias for your corpus.

### 11.3 If Verifier Fails to Load

- check model file path,
- verify required dependencies for unpickling are installed,
- verify tokenizer path/base model configuration.

## 12) Why This Backend Structure Works

- Shared retriever state improves consistency across features.
- Route isolation keeps QA and verifier evolution independent.
- Config centralization makes ops tuning straightforward.
- Compatibility wrapper in verifier service supports mixed model artifact ecosystems.

This backend is the stable API bridge between App UX and Model/data pipelines.
