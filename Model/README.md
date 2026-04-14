# VeriFact Model Workspace

This directory is the model and data-engineering workspace for VeriFact.

It contains the systems that produce and maintain the evidence layer consumed by application APIs, including:

- ingestion/scraping adapters,
- embedding + FAISS indexing pipelines,
- incremental indexing automation,
- QA/retrieval local services,
- claim labeling and dataset-generation workflows.

## 1) Why This Workspace Exists

The production backend should stay lean and request-focused. Heavy data/model operations are kept in this workspace to:

- isolate long-running and compute-intensive jobs,
- support iterative experimentation safely,
- allow frequent index refresh cycles without coupling to API release cadence.

## 2) Directory Overview

```text
Model/
├─ main.py                    # Interactive CLI orchestrator menu
├─ api/                       # FastAPI retriever service wrapper
├─ retrieval/                 # Retrieval engine, scripts, quick references
├─ qa/                        # QA local system and QA API wrappers
├─ harvester/                 # Data adapters + indexing/seed pipelines
├─ scheduler/                 # FastAPI-friendly scheduling wrappers
├─ labelling/                 # Claim extraction and labeling workflows
├─ docker-compose.yml         # Elasticsearch + retriever service stack
├─ Dockerfile.retriever       # Retriever API container image
└─ scripts/bootstrap.ps1      # Convenience bootstrap for containers + indexing
```

## 3) Top-Level Execution Surfaces

## 3.1 Interactive Operations Menu (`main.py`)

`main.py` provides a terminal menu that orchestrates common tasks:

- run retrieval workflow,
- run labelling workflow,
- start Docker services,
- run schedulers (WHO scraping / FAISS update).

This is the quickest way to operate multiple subsystems without remembering each script path.

## 3.2 Retrieval API Service (`api/main.py`)

`api/main.py` exposes a retrieval-focused FastAPI service:

- startup initializes retriever from configured index dir,
- `POST /search` executes search with optional overrides,
- `POST /cache/clear` clears retriever entity cache,
- `GET /health` reports readiness.

Default index path is environment-driven (`INDEX_DIR`) and resolves to `Model/retrieval/storage` when unset.

## 3.3 Container Runtime

`docker-compose.yml` defines:

- Elasticsearch service,
- retriever service built from `Dockerfile.retriever`.

This supports local containerized retrieval testing.

## 4) Retrieval Subsystem (`retrieval/`)

The retrieval subsystem is a standalone hybrid retrieval engine with setup and interactive search workflows.

### Core Files

- `simple_retriever.py`
- `run_retriever.py`
- `QUICK_REFERENCE.md`
- `SIMPLE_README.md`

### Retrieval Pipeline

The retriever executes a multi-stage hybrid process:

1. FAISS semantic search
2. Elasticsearch lexical search
3. RRF fusion
4. Cross-encoder reranking
5. Multi-signal final scoring
6. score filtering and top-k return

Depending on variant/config, it can also include deduplication and MeSH/SapBERT-assisted entity scoring.

### Model Assets

`MinimalModelManager` loads:

- sentence embedding model (`all-mpnet-base-v2` in retrieval implementation),
- cross-encoder reranker,
- optional SapBERT model,
- FAISS index + metadata,
- MeSH concept dictionary and optional precomputed SapBERT embeddings.

### Why This Design

- Hybrid retrieval mitigates single-model blind spots.
- Explicit scoring components increase debuggability and ranking control.
- Scripted setup (`run_retriever.py setup`) reduces onboarding friction.

## 5) QA Subsystem (`qa/`)

The QA subsystem exists in two forms:

1. **Local QA runtime** (`qa_system.py`, `run_qa.py`)
2. **QA API wrapper** (`qa_api.py`)

### Local QA Flow

- retrieve passages,
- score-filter and clip,
- build constrained source context,
- prompt Ollama model,
- return answer + cited sources.

### QA API Flow

`qa_api.py` calls retriever endpoint over HTTP and then Ollama, exposing:

- `GET /health`
- `POST /answer`

### Config

`qa/config.py` centralizes defaults:

- top_k, min_score
- Ollama URL/model
- max context chars
- retriever URL

### Why This Design

- Enables side-by-side local QA experimentation without touching backend runtime.
- Keeps prompt strategy and retrieval grounding in one focused module set.

## 6) Harvester Subsystem (`harvester/`)

The harvester is the data-engineering core for source ingestion and index preparation.

It combines adapters and pipelines.

## 6.1 Adapters (`harvester/adapters/`)

Adapters target multiple source families:

- WHO scraping adapters (news, outbreak, feature stories, fact sheets)
- WebMD scraping/indexing/preprocessing adapters
- PubMed-related adapter(s)

Adapter responsibilities include:

- source-specific fetch/parsing,
- serialization of normalized article structures,
- producing source-specific JSON artifacts for downstream indexing.

## 6.2 Pipeline Scripts (`harvester/pipeline/`)

Key pipeline scripts:

1. `build_combined_faiss.py`
- combines WHO + WebMD content,
- extracts passages,
- deduplicates,
- computes embeddings,
- writes `embeddings.npy`, `metadata.json`, optional `sapbert_embeddings.npy`, and `index.faiss`.

2. `build_concept_dictionary.py`
- builds MeSH-like concept dictionary from seed JSONs,
- generates split embedding collections and label embeddings for concept workflows.

3. `build_webmd_indxeing.py`
- interactive retrieval/search utility around FAISS + reranking + trust heuristics.

### Output Pattern

Pipelines write aligned artifacts where metadata ordering matches vector ordering. This alignment is critical for accurate retrieval result reconstruction.

## 7) Incremental Indexing Scheduler (`harvester/pipeline/indexing_scheduler/`)

This subsystem supports incremental FAISS refreshes.

Primary script: `update_faiss_incremental.py`

Capabilities:

- tracks processed `passage_id` values in checkpoint JSON,
- scans all configured source directories,
- computes embeddings only for unseen passages,
- merges new and existing embeddings/metadata,
- updates FAISS index incrementally,
- updates checkpoint metadata.

Auxiliary script: `create_initial_checkpoint.py` for bootstrapping checkpoint state from existing metadata.

Why this exists:

- full re-indexing is expensive,
- incremental refresh supports regular content updates with lower cost.

## 8) WHO Scraping Scheduler (`harvester/pipeline/who_scraping_scheduler/`)

Main orchestrator: `who_scheduler.py`

Modes:

- `full`
- `news`
- `outbreak`
- `features`

Functions:

- runs relevant headline + details adapters in sequence,
- supports configurable lookback window (`--days`),
- emits timestamped logs and exit status.

Additional helpers:

- `scheduler_config.py` for scheduling/retry defaults,
- `fastapi_integration.py` for APScheduler integration,
- `verify_setup.py` to validate environment and script layout.

## 9) Global Scheduler Wrappers (`scheduler/`)

The `Model/scheduler/` package provides higher-level scheduling wrappers for FastAPI integration.

Files:

- `config.py`: central scheduler config for WHO scraping and FAISS indexing
- `fastapi_who_scheduler.py`: periodic WHO scraping job setup
- `fastapi_indexing_scheduler.py`: periodic FAISS update/rebuild job setup

This layer gives a clean entrypoint for embedding scheduled tasks into service lifecycles.

## 10) Labelling Subsystem (`labelling/`)

The labelling subsystem turns claims and retrieval evidence into structured labeled outputs for downstream model workflows.

## 10.1 Core Components

- `config.py`: paths, thresholds, output files, model names
- `claim_extraction.py`: sentence splitting, claim extraction, claim negation generation
- `stance_detector.py`: heuristic and NLI-based stance prediction
- `scoring.py`: passage scoring and auto-label decision support utilities
- `persistence.py`: load/save trackers and labeled data persistence
- `label_passages.py`: orchestration script with multiple labeling/export workflows
- `negate_labeled_claims.py`: polarity-flipped labeled claim generation

## 10.2 Representative Workflows in `label_passages.py`

- label unlabeled claims with retrieval + NLI stance scoring
- remove irrelevant unlabeled claims based on thresholds
- extract queries and negations
- generate negations from labeled claims
- generate claims from verified claims
- export labels/statistics

### Why This Design

- preserves modularity while supporting large workflow orchestration,
- enables configurable automation thresholds with fallback/manual review paths,
- provides reproducible artifact output for training/evaluation pipelines.

## 11) Dependency Notes

Known requirements files:

- `api/requirements.txt`
- `qa/requirements.txt`
- `harvester/pipeline/who_scraping_scheduler/requirements.txt`

Some scripts additionally depend on ecosystem libraries not always pinned in one place (for example sentence-transformers, faiss-cpu, transformers, APScheduler, BeautifulSoup/selenium stack), so environment planning should account for subsystem-specific needs.

## 12) Common Commands

From `Model/`:

### Interactive menu

```bash
python main.py
```

### Retriever API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Retrieval setup and query

```bash
cd retrieval
python run_retriever.py setup
python run_retriever.py search "vaccine safety during pregnancy"
```

### QA interactive

```bash
cd qa
python run_qa.py
```

### Docker bootstrap (PowerShell)

```powershell
cd Model
./scripts/bootstrap.ps1
```

### WHO scheduler manual run

```bash
cd harvester/pipeline/who_scraping_scheduler
python who_scheduler.py --mode full --days 3
```

### Incremental FAISS update

```bash
cd harvester/pipeline/indexing_scheduler
python update_faiss_incremental.py
```

## 13) Recommended Operational Sequence

For ongoing evidence refresh:

1. run WHO/WebMD scraping adapters or scheduler,
2. run incremental indexing update,
3. validate retrieval outputs locally,
4. reindex Elasticsearch where required by serving stack,
5. roll updated artifacts into backend/model serving environments.

## 14) Reliability and Safety Considerations

- Keep metadata and embedding rows strictly aligned.
- Preserve checkpoint files; accidental deletion can trigger expensive rebuilds.
- Validate scheduler paths when restructuring directories.
- Treat generated labels as machine-assisted outputs requiring quality checks.
- Monitor resource usage for embedding generation and NLI-heavy labeling operations.

## 15) Relationship to Backend and App

- **Model** produces and maintains retrieval/labelling assets.
- **Backend** serves stable API contracts over those assets.
- **App** provides user-facing interactions over Backend APIs.

This layered separation is intentional and is central to VeriFact maintainability.
