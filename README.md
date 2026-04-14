# VeriFact

<div align="center">
  <img src="Assets/brand.png" alt="VeriFact Brand Logo" width="180" />
  <h3>App-first medical fact-checking platform for evidence-driven health claim verification</h3>
</div>

VeriFact starts with a simple mobile app experience and is powered by a retrieval-focused backend and model/data pipelines behind the scenes.

## Project Links

<div align="center">

[![Overleaf](https://img.shields.io/badge/Overleaf-Project-47A141?style=for-the-badge&logo=overleaf&logoColor=white)](https://www.overleaf.com/project/69350ec7bb5450bfb16872e9)
[![Figma](https://img.shields.io/badge/Figma-Design-1F1F1F?style=for-the-badge&logo=figma&logoColor=white)](https://www.figma.com/design/BqdgAEriXKzZm9ebEsSUkf/VeriFact-App)
[![Google%20Drive](https://img.shields.io/badge/Google%20Drive-Files-0F9D58?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1DYucnJS13Kk1Zyj8gGQGmIgIlD-9kA10?usp=drive_link)
[![Notion](https://img.shields.io/badge/Notion-Workspace-111111?style=for-the-badge&logo=notion&logoColor=white)](https://www.notion.so/VeriFact-Final-Year-Major-Project-2c5728051d2681e9b29ef8dda96482c4?source=copy_link)

</div>

## App First

The first thing VeriFact is built for is user experience:

- capture claims quickly (typing, camera, gallery),
- extract text with OCR,
- run verification and QA against indexed medical evidence,
- review confidence and sources,
- keep a searchable local history of prior checks.

## App Features

### Core user capabilities

- **Claim verification** with verdict and confidence.
- **Evidence browsing** with source cards and scores.
- **Question answering** from retrieved context.
- **Document search** over indexed passages.
- **OCR capture flow** from camera and gallery.
- **History timeline** with previews and re-run paths.
- **Environment-aware setup** (dev/prod flavors, runtime base URL override).

### UX highlights

- Fast mode switching (`Verifier`, `QA`, `Doc Search`) from the home flow.
- Local persistence through Hive for reliable history access.
- Transparent result presentation with source-level evidence context.

## App Screenshots

<table>
  <tr>
    <td align="center"><img src="Assets/App%20Screenshots/1.jpeg" alt="VeriFact app screenshot 1" width="220" /><br /><sub>Home and search workflow</sub></td>
    <td align="center"><img src="Assets/App%20Screenshots/2.jpeg" alt="VeriFact app screenshot 2" width="220" /><br /><sub>Verification interaction</sub></td>
    <td align="center"><img src="Assets/App%20Screenshots/3.jpeg" alt="VeriFact app screenshot 3" width="220" /><br /><sub>Result and confidence view</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="Assets/App%20Screenshots/4.jpeg" alt="VeriFact app screenshot 4" width="220" /><br /><sub>Evidence and sources</sub></td>
    <td align="center"><img src="Assets/App%20Screenshots/5.jpeg" alt="VeriFact app screenshot 5" width="220" /><br /><sub>History and replay flow</sub></td>
    <td align="center"><img src="Assets/App%20Screenshots/6.jpeg" alt="VeriFact app screenshot 6" width="220" /><br /><sub>Settings and controls</sub></td>
  </tr>
</table>

## How VeriFact Works End-to-End

1. User submits a claim/question from the app.
2. App calls backend APIs (`/retrieval`, `/qa`, `/verifier`).
3. Backend retriever ranks passages using hybrid retrieval.
4. QA/verifier layers produce grounded outputs.
5. App displays verdict, confidence, and sources.
6. Model pipelines keep the knowledge/index layer fresh.

## Platform Architecture

VeriFact consists of three cooperating systems.

### 1) Mobile App (`App/`)

- Flutter-based product layer.
- Handles OCR, user input, navigation, and local history.
- Uses Riverpod notifiers + repository/controller pattern.

### 2) Backend API (`Backend/`)

- FastAPI service exposing retrieval, QA, and verifier routes.
- Shared retriever is initialized at startup and reused in-process.
- Verifier supports multiple model interfaces and confidence thresholding.

### 3) Model Workspace (`Model/`)

- Data ingestion (WHO/WebMD-style adapters).
- FAISS embedding/index pipelines (full and incremental).
- Scheduler jobs for scraping and index updates.
- Labeling workflows for dataset generation and stance pipelines.

## Repository Layout

```text
VeriFact/
├─ App/          # Flutter app (UI, OCR, state, API clients)
├─ Backend/      # FastAPI backend (retrieval, QA, verifier APIs)
├─ Model/        # Model/data workspace (harvester, indexing, labeling)
├─ Assets/       # Brand, screenshots, architecture assets
└─ Reports/      # Project reports
```

## Quick Start

### Run the app (dev)

```bash
cd App
flutter pub get
flutter run -t lib/main_dev.dart --flavor dev
```

### Run backend API

```bash
cd Backend
uvicorn main:app --reload
```

### Run model retriever API

```bash
cd Model
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Bootstrap model stack (PowerShell)

```powershell
cd Model
./scripts/bootstrap.ps1
```

## Backend API Surface

- `POST /retrieval/search`
- `POST /retrieval/clear-cache`
- `GET /qa/health`
- `POST /qa/answer`
- `GET /verifier/health`
- `POST /verifier/verify`
- `GET /health`

## Retrieval and Evidence Strategy

Retrieval quality is built through a hybrid ranking strategy:

- vector search (semantic relevance),
- Elasticsearch lexical matching,
- reciprocal rank fusion,
- cross-encoder reranking,
- domain/freshness/metadata-aware final scoring.

This improves factual grounding and reduces dependence on any single ranking signal.

## Typical Team Workflows

### Mobile + backend integration

1. Start backend API.
2. Configure app flavor or base URL override.
3. Run app and validate verifier/QA/doc flows.

### Retrieval refresh cycle

1. Run model scraping/ingestion jobs.
2. Run full or incremental indexing.
3. Reindex Elasticsearch if needed.
4. Restart serving layers where required.

### Labeling pipeline

1. Prepare claims input files.
2. Run `Model/labelling/label_passages.py` workflows.
3. Export JSON/CSV training artifacts.

## Where To Read Next

- App deep dive: `App/readme.md`
- Backend deep dive: `Backend/README.md`
- Model deep dive: `Model/README.md`

---

For new contributors: read this file first, then App -> Backend -> Model.
