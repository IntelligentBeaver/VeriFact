# VeriFact Admin Tools

Administrative tools for labeling passages and building training data for the medical fact verification system.

## Quick Start

```bash
cd harvester/admin
pip install sentence-transformers faiss-cpu numpy
python label_passages.py
```

## What’s Here

- **label_passages.py** — interactive CLI to label passages as RELEVANT or UNRELATED
- **config.py** — central settings (models, paths, search params)
- **output/** — labeled data written here (created automatically)

## How the Labeler Works

1. Load index files from `harvester/storage/outputs/webmd/faiss` (FAISS index, embeddings, metadata).
2. Embed your query with the biomedical model and retrieve top passages via FAISS.
3. Rerank with the cross-encoder; optionally use SapBERT embeddings for medical entity boost.
4. Show you each passage (title, section, URL, scores, truncated text).
5. You mark each passage as RELEVANT or UNRELATED; results are saved immediately.

## Usage Flow

1. **Run**: `python label_passages.py`
2. **Choose option 1** to start labeling
3. **Enter a medical query/claim** (e.g., "Does vitamin D prevent COVID?")
4. **Review top passages** (defaults: retrieve 100, show 10, rerank top 20)
5. **Answer y/n** for each passage → files update in `output/`
6. **Option 2** shows totals; option 3 reserved for exports; option 4 exits

## Configuration (config.py)

```python
# Models
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Search defaults
DEFAULT_TOPK_FAISS = 100      # retrieve from FAISS
DEFAULT_RERANK_K = 20         # rerank this many with cross-encoder
DEFAULT_DISPLAY_RESULTS = 10  # passages shown per session

# Paths (auto-set relative to this file)
INDEX_DIR = HARVESTER_DIR / "storage" / "outputs" / "webmd" / "faiss"
OUTPUT_DIR = ADMIN_DIR / "output"
```

## Outputs

Written to `output/` during labeling:
- `labeled_relevant_passages.json` — passages marked relevant
- `labeled_unrelated_passages.json` — passages marked unrelated
- `labeling_session.json` — last session details

Example record:

```json
{
  "passage_id": "webmd_p12345",
  "query": "Does vitamin D prevent COVID?",
  "relevant": true,
  "text": "Recent studies show...",
  "title": "Vitamin D and COVID",
  "url": "webmd.com/...",
  "author": "Dr. Jane Smith",
  "medically_reviewed_by": "Medical Review Board",
  "scores": {
    "faiss_similarity": 0.72,
    "cross_encoder_relevance": 0.90
  },
  "labeled_at": "2024-01-15T10:30:45.123456"
}
```

## Scoring Signals (handled automatically)

- **FAISS similarity** (25% weight): vector match between query and passage
- **Cross-encoder relevance** (70% weight): fine-grained relevance
- **SapBERT entity boost** (5% weight): medical concept overlap

## Good Queries vs Bad Queries

- Good: "Does zinc shorten colds?" | "Is Ozempic safe long term?" | "Do masks reduce COVID spread?"
- Bad: "health" (too broad) | empty input

## Common Issues

- **metadata.json not found** → run the index build first (`build_webmd_faiss_biomed.py`).
- **Model download fails** → check internet; first run downloads ~500MB.
- **Slow startup** → models load once; searches are fast afterward.
- **Memory pressure** → lower `DEFAULT_TOPK_FAISS` to 50.

## File Map

- [harvester/admin/label_passages.py](harvester/admin/label_passages.py) — interactive labeling workflow
- [harvester/admin/config.py](harvester/admin/config.py) — models, paths, defaults
- [harvester/admin/output/](harvester/admin/output/) — labeled data artifacts
