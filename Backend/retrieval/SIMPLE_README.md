# Simple Retrieval System

**One file. One process. No complexity.**

## Overview

The entire retrieval system is now contained in a single file: `simple_retriever.py`

**Single Process Flow:**
```
Query → FAISS Search → ElasticSearch → RRF Fusion → Cross-Encoder → Multi-Signal Scoring → Results
```

No alternatives. No optional paths. Just one clear flow.

---

## Quick Start

### 1. One-Time Setup (5 minutes)

```bash
# Start ElasticSearch
docker run -d --name elasticsearch \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -p 9200:9200 \
  docker.elastic.co/elasticsearch/elasticsearch:8.5.0
```

```python
# Create index and load passages
from simple_retriever import create_elasticsearch_index, index_passages_to_elasticsearch
import json

# Create index
create_elasticsearch_index()

# Load passages
with open('storage/outputs/combined/metadata.json', 'r') as f:
    passages = json.load(f)

# Index them
index_passages_to_elasticsearch(passages)
```

### 2. Search (Every Time)

```python
from models import ModelManager
from simple_retriever import SimpleRetriever
from pathlib import Path

# Initialize once
model_manager = ModelManager(Path("storage/outputs/combined"), {})
model_manager.initialize_all()

retriever = SimpleRetriever(model_manager, "storage/outputs/combined")

# Search multiple times
results = retriever.search("vaccine safety during pregnancy")

# Results are ready to use
for result in results:
    print(f"Score: {result['final_score']:.3f}")
    print(f"Title: {result['passage']['title']}")
    print(f"Domain: {result['passage']['domain_tier']}")
```

---

## Configuration

All settings are class constants at the top of `SimpleRetriever`:

```python
# Want to change retrieval numbers?
FAISS_TOPK = 200
ES_TOPK = 200
FINAL_TOPK = 10

# Want to adjust scoring weights?
WEIGHT_FAISS = 0.25
WEIGHT_CROSS_ENCODER = 0.40
WEIGHT_ENTITY_MATCH = 0.10      # Medical entity matching (using SapBERT)
WEIGHT_LEXICAL = 0.10
WEIGHT_DOMAIN = 0.10
WEIGHT_FRESHNESS = 0.05

# Want to change quality threshold?
MIN_SCORE = 0.4

# Want to add domain authority?
DOMAIN_SCORES = {
    'who.int': 1.0,
    'cdc.gov': 1.0,
    # ... add more
}
```

Just edit the class constants. No separate config files.

---

## What's Inside

The `simple_retriever.py` file contains:

1. **SimpleRetriever class** - Everything in one place
   - All configuration (class constants)
   - All search logic (single `search()` method)
   - All scoring components (inline)
   - All utilities (embedded)

2. **Setup utilities** - Functions to create ES index and load passages

3. **Example usage** - Copy-paste ready code

**That's it. One file. 600 lines. Self-contained.**

---

## The Process

When you call `retriever.search("query")`:

```
[1/7] FAISS semantic search
      ↓ 200 candidates

[2/7] ElasticSearch lexical search
      ↓ 200 candidates

[3/7] RRF fusion (combine rankings)
      ↓ 100 unique passages

[4/7] Cross-encoder reranking
      ↓ relevance scores

[5/7] Multi-signal scoring (6 components)
      ↓ final scores

[6/7] Deduplication
      ↓ remove duplicates

[7/7] Filter by MIN_SCORE
      ↓ top 10 results
```

Progress printed at each step. No mystery.

---

## What Changed From Before

**Before:** 
- 12 files (config.py, retriever.py, domain_classifier.py, elasticsearch_wrapper.py, scoring.py, etc.)
- Multiple alternatives (hybrid mode, FAISS-only mode, with ES, without ES)
- Complex configuration system
- Interlinked dependencies
- Hard to understand flow

**Now:**
- 1 file (`simple_retriever.py`)
- 1 process (no alternatives)
- Constants at top (no separate config)
- Self-contained (no dependencies)
- Clear linear flow

---

## Scoring Breakdown

Every result has a `final_score` composed of:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| FAISS | 25% | Semantic similarity |
| Cross-Encoder | 40% | Relevance ranking |
| Entity Match | 10% | Medical entity overlap (SapBERT) |
| Lexical | 10% | Exact word matches |
| Domain | 10% | Source authority |
| Freshness | 5% | How recent |

Plus bonuses:
- Medical review: +0.10
- Has author: +0.02

---

## Domain Tiers

Sources are classified as:

- **Gold** (1.0): WHO, CDC, NIH, FDA
- **Silver** (0.95): Mayo Clinic, Johns Hopkins, Harvard
- **Bronze** (0.85): WebMD, Healthline
- **Default** (0.60): Unknown sources

Add more in `DOMAIN_SCORES` dict.

---

## SapBERT Entity Extraction

The system uses **SapBERT (Specialized Biomedical BERT)** for **medical entity extraction** combined with your **MeSH (Medical Subject Headings) ontology**.

### How It Works

**Entity Matching Process:**

1. **MeSH Keyword Extraction** (always available)
   - Fast dictionary lookup against your **21,887 MeSH concepts**
   - **206,133 indexed keyword terms** (preferred terms + synonyms)
   - Covers diseases, treatments, diagnostics, symptoms, etc.

2. **SapBERT Enhancement** (optional, more accurate)
   - Uses cambridgeltl/SapBERT-from-PubMedBERT for semantic entity matching
   - Finds medical terms even with different wording or phrasing
   - Example: Finds "shortness of breath" when looking for "dyspnea"

3. **Pre-computed Embeddings** (optional)
   - Uses pre-computed SapBERT embeddings from `storage/seeds/embeddings/sapbert_embeddings.npy`
   - Speeds up semantic matching if available

**Entity Scoring:**

For a query "vaccine safety":
- Extracts entities from MeSH: `{mesh:D014687, mesh:D023241, ...}`
- Scores passages by MeSH entity overlap (Jaccard similarity)
- Passage with matching vaccine + safety concepts gets high entity match score (0.5-0.95)
- Passage without matching medical entities gets lower score (0.0-0.3)

### Why MeSH + SapBERT?

✅ **Comprehensive Medical Coverage:**
- Uses your existing MeSH ontology: **21,887 medical concepts**
- **206,133 keyword terms** for fast matching
- Covers entire spectrum of medical knowledge

✅ **Semantic Matching:**
- SapBERT trained on biomedical literature
- Understands medical terminology and synonyms
- Recognizes related concepts

✅ **Fast + Accurate:**
- Keyword matching for speed (O(1) lookup)
- SapBERT for semantic accuracy when available
- Pre-computed embeddings for efficiency

### Why Not General Embedding?

❌ **Don't use SapBERT for general passage embedding:**
- SapBERT embeddings ≠ all-mpnet-base-v2 embeddings
- FAISS index built with all-mpnet-base-v2 (incompatible semantic spaces)
- Would compare apples to oranges

✅ **Do use SapBERT for entity extraction:**
- Designed specifically for biomedical text
- Recognizes medical concepts better than general models
- Boosts relevance for medical fact-checking
- Falls back gracefully to keyword matching if model unavailable

---

## Troubleshooting

**Q: ElasticSearch won't connect**
```bash
# Check if running
docker ps | grep elasticsearch

# Check logs
docker logs elasticsearch

# Restart
docker restart elasticsearch
```

**Q: No results returned**
- Lower `MIN_SCORE` threshold
- Check if passages are indexed in ES
- Verify FAISS index exists

**Q: Results are wrong**
- Adjust scoring weights in class constants
- Check domain scores for your sources
- Review freshness thresholds

**Q: Too slow**
- Reduce `FAISS_TOPK` and `ES_TOPK`
- Increase `MIN_SCORE` to filter more aggressively

---

## That's It

No complex documentation. No multiple guides. Just:

1. **Read this file** (you just did)
2. **Run setup** (5 minutes)
3. **Search** (one method call)

Simple.
