# Quick Reference Guide

## Files Overview

### simple_retriever.py (38 KB)
Main retrieval system with 1000+ lines of well-organized code.

**Key Classes:**
- `MinimalModelManager`: Loads models, MeSH concepts, FAISS index
- `SimpleRetriever`: Main 7-step retrieval pipeline

**Key Functions:**
- `create_elasticsearch_index()`: Creates ES index with medical analyzer
- `index_passages_to_elasticsearch()`: Bulk indexes passages (progress reporting)

### run_retriever.py (5.8 KB)
User-facing interface for setup and search.

**Key Functions:**
- `setup_elasticsearch()`: One-time setup (index creation + passage loading)
- `search_and_display()`: Search and display formatted results

---

## Quick Start

### 1. One-Time Setup
```bash
# Start ElasticSearch
docker run -d --name elasticsearch \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -p 9200:9200 \
  docker.elastic.co/elasticsearch/elasticsearch:9.2.4

# Setup indexing
python run_retriever.py setup
```

### 2. Search
```bash
python run_retriever.py search "vaccine safety during pregnancy"
```

### 3. Programmatic Use
```python
from simple_retriever import SimpleRetriever, MinimalModelManager
from pathlib import Path

# Initialize
model_manager = MinimalModelManager(Path("../../storage/outputs/combined"))
retriever = SimpleRetriever(model_manager, "../../storage/outputs/combined")

# Search
results = retriever.search("query here")

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['final_score']:.3f}")
    print(f"   Title: {result['passage']['title']}")
    
# Free memory (optional)
retriever.clear_entity_cache()
```

---

## Retrieval Pipeline (7 Steps)

```
[1/7] FAISS semantic search      → 200 candidates
[2/7] ElasticSearch lexical      → 200 candidates
[3/7] RRF fusion                 → 100 unique passages
[4/7] Cross-encoder reranking    → relevance scores
[5/7] Multi-signal scoring       → final scores
[6/7] Deduplication              → remove ~10-20% duplicates
[7/7] Filter & return            → Top 10 results
```

---

## Scoring Components (6 + 2)

| Component | Weight | Source |
|-----------|--------|--------|
| FAISS | 25% | Semantic similarity |
| Cross-Encoder | 40% | Relevance ranking |
| MeSH Entities | 10% | Medical entity overlap |
| Lexical | 10% | Exact word matches |
| Domain | 10% | Source authority |
| Freshness | 5% | Publication date |
| **Bonuses** | | |
| Medical Review | +0.10 | If reviewed by MD |
| Author | +0.02 | If author specified |

---

## Configuration

All settings in `SimpleRetriever` class constants:

```python
# Retrieval
FAISS_TOPK = 200          # Candidates from FAISS
ES_TOPK = 200             # Candidates from ES
FINAL_TOPK = 10           # Final results

# RRF
RRF_K = 60
RRF_WEIGHT_FAISS = 0.5
RRF_WEIGHT_ES = 0.5

# Scoring weights (sum = 1.0)
WEIGHT_FAISS = 0.25
WEIGHT_CROSS_ENCODER = 0.40
WEIGHT_ENTITY_MATCH = 0.10
WEIGHT_LEXICAL = 0.10
WEIGHT_DOMAIN = 0.10
WEIGHT_FRESHNESS = 0.05

# Filtering
MIN_SCORE = 0.4
DEDUP_THRESHOLD = 0.92

# Bonuses
MEDICAL_REVIEW_BONUS = 0.10
AUTHOR_BONUS = 0.02

# Domains (3-tier: gold/silver/bronze/default)
DOMAIN_SCORES = {...}

# Freshness (days)
FRESHNESS_RECENT = 365      # 1 year
FRESHNESS_MODERATE = 1095   # 3 years
FRESHNESS_OLD = 1825        # 5 years
```

---

## Medical Entity Extraction

Using 21,887 MeSH concepts with 206,133 keyword terms.

**Two-stage extraction:**
1. **Keyword matching** (fast, always available)
   - Matches against preferred terms + synonyms
   - O(1) lookup

2. **SapBERT semantic matching** (accurate, optional)
   - Uses biomedical BERT model
   - Finds synonyms and related concepts
   - With caching for performance

---

## Data Sources

### Input
- Passages: `storage/outputs/combined/metadata.json` (363,511 passages)
- FAISS index: `storage/outputs/combined/index.faiss`
- MeSH concepts: `storage/mesh_concepts.json` (21,887 concepts)
- Pre-computed embeddings: `storage/seeds/embeddings/sapbert_embeddings.npy` (optional)

### Output
- ElasticSearch index: `medical_passages` (localhost:9200)
- Search results: Python dictionaries with scores & metadata

---

## Performance

### Indexing
- 363,511 passages → ~30 seconds
- Batch size: 1000 documents
- With progress reporting

### Searching
- Query → Results: ~2-5 seconds
- Entity extraction: <1ms with caching
- FAISS+ES combined: ~200ms
- Cross-encoder reranking: ~500ms

### Memory
- FAISS index: ~500MB
- Passages in memory: ~200MB
- Models: ~1GB
- Entity cache: <50MB (cleared on demand)

---

## Troubleshooting

### ElasticSearch won't connect
```bash
# Check if running
docker ps | grep elasticsearch

# Check logs
docker logs elasticsearch

# Restart
docker restart elasticsearch

# Wait 30 seconds for startup
```

### No results returned
- Lower `MIN_SCORE` threshold (currently 0.4)
- Check ES index with: `curl http://localhost:9200/medical_passages/_count`
- Verify FAISS index exists

### Too slow
- Reduce `FAISS_TOPK` and `ES_TOPK`
- Increase `MIN_SCORE` for more aggressive filtering
- Clear entity cache: `retriever.clear_entity_cache()`

### Wrong results
- Adjust scoring weights in `SimpleRetriever` constants
- Review domain scores for your sources
- Check freshness thresholds

---

## API Reference

### SimpleRetriever Methods

**Main:**
- `search(query: str) → List[Dict]`: Execute full 7-step retrieval

**Utilities:**
- `clear_entity_cache()`: Free cached entity extractions
- `_connect_elasticsearch()`: Reconnect to ES

**Internal:**
- `_faiss_search()`: Step 1 - Semantic search
- `_elasticsearch_search()`: Step 2 - Lexical search
- `_rrf_fusion()`: Step 3 - Combine rankings
- `_cross_encoder_rerank()`: Step 4 - Reranking
- `_compute_scores()`: Step 5 - Scoring
- `_deduplicate()`: Step 6 - Remove duplicates
- `_extract_medical_entities()`: Entity extraction (with caching)
- `_compute_entity_match_score()`: Jaccard similarity for entities
- `_get_domain_score()`: Domain authority lookup
- `_get_freshness_score()`: Date-based scoring

### Utility Functions

**Setup:**
- `create_elasticsearch_index(host, port, name)`: Create ES index
- `index_passages_to_elasticsearch(passages, host, port, name, batch_size)`: Bulk index

---

## Result Format

```python
{
    'final_score': 0.85,
    'scores': {
        'faiss': 0.82,
        'cross_encoder': 0.88,
        'entity_match': 0.75,
        'lexical': 0.60,
        'domain': 0.95,
        'freshness': 1.0
    },
    'passage': {
        'id': 'unique-id',
        'title': 'Article Title',
        'text': 'Full passage text...',
        'url': 'https://source.com/article',
        'author': 'Dr. Smith',
        'medically_reviewed_by': 'Dr. Jones',
        'published_date': '2024-01-15',
        'section_heading': 'Treatment',
        'domain_tier': 'gold'  # Added by retriever
    },
    'faiss_score': 0.82,
    'faiss_rank': 1,
    'es_score': 0.55,
    'es_rank': 45,
    'rrf_score': 0.72
}
```

---

## Important Notes

1. **Entity Extraction**: Uses MeSH ontology with 21,887 concepts
2. **Caching**: Automatically caches entity extractions for performance
3. **Backward Compatible**: All changes are 100% backward compatible
4. **Type Safe**: Type validation in SimpleRetriever initialization
5. **Error Handling**: Graceful fallbacks for missing models

