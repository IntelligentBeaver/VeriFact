"""
Simplified Hybrid Retrieval System
Single file, single process flow, self-contained.

Process: Query → [FAISS + ElasticSearch] → RRF Fusion → Cross-Encoder → Combined Score → Results
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime
from urllib.parse import urlparse
from elasticsearch import Elasticsearch
import time
import requests
from config import load_retriever_config


class MinimalModelManager:
    """Minimal model manager for the simplified retriever."""
    
    def __init__(self, index_dir: str):
        """Initialize with just what we need."""
        from sentence_transformers import SentenceTransformer, CrossEncoder
        import faiss
        import numpy as np
        import json
        from pathlib import Path
        import os
        
        self.index_dir = Path(index_dir).resolve()  # Make absolute path
        
        print(f"\n  Index directory: {self.index_dir}")
        print(f"  Directory exists: {self.index_dir.exists()}")
        
        if self.index_dir.exists():
            print(f"  Files in directory:")
            for f in self.index_dir.iterdir():
                print(f"    - {f.name}")
        
        # Load embedding model
        print("  Loading embedding model (all-mpnet-base-v2)...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Load cross-encoder for reranking (use public ranking model)
        print("  Loading cross-encoder...")
        try:
            # Try to load the better model first
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
        except Exception as e:
            print(f"  Warning: Could not load cross-encoder, using fallback...")
            self.cross_encoder = None
        
        # Load SapBERT for medical entity extraction
        print("  Loading SapBERT for entity extraction...")
        try:
            self.sapbert_model = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token')
        except Exception as e:
            print(f"  Warning: Could not load SapBERT, entity extraction will be keyword-based...")
            self.sapbert_model = None
        
        # Load MeSH concepts from storage/seeds directory
        print("  Loading MeSH medical concepts...")
        mesh_concepts_path = Path(__file__).parent.parent / "storage" / "mesh_concepts.json"
        self.mesh_concepts = {}
        self.mesh_keywords = {}  # Map keywords to MeSH IDs
        
        try:
            if mesh_concepts_path.exists():
                with open(mesh_concepts_path, 'r', encoding='utf-8') as f:
                    self.mesh_concepts = json.load(f)
                
                # Build keyword index for fast lookup
                for mesh_id, concept in self.mesh_concepts.items():
                    # Index preferred term
                    if 'preferred_term' in concept:
                        term = concept['preferred_term'].lower()
                        if term not in self.mesh_keywords:
                            self.mesh_keywords[term] = []
                        self.mesh_keywords[term].append(mesh_id)
                    
                    # Index synonyms
                    for synonym in concept.get('synonyms', []):
                        syn = synonym.lower()
                        if syn not in self.mesh_keywords:
                            self.mesh_keywords[syn] = []
                        self.mesh_keywords[syn].append(mesh_id)
                
                print(f"    Loaded {len(self.mesh_concepts)} MeSH concepts")
                print(f"    Built keyword index: {len(self.mesh_keywords)} terms")
            else:
                print(f"    Warning: MeSH concepts file not found at {mesh_concepts_path}")
                print(f"    Entity extraction will use substring matching only")
        except Exception as e:
            print(f"    Warning: Could not load MeSH concepts: {e}")
        
        # Load pre-computed SapBERT embeddings if available
        self.sapbert_embeddings = None
        embeddings_path = Path(__file__).parent.parent / "storage" / "seeds" / "embeddings" / "sapbert_embeddings.npy"
        try:
            if embeddings_path.exists():
                self.sapbert_embeddings = np.load(str(embeddings_path))
                print(f"    Loaded pre-computed SapBERT embeddings: {self.sapbert_embeddings.shape}")
        except Exception as e:
            print(f"    Note: Pre-computed embeddings not available: {e}")
        
        # Load FAISS index
        print("  Loading FAISS index...")
        faiss_path = self.index_dir / "index.faiss"
        passages_path = self.index_dir / "metadata.json"
        
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {self.index_dir}")
        
        if not faiss_path.exists():
            print(f"  Warning: FAISS index not found at {faiss_path}")
            print(f"  Expected path: {faiss_path.absolute()}")
            raise FileNotFoundError(f"FAISS index not found. Try running setup first.")
        
        self.faiss_index = faiss.read_index(str(faiss_path))
        
        # Load passages
        print("  Loading passages...")
        if not passages_path.exists():
            raise FileNotFoundError(f"Passages not found at {passages_path}")
        
        with open(passages_path, 'r', encoding='utf-8') as f:
            self.passages = json.load(f)
        
        print(f"✓ Models loaded successfully")
        print(f"  - FAISS index: {len(self.passages)} passages")
        print(f"  - Embedding model: all-mpnet-base-v2")
        print(f"  - Cross-encoder: {'cross-encoders/ms-marco-MiniLM-L6-v2' if self.cross_encoder else 'DISABLED'}")
        print(f"  - SapBERT: {'Entity extraction enabled' if self.sapbert_model else 'Keyword-based extraction'}")
        print(f"  - MeSH concepts: {len(self.mesh_concepts)} loaded")


class SimpleRetriever:
    """
    Self-contained hybrid retrieval system.
    Combines FAISS semantic search + ElasticSearch lexical search.
    """
    
    # ============================================================================
    # CONFIGURATION - All settings in one place
    # ============================================================================
    
    # ElasticSearch settings
    ES_HOST = "127.0.0.1"
    ES_PORT = 9200
    ES_INDEX = "medical_passages"
    
    # Retrieval settings
    FAISS_TOPK = 50          # Get 50 candidates from FAISS
    ES_TOPK = 50           # Get 50 candidates from ElasticSearch
    FINAL_TOPK = 5          # Return top 10 results

    # Feature flags (performance vs quality)
    ENABLE_CROSS_ENCODER = True
    ENABLE_ENTITY_MATCH = False
    ENABLE_SAPBERT_SEMANTIC = True  # Set True for higher accuracy (slower)
    
    # RRF Fusion settings
    RRF_K = 60
    RRF_WEIGHT_FAISS = 0.5
    RRF_WEIGHT_ES = 0.5
    
    # Scoring weights (must sum to 1.0)
    WEIGHT_FAISS = 0.25
    WEIGHT_CROSS_ENCODER = 0.40
    WEIGHT_ENTITY_MATCH = 0.10    # Medical entity matching bonus (was SapBERT embedding)
    WEIGHT_LEXICAL = 0.10
    WEIGHT_DOMAIN = 0.10
    WEIGHT_FRESHNESS = 0.05
    
    # Quality filters
    MIN_SCORE = 0.4
    
    # Bonuses
    MEDICAL_REVIEW_BONUS = 0.10
    AUTHOR_BONUS = 0.02
    
    # Domain authority (3-tier system)
    DOMAIN_SCORES = {
        # Gold tier - Maximum authority
        'who.int': 1.0,
        'cdc.gov': 1.0,
        'nih.gov': 1.0,
        'fda.gov': 1.0,
        
        # Silver tier - High authority
        'mayoclinic.org': 0.95,
        'hopkinsmedicine.org': 0.95,
        'health.harvard.edu': 0.95,
        'clevelandclinic.org': 0.90,
        'jamanetwork.com': 0.90,
        
        # Bronze tier - Good authority
        'webmd.com': 0.85,
        'healthline.com': 0.80,
        'medicalnewstoday.com': 0.80,
        'medlineplus.gov': 0.85,
        
        # Default for unknown sources
        'default': 0.60
    }
    
    # Freshness thresholds (days)
    FRESHNESS_RECENT = 365      # < 1 year = recent (score 1.0)
    FRESHNESS_MODERATE = 1095   # < 3 years = moderate (score 0.7)
    FRESHNESS_OLD = 1825        # < 5 years = old (score 0.4)
    # > 5 years = very old (score 0.1)
    
    
    def __init__(self, model_manager, index_dir: str):
        """
        Initialize retriever with models and index.
        
        Args:
            model_manager: MinimalModelManager instance with initialized models
            index_dir: Path to FAISS index directory
        """
        if not isinstance(model_manager, MinimalModelManager):
            raise TypeError("model_manager must be a MinimalModelManager instance")
        
        self.model_manager = model_manager
        self.index_dir = Path(index_dir)

        # Load centralized retriever config (env vars are respected by loader)
        try:
            cfg = load_retriever_config()
        except Exception:
            cfg = None

        if cfg is not None:
            self.ES_HOST = cfg.es_host
            self.ES_PORT = int(cfg.es_port)
            self.ES_INDEX = cfg.es_index

            # Retrieval tuning
            self.FAISS_TOPK = cfg.faiss_topk
            self.ES_TOPK = cfg.es_topk
            self.FINAL_TOPK = cfg.final_topk

            # RRF
            self.RRF_K = cfg.rrf_k
            self.RRF_WEIGHT_FAISS = cfg.rrf_weight_faiss
            self.RRF_WEIGHT_ES = cfg.rrf_weight_es

            # Scoring weights & thresholds
            self.WEIGHT_FAISS = cfg.weight_faiss
            self.WEIGHT_CROSS_ENCODER = cfg.weight_cross_encoder
            self.WEIGHT_ENTITY_MATCH = cfg.weight_entity_match
            self.WEIGHT_LEXICAL = cfg.weight_lexical
            self.WEIGHT_DOMAIN = cfg.weight_domain
            self.WEIGHT_FRESHNESS = cfg.weight_freshness
            self.MIN_SCORE = cfg.min_score

            # Bonuses
            self.MEDICAL_REVIEW_BONUS = cfg.medical_review_bonus
            self.AUTHOR_BONUS = cfg.author_bonus

            # Domain scores & freshness
            self.DOMAIN_SCORES = cfg.domain_scores or self.DOMAIN_SCORES
            self.FRESHNESS_RECENT = cfg.freshness_recent
            self.FRESHNESS_MODERATE = cfg.freshness_moderate
            self.FRESHNESS_OLD = cfg.freshness_old
        else:
            # fallback to env vars if loader isn't available
            self.ES_HOST = os.getenv("ES_HOST", self.ES_HOST)
            self.ES_PORT = int(os.getenv("ES_PORT", str(self.ES_PORT)))
            self.ES_INDEX = os.getenv("ES_INDEX", self.ES_INDEX)
        
        # Cache for entity extraction (avoid re-extracting same text)
        self._entity_cache = {}
        
        # Initialize ElasticSearch
        self.es = None
        self._connect_elasticsearch()
        
        print(f"✓ SimpleRetriever initialized")
        print(f"  - FAISS index: {self.index_dir}")
        print(f"  - ElasticSearch: {self.ES_HOST}:{self.ES_PORT}")
        print(f"  - Process: FAISS+ES → RRF → Cross-Encoder → Score → Top {self.FINAL_TOPK}")
    
    
    def clear_entity_cache(self):
        """Clear entity extraction cache to free memory."""
        self._entity_cache.clear()
        print(f"✓ Entity cache cleared")
    
    
    def _connect_elasticsearch(self):
        """Connect to ElasticSearch."""
        try:
            self.es = Elasticsearch(
                [f"http://{self.ES_HOST}:{self.ES_PORT}"],
                request_timeout=30,
                verify_certs=False,
                ssl_show_warn=False
            )
            if self.es.ping():
                print(f"✓ Connected to ElasticSearch at {self.ES_HOST}:{self.ES_PORT}")
            else:
                print(f"✗ ElasticSearch not responding at {self.ES_HOST}:{self.ES_PORT}")
                self.es = None
        except Exception as e:
            print(f"✗ ElasticSearch connection failed: {e}")
            self.es = None
    
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Main search method - single, clear process flow.
        
        Process:
        1. FAISS semantic search (200 candidates)
        2. ElasticSearch lexical search (200 candidates)
        3. RRF fusion to combine rankings
        4. Cross-encoder reranking
        5. Multi-signal scoring (6 components)
        6. Deduplication
        7. Return top 10 results
        
        Args:
            query: Search query string
            
        Returns:
            List of scored results with metadata
        """
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Step 1: FAISS semantic search
        print(f"\n[1/7] FAISS semantic search...")
        faiss_results = self._faiss_search(query)
        print(f"  ✓ Retrieved {len(faiss_results)} candidates")
        
        # Step 2: ElasticSearch lexical search
        print(f"\n[2/7] ElasticSearch lexical search...")
        es_results = self._elasticsearch_search(query)
        print(f"  ✓ Retrieved {len(es_results)} candidates")
        
        # Step 3: RRF fusion
        print(f"\n[3/7] RRF fusion...")
        fused_results = self._rrf_fusion(faiss_results, es_results)
        print(f"  ✓ Fused to {len(fused_results)} unique passages")
        
        # Step 4: Cross-encoder reranking
        print(f"\n[4/7] Cross-encoder reranking...")
        reranked_results = self._cross_encoder_rerank(query, fused_results)
        print(f"  ✓ Reranked {len(reranked_results)} passages")
        
        # Step 5: Multi-signal scoring
        print(f"\n[5/7] Computing final scores...")
        scored_results = self._compute_scores(query, reranked_results)
        print(f"  ✓ Scored {len(scored_results)} passages")
        
        # Step 6: Final filtering
        print(f"\n[6/6] Final filtering...")
        final_results = [r for r in scored_results if r['final_score'] >= self.MIN_SCORE]
        final_results = final_results[:self.FINAL_TOPK]
        print(f"  ✓ Returning {len(final_results)} results (min_score={self.MIN_SCORE})")
        
        # Print top 3 results
        print(f"\n{'='*80}")
        print("Top 3 Results:")
        print(f"{'='*80}")
        for i, result in enumerate(final_results[:3], 1):
            p = result['passage']
            print(f"\n{i}. Score: {result['final_score']:.3f} | Domain: {result.get('domain_tier', 'N/A')}")
            print(f"   Title: {p.get('title', 'N/A')}")
            print(f"   Source: {p.get('url', 'N/A')[:70]}")
        
        return final_results
    
    
    def _faiss_search(self, query: str) -> List[Dict[str, Any]]:
        """FAISS semantic search."""
        # Get query embedding
        query_embedding = self.model_manager.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search FAISS index
        faiss_index = self.model_manager.faiss_index
        distances, indices = faiss_index.search(
            query_embedding.reshape(1, -1),
            self.FAISS_TOPK
        )
        
        # Load passages
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:
                continue
            passage = self.model_manager.passages[idx]
            results.append({
                'passage': passage,
                'faiss_score': float(score),
                'faiss_rank': rank + 1,
                'passage_idx': int(idx)
            })
        
        return results
    
    
    def _elasticsearch_search(self, query: str) -> List[Dict[str, Any]]:
        """ElasticSearch BM25 lexical search."""
        if not self.es:
            return []
        
        try:
            response = self.es.search(
                index=self.ES_INDEX,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["text^2", "title^1.5", "section_heading"],
                            "type": "best_fields"
                        }
                    },
                    "size": self.ES_TOPK
                }
            )
            
            results = []
            for rank, hit in enumerate(response['hits']['hits']):
                passage = hit['_source']
                if 'passage_id' not in passage:
                    passage['passage_id'] = hit['_id']
                results.append({
                    'passage': passage,
                    'es_score': hit['_score'],
                    'es_rank': rank + 1,
                    'passage_id': hit['_id']
                })
            
            return results
            
        except Exception as e:
            print(f"  ✗ ElasticSearch error: {e}")
            return []
    
    
    def _rrf_fusion(self, faiss_results: List[Dict], es_results: List[Dict]) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) - combines FAISS and ES rankings.
        Formula: score = w1/(k+rank_faiss) + w2/(k+rank_es)
        """
        # Build passage lookup
        passage_map = {}
        
        # Add FAISS results
        for r in faiss_results:
            pid = (
                r['passage'].get('passage_id')
                or r['passage'].get('id')
                or r['passage'].get('url')
            )
            passage_map[pid] = {
                'passage': r['passage'],
                'faiss_rank': r['faiss_rank'],
                'faiss_score': r['faiss_score'],
                'es_rank': None,
                'es_score': 0.0
            }
        
        # Add ES results
        for r in es_results:
            pid = (
                r['passage'].get('passage_id')
                or r['passage'].get('id')
                or r['passage'].get('url')
                or r.get('passage_id')
            )
            if pid in passage_map:
                passage_map[pid]['es_rank'] = r['es_rank']
                passage_map[pid]['es_score'] = r['es_score']
            else:
                passage_map[pid] = {
                    'passage': r['passage'],
                    'faiss_rank': None,
                    'faiss_score': 0.0,  # Default to 0.0 instead of None
                    'es_rank': r['es_rank'],
                    'es_score': r['es_score']
                }
        
        # Compute RRF scores
        results = []
        for pid, data in passage_map.items():
            rrf_score = 0.0
            
            if data['faiss_rank']:
                rrf_score += self.RRF_WEIGHT_FAISS / (self.RRF_K + data['faiss_rank'])
            
            if data['es_rank']:
                rrf_score += self.RRF_WEIGHT_ES / (self.RRF_K + data['es_rank'])
            
            results.append({
                'passage': data['passage'],
                'rrf_score': rrf_score,
                'faiss_score': data['faiss_score'],
                'es_score': data['es_score'],
                'faiss_rank': data['faiss_rank'],
                'es_rank': data['es_rank']
            })
        
        # Sort by RRF score
        results.sort(key=lambda x: x['rrf_score'], reverse=True)
        return results[:100]  # Keep top 100 for reranking
    
    
    def _cross_encoder_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Cross-encoder reranking for relevance."""
        if not results or not self.ENABLE_CROSS_ENCODER or not self.model_manager.cross_encoder:
            # If no cross-encoder, just return results as-is
            for result in results:
                result['cross_score'] = 0.5  # Neutral score
            return results
        
        # Prepare pairs
        pairs = [[query, r['passage']['text'][:512]] for r in results]  # Limit text length
        
        try:
            # Get cross-encoder scores
            cross_scores = self.model_manager.cross_encoder.predict(pairs)
            
            # Add scores to results
            for result, score in zip(results, cross_scores):
                result['cross_score'] = float(score)
        except Exception as e:
            print(f"  Warning: Cross-encoder failed: {e}")
            for result in results:
                result['cross_score'] = 0.5
        
        return results
    
    
    def _compute_scores(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Multi-signal scoring with 6 components:
        1. FAISS semantic similarity (25%)
        2. Cross-encoder relevance (40%)
        3. Medical entity matching (10%) - using SapBERT entity extraction
        4. Lexical overlap (10%)
        5. Domain authority (10%)
        6. Freshness (5%)
        + Medical review bonus
        + Author bonus
        """
        if not results:
            return results
        
        # Get query tokens for lexical matching
        query_tokens = set(query.lower().split())
        
        # Extract medical entities from query using SapBERT when enabled
        query_entities = self._extract_medical_entities(query) if self.ENABLE_ENTITY_MATCH else set()
        
        for result in results:
            passage = result['passage']
            text = passage.get('text', '')
            
            # 1. Normalize FAISS score (already 0-1 from cosine)
            faiss_score = result.get('faiss_score', 0.0)
            if faiss_score is None:
                faiss_score = 0.0
            
            # 2. Normalize cross-encoder score (sigmoid to 0-1)
            cross_score = result.get('cross_score', 0.0)
            if cross_score is None:
                cross_score = 0.0
            cross_score = 1 / (1 + np.exp(-cross_score))  # sigmoid
            
            # 3. Medical entity matching using SapBERT entity extraction (if enabled)
            if self.ENABLE_ENTITY_MATCH:
                passage_entities = self._extract_medical_entities(text)
                entity_match_score = self._compute_entity_match_score(query_entities, passage_entities)
            else:
                entity_match_score = 0.0
            
            # 4. Lexical overlap
            passage_tokens = set(text.lower().split())
            if passage_tokens:
                lexical_score = len(query_tokens & passage_tokens) / len(query_tokens)
            else:
                lexical_score = 0.0
            
            # 5. Domain authority
            domain_score = self._get_domain_score(passage)
            
            # 6. Freshness
            freshness_score = self._get_freshness_score(passage)
            
            # Weighted combination
            final_score = (
                self.WEIGHT_FAISS * faiss_score +
                self.WEIGHT_CROSS_ENCODER * cross_score +
                self.WEIGHT_ENTITY_MATCH * entity_match_score +
                self.WEIGHT_LEXICAL * lexical_score +
                self.WEIGHT_DOMAIN * domain_score +
                self.WEIGHT_FRESHNESS * freshness_score
            )
            
            # Apply bonuses
            if passage.get('medically_reviewed_by'):
                final_score += self.MEDICAL_REVIEW_BONUS
            
            if passage.get('author'):
                final_score += self.AUTHOR_BONUS
            
            # Cap at 1.0
            final_score = min(final_score, 1.0)
            
            # Store all scores
            result['final_score'] = final_score
            result['domain_tier'] = passage.get('domain_tier', 'N/A')  # Copy from passage
            result['scores'] = {
                'faiss': faiss_score,
                'cross_encoder': cross_score,
                'entity_match': entity_match_score,
                'lexical': lexical_score,
                'domain': domain_score,
                'freshness': freshness_score
            }
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
    
    
    def _extract_medical_entities(self, text: str) -> Set[str]:
        """
        Extract medical entities from text using MeSH concepts and SapBERT.
        
        Uses caching to avoid re-extracting entities from same text.
        
        Process:
        1. Keyword matching against MeSH concept terms (fast)
        2. Optional SapBERT embedding similarity for semantic matching (accurate)
        
        Returns set of matched MeSH IDs.
        """
        # Check cache first
        text_hash = hash(text[:256])  # Hash first 256 chars for cache key
        if text_hash in self._entity_cache:
            return self._entity_cache[text_hash]
        
        text_lower = text.lower()
        matched_mesh_ids = set()
        
        # Step 1: Keyword matching against MeSH concept index
        # Check both preferred terms and synonyms
        for keyword, mesh_ids in self.model_manager.mesh_keywords.items():
            if keyword in text_lower:
                # Add all MeSH IDs for this keyword
                matched_mesh_ids.update(mesh_ids)
        
        # Step 2: SapBERT-based semantic matching (optional, more accurate)
        # Only run SapBERT semantic matching when the flag is enabled and the model exists
        if self.ENABLE_SAPBERT_SEMANTIC and self.model_manager.sapbert_model is not None and self.model_manager.mesh_concepts:
            try:
                # Encode text using SapBERT
                text_embedding = self.model_manager.sapbert_model.encode(
                    text[:512],  # Limit length for efficiency
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                # Check semantic similarity to MeSH preferred terms
                for mesh_id, concept in self.model_manager.mesh_concepts.items():
                    preferred_term = concept.get('preferred_term', '')
                    if preferred_term:
                        try:
                            term_embedding = self.model_manager.sapbert_model.encode(
                                preferred_term[:128],
                                convert_to_numpy=True,
                                normalize_embeddings=True
                            )
                            
                            # Compute cosine similarity
                            similarity = float(np.dot(text_embedding, term_embedding))
                            
                            # High threshold for medical terms (SapBERT typically 0.5-1.0 for medical)
                            if similarity > 0.65:
                                matched_mesh_ids.add(mesh_id)
                        except:
                            pass
            except Exception as e:
                # Fallback to keyword-only if SapBERT fails
                pass
        
        # Cache result
        self._entity_cache[text_hash] = matched_mesh_ids
        return matched_mesh_ids
    
    
    def _compute_entity_match_score(self, query_entities: Set[str], passage_entities: Set[str]) -> float:
        """
        Compute score based on MeSH medical entity matching.
        
        Returns higher score when passage contains more of the medical entities from the query.
        This is critical for medical fact-checking - matching on actual medical concepts.
        
        Uses Jaccard similarity to measure entity overlap:
        - Perfect overlap (all query entities in passage) → 0.95
        - Partial overlap → 0.5 + (0.4 * jaccard_score)
        - No overlap → 0.0 or low (0.3 if passage has medical content)
        """
        if not query_entities:
            # No medical entities in query
            if passage_entities:
                return 0.3  # Slight boost if passage has medical content
            else:
                return 0.5  # Neutral if neither has medical entities
        
        # Compute Jaccard similarity (intersection / union) for MeSH entities
        intersection = len(query_entities & passage_entities)
        union = len(query_entities | passage_entities)
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        # Boost for high overlap (medical relevance)
        if intersection == len(query_entities):
            # Passage contains all query entities
            return 0.95
        elif intersection > 0:
            # Passage contains some query entities
            return 0.5 + (0.4 * jaccard_score)
        else:
            # No matching entities
            return 0.0
    
    
    def _get_domain_score(self, passage: Dict) -> float:
        """Get domain authority score."""
        url = passage.get('url', '')
        if not url:
            return self.DOMAIN_SCORES['default']
        
        # Extract domain
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '')
        except:
            return self.DOMAIN_SCORES['default']
        
        # Look up score
        score = self.DOMAIN_SCORES.get(domain, self.DOMAIN_SCORES['default'])
        
        # Store domain tier for display
        if score >= 0.95:
            passage['domain_tier'] = 'gold'
        elif score >= 0.85:
            passage['domain_tier'] = 'silver'
        else:
            passage['domain_tier'] = 'bronze'
        
        return score
    
    
    def _get_freshness_score(self, passage: Dict) -> float:
        """Get freshness score based on published date."""
        published_date = passage.get('published_date')
        if not published_date:
            return 0.5  # Neutral for unknown dates
        
        try:
            # Parse date
            if isinstance(published_date, str):
                pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            else:
                pub_date = published_date
            
            # Calculate age in days
            age_days = (datetime.now(pub_date.tzinfo) - pub_date).days
            
            # Score based on age
            if age_days < self.FRESHNESS_RECENT:
                return 1.0
            elif age_days < self.FRESHNESS_MODERATE:
                return 0.7
            elif age_days < self.FRESHNESS_OLD:
                return 0.4
            else:
                return 0.1
                
        except:
            return 0.5
    
    


# ============================================================================
# SETUP UTILITIES
# ============================================================================

def create_elasticsearch_index(es_host="127.0.0.1", es_port=9200, index_name="medical_passages"):
    """Create ElasticSearch index with medical analyzer."""
    
    # First, test with basic HTTP request
    print(f"Testing connection to {es_host}:{es_port}...")
    try:
        response = requests.get(f"http://{es_host}:{es_port}/", timeout=5)
        if response.status_code == 200:
            print(f"✓ HTTP connection works")
        else:
            print(f"✗ HTTP connection failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Cannot reach {es_host}:{es_port}: {e}")
        print(f"\nTry using 127.0.0.1 instead of localhost")
        es_host = "127.0.0.1"
        print(f"Retrying with {es_host}...")
    
    try:
        es = Elasticsearch(
            [f"http://{es_host}:{es_port}"],
            request_timeout=10,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Try to ping with retries
        print(f"Checking ElasticSearch client at {es_host}:{es_port}...")
        for attempt in range(5):
            try:
                if es.ping():
                    print(f"✓ Connected to ElasticSearch")
                    break
                else:
                    if attempt < 4:
                        print(f"  Waiting... (attempt {attempt+1}/5)")
                        time.sleep(3)
                    else:
                        raise Exception("Could not connect to ElasticSearch")
            except Exception as ping_error:
                print(f"  Ping error: {ping_error}")
                if attempt < 4:
                    print(f"  Retrying... (attempt {attempt+1}/5)")
                    time.sleep(3)
                else:
                    raise Exception(f"Could not connect to ElasticSearch: {ping_error}")
        
        # Delete if exists
        try:
            if es.indices.exists(index=index_name):
                es.indices.delete(index=index_name)
                print(f"✓ Deleted existing index: {index_name}")
        except Exception as e:
            print(f"  Note: Could not delete existing index: {e}")
        
        # Create with settings
        es.indices.create(
            index=index_name,
            body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "medical_analyzer": {
                                "type": "standard",
                                "stopwords": "_english_"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text", "analyzer": "medical_analyzer"},
                        "title": {"type": "text", "analyzer": "medical_analyzer"},
                        "section_heading": {"type": "text"},
                        "url": {"type": "keyword"},
                        "author": {"type": "text"},
                        "medically_reviewed_by": {"type": "text"},
                        "published_date": {"type": "date"},
                    }
                }
            }
        )
        print(f"✓ Created index: {index_name}")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        print(f"\nElasticSearch is not responding at {es_host}:{es_port}")
        print(f"\nSteps to fix:")
        print(f"  1. If container already exists, remove it:")
        print(f"     docker rm -f elasticsearch")
        print(f"  2. Start ElasticSearch:")
        print(f"     docker run -d --name elasticsearch \\")
        print(f"       -e \"discovery.type=single-node\" \\")
        print(f"       -e \"xpack.security.enabled=false\" \\")
        print(f"       -p 9200:9200 \\")
        print(f"       docker.elastic.co/elasticsearch/elasticsearch:9.2.4")
        print(f"  3. Wait 20-30 seconds for it to fully start")
        print(f"  4. Check it's running:")
        print(f"     docker ps | grep elasticsearch")
        print(f"  5. Then try again")
        return False


def index_passages_to_elasticsearch(passages: List[Dict], es_host="127.0.0.1", es_port=9200,
                                     index_name="medical_passages", batch_size=500):
    """
    Index passages to ElasticSearch with bulk API.
    
    Args:
        passages: List of passage dictionaries to index
        es_host: ElasticSearch host (default: localhost)
        es_port: ElasticSearch port (default: 9200)
        index_name: Index name (default: medical_passages)
        batch_size: Bulk indexing batch size (default: 1000)
    """
    from elasticsearch.helpers import bulk
    
    # Create client
    es = Elasticsearch(
        [f"http://{es_host}:{es_port}"],
        request_timeout=120,
        verify_certs=False,
        ssl_show_warn=False,
        retry_on_timeout=True,
        max_retries=3
    )
    
    print(f"\nIndexing {len(passages)} passages to ElasticSearch...")
    
    # Prepare bulk actions
    actions = []
    total_failures = 0
    failure_reasons = {}
    skipped = 0
    
    for i, passage in enumerate(passages):
        text = passage.get("text") or ""
        title = passage.get("title") or ""
        section_heading = passage.get("section_heading") or ""
        if not text and not title:
            skipped += 1
            continue

        published_date = passage.get("published_date")
        if published_date:
            try:
                if isinstance(published_date, str):
                    datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            except Exception:
                passage.pop("published_date", None)

        passage["text"] = str(text)
        passage["title"] = str(title)
        passage["section_heading"] = str(section_heading)

        action = {
            "_index": index_name,
            "_id": passage.get('id', i),
            "_source": passage
        }
        actions.append(action)
        
        if len(actions) >= batch_size:
            try:
                success, failed = bulk(es, actions, raise_on_error=False)
                print(f"  ✓ Indexed {len(actions)} passages (progress: {i+1}/{len(passages)})")
                if failed:
                    total_failures += len(failed)
                    print(f"    Warning: {len(failed)} failures in this batch")
                    # Log failure reasons
                    for fail_item in failed:
                        error_type = fail_item.get('error', {}).get('type', 'unknown')
                        reason = fail_item.get('error', {}).get('reason', 'unknown')
                        if error_type not in failure_reasons:
                            failure_reasons[error_type] = 0
                        failure_reasons[error_type] += 1
                actions = []
            except Exception as e:
                print(f"  ✗ Error indexing batch: {e}")
                actions = []
    
    # Index remaining
    if actions:
        try:
            success, failed = bulk(es, actions, raise_on_error=False)
            print(f"  ✓ Indexed {len(actions)} passages (final batch)")
            if failed:
                total_failures += len(failed)
                print(f"    Warning: {len(failed)} failures in final batch")
                for fail_item in failed:
                    error_type = fail_item.get('error', {}).get('type', 'unknown')
                    if error_type not in failure_reasons:
                        failure_reasons[error_type] = 0
                    failure_reasons[error_type] += 1
        except Exception as e:
            print(f"  ✗ Error indexing final batch: {e}")

    try:
        es.indices.refresh(index=index_name)
        print(f"✓ Refreshed index: {index_name}")
    except Exception as e:
        print(f"  Note: Could not refresh index: {e}")
    
    print(f"\n✓ Indexing complete: {len(passages) - skipped} passages indexed")
    if skipped:
        print(f"  Skipped {skipped} passages with empty text/title")
    if total_failures > 0:
        print(f"\nFailure Summary:")
        print(f"  Total failures: {total_failures}")
        for error_type, count in failure_reasons.items():
            print(f"    - {error_type}: {count}")
        print(f"\nNote: Failures are usually due to:")
        print(f"  1. Document too large (Elasticsearch has a 100MB limit per document)")
        print(f"  2. Invalid field types (numeric fields with non-numeric values)")
        print(f"  3. Field name conflicts or mapping issues")
        print(f"  4. Missing required fields")
        print(f"\nYour search will still work fine with {len(passages) - total_failures} indexed passages.")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage:
    
    1. Setup (one-time):
       - Start ElasticSearch
       - Create index
       - Index passages
    
    2. Search:
       - Initialize retriever
       - Search with queries
    """
    
    # Setup example
    print("="*80)
    print("SETUP EXAMPLE")
    print("="*80)
    print("""
    # 1. Start ElasticSearch
    docker run -d --name elasticsearch \\
      -e "discovery.type=single-node" \\
      -e "xpack.security.enabled=false" \\
      -p 9200:9200 \\
      docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    
    # 2. Create index and load passages
    from simple_retriever import create_elasticsearch_index, index_passages_to_elasticsearch
    import json
    
    create_elasticsearch_index()
    
    with open('storage/outputs/combined/metadata.json', 'r') as f:
        passages = json.load(f)
    
    index_passages_to_elasticsearch(passages)
    """)
    
    # Search example
    print("\n" + "="*80)
    print("SEARCH EXAMPLE")
    print("="*80)
    print("""
    from models import ModelManager
    from simple_retriever import SimpleRetriever
    from pathlib import Path
    
    # Initialize
    model_manager = ModelManager(Path("storage/outputs/combined"), {})
    model_manager.initialize_all()
    
    retriever = SimpleRetriever(model_manager, "storage/outputs/combined")
    
    # Search
    results = retriever.search("vaccine safety during pregnancy")
    
    # View results
    for i, result in enumerate(results, 1):
        p = result['passage']
        print(f"{i}. Score: {result['final_score']:.3f}")
        print(f"   Title: {p['title']}")
        print(f"   Domain: {result.get('domain_tier', 'N/A')}")
        print(f"   URL: {p['url']}")
    """)
