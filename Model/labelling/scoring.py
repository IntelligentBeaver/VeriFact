"""
Scoring Module
Handles relevance scoring calculations for passage ranking.
Includes hybrid FAISS + ElasticSearch RRF fusion and domain-aware scoring.
"""

import re
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


class PassageScorer:
    """Calculates relevance scores for passages using multiple signals."""
    
    # Stop words for lexical overlap calculation
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'than', 'to', 'of', 'in', 'on',
        'for', 'with', 'by', 'from', 'as', 'at', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'it', 'this', 'that', 'these', 'those', 'which', 'who', 'whom', 'what',
        'when', 'where', 'why', 'how', 'can', 'may', 'might', 'could', 'should', 'would',
        'will', 'do', 'does', 'did', 'not', 'no'
    }
    
    # Default weight configuration
    DEFAULT_WEIGHTS = {
        'faiss_with_sapbert': 0.35,
        'faiss_without_sapbert': 0.40,
        'cross_encoder': 0.50,
        'sapbert': 0.05,
        'lexical': 0.10,
        'credibility_boost': 1.05,
        'credibility_penalty': 0.95
    }
    
    def __init__(self, weights: Optional[dict] = None):
        """
        Initialize scorer with optional custom weights.
        
        Args:
            weights: Custom weight configuration (uses defaults if None)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
    
    def calculate_combined_score(
        self,
        faiss_score: float,
        cross_encoder_score: float,
        sapbert_score: Optional[float] = None,
        lexical_score: Optional[float] = None,
        is_medically_reviewed: bool = False,
        has_author: bool = False
    ) -> float:
        """
        Calculate weighted combined relevance score from multiple signals.
        
        Combines:
        - FAISS embedding similarity (35-40%)
        - Cross-encoder relevance (50%)
        - SapBERT medical entity similarity (5%, optional)
        - Lexical overlap (10%)
        - Credibility adjustments based on metadata
        
        Args:
            faiss_score: FAISS similarity [0, 1]
            cross_encoder_score: Cross-encoder score (normalized to [0, 1])
            sapbert_score: SapBERT medical similarity [0, 1] or None
            lexical_score: Lexical overlap score [0, 1] or None
            is_medically_reviewed: Whether passage is medically reviewed
            has_author: Whether passage has an identifiable author
            
        Returns:
            Combined relevance score [0, 1]
        """
        # Normalize cross-encoder score to [0, 1]
        ce_normalized = max(0.0, min(1.0, cross_encoder_score))
        
        # Calculate base weighted score
        if sapbert_score is not None:
            lex = lexical_score if lexical_score is not None else 0.0
            combined = (
                self.weights['faiss_with_sapbert'] * faiss_score +
                self.weights['cross_encoder'] * ce_normalized +
                self.weights['sapbert'] * sapbert_score +
                self.weights['lexical'] * lex
            )
        else:
            lex = lexical_score if lexical_score is not None else 0.0
            combined = (
                self.weights['faiss_without_sapbert'] * faiss_score +
                self.weights['cross_encoder'] * ce_normalized +
                self.weights['lexical'] * lex
            )
        
        # Apply credibility adjustments
        if is_medically_reviewed:
            combined *= self.weights['credibility_boost']
        if not has_author:
            combined *= self.weights['credibility_penalty']
        
        # Clamp to valid range
        return max(0.0, min(1.0, combined))
    
    def compute_lexical_overlap(self, query: str, passage_text: str) -> float:
        """
        Compute lexical overlap between query and passage.
        
        Calculates Jaccard-style similarity based on significant word overlap,
        filtering out stop words and requiring minimum word length.
        
        Args:
            query: Query string
            passage_text: Passage text
            
        Returns:
            Overlap score [0, 1]
        """
        if not query or not passage_text:
            return 0.0
        
        # Extract significant tokens
        query_tokens = self._extract_tokens(query)
        if not query_tokens:
            return 0.0
        
        passage_tokens = self._extract_tokens(passage_text)
        
        # Calculate overlap
        overlap = query_tokens.intersection(passage_tokens)
        return min(1.0, len(overlap) / len(query_tokens))
    
    def _extract_tokens(self, text: str) -> set:
        """
        Extract significant tokens from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of lowercase tokens (length >= 3, excluding stop words)
        """
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return {w for w in words if len(w) >= 3 and w not in self.STOP_WORDS}
    
    def compute_sapbert_similarity(
        self,
        query_embedding: np.ndarray,
        passage_embedding: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between SapBERT embeddings.
        
        Args:
            query_embedding: Query embedding vector (normalized)
            passage_embedding: Passage embedding vector (normalized)
            
        Returns:
            Cosine similarity [0, 1]
        """
        # Dot product of normalized vectors = cosine similarity
        similarity = np.dot(query_embedding, passage_embedding)
        # Clamp to valid range
        return max(0.0, min(1.0, float(similarity)))
    
    def normalize_scores(self, scores: List[float]) -> np.ndarray:
        """
        Apply sigmoid normalization to raw scores.
        
        Args:
            scores: List of raw scores
            
        Returns:
            Normalized scores [0, 1]
        """
        try:
            arr = np.array(scores, dtype=np.float32)
            normalized = 1.0 / (1.0 + np.exp(-arr))
            return normalized
        except Exception:
            return np.zeros(len(scores), dtype=np.float32)


class PassageFilter:
    """Filters passages based on content quality."""
    
    @staticmethod
    def is_substantive(
        passage_text: str,
        min_length: int = 100,
        min_words: int = 15
    ) -> bool:
        """
        Check if passage has substantive content.
        
        Filters out:
        - Too short passages (< min_length chars or < min_words words)
        - All uppercase text (likely headings)
        - Single-word passages
        
        Args:
            passage_text: Passage text to check
            min_length: Minimum character count
            min_words: Minimum word count
            
        Returns:
            True if passage is substantive, False otherwise
        """
        text = (passage_text or "").strip()
        
        # Check length requirements
        if len(text) < min_length or len(text.split()) < min_words:
            return False
        
        # Check for heading-like patterns
        if text.isupper() or text.count(' ') == 0:
            return False
        
        return True
    
    @staticmethod
    def is_question(passage_text: str) -> bool:
        """
        Check if passage is primarily a question.
        
        Detects:
        1. Single sentence ending with ?
        2. Multiple sentences where >50% are questions
        3. Short text (<100 chars) ending with ?
        
        Args:
            passage_text: Passage text to check
            
        Returns:
            True if passage is question-like
        """
        text = passage_text.strip()
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        question_count = text.count('?')
        total_sentences = len(sentences)
        
        # Single sentence question
        if total_sentences == 1 and text.endswith('?'):
            return True
        
        # Majority questions (FAQ-style)
        if total_sentences > 1 and question_count >= total_sentences * 0.5:
            return True
        
        # Short question
        if len(text) < 100 and text.endswith('?'):
            return True
        
        return False


class AutoLabeler:
    """Determines auto-labeling decisions based on score thresholds."""
    
    def __init__(
        self,
        relevant_threshold: float = 0.66,
        unrelated_threshold: float = 0.52,
        ce_max_unrelated: float = 0.10,
        lex_max_unrelated: float = 0.40
    ):
        """
        Initialize auto-labeler with thresholds.
        
        Args:
            relevant_threshold: Score >= this → auto-label as relevant
            unrelated_threshold: Score <= this → auto-label as unrelated
            ce_max_unrelated: Cross-encoder score threshold for unrelated
            lex_max_unrelated: Lexical score threshold for unrelated
        """
        self.relevant_threshold = relevant_threshold
        self.unrelated_threshold = unrelated_threshold
        self.ce_max_unrelated = ce_max_unrelated
        self.lex_max_unrelated = lex_max_unrelated
    
    def determine_label(
        self,
        combined_score: float,
        rerank_score: float,
        lexical_score: float,
        is_question: bool
    ) -> Tuple[str, Optional[str]]:
        """
        Determine auto-label for passage.
        
        Args:
            combined_score: Combined relevance score [0, 1]
            rerank_score: Cross-encoder reranking score
            lexical_score: Lexical overlap score [0, 1]
            is_question: Whether passage is a question
            
        Returns:
            Tuple of (label, reason) where label is one of:
            - 'relevant': High confidence relevant
            - 'unrelated': High confidence unrelated
            - 'question': Detected as question
            - 'review': Needs human review
        """
        # Check for question first
        if is_question:
            return 'question', 'question'
        
        # Check low CE + low lexical → likely unrelated
        if rerank_score <= self.ce_max_unrelated and lexical_score <= self.lex_max_unrelated:
            return 'unrelated', 'low_ce_lex'
        
        # Check combined score thresholds
        if combined_score >= self.relevant_threshold:
            return 'relevant', 'score_high'
        
        if combined_score <= self.unrelated_threshold:
            return 'unrelated', 'score_low'
        
        # Middle zone → needs human review
        return 'review', None


class RRFFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple rankers.
    Fuses FAISS dense retrieval with ElasticSearch BM25 lexical search.
    """
    
    @staticmethod
    def compute_rrf_score(
        faiss_rank: Optional[int],
        es_rank: Optional[int],
        k: int = 60,
        faiss_weight: float = 0.5,
        es_weight: float = 0.5
    ) -> float:
        """
        Compute RRF score for a passage ranked by multiple systems.
        
        RRF formula: score = Σ(weight_i / (k + rank_i))
        
        Args:
            faiss_rank: FAISS ranking position (1-indexed) or None if not ranked
            es_rank: ElasticSearch ranking position (1-indexed) or None if not ranked
            k: RRF constant (default 60) - higher k smooths differences
            faiss_weight: Weight for FAISS component
            es_weight: Weight for ElasticSearch component
            
        Returns:
            RRF score [0, ∞) - higher is better
        """
        score = 0.0
        
        if faiss_rank is not None and faiss_rank > 0:
            score += faiss_weight / (k + faiss_rank)
        
        if es_rank is not None and es_rank > 0:
            score += es_weight / (k + es_rank)
        
        return score
    
    @staticmethod
    def fuse_rankings(
        faiss_results: List[Dict[str, Any]],
        es_results: List[Dict[str, Any]],
        k: int = 60,
        faiss_weight: float = 0.5,
        es_weight: float = 0.5,
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fuse FAISS and ElasticSearch rankings using RRF.
        
        Args:
            faiss_results: List of FAISS results with 'passage_id' and optionally 'rank'
            es_results: List of ES results with 'passage_id' and optionally 'rank'
            k: RRF constant
            faiss_weight: Weight for FAISS
            es_weight: Weight for ElasticSearch
            normalize: If True, normalize RRF scores to [0, 1]
            
        Returns:
            Fused and sorted results list
        """
        # Build passage score map
        passage_scores = {}
        
        # Add FAISS results
        for rank, result in enumerate(faiss_results, 1):
            passage_id = result.get('passage_id')
            if passage_id:
                if passage_id not in passage_scores:
                    passage_scores[passage_id] = {
                        'passage_id': passage_id,
                        'faiss_rank': None,
                        'es_rank': None,
                        'source': result
                    }
                passage_scores[passage_id]['faiss_rank'] = rank
                passage_scores[passage_id]['source'] = result
        
        # Add ElasticSearch results
        for rank, result in enumerate(es_results, 1):
            passage_id = result.get('passage_id')
            if passage_id:
                if passage_id not in passage_scores:
                    passage_scores[passage_id] = {
                        'passage_id': passage_id,
                        'faiss_rank': None,
                        'es_rank': None,
                        'source': result
                    }
                passage_scores[passage_id]['es_rank'] = rank
                if 'source' not in passage_scores[passage_id]:
                    passage_scores[passage_id]['source'] = result
        
        # Compute RRF scores
        fused_results = []
        for passage_id, data in passage_scores.items():
            rrf_score = RRFFusion.compute_rrf_score(
                data['faiss_rank'],
                data['es_rank'],
                k=k,
                faiss_weight=faiss_weight,
                es_weight=es_weight
            )
            
            fused_results.append({
                'passage_id': passage_id,
                'rrf_score': rrf_score,
                'faiss_rank': data['faiss_rank'],
                'es_rank': data['es_rank'],
                'source': data['source']
            })
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        # Normalize RRF scores if requested
        if normalize and fused_results:
            max_score = max(r['rrf_score'] for r in fused_results)
            if max_score > 0:
                for result in fused_results:
                    result['rrf_score_normalized'] = result['rrf_score'] / max_score
        
        return fused_results
    
    @staticmethod
    def explain_fusion(
        passage_id: str,
        faiss_rank: Optional[int],
        es_rank: Optional[int],
        k: int = 60,
        faiss_weight: float = 0.5,
        es_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Explain RRF score calculation for a passage.
        
        Args:
            passage_id: Passage ID
            faiss_rank: FAISS rank
            es_rank: ElasticSearch rank
            k: RRF constant
            faiss_weight: FAISS weight
            es_weight: ES weight
            
        Returns:
            Dictionary explaining score components
        """
        faiss_component = 0.0
        if faiss_rank is not None and faiss_rank > 0:
            faiss_component = faiss_weight / (k + faiss_rank)
        
        es_component = 0.0
        if es_rank is not None and es_rank > 0:
            es_component = es_weight / (k + es_rank)
        
        total_score = faiss_component + es_component
        
        return {
            'passage_id': passage_id,
            'faiss_rank': faiss_rank,
            'es_rank': es_rank,
            'faiss_component': faiss_component,
            'es_component': es_component,
            'total_rrf_score': total_score,
            'explanation': f"RRF = {faiss_weight}/(60+{faiss_rank or 'N/A'}) + {es_weight}/(60+{es_rank or 'N/A'})"
        }

