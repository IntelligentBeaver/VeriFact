"""
Stance Detection Module
Handles stance detection using heuristics and NLI models.
"""

import re
from typing import Optional, Tuple
import numpy as np


class StanceDetector:
    """Detects stance (supports/refutes/neutral) using medical heuristics and NLI."""
    
    # Multi-word support phrases (2x weight)
    SUPPORT_PHRASES = [
        'lead to', 'leads to', 'leading to',
        'linked to', 'links to',
        'associated with',
        'cause', 'causes', 'caused by',
        'result in', 'results in',
        'increase risk', 'increases risk',
        'increased risk', 'higher risk', 'more likely',
        'risk of', 'raises risk', 'raise risk',
        'may increase', 'may increase risk',
        'responsible for',
        'result of', 'consequence of'
    ]
    
    # Single-word support keywords (1x weight)
    SUPPORT_KEYWORDS = [
        'helps', 'helpful', 'beneficial', 'benefit', 'benefits',
        'improves', 'improvement', 'improve',
        'prevents', 'prevention', 'preventive', 'protective', 'protection',
        'treats', 'treatment', 'therapeutic',
        'supports', 'support', 'supporting',
        'effective', 'efficacy', 'efficacious',
        'proven', 'evidence', 'robust',
        'good', 'positive', 'positively',
        'promotes', 'promotion',
        'enhances', 'enhancement', 'enhance',
        'strengthens', 'strengthen', 'strength',
        'aids', 'aid', 'assists', 'assist',
        'trigger', 'triggers', 'triggered'
    ]
    
    # Multi-word refute phrases (2x weight)
    REFUTE_PHRASES = [
        'no evidence', 'lack of evidence', 'insufficient evidence',
        'does not', 'do not', 'did not',
        'does not cause', 'does not lead',
        'not linked', 'not associated',
        'no association', 'no link',
        'not responsible'
    ]
    
    # Single-word refute keywords (1x weight)
    REFUTE_KEYWORDS = [
        'not', 'no', 'none', 'neither',
        'harmful', 'harm', 'danger', 'dangerous',
        'adverse', 'adversely',
        'ineffective', 'ineffectiveness', 'inefficacious',
        'contraindicated', 'contraindication',
        'avoid', 'avoidance', 'avoiding',
        'problem', 'problematic', 'problems',
        'issue', 'issues',
        'negative', 'negatively',
        'failed', 'failure', 'fail',
        'worsen', 'worsens', 'worse', 'worsening',
        'caution', 'cautious'
    ]
    
    def __init__(self, nli_cross_encoder=None):
        """
        Initialize stance detector.
        
        Args:
            nli_cross_encoder: Optional NLI cross-encoder model
        """
        self.nli_cross_encoder = nli_cross_encoder
    
    def detect_stance(
        self,
        query: str,
        passage_text: str
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Detect stance using medical heuristics.
        
        Args:
            query: Query/claim text
            passage_text: Passage text
            
        Returns:
            Tuple of (stance_label, confidence) or (None, None) if inconclusive
            stance_label: 'supports', 'refutes', 'neutral', or None
        """
        if not passage_text:
            return None, None
        
        # Use medical-aware heuristic detection
        return self._detect_stance_heuristic(passage_text)
    
    def detect_stance_nli(
        self,
        query: str,
        passage_text: str
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Detect stance using NLI cross-encoder.
        
        Args:
            query: Query/claim text
            passage_text: Passage text
            
        Returns:
            Tuple of (stance_label, confidence) or (None, None) if NLI unavailable
        """
        if self.nli_cross_encoder is None:
            return None, None
        
        try:
            pair = [[query, passage_text]]
            scores = self.nli_cross_encoder.predict(pair)[0]
            
            # Handle scalar scores
            if isinstance(scores, np.ndarray) and len(scores.shape) == 0:
                return None, None
            
            # Extract entailment, neutral, contradiction (standard NLI order)
            if isinstance(scores, np.ndarray) and scores.shape[0] >= 3:
                entail_score = float(scores[2])  # ENTAILMENT
                contra_score = float(scores[0])  # CONTRADICTION
                
                if entail_score > contra_score:
                    confidence = float(np.tanh(entail_score / 2.0))
                    return 'supports', confidence
                elif contra_score > entail_score:
                    confidence = float(np.tanh(contra_score / 2.0))
                    return 'refutes', confidence
                else:
                    return 'neutral', 0.5
            else:
                return None, None
        except Exception:
            return None, None
    
    def _detect_stance_heuristic(
        self,
        passage_text: str
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Detect stance using keyword-based heuristics.
        
        Optimized for medical fact-checking by recognizing:
        - Causal language: "lead to", "cause", "associated with"
        - Risk/benefit language: "increase risk", "improve", "worsen"
        - Negations: "not", "no evidence", "contraindicated"
        
        Args:
            passage_text: Passage text to analyze
            
        Returns:
            Tuple of (stance_label, confidence) or (None, None) if inconclusive
        """
        text_lower = passage_text.lower()
        
        support_count = 0
        refute_count = 0
        strong_support_hit = False
        
        # Count multi-word support phrases (2x weight)
        for phrase in self.SUPPORT_PHRASES:
            occurrences = text_lower.count(phrase)
            if occurrences > 0:
                support_count += occurrences * 2
                strong_support_hit = True
        
        # Count multi-word refute phrases (2x weight)
        for phrase in self.REFUTE_PHRASES:
            occurrences = text_lower.count(phrase)
            if occurrences > 0:
                refute_count += occurrences * 2
        
        # Count single-word support keywords (1x weight)
        for keyword in self.SUPPORT_KEYWORDS:
            occurrences = self._count_whole_word(text_lower, keyword)
            support_count += occurrences
        
        # Count single-word refute keywords (1x weight)
        for keyword in self.REFUTE_KEYWORDS:
            occurrences = self._count_whole_word(text_lower, keyword)
            refute_count += occurrences
        
        # Determine stance
        total_signals = support_count + refute_count
        
        if total_signals == 0:
            return None, None
        
        if support_count > refute_count:
            confidence = min(1.0, support_count / (total_signals + 1))
            if strong_support_hit and refute_count == 0:
                confidence = max(confidence, 0.85)
            return 'supports', confidence
        
        elif refute_count > support_count:
            confidence = min(1.0, refute_count / (total_signals + 1))
            return 'refutes', confidence
        
        else:
            # Equal signals = inconclusive
            return None, None
    
    @staticmethod
    def _count_whole_word(text: str, word: str) -> int:
        """
        Count occurrences of whole word in text.
        
        Args:
            text: Text to search in (lowercase)
            word: Word to search for
            
        Returns:
            Count of whole-word matches
        """
        pattern = r'\b' + re.escape(word) + r'\b'
        return len(re.findall(pattern, text))


class StanceAutoLabeler:
    """Handles auto-labeling of stance with confidence thresholds."""
    
    def __init__(self, auto_threshold: float = 0.66):
        """
        Initialize auto-labeler.
        
        Args:
            auto_threshold: Confidence threshold for auto-labeling
        """
        self.auto_threshold = auto_threshold
    
    def should_auto_label(
        self,
        stance: Optional[str],
        confidence: Optional[float]
    ) -> bool:
        """
        Determine if stance should be auto-labeled.
        
        Args:
            stance: Predicted stance
            confidence: Prediction confidence
            
        Returns:
            True if confidence meets threshold and stance is not neutral
        """
        if stance is None or stance == 'neutral':
            return False
        
        if confidence is None:
            return False
        
        return confidence >= self.auto_threshold
    
    def get_stance_with_fallback(
        self,
        stance: Optional[str],
        confidence: Optional[float]
    ) -> Tuple[str, float]:
        """
        Get stance with fallback to neutral if no prediction.
        
        Args:
            stance: Predicted stance
            confidence: Prediction confidence
            
        Returns:
            Tuple of (stance, confidence) with neutral fallback
        """
        if stance is None:
            return 'neutral', 0.50
        
        if confidence is None:
            return stance, 0.50
        
        return stance, confidence
