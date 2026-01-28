"""
Interactive Passage Labeling System
Labels passages from the medical index as RELEVANT or UNRELATED to a query.
Stores results with full metadata and query information.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline

from config import (
    INDEX_DIR, EMBEDDING_MODEL, CROSS_ENCODER_MODEL, NLI_CROSS_ENCODER_MODEL, SAPBERT_MODEL, STANCE_MODEL,
    DEFAULT_TOPK_FAISS, DEFAULT_DISPLAY_RESULTS,
    AUTO_RELEVANT_THRESHOLD, AUTO_UNRELATED_THRESHOLD, STANCE_AUTO_THRESHOLD,
    AUTO_LABELING_MODE,
    AUTO_UNRELATED_CE_MAX, AUTO_UNRELATED_LEX_MAX,
    RELEVANT_PASSAGES_FILE, UNRELATED_PASSAGES_FILE, QUESTION_PASSAGES_FILE,
    LABELING_SESSION_FILE, CONCEPT_LABELS_VECTORS, CONCEPT_LABELS_METADATA, CONCEPT_TOPK,
    OUTPUT_DIR,
    VERIFIED_CLAIMS_FILE, FAKE_CLAIMS_FILE
)


class PassageLabeler:
    """Interactive labeling system for medical passages."""
    
    def __init__(self):
        # Index + model locations and cached artifacts
        self.index_dir = Path(INDEX_DIR)
        self.metadata = None
        self.index = None
        self.embed_model = None
        self.cross_encoder = None
        self.nli_cross_encoder = None
        self.sapbert_model = None
        self.sapbert_embeddings = None
        self.stance_classifier = None
        self.concept_label_vectors = None
        self.concept_label_meta = None
        self.last_query_concepts = []
        self.auto_labeling_mode = AUTO_LABELING_MODE
        
        # Persisted labels from prior sessions
        self.relevant_passages = self._load_existing("relevant")
        self.unrelated_passages = self._load_existing("unrelated")
        self.question_passages = self._load_existing("question")
        self.session_start = datetime.now()
        self.models_initialized = False
        
    def _load_existing(self, label_type):
        """Load existing labeled passages."""
        if label_type == "relevant":
            file_path = RELEVANT_PASSAGES_FILE
        elif label_type == "unrelated":
            file_path = UNRELATED_PASSAGES_FILE
        elif label_type == "question":
            file_path = QUESTION_PASSAGES_FILE
        else:
            return []
        
        if file_path.exists():
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _open_output_file(self, path_obj):
        """Open output file safely on Windows paths."""
        try:
            return path_obj.open('w', encoding='utf-8')
        except OSError as e:
            if os.name == "nt":
                long_path = r"\\?\\" + str(path_obj)
                return open(long_path, 'w', encoding='utf-8')
            raise e
    
    def initialize_models(self, skip_stance=False):
        """Load all required models and indices.
        
        Args:
            skip_stance: If True, skip loading stance detection models (NLI cross-encoder and MNLI classifier).
                        Useful for workflows that don't need stance prediction (e.g., fake claims with hardcoded refutes).
        """
        print("\n" + "="*60)
        print("Initializing Models and Indices")
        print("="*60)
        
        # Load metadata
        metadata_path = self.index_dir / 'metadata.json'
        if not metadata_path.exists():
            print(f"Error: metadata.json not found at {metadata_path}")
            return False
        
        print("\nStep 1: Loading metadata...")
        with metadata_path.open('r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"  ✓ Loaded {len(self.metadata)} passages")
        
        # Load FAISS index
        index_path = self.index_dir / 'index.faiss'
        if not index_path.exists():
            print(f"Error: index.faiss not found at {index_path}")
            return False
        
        print("\nStep 2: Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))
        print(f"  ✓ FAISS index loaded")
        
        # Load embedding model used for query + passage embeddings
        print("\nStep 3: Loading embedding model...")
        print(f"  Model: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  ✓ Embedding model loaded")
        
        # Load cross-encoder reranker (optional but recommended)
        print("\nStep 4: Loading cross-encoder for relevance ranking...")
        print(f"  Model: {CROSS_ENCODER_MODEL}")
        try:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            print(f"  ✓ Cross-encoder loaded")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load cross-encoder: {e}")
        
        # Load NLI cross-encoder for stance detection only
        if skip_stance:
            print("\nStep 4b: Skipping NLI cross-encoder (not needed for this workflow)")
            self.nli_cross_encoder = None
        else:
            print("\nStep 4b: Loading NLI cross-encoder for stance detection...")
            print(f"  Model: {NLI_CROSS_ENCODER_MODEL}")
            try:
                self.nli_cross_encoder = CrossEncoder(NLI_CROSS_ENCODER_MODEL)
                print(f"  ✓ NLI cross-encoder loaded")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to load NLI cross-encoder: {e}")
        
        # Load SapBERT embeddings for medical-entity similarity (optional)
        sapbert_path = self.index_dir / 'sapbert_embeddings.npy'
        if sapbert_path.exists():
            print("\nStep 5: Loading SapBERT embeddings...")
            self.sapbert_embeddings = np.load(str(sapbert_path))
            print(f"  ✓ SapBERT embeddings loaded: {self.sapbert_embeddings.shape}")
            
            try:
                self.sapbert_model = SentenceTransformer(SAPBERT_MODEL)
                print(f"  ✓ SapBERT model loaded")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to load SapBERT model: {e}")
        else:
            print("\nStep 5: SapBERT embeddings not found (optional)")
        
        # Load stance classifier (MNLI)
        if skip_stance:
            print("\nStep 6: Skipping stance classifier (not needed for this workflow)")
            self.stance_classifier = None
        else:
            print("\nStep 6: Loading stance classifier (MNLI)...")
            print(f"  Model: {STANCE_MODEL}")
            try:
                self.stance_classifier = pipeline(
                    task="text-classification",
                    model=STANCE_MODEL,
                    tokenizer=STANCE_MODEL,
                    return_all_scores=True
                )
                print(f"  ✓ Stance classifier loaded")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to load stance classifier: {e}")
                self.stance_classifier = None
            
            if self.stance_classifier is None:
                print("  ⚠ No stance classifier available; stance will require manual input.")

        # Load SapBERT concept label embeddings (optional, for query expansion)
        labels_vec_path = CONCEPT_LABELS_VECTORS
        labels_meta_path = CONCEPT_LABELS_METADATA
        if self.sapbert_model is not None and labels_vec_path.exists() and labels_meta_path.exists():
            try:
                self.concept_label_vectors = np.load(str(labels_vec_path))
                with labels_meta_path.open('r', encoding='utf-8') as f:
                    self.concept_label_meta = json.load(f)
                count = len(self.concept_label_meta) if self.concept_label_meta else 0
                print(f"  ✓ Concept label embeddings loaded: {self.concept_label_vectors.shape} (labels={count})")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to load concept label embeddings: {e}")
                self.concept_label_vectors = None
                self.concept_label_meta = None
        else:
            print("  ℹ Concept label embeddings not found (optional)")
        
        print("\n" + "="*60)
        print("✓ All models loaded successfully!")
        print("="*60)
        self.models_initialized = True
        return True

    def ensure_models_initialized(self, skip_stance=False):
        """Initialize models on demand.
        
        Args:
            skip_stance: If True, skip loading stance detection models.
        """
        if not self.models_initialized:
            return self.initialize_models(skip_stance=skip_stance)
        return True
    
    def embed_query(self, query):
        """Embed a query using the same model as the index.
        
        Converts user query text into a dense vector representation using
        the S-PubMedBert model, then L2-normalizes for FAISS similarity search.
        """
        v = self.embed_model.encode([query], convert_to_numpy=True)[0]
        # L2 normalization: divide by vector magnitude for consistent similarity scoring
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v
    
    def search(self, query, topk=DEFAULT_TOPK_FAISS):
        """Search for passages related to query using FAISS similarity.
        
        Uses inner product (dot product) search on normalized embeddings.
        Returns top-k most similar passages based on embedding space proximity.
        """
        qvec = self.build_query_vector(query)
        # Reshape for FAISS: expects (batch_size, embedding_dim)
        q = np.expand_dims(qvec.astype(np.float32), axis=0)
        # FAISS returns scores (similarity values) and indices of top-k results
        scores, indices = self.index.search(q, topk)
        return scores[0], indices[0]
    
    def rerank_results(self, query, passages):
        """Rerank passages using a general relevance cross-encoder (MS-MARCO).
        
        Cross-encoder provides higher-quality relevance scoring by computing
        query-passage interaction directly. Uses MS-MARCO MiniLM which is trained
        for passage ranking and returns scores in [-1, 1] range.
        
        Returns array of relevance scores, one per passage, in [-1, 1] range.
        """
        if self.cross_encoder is None:
            # Fallback: if cross-encoder not available, return neutral scores
            return [0.0] * len(passages)
        
        texts = [p.get('text', '') for p in passages]
        # Cross-encoder expects [query, passage] pairs
        pairs = [[query, t] for t in texts]
        scores = self.cross_encoder.predict(pairs)

        # MS-MARCO cross-encoder outputs logits; normalize to [0, 1] with sigmoid
        try:
            arr = np.array(scores, dtype=np.float32)
            normalized = 1.0 / (1.0 + np.exp(-arr))
            return normalized.tolist()
        except Exception:
            return [0.0] * len(passages)
    
    def compute_sapbert_similarities(self, query, passage_indices):
        """Compute SapBERT medical entity similarity scores for passages.
        
        SapBERT is specialized for medical concepts and entities. This method:
        1. Embeds the query using SapBERT
        2. Computes cosine similarity between query and each passage embedding
        3. Returns normalized [0, 1] scores reflecting medical entity relevance
        
        Returns None for missing embeddings (prevents artificial score inflation).
        """
        if self.sapbert_model is None or self.sapbert_embeddings is None:
            # Return None marker - will be handled gracefully in calculate_combined_score
            # This prevents missing embeddings from being treated as neutral (0.5)
            return [None] * len(passage_indices)
        
        # Embed query with SapBERT medical model
        query_emb = self.sapbert_model.encode([query], convert_to_numpy=True)[0]
        # L2 normalize: ensures cosine similarity is computed correctly
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        # Compute similarity with passage embeddings
        scores = []
        for idx in passage_indices:
            if idx >= 0 and idx < len(self.sapbert_embeddings):
                passage_emb = self.sapbert_embeddings[idx]
                # Normalize passage embedding for fair comparison
                passage_emb = passage_emb / (np.linalg.norm(passage_emb) + 1e-8)
                # Dot product of normalized vectors = cosine similarity
                similarity = np.dot(query_emb, passage_emb)
                # Clamp to [0, 1]: cosine similarity can theoretically go outside this
                scores.append(max(0.0, similarity))
            else:
                # Mark missing embedding as None, not neutral 0.5
                scores.append(None)
        
        return scores

    def get_top_concepts(self, query, topk=CONCEPT_TOPK):
        """Retrieve top SapBERT concept labels related to the query."""
        if self.sapbert_model is None or self.concept_label_vectors is None or self.concept_label_meta is None:
            return []

        try:
            q_emb = self.sapbert_model.encode([query], convert_to_numpy=True)[0]
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            sims = self.concept_label_vectors.dot(q_emb)
            # Pull more candidates to ensure we can dedupe by label
            top_idxs = np.argsort(-sims)[: max(topk * 5, topk)]
            labels = []
            seen = set()
            for i in top_idxs:
                if i < len(self.concept_label_meta):
                    meta = self.concept_label_meta[i]
                    label = (
                        meta.get("canonical_label")
                        or meta.get("canonical")
                        or meta.get("label")
                        or meta.get("preferred_label")
                        or meta.get("name")
                        or meta.get("original_text")
                        or meta.get("mesh_id")
                    )
                    if label and label not in seen:
                        labels.append(label)
                        seen.add(label)
                    if len(labels) >= topk:
                        break
            return labels
        except Exception:
            return []

    def build_query_vector(self, query):
        """Build query vector with optional concept expansion."""
        self.last_query_concepts = []
        expanded_query = query

        # Use SapBERT concept labels to expand the query text (if available)
        concepts = self.get_top_concepts(query, topk=CONCEPT_TOPK)
        if concepts:
            self.last_query_concepts = concepts
            expanded_query = f"{query}; " + "; ".join(concepts)

        return self.embed_query(expanded_query)
    
    def is_question(self, text):
        """Check if passage text is itself a question."""
        text = text.strip()
        return text.endswith('?')
    
    def is_substantive_passage(self, passage_text, min_length=100, min_words=15):
        """Filter out title-like or heading-only passages.
        
        Checks if passage has substantive content (not just a heading/label).
        Skips passages that are:
        - Too short (< min_length chars or < min_words words)
        - All uppercase (likely headings)
        - Single word with no spaces
        
        Returns True if passage is substantive, False if it should be skipped.
        """
        text = (passage_text or "").strip()
        
        # Check minimum length and word count
        if len(text) < min_length or len(text.split()) < min_words:
            return False
        
        # Check if it's just repeated terms or all caps (likely heading)
        if text.isupper() or text.count(' ') == 0:
            return False
        
        return True
    
    def is_single_sentence_question(self, text):
        """Check if passage is a question or primarily questions.
        
        Detects three types of passages that should be auto-labeled as questions:
        1. Single sentence ending with ? (simple question)
        2. Multiple sentences where >50% are questions (mostly questions)
        3. Short text (<100 chars) ending with ? (brief question)
        
        This catches both direct questions and meta-passages used in FAQs.
        """
        import re
        text = text.strip()
        # Split by sentence delimiters to count sentences
        # Regex handles multiple punctuation marks (e.g., "What?!")
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Count how many sentences end with question mark
        question_count = text.count('?')
        total_sentences = len(sentences)
        
        # Mark as question if any of these conditions are true:
        # 1. Single sentence ending with ? (basic question detection)
        if total_sentences == 1 and text.endswith('?'):
            return True
        # 2. Multiple sentences but majority are questions (FAQ-style content)
        if total_sentences > 1 and question_count >= total_sentences * 0.5:
            return True
        # 3. Very short passage ending with ? (common in educational content)
        if len(text) < 100 and text.endswith('?'):
            return True
        
        return False
    
    def calculate_combined_score(self, faiss_score, cross_encoder_score, sapbert_score=None, 
                                   lexical_score=None, is_medically_reviewed=False, has_author=False):
        """Calculate weighted combined score from multiple relevance signals.
        
        This is the core scoring algorithm that combines three independent relevance signals:
        - FAISS (25%): Embedding similarity in semantic space
        - Cross-Encoder (70%): High-quality query-passage interaction scoring
        - SapBERT (5%): Medical entity/concept alignment
        
        Then applies credibility adjustments based on source metadata.
        
        Args:
            faiss_score: FAISS similarity [0, 1]
            cross_encoder_score: Cross-encoder score, typically [-1, +1]
            sapbert_score: SapBERT medical entity similarity [0, 1] or None
            is_medically_reviewed: Whether passage is medically reviewed
            has_author: Whether passage has an identifiable author
            
        Returns:
            Combined relevance score [0, 1]
        """
        # Cross-encoder scores are already normalized to [0, 1] in rerank_results
        ce_normalized = max(0.0, min(1.0, cross_encoder_score))

        # Combine the signals using a more balanced weighted average.
        # Rationale: in some queries the cross-encoder can be overly decisive
        # (pushing many high-FAISS results into 'unrelated'). Increase FAISS
        # contribution and reduce CE dominance for more robust behavior.
        # When SapBERT is present: 35% FAISS + 50% CE + 5% SapBERT + 10% Lexical
        # When SapBERT is missing: 40% FAISS + 50% CE + 10% Lexical
        if sapbert_score is not None:
            lex = 0.0 if lexical_score is None else lexical_score
            combined = 0.35 * faiss_score + 0.50 * ce_normalized + 0.05 * sapbert_score + 0.10 * lex
        else:
            lex = 0.0 if lexical_score is None else lexical_score
            combined = 0.40 * faiss_score + 0.50 * ce_normalized + 0.10 * lex

        # Apply milder credibility adjustments based on source metadata
        # Slight boost for medically reviewed content, slight penalty for anonymous
        if is_medically_reviewed:
            combined *= 1.05
        if not has_author:
            combined *= 0.95
        
        # Final clamp to [0, 1] range after all transformations
        return max(0.0, min(1.0, combined))
    
    def display_passage(self, idx, passage, query, faiss_score, cross_score, sapbert_score, combined_score, auto_label, lexical_score=None, auto_label_reason=None):
        """Display a compact view of the passage with auto-label status and scores.
        
        Shows:
        - Passage title with auto-label indicator
        - Metadata: section, author presence, medical review status
        - All four relevance scores for transparency
        - Text snippet for human review
        
        Auto-label indicators:
        - [AUTO: RELEVANT ✓]: Combined score >= 0.75 (high confidence)
        - [AUTO: UNRELATED ✗]: Combined score <= 0.35 (low confidence)
        - [AUTO: QUESTION ❓]: Single/multi-sentence question detected
        - [NEEDS REVIEW]: Score in middle range (0.35-0.75, requires human judgment)
        """
        has_author = bool(passage.get('author'))
        has_review = bool(passage.get('medically_reviewed_by'))
        title = passage.get('title', 'Untitled')
        section = passage.get('section_heading', 'N/A')
        text = (passage.get('text', '') or '').strip()
        # Show first 800 chars of passage text for context
        if text:
            text_snippet = text if len(text) <= 800 else f"{text[:800]}..."
        else:
            # Fallback to show something when passage text is missing
            text_snippet = "[NO TEXT AVAILABLE]"

        # Show auto-label status
        auto_label_display = ""
        reason_display = f" | {auto_label_reason}" if auto_label_reason else ""
        if auto_label == 'relevant':
            auto_label_display = f" [AUTO: RELEVANT ✓{reason_display}]"
        elif auto_label == 'unrelated':
            auto_label_display = f" [AUTO: UNRELATED ✗{reason_display}]"
        elif auto_label == 'question':
            auto_label_display = f" [AUTO: QUESTION ❓{reason_display}]"
        elif auto_label == 'review':
            auto_label_display = " [NEEDS REVIEW]"
        
        # Display formatted passage information
        print(f"\n{'='*70}")
        print(f"[{idx}] {title}{auto_label_display}")
        print(f"Section: {section} | Author: {'yes' if has_author else 'no'} | Reviewed: {'yes' if has_review else 'no'}")
        # Show all four scores for complete transparency
        if lexical_score is None:
            print(f"Scores → FAISS: {faiss_score:.4f} | Cross: {cross_score:.4f} | SapBERT: {sapbert_score:.4f} | Combined: {combined_score:.4f}")
        else:
            print(f"Scores → FAISS: {faiss_score:.4f} | Cross: {cross_score:.4f} | SapBERT: {sapbert_score:.4f} | Lexical: {lexical_score:.4f} | Combined: {combined_score:.4f}")
        print("Text:\n" + text_snippet)
        print(f"{'='*70}")

    def compute_lexical_overlap(self, query, passage_text):
        """Compute simple lexical overlap between query and passage (0..1)."""
        import re
        if not query or not passage_text:
            return 0.0

        stop = {
            'the','a','an','and','or','but','if','then','than','to','of','in','on','for','with','by','from','as','at',
            'is','are','was','were','be','been','being','it','this','that','these','those','which','who','whom','what',
            'when','where','why','how','can','may','might','could','should','would','will','do','does','did','not','no'
        }

        def tokens(text):
            words = re.findall(r"[a-zA-Z]+", text.lower())
            return [w for w in words if len(w) >= 3 and w not in stop]

        q_tokens = set(tokens(query))
        if not q_tokens:
            return 0.0
        p_tokens = set(tokens(passage_text))
        overlap = q_tokens.intersection(p_tokens)
        return min(1.0, len(overlap) / max(1, len(q_tokens)))
    
    def get_relevance_decision(self, auto_label=None, fallback_label=None):
        """Get user input for labeling or use auto-label decision.
        
        If auto_label is set to a decision (relevant/unrelated/question),
        skips the prompt and returns that label immediately. This reduces
        user fatigue by only prompting for uncertain passages.
        
        If auto_label is 'review', prompts user with keyboard shortcuts:
        - Enter/Y: Mark as relevant (default)
        - N: Mark as unrelated
        - Q: Mark as question
        
        This efficiency measure means users only interact with ~20-40%
        of passages (those in the uncertain middle range).
        """
        if auto_label in ['relevant', 'unrelated', 'question']:
            # Auto-labeled passage: return decision without prompting user
            return auto_label
        
        if self.auto_labeling_mode:
            return fallback_label or 'relevant'

        # Passage needs human review: show prompt
        while True:
            decision = input("\nLabel: [Enter=relevant] n=unrelated q=question: ").strip().lower()
            # Default to 'relevant' if user just presses Enter
            if decision == '' or decision in ['y', 'yes']:
                return 'relevant'
            elif decision in ['n', 'no']:
                return 'unrelated'
            elif decision in ['q', 'question']:
                return 'question'
            else:
                print("Invalid input. Press Enter for relevant, 'n' for unrelated, or 'q' for question.")

    def detect_stance_heuristic(self, passage_text):
        """Detect stance using medical-aware keyword heuristics.
        
        Optimized for medical fact-checking by recognizing:
        - Causal medical language: "lead to", "linked to", "associated with", "cause"
        - Risk/benefit language: "increase risk", "protective", "improve", "worsen"
        - Negations: "not", "no evidence", "contraindicated", "avoid"
        
        Separates multi-word phrases (2x weight) from single keywords (1x weight)
        to avoid double-counting and improve accuracy.
        
        Returns (stance, confidence) or (None, None) if inconclusive.
        """
        import re

        text_lower = passage_text.lower()
        
        # Multi-word support phrases (2x weight due to higher specificity)
        # These capture medical causality language that's critical for fact-checking
        support_phrases = [
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
        # These capture benefit/protective language
        support_single = [
            'helps', 'helpful', 'beneficial', 'benefit', 'benefits',
            'improves', 'improvement', 'improve',
            'prevents', 'prevention', 'preventive', 'protective', 'protection',
            'treats', 'treatment', 'therapeutic',
            'supports', 'support', 'supporting',
            'effective', 'efficacy', 'efficacious',
            'proven', 'evidence', 'robust',
            'good', 'positive', 'positively',
            'promotes', 'promotion', 'promotes',
            'enhances', 'enhancement', 'enhance',
            'strengthens', 'strengthen', 'strength',
            'promotes', 'promotion', 'promoter',
            'aids', 'aid', 'assists', 'assist',
            'trigger', 'triggers', 'triggered'
        ]
        
        # Multi-word refute phrases (2x weight)
        refute_phrases = [
            'no evidence', 'lack of evidence', 'insufficient evidence',
            'does not', 'do not', 'did not',
            'does not cause', 'does not lead',
            'not linked', 'not associated',
            'no association', 'no link',
            'not responsible'
        ]
        
        # Single-word refute keywords (1x weight)
        refute_single = [
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
        
        support_count = 0
        refute_count = 0
        
        # Count multi-word support phrases (2x weight)
        strong_support_hit = False
        for phrase in support_phrases:
            occurrences = text_lower.count(phrase)
            if occurrences > 0:
                support_count += occurrences * 2
                strong_support_hit = True
        
        # Count multi-word refute phrases (2x weight)
        for phrase in refute_phrases:
            occurrences = text_lower.count(phrase)
            if occurrences > 0:
                refute_count += occurrences * 2
        
        def count_whole_word(word):
            pattern = r'\b' + re.escape(word) + r'\b'
            return len(re.findall(pattern, text_lower))

        # Count single-word support keywords (1x weight) using word boundaries
        for keyword in support_single:
            occurrences = count_whole_word(keyword)
            support_count += occurrences

        # Count single-word refute keywords (1x weight) using word boundaries
        for keyword in refute_single:
            occurrences = count_whole_word(keyword)
            refute_count += occurrences
        
        # Decision logic
        total_signals = support_count + refute_count
        
        # No clear signal either way
        if total_signals == 0:
            return None, None
        
        # Determine stance based on signal strength
        if support_count > refute_count:
            # Confidence: higher weight on support signals
            # Normalized: support_count / (support_count + refute_count)
            # Add 1 to denominator to prevent extreme confidence from single strong signal
            confidence = min(1.0, support_count / (total_signals + 1))
            if strong_support_hit and refute_count == 0:
                confidence = max(confidence, 0.85)
            return 'supports', confidence
        elif refute_count > support_count:
            confidence = min(1.0, refute_count / (total_signals + 1))
            return 'refutes', confidence
        else:
            # Equal support and refute signals = unclear stance
            return None, None
    
    def predict_stance_nli(self, query, passage_text):
        """Predict stance using NLI cross-encoder (entailment/contradiction).
        
        Uses the NLI cross-encoder to determine if passage entails (supports),
        contradicts (refutes), or is neutral to the query claim.
        
        Returns tuple: (stance_label, confidence)
        """
        if self.nli_cross_encoder is None:
            return None, None
        
        try:
            pair = [[query, passage_text]]
            scores = self.nli_cross_encoder.predict(pair)[0]  # Returns logits for one pair
            
            # Extract entailment, neutral, contradiction (standard NLI order)
            if isinstance(scores, np.ndarray) and len(scores.shape) == 0:
                # Single scalar
                return None, None
            
            if isinstance(scores, np.ndarray) and scores.shape[0] >= 3:
                entail_score = float(scores[2])  # ENTAILMENT
                contra_score = float(scores[0])  # CONTRADICTION
                
                if entail_score > contra_score:
                    confidence = float(np.tanh(entail_score / 2.0))  # Normalize to [0,1]
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
    
    def predict_stance(self, query, passage_text):
        """Predict stance (supports/refutes/neutral) using medical heuristics ONLY.
        
        For medical fact-checking, heuristics are more reliable than NLI/MNLI models
        because they capture medical causality language ("lead to", "linked to", "cause").
        
        NLI/MNLI models treat statistical/hedged medical language as "neutral" which is
        fundamentally wrong for medical fact-checking. We NEVER use NLI/MNLI fallback.
        
        Returns tuple: (stance_label, confidence) or (None, None) if no clear signal.
        """
        if not passage_text:
            return None, None
        
        # ONLY use medical-aware heuristic keyword detection
        # No fallback to MNLI - heuristics are domain-appropriate for medical facts
        heuristic_stance, heuristic_conf = self.detect_stance_heuristic(passage_text)
        return heuristic_stance, heuristic_conf

    def get_stance_decision(self, suggest=None, suggest_confidence=None):
        """Prompt user for stance when relevance is 'relevant'.
        If a suggestion is provided, show it but require explicit confirmation.
        """
        suggestion_text = ""
        if suggest:
            if suggest_confidence is not None:
                suggestion_text = f" [suggested: {suggest} ({suggest_confidence:.2f})]"
            else:
                suggestion_text = f" [suggested: {suggest}]"

        if self.auto_labeling_mode:
            if suggest:
                conf = suggest_confidence if suggest_confidence is not None else 0.50
                return suggest, conf
            return 'neutral', 0.50

        prompt = f"\nStance: s=supports r=refutes n=neutral{suggestion_text}: "
        while True:
            decision = input(prompt).strip().lower()
            if decision == 's':
                conf = suggest_confidence if suggest == 'supports' and suggest_confidence is not None else 0.50
                return 'supports', conf
            if decision == 'r':
                conf = suggest_confidence if suggest == 'refutes' and suggest_confidence is not None else 0.50
                return 'refutes', conf
            if decision == 'n':
                conf = suggest_confidence if suggest == 'neutral' and suggest_confidence is not None else 0.50
                return 'neutral', conf
            print("Invalid input. Choose: s (supports), r (refutes), n (neutral).")
    
    def save_labeled_passage(self, passage, query, decision, rerank_score, faiss_score, stance=None, stance_confidence=None):
        """Save labeled passage to appropriate file based on decision.
        
        Creates a labeled item with complete metadata, stance (for relevant), and scores,
        then appends to the appropriate JSON file based on the label.
        Saves immediately to disk to protect against data loss from crashes.
        """
        labeled_item = {
            'passage_id': passage.get('passage_id'),
            'query': query,
            'label': decision,  # 'relevant', 'unrelated', or 'question'
            'stance': stance if decision == 'relevant' else None,
            'stance_confidence': float(stance_confidence) if (stance_confidence is not None and decision == 'relevant') else None,
            'text': passage.get('text'),
            'section_heading': passage.get('section_heading'),
            'title': passage.get('title'),
            'url': passage.get('url'),
            'author': passage.get('author'),
            'medically_reviewed_by': passage.get('medically_reviewed_by'),
            'sources': passage.get('sources', []),
            'published_date': passage.get('published_date'),
            'scores': {
                'faiss_similarity': float(faiss_score),
                'cross_encoder_relevance': float(rerank_score)
            },
            'labeled_at': datetime.now().isoformat()
        }
        
        # Save to appropriate file based on decision
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if decision == 'relevant':
            self.relevant_passages.append(labeled_item)
            with self._open_output_file(RELEVANT_PASSAGES_FILE) as f:
                json.dump(self.relevant_passages, f, indent=2, ensure_ascii=False)
        elif decision == 'unrelated':
            self.unrelated_passages.append(labeled_item)
            with self._open_output_file(UNRELATED_PASSAGES_FILE) as f:
                json.dump(self.unrelated_passages, f, indent=2, ensure_ascii=False)
        elif decision == 'question':
            self.question_passages.append(labeled_item)
            with self._open_output_file(QUESTION_PASSAGES_FILE) as f:
                json.dump(self.question_passages, f, indent=2, ensure_ascii=False)
        
        return labeled_item
    
    def label_session(self, query, num_results=DEFAULT_DISPLAY_RESULTS, topk=DEFAULT_TOPK_FAISS):
        """Interactive labeling session for a query.
        
        Workflow:
        1. Search for top-k passages using FAISS similarity
        2. Rerank with cross-encoder for higher quality
        3. Compute SapBERT medical entity scores
        4. Calculate combined weighted score
        5. Detect single-sentence questions
        6. Auto-label high/low confidence passages; prompt for uncertain ones
        7. Collect statistics on auto-labeling distribution
        8. Display comprehensive score distribution analysis
        """
        print(f"\nSearching for passages related to: '{query}'")
        print("Please wait...\n")
        
        # Search top-k passages using FAISS (fast semantic similarity)
        faiss_scores, indices = self.search(query, topk=topk)
        if self.last_query_concepts:
            print(f"Concept expansion: {', '.join(self.last_query_concepts)}")
        
        # Filter valid results: FAISS padding uses -1 for invalid indices
        candidates = []
        for score, idx in zip(faiss_scores, indices):
            if idx >= 0 and idx < len(self.metadata):
                # Cap FAISS scores at 0.95 (exact 1.0 is suspicious and usually indicates incomplete passages)
                capped_score = min(0.95, float(score))
                candidates.append((idx, capped_score))
        
        # Take top-N results for display (default 10)
        top_candidates = candidates[:num_results]
        passages = [self.metadata[idx] for idx, _ in top_candidates]
        faiss_scores_top = [score for _, score in top_candidates]
        
        print(f"Found {len(candidates)} passages, displaying top {len(top_candidates)}\n")
        
        # Get reranking and SapBERT scores for more accurate relevance assessment
        rerank_scores = self.rerank_results(query, passages)
        passage_indices = [idx for idx, _ in top_candidates]
        sapbert_scores = self.compute_sapbert_similarities(query, passage_indices)
        
        # Initialize session data and analytics counters
        session_data = {
            'query': query,
            'session_start': self.session_start.isoformat(),
            'labeled_passages': []
        }
        
        # Track scores for analysis
        all_scores = []
        auto_relevant_count = 0
        auto_unrelated_count = 0
        auto_question_count = 0
        human_review_count = 0
        
        # Process each passage with scoring and auto-labeling
        for i, (passage, faiss_score, rerank_score, sapbert_score) in enumerate(zip(passages, faiss_scores_top, rerank_scores, sapbert_scores), 1):
            # Check if passage has substantive content (skip title-only passages)
            if not self.is_substantive_passage(passage.get('text', '')):
                print(f"[{i}] {passage.get('title', 'Untitled')} [SKIPPED: Insufficient text content]")
                continue
            
            lexical_score = self.compute_lexical_overlap(query, passage.get('text', ''))

            # Calculate combined score with credibility signals
            is_reviewed = bool(passage.get('medically_reviewed_by'))
            has_author = bool(passage.get('author'))
            combined_score = self.calculate_combined_score(
                faiss_score, rerank_score, sapbert_score, lexical_score,
                is_medically_reviewed=is_reviewed,
                has_author=has_author
            )
            all_scores.append(combined_score)
            
            # Determine auto-label using multiple criteria
            auto_label = None
            auto_label_reason = None
            # Check 1: Is this passage a question?
            if self.is_single_sentence_question(passage.get('text', '')):
                auto_label = 'question'
                auto_label_reason = 'question'
                auto_question_count += 1
            # Check 1b: Very low CE + low lexical overlap → likely unrelated
            elif rerank_score <= AUTO_UNRELATED_CE_MAX and lexical_score <= AUTO_UNRELATED_LEX_MAX:
                auto_label = 'unrelated'
                auto_label_reason = 'low_ce_lex'
                auto_unrelated_count += 1
            # Check 2: High confidence relevant (score >= 0.75)
            elif combined_score >= AUTO_RELEVANT_THRESHOLD:
                auto_label = 'relevant'
                auto_label_reason = 'score_high'
                auto_relevant_count += 1
            # Check 3: High confidence unrelated (score <= 0.35)
            elif combined_score <= AUTO_UNRELATED_THRESHOLD:
                auto_label = 'unrelated'
                auto_label_reason = 'score_low'
                auto_unrelated_count += 1
            # Otherwise: Uncertain - needs human review
            else:
                auto_label = 'review'
                human_review_count += 1
            
            # Display passage with auto-label status and all scores
            self.display_passage(
                i,
                passage,
                query,
                faiss_score,
                rerank_score,
                sapbert_score,
                combined_score,
                auto_label,
                lexical_score,
                auto_label_reason=auto_label_reason,
            )
            
            # Get labeling decision (auto-labeled or human review)
            # Skips prompt for auto-labeled passages, only prompts for 'review' label
            mid = (AUTO_RELEVANT_THRESHOLD + AUTO_UNRELATED_THRESHOLD) / 2.0
            fallback_label = 'relevant' if combined_score >= mid else 'unrelated'
            decision = self.get_relevance_decision(auto_label=auto_label, fallback_label=fallback_label)
            stance = None
            stance_confidence = None
            stance_source = None

            # If passage is relevant, attempt stance auto-detection with human fallback
            if decision == 'relevant':
                # Prefer NLI stance if available, then fallback to medical heuristics
                stance_suggest, stance_conf = self.predict_stance_nli(query, passage.get('text', ''))
                if stance_suggest is None:
                    stance_suggest, stance_conf = self.predict_stance(query, passage.get('text', ''))

                stance_confidence = stance_conf

                if self.auto_labeling_mode:
                    if stance_suggest is None:
                        stance = 'neutral'
                        stance_confidence = 0.50
                    else:
                        stance = stance_suggest
                        stance_confidence = stance_confidence if stance_confidence is not None else 0.50
                    stance_source = 'auto'
                else:
                    # Auto-accept only if high confidence AND not neutral
                    if (
                        stance_suggest
                        and stance_suggest != 'neutral'
                        and stance_confidence is not None
                        and stance_confidence >= STANCE_AUTO_THRESHOLD
                    ):
                        stance = stance_suggest
                        stance_source = 'auto'
                    else:
                        # Ask user, showing suggestion (even if neutral) for human oversight
                        stance, stance_confidence = self.get_stance_decision(
                            suggest=stance_suggest,
                            suggest_confidence=stance_confidence,
                        )
                        stance_source = 'manual'

            # Save label immediately to disk for crash safety
            # This ensures no work is lost if the program crashes mid-session
            labeled = self.save_labeled_passage(
                passage, query, decision, rerank_score, faiss_score,
                stance=stance, stance_confidence=stance_confidence
            )
            session_data['labeled_passages'].append(labeled)

            # Show user confirmation of label assignment
            label_symbols = {'relevant': 'RELEVANT ✓', 'unrelated': 'UNRELATED ✗', 'question': 'QUESTION ❓'}
            print(f"→ Labeled as: {label_symbols[decision]}")
            if decision == 'relevant' and stance:
                conf_text = f" ({stance_confidence:.2f})" if stance_confidence is not None else ""
                source_text = f"[{stance_source}]" if stance_source else ""
                print(f"   Stance: {stance}{conf_text} {source_text}")

        # Save complete session record
        with LABELING_SESSION_FILE.open('w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        # Summary with auto-labeling distribution
        print(f"\n{'='*70}")
        print("Session Summary")
        print(f"{'='*70}")
        print(f"Total labeled: {len(session_data['labeled_passages'])}")
        # Show breakdown of auto-labeling effectiveness
        print(f"  Auto-Relevant: {auto_relevant_count} ({100*auto_relevant_count/len(passages):.1f}%)")
        print(f"  Auto-Unrelated: {auto_unrelated_count} ({100*auto_unrelated_count/len(passages):.1f}%)")
        print(f"  Auto-Question: {auto_question_count} ({100*auto_question_count/len(passages):.1f}%)")
        print(f"  Human Review: {human_review_count} ({100*human_review_count/len(passages):.1f}%)")

        # Final label distribution after human decisions
        if session_data['labeled_passages']:
            final_relevant = sum(1 for p in session_data['labeled_passages'] if p['label'] == 'relevant')
            final_unrelated = sum(1 for p in session_data['labeled_passages'] if p['label'] == 'unrelated')
            final_questions = sum(1 for p in session_data['labeled_passages'] if p['label'] == 'question')
            print(f"\nFinal Results:")
            print(f"  Relevant: {final_relevant}")
            print(f"  Unrelated: {final_unrelated}")
            print(f"  Questions: {final_questions}")
            # Stance breakdown (only for relevant passages)
            stance_supports = sum(1 for p in session_data['labeled_passages'] if p.get('label') == 'relevant' and p.get('stance') == 'supports')
            stance_refutes = sum(1 for p in session_data['labeled_passages'] if p.get('label') == 'relevant' and p.get('stance') == 'refutes')
            stance_neutral = sum(1 for p in session_data['labeled_passages'] if p.get('label') == 'relevant' and p.get('stance') == 'neutral')
            if final_relevant:
                print(f"  Stance (relevant): supports={stance_supports}, refutes={stance_refutes}, neutral={stance_neutral}")

        # Score distribution analysis: helps understand score calibration
        # If many scores cluster near thresholds, thresholds may need tuning
        if all_scores:
            print(f"\nScore Distribution Analysis:")
            print(f"  Mean: {np.mean(all_scores):.3f}")
            print(f"  Median: {np.median(all_scores):.3f}")
            print(f"  Std Dev: {np.std(all_scores):.3f}")
            print(f"  Min: {np.min(all_scores):.3f} | Max: {np.max(all_scores):.3f}")

            # Breakdown by threshold zones for calibration assessment
            below_unrelated = sum(1 for s in all_scores if s < AUTO_UNRELATED_THRESHOLD)
            review_zone = sum(1 for s in all_scores if AUTO_UNRELATED_THRESHOLD <= s <= AUTO_RELEVANT_THRESHOLD)
            above_relevant = sum(1 for s in all_scores if s > AUTO_RELEVANT_THRESHOLD)
            print(f"\n  Below unrelated threshold (<{AUTO_UNRELATED_THRESHOLD}): {below_unrelated}")
            print(f"  In review zone ({AUTO_UNRELATED_THRESHOLD}-{AUTO_RELEVANT_THRESHOLD}): {review_zone}")
            print(f"  Above relevant threshold (>{AUTO_RELEVANT_THRESHOLD}): {above_relevant}")

        # Show output file paths for reference
        print(f"\nOutput files:")
        print(f"  Relevant: {RELEVANT_PASSAGES_FILE}")
        print(f"  Unrelated: {UNRELATED_PASSAGES_FILE}")
        print(f"  Questions: {QUESTION_PASSAGES_FILE}")
        print(f"  Session: {LABELING_SESSION_FILE}")
        print(f"{'='*70}\n")

    def _load_verified_claims(self):
        """Load verified claims from disk for auto-labeling mode."""
        if not VERIFIED_CLAIMS_FILE.exists():
            print(f"Verified claims file not found at {VERIFIED_CLAIMS_FILE}")
            return []
        try:
            with VERIFIED_CLAIMS_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
            claims = []
            for item in data:
                if isinstance(item, dict):
                    text = item.get('claim') or item.get('text')
                    if text:
                        claims.append({"id": item.get("id"), "claim": text.strip()})
                elif isinstance(item, str):
                    claims.append({"id": None, "claim": item.strip()})
            return [c for c in claims if c.get("claim")]
        except Exception as e:
            print(f"Failed to read verified claims: {e}")
            return []

    def _load_processed_claim_ids(self):
        """Load processed claim IDs from disk to avoid reprocessing."""
        path = OUTPUT_DIR / "processed_claim_ids.json"
        if not path.exists():
            return set()
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data) if isinstance(data, list) else set()
        except Exception:
            return set()

    def _save_processed_claim_ids(self, ids_set):
        """Persist processed claim IDs to disk."""
        path = OUTPUT_DIR / "processed_claim_ids.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(sorted(ids_set), f, indent=2, ensure_ascii=False)

    def auto_label_from_claims(self, num_results=DEFAULT_DISPLAY_RESULTS):
        """Run fully automated labeling over verified claims."""
        if not self.ensure_models_initialized():
            print("Failed to initialize models. Aborting auto-label run.")
            return

        claims = self._load_verified_claims()
        if not claims:
            print("No verified claims found. Aborting auto-label run.")
            return

        processed_ids = self._load_processed_claim_ids()

        prev_mode = self.auto_labeling_mode
        self.auto_labeling_mode = True

        print(f"\nAuto-labeling enabled. Running {len(claims)} claims...")
        for i, item in enumerate(claims, 1):
            claim_id = item.get("id")
            claim_text = item.get("claim")
            if claim_id is not None and claim_id in processed_ids:
                print(f"\n[{i}/{len(claims)}] Skipping claim id {claim_id} (already processed)")
                continue
            print(f"\n[{i}/{len(claims)}] Claim: {claim_text}")
            self.label_session(claim_text, num_results=num_results)
            if claim_id is not None:
                processed_ids.add(claim_id)

        if processed_ids:
            self._save_processed_claim_ids(processed_ids)

        self.auto_labeling_mode = prev_mode

    def _load_fake_claims(self):
        """Load fake/false claims from disk for refute labeling."""
        if not FAKE_CLAIMS_FILE.exists():
            print(f"Fake claims file not found at {FAKE_CLAIMS_FILE}")
            return []
        try:
            with FAKE_CLAIMS_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
            claims = []
            for item in data:
                if isinstance(item, dict):
                    text = item.get('claim') or item.get('text')
                    if text:
                        claims.append({"id": item.get("id"), "claim": text.strip()})
                elif isinstance(item, str):
                    claims.append({"id": None, "claim": item.strip()})
            return [c for c in claims if c.get("claim")]
        except Exception as e:
            print(f"Failed to read fake claims: {e}")
            return []

    def auto_label_refutes_from_fake_claims(self, num_results=DEFAULT_DISPLAY_RESULTS):
        """Auto-label fake claims as REFUTES based on relevant passages found.
        
        For each fake claim, search for related passages. Relevant passages are
        automatically labeled with stance=refutes (since they refute the false claim).
        This efficiently builds the refute portion of the training dataset.
        
        Note: Skips loading stance detection models since stance is hardcoded to 'refutes'.
        """
        if not self.ensure_models_initialized(skip_stance=True):
            print("Failed to initialize models. Aborting fake claim run.")
            return

        fake_claims = self._load_fake_claims()
        if not fake_claims:
            print("No fake claims found. Aborting fake claim labeling run.")
            return

        processed_ids = self._load_processed_claim_ids()

        prev_mode = self.auto_labeling_mode
        self.auto_labeling_mode = True

        print(f"\nAuto-labeling refutes from {len(fake_claims)} fake claims...")
        for i, item in enumerate(fake_claims, 1):
            claim_id = item.get("id")
            claim_text = item.get("claim")
            if claim_id is not None and claim_id in processed_ids:
                print(f"\n[{i}/{len(fake_claims)}] Skipping fake claim id {claim_id} (already processed)")
                continue
            print(f"\n[{i}/{len(fake_claims)}] Fake claim: {claim_text}")
            self._label_session_for_fake_claim(claim_text, num_results=num_results)
            if claim_id is not None:
                processed_ids.add(claim_id)

        if processed_ids:
            self._save_processed_claim_ids(processed_ids)

        self.auto_labeling_mode = prev_mode
        print(f"\n✓ Fake claim refute labeling complete!")

    def _label_session_for_fake_claim(self, query, num_results=DEFAULT_DISPLAY_RESULTS, topk=DEFAULT_TOPK_FAISS):
        """Labeling session for fake claims - auto-forces refutes stance for relevant passages.
        
        Only relevant passages are labeled (unrelated are discarded).
        All relevant passages get stance=refutes since they refute the false claim.
        """
        print(f"\nSearching for passages related to: '{query}'")
        print("Please wait...\n")
        
        faiss_scores, indices = self.search(query, topk=topk)
        if self.last_query_concepts:
            print(f"Concept expansion: {', '.join(self.last_query_concepts)}")
        
        candidates = []
        for score, idx in zip(faiss_scores, indices):
            if idx >= 0 and idx < len(self.metadata):
                capped_score = min(0.95, float(score))
                candidates.append((idx, capped_score))
        
        top_candidates = candidates[:num_results]
        passages = [self.metadata[idx] for idx, _ in top_candidates]
        faiss_scores_top = [score for _, score in top_candidates]
        
        print(f"Found {len(candidates)} passages, displaying top {len(top_candidates)}\n")
        
        rerank_scores = self.rerank_results(query, passages)
        passage_indices = [idx for idx, _ in top_candidates]
        sapbert_scores = self.compute_sapbert_similarities(query, passage_indices)
        
        session_data = {
            'query': query,
            'session_start': self.session_start.isoformat(),
            'labeled_passages': []
        }
        
        all_scores = []
        labeled_count = 0
        skipped_count = 0
        
        for i, (passage, faiss_score, rerank_score, sapbert_score) in enumerate(zip(passages, faiss_scores_top, rerank_scores, sapbert_scores), 1):
            if not self.is_substantive_passage(passage.get('text', '')):
                print(f"[{i}] {passage.get('title', 'Untitled')} [SKIPPED: Insufficient text content]")
                skipped_count += 1
                continue
            
            lexical_score = self.compute_lexical_overlap(query, passage.get('text', ''))
            is_reviewed = bool(passage.get('medically_reviewed_by'))
            has_author = bool(passage.get('author'))
            combined_score = self.calculate_combined_score(
                faiss_score, rerank_score, sapbert_score, lexical_score,
                is_medically_reviewed=is_reviewed,
                has_author=has_author
            )
            all_scores.append(combined_score)
            
            # Auto-determine relevance
            auto_label = None
            auto_label_reason = None
            
            if self.is_single_sentence_question(passage.get('text', '')):
                auto_label = 'question'
                auto_label_reason = 'question'
            elif rerank_score <= AUTO_UNRELATED_CE_MAX and lexical_score <= AUTO_UNRELATED_LEX_MAX:
                auto_label = 'unrelated'
                auto_label_reason = 'low_ce_lex'
            elif combined_score >= AUTO_RELEVANT_THRESHOLD:
                auto_label = 'relevant'
                auto_label_reason = 'score_high'
            elif combined_score <= AUTO_UNRELATED_THRESHOLD:
                auto_label = 'unrelated'
                auto_label_reason = 'score_low'
            else:
                auto_label = 'review'
            
            self.display_passage(
                i, passage, query, faiss_score, rerank_score, sapbert_score,
                combined_score, auto_label, lexical_score, auto_label_reason=auto_label_reason,
            )
            
            # For fake claims: ONLY label relevant passages with refutes stance
            # Skip unrelated/question passages (no training value)
            if auto_label == 'relevant':
                labeled = self.save_labeled_passage(
                    passage, query, 'relevant', rerank_score, faiss_score,
                    stance='refutes', stance_confidence=0.95  # High confidence: false claim → refutes
                )
                session_data['labeled_passages'].append(labeled)
                labeled_count += 1
                print(f"→ Labeled as: RELEVANT ✓ [REFUTES (fake claim)]")
            else:
                skipped_count += 1
                print(f"→ Skipped: {auto_label.upper()} (no training value for fake claims)")
        
        with LABELING_SESSION_FILE.open('w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"Fake Claim Session Summary")
        print(f"{'='*70}")
        print(f"Passages labeled (refutes): {labeled_count}")
        print(f"Passages skipped: {skipped_count}")
        if all_scores:
            print(f"\nScore Distribution:")
            print(f"  Mean: {np.mean(all_scores):.3f}")
            print(f"  Median: {np.median(all_scores):.3f}")
        print(f"{'='*70}\n")

    def _collect_labels(self, subset="all"):
        """Collect labeled items by subset."""
        subset = (subset or "all").strip().lower()
        if subset == "relevant":
            return list(self.relevant_passages)
        if subset == "unrelated":
            return list(self.unrelated_passages)
        if subset == "question":
            return list(self.question_passages)
        if subset in {"rel+unrel", "relevant+unrelated", "relevant_unrelated"}:
            return list(self.relevant_passages) + list(self.unrelated_passages)
        return list(self.relevant_passages) + list(self.unrelated_passages) + list(self.question_passages)

    def export_labeled_data(self, export_format="csv", out_path=None, subset="all"):
        """Export labeled data to CSV/TSV/JSON/JSONL."""
        export_format = (export_format or "csv").strip().lower()
        data = self._collect_labels(subset=subset)
        if not data:
            print("No labeled data to export.")
            return

        if export_format not in {"csv", "tsv", "json", "jsonl"}:
            print("Unsupported format. Use csv, tsv, json, or jsonl.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "jsonl" if export_format == "jsonl" else export_format
        safe_subset = (subset or "all").strip().lower().replace("+", "_").replace(" ", "_")
        out_file = out_path or (OUTPUT_DIR / f"labeled_export_{safe_subset}_{timestamp}.{ext}")

        if export_format in {"csv", "tsv"}:
            import csv

            delimiter = "," if export_format == "csv" else "\t"
            fieldnames = [
                "passage_id",
                "query",
                "label",
                "stance",
                "stance_confidence",
                "title",
                "section_heading",
                "url",
                "author",
                "medically_reviewed_by",
                "published_date",
                "text",
                "faiss_similarity",
                "cross_encoder_relevance",
                "labeled_at",
            ]

            with Path(out_file).open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                for item in data:
                    scores = item.get("scores") or {}
                    row = {
                        "passage_id": item.get("passage_id"),
                        "query": item.get("query"),
                        "label": item.get("label"),
                        "stance": item.get("stance"),
                        "stance_confidence": item.get("stance_confidence"),
                        "title": item.get("title"),
                        "section_heading": item.get("section_heading"),
                        "url": item.get("url"),
                        "author": item.get("author"),
                        "medically_reviewed_by": item.get("medically_reviewed_by"),
                        "published_date": item.get("published_date"),
                        "text": item.get("text"),
                        "faiss_similarity": scores.get("faiss_similarity"),
                        "cross_encoder_relevance": scores.get("cross_encoder_relevance"),
                        "labeled_at": item.get("labeled_at"),
                    }
                    writer.writerow(row)

        elif export_format == "json":
            with Path(out_file).open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif export_format == "jsonl":
            with Path(out_file).open("w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Exported {len(data)} records to {out_file}")


def main():
    """Main menu loop for the VeriFact admin labeling system.
    
    Provides interactive CLI with these options:
    1. Label passages: Start an interactive session for a query
    2. Use random question: Auto-select a random question from labeled questions
    3. View statistics: Show count of labeled passages by category
    4. Export data: (Future feature)
    5. Exit: Close the application
    """
    labeler = PassageLabeler()
    
    print("\n" + "="*70)
    print("VERIFACT - Medical Fact Verification Admin Tool")
    print("="*70)
    
    # Main menu loop
    while True:
        print("\n" + "="*70)
        print("Main Menu")
        print("="*70)
        print("1. Label passages (interactive mode)")
        print("2. Use random question from labeled passages")
        print("3. View labeling statistics")
        print("4. Export labeled data")
        print("5. Auto-label using verified claims")
        print("6. Auto-label REFUTES from fake claims")
        print("7. Exit")
        print("="*70)
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            # Start interactive labeling session
            print("\n" + "-"*70)
            if not labeler.ensure_models_initialized():
                print("Failed to initialize models. Returning to menu.")
                continue
            query = input("Enter medical query/claim to investigate: ").strip()
            if not query:
                print("Query cannot be empty.")
                continue
            
            # Get number of passages to review (default: 10)
            print("\nEnter number of passages to review (default: 10): ", end="")
            try:
                num_results = int(input().strip() or DEFAULT_DISPLAY_RESULTS)
            except ValueError:
                num_results = DEFAULT_DISPLAY_RESULTS
            
            # Run the labeling session
            labeler.label_session(query, num_results=num_results)
        
        elif choice == '2':
            # Use a random question from labeled questions
            if not labeler.ensure_models_initialized():
                print("Failed to initialize models. Returning to menu.")
                continue
            if not labeler.question_passages:
                print("\nNo labeled questions available. Label some questions first.")
                continue
            
            random_question = random.choice(labeler.question_passages)
            query = random_question.get('text', '').strip()
            
            if not query:
                print("\nSelected question has no text. Try again.")
                continue
            
            print(f"\nUsing random question: '{query}'")
            
            # Get number of passages to review (default: 10)
            print("\nEnter number of passages to review (default: 10): ", end="")
            try:
                num_results = int(input().strip() or DEFAULT_DISPLAY_RESULTS)
            except ValueError:
                num_results = DEFAULT_DISPLAY_RESULTS
            
            # Run the labeling session with the randomly selected question
            labeler.label_session(query, num_results=num_results)
        
        elif choice == '3':
            # Show labeling statistics
            print("\n" + "-"*70)
            print("Labeling Statistics")
            print("-"*70)
            print(f"Relevant passages labeled: {len(labeler.relevant_passages)}")
            print(f"Unrelated passages labeled: {len(labeler.unrelated_passages)}")
            print(f"Question passages labeled: {len(labeler.question_passages)}")
            total = len(labeler.relevant_passages) + len(labeler.unrelated_passages) + len(labeler.question_passages)
            print(f"Total labeled: {total}")

            if labeler.relevant_passages:
                stance_supports = sum(1 for p in labeler.relevant_passages if p.get('stance') == 'supports')
                stance_refutes = sum(1 for p in labeler.relevant_passages if p.get('stance') == 'refutes')
                stance_neutral = sum(1 for p in labeler.relevant_passages if p.get('stance') == 'neutral')
                print("\nStance distribution (relevant):")
                print(f"  Supports: {stance_supports}")
                print(f"  Refutes: {stance_refutes}")
                print(f"  Neutral: {stance_neutral}")
            
            # Show unique queries per category (for diversity analysis)
            if labeler.relevant_passages:
                print(f"\nUnique queries in relevant: {len(set(p['query'] for p in labeler.relevant_passages))}")
            if labeler.unrelated_passages:
                print(f"Unique queries in unrelated: {len(set(p['query'] for p in labeler.unrelated_passages))}")
            if labeler.question_passages:
                print(f"Unique queries in questions: {len(set(p['query'] for p in labeler.question_passages))}")
        
        elif choice == '4':
            print("\nExport labeled data")
            print("Choose format: csv, tsv, json, jsonl")
            export_format = input("Format [csv]: ").strip().lower() or "csv"
            print("Choose subset: relevant, unrelated, question, rel+unrel, all")
            subset = input("Subset [all]: ").strip().lower() or "all"
            labeler.export_labeled_data(export_format=export_format, subset=subset)

        elif choice == '5':
            # Fully automated labeling using verified claims
            labeler.auto_label_from_claims(num_results=DEFAULT_DISPLAY_RESULTS)

        elif choice == '6':
            # Auto-label refutes from fake claims
            labeler.auto_label_refutes_from_fake_claims(num_results=DEFAULT_DISPLAY_RESULTS)

        elif choice == '7':
            print("\nThank you for using VeriFact Admin Tool. Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-7.")


if __name__ == '__main__':
    """Entry point with error handling.
    
    Gracefully handles:
    - KeyboardInterrupt: User presses Ctrl+C
    - Exceptions: Unexpected errors are caught and displayed
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
