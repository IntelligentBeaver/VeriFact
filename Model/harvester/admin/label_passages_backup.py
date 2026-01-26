"""
Interactive Passage Labeling System
Labels passages from the medical index as RELEVANT or UNRELATED to a query.
Stores results with full metadata and query information.
"""

import json
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from config import (
    INDEX_DIR, EMBEDDING_MODEL, CROSS_ENCODER_MODEL, SAPBERT_MODEL,
    DEFAULT_TOPK_FAISS, DEFAULT_RERANK_K, DEFAULT_DISPLAY_RESULTS,
    AUTO_RELEVANT_THRESHOLD, AUTO_UNRELATED_THRESHOLD,
    RELEVANT_PASSAGES_FILE, UNRELATED_PASSAGES_FILE, QUESTION_PASSAGES_FILE,
    LABELING_SESSION_FILE
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
        self.sapbert_model = None
        self.sapbert_embeddings = None
        
        # Persisted labels from prior sessions
        self.relevant_passages = self._load_existing("relevant")
        self.unrelated_passages = self._load_existing("unrelated")
        self.question_passages = self._load_existing("question")
        self.session_start = datetime.now()
        
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
    
    def initialize_models(self):
        """Load all required models and indices."""
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
        print("\nStep 4: Loading cross-encoder...")
        print(f"  Model: {CROSS_ENCODER_MODEL}")
        try:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            print(f"  ✓ Cross-encoder loaded")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load cross-encoder: {e}")
        
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
        
        print("\n" + "="*60)
        print("✓ All models loaded successfully!")
        print("="*60)
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
        qvec = self.embed_query(query)
        # Reshape for FAISS: expects (batch_size, embedding_dim)
        q = np.expand_dims(qvec.astype(np.float32), axis=0)
        # FAISS returns scores (similarity values) and indices of top-k results
        scores, indices = self.index.search(q, topk)
        return scores[0], indices[0]
    
    def rerank_results(self, query, passages):
        """Rerank passages using a cross-encoder model.
        
        Cross-encoder provides higher-quality relevance scoring by computing
        query-passage interaction directly. Outputs real-valued scores (typically -1 to +1).
        Higher values = more relevant to the query.
        
        Returns array of relevance scores, one per passage.
        """
        if self.cross_encoder is None:
            # Fallback: if cross-encoder not available, return neutral scores
            return [1.0] * len(passages)
        
        texts = [p.get('text', '') for p in passages]
        # Cross-encoder expects [query, passage] pairs
        pairs = [[query, t] for t in texts]
        scores = self.cross_encoder.predict(pairs)
        return scores
    
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
    
    def is_question(self, text):
        """Check if passage text is itself a question."""
        text = text.strip()
        return text.endswith('?')
    
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
                                   is_medically_reviewed=False, has_author=False):
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
        # Handle cross-encoder score: clip negative values to 0
        # Rationale: Negative CE scores indicate irrelevance; converting them to 0.25
        # (via linear normalization) would falsely boost irrelevant passages.
        # Clipping preserves the distinction: negatives stay low, positives map 0-1.
        ce_normalized = max(0.0, min(1.0, cross_encoder_score))
        
        # Combine the three signals using weighted average
        if sapbert_score is not None:
            # All three signals available: use base weights
            # 25% FAISS + 70% CE + 5% SapBERT = 100%
            combined = 0.25 * faiss_score + 0.70 * ce_normalized + 0.05 * sapbert_score
        else:
            # SapBERT missing: redistribute weights proportionally
            # Instead of dropping the weight, scale up remaining signals
            # 26% FAISS + 74% CE = 100% (maintains signal importance ratio)
            combined = 0.26 * faiss_score + 0.74 * ce_normalized
        
        # Apply credibility adjustments based on source metadata
        if is_medically_reviewed:
            # Boost passages from credible medical sources
            # 1.08x multiplier = +8% score increase
            combined *= 1.08
        if not has_author:
            # Penalize anonymous sources (harder to verify credibility)
            # 0.92x multiplier = -8% score decrease
            combined *= 0.92
        
        # Final clamp to [0, 1] range after all transformations
        return max(0.0, min(1.0, combined))
    
    def display_passage(self, idx, passage, query, faiss_score, cross_score, sapbert_score, combined_score, auto_label):
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
        text = passage.get('text', '')
        # Show first 500 chars of passage text for context
        text_snippet = text if len(text) <= 500 else f"{text[:500]}..."

        # Show auto-label status
        auto_label_display = ""
        if auto_label == 'relevant':
            auto_label_display = " [AUTO: RELEVANT ✓]"
        elif auto_label == 'unrelated':
            auto_label_display = " [AUTO: UNRELATED ✗]"
        elif auto_label == 'question':
            auto_label_display = " [AUTO: QUESTION ❓]"
        elif auto_label == 'review':
            auto_label_display = " [NEEDS REVIEW]"
        
        # Display formatted passage information
        print(f"\n{'='*70}")
        print(f"[{idx}] {title}{auto_label_display}")
        print(f"Section: {section} | Author: {'yes' if has_author else 'no'} | Reviewed: {'yes' if has_review else 'no'}")
        # Show all four scores for complete transparency
        print(f"Scores → FAISS: {faiss_score:.4f} | Cross: {cross_score:.4f} | SapBERT: {sapbert_score:.4f} | Combined: {combined_score:.4f}")
        print(f"Text: {text_snippet}")
        print(f"{'='*70}")
    
    def get_relevance_decision(self, auto_label=None):
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
    
    def save_labeled_passage(self, passage, query, decision, rerank_score, faiss_score):
        """Save labeled passage to appropriate file based on decision.
        
        Creates a labeled item with complete metadata and scores, then
        appends to the appropriate JSON file based on the label.
        
        Saves immediately to disk to protect against data loss from crashes.
        """
        labeled_item = {
            'passage_id': passage.get('passage_id'),
            'query': query,
            'label': decision,  # 'relevant', 'unrelated', or 'question'
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
        if decision == 'relevant':
            self.relevant_passages.append(labeled_item)
            with RELEVANT_PASSAGES_FILE.open('w', encoding='utf-8') as f:
                json.dump(self.relevant_passages, f, indent=2, ensure_ascii=False)
        elif decision == 'unrelated':
            self.unrelated_passages.append(labeled_item)
            with UNRELATED_PASSAGES_FILE.open('w', encoding='utf-8') as f:
                json.dump(self.unrelated_passages, f, indent=2, ensure_ascii=False)
        elif decision == 'question':
            self.question_passages.append(labeled_item)
            with QUESTION_PASSAGES_FILE.open('w', encoding='utf-8') as f:
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
        
        # Filter valid results: FAISS padding uses -1 for invalid indices
        candidates = []
        for score, idx in zip(faiss_scores, indices):
            if idx >= 0 and idx < len(self.metadata):
                candidates.append((idx, score))
        
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
            # Calculate combined score with credibility signals
            is_reviewed = bool(passage.get('medically_reviewed_by'))
            has_author = bool(passage.get('author'))
            combined_score = self.calculate_combined_score(
                faiss_score, rerank_score, sapbert_score,
                is_medically_reviewed=is_reviewed,
                has_author=has_author
            )
            all_scores.append(combined_score)
            
            # Determine auto-label using three criteria
            auto_label = None
            # Check 1: Is this passage a question?
            if self.is_single_sentence_question(passage.get('text', '')):
                auto_label = 'question'
                auto_question_count += 1
            # Check 2: High confidence relevant (score >= 0.75)
            elif combined_score >= AUTO_RELEVANT_THRESHOLD:
                auto_label = 'relevant'
                auto_relevant_count += 1
            # Check 3: High confidence unrelated (score <= 0.35)
            elif combined_score <= AUTO_UNRELATED_THRESHOLD:
                auto_label = 'unrelated'
                auto_unrelated_count += 1
            # Otherwise: Uncertain - needs human review
            else:
                auto_label = 'review'
                human_review_count += 1
            
            # Display passage with auto-label status and all scores
            self.display_passage(i, passage, query, faiss_score, rerank_score, sapbert_score, combined_score, auto_label)
            
            # Get labeling decision (auto-labeled or human review)
            # Skips prompt for auto-labeled passages, only prompts for 'review' label
            decision = self.get_relevance_decision(auto_label=auto_label)
            
            # Save label immediately to disk for crash safety
            # This ensures no work is lost if the program crashes mid-session
            labeled = self.save_labeled_passage(passage, query, decision, rerank_score, faiss_score)
            session_data['labeled_passages'].append(labeled)
            
            # Show user confirmation of label assignment
            label_symbols = {'relevant': 'RELEVANT ✓', 'unrelated': 'UNRELATED ✗', 'question': 'QUESTION ❓'}
            print(f"→ Labeled as: {label_symbols[decision]}")
        
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


def main():
    """Main menu loop for the VeriFact admin labeling system.
    
    Provides interactive CLI with these options:
    1. Label passages: Start an interactive session for a query
    2. View statistics: Show count of labeled passages by category
    3. Export data: (Future feature)
    4. Exit: Close the application
    """
    labeler = PassageLabeler()
    
    print("\n" + "="*70)
    print("VERIFACT - Medical Fact Verification Admin Tool")
    print("="*70)
    
    # Initialize models and indices before any labeling
    if not labeler.initialize_models():
        print("Failed to initialize. Exiting.")
        return
    
    # Main menu loop
    while True:
        print("\n" + "="*70)
        print("Main Menu")
        print("="*70)
        print("1. Label passages (interactive mode)")
        print("2. View labeling statistics")
        print("3. Export labeled data")
        print("4. Exit")
        print("="*70)
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            # Start interactive labeling session
            print("\n" + "-"*70)
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
            # Show labeling statistics
            print("\n" + "-"*70)
            print("Labeling Statistics")
            print("-"*70)
            print(f"Relevant passages labeled: {len(labeler.relevant_passages)}")
            print(f"Unrelated passages labeled: {len(labeler.unrelated_passages)}")
            print(f"Question passages labeled: {len(labeler.question_passages)}")
            total = len(labeler.relevant_passages) + len(labeler.unrelated_passages) + len(labeler.question_passages)
            print(f"Total labeled: {total}")
            
            # Show unique queries per category (for diversity analysis)
            if labeler.relevant_passages:
                print(f"\nUnique queries in relevant: {len(set(p['query'] for p in labeler.relevant_passages))}")
            if labeler.unrelated_passages:
                print(f"Unique queries in unrelated: {len(set(p['query'] for p in labeler.unrelated_passages))}")
            if labeler.question_passages:
                print(f"Unique queries in questions: {len(set(p['query'] for p in labeler.question_passages))}")
        
        elif choice == '3':
            print("\nExport functionality coming soon...")
        
        elif choice == '4':
            print("\nThank you for using VeriFact Admin Tool. Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-4.")


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
