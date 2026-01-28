"""
Interactive Passage Labeling System (Refactored)
Labels passages from the medical index as RELEVANT or UNRELATED to a query.
Now using modular components for scoring, stance detection, and persistence.
"""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from models import ModelManager
from scoring import PassageScorer, PassageFilter, AutoLabeler
from stance_detector import StanceDetector, StanceAutoLabeler
from persistence import LabeledDataStore, ClaimsLoader, ProcessedClaimTracker

from config import (
    INDEX_DIR, 
    DEFAULT_TOPK_FAISS, DEFAULT_DISPLAY_RESULTS,
    AUTO_RELEVANT_THRESHOLD, AUTO_UNRELATED_THRESHOLD, STANCE_AUTO_THRESHOLD,
    AUTO_LABELING_MODE,
    AUTO_UNRELATED_CE_MAX, AUTO_UNRELATED_LEX_MAX,
    RELEVANT_PASSAGES_FILE, UNRELATED_PASSAGES_FILE, QUESTION_PASSAGES_FILE,
    LABELING_SESSION_FILE, CONCEPT_LABELS_VECTORS, CONCEPT_LABELS_METADATA, CONCEPT_TOPK,
    OUTPUT_DIR,
    VERIFIED_CLAIMS_FILE, FAKE_CLAIMS_FILE, FAKE_CLAIMS_LABELED_FILE
)


class PassageLabeler:
    """Interactive labeling system for medical passages using modular components."""
    
    def __init__(self):
        """Initialize labeler with modular components."""
        self.index_dir = Path(INDEX_DIR)
        
        # Initialize modular components
        self.model_manager = None
        self.passage_scorer = PassageScorer()
        self.passage_filter = PassageFilter()
        self.auto_labeler = AutoLabeler(
            relevant_threshold=AUTO_RELEVANT_THRESHOLD,
            unrelated_threshold=AUTO_UNRELATED_THRESHOLD,
            ce_max_unrelated=AUTO_UNRELATED_CE_MAX,
            lex_max_unrelated=AUTO_UNRELATED_LEX_MAX
        )
        self.stance_detector = None
        self.stance_auto_labeler = StanceAutoLabeler(auto_threshold=STANCE_AUTO_THRESHOLD)
        
        # Data store for persistence
        self.data_store = LabeledDataStore(
            relevant_file=RELEVANT_PASSAGES_FILE,
            unrelated_file=UNRELATED_PASSAGES_FILE,
            question_file=QUESTION_PASSAGES_FILE,
            session_file=LABELING_SESSION_FILE,
            output_dir=OUTPUT_DIR
        )
        
        # Load existing labeled data
        self.relevant_passages = self.data_store.relevant_passages
        self.unrelated_passages = self.data_store.unrelated_passages
        self.question_passages = self.data_store.question_passages
        self.fake_claims_passages = self._load_fake_claims_labeled()
        
        self.session_start = datetime.now()
        self.models_initialized = False
        self.last_query_concepts = []
        self.auto_labeling_mode = AUTO_LABELING_MODE

    def _load_fake_claims_labeled(self):
        """Load labeled fake claims from file."""
        if FAKE_CLAIMS_LABELED_FILE.exists():
            try:
                with FAKE_CLAIMS_LABELED_FILE.open('r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load fake claims: {e}")
        return []

    def initialize_models(self, skip_stance=False):
        """Initialize all models using ModelManager.
        
        Args:
            skip_stance: If True, skip loading stance detection models
        """
        config = {
            'EMBEDDING_MODEL': 'pritamdeka/S-PubMedBert-MS-MARCO',
            'CROSS_ENCODER_MODEL': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
            'NLI_CROSS_ENCODER_MODEL': 'cross-encoder/nli-deberta-v3-large',
            'SAPBERT_MODEL': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
            'STANCE_MODEL': 'roberta-large-mnli',
            'CONCEPT_LABELS_VECTORS': str(CONCEPT_LABELS_VECTORS),
            'CONCEPT_LABELS_METADATA': str(CONCEPT_LABELS_METADATA)
        }
        
        self.model_manager = ModelManager(self.index_dir, config)
        success = self.model_manager.initialize_all(skip_stance=skip_stance)
        
        if success:
            self.stance_detector = StanceDetector(
                nli_cross_encoder=self.model_manager.nli_cross_encoder
            )
            self.models_initialized = True
        
        return success

    def ensure_models_initialized(self, skip_stance=False):
        """Initialize models on demand."""
        if not self.models_initialized:
            return self.initialize_models(skip_stance=skip_stance)
        return True

    def search(self, query, topk=DEFAULT_TOPK_FAISS):
        """Search for passages using FAISS via ModelManager."""
        if self.model_manager is None:
            raise RuntimeError("Models not initialized")
        
        # Build query vector with concept expansion
        self.last_query_concepts = []
        expanded_query = query
        
        concepts = self.get_top_concepts(query, topk=CONCEPT_TOPK)
        if concepts:
            self.last_query_concepts = concepts
            expanded_query = f"{query}; " + "; ".join(concepts)
        
        # Embed and search
        query_vector = self.model_manager.embed_text(expanded_query, normalize=True)
        query_vector = np.expand_dims(query_vector.astype(np.float32), axis=0)
        scores, indices = self.model_manager.index.search(query_vector, topk)
        return scores[0], indices[0]

    def rerank_results(self, query, passages):
        """Rerank passages using cross-encoder via ModelManager."""
        if self.model_manager is None or self.model_manager.cross_encoder is None:
            return [0.0] * len(passages)
        
        texts = [p.get('text', '') for p in passages]
        pairs = [[query, t] for t in texts]
        scores = self.model_manager.cross_encoder.predict(pairs)
        
        # Normalize to [0, 1] using sigmoid
        try:
            arr = np.array(scores, dtype=np.float32)
            normalized = 1.0 / (1.0 + np.exp(-arr))
            return normalized.tolist()
        except Exception:
            return [0.0] * len(passages)

    def compute_sapbert_similarities(self, query, passage_indices):
        """Compute SapBERT medical entity similarities."""
        if self.model_manager is None or self.model_manager.sapbert_embeddings is None:
            return [None] * len(passage_indices)
        
        query_emb = self.model_manager.embed_with_sapbert(query, normalize=True)
        if query_emb is None:
            return [None] * len(passage_indices)
        
        scores = []
        for idx in passage_indices:
            if idx >= 0 and idx < len(self.model_manager.sapbert_embeddings):
                passage_emb = self.model_manager.sapbert_embeddings[idx]
                passage_emb = passage_emb / (np.linalg.norm(passage_emb) + 1e-8)
                similarity = np.dot(query_emb, passage_emb)
                scores.append(max(0.0, similarity))
            else:
                scores.append(None)
        
        return scores

    def get_top_concepts(self, query, topk=CONCEPT_TOPK):
        """Retrieve top concept labels related to query."""
        if (self.model_manager is None or 
            self.model_manager.sapbert_model is None or 
            self.model_manager.concept_label_vectors is None):
            return []
        
        try:
            q_emb = self.model_manager.embed_with_sapbert(query, normalize=True)
            if q_emb is None:
                return []
            
            sims = self.model_manager.concept_label_vectors.dot(q_emb)
            top_idxs = np.argsort(-sims)[: max(topk * 5, topk)]
            
            labels = []
            seen = set()
            for i in top_idxs:
                if i < len(self.model_manager.concept_label_meta):
                    meta = self.model_manager.concept_label_meta[i]
                    label = (
                        meta.get("canonical_label") or meta.get("canonical") or
                        meta.get("label") or meta.get("preferred_label") or
                        meta.get("name") or meta.get("original_text") or
                        meta.get("mesh_id")
                    )
                    if label and label not in seen:
                        labels.append(label)
                        seen.add(label)
                    if len(labels) >= topk:
                        break
            return labels
        except Exception:
            return []

    def display_passage(self, idx, passage, query, faiss_score, cross_score, 
                       sapbert_score, combined_score, auto_label, 
                       lexical_score=None, auto_label_reason=None):
        """Display passage with scores and auto-label status."""
        has_author = bool(passage.get('author'))
        has_review = bool(passage.get('medically_reviewed_by'))
        title = passage.get('title', 'Untitled')
        section = passage.get('section_heading', 'N/A')
        text = (passage.get('text', '') or '').strip()
        text_snippet = text if len(text) <= 800 else f"{text[:800]}..."
        if not text_snippet:
            text_snippet = "[NO TEXT AVAILABLE]"

        # Auto-label display
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
        
        print(f"\n{'='*70}")
        print(f"[{idx}] {title}{auto_label_display}")
        print(f"Section: {section} | Author: {'yes' if has_author else 'no'} | Reviewed: {'yes' if has_review else 'no'}")
        if lexical_score is None:
            print(f"Scores → FAISS: {faiss_score:.4f} | Cross: {cross_score:.4f} | SapBERT: {sapbert_score:.4f} | Combined: {combined_score:.4f}")
        else:
            print(f"Scores → FAISS: {faiss_score:.4f} | Cross: {cross_score:.4f} | SapBERT: {sapbert_score:.4f} | Lexical: {lexical_score:.4f} | Combined: {combined_score:.4f}")
        print("Text:\n" + text_snippet)
        print(f"{'='*70}")

    def get_relevance_decision(self, auto_label=None, fallback_label=None):
        """Get user input for labeling decision."""
        if auto_label in ['relevant', 'unrelated', 'question']:
            return auto_label
        
        if self.auto_labeling_mode:
            return fallback_label or 'relevant'

        while True:
            decision = input("\nLabel: [Enter=relevant] n=unrelated q=question: ").strip().lower()
            if decision == '' or decision in ['y', 'yes']:
                return 'relevant'
            elif decision in ['n', 'no']:
                return 'unrelated'
            elif decision in ['q', 'question']:
                return 'question'
            else:
                print("Invalid input. Press Enter for relevant, 'n' for unrelated, or 'q' for question.")

    def get_stance_decision(self, suggest=None, suggest_confidence=None):
        """Get user input for stance decision."""
        if self.auto_labeling_mode:
            return suggest or 'neutral', suggest_confidence or 0.5

        while True:
            prompt = "\nStance: [Enter=use suggestion] s=supports r=refutes n=neutral: "
            if suggest is not None:
                prompt = f"\nStance (suggested: {suggest}): [Enter=use suggestion] s=supports r=refutes n=neutral: "
            
            decision = input(prompt).strip().lower()
            if decision == '':
                return suggest or 'neutral', suggest_confidence or 0.5
            elif decision in ['s', 'supports']:
                return 'supports', 0.7
            elif decision in ['r', 'refutes']:
                return 'refutes', 0.7
            elif decision in ['n', 'neutral']:
                return 'neutral', 0.5
            else:
                print("Invalid input.")

    def predict_stance(self, query, passage_text):
        """Predict stance using heuristic detection."""
        if self.stance_detector is None:
            return None, None
        return self.stance_detector.detect_stance(query, passage_text)

    def predict_stance_nli(self, query, passage_text):
        """Predict stance using NLI cross-encoder."""
        if self.stance_detector is None:
            return None, None
        return self.stance_detector.detect_stance_nli(query, passage_text)

    def save_labeled_passage(self, passage, query, decision, rerank_score, faiss_score, 
                            stance=None, stance_confidence=None, from_fake_claim=False):
        """Save labeled passage using LabeledDataStore."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        labeled_item = {
            'passage_id': passage.get('passage_id'),
            'query': query,
            'label': decision,
            'stance': stance if decision == 'relevant' else None,
            'stance_confidence': float(stance_confidence) if (
                stance_confidence is not None and decision == 'relevant'
            ) else None,
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
            'labeled_at': datetime.now().isoformat(),
            'from_fake_claim': from_fake_claim
        }
        
        # If from fake claims workflow, ONLY save to dedicated fake claims file
        if from_fake_claim and decision == 'relevant' and stance == 'refutes':
            self.fake_claims_passages.append(labeled_item)
            FAKE_CLAIMS_LABELED_FILE.parent.mkdir(parents=True, exist_ok=True)
            with FAKE_CLAIMS_LABELED_FILE.open('w', encoding='utf-8') as f:
                json.dump(self.fake_claims_passages, f, indent=2, ensure_ascii=False)
        else:
            # Normal workflow: use LabeledDataStore
            self.data_store.save_labeled_passage(
                passage, query, decision, faiss_score, rerank_score, stance, stance_confidence
            )
            # Update in-memory cache
            if decision == 'relevant':
                self.relevant_passages = self.data_store.relevant_passages
            elif decision == 'unrelated':
                self.unrelated_passages = self.data_store.unrelated_passages
            elif decision == 'question':
                self.question_passages = self.data_store.question_passages
        
        return labeled_item

    def label_session(self, query, num_results=DEFAULT_DISPLAY_RESULTS, topk=DEFAULT_TOPK_FAISS):
        """Interactive labeling session for a query."""
        print(f"\nSearching for passages related to: '{query}'")
        print("Please wait...\n")
        
        faiss_scores, indices = self.search(query, topk=topk)
        if self.last_query_concepts:
            print(f"Concept expansion: {', '.join(self.last_query_concepts)}")
        
        # Filter valid results
        candidates = []
        for score, idx in zip(faiss_scores, indices):
            if idx >= 0 and idx < len(self.model_manager.metadata):
                capped_score = min(0.95, float(score))
                candidates.append((idx, capped_score))
        
        top_candidates = candidates[:num_results]
        passages = [self.model_manager.metadata[idx] for idx, _ in top_candidates]
        faiss_scores_top = [score for _, score in top_candidates]
        
        print(f"Found {len(candidates)} passages, displaying top {len(top_candidates)}\n")
        
        # Get scores
        rerank_scores = self.rerank_results(query, passages)
        passage_indices = [idx for idx, _ in top_candidates]
        sapbert_scores = self.compute_sapbert_similarities(query, passage_indices)
        
        session_data = {
            'query': query,
            'session_start': self.session_start.isoformat(),
            'labeled_passages': []
        }
        
        all_scores = []
        auto_relevant_count = 0
        auto_unrelated_count = 0
        auto_question_count = 0
        human_review_count = 0
        
        for i, (passage, faiss_score, rerank_score, sapbert_score) in enumerate(
            zip(passages, faiss_scores_top, rerank_scores, sapbert_scores), 1
        ):
            # Check substantiveness
            if not self.passage_filter.is_substantive(passage.get('text', '')):
                print(f"[{i}] {passage.get('title', 'Untitled')} [SKIPPED: Insufficient text content]")
                continue
            
            lexical_score = self.passage_scorer.compute_lexical_overlap(query, passage.get('text', ''))
            
            is_reviewed = bool(passage.get('medically_reviewed_by'))
            has_author = bool(passage.get('author'))
            combined_score = self.passage_scorer.calculate_combined_score(
                faiss_score, rerank_score, sapbert_score, lexical_score,
                is_medically_reviewed=is_reviewed,
                has_author=has_author
            )
            all_scores.append(combined_score)
            
            # Determine auto-label
            is_question = self.passage_filter.is_question(passage.get('text', ''))
            auto_label, auto_label_reason = self.auto_labeler.determine_label(
                combined_score, rerank_score, lexical_score, is_question
            )
            
            if auto_label == 'relevant':
                auto_relevant_count += 1
            elif auto_label == 'unrelated':
                auto_unrelated_count += 1
            elif auto_label == 'question':
                auto_question_count += 1
            else:
                human_review_count += 1
            
            self.display_passage(i, passage, query, faiss_score, rerank_score, 
                               sapbert_score, combined_score, auto_label,
                               lexical_score, auto_label_reason)
            
            mid = (AUTO_RELEVANT_THRESHOLD + AUTO_UNRELATED_THRESHOLD) / 2.0
            fallback_label = 'relevant' if combined_score >= mid else 'unrelated'
            decision = self.get_relevance_decision(auto_label=auto_label, fallback_label=fallback_label)
            
            stance = None
            stance_confidence = None
            
            if decision == 'relevant':
                stance_suggest, stance_conf = self.predict_stance_nli(query, passage.get('text', ''))
                if stance_suggest is None:
                    stance_suggest, stance_conf = self.predict_stance(query, passage.get('text', ''))
                
                if self.auto_labeling_mode:
                    if stance_suggest is None:
                        stance = 'neutral'
                        stance_confidence = 0.50
                    else:
                        stance = stance_suggest
                        stance_confidence = stance_conf or 0.50
                else:
                    if (stance_suggest and stance_suggest != 'neutral' and 
                        stance_conf is not None and stance_conf >= STANCE_AUTO_THRESHOLD):
                        stance = stance_suggest
                    else:
                        stance, stance_confidence = self.get_stance_decision(stance_suggest, stance_conf)
            
            self.save_labeled_passage(passage, query, decision, rerank_score, faiss_score, 
                                     stance, stance_confidence, from_fake_claim=False)
            session_data['labeled_passages'].append({
                'passage_id': passage.get('passage_id'),
                'decision': decision,
                'stance': stance,
                'auto_label': auto_label
            })
        
        # Display session statistics
        print(f"\n{'='*70}")
        print("Session Statistics")
        print(f"{'='*70}")
        print(f"Auto-labeled as RELEVANT: {auto_relevant_count}")
        print(f"Auto-labeled as UNRELATED: {auto_unrelated_count}")
        print(f"Auto-labeled as QUESTION: {auto_question_count}")
        print(f"Required human review: {human_review_count}")
        
        if all_scores:
            print(f"\nScore Distribution:")
            print(f"  Min: {min(all_scores):.4f}, Max: {max(all_scores):.4f}, Avg: {np.mean(all_scores):.4f}")
        
        self.data_store.save_session(session_data)

    def _load_verified_claims(self):
        """Load verified claims from file."""
        return ClaimsLoader.load_claims(VERIFIED_CLAIMS_FILE)

    def _load_fake_claims(self):
        """Load fake claims from file."""
        return ClaimsLoader.load_claims(FAKE_CLAIMS_FILE)

    def _load_processed_claim_ids(self):
        """Load set of processed claim IDs."""
        tracker = ProcessedClaimTracker(OUTPUT_DIR)
        return tracker.processed_ids

    def _save_processed_claim_ids(self, ids_set):
        """Save set of processed claim IDs."""
        tracker = ProcessedClaimTracker(OUTPUT_DIR)
        tracker.processed_ids = ids_set
        tracker.save()

    def auto_label_from_claims(self, num_results=DEFAULT_DISPLAY_RESULTS):
        """Auto-label passages for verified claims."""
        self.ensure_models_initialized(skip_stance=False)
        
        claims = self._load_verified_claims()
        if not claims:
            print("No verified claims found.")
            return
        
        processed_ids = self._load_processed_claim_ids()
        
        for claim in claims:
            claim_id = claim.get('id')
            if claim_id and ProcessedClaimTracker(OUTPUT_DIR).is_processed(claim_id):
                continue
            
            query = claim.get('claim')
            if not query:
                continue
            
            print(f"\nProcessing claim: {query}")
            self.label_session(query, num_results=num_results)
            
            processed_ids.add(claim_id)
        
        self._save_processed_claim_ids(processed_ids)

    def auto_label_refutes_from_fake_claims(self, num_results=DEFAULT_DISPLAY_RESULTS):
        """Auto-label passages that refute fake claims."""
        self.ensure_models_initialized(skip_stance=True)
        
        claims = self._load_fake_claims()
        if not claims:
            print("No fake claims found.")
            return
        
        for claim in claims:
            query = claim.get('claim')
            if not query:
                continue
            
            print(f"\nProcessing fake claim: {query}")
            self._label_session_for_fake_claim(query, num_results=num_results)

    def _label_session_for_fake_claim(self, query, num_results=DEFAULT_DISPLAY_RESULTS, topk=DEFAULT_TOPK_FAISS):
        """Label session specifically for fake claims (auto-sets stance to refutes)."""
        print(f"Searching for passages that refute: '{query}'")
        print("Please wait...\n")
        
        faiss_scores, indices = self.search(query, topk=topk)
        
        candidates = []
        for score, idx in zip(faiss_scores, indices):
            if idx >= 0 and idx < len(self.model_manager.metadata):
                capped_score = min(0.95, float(score))
                candidates.append((idx, capped_score))
        
        top_candidates = candidates[:num_results]
        passages = [self.model_manager.metadata[idx] for idx, _ in top_candidates]
        faiss_scores_top = [score for _, score in top_candidates]
        
        print(f"Found {len(candidates)} passages, displaying top {len(top_candidates)}\n")
        
        rerank_scores = self.rerank_results(query, passages)
        passage_indices = [idx for idx, _ in top_candidates]
        sapbert_scores = self.compute_sapbert_similarities(query, passage_indices)
        
        for i, (passage, faiss_score, rerank_score, sapbert_score) in enumerate(
            zip(passages, faiss_scores_top, rerank_scores, sapbert_scores), 1
        ):
            if not self.passage_filter.is_substantive(passage.get('text', '')):
                print(f"[{i}] {passage.get('title', 'Untitled')} [SKIPPED: Insufficient text content]")
                continue
            
            lexical_score = self.passage_scorer.compute_lexical_overlap(query, passage.get('text', ''))
            
            is_reviewed = bool(passage.get('medically_reviewed_by'))
            has_author = bool(passage.get('author'))
            combined_score = self.passage_scorer.calculate_combined_score(
                faiss_score, rerank_score, sapbert_score, lexical_score,
                is_medically_reviewed=is_reviewed,
                has_author=has_author
            )
            
            is_question = self.passage_filter.is_question(passage.get('text', ''))
            auto_label, auto_label_reason = self.auto_labeler.determine_label(
                combined_score, rerank_score, lexical_score, is_question
            )
            
            self.display_passage(i, passage, query, faiss_score, rerank_score, 
                               sapbert_score, combined_score, auto_label,
                               lexical_score, auto_label_reason)
            
            decision = self.get_relevance_decision(auto_label=auto_label, fallback_label='relevant')
            
            # For fake claims, hardcode stance to 'refutes' with high confidence
            stance = 'refutes'
            stance_confidence = 0.95
            
            self.save_labeled_passage(passage, query, decision, rerank_score, faiss_score,
                                     stance, stance_confidence, from_fake_claim=True)

    def _collect_labels(self, subset="all"):
        """Collect labeled passages by subset."""
        return self.data_store.get_all_labeled(subset=subset)

    def export_labeled_data(self, export_format="csv", out_path=None, subset="all"):
        """Export labeled data in various formats."""
        # Placeholder - implement based on requirements
        pass

    def view_statistics(self):
        """Display labeling statistics."""
        stats = self.data_store.get_statistics()
        
        print(f"\n{'='*70}")
        print("Labeling Statistics")
        print(f"{'='*70}")
        print(f"Total labeled passages: {stats['total_count']}")
        print(f"  - Relevant: {stats['relevant_count']}")
        print(f"  - Unrelated: {stats['unrelated_count']}")
        print(f"  - Questions: {stats['question_count']}")
        print(f"  - Fake claims (refutes): {len(self.fake_claims_passages)}")
        
        print(f"\nStance breakdown (relevant passages):")
        stance = stats['stance_breakdown']
        print(f"  - Supports: {stance['supports']}")
        print(f"  - Refutes: {stance['refutes']}")
        print(f"  - Neutral: {stance['neutral']}")
        print(f"{'='*70}\n")


def main():
    """Main menu loop for the VeriFact admin labeling system."""
    labeler = PassageLabeler()
    
    print("\n" + "="*70)
    print("VERIFACT - Medical Fact Verification Admin Tool")
    print("="*70)
    
    while True:
        print("\n" + "="*70)
        print("Main Menu")
        print("="*70)
        print("1. Label passages from a query")
        print("2. Auto-label from verified claims")
        print("3. Auto-label REFUTES from fake claims")
        print("4. View statistics")
        print("5. Exit")
        print("="*70)
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            query = input("\nEnter query/claim: ").strip()
            if query:
                labeler.ensure_models_initialized()
                labeler.label_session(query)
        elif choice == '2':
            labeler.ensure_models_initialized(skip_stance=False)
            labeler.auto_label_from_claims()
        elif choice == '3':
            labeler.ensure_models_initialized(skip_stance=True)
            labeler.auto_label_refutes_from_fake_claims()
        elif choice == '4':
            labeler.view_statistics()
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
