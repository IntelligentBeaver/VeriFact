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
        """Embed a query using the same model as the index."""
        v = self.embed_model.encode([query], convert_to_numpy=True)[0]
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v
    
    def search(self, query, topk=DEFAULT_TOPK_FAISS):
        """Search for passages related to query."""
        qvec = self.embed_query(query)
        q = np.expand_dims(qvec.astype(np.float32), axis=0)
        scores, indices = self.index.search(q, topk)
        return scores[0], indices[0]
    
    def rerank_results(self, query, passages):
        """Rerank passages using cross-encoder."""
        if self.cross_encoder is None:
            return [1.0] * len(passages)
        
        texts = [p.get('text', '') for p in passages]
        pairs = [[query, t] for t in texts]
        scores = self.cross_encoder.predict(pairs)
        return scores
    
    def is_question(self, text):
        """Check if passage text is itself a question."""
        text = text.strip()
        return text.endswith('?')
    
    def display_passage(self, idx, passage, query, faiss_score, rerank_score):
        """Display a compact view of the passage."""
        has_author = bool(passage.get('author'))
        has_review = bool(passage.get('medically_reviewed_by'))
        is_q = self.is_question(passage.get('text', ''))
        title = passage.get('title', 'Untitled')
        section = passage.get('section_heading', 'N/A')
        text = passage.get('text', '')
        text_snippet = text if len(text) <= 400 else f"{text[:400]}..."

        q_flag = " [QUESTION]" if is_q else ""
        print(f"\n{'='*70}")
        print(f"[{idx}] {title}{q_flag}")
        print(f"Section: {section} | Author: {'yes' if has_author else 'no'} | Reviewed: {'yes' if has_review else 'no'}")
        print(f"Scores → FAISS: {faiss_score:.4f} | Cross-encoder: {rerank_score:.4f}")
        print(f"Text: {text_snippet}")
        print(f"{'='*70}")
    
    def get_relevance_decision(self):
        """Get user input for labeling: Enter=relevant, n=unrelated, q=question."""
        while True:
            decision = input("\nLabel: [Enter=relevant] n=unrelated q=question: ").strip().lower()
            # Default to 'y' (relevant) if user just presses Enter
            if decision == '' or decision in ['y', 'yes']:
                return 'relevant'
            elif decision in ['n', 'no']:
                return 'unrelated'
            elif decision in ['q', 'question']:
                return 'question'
            else:
                print("Invalid input. Press Enter for relevant, 'n' for unrelated, or 'q' for question.")
    
    def save_labeled_passage(self, passage, query, decision, rerank_score, faiss_score):
        """Save labeled passage to appropriate file based on decision."""
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
        """Interactive labeling session for a query."""
        print(f"\nSearching for passages related to: '{query}'")
        print("Please wait...\n")
        
        # Search top-k passages using FAISS
        faiss_scores, indices = self.search(query, topk=topk)
        
        # Collect valid results (filter out FAISS padding with -1)
        candidates = []
        for score, idx in zip(faiss_scores, indices):
            if idx >= 0 and idx < len(self.metadata):
                candidates.append((idx, score))
        
        # Rerank top results and prepare display set
        top_candidates = candidates[:num_results]
        passages = [self.metadata[idx] for idx, _ in top_candidates]
        faiss_scores_top = [score for _, score in top_candidates]
        
        print(f"Found {len(candidates)} passages, displaying top {len(top_candidates)}\n")
        
        rerank_scores = self.rerank_results(query, passages)
        
        # Interactive labeling
        session_data = {
            'query': query,
            'session_start': self.session_start.isoformat(),
            'labeled_passages': []
        }
        
        for i, (passage, faiss_score, rerank_score) in enumerate(zip(passages, faiss_scores_top, rerank_scores), 1):
            self.display_passage(i, passage, query, faiss_score, rerank_score)
            
            # Get labeling decision: relevant, unrelated, or question
            decision = self.get_relevance_decision()
            
            # Save label immediately to disk for crash safety
            labeled = self.save_labeled_passage(passage, query, decision, rerank_score, faiss_score)
            session_data['labeled_passages'].append(labeled)
            
            label_symbols = {'relevant': 'RELEVANT ✓', 'unrelated': 'UNRELATED ✗', 'question': 'QUESTION ❓'}
            print(f"→ Labeled as: {label_symbols[decision]}")
        
        # Save session
        with LABELING_SESSION_FILE.open('w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        # Summary
        print(f"\n{'='*70}")
        print("Session Summary")
        print(f"{'='*70}")
        print(f"Total labeled: {len(session_data['labeled_passages'])}")
        print(f"Relevant: {sum(1 for p in session_data['labeled_passages'] if p['label'] == 'relevant')}")
        print(f"Unrelated: {sum(1 for p in session_data['labeled_passages'] if p['label'] == 'unrelated')}")
        print(f"Questions: {sum(1 for p in session_data['labeled_passages'] if p['label'] == 'question')}")
        print(f"\nOutput files:")
        print(f"  Relevant: {RELEVANT_PASSAGES_FILE}")
        print(f"  Unrelated: {UNRELATED_PASSAGES_FILE}")
        print(f"  Questions: {QUESTION_PASSAGES_FILE}")
        print(f"  Session: {LABELING_SESSION_FILE}")
        print(f"{'='*70}\n")


def main():
    """Main menu loop."""
    labeler = PassageLabeler()
    
    print("\n" + "="*70)
    print("VERIFACT - Medical Fact Verification Admin Tool")
    print("="*70)
    
    # Initialize
    if not labeler.initialize_models():
        print("Failed to initialize. Exiting.")
        return
    
    # Main menu
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
            print("\n" + "-"*70)
            query = input("Enter medical query/claim to investigate: ").strip()
            if not query:
                print("Query cannot be empty.")
                continue
            
            print("\nEnter number of passages to review (default: 10): ", end="")
            try:
                num_results = int(input().strip() or DEFAULT_DISPLAY_RESULTS)
            except ValueError:
                num_results = DEFAULT_DISPLAY_RESULTS
            
            labeler.label_session(query, num_results=num_results)
        
        elif choice == '2':
            print("\n" + "-"*70)
            print("Labeling Statistics")
            print("-"*70)
            print(f"Relevant passages labeled: {len(labeler.relevant_passages)}")
            print(f"Unrelated passages labeled: {len(labeler.unrelated_passages)}")
            print(f"Question passages labeled: {len(labeler.question_passages)}")
            total = len(labeler.relevant_passages) + len(labeler.unrelated_passages) + len(labeler.question_passages)
            print(f"Total labeled: {total}")
            
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
