"""
Interactive Passage Labelling System
Labels passages from the medical index as RELEVANT or UNRELATED to a query.
Uses the retrieval system for scoring and ranking.
"""

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from retrieval.simple_retriever import SimpleRetriever, MinimalModelManager
from stance_detector import StanceDetector
from claim_extraction import SentenceSplitter, ClaimSentenceFilter, ClaimExtractor, RefuteGenerator
from persistence import ClaimsLoader, ProcessedClaimTracker

from config import (
    INDEX_DIR,
    DEFAULT_DISPLAY_RESULTS,
    AUTO_RELEVANT_THRESHOLD, AUTO_UNRELATED_THRESHOLD, STANCE_AUTO_THRESHOLD,
    
    AUTO_UNRELATED_CE_MAX, AUTO_UNRELATED_LEX_MAX,
    
    OUTPUT_DIR,
    VERIFIED_CLAIMS_FILE,
    CLAIMS_OUTPUT_FILE
)


class AutoLabeler:
    """Determines auto-labeling decisions based on score thresholds."""

    def __init__(
        self,
        relevant_threshold: float = 0.66,
        unrelated_threshold: float = 0.52,
        ce_max_unrelated: float = 0.10,
        lex_max_unrelated: float = 0.40
    ):
        self.relevant_threshold = relevant_threshold
        self.unrelated_threshold = unrelated_threshold
        self.ce_max_unrelated = ce_max_unrelated
        self.lex_max_unrelated = lex_max_unrelated

    def determine_label(
        self,
        combined_score: float,
        rerank_score: float,
        lexical_score: float
    ) -> Tuple[str, Optional[str]]:
        if rerank_score <= self.ce_max_unrelated and lexical_score <= self.lex_max_unrelated:
            return 'unrelated', 'low_ce_lex'

        if combined_score >= self.relevant_threshold:
            return 'relevant', 'score_high'

        if combined_score <= self.unrelated_threshold:
            return 'unrelated', 'score_low'

        return 'review', None


class PassageLabeler:
    """Interactive labeling system for medical passages using modular components."""
    
    def __init__(self):
        """Initialize labeler with modular components."""
        self.index_dir = Path(INDEX_DIR)
        
        # Initialize modular components
        self.model_manager = None
        self.retriever = None
        self.auto_labeler = AutoLabeler(
            relevant_threshold=AUTO_RELEVANT_THRESHOLD,
            unrelated_threshold=AUTO_UNRELATED_THRESHOLD,
            ce_max_unrelated=AUTO_UNRELATED_CE_MAX,
            lex_max_unrelated=AUTO_UNRELATED_LEX_MAX
        )
        self.stance_detector = None

        self.sentence_splitter = SentenceSplitter()
        self.claim_filter = ClaimSentenceFilter()
        self.claim_extractor = ClaimExtractor()
        self.refute_generator = RefuteGenerator()
        
        self.session_start = datetime.now()
        self.models_initialized = False

    def initialize_models(self, skip_stance=False):
        """Initialize retrieval system models."""
        try:
            self.model_manager = MinimalModelManager(str(self.index_dir))
            self.retriever = SimpleRetriever(self.model_manager, str(self.index_dir))
            self.stance_detector = StanceDetector()
            self.models_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize retrieval system: {e}")
            return False

    def ensure_models_initialized(self, skip_stance=False):
        """Initialize models on demand."""
        if not self.models_initialized:
            return self.initialize_models(skip_stance=skip_stance)
        return True

    def predict_stance(self, query, passage_text):
        """Predict stance using heuristic detection."""
        if self.stance_detector is None:
            return None, None
        return self.stance_detector.detect_stance(query, passage_text)

    def _load_existing_claims(self):
        if not CLAIMS_OUTPUT_FILE.exists():
            return []
        try:
            with CLAIMS_OUTPUT_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_claims_output(self, items):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        existing = self._load_existing_claims()
        existing.extend(items)
        with CLAIMS_OUTPUT_FILE.open('w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def _load_verified_claims(self):
        """Load verified claims from file."""
        return ClaimsLoader.load_claims(VERIFIED_CLAIMS_FILE)

    def _load_processed_claim_ids(self):
        """Load set of processed claim IDs."""
        tracker = ProcessedClaimTracker(OUTPUT_DIR)
        return tracker.processed_ids

    def _save_processed_claim_ids(self, ids_set):
        """Save set of processed claim IDs."""
        tracker = ProcessedClaimTracker(OUTPUT_DIR)
        tracker.processed_ids = ids_set
        tracker.save()

    def _claim_id(self, passage_id, claim_text, label):
        raw = f"{passage_id}|{label}|{claim_text}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def generate_claims_from_verified_claims(self, num_results=DEFAULT_DISPLAY_RESULTS):
        """Generate atomic claims from retrieved passages for verified queries."""
        self.ensure_models_initialized(skip_stance=False)

        claims = self._load_verified_claims()
        if not claims:
            print("No verified claims found.")
            return

        processed_ids = self._load_processed_claim_ids()
        generated = []
        seen = set()

        total_sentences = 0
        kept_sentences = 0
        supports_count = 0
        refutes_count = 0

        for claim in claims:
            claim_id = claim.get('id')
            if claim_id and ProcessedClaimTracker(OUTPUT_DIR).is_processed(claim_id):
                continue

            query = claim.get('claim')
            if not query:
                continue

            print(f"\nProcessing query: {query}")
            if self.retriever and num_results:
                self.retriever.FINAL_TOPK = int(num_results)

            results = self.retriever.search(query)

            for result in results:
                passage = result.get('passage', {})
                passage_text = passage.get('text', '') or ''
                if not passage_text.strip():
                    continue

                scores = result.get('scores', {})
                final_score = float(result.get('final_score', 0.0))
                cross_score = float(scores.get('cross_encoder', 0.0))
                lexical_score = float(scores.get('lexical', 0.0))

                decision, _reason = self.auto_labeler.determine_label(
                    final_score,
                    cross_score,
                    lexical_score
                )

                if decision != 'relevant':
                    continue

                passage_id = passage.get('passage_id') or passage.get('id') or passage.get('url')
                sentences = self.sentence_splitter.split(passage_text)
                total_sentences += len(sentences)

                for sentence in sentences:
                    if not self.claim_filter.is_claim_worthy(sentence):
                        continue
                    kept_sentences += 1

                    extracted = self.claim_extractor.extract(sentence)
                    for claim_text in extracted:
                        claim_key = (passage_id, claim_text, 'supports')
                        if claim_key in seen:
                            continue
                        seen.add(claim_key)

                        stance, stance_confidence = self.predict_stance(claim_text, passage_text)
                        if stance is None:
                            stance = 'neutral'
                            stance_confidence = 0.50

                        generated.append({
                            'claim_id': self._claim_id(passage_id, claim_text, 'supports'),
                            'query': query,
                            'source_passage_id': passage_id,
                            'source_title': passage.get('title'),
                            'source_url': passage.get('url'),
                            'sentence': sentence,
                            'claim': claim_text,
                            'label': 'supports',
                            'stance': stance,
                            'stance_confidence': float(stance_confidence),
                            'created_at': datetime.now().isoformat()
                        })
                        supports_count += 1

                        for refute in self.refute_generator.generate(claim_text):
                            refute_key = (passage_id, refute, 'refutes')
                            if refute_key in seen:
                                continue
                            seen.add(refute_key)

                            generated.append({
                                'claim_id': self._claim_id(passage_id, refute, 'refutes'),
                                'query': query,
                                'source_passage_id': passage_id,
                                'source_title': passage.get('title'),
                                'source_url': passage.get('url'),
                                'sentence': sentence,
                                'claim': refute,
                                'label': 'refutes',
                                'stance': 'refutes',
                                'stance_confidence': 0.0,
                                'created_at': datetime.now().isoformat()
                            })
                            refutes_count += 1

            processed_ids.add(claim_id)

        if generated:
            self._save_claims_output(generated)

        self._save_processed_claim_ids(processed_ids)
        print("\nClaim generation complete.")
        print(f"Sentences scanned: {total_sentences}")
        print(f"Sentences kept: {kept_sentences}")
        print(f"Claims (supports): {supports_count}")
        print(f"Claims (refutes): {refutes_count}")
        print(f"Output: {CLAIMS_OUTPUT_FILE}")

    def view_statistics(self):
        """Display labeling statistics."""
        items = self._load_existing_claims()
        total = len(items)
        supports = sum(1 for item in items if item.get('label') == 'supports')
        refutes = sum(1 for item in items if item.get('label') == 'refutes')
        
        print(f"\n{'='*70}")
        print("Claim Generation Statistics")
        print(f"{'='*70}")
        print(f"Total claims: {total}")
        print(f"  - Supports: {supports}")
        print(f"  - Refutes: {refutes}")
        print(f"Output file: {CLAIMS_OUTPUT_FILE}")
        print(f"{'='*70}\n")

    def export_claims(self, export_format: str):
        """Export generated claims to JSON, CSV, or TSV."""
        items = self._load_existing_claims()
        if not items:
            print("No claims found to export.")
            return

        export_format = (export_format or "").strip().lower()
        if export_format not in {"json", "csv", "tsv"}:
            print("Unsupported format. Use json, csv, or tsv.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"claims_export_{timestamp}.{export_format}"
        out_path = OUTPUT_DIR / file_name

        if export_format == "json":
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"Exported to {out_path}")
            return

        fieldnames = sorted({
            key for item in items for key in item.keys()
        })
        delimiter = "\t" if export_format == "tsv" else ","

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for item in items:
                writer.writerow(item)

        print(f"Exported to {out_path}")


def main():
    """Main menu loop for the VeriFact labelling system."""
    labeler = PassageLabeler()
    
    print("\n" + "="*70)
    print("VERIFACT - Medical Fact Verification Labelling Tool")
    print("="*70)
    
    while True:
        print("\n" + "="*70)
        print("Main Menu")
        print("="*70)
        print("1. Generate claims from verified queries")
        print("2. View statistics")
        print("3. Export claims (json/csv/tsv)")
        print("4. Exit")
        print("="*70)
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            labeler.ensure_models_initialized(skip_stance=False)
            labeler.generate_claims_from_verified_claims()
        elif choice == '2':
            labeler.view_statistics()
        elif choice == '3':
            export_format = input("Export format (json/csv/tsv): ").strip().lower()
            labeler.export_claims(export_format)
        elif choice == '4':
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
