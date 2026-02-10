"""
Interactive Passage Labelling System
Labels passages from the medical index as RELEVANT or UNRELATED to a query.
Uses the retrieval system for scoring and ranking.
"""

import csv
import hashlib
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers.cross_encoder import CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from retrieval.simple_retriever import SimpleRetriever, MinimalModelManager
from claim_extraction import SentenceSplitter, ClaimSentenceFilter, ClaimExtractor, RefuteGenerator
from persistence import ClaimsLoader, ProcessedClaimTracker
from stance_detector import StanceDetector

from config import (
    INDEX_DIR,
    DEFAULT_DISPLAY_RESULTS,
    AUTO_RELEVANT_THRESHOLD, AUTO_UNRELATED_THRESHOLD,
    
    AUTO_UNRELATED_CE_MAX, AUTO_UNRELATED_LEX_MAX,
    
    OUTPUT_DIR,
    VERIFIED_CLAIMS_FILE,
    UNLABELED_CLAIMS_OUTPUT_FILE,
    UNLABELED_TOPK,
    NLI_CROSS_ENCODER_MODEL,
    CLAIMS_OUTPUT_FILE,
    CLAIMS_EXPORT_STATE_FILE,
    QUERIES_OUTPUT_FILE,
    QUERIES_NEGATED_OUTPUT_FILE
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

        self.sentence_splitter = SentenceSplitter()
        self.claim_filter = ClaimSentenceFilter()
        self.claim_extractor = ClaimExtractor()
        self.refute_generator = RefuteGenerator()
        self._retriever_local = threading.local()
        self._nlp_lock = threading.Lock()
        self._nli_model = None
        self._stance_detector = None
        
        self.session_start = datetime.now()
        self.models_initialized = False

    def initialize_models(self, skip_stance=False):
        """Initialize retrieval system models."""
        try:
            self.model_manager = MinimalModelManager(str(self.index_dir))
            self.retriever = SimpleRetriever(self.model_manager, str(self.index_dir))
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

    def _ensure_stance_detector(self) -> bool:
        if self._stance_detector is not None:
            return True
        try:
            self._nli_model = CrossEncoder(NLI_CROSS_ENCODER_MODEL)
            self._stance_detector = StanceDetector(self._nli_model)
            return True
        except Exception as e:
            print(f"Warning: Failed to load NLI model ({NLI_CROSS_ENCODER_MODEL}): {e}")
            self._stance_detector = StanceDetector(None)
            return False

    def _load_existing_claims(self):
        if not CLAIMS_OUTPUT_FILE.exists():
            return []
        try:
            with CLAIMS_OUTPUT_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _load_existing_claim_ids(self):
        return {
            item.get('claim_id') for item in self._load_existing_claims()
            if item.get('claim_id')
        }

    def _load_export_state(self):
        if not CLAIMS_EXPORT_STATE_FILE.exists():
            return set()
        try:
            with CLAIMS_EXPORT_STATE_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return set(str(item) for item in data)
        except Exception:
            pass
        return set()

    def _save_export_state(self, exported_ids):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with CLAIMS_EXPORT_STATE_FILE.open('w', encoding='utf-8') as f:
            json.dump(sorted(exported_ids), f, indent=2, ensure_ascii=False)

    def _save_claims_output(self, items):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        existing = self._load_existing_claims()
        existing_ids = {item.get('claim_id') for item in existing if item.get('claim_id')}
        new_items = [item for item in items if item.get('claim_id') not in existing_ids]
        if not new_items:
            return
        existing.extend(new_items)
        with CLAIMS_OUTPUT_FILE.open('w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def _load_verified_claims(self):
        """Load verified claims from file."""
        return ClaimsLoader.load_claims(VERIFIED_CLAIMS_FILE)

    def _load_unlabeled_labeled_claims(self):
        if not UNLABELED_CLAIMS_OUTPUT_FILE.exists():
            return []
        try:
            with UNLABELED_CLAIMS_OUTPUT_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_unlabeled_labeled_claims(self, items):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        existing = self._load_unlabeled_labeled_claims()
        existing_ids = {item.get('claim_id') for item in existing if item.get('claim_id')}
        new_items = [item for item in items if item.get('claim_id') not in existing_ids]
        if not new_items:
            return
        existing.extend(new_items)
        with UNLABELED_CLAIMS_OUTPUT_FILE.open('w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def _load_processed_claim_ids(self):
        """Load set of processed claim IDs."""
        tracker = ProcessedClaimTracker(OUTPUT_DIR)
        return tracker.processed_ids

    def _save_processed_claim_ids(self, ids_set):
        """Save set of processed claim IDs."""
        tracker = ProcessedClaimTracker(OUTPUT_DIR)
        tracker.processed_ids = ids_set
        tracker.save()

    def _claim_id(self, passage_id, claim_text, label, query_text):
        raw = f"{passage_id}|{label}|{query_text}|{claim_text}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _claim_text_id(self, claim_text: str) -> str:
        return hashlib.sha1(claim_text.encode("utf-8")).hexdigest()

    def _negate_query(self, query: str) -> str:
        negated = self.refute_generator.negate_text(query, allow_rules=False)
        return negated or ""

    def _save_query_list(self, queries, out_path: Path):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)

    def _load_negated_query_pairs(self):
        if not QUERIES_NEGATED_OUTPUT_FILE.exists():
            print(f"Missing file: {QUERIES_NEGATED_OUTPUT_FILE}")
            return []
        try:
            with QUERIES_NEGATED_OUTPUT_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
        except Exception:
            return []

        pairs = []
        for item in data:
            if not isinstance(item, dict):
                continue
            query = (item.get("query") or "").strip()
            negated = (item.get("negated_query") or "").strip()
            if not query:
                continue
            pairs.append((query, negated))
        return pairs

    def _get_thread_retriever(self):
        if not hasattr(self._retriever_local, "retriever"):
            model_manager = MinimalModelManager(str(self.index_dir))
            self._retriever_local.retriever = SimpleRetriever(model_manager, str(self.index_dir))
        return self._retriever_local.retriever

    def _export_simple_labels_csv(self, items):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"query_labels_{timestamp}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["claim_id", "query", "label"])
            writer.writeheader()
            for item in items:
                writer.writerow(item)
        return out_path

    def label_from_extracted_queries_csv(self):
        """Label extracted queries without retrieval and export to CSV."""
        pairs = self._load_negated_query_pairs()
        if not pairs:
            print("No query pairs found.")
            return

        rows = []
        seen = set()

        for query, negated in pairs:
            query_text = (query or "").strip()
            if query_text:
                claim_id = self._claim_id("query", query_text, "supports", query_text)
                if claim_id not in seen:
                    rows.append({
                        "claim_id": claim_id,
                        "query": query_text,
                        "label": "supports"
                    })
                    seen.add(claim_id)

            negated_text = (negated or "").strip()
            if negated_text:
                claim_id = self._claim_id("negated_query", negated_text, "refutes", negated_text)
                if claim_id not in seen:
                    rows.append({
                        "claim_id": claim_id,
                        "query": negated_text,
                        "label": "refutes"
                    })
                    seen.add(claim_id)

        out_path = self._export_simple_labels_csv(rows)
        print("\nLabel export complete.")
        print(f"Total rows: {len(rows)}")
        print(f"CSV saved: {out_path}")

    def label_unlabeled_claims_with_nli(self, top_k: int = UNLABELED_TOPK):
        """Label claims from claims_unlabeled.json using retrieval + NLI stance."""
        if not self.ensure_models_initialized(skip_stance=True):
            return

        self._ensure_stance_detector()
        claims = self._load_verified_claims()
        if not claims:
            print("No claims found in claims_unlabeled.json.")
            return

        processed_ids = self._load_processed_claim_ids()
        output_items = []

        if self.retriever and top_k:
            self.retriever.FINAL_TOPK = int(top_k)

        for claim in claims:
            claim_text = (claim.get("claim") or "").strip()
            if not claim_text:
                continue

            claim_id = claim.get("id") or self._claim_text_id(claim_text)
            if claim_id in processed_ids:
                continue

            results = self.retriever.search(claim_text)
            best = None
            best_conf = -1.0
            best_stance = None
            best_reason = "nli"

            for result in results:
                passage = result.get("passage", {})
                passage_text = (passage.get("text") or "").strip()
                if not passage_text:
                    continue

                stance, confidence = self._stance_detector.detect_stance_nli(
                    claim_text,
                    passage_text
                )
                if stance is None:
                    stance, confidence = self._stance_detector.detect_stance(
                        claim_text,
                        passage_text
                    )
                    best_reason = "heuristic"

                if stance is None:
                    continue

                if stance == "neutral":
                    if best is None:
                        best = result
                        best_conf = confidence or 0.5
                        best_stance = "neutral"
                    continue

                conf_val = confidence if confidence is not None else 0.0
                if conf_val > best_conf:
                    best = result
                    best_conf = conf_val
                    best_stance = stance

            if best is None:
                best = results[0] if results else {}
                best_stance = "neutral"
                best_conf = 0.5
                best_reason = "fallback"

            passage = best.get("passage", {}) if isinstance(best, dict) else {}
            scores = best.get("scores", {}) if isinstance(best, dict) else {}
            output_items.append({
                "claim_id": claim_id,
                "claim": claim_text,
                "label": best_stance,
                "stance_confidence": float(best_conf) if best_conf is not None else None,
                "evidence": {
                    "passage_id": passage.get("passage_id") or passage.get("id"),
                    "title": passage.get("title"),
                    "url": passage.get("url"),
                    "text": passage.get("text"),
                    "scores": {
                        "final_score": float(best.get("final_score", 0.0)) if isinstance(best, dict) else 0.0,
                        "cross_encoder": float(scores.get("cross_encoder", 0.0)) if isinstance(scores, dict) else 0.0,
                        "lexical": float(scores.get("lexical", 0.0)) if isinstance(scores, dict) else 0.0
                    }
                },
                "stance_method": best_reason,
                "created_at": datetime.now().isoformat()
            })

            processed_ids.add(claim_id)

            if len(output_items) >= 25:
                self._save_unlabeled_labeled_claims(output_items)
                output_items = []

        if output_items:
            self._save_unlabeled_labeled_claims(output_items)

        self._save_processed_claim_ids(processed_ids)
        print("\nUnlabeled claim stance labeling complete.")
        print(f"Output: {UNLABELED_CLAIMS_OUTPUT_FILE}")

    def extract_queries_and_negations(self):
        """Extract query list and a negated-claims list without indexing/labeling."""
        def _load_list(path: Path):
            if not path.exists():
                return []
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else []
            except Exception:
                return []

        def _fallback_negation(text: str) -> str:
            cleaned = (text or "").strip()
            if not cleaned:
                return ""
            if cleaned.endswith("."):
                cleaned = cleaned[:-1]
            return f"There is no evidence that {cleaned}."

        use_model_choice = input("Use T5 model for negation? (y/N): ").strip().lower()
        use_model = use_model_choice == "y"
        if use_model:
            print("Loading T5 negation model (may take a while)...")

        negator = self.refute_generator if use_model else RefuteGenerator(use_model=False)

        claims = self._load_verified_claims()
        if not claims:
            print("No verified claims found.")
            return

        seen = set()
        queries = []
        for claim in claims:
            text = (claim.get("claim") or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(text)

        if not queries:
            print("No valid query text found.")
            return

        existing_queries = _load_list(QUERIES_OUTPUT_FILE)
        existing_query_set = {str(item).strip().lower() for item in existing_queries}
        existing_negated = _load_list(QUERIES_NEGATED_OUTPUT_FILE)
        existing_negated_map = {}
        for item in existing_negated:
            if isinstance(item, dict) and item.get("query"):
                existing_negated_map[item["query"].strip().lower()] = item

        all_queries = list(existing_queries)
        for query in queries:
            key = query.lower()
            if key in existing_query_set:
                continue
            existing_query_set.add(key)
            all_queries.append(query)

        negated_items = list(existing_negated)
        total = len(queries)
        processed = 0
        saved_negations = 0
        skipped = 0
        fallback_used = 0
        batch_size = 25

        for query in queries:
            processed += 1
            key = query.lower()
            if key in existing_negated_map:
                skipped += 1
            else:
                negated = negator.negate_text(query, allow_rules=True)
                if not negated:
                    negated = _fallback_negation(query)
                    fallback_used += 1
                negated_items.append({
                    "query": query,
                    "negated_query": negated
                })
                existing_negated_map[key] = negated_items[-1]
                saved_negations += 1

            if processed % batch_size == 0 or processed == total:
                self._save_query_list(all_queries, QUERIES_OUTPUT_FILE)
                self._save_query_list(negated_items, QUERIES_NEGATED_OUTPUT_FILE)
                print(
                    f"Processed {processed}/{total} | "
                    f"negated new: {saved_negations} | "
                    f"skipped: {skipped} | "
                    f"fallback: {fallback_used}"
                )

        print("\nQuery extraction complete.")
        print(f"Queries saved: {QUERIES_OUTPUT_FILE}")
        print(f"Negated queries saved: {QUERIES_NEGATED_OUTPUT_FILE}")

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
        existing_ids = self._load_existing_claim_ids()

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

            negated_query = self._negate_query(query)

            query_variants = [(query, "positive")]
            if negated_query:
                query_variants.append((negated_query, "negative"))

            for query_text, query_polarity in query_variants:
                if not query_text:
                    continue

                print(f"\nProcessing query: {query_text}")
                if self.retriever and num_results:
                    self.retriever.FINAL_TOPK = int(num_results)

                results = self.retriever.search(query_text)

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
                            claim_key = (query_text, passage_id, claim_text, 'supports')
                            if claim_key in seen:
                                continue
                            seen.add(claim_key)

                            claim_id = self._claim_id(passage_id, claim_text, 'supports', query_text)
                            if claim_id in existing_ids:
                                continue
                            existing_ids.add(claim_id)

                            generated.append({
                                'claim_id': claim_id,
                                'query': query_text,
                                'query_negated': negated_query,
                                'query_polarity': query_polarity,
                                'source_query': query,
                                'source_passage_id': passage_id,
                                'source_title': passage.get('title'),
                                'source_url': passage.get('url'),
                                'sentence': sentence,
                                'claim': claim_text,
                                'label': 'supports',
                                'is_negated': False,
                                'source_claim': None,
                                'created_at': datetime.now().isoformat()
                            })
                            supports_count += 1

                            for refute in self.refute_generator.generate(claim_text):
                                refute_key = (query_text, passage_id, refute, 'refutes')
                                if refute_key in seen:
                                    continue
                                seen.add(refute_key)

                                refute_id = self._claim_id(passage_id, refute, 'refutes', query_text)
                                if refute_id in existing_ids:
                                    continue
                                existing_ids.add(refute_id)

                                generated.append({
                                    'claim_id': refute_id,
                                    'query': query_text,
                                    'query_negated': negated_query,
                                    'query_polarity': query_polarity,
                                    'source_query': query,
                                    'source_passage_id': passage_id,
                                    'source_title': passage.get('title'),
                                    'source_url': passage.get('url'),
                                    'sentence': sentence,
                                    'claim': refute,
                                    'label': 'refutes',
                                    'is_negated': True,
                                    'source_claim': claim_text,
                                    'created_at': datetime.now().isoformat()
                                })
                                refutes_count += 1

            processed_ids.add(claim_id)

            if generated:
                self._save_claims_output(generated)
                generated = []

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

        exported_ids = self._load_export_state()
        items_to_export = [
            item for item in items
            if item.get('claim_id') and item.get('claim_id') not in exported_ids
        ]

        if not items_to_export:
            print("No new claims to export.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"claims_export_{timestamp}.{export_format}"
        out_path = OUTPUT_DIR / file_name

        if export_format == "json":
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(items_to_export, f, indent=2, ensure_ascii=False)
            exported_ids.update(item.get('claim_id') for item in items_to_export)
            self._save_export_state(exported_ids)
            print(f"Exported to {out_path}")
            return

        fieldnames = sorted({
            key for item in items_to_export for key in item.keys()
        })
        delimiter = "\t" if export_format == "tsv" else ","

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for item in items_to_export:
                writer.writerow(item)

        exported_ids.update(item.get('claim_id') for item in items_to_export)
        self._save_export_state(exported_ids)
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
        print("1. Label unlabeled claims (NLI stance)")
        print("2. Generate claims from verified queries")
        print("3. Extract queries + negated queries")
        print("4. View statistics")
        print("5. Export claims (json/csv/tsv)")
        print("6. Export query labels (csv)")
        print("7. Exit")
        print("="*70)
        
        choice = input("Select option (1-7): ").strip()
        
        if choice == '1':
            labeler.label_unlabeled_claims_with_nli()
        elif choice == '2':
            labeler.ensure_models_initialized(skip_stance=False)
            labeler.generate_claims_from_verified_claims()
        elif choice == '3':
            labeler.extract_queries_and_negations()
        elif choice == '4':
            labeler.view_statistics()
        elif choice == '5':
            export_format = input("Export format (json/csv/tsv): ").strip().lower()
            labeler.export_claims(export_format)
        elif choice == '6':
            labeler.label_from_extracted_queries_csv()
        elif choice == '7':
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
