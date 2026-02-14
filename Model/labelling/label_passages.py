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
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from retrieval.simple_retriever import SimpleRetriever, MinimalModelManager
from claim_extraction import (
    SentenceSplitter,
    ClaimSentenceFilter,
    ClaimExtractor,
    RefuteGenerator,
    set_negator_model
)
from persistence import ClaimsLoader, ProcessedClaimTracker
from stance_detector import StanceDetector

from config import (
    INDEX_DIR,
    DEFAULT_DISPLAY_RESULTS,
    AUTO_RELEVANT_THRESHOLD, AUTO_UNRELATED_THRESHOLD,
    
    AUTO_UNRELATED_CE_MAX, AUTO_UNRELATED_LEX_MAX,
    
    OUTPUT_DIR,
    VERIFIED_CLAIMS_FILE,
    LABELED_CLAIMS_FILE,
    LABELED_CLAIMS_NEGATED_FILE,
    UNLABELED_CLAIMS_OUTPUT_FILE,
    UNLABELED_TOPK,
    UNLABELED_FILTER_TOPK,
    UNLABELED_FILTER_MIN_SCORE,
    UNLABELED_FILTER_SAVE_EVERY,
    UNLABELED_FILTER_WORKERS,
    UNLABELED_FILTER_CHECKPOINT_FILE,
    NLI_CROSS_ENCODER_MODEL,
    UNLABELED_MIN_CONFIDENCE,
    UNLABELED_SAVE_EVERY,
    NLI_MIN_CONFIDENCE,
    NLI_MIN_MARGIN,
    NLI_MIN_RELEVANCE,
    CLAIMS_OUTPUT_FILE,
    CLAIMS_EXPORT_STATE_FILE,
    QUERIES_OUTPUT_FILE,
    QUERIES_NEGATED_OUTPUT_FILE
    ,
    NEGATION_MODEL_LARGE
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
        self._nli_local = threading.local()
        self._stance_local = threading.local()
        
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
            from sentence_transformers.cross_encoder import CrossEncoder
            self._nli_model = CrossEncoder(
                NLI_CROSS_ENCODER_MODEL,
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": False}
            )
            self._stance_detector = StanceDetector(
                self._nli_model,
                min_confidence=NLI_MIN_CONFIDENCE,
                min_margin=NLI_MIN_MARGIN
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to load NLI model ({NLI_CROSS_ENCODER_MODEL}): {e}")
            self._stance_detector = StanceDetector(None)
            return False

    def _get_thread_stance_detector(self) -> StanceDetector:
        if not hasattr(self._stance_local, "detector"):
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
                self._stance_local.detector = StanceDetector(
                    CrossEncoder(
                        NLI_CROSS_ENCODER_MODEL,
                        device="cpu",
                        model_kwargs={"low_cpu_mem_usage": False}
                    ),
                    min_confidence=NLI_MIN_CONFIDENCE,
                    min_margin=NLI_MIN_MARGIN
                )
            except Exception as e:
                print(f"Warning: Failed to load NLI model ({NLI_CROSS_ENCODER_MODEL}): {e}")
                self._stance_local.detector = StanceDetector(None)
        return self._stance_local.detector

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

    def _load_labeled_claims(self):
        """Load labeled claims from file."""
        if not LABELED_CLAIMS_FILE.exists():
            print(f"Claims file not found at {LABELED_CLAIMS_FILE}")
            return []
        try:
            with LABELED_CLAIMS_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as exc:
            print(f"Failed to read claims from {LABELED_CLAIMS_FILE}: {exc}")
            return []

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

    def _unlabeled_claim_key(self, item) -> str:
        claim_id = (item.get("id") or "").strip()
        if claim_id:
            return claim_id
        claim_text = (item.get("claim") or "").strip()
        if claim_text:
            return self._claim_text_id(claim_text)
        return hashlib.sha1(json.dumps(item, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_filter_checkpoint(self):
        if not UNLABELED_FILTER_CHECKPOINT_FILE.exists():
            return set()
        try:
            with UNLABELED_FILTER_CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                processed = data.get("processed_ids") or []
                return {str(item) for item in processed}
        except Exception:
            pass
        return set()

    def _save_filter_checkpoint(self, processed_ids, stats):
        payload = {
            "processed_ids": sorted(processed_ids),
            "stats": stats,
            "saved_at": datetime.now().isoformat()
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with UNLABELED_FILTER_CHECKPOINT_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

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

        claims = self._load_verified_claims()
        if not claims:
            print("No claims found in claims_unlabeled.json.")
            return

        processed_ids = self._load_processed_claim_ids()
        existing_items = self._load_unlabeled_labeled_claims()
        existing_claim_texts = {
            (item.get("claim") or "").strip().lower()
            for item in existing_items
            if (item.get("claim") or "").strip()
        }
        pending_claims = []
        for item in claims:
            claim_text = (item.get("claim") or "").strip()
            if not claim_text:
                continue
            if claim_text.lower() in existing_claim_texts:
                continue
            claim_id = item.get("id") or self._claim_text_id(claim_text)
            if claim_id in processed_ids:
                continue
            pending_claims.append({"claim_id": claim_id, "claim": claim_text})

        if not pending_claims:
            print("All claims already processed.")
            return
        output_items = []
        processed_count = 0
        successful_count = 0
        skipped_count = 0
        save_period = 5  # Log progress every 5 claims

        def _process_claim(item):
            try:
                claim_text = item.get("claim")
                claim_id = item.get("claim_id")
                if not claim_text or not claim_id:
                    return None

                retriever = self._get_thread_retriever()
                if retriever and top_k:
                    retriever.FINAL_TOPK = int(top_k)

                results = retriever.search(claim_text)
                if top_k:
                    results = results[:int(top_k)]
                stance_detector = self._get_thread_stance_detector()

                best = None
                best_conf = -1.0
                best_stance = None

                print("\n" + "-" * 70)
                print(f"Claim: {claim_text}")
                print(f"Processing {len(results)} retrieved passages...")
                passages_processed = 0
                passages_filtered = 0

                for result in results:
                    passage = result.get("passage", {})
                    passage_text = (passage.get("text") or "").strip()
                    if not passage_text:
                        continue

                    final_score = float(result.get("final_score", 0.0))
                    title = (passage.get("title") or "").strip()
                    passage_id = passage.get("passage_id") or passage.get("id")
                    
                    if final_score < NLI_MIN_RELEVANCE:
                        passages_filtered += 1
                        print(f"  [SKIP] Passage {passage_id} | score={final_score:.3f} < threshold {NLI_MIN_RELEVANCE}")
                        continue
                    
                    passages_processed += 1
                    stance, confidence = stance_detector.detect_stance_nli(
                        claim_text,
                        passage_text
                    )
                    if stance is None:
                        stance, confidence = stance_detector.detect_stance(
                            claim_text,
                            passage_text
                        )
                        

                    conf_display = f"{confidence:.3f}" if confidence is not None else "n/a"
                    stance_display = stance or "unknown"
                    print(f"  - Passage {passage_id} | {title[:60]} | stance={stance_display} | conf={conf_display} | score={final_score:.3f}")

                    if stance is None:
                        continue

                    if stance == "neutral":
                        if best is None:
                            best = result
                            best_conf = confidence or 0.5
                            best_stance = "neutral"
                        continue

                    conf_val = confidence if confidence is not None else 0.0
                    # Use NLI confidence primarily, use retrieval score as tiebreaker
                    # Don't multiply (kills high NLI confidence), instead blend them
                    weighted_conf = 0.85 * conf_val + 0.15 * max(final_score, 0.0)
                    print(f"    → NLI={conf_val:.3f}, retrieval={final_score:.3f}, weighted={weighted_conf:.3f}")
                    if weighted_conf > best_conf:
                        best = result
                        best_conf = weighted_conf
                        best_stance = stance
                        print(f"    ✓ New best: {stance} ({weighted_conf:.3f})")

                if best is None:
                    best = results[0] if results else {}
                    best_stance = "neutral"
                    best_conf = 0.5

                final_conf = best_conf if best_conf is not None else 0.0
                print(f"\n[Summary] Processed {passages_processed}, Filtered {passages_filtered}, Final: {best_stance} ({final_conf:.3f})")
                
                if final_conf < UNLABELED_MIN_CONFIDENCE:
                    final_label = "unproven"
                elif best_stance in {"supports", "refutes"}:
                    final_label = best_stance
                else:
                    final_label = "unproven"

                print(f"=> Final label: {final_label} (threshold: {UNLABELED_MIN_CONFIDENCE})")

                return {
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "label": final_label
                }
            except Exception as exc:
                print(f"Error processing claim: {exc}")
                import traceback
                traceback.print_exc()
                return None

        for item in pending_claims:
            processed_count += 1
            result = _process_claim(item)
            
            if not result:
                skipped_count += 1
                if processed_count % save_period == 0:
                    print(f"\n[Progress] Processed {processed_count}, successful {successful_count}, skipped {skipped_count}")
                continue

            if result["claim_id"] in processed_ids:
                skipped_count += 1
                continue
            processed_ids.add(result["claim_id"])
            successful_count += 1

            output_items.append(result)
            if len(output_items) >= UNLABELED_SAVE_EVERY:
                self._save_unlabeled_labeled_claims(output_items)
                print(f"Saved {len(output_items)} items to {UNLABELED_CLAIMS_OUTPUT_FILE}")
                output_items = []
            elif processed_count % save_period == 0:
                print(f"\n[Progress] Processed {processed_count}, successful {successful_count}, skipped {skipped_count}")

        if output_items:
            self._save_unlabeled_labeled_claims(output_items)
            print(f"Saved {len(output_items)} items to {UNLABELED_CLAIMS_OUTPUT_FILE}")

        self._save_processed_claim_ids(processed_ids)
        print("\nUnlabeled claim stance labeling complete.")
        print(f"Output: {UNLABELED_CLAIMS_OUTPUT_FILE}")

    def remove_irrelevant_unlabeled_claims(
        self,
        top_k: int = UNLABELED_FILTER_TOPK,
        min_score: float = UNLABELED_FILTER_MIN_SCORE,
        reset_checkpoint: bool = False
    ):
        """Remove claims with weak retrieval scores and overwrite claims_unlabeled.json."""
        if not self.ensure_models_initialized(skip_stance=True):
            return

        claims = self._load_verified_claims()
        if not claims:
            print("No claims found in claims_unlabeled.json.")
            return

        processed_ids = set()
        if not reset_checkpoint:
            processed_ids = self._load_filter_checkpoint()
        else:
            if UNLABELED_FILTER_CHECKPOINT_FILE.exists():
                try:
                    UNLABELED_FILTER_CHECKPOINT_FILE.unlink()
                except Exception:
                    pass
        current_keys = {self._unlabeled_claim_key(item) for item in claims}
        processed_ids = processed_ids.intersection(current_keys)

        status_map = {}
        for item in claims:
            key = self._unlabeled_claim_key(item)
            if key in processed_ids:
                status_map[key] = True

        total = len(claims)
        processed = len(processed_ids)
        score_sum = 0.0
        score_count = 0

        if not processed_ids:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = OUTPUT_DIR / f"claims_unlabeled_backup_{timestamp}.json"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with backup_path.open("w", encoding="utf-8") as f:
                json.dump(claims, f, indent=2, ensure_ascii=False)
        else:
            backup_path = None

        pending = []
        for item in claims:
            key = self._unlabeled_claim_key(item)
            if key in processed_ids:
                continue
            claim_text = (item.get("claim") or "").strip()
            if not claim_text:
                processed_ids.add(key)
                status_map[key] = True
                continue
            pending.append((key, item, claim_text))

        if not pending and processed_ids:
            print(
                "All claims already processed. "
                "Run with reset to reprocess using retrieval."
            )
            return

        def _process_claim(key, item, claim_text):
            try:
                retriever = self._get_thread_retriever()
                if retriever and top_k:
                    retriever.FINAL_TOPK = int(top_k)
                results = retriever.search(claim_text)
                if top_k:
                    results = results[:int(top_k)]
                best_score = max(
                    (float(result.get("final_score", 0.0)) for result in results),
                    default=0.0
                )
                return key, item, best_score
            except Exception:
                return key, item, 0.0

        save_every = max(int(UNLABELED_FILTER_SAVE_EVERY), 1)
        max_workers = max(int(UNLABELED_FILTER_WORKERS), 1)
        def _build_updated_list():
            return [
                item for item in claims
                if status_map.get(self._unlabeled_claim_key(item)) is not False
            ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process_claim, key, item, claim_text)
                for key, item, claim_text in pending
            ]

            for future in as_completed(futures):
                key, item, best_score = future.result()
                processed_ids.add(key)
                processed += 1
                score_sum += best_score
                score_count += 1

                status_map[key] = best_score >= min_score

                if processed % save_every == 0 or processed == total:
                    updated_claims = _build_updated_list()
                    with VERIFIED_CLAIMS_FILE.open("w", encoding="utf-8") as f:
                        json.dump(updated_claims, f, indent=2, ensure_ascii=False)

                    kept_count = sum(1 for val in status_map.values() if val is True)
                    removed_count = sum(1 for val in status_map.values() if val is False)
                    stats = {
                        "total": total,
                        "processed": processed,
                        "pending": max(total - processed, 0),
                        "kept": kept_count,
                        "removed": removed_count,
                        "avg_best_score": (
                            score_sum / score_count if score_count else 0.0
                        )
                    }
                    self._save_filter_checkpoint(processed_ids, stats)
                    print(
                        f"Processed {processed}/{total} | kept: {kept_count} | "
                        f"removed: {removed_count} | avg_score: {stats['avg_best_score']:.3f}"
                    )

        updated_claims = _build_updated_list()
        with VERIFIED_CLAIMS_FILE.open("w", encoding="utf-8") as f:
            json.dump(updated_claims, f, indent=2, ensure_ascii=False)

        kept_count = sum(1 for val in status_map.values() if val is True)
        removed_count = sum(1 for val in status_map.values() if val is False)
        stats = {
            "total": total,
            "processed": processed,
            "pending": max(total - processed, 0),
            "kept": kept_count,
            "removed": removed_count,
            "avg_best_score": score_sum / score_count if score_count else 0.0
        }
        self._save_filter_checkpoint(processed_ids, stats)

        print("\nIrrelevant claim removal complete.")
        print(f"Kept: {kept_count} | Removed: {removed_count}")
        if backup_path is not None:
            print(f"Backup saved: {backup_path}")
        print(f"Updated file: {VERIFIED_CLAIMS_FILE}")

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
        label_map = {
            "true": "false",
            "false": "true",
            "supports": "refutes",
            "support": "refute",
            "refutes": "supports",
            "refute": "support",
        }

        for claim in claims:
            text = (claim.get("claim") or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)

            raw_label = (claim.get("label") or "").strip()
            label_key = raw_label.lower()
            negated_label = label_map.get(label_key)
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

    def generate_negations_from_labeled_claims(self):
        """Generate opposite-polarity sentences from claims_labeled.json."""
        claims = self._load_labeled_claims()
        if not claims:
            print("No labeled claims found in claims_labeled.json.")
            return

        set_negator_model(NEGATION_MODEL_LARGE)
        negator = RefuteGenerator(use_model=True)

        items = []
        seen = set()
        total = len(claims)
        processed = 0
        saved = 0
        save_every = 10
        label_map = {
            "true": "false",
            "false": "true",
            "supports": "refutes",
            "support": "refute",
            "refutes": "supports",
            "refute": "support",
        }
        for claim in claims:
            text = (claim.get("claim") or "").strip()
            if not text:
                continue
            processed += 1
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)

            raw_label = (claim.get("label") or "").strip()
            label_key = raw_label.lower()
            negated_label = label_map.get(label_key)

            negated = negator.negate_text(text, allow_rules=True)
            if not negated:
                continue

            items.append({
                "id": claim.get("id"),
                "claim": text,
                "negated_claim": negated,
                "label_original": raw_label or None,
                "label_negated": negated_label
            })

            if len(items) >= save_every:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with LABELED_CLAIMS_NEGATED_FILE.open("w", encoding="utf-8") as f:
                    json.dump(items, f, indent=2, ensure_ascii=False)
                saved = len(items)
                print(f"Saved {saved} items to {LABELED_CLAIMS_NEGATED_FILE}")

            if processed % 25 == 0 or processed == total:
                print(
                    f"Processed {processed}/{total} | "
                    f"negated: {len(items)} | saved: {saved}"
                )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with LABELED_CLAIMS_NEGATED_FILE.open("w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        saved = len(items)
        print(f"Saved {saved} items to {LABELED_CLAIMS_NEGATED_FILE}")

        print("\nNegation generation complete.")
        print(f"Negated claims saved: {LABELED_CLAIMS_NEGATED_FILE}")

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
        print("3. Generate negations from labeled claims")
        print("4. Extract queries + negated queries")
        print("5. View statistics")
        print("6. Export claims (json/csv/tsv)")
        print("7. Export query labels (csv)")
        print("8. Remove irrelevant unlabeled claims")
        print("9. Exit")
        print("="*70)
        
        choice = input("Select option (1-9): ").strip()
        
        if choice == '1':
            labeler.label_unlabeled_claims_with_nli()
        elif choice == '2':
            labeler.ensure_models_initialized(skip_stance=False)
            labeler.generate_claims_from_verified_claims()
        elif choice == '3':
            labeler.generate_negations_from_labeled_claims()
        elif choice == '4':
            labeler.extract_queries_and_negations()
        elif choice == '5':
            labeler.view_statistics()
        elif choice == '6':
            export_format = input("Export format (json/csv/tsv): ").strip().lower()
            labeler.export_claims(export_format)
        elif choice == '7':
            labeler.label_from_extracted_queries_csv()
        elif choice == '8':
            threshold_raw = input(
                f"Min retrieval score (default {UNLABELED_FILTER_MIN_SCORE}): "
            ).strip()
            top_k_raw = input(
                f"Top-k results (default {UNLABELED_FILTER_TOPK}): "
            ).strip()
            reset_raw = input("Reset filter checkpoint? (y/N): ").strip().lower()

            min_score = UNLABELED_FILTER_MIN_SCORE
            top_k = UNLABELED_FILTER_TOPK
            reset_checkpoint = reset_raw == "y"

            if threshold_raw:
                try:
                    min_score = float(threshold_raw)
                except ValueError:
                    print("Invalid score; using default.")

            if top_k_raw:
                try:
                    top_k = int(top_k_raw)
                except ValueError:
                    print("Invalid top-k; using default.")

            labeler.remove_irrelevant_unlabeled_claims(
                top_k=top_k,
                min_score=min_score,
                reset_checkpoint=reset_checkpoint
            )
        elif choice == '9':
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
