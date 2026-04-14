#!/usr/bin/env python3
"""
build_concept_dictionary_with_collections.py

Builds a concept dictionary from MeSH seed JSON files and generates
central SapBERT embeddings organized into collections (train/dev/test)
by splitting each seed's synonyms+aliases into parts.

Outputs:
  embeddings/{split}/vectors.npy
  embeddings/{split}/metadata.json
  embeddings/labels/vectors.npy
  embeddings/labels/metadata.json
  (per-concept label files also written to embeddings/labels/ for auditing)

Notes:
- Only canonical_label, synonyms, and aliases are used (no preferred_search_terms nor keyword_candidates).
- Each seed's variants (synonyms + aliases) are split into len(splits) parts (default 3 -> train/dev/test).
- Threads collect texts; encoding is done centrally after threads complete to avoid model-loading-in-threads issues.
"""

import argparse
import glob
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# defaults
SAPBERTMODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
INPUT_DIR = "../storage/seeds"
OUTPUT_JSON = "../storage/mesh_concepts.json"

# Global shared objects (initialized in generate_collections_from_seed_files)
_SAPBERT_MODEL = None
_SAPBERT_LOCK = None

# Central accumulators (thread-safe via CENTRAL_LOCK)
CENTRAL = {}   # CENTRAL[split] = {"texts": [...], "meta": [...]}
CENTRAL_LOCK = None

# Labels accumulator (mesh_id -> {"canonical": str or None, "variants": set(), "source_file": str})
LABELS = {}
LABELS_LOCK = None

class Stats:
    def __init__(self):
        self.counters = defaultdict(int)
        self.start_time = time.time()

    def inc(self, key, n=1):
        self.counters[key] += n

    def elapsed(self):
        return time.time() - self.start_time

    def rate(self, key):
        t = self.elapsed()
        return (self.counters[key] / t) if t > 0 else 0.0

    def snapshot(self):
        return dict(self.counters)

    def log(self, prefix="[STATS]"):
        elapsed = self.elapsed()
        msg = f"{prefix} elapsed={elapsed:.1f}s"
        for k, v in sorted(self.counters.items()):
            msg += f" | {k}={v}"
        print(msg)

# -----------------------
# Term extraction utilities (modified per your request)
# Only: canonical_label, synonyms, aliases
# -----------------------
def extract_terms_from_entry(entry: Dict[str, Any],
                             lowercase: bool = False,
                             excluded_terms: List[str] = None) -> Dict[str, Any]:
    """
    Extract canonical_label, synonyms, aliases from an entry.
    Returns dict: {
        "canonical_label": str or None,
        "synonyms": list[str],
        "aliases": list[str]
    }
    """
    def _proc_text(t):
        if t is None:
            return None
        s = str(t).strip()
        if s == "":
            return None
        return s.lower() if lowercase else s

    excluded_terms = excluded_terms or []

    canonical = entry.get("canonical_label") or entry.get("label") or None
    canonical = _proc_text(canonical)

    synonyms = []
    if "synonyms" in entry and isinstance(entry["synonyms"], (list, tuple)):
        for t in entry["synonyms"]:
            s = _proc_text(t)
            if not s:
                continue
            skip = False
            for ex in excluded_terms:
                if ex is None:
                    continue
                ex_match = ex.lower() if lowercase else ex
                if ex_match in s:
                    skip = True
                    break
            if not skip:
                synonyms.append(s)

    aliases = []
    if "aliases" in entry and isinstance(entry["aliases"], (list, tuple)):
        for t in entry["aliases"]:
            s = _proc_text(t)
            if not s:
                continue
            skip = False
            for ex in excluded_terms:
                if ex is None:
                    continue
                ex_match = ex.lower() if lowercase else ex
                if ex_match in s:
                    skip = True
                    break
            if not skip:
                aliases.append(s)

    return {
        "canonical_label": canonical,
        "synonyms": synonyms,
        "aliases": aliases
    }

# -----------------------
# Build concept dictionary from JSON seeds (unchanged semantics except extraction)
# -----------------------
def build_concept_dictionary(input_dir: str,
                              lowercase: bool = False,
                              output_path: str = "concepts.json",
                              write_pretty: bool = True,
                              excluded_tokens_field: str = "excluded_terms") -> Dict[str, Any]:
    """
    Reads all JSON files in input_dir (glob *.json) and builds a concept dictionary.
    Returns the dictionary object and writes to output_path as JSON.
    """
    stats = Stats()
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    stats.inc("seed_files", len(files))
    print(f"[INFO] Found {len(files)} seed files")

    concept_dict: Dict[str, Dict[str, Any]] = {}

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
            continue

        entries = data if isinstance(data, list) else [data]

        for entry in entries:
            stats.inc("entries_seen")
            if not isinstance(entry, dict):
                continue
            seed_id = entry.get("seed_id") or entry.get("mesh_id") or entry.get("id")
            if not seed_id:
                stats.inc("entries_skipped_no_id")
                continue

            excluded_list = entry.get(excluded_tokens_field, []) if entry.get(excluded_tokens_field) else []
            extracted = extract_terms_from_entry(entry, lowercase=lowercase, excluded_terms=excluded_list)
            terms = []
            # canonical_label + synonyms + aliases combined for simple downstream use
            if extracted.get("canonical_label"):
                terms.append(extracted["canonical_label"])
            terms.extend(extracted.get("synonyms", []))
            terms.extend(extracted.get("aliases", []))

            if not terms:
                stats.inc("concepts_no_terms")
                continue

            concept_meta = {
                "concept_id": seed_id,
                "canonical_label": extracted.get("canonical_label"),
                "terms": terms,
                "synonyms": extracted.get("synonyms", []),
                "aliases": extracted.get("aliases", []),
                "source_file": os.path.basename(fp),
            }

            concept_dict[seed_id] = concept_meta
            stats.inc("concepts_created")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outf:
        if write_pretty:
            json.dump(concept_dict, outf, indent=2, ensure_ascii=False)
        else:
            json.dump(concept_dict, outf, ensure_ascii=False)

    print(f"[INFO] Wrote {len(concept_dict)} concepts to {output_path}")
    stats.log("[BUILD]")
    return concept_dict

# -----------------------
# Utils: splitting variants into N parts (round-robin for even distribution)
# -----------------------
def split_variants_round_robin(variants: List[str], n: int) -> List[List[str]]:
    buckets = [[] for _ in range(n)]
    if not variants:
        return buckets
    for i, v in enumerate(variants):
        buckets[i % n].append(v)
    return buckets

# -----------------------
# Worker: process one seed file (this is the unit of threading)
# Workers only collect texts & metadata into CENTRAL and LABELS (do not encode).
# -----------------------
def _process_seed_file(fp: str,
                       embeddings_dir: str,
                       splits: List[str],
                       lowercase: bool,
                       excluded_tokens_field: str,
                       checkpoint: Dict[str, Any],
                       checkpoint_file: str,
                       force: bool) -> Dict[str, Any]:
    """
    Parses entries in `fp` and appends variant texts into CENTRAL per-split and
    updates LABELS for label embedding construction.

    Returns summary dict with counts and status.
    """
    safe_fp = os.path.basename(fp)
    summary = {"file": safe_fp, "entries_seen": 0, "added": 0, "skipped": 0, "failed": 0}
    try:
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        return {"file": safe_fp, "error": f"failed_load:{e}"}

    entries = data if isinstance(data, list) else [data]

    # checkpoint skip whole file if already processed
    ck_processed_files = checkpoint.get("processed_files", {}) if checkpoint else {}
    if not force and ck_processed_files.get(safe_fp):
        return {"file": safe_fp, "skipped": "already_in_checkpoint"}

    for entry in entries:
        summary["entries_seen"] += 1
        try:
            seed_id = entry.get("seed_id") or entry.get("mesh_id") or entry.get("id")
            if not seed_id:
                summary["failed"] += 1
                continue

            excluded_list = entry.get(excluded_tokens_field, []) if entry.get(excluded_tokens_field) else []
            extracted = extract_terms_from_entry(entry, lowercase=lowercase, excluded_terms=excluded_list)
            canonical = extracted.get("canonical_label")
            synonyms = extracted.get("synonyms", []) or []
            aliases = extracted.get("aliases", []) or []

            variants = list(synonyms) + list(aliases)  # combine synonyms+aliases for splitting
            k = len(splits)
            buckets = split_variants_round_robin(variants, k)

            # Append to CENTRAL under lock
            global CENTRAL, CENTRAL_LOCK, LABELS, LABELS_LOCK
            if CENTRAL_LOCK is None or LABELS_LOCK is None:
                return {"file": safe_fp, "error": "central_locks_not_initialized"}

            for i, split_name in enumerate(splits):
                texts = buckets[i] if i < len(buckets) else []
                if not texts:
                    continue
                with CENTRAL_LOCK:
                    for t in texts:
                        CENTRAL[split_name]["texts"].append(t)
                        CENTRAL[split_name]["meta"].append({
                            "mesh_id": seed_id,
                            "canonical_label": canonical,
                            "source_file": safe_fp,
                            "original_text": t,
                            "split": split_name
                        })
                        summary["added"] += 1

            # update LABELS: keep canonical + variants (for label embedding later)
            with LABELS_LOCK:
                rec = LABELS.setdefault(seed_id, {"canonical": canonical, "variants": set(), "source_file": safe_fp})
                # update canonical if not set but present
                if not rec.get("canonical") and canonical:
                    rec["canonical"] = canonical
                for v in variants:
                    rec["variants"].add(v)

        except Exception as e:
            summary["failed"] += 1
            print(f"[WARN] Failed processing entry in {fp}: {e}")
            continue

    # update checkpoint for file
    if checkpoint is not None:
        checkpoint.setdefault("processed_files", {})[safe_fp] = {"ts": time.time(), "entries": len(entries)}
        checkpoint["updated_at"] = time.time()
        if checkpoint_file:
            try:
                with open(checkpoint_file, "w", encoding="utf-8") as cf:
                    json.dump(checkpoint, cf, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[WARN] Failed writing checkpoint file: {e}")

    return summary

# -----------------------
# High-level embedding generation that uses threads per seed file to collect texts,
# then encodes centrally per split and builds label embeddings.
# -----------------------
def generate_collections_from_seed_files(input_dir: str,
                                         model_name: str = SAPBERTMODEL,
                                         embeddings_dir: str = "embeddings",
                                         splits: List[str] = None,
                                         normalize: bool = True,
                                         batch_size: int = 64,
                                         num_file_workers: int = 4,
                                         lowercase: bool = False,
                                         checkpoint_file: str = None,
                                         force: bool = False,
                                         excluded_tokens_field: str = "excluded_terms"):
    """
    Processes each JSON file under input_dir in its own thread (ThreadPoolExecutor) to
    collect variant texts into CENTRAL, then encodes centrally per split and computes label embeddings.
    """
    from sentence_transformers import SentenceTransformer
    import threading
    import numpy as np

    global _SAPBERT_MODEL, _SAPBERT_LOCK, CENTRAL, CENTRAL_LOCK, LABELS, LABELS_LOCK

    if splits is None:
        splits = ["train", "dev", "test"]
    splits = list(splits)

    # initialize central structures + locks
    CENTRAL = {s: {"texts": [], "meta": []} for s in splits}
    CENTRAL_LOCK = threading.Lock()
    LABELS = {}
    LABELS_LOCK = threading.Lock()

    # initialize shared model once (main thread) to avoid meta-tensor threading issues
    if _SAPBERT_MODEL is None:
        print(f"[INFO] Loading SentenceTransformer model '{model_name}' in main thread...")
        _SAPBERT_MODEL = SentenceTransformer(model_name, device="cpu")
        _SAPBERT_LOCK = threading.Lock()

    os.makedirs(embeddings_dir, exist_ok=True)
    # load checkpoint if present
    checkpoint = {"processed_files": {}, "updated_at": None}
    if checkpoint_file and os.path.exists(checkpoint_file) and not force:
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as cf:
                checkpoint = json.load(cf)
        except Exception:
            checkpoint = {"processed_files": {}, "updated_at": None}

    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    total_files = len(files)
    print(f"[INFO] Found {total_files} seed files. Spawning up to {num_file_workers} threads (one thread per file).")

    stats = Stats()
    stats.inc("files_total", total_files)

    # Phase 1: collect texts & metadata with threads
    results = []
    with ThreadPoolExecutor(max_workers=num_file_workers) as exe:
        future_to_fp = {}
        for fp in files:
            future = exe.submit(
                _process_seed_file,
                fp,
                embeddings_dir,
                splits,
                lowercase,
                excluded_tokens_field,
                checkpoint,
                checkpoint_file,
                force
            )
            future_to_fp[future] = fp

        for fut in as_completed(future_to_fp):
            fp = future_to_fp[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[ERROR] Worker for {fp} raised: {e}")
                res = {"file": os.path.basename(fp), "error": str(e)}
            results.append(res)
            stats.inc("files_done")
            print(f"[INFO] Completed file {os.path.basename(fp)} -> {res}")

    # Phase 2: central encoding per split (main thread)
    print("[INFO] Phase 1 complete. Beginning central encoding per split...")

    # ensure directories
    for s in splits:
        os.makedirs(os.path.join(embeddings_dir, s), exist_ok=True)

    # encode each split centrally
    all_split_stats = {}
    model = _SAPBERT_MODEL  # use shared model loaded earlier
    for s in splits:
        texts = CENTRAL[s]["texts"]
        meta = CENTRAL[s]["meta"]
        n = len(texts)
        all_split_stats[s] = {"n_texts": n}
        print(f"[INFO] Encoding split '{s}' with {n} texts (batch_size={batch_size})")
        vectors = None
        if n == 0:
            vectors = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
        else:
            # encode in one call (SentenceTransformer handles internal batching)
            with _SAPBERT_LOCK:
                emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize)
            vectors = np.asarray(emb, dtype=np.float32)

        # save vectors + metadata for this split
        vec_path = os.path.join(embeddings_dir, s, "vectors.npy")
        meta_path = os.path.join(embeddings_dir, s, "metadata.json")
        try:
            np.save(vec_path, vectors)
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2, ensure_ascii=False)
            print(f"[INFO] Wrote split '{s}' vectors -> {vec_path} (shape={vectors.shape}) and metadata -> {meta_path}")
        except Exception as e:
            print(f"[WARN] Failed saving split {s}: {e}")

    # Phase 3: build label embeddings (one vector per concept)
    print("[INFO] Building label embeddings for each concept (canonical + variants centroid)...")
    os.makedirs(os.path.join(embeddings_dir, "labels"), exist_ok=True)
    label_meta_list = []
    label_vectors = []
    # For memory safety, encode in batches per-label (each label usually small)
    for i, (mesh_id, rec) in enumerate(sorted(LABELS.items())):
        canonical = rec.get("canonical")
        variants = sorted(list(rec.get("variants", [])))
        texts_for_label = []
        if canonical:
            texts_for_label.append(canonical)
        texts_for_label.extend(variants)

        # if no texts at all, produce zero vector
        if not texts_for_label:
            vec = np.zeros((model.get_sentence_embedding_dimension(),), dtype=np.float32)
        else:
            with _SAPBERT_LOCK:
                emb = model.encode(texts_for_label, batch_size=min(32, len(texts_for_label)), show_progress_bar=False, normalize_embeddings=normalize)
            emb = np.asarray(emb, dtype=np.float32)
            # centroid & normalize
            centroid = emb.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / (norm + 1e-12)
            vec = centroid.astype(np.float32)

        label_vectors.append(vec)
        label_meta_list.append({
            "mesh_id": mesh_id,
            "canonical": canonical,
            "n_variants": len(variants),
            "source_file": rec.get("source_file")
        })

        # also save per-concept label file for auditing
        safe_cid = str(mesh_id).replace(":", "_").replace("/", "_")
        try:
            np.save(os.path.join(embeddings_dir, "labels", f"{safe_cid}_label.npy"), vec)
        except Exception:
            pass

    # save all label vectors + metadata as arrays
    try:
        label_vecs_arr = np.stack(label_vectors, axis=0) if label_vectors else np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
        np.save(os.path.join(embeddings_dir, "labels", "vectors.npy"), label_vecs_arr)
        with open(os.path.join(embeddings_dir, "labels", "metadata.json"), "w", encoding="utf-8") as lf:
            json.dump(label_meta_list, lf, indent=2, ensure_ascii=False)
        print(f"[INFO] Wrote labels vectors -> {os.path.join(embeddings_dir, 'labels','vectors.npy')} (shape={label_vecs_arr.shape}) and metadata")
    except Exception as e:
        print(f"[WARN] Failed saving label collections: {e}")

    stats.log("[COLLECTION EMBEDDING]")
    summary = {
        "split_stats": all_split_stats,
        "n_labels": len(label_meta_list),
        "file_results": results
    }
    return summary

# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Build concept dictionary from MeSH seed JSON files and generate SapBERT embeddings split into central collections.")
    p.add_argument("--input-dir", type=str, default=INPUT_DIR, help="Directory containing mesh seed JSON files (glob *.json).")
    p.add_argument("--output", type=str, default=OUTPUT_JSON, help="Path for output concept dictionary JSON.")
    p.add_argument("--lowercase", action="store_true", help="Lowercase all extracted terms (useful for normalization).")
    p.add_argument("--generate-embeddings", action="store_true", default=True, help="Also generate SapBERT embeddings for each split/collection.")
    p.add_argument("--sapbert-model", type=str, default=SAPBERTMODEL, help="SapBERT model name (if generating embeddings).")
    p.add_argument("--embeddings-dir", type=str, default="embeddings", help="Directory to write per-split .npy embedding files.")
    p.add_argument("--embeddings-batch-size", type=int, default=64, help="Batch size passed to SentenceTransformer.encode().")
    p.add_argument("--num-file-workers", type=int, default=4, help="Number of threads (files processed in parallel).")
    p.add_argument("--checkpoint", type=str, default="collections_checkpoint.json", help="Checkpoint JSON path for resume/resilience.")
    p.add_argument("--force", action="store_true", help="Force re-generate embeddings even if files exist / checkpoint says done.")
    p.add_argument("--splits", type=str, default="train,dev,test", help="Comma-separated split names in order (default 'train,dev,test').")
    p.add_argument("--excluded-tokens-field", type=str, default="excluded_terms", help="Field name in seed entries listing excluded tokens.")
    args = p.parse_args()

    # build concepts dictionary (for convenience / auditing)
    concept_dict = build_concept_dictionary(
        input_dir=args.input_dir,
        lowercase=args.lowercase,
        output_path=args.output
    )

    if args.generate_embeddings:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        try:
            summary = generate_collections_from_seed_files(
                input_dir=args.input_dir,
                model_name=args.sapbert_model,
                embeddings_dir=args.embeddings_dir,
                splits=splits,
                normalize=True,
                batch_size=args.embeddings_batch_size,
                num_file_workers=args.num_file_workers,
                lowercase=args.lowercase,
                checkpoint_file=args.checkpoint,
                force=args.force,
                excluded_tokens_field=args.excluded_tokens_field
            )
            print("[INFO] Collection embedding generation complete.")
            # write a summary results file
            try:
                summary_path = os.path.join(os.path.dirname(args.checkpoint) or ".", "collections_summary.json")
                with open(summary_path, "w", encoding="utf-8") as sf:
                    json.dump(summary, sf, indent=2, ensure_ascii=False)
                print(f"[INFO] Wrote collections summary to {summary_path}")
            except Exception:
                pass
        except Exception as e:
            print(f"[ERROR] Failed to generate collections: {e}")

if __name__ == "__main__":
    main()
