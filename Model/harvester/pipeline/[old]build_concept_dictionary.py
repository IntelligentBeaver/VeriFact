#!/usr/bin/env python3
"""
[old]build_concept_dictionary.py

Builds a concept dictionary from MeSH seed JSON files.

Example:
  python [old]build_concept_dictionary.py \
    --input-dir storage/seeds \
    --output concepts.json \
    --min-keyword-score 0.6 \
    --lowercase False \
    --generate-embeddings False

To generate embeddings (requires sentence-transformers installed):
  python [old]build_concept_dictionary.py --input-dir storage/seeds \
    --output concepts.json --generate-embeddings True \
    --sapbert-model cambridgeltl/SapBERT-from-PubMedBERT-fulltext
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set


SAPBERTMODEL="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
INPUT_DIR="../storage/seeds"
OUTPUT_DIR="../storage/concepts.json"

def extract_terms_from_entry(entry: Dict[str, Any],
                             min_keyword_score: float = 0.6,
                             include_fields: List[str] = None,
                             lowercase: bool = False,
                             excluded_terms: List[str] = None) -> List[str]:
    """
    Given a mesh seed entry dict, return a cleaned list of surface forms (terms).
    """
    if include_fields is None:
        include_fields = ["canonical_label", "synonyms", "aliases", "preferred_search_terms", "keyword_candidates"]

    terms: Set[str] = set()
    def add_term(t):
        if t is None:
            return
        s = str(t).strip()
        if s == "":
            return
        if lowercase:
            s_proc = s.lower()
        else:
            s_proc = s
        # exclude numeric or explicit excluded tokens
        if excluded_terms:
            for ex in excluded_terms:
                if ex is None:
                    continue
                # simple containment check; excluded_terms expected to be exact tokens/fragments
                if (lowercase and ex.lower() in s_proc) or (not lowercase and ex in s_proc):
                    return
        terms.add(s_proc)

    # canonical_label
    if "canonical_label" in entry and entry.get("canonical_label"):
        add_term(entry["canonical_label"])

    # synonyms, aliases, preferred_search_terms are expected to be lists
    for fld in ("synonyms", "aliases", "preferred_search_terms"):
        if fld in entry and isinstance(entry[fld], (list, tuple)):
            for t in entry[fld]:
                add_term(t)

    # keyword_candidates: use only ones >= min_keyword_score if present
    if "keyword_candidates" in entry and isinstance(entry["keyword_candidates"], list):
        for candidate in entry["keyword_candidates"]:
            try:
                term = candidate.get("term")
                score = float(candidate.get("score", 0.0))
            except Exception:
                continue
            if score >= min_keyword_score:
                add_term(term)

    # also include `label` if canonical_label missing
    if not terms and "label" in entry and entry.get("label"):
        add_term(entry["label"])

    # final: return sorted list for determinism
    return sorted(terms)

def build_concept_dictionary(input_dir: str,
                              min_keyword_score: float = 0.6,
                              lowercase: bool = False,
                              include_fields: List[str] = None,
                              output_path: str = "concepts.json",
                              write_pretty: bool = True,
                              excluded_tokens_field: str = "excluded_terms") -> Dict[str, Any]:
    """
    Reads all JSON files in input_dir (glob *.json) and builds a concept dictionary.
    Returns the dictionary object and writes to output_path as JSON.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    concept_dict: Dict[str, Dict[str, Any]] = {}

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
            continue

        # the file may contain a list of entries or a single entry
        entries = data if isinstance(data, list) else [data]

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            seed_id = entry.get("seed_id") or entry.get("mesh_id") or entry.get("id")
            if not seed_id:
                # try to derive an id from canonical_label as fallback (not ideal)
                fallback = entry.get("canonical_label") or entry.get("label")
                if fallback:
                    seed_id = f"auto:{fallback.replace(' ', '_')}"
                else:
                    print("[WARN] Skipping entry with no seed_id and no label.")
                    continue

            excluded_terms_list = entry.get(excluded_tokens_field, []) if entry.get(excluded_tokens_field) else []
            terms = extract_terms_from_entry(
                entry,
                min_keyword_score=min_keyword_score,
                include_fields=include_fields,
                lowercase=lowercase,
                excluded_terms=excluded_terms_list
            )

            if not terms:
                # still include canonical label if possible
                can_label = entry.get("canonical_label") or entry.get("label")
                if can_label:
                    term_to_add = can_label.lower().strip() if lowercase else can_label.strip()
                    if excluded_terms_list and any(ex in term_to_add for ex in excluded_terms_list):
                        pass
                    else:
                        terms = [term_to_add]

            # build concept meta
            concept_meta = {
                "concept_id": seed_id,
                "canonical_label": entry.get("canonical_label") or entry.get("label"),
                "terms": terms,
                # preserve some helpful metadata for downstream use / auditing
                "sapbert_metadata": entry.get("sapbert_metadata", None),
                "source_file": os.path.basename(fp),
            }

            concept_dict[seed_id] = concept_meta

    # write out
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outf:
        if write_pretty:
            json.dump(concept_dict, outf, indent=2, ensure_ascii=False)
        else:
            json.dump(concept_dict, outf, ensure_ascii=False)

    print(f"[INFO] Wrote {len(concept_dict)} concepts to {output_path}")
    return concept_dict

def generate_sapbert_embeddings(concept_dict: Dict[str, Any],
                                model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                                out_embeddings_path: str = "embeddings/concept_embeddings.npz",
                                normalize: bool = True,
                                batch_size: int = 64):
    """
    Optionally generate SapBERT embeddings for every surface form (term).
    Saves a single .npz containing:
      - embeddings: (N, D) float32
      - term_texts: list of term texts (length N)
      - concept_ids: list of concept ids (length N)
    Requires: sentence_transformers, numpy
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception as e:
        raise RuntimeError("sentence-transformers is required for embedding generation. Install via `pip install sentence-transformers`") from e

    import numpy as np

    # flatten terms -> texts + concept mapping
    term_texts: List[str] = []
    concept_ids: List[str] = []
    for cid, meta in concept_dict.items():
        for t in meta.get("terms", []):
            term_texts.append(t)
            concept_ids.append(cid)

    if not term_texts:
        raise ValueError("No terms found in concept_dict to embed.")

    print(f"[INFO] Encoding {len(term_texts)} terms with model {model_name} ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(term_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    os.makedirs(os.path.dirname(out_embeddings_path) or ".", exist_ok=True)
    np.savez_compressed(out_embeddings_path,
                        embeddings=embeddings,
                        term_texts=term_texts,
                        concept_ids=concept_ids)
    print(f"[INFO] Saved embeddings: {out_embeddings_path} (shape {embeddings.shape})")
    return out_embeddings_path

def main():
    p = argparse.ArgumentParser(description="Build concept dictionary from MeSH seed JSON files.")
    p.add_argument("--input-dir", type=str, default=INPUT_DIR, help="Directory containing mesh seed JSON files (glob *.json).")
    p.add_argument("--output", type=str, default=OUTPUT_DIR, help="Path for output concept dictionary JSON.")
    p.add_argument("--min-keyword-score", type=float, default=0.6, help="Minimum keyword candidate score to include candidate.")
    p.add_argument("--lowercase", action="store_true", help="Lowercase all extracted terms (useful for normalization).")
    p.add_argument("--generate-embeddings", action="store_true", help="Also generate SapBERT embeddings for each term (requires sentence-transformers).")
    p.add_argument("--sapbert-model", type=str, default=SAPBERTMODEL, help="SapBERT model name (if generating embeddings).")
    p.add_argument("--embeddings-output", type=str, default="embeddings/concept_embeddings.npz", help="Output path for embeddings .npz when --generate-embeddings is used.")
    args = p.parse_args()

    concept_dict = build_concept_dictionary(
        input_dir=args.input_dir,
        min_keyword_score=args.min_keyword_score,
        lowercase=args.lowercase,
        output_path=args.output
    )

    if args.generate_embeddings:
        try:
            emb_path = generate_sapbert_embeddings(concept_dict, model_name=args.sapbert_model, out_embeddings_path=args.embeddings_output)
            print(f"[INFO] Embeddings created at: {emb_path}")
        except Exception as e:
            print(f"[ERROR] Failed to generate embeddings: {e}")

if __name__ == "__main__":
    main()
