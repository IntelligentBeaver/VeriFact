import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import re
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

OUTPUT_DIR = Path("../../storage/outputs/harvest_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
# ========================
# Domain logic
# ========================
STOPWORDS = {
    "the", "and", "of", "in", "to", "with", "for", "on",
    "by", "an", "a", "is", "are", "was", "were", "that"
}

def _normalize_phrase_for_dedupe(s: str) -> str:
    """Normalize a clause for deduping: lowercase, remove punctuation/hyphens, collapse spaces."""
    s = s.lower().strip()
    # remove punctuation (but keep spaces and alphanumerics)
    s = re.sub(r"[-_\\/&]", " ", s)         # convert hyphens/slashes/& to spaces first
    s = re.sub(r"[^a-z0-9\s]", "", s)      # remove any remaining non-alphanumeric
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def generate_ngrams(tokens, min_n=2, max_n=4):
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            chunk = tokens[i:i+n]
            if not any(t in STOPWORDS for t in chunk):
                ngrams.append(" ".join(chunk))
    return ngrams

def get_embedding(text):
    """Generates a vector embedding using SapBERT."""
    import torch
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def extract_bio_phrases(label, scope_note, top_k=3):
    """
    Extract biologically relevant phrases using:
    - sentence splitting
    - n-gram generation
    - semantic similarity to label
    """
    if not scope_note or not isinstance(scope_note, str):
        print(f"⚠️ Warning: 'scope_note' for {label} is missing or empty.")
        return []

    sentences = re.split(r"[.;]", scope_note)
    candidates = []

    for sent in sentences:
        tokens = clean_text(sent).split()
        if len(tokens) < 2:
            continue
        candidates.extend(generate_ngrams(tokens))

    # Deduplicate
    candidates = list(set(candidates))
    if not candidates:
        return []

    # Semantic ranking
    label_vec = get_embedding(label)
    candidate_vecs = np.concatenate([get_embedding(c) for c in candidates])

    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(label_vec, candidate_vecs)[0]

    top_indices = sims.argsort()[-top_k:][::-1]
    return [candidates[i] for i in top_indices]


def generate_harvester_matrix_with_scoring(seed_data, top_k_keywords=3, max_results=10):
    """
    Generates a harvest list with semantic scoring, deduplicates similar queries,
    and returns at most `max_results` entries (keeps highest scoring).
    """
    label = seed_data.get('label', 'Unknown')
    scope_note = seed_data.get('semantic_ground_truth', "")
    qualifiers = seed_data.get('harvester_modifiers', [])
    synonyms = [s.lower() for s in seed_data.get('synonyms', [])]
    mest_parent = seed_data.get('MESTry_parent', None)
    #
    # --- Tier 1: Identity queries ---
    identity_query = f'"{label}"[Mesh]'
    identity_queries = [identity_query]

    # --- Tier 2: Action queries ---
    action_queries = [f'"{label} {q}"' for q in qualifiers[:5]]

    # --- Tier 3: Context queries ---
    bio_keywords = extract_bio_phrases(label, scope_note, top_k=top_k_keywords)
    context_queries = [f'"{label}" AND "{bk}"' for bk in bio_keywords]
    context_queries += [f'"{label}" AND "{s}"' for s in synonyms]

    for q in qualifiers[:5]:
        for bk in bio_keywords:
            context_queries.append(f'"{label}" AND "{bk}" AND "{q}"')

    if mest_parent:
        context_queries.append(f'"{label}" AND "{mest_parent}"')

    # --- Semantic scoring for every candidate query ---
    all_queries = identity_queries + action_queries + context_queries
    label_vec = get_embedding(label)

    scored_queries = []
    for q in all_queries:
        try:
            q_vec = get_embedding(q)
            score = float(cosine_similarity(label_vec, q_vec)[0][0])
        except Exception as e:
            # if embedding fails for some query, skip it but keep processing others
            print(f"Embedding error for query: {q} -> {e}")
            continue
        scored_queries.append((q, score))

    # sort by descending score
    scored_queries.sort(key=lambda x: x[1], reverse=True)

    # --- Deduplicate using normalized sets of *other* clauses (exclude the label itself) ---
    label_norm = _normalize_phrase_for_dedupe(label)
    seen_keys = set()
    final_list = []

    # Ensure identity_query is present (and reserve its spot at the top if it exists in scored list)
    # Find identity_query in scored list and keep it as highest priority (if present).
    for i, (q, s) in enumerate(scored_queries):
        if q == identity_query:
            final_list.append((q, s))
            seen_keys.add(())  # empty tuple key meaning "no other clause"
            break

    # iterate scored queries in order and add deduped ones
    for q, s in scored_queries:
        # skip the identity query already pushed
        if q == identity_query:
            continue

        # extract quoted clauses from the query
        clauses = re.findall(r'"([^"]+)"', q)
        if not clauses:
            # fallback: treat entire query as single clause if no quoted part
            clauses = [q]

        # normalize and remove label itself from clause list
        other_norms = []
        for c in clauses:
            n = _normalize_phrase_for_dedupe(c)
            if n and n != label_norm:
                other_norms.append(n)

        # create dedupe key: sorted tuple of unique other clause norms
        key = tuple(sorted(set(other_norms)))

        # If key already seen, skip
        if key in seen_keys:
            continue

        # If key is empty (means only label, e.g., '"label"' or '"label" AND "label"'), skip because identity already present
        if key == ():
            # identity already included or redundant
            continue

        # Add to results
        final_list.append((q, s))
        seen_keys.add(key)

        # stop if we have enough
        if len(final_list) >= max_results:
            break

    # If we don't yet have max_results and identity wasn't present earlier,
    # ensure we include the identity (fallback)
    if len(final_list) < max_results and identity_query not in [q for q, _ in final_list]:
        # find identity in scored queries (if present) and add it at front
        for q, s in scored_queries:
            if q == identity_query:
                final_list.insert(0, (q, s))
                break

    # final trim (in case identity insertion went over limit)
    final_list = final_list[:max_results]

    return {"seed": label, "harvest_list": final_list}


# ========================
# Parallel engine
# ========================
def safe_filename(name: str) -> str:
    # Make filenames OS-safe
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

def _worker(seed):
    label = seed.get("label", "Unknown")
    filename = safe_filename(label) + ".json"
    output_path = OUTPUT_DIR / filename

    try:
        result = generate_harvester_matrix_with_scoring(seed)
        payload = {"label": label, "result": result, "error": None}
    except Exception:
        payload = {
            "label": label,
            "result": None,
            "error": traceback.format_exc()
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return label

def run_parallel(seeds, max_workers=None):
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    total = len(seeds)
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, seed) for seed in seeds]

        for future in as_completed(futures):
            label = future.result()
            completed += 1
            print(f"[{completed}/{total}] finished → {label}", flush=True)

# ========================
# CLI entry point
# ========================
def main(input_json):
    with open(input_json, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    run_parallel(seeds)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python harvest_parallel.py seeds.json")

    main(sys.argv[1])
