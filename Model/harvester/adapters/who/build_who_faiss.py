#!/usr/bin/env python3
"""
FAISS + Cross-Encoder retrieval for WHO dataset with optional SapBERT filtering.

See original script for base behavior. New behavior:
 - If --use-sapbert is set, the query is first embedded with SapBERT and mapped to top MeSH concepts.
 - A concept->doc_id inverted map is loaded (or built once) to restrict FAISS retrieval to doc ids
   associated with those concepts.
"""
import os
import json
import argparse
import hashlib
from glob import glob
from tqdm import tqdm

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# -----------------------
# CONFIG - change here
# -----------------------
INPUT_FILE = "../../storage/who/news/2025.json"
INPUT_DIR = "../../storage/who/news"
OUTPUT_DIR="../../storage/outputs/who/faiss"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # embedder
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # reranker
MIN_CHARS = 30
CHUNK_MAX_CHARS = 750
EMBED_BATCH_SIZE = 64
USE_COSINE = True

# SapBERT / concept config
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CONCEPT_JSON = "../../storage/concepts.json"  # path to your MeSH concept JSON
CONCEPT_DOCMAP_FN = "concept_doc_map.json"  # will be saved in OUTPUT_DIR
SAPBERT_BATCH = 64
SAPBERT_TOP_K_CONCEPTS = 6
SAPBERT_CONCEPT_THRESHOLD = 0.60  # cosine threshold to include a concept for a doc or query

# Retrieval/rerank params (defaults)
TOP_K_RETRIEVE = 100
TOP_K_RERANK = 20
COLLAPSE_BY_URL = True

# -----------------------
# helpers
# -----------------------
def sha1(s: str):
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def list_input_files(input_dir=None, input_file=None):
    files = []
    if input_file:
        files.append(input_file)
    if input_dir:
        pattern = os.path.join(input_dir, "*.json")
        files.extend(sorted(glob(pattern)))
    return files

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_whitespace(text: str):
    return " ".join(text.split())

def chunk_text_by_chars(text: str, max_chars=CHUNK_MAX_CHARS):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    pieces = []
    for part in text.split(". "):
        part = part.strip()
        if not part:
            continue
        if not part.endswith("."):
            part = part + "."
        pieces.append(part)
    if not pieces:
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    chunks = []
    cur = ""
    for p in pieces:
        if len(cur) + 1 + len(p) <= max_chars:
            cur = (cur + " " + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks

# -----------------------
# fact unit extraction (unchanged)
# -----------------------
def extract_fact_units_from_article(article):
    units = []
    url = article.get("url")
    title = article.get("title")
    date = article.get("date")
    art_type = article.get("type")

    def add_unit(text, kind, heading=None, sec_idx=None, para_idx=None):
        if not text:
            return
        text = normalize_whitespace(text)
        if len(text) < MIN_CHARS:
            return
        uid = sha1(f"{url}||{kind}||{heading or ''}||{sec_idx}||{para_idx}||{text[:120]}")
        units.append({
            "id": uid,
            "text": text,
            "source_url": url,
            "title": title,
            "date": date,
            "section_heading": heading,
            "type": art_type,
            "kind": kind
        })

    if title:
        add_unit(title, kind="title", heading=None, sec_idx=None, para_idx=0)
    if article.get("lead"):
        add_unit(article.get("lead"), kind="lead")

    content = article.get("content") or []
    for sec_idx, section in enumerate(content):
        heading = section.get("heading")
        sec_content = section.get("content") or []
        if isinstance(sec_content, str):
            sec_content = [sec_content]
        for para_idx, paragraph in enumerate(sec_content):
            add_unit(paragraph, kind="paragraph", heading=heading, sec_idx=sec_idx, para_idx=para_idx)
        for b_idx, b in enumerate(section.get("bullets") or []):
            add_unit(b, kind="bullet", heading=heading, sec_idx=sec_idx, para_idx=b_idx)

    for b_idx, b in enumerate(article.get("bullets") or []):
        add_unit(b, kind="bullet", heading=None, sec_idx="top", para_idx=b_idx)

    for r_idx, r in enumerate(article.get("references") or []):
        text = r.get("text") or r.get("url") or ""
        add_unit(text, kind="reference", sec_idx="refs", para_idx=r_idx)

    return units

# -----------------------
# global model/index/cache
# -----------------------
GLOBAL = {
    "device": None,
    "embedder": None,
    "cross_encoder": None,
    "index": None,
    "metadata": None,
    # sapbert related
    "sapbert_tokenizer": None,
    "sapbert_model": None,
    "concept_cache": None,            # dict of concept_id -> {centroid, entry}
    "concept_doc_map": None,          # dict of concept_id -> [doc_idx,...]
    "sapbert_device": None
}

def get_device(override=None):
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load concept JSON + precomputed candidate embeddings (fast)
# -----------------------
def load_concept_cache(concept_json_path):
    """
    Loads the MeSH concept JSON and the candidate/label embedding .npy refs (if present),
    computes centroids for each concept and caches them for fast similarity.
    """
    if GLOBAL["concept_cache"] is not None:
        return GLOBAL["concept_cache"]

    concepts = load_json_file(concept_json_path)
    cache = {}
    for cid, entry in concepts.items():
        meta = entry.get("sapbert_metadata", {})
        cand_ref = meta.get("candidate_embeddings_ref")
        label_ref = meta.get("label_embedding_ref")

        cand_emb = None
        label_emb = None
        if cand_ref and os.path.exists(cand_ref):
            cand_emb = np.load(cand_ref).astype("float32")
        if label_ref and os.path.exists(label_ref):
            label_emb = np.load(label_ref).astype("float32")

        # normalize
        if cand_emb is not None:
            norms = np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-12
            cand_emb = cand_emb / norms
            centroid = cand_emb.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        elif label_emb is not None:
            if label_emb.ndim == 1:
                centroid = label_emb / (np.linalg.norm(label_emb) + 1e-12)
            else:
                norms = np.linalg.norm(label_emb, axis=1, keepdims=True) + 1e-12
                label_emb = label_emb / norms
                centroid = label_emb.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        else:
            centroid = None

        cache[cid] = {
            "entry": entry,
            "candidate_embeddings": cand_emb,
            "label_embeddings": label_emb,
            "centroid": centroid
        }
    GLOBAL["concept_cache"] = cache
    return cache

# -----------------------
# SapBERT model utils (embed queries / batch embed)
# -----------------------
from transformers import AutoTokenizer, AutoModel

def prepare_sapbert(model_name=SAPBERT_MODEL, device="cpu"):
    if GLOBAL["sapbert_model"] is None or GLOBAL["sapbert_device"] != device:
        print(f"Loading SapBERT {model_name} on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        GLOBAL["sapbert_tokenizer"] = tokenizer
        GLOBAL["sapbert_model"] = model
        GLOBAL["sapbert_device"] = device
    return GLOBAL["sapbert_tokenizer"], GLOBAL["sapbert_model"]

def sapbert_embed_texts(texts, device="cpu", batch_size=SAPBERT_BATCH):
    """Embed a list of texts with SapBERT, returns (N,dim) float32 unit-normalized vectors"""
    tok, model = prepare_sapbert(device=device)
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tok(batch, truncation=True, padding=True, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            token_mask = inputs['attention_mask'].unsqueeze(-1)
            seq_emb = (out.last_hidden_state * token_mask).sum(1) / token_mask.sum(1)
            vecs = seq_emb.cpu().numpy().astype("float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
            all_vecs.append(vecs)
    if not all_vecs:
        return np.zeros((0, model.config.hidden_size), dtype="float32")
    return np.vstack(all_vecs)

# -----------------------
# Build or load concept->doc mapping
# -----------------------
def build_or_load_concept_doc_map(output_dir, metadata, concept_json_path, device="cpu",
                                  force_rebuild=False, threshold=SAPBERT_CONCEPT_THRESHOLD):
    """
    Creates a mapping: concept_id -> list(doc_idx) by embedding each metadata['text'] using SapBERT
    and matching against concept centroids. Saves map to OUTPUT_DIR/CONCEPT_DOCMAP_FN
    """
    out_path = os.path.join(output_dir, CONCEPT_DOCMAP_FN)
    if os.path.exists(out_path) and not force_rebuild:
        print("Loading existing concept->doc map from", out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            cmap = json.load(f)
        # convert keys to str and values to lists of ints (they are saved as ints already)
        for k,v in cmap.items():
            cmap[k] = [int(x) for x in v]
        GLOBAL["concept_doc_map"] = cmap
        return cmap

    print("Building concept->doc map with SapBERT. This may take a while (one-time).")
    concept_cache = load_concept_cache(concept_json_path)
    # build centroid matrix and id list
    cids = []
    cents = []
    for cid, info in concept_cache.items():
        if info["centroid"] is not None:
            cids.append(cid)
            cents.append(info["centroid"])
    if not cents:
        raise ValueError("No concept centroids found in concept cache.")
    cent_mat = np.vstack(cents).astype("float32")  # shape (C, dim)
    # embed all unit texts
    texts = [m["text"] for m in metadata]
    unit_vecs = sapbert_embed_texts(texts, device=device, batch_size=SAPBERT_BATCH)
    # compute similarity matrix in batches (to save memory)
    cmap = {cid: [] for cid in cids}
    B = 256  # batch size for similarity
    for i in tqdm(range(0, unit_vecs.shape[0], B), desc="Matching units -> concepts"):
        batch_vecs = unit_vecs[i:i+B]  # (b, dim)
        sim = np.dot(batch_vecs, cent_mat.T)  # (b, C)
        # for each unit, find concepts above threshold
        for j in range(sim.shape[0]):
            idxs = np.where(sim[j] >= threshold)[0]
            for idx in idxs:
                cid = cids[idx]
                cmap[cid].append(i + j)
    # also ensure concept ids that had none are present as empty lists
    for cid in list(concept_cache.keys()):
        if cid not in cmap:
            cmap[cid] = []
    # save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cmap, f, ensure_ascii=False)
    GLOBAL["concept_doc_map"] = cmap
    print("Saved concept->doc map to", out_path)
    return cmap

# -----------------------
# Utility: get top-K concepts for a query (use centroids)
# -----------------------
def get_top_concepts_for_query(query, top_k=SAPBERT_TOP_K_CONCEPTS, device="cpu"):
    concept_cache = load_concept_cache(CONCEPT_JSON)
    # build centroid arrays if not already computed once
    if "centroid_matrix" not in GLOBAL:
        cids = []
        cents = []
        for cid, info in concept_cache.items():
            if info["centroid"] is not None:
                cids.append(cid)
                cents.append(info["centroid"])
        GLOBAL["centroid_ids"] = cids
        GLOBAL["centroid_matrix"] = np.vstack(cents).astype("float32") if cents else np.zeros((0,768),dtype="float32")
    # embed query
    qv = sapbert_embed_texts([query], device=device, batch_size=1)[0]
    if GLOBAL["centroid_matrix"].size == 0:
        return []
    sims = np.dot(GLOBAL["centroid_matrix"], qv)  # (C,)
    order = np.argsort(-sims)[:top_k]
    top = []
    for idx in order:
        cid = GLOBAL["centroid_ids"][idx]
        top.append((cid, float(sims[idx])))
    return top

# -----------------------
# load + prepare models (unchanged)
# -----------------------
def load_index_and_metadata(output_dir):
    """Load existing FAISS index + metadata + embeddings if present."""
    index_path = os.path.join(output_dir, "index.faiss")
    metadata_path = os.path.join(output_dir, "metadata.json")
    np_path = os.path.join(output_dir, "embeddings.npy")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None, None

    print("Loading FAISS index from", index_path)
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = None
    if os.path.exists(np_path):
        try:
            embeddings = np.load(np_path)
        except Exception:
            embeddings = None
    return index, metadata, embeddings

def prepare_models(model_name, cross_model_name, device):
    """Instantiate and cache embedder and cross-encoder on `device`."""
    if GLOBAL["embedder"] is None:
        print(f"Loading embedder {model_name} (device={device})...")
        GLOBAL["embedder"] = SentenceTransformer(model_name, device=device)
    if GLOBAL["cross_encoder"] is None:
        print(f"Loading cross-encoder {cross_model_name} (device={device})...")
        GLOBAL["cross_encoder"] = CrossEncoder(cross_model_name, device=device)
    return GLOBAL["embedder"], GLOBAL["cross_encoder"]

def load_articles_from_file(path):
    """
    Accepts JSON in three forms:
      - list of dicts
      - single dict (fact-sheet)
      - mapping slug->dict
    Returns a list of dicts (articles).
    """
    data = load_json_file(path)
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    elif isinstance(data, dict):
        # Single article (like abortion.json)
        if any(k in data for k in ("url", "title", "sections", "content", "published_date")):
            return [data]
        # mapping slug->article dict
        vals = [v for v in data.values() if isinstance(v, dict) and any(k in v for k in ("url","title","sections","content"))]
        return vals
    else:
        return []

# -----------------------
# build + embed pipeline (unchanged) -- but note: maintain same metadata format
# -----------------------
def build_index_sentence_transformer(input_dir=None, input_file=None,
                                     output_dir="storage/who/faiss_mpnet", model_name=MODEL_NAME,
                                     device=None):
    os.makedirs(output_dir, exist_ok=True)
    files = list_input_files(input_dir=input_dir, input_file=input_file)
    if not files:
        raise ValueError("No input files found. Provide --input-file or --input-dir")

    embedder = SentenceTransformer(model_name, device=device)
    print("Embedding model loaded on device:", device)

    # load articles and extract units
    all_units = []
    for f in files:
        print("Loading", f)
        articles = load_articles_from_file(f)
        if not articles:
            print(f"Warning: no articles found in {f} - skipping")
            continue
        for art in articles:
            units = extract_fact_units_from_article(art)
            all_units.extend(units)

    print(f"Extracted {len(all_units)} fact units (before dedupe)")

    # dedupe by id
    unique = {}
    for u in all_units:
        unique[u["id"]] = u
    units = list(unique.values())
    texts = [u["text"] for u in units]

    print(f"{len(units)} unique units to embed")

    # chunk all texts into chunks; keep mapping to units
    all_chunks = []
    chunk_owner = []  # chunk_owner[i] = index of unit owning this chunk
    for idx, t in enumerate(texts):
        chunks = chunk_text_by_chars(t, max_chars=CHUNK_MAX_CHARS)
        if not chunks:
            chunks = [t]
        for c in chunks:
            all_chunks.append(c)
            chunk_owner.append(idx)

    print(f"Total chunks to embed: {len(all_chunks)}")

    # embed chunks in batches
    embeddings_chunks = []
    for i in tqdm(range(0, len(all_chunks), EMBED_BATCH_SIZE), desc="Encoding chunks"):
        batch = all_chunks[i:i+EMBED_BATCH_SIZE]
        embs = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings_chunks.append(embs)
    embeddings_chunks = np.vstack(embeddings_chunks).astype("float32")

    # average chunk embeddings per original unit
    dim = embeddings_chunks.shape[1]
    unit_embeddings = np.zeros((len(units), dim), dtype="float32")
    counts = np.zeros((len(units),), dtype="int32")
    for i, owner in enumerate(chunk_owner):
        unit_embeddings[owner] += embeddings_chunks[i]
        counts[owner] += 1
    for i in range(len(units)):
        if counts[i] > 0:
            unit_embeddings[i] /= counts[i]
        else:
            unit_embeddings[i] = np.zeros((dim,), dtype="float32")

    if USE_COSINE:
        faiss.normalize_L2(unit_embeddings)

    # build FAISS index
    if USE_COSINE:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    print("Adding vectors to FAISS index...")
    index.add(unit_embeddings)
    print("Index size:", index.ntotal)

    index_path = os.path.join(output_dir, "index.faiss")
    metadata_path = os.path.join(output_dir, "metadata.json")
    np_path = os.path.join(output_dir, "embeddings.npy")

    faiss.write_index(index, index_path)
    np.save(np_path, unit_embeddings)
    print("Saved FAISS index to", index_path, "and embeddings to", np_path)

    metadata = []
    for u in units:
        metadata.append({
            "id": u["id"],
            "text": u["text"],
            "source_url": u["source_url"],
            "title": u["title"],
            "date": u.get("date"),
            "section_heading": u.get("section_heading"),
            "type": u.get("type"),
            "kind": u.get("kind")
        })
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("Saved metadata to", metadata_path)

    # cache
    GLOBAL["index"] = index
    GLOBAL["metadata"] = metadata
    return index, metadata

# -----------------------
# helper: collapse duplicates by URL (keep best faiss score)
# -----------------------
def collapse_by_url(candidates):
    best = {}
    for c in candidates:
        url = c["metadata"]["source_url"]
        if url not in best or c["faiss_score"] > best[url]["faiss_score"]:
            best[url] = c
    return sorted(best.values(), key=lambda x: x["faiss_score"], reverse=True)

# -----------------------
# FAISS restricted search util
# -----------------------
def faiss_search_restricted(index, query_vec, allowed_doc_ids: set, top_k=20, search_k_multiplier=10):
    """
    Do an initial FAISS search for search_k (top_k*multiplier), then filter by allowed_doc_ids
    and return top_k hits. query_vec must be (1,dim) np.float32 and normalized if using cosine.
    """
    search_k = min(index.ntotal, max(top_k * search_k_multiplier, 200))
    D, I = index.search(query_vec, search_k)
    hits = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        if idx in allowed_doc_ids:
            hits.append((idx, float(dist)))
            if len(hits) >= top_k:
                break
    return hits

# -----------------------
# search demo + cross-encoder rerank (updated to include SapBERT option)
# -----------------------
def search_index_sentence_transformer(output_dir, query, model_name=MODEL_NAME,
                                      cross_encoder_model=CROSS_ENCODER_MODEL,
                                      top_k_retrieve=TOP_K_RETRIEVE, top_k_rerank=TOP_K_RERANK,
                                      collapse_by_url_flag=COLLAPSE_BY_URL, no_rerank=False,
                                      device=None, use_sapbert=True, rebuild_concept_map=False):
    # lazy load index + metadata into GLOBAL cache
    if GLOBAL["index"] is None or GLOBAL["metadata"] is None:
        idx, md, _ = load_index_and_metadata(output_dir)
        if idx is None or md is None:
            raise FileNotFoundError("Index or metadata not found. Build index first or pass --input-file/--input-dir with --rebuild.")
        GLOBAL["index"] = idx
        GLOBAL["metadata"] = md

    # lazy load models
    if GLOBAL["embedder"] is None or (device and GLOBAL["device"] != device):
        GLOBAL["device"] = device
        GLOBAL["embedder"] = SentenceTransformer(model_name, device=device)

    embedder = GLOBAL["embedder"]
    index = GLOBAL["index"]
    metadata = GLOBAL["metadata"]

    allowed_doc_ids = None
    # SapBERT concept filtering path
    if use_sapbert:
        # load concept cache and build/load concept->doc map
        load_concept_cache(CONCEPT_JSON)  # cache centroids
        # ensure sapbert is prepared on same device as requested for sapbert (we'll use same device)
        sapbert_device = device if device else get_device(None)
        # build or load map (may take a while first run)
        concept_map = build_or_load_concept_doc_map(output_dir, metadata, CONCEPT_JSON, device=sapbert_device, force_rebuild=rebuild_concept_map, threshold=SAPBERT_CONCEPT_THRESHOLD)
        # get top concepts for query
        top_concepts = get_top_concepts_for_query(query, top_k=SAPBERT_TOP_K_CONCEPTS, device=sapbert_device)
        # filter by threshold
        top_concepts = [(c,s) for c,s in top_concepts if s >= SAPBERT_CONCEPT_THRESHOLD]
        if top_concepts:
            # union doc ids
            allowed_set = set()
            for c,s in top_concepts:
                ids = concept_map.get(c, []) or []
                allowed_set.update(ids)
            allowed_doc_ids = allowed_set
            # debug print
            print("SapBERT top concepts:", top_concepts)
            print(f"Allowed doc ids from concepts: {len(allowed_doc_ids)} items")
        else:
            print("No SapBERT concepts matched above threshold for query; falling back to global search.")

    # embed query (for FAISS retrieval)
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    if USE_COSINE:
        faiss.normalize_L2(q_emb)

    # FAISS retrieve (first stage)
    if allowed_doc_ids:
        # restricted search: we search global and filter by allowed set
        hits = faiss_search_restricted(index, q_emb, allowed_doc_ids, top_k=min(top_k_retrieve, index.ntotal))
        candidates = []
        for idx, dist in hits:
            meta = metadata[idx]
            candidates.append({
                "faiss_score": float(dist),
                "metadata": meta
            })
    else:
        retrieve_k = min(top_k_retrieve, index.ntotal)
        D, I = index.search(q_emb, retrieve_k)
        candidates = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = metadata[idx]
            candidates.append({
                "faiss_score": float(dist),
                "metadata": meta
            })

    # optional collapse
    if collapse_by_url_flag:
        candidates = collapse_by_url(candidates)

    # keep top-N for reranking
    rerank_candidates = candidates[:top_k_rerank]

    if not rerank_candidates:
        return []

    # if no rerank requested, return faiss top-K
    if no_rerank:
        return rerank_candidates

    # prepare pairs for cross-encoder: [query, candidate_text]
    pair_texts = [[query, c["metadata"]["text"]] for c in rerank_candidates]

    # lazy load cross-encoder (respect device)
    if GLOBAL["cross_encoder"] is None or (device and GLOBAL["device"] != device):
        GLOBAL["cross_encoder"] = CrossEncoder(cross_encoder_model, device=device)

    cross = GLOBAL["cross_encoder"]
    rerank_scores = cross.predict(pair_texts)  # higher = more relevant

    # attach rerank scores
    for c, s in zip(rerank_candidates, rerank_scores):
        c["rerank_score"] = float(s)

    # sort by rerank_score desc
    reranked = sorted(rerank_candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default=None, help="Directory with yearly JSON files")
    p.add_argument("--input-file", type=str, default=INPUT_FILE, help="Single JSON file")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output dir for FAISS index and metadata")
    p.add_argument("--model-name", type=str, default=MODEL_NAME, help="SentenceTransformer model name")
    p.add_argument("--cross-encoder-model", type=str, default=CROSS_ENCODER_MODEL, help="CrossEncoder model name")
    p.add_argument("--top-k-retrieve", type=int, default=TOP_K_RETRIEVE, help="How many to retrieve from FAISS before reranking")
    p.add_argument("--top-k-rerank", type=int, default=TOP_K_RERANK, help="How many to rerank with cross-encoder")
    p.add_argument("--collapse-by-url", action="store_true", help="Collapse duplicates by source URL before rerank")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild of index from input files")
    p.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    p.add_argument("--device", type=str, default=None, help="Device to use: 'cuda' or 'cpu' (auto-detect if not provided)")
    p.add_argument("--use-sapbert", action="store_true", help="Use SapBERT concept filtering before FAISS retrieval")
    p.add_argument("--rebuild-concept-map", action="store_true", help="Force rebuild of concept->doc map (SapBERT step)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = get_device(args.device)
    print("Using device:", device)

    # if index exists and user didn't request rebuild, just load
    idx_path = os.path.join(args.output_dir, "index.faiss")
    md_path = os.path.join(args.output_dir, "metadata.json")
    index_exists = os.path.exists(idx_path) and os.path.exists(md_path)

    if (args.rebuild or not index_exists) and (args.input_dir or args.input_file):
        # build index (will also populate GLOBAL cache)
        build_index_sentence_transformer(
            input_dir=args.input_dir,
            input_file=args.input_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
            device=device

        )
    else:
        if index_exists:
            print("Loading existing index and metadata from", args.output_dir)
            idx, md, _ = load_index_and_metadata(args.output_dir)
            GLOBAL["index"] = idx
            GLOBAL["metadata"] = md
        else:
            print("No index found. To build, pass --input-file or --input-dir and --rebuild.")

    # prepare models (embedder + cross encoder) lazily but show loading message now
    GLOBAL["device"] = device
    GLOBAL["embedder"] = SentenceTransformer(args.model_name, device=device)
    if not args.no_rerank:
        GLOBAL["cross_encoder"] = CrossEncoder(args.cross_encoder_model, device=device)

    print("\nIndex built / loaded. Demo search (empty query to exit).")
    while True:
        q = input("\nquery> ").strip()
        if not q:
            break
        results = search_index_sentence_transformer(
            args.output_dir,
            q,
            model_name=args.model_name,
            cross_encoder_model=args.cross_encoder_model,
            top_k_retrieve=args.top_k_retrieve,
            top_k_rerank=args.top_k_rerank,
            collapse_by_url_flag=args.collapse_by_url,
            no_rerank=args.no_rerank,
            device=device,
            # CHANGE THIS
            use_sapbert=False,
            rebuild_concept_map=args.rebuild_concept_map
        )
        for r in results:
            meta = r["metadata"]
            print(f"\nfaiss_score={r.get('faiss_score',0):.4f} rerank_score={r.get('rerank_score',0):0.4f} url={meta['source_url']}")
            print("kind:", meta.get("kind"), "heading:", meta.get("section_heading"))
            print("text:", meta.get("text")[:400].replace("\n", " "))
