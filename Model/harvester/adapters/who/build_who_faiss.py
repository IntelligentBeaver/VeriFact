#!/usr/bin/env python3
"""
FAISS + Cross-Encoder retrieval for WHO dataset.

Behavior:
 - If an existing index (index.faiss) + metadata.json exist in --output-dir,
   the script loads them and **does not** rebuild unless --rebuild is passed.
 - If --input-file or --input-dir is provided AND --rebuild is passed, the script rebuilds.
 - Models (embedder + cross-encoder) are loaded once and reused.
 - Device autodetect: use CUDA if available; override with --device.
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
# fact unit extraction
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
# global model/index cache
# -----------------------
GLOBAL = {
    "device": None,
    "embedder": None,
    "cross_encoder": None,
    "index": None,
    "metadata": None
}

def get_device(override=None):
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"

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
# build + embed pipeline (sentence-transformers)
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
# search demo + cross-encoder rerank
# -----------------------
def search_index_sentence_transformer(output_dir, query, model_name=MODEL_NAME,
                                      cross_encoder_model=CROSS_ENCODER_MODEL,
                                      top_k_retrieve=TOP_K_RETRIEVE, top_k_rerank=TOP_K_RERANK,
                                      collapse_by_url_flag=COLLAPSE_BY_URL, no_rerank=False,
                                      device=None):
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

    # embed query
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    if USE_COSINE:
        faiss.normalize_L2(q_emb)

    # FAISS retrieve (first stage)
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
            # continue: user may still want to run build later

    # prepare models (embedder + cross encoder) lazily but show loading message now
    # We don't force loading cross-encoder if user wants to skip rerank
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
            device=device
        )
        for r in results:
            meta = r["metadata"]
            print(f"\nfaiss_score={r.get('faiss_score',0):.4f} rerank_score={r.get('rerank_score',0):0.4f} url={meta['source_url']}")
            print("kind:", meta.get("kind"), "heading:", meta.get("section_heading"))
            print("text:", meta.get("text")[:400].replace("\n", " "))