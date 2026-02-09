import os
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from routes.auth import router as auth_router
from security import hash_password, verify_password

try:
    from sentence_transformers.cross_encoder import CrossEncoder  
except Exception:
    CrossEncoder = None  

from datetime import datetime
from dateutil import parser as date_parser
import math

INDEX_DIR = os.getenv("INDEX_DIR", "./index_output")   
FAISS_INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
SAPBERT_FILENAME = "sapbert_embeddings.npy"  


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


DEFAULT_TOP_K = 100
DEFAULT_RERANK_K = 20
DEFAULT_SAPBERT_BOOST = 0.1


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faiss-rerank-server")

app = FastAPI(title="FAISS Search + Cross-Encoder Rerank", version="1.0.0")
app.include_router(auth_router)

faiss_index = None
index_dir = Path(INDEX_DIR)
metadata = None
embedder = None
cross_encoder = None
sapbert_embeddings = None
model_dim = None



class QueryRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    rerank_k: int = DEFAULT_RERANK_K
    sapbert_threshold: Optional[float] = None  
    sapbert_boost: float = DEFAULT_SAPBERT_BOOST
    use_cross_encoder: bool = True


class ResultItem(BaseModel):
    passage_id: Optional[str]
    title: Optional[str]
    url: Optional[str]
    snippet: Optional[str]

    faiss_score: float
    cross_score: Optional[float] = None
    sapbert_score: Optional[float] = None
    trust_score: float
    final_score: float


class QueryResponse(BaseModel):
    query: str
    results: List[ResultItem]

def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return x / norms


def compute_trust_score(meta_item: dict) -> float:
    """Same heuristics as your interactive script."""
    score = 0.0
    if meta_item.get("medically_reviewed_by"):
        score += 0.5

    sources = meta_item.get("sources") or []
    try:
        src_count = len(sources)
    except Exception:
        src_count = 0
    score += min(src_count / 3.0, 0.3)

    pub = meta_item.get("published_date")
    if pub:
        try:
            dt = date_parser.parse(pub)
            days = (datetime.utcnow() - dt).days
            years = days / 365.25
            recency = max(0.0, 0.2 * math.exp(-years / 2.0))
            score += recency
        except Exception:
            pass

    return max(0.0, min(1.0, score))


@app.on_event("startup")
def startup():
    global faiss_index, metadata, embedder, cross_encoder, sapbert_embeddings, model_dim

    logger.info("Starting server; loading index + metadata + models...")

    # 1) index path
    idx_path = index_dir / FAISS_INDEX_FILENAME
    meta_path = index_dir / METADATA_FILENAME
    sapbert_path = index_dir / SAPBERT_FILENAME

    if not idx_path.exists():
        raise RuntimeError(f"FAISS index not found at {idx_path}. Build index and place it there.")

    if not meta_path.exists():
        raise RuntimeError(f"metadata.json not found at {meta_path} (must be produced by build_faiss_index.py).")

    # load FAISS index
    try:
        faiss_index = faiss.read_index(str(idx_path))
    except Exception as e:
        logger.error("Failed to read FAISS index: %s", e)
        raise RuntimeError(f"Failed to read FAISS index: {e}")

    try:
        idx_dim = int(faiss_index.d)
        ntotal = int(getattr(faiss_index, "ntotal", -1))
        logger.info(f"Loaded FAISS index: dim={idx_dim}, ntotal={ntotal}")
    except Exception:
        logger.warning("Couldn't read index diagnostics (d/ntotal).")

    # load metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"Loaded metadata records: {len(metadata)}")

    # load optional sapbert embeddings
    if sapbert_path.exists():
        try:
            sapbert_embeddings = np.load(str(sapbert_path))
            logger.info(f"Loaded sapbert embeddings shape: {sapbert_embeddings.shape}")
        except Exception as e:
            logger.warning("Failed to load sapbert embeddings: %s. Continuing without.", e)
            sapbert_embeddings = None
    else:
        sapbert_embeddings = None
        logger.info("No sapbert_embeddings.npy found; skipping SapBERT features.")

    # load embedder model
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model_dim = int(embedder.get_sentence_embedding_dimension())
        logger.info(f"Loaded embedding model '{EMBEDDING_MODEL_NAME}', dim={model_dim}")
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        raise RuntimeError(f"Failed to load embedding model: {e}")

  
    if model_dim != int(faiss_index.d):
        raise RuntimeError(
            f"Dimension mismatch: model_dim={model_dim} but index_dim={int(faiss_index.d)}. "
            "Make server EMBEDDING_MODEL_NAME match the model used to build the index or rebuild the index."
        )

    if CrossEncoder is not None:
        try:
            cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info(f"Cross-Encoder loaded: {CROSS_ENCODER_MODEL}")
        except Exception as e:
            logger.warning("Failed to load Cross-Encoder (will skip rerank): %s", e)
            cross_encoder = None
    else:
        cross_encoder = None
        logger.info("CrossEncoder not available in environment; skipping rerank functionality.")


@app.get("/")
def root():
    return {"status": "running", "note": "use /info and POST /query"}


@app.get("/info")
def info():
    if faiss_index is None or embedder is None or metadata is None:
        raise HTTPException(status_code=500, detail="server not fully initialized")

    return {
        "index_dir": str(index_dir),
        "index_dim": int(faiss_index.d),
        "index_ntotal": int(getattr(faiss_index, "ntotal", -1)),
        "model": EMBEDDING_MODEL_NAME,
        "model_dim": model_dim,
        "metadata_count": len(metadata),
        "cross_encoder_loaded": bool(cross_encoder),
        "sapbert_loaded": bool(sapbert_embeddings is not None),
    }


def faiss_search_topk(query_vec: np.ndarray, topk: int):
    """Run faiss search. query_vec: (d,) or (1,d). returns (scores, indices) 1d arrays length k."""
    q = np.asarray(query_vec, dtype=np.float32)
    if q.ndim == 1:
        q = np.expand_dims(q, axis=0)

    idx_classname = faiss_index.__class__.__name__.lower()
    if "indexflatip" in idx_classname or "indexivf" in idx_classname:
        q = l2_normalize_rows(q)
    k = min(topk, int(getattr(faiss_index, "ntotal", topk)))
    scores, indices = faiss_index.search(q, k)
    return scores[0], indices[0]


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # 1) embed query
    try:
        q_emb = embedder.encode([req.query], convert_to_numpy=True, show_progress_bar=False)[0]
    except Exception as e:
        logger.error("Encoding error: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # ensure shape and dim
    if q_emb.ndim == 1:
        q_emb = np.expand_dims(q_emb, axis=0)[0]
    if q_emb.shape[0] != model_dim:
        raise HTTPException(
            status_code=500,
            detail=f"Model produced embedding dim={q_emb.shape[0]}, but index expects {faiss_index.d}"
        )

    faiss_scores, faiss_idxs = faiss_search_topk(q_emb, req.top_k)

    candidates = []
    for sc, idx in zip(faiss_scores.tolist(), faiss_idxs.tolist()):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        candidates.append({
            "idx": idx,
            "meta": m,
            "faiss_score": float(sc)
        })

    if not candidates:
        return QueryResponse(query=req.query, results=[])

    if sapbert_embeddings is not None:
        try:
            q_sap = embedder.encode([req.query], convert_to_numpy=True, show_progress_bar=False)[0]
            # normalize
            q_sap = q_sap / (np.linalg.norm(q_sap) + 1e-10)
            for c in candidates:
                idx = c["idx"]
                if idx < sapbert_embeddings.shape[0]:
                    c["sapbert_score"] = float(np.dot(q_sap, sapbert_embeddings[idx]))
                else:
                    c["sapbert_score"] = 0.0
                    
            if req.sapbert_threshold is not None:
                candidates = [c for c in candidates if c.get("sapbert_score", 0.0) >= req.sapbert_threshold]
                if not candidates:
                    return QueryResponse(query=req.query, results=[])
        except Exception as e:
            logger.warning("SapBERT scoring failed: %s", e)
            # continue without sapbert scores

    # Rerank top rerank_k candidates with cross-encoder if available
    rerank_k = min(req.rerank_k, len(candidates))
    rerank_slice = candidates[:rerank_k]
    texts = [c["meta"].get("text", "") for c in rerank_slice]

    if req.use_cross_encoder and cross_encoder is not None and texts:
        try:
            pairs = [[req.query, t] for t in texts]
            ce_scores = cross_encoder.predict(pairs)
            for c, ce in zip(rerank_slice, ce_scores.tolist()):
                c["cross_score"] = float(ce)
        except Exception as e:
            logger.warning("Cross-encoder rerank failed: %s", e)
            # fallback: don't set cross_score
            for c in rerank_slice:
                c["cross_score"] = None
    else:
        # not using cross-encoder: set cross_score None
        for c in rerank_slice:
            c["cross_score"] = None

    # compute trust score and final_score for rerank_slice
    results_ready = []
    for c in rerank_slice:
        meta = c["meta"]
        faiss_sc = c["faiss_score"]
        cross_sc = c.get("cross_score")
        sap_sc = c.get("sapbert_score", 0.0)
        trust_sc = compute_trust_score(meta)

       
        final = None
        if cross_sc is not None:
            cross_norm = 1.0 / (1.0 + math.exp(-float(cross_sc) / 1.0)) 
            final = 0.7 * cross_norm + 0.2 * trust_sc + req.sapbert_boost * float(sap_sc)
            cross_used = cross_norm
        else:
            idx_classname = faiss_index.__class__.__name__.lower()
            if "indexflatip" in idx_classname or "indexivf" in idx_classname:
                # assume faiss_sc is cosine in [-1,1]
                faiss_norm = (faiss_sc + 1.0) / 2.0
            else:
                faiss_norm = 1.0 / (1.0 + float(abs(faiss_sc)))
            final = 0.7 * faiss_norm + 0.2 * trust_sc + req.sapbert_boost * float(sap_sc)
            cross_used = None

        result = ResultItem(
            passage_id=meta.get("passage_id"),
            title=meta.get("title"),
            url=meta.get("url"),
            snippet=(' '.join(meta.get("text", "").split())[:400]) if meta.get("text") else None,
            faiss_score=float(faiss_sc),
            cross_score=(float(cross_sc) if cross_sc is not None else None),
            sapbert_score=(float(sap_sc) if sap_sc is not None else None),
            trust_score=float(trust_sc),
            final_score=float(final),
        )
        results_ready.append(result)

    # sort by final_score
    results_ready.sort(key=lambda r: r.final_score, reverse=True)

    return QueryResponse(query=req.query, results=results_ready)


# quick run helper
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_with_rerank:app", host="127.0.0.1", port=8000, reload=True)
