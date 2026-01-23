"""
Interactive search + FAISS retrieval + Cross-Encoder rerank script.

Inputs (directory with files produced by build_faiss_index.py):
 - index.faiss
 - embeddings.npy
 - metadata.json
 - (optional) sapbert_embeddings.npy

Flow:
 1. Load FAISS index and metadata (metadata order must match embeddings order used to build the index).
 2. Embed the user query using the same embedding model used to build the index.
 3. Perform FAISS search to get top-K passages (returns FAISS inner-product scores if the index is IP).
 4. Optionally compute SapBERT cosine similarities between query sapbert vector and sapbert_embeddings to filter/boost.
 5. Re-rank top candidates using a Cross-Encoder and present FAISS score, Cross-Encoder score, trust score, optional sapbert score, and final combined score.

Usage example:
python search_and_rerank.py --index-dir ./index_output --embedding-model all-mpnet-base-v2 --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 --topk 100 --rerank 20

Requirements:
pip install sentence-transformers faiss-cpu numpy tqdm python-dateutil

"""

import argparse
import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm
from dateutil import parser as date_parser
from datetime import datetime
import math


def load_metadata(metadata_path: Path):
    with metadata_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_faiss_index(index_path: Path):
    idx = faiss.read_index(str(index_path))
    return idx


def embed_query(query: str, model: SentenceTransformer, normalize=True):
    v = model.encode([query], convert_to_numpy=True)[0]
    if normalize:
        n = np.linalg.norm(v)
        if n == 0:
            return v
        v = v / n
    return v


def cosine(a, b):
    # a, b are 1D numpy arrays
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def faiss_search(index, query_vec, topk=100):
    q = np.expand_dims(query_vec.astype(np.float32), axis=0)
    scores, indices = index.search(q, topk)
    # faiss returns scores shape (1, k) and indices (1, k)
    return scores[0], indices[0]


def compute_trust_score(meta_item):
    """Compute a simple trust score in [0,1] based on metadata heuristics.

    Heuristics used (simple and tunable):
      - medically_reviewed_by present: +0.5
      - number of reputable sources: scaled up to +0.3
      - recency: published within last 3 years -> up to +0.2
    """
    score = 0.0
    if meta_item.get('medically_reviewed_by'):
        score += 0.5

    sources = meta_item.get('sources') or []
    try:
        src_count = len(sources)
    except Exception:
        src_count = 0
    score += min(src_count / 3.0, 0.3)

    # recency: newer gets higher score (exponential decay over years)
    pub = meta_item.get('published_date')
    if pub:
        try:
            dt = date_parser.parse(pub)
            days = (datetime.utcnow() - dt).days
            years = days / 365.25
            # decay: 0 years -> +0.2, 5 years -> ~0.0
            recency = max(0.0, 0.2 * math.exp(-years / 2.0))
            score += recency
        except Exception:
            pass

    # clamp
    return max(0.0, min(1.0, score))


def rerank_with_cross_encoder(query, candidates_texts, cross_encoder_model_name):
    model = CrossEncoder(cross_encoder_model_name)
    pairs = [[query, t] for t in candidates_texts]
    scores = model.predict(pairs)
    return scores


def interactive_loop(index_dir: Path, embedding_model_name: str, cross_encoder_model_name: str,
                     sapbert_path: Path = None, sapbert_threshold: float = None,
                     sapbert_boost: float = 0.1, topk: int = 100, rerank_k: int = 20):

    index_path = index_dir / 'index.faiss'
    metadata_path = index_dir / 'metadata.json'

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError('index.faiss and metadata.json must exist in index-dir')

    print('Loading metadata...')
    metadata = load_metadata(metadata_path)
    print(f'Loaded {len(metadata)} metadata entries')

    print('Loading FAISS index...')
    index = load_faiss_index(index_path)

    print(f'Loading embedding model {embedding_model_name}...')
    embed_model = SentenceTransformer(embedding_model_name)

    sapbert_embeddings = None
    if sapbert_path and sapbert_path.exists():
        print('Loading optional SapBERT embeddings...')
        sapbert_embeddings = np.load(str(sapbert_path))
        if sapbert_embeddings.shape[0] != len(metadata):
            print('Warning: sapbert_embeddings length does not match metadata length')

    print('Interactive search ready — type your query (or Ctrl+C to exit)')

    while True:
        try:
            query = input('\nQuery> ').strip()
            if not query:
                print('Empty query; try again')
                continue

            # 1) embed query
            qvec = embed_query(query, embed_model, normalize=True)

            # 2) FAISS search
            faiss_scores, faiss_idxs = faiss_search(index, qvec, topk=topk)

            # Collect candidate metadata and texts
            candidates = []
            for score, idx in zip(faiss_scores, faiss_idxs):
                if idx < 0 or idx >= len(metadata):
                    continue
                m = metadata[idx]
                candidates.append({'idx': idx, 'faiss_score': float(score), 'meta': m})

            # 3) Optional SapBERT scoring/ filtering
            if sapbert_embeddings is not None:
                # Attempt to load a SapBERT-like encoder using the same embedding_model_name for query
                # or assume user will provide query sapbert vector via same model name.
                # We'll compute a sapbert query vector by encoding the query with the same embedding
                # model as used to create sapbert vectors, if possible. If not available, skip.
                # Here we assume sapbert_embeddings are normalized.
                try:
                    query_sap_vec = embed_model.encode([query], convert_to_numpy=True)[0]
                    # normalize
                    n = np.linalg.norm(query_sap_vec)
                    if n != 0:
                        query_sap_vec = query_sap_vec / n
                    # compute cosine to candidates
                    for c in candidates:
                        idx = c['idx']
                        if idx < sapbert_embeddings.shape[0]:
                            s = float(np.dot(query_sap_vec, sapbert_embeddings[idx]))
                        else:
                            s = 0.0
                        c['sapbert_score'] = s
                    # optional hard filter
                    if sapbert_threshold is not None:
                        candidates = [c for c in candidates if c.get('sapbert_score', 0.0) >= sapbert_threshold]
                        if not candidates:
                            print('No candidates remain after SapBERT filtering')
                            continue
                except Exception as e:
                    print('SapBERT scoring failed:', e)

            # 4) Rerank top rerank_k with cross-encoder
            rerank_candidates = candidates[:rerank_k]
            texts = [c['meta']['text'] for c in rerank_candidates]
            if not texts:
                print('No candidates to rerank')
                continue

            print('Running cross-encoder rerank (this may take a little time)...')
            ce_scores = rerank_with_cross_encoder(query, texts, cross_encoder_model_name)

            # attach rerank scores and compute final combined score
            for c, ce in zip(rerank_candidates, ce_scores):
                c['cross_score'] = float(ce)
                # compute trust score
                c['trust_score'] = compute_trust_score(c['meta'])
                # optional sapbert boost
                sap = c.get('sapbert_score', 0.0)
                # combine: weights (tunable)
                final = 0.7 * c['cross_score'] + 0.2 * c['trust_score'] + 0.1 * sap
                c['final_score'] = final

            # sort by final_score
            rerank_candidates.sort(key=lambda x: x['final_score'], reverse=True)

            # Display results
            print('\nTop results (after rerank):')
            for rank, c in enumerate(rerank_candidates[:min(20, len(rerank_candidates))], 1):
                m = c['meta']
                print(f"\n{rank}. final={c['final_score']:.4f} cross={c['cross_score']:.4f} faiss={c['faiss_score']:.4f} trust={c['trust_score']:.3f} sapbert={c.get('sapbert_score',0.0):.3f}")
                print(f"   passage_id: {m.get('passage_id')} | title: {m.get('title')} | url: {m.get('url')}")
                # print short snippet (first 240 chars)
                snippet = m.get('text','')
                snippet = ' '.join(snippet.split())
                print('   snippet:', (snippet[:240] + '...') if len(snippet) > 240 else snippet)

        except KeyboardInterrupt:
            print('\nExiting.')
            break
        except Exception as e:
            print('Search failed:', e)


INDEX_DIR="../storage/outputs/webmd/faiss"
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index-dir', type=str, default=INDEX_DIR, help='Directory containing index.faiss, metadata.json, embeddings.npy')
    p.add_argument('--embedding-model', type=str, default='all-mpnet-base-v2', help='SentenceTransformer model name for query embedding')
    p.add_argument('--cross-encoder-model', type=str, default='cross-encoder/ms-marco-MiniLM-L-6-v2', help='Cross-Encoder model name for reranking')
    p.add_argument('--sapbert-embeddings', type=str, default=None, help='Optional path to sapbert_embeddings.npy')
    p.add_argument('--sapbert-threshold', type=float, default=None, help='Optional hard threshold for sapbert cosine filter')
    p.add_argument('--sapbert-boost', type=float, default=0.1, help='Amount to include sapbert score in final score')
    p.add_argument('--topk', type=int, default=100, help='Number of passages to retrieve from FAISS')
    p.add_argument('--rerank', type=int, default=20, help='Number of top FAISS results to rerank with cross-encoder')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    interactive_loop(
        index_dir=Path(args.index_dir),
        embedding_model_name=args.embedding_model,
        cross_encoder_model_name=args.cross_encoder_model,
        sapbert_path=Path(args.sapbert_embeddings) if args.sapbert_embeddings else None,
        sapbert_threshold=args.sapbert_threshold,
        sapbert_boost=args.sapbert_boost,
        topk=args.topk,
        rerank_k=args.rerank,
    )
