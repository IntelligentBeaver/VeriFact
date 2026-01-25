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

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


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
            # compute difference in days as integer
            delta = datetime.utcnow() - dt
            days = delta.total_seconds() / 86400  # 86400 seconds in a day
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
                     sapbert_path: Path = None, sapbert_model_name: str = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
                     sapbert_threshold: float = None, sapbert_boost: float = 0.1,
                     sapbert_concepts_topk: int = 5,
                     topk: int = 100, rerank_k: int = 20):

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

    # pre-load cross-encoder once (re-using repeatedly is faster)
    print(f'Loading cross-encoder {cross_encoder_model_name} (for rerank)...')
    try:
        cross_encoder = CrossEncoder(cross_encoder_model_name)
    except Exception as e:
        print("Warning: failed to load cross-encoder at startup:", e)
        cross_encoder = None

    # --- SapBERT assets (concepts or doc-level)
    sapbert_embeddings = None
    sapbert_mode = None   # 'doc' or 'concept' or None
    sap_model = None
    concept_meta = None   # list/dict with per-concept labels (aligned to sapbert_embeddings for concept-mode)

    if sapbert_path and sapbert_path.exists():
        print('Loading optional SapBERT embeddings...')
        sapbert_embeddings = np.load(str(sapbert_path))
        # quick debug sizes
        print("SapBERT embeddings shape:", sapbert_embeddings.shape)
        if sapbert_embeddings.shape[0] == len(metadata):
            sapbert_mode = 'doc'
            print("SapBERT looks like document-level embeddings (will compute query->doc SapBERT similarity).")
            # Load SapBERT encoder to encode queries in the same space:
            try:
                sap_model = SentenceTransformer(sapbert_model_name)
            except Exception as e:
                print("Failed to load sapbert encoder, doc-level SapBERT scoring will be disabled:", e)
                sap_model = None
        else:
            sapbert_mode = 'concept'
            print("SapBERT is concept embeddings.")
            # attempt to find concept metadata (labels) in same dir as sapbert_path
            cand_meta_names = ['metadata.json', 'combined_metadata.json', 'concepts.json', 'mesh_concepts.json']
            found = None
            for nm in cand_meta_names:
                p = sapbert_path.parent / nm
                if p.exists():
                    found = p
                    break
            if found:
                try:
                    with found.open('r', encoding='utf-8') as f:
                        concept_meta_raw = json.load(f)
                        if isinstance(concept_meta_raw, dict):
                            concept_meta = list(concept_meta_raw.values())
                        else:
                            concept_meta = concept_meta_raw
                    print(f"Loaded concept metadata from {found} (n={len(concept_meta)})")
                except Exception as e:
                    print("Failed to load concept metadata:", e)
                    concept_meta = None
            else:
                print("Concept metadata file not found next to sapbert embeddings; concept labels will be unavailable for expansion.")
                concept_meta = None

            # load sapbert encoder for queries (required for mapping query -> concept space)
            try:
                sap_model = SentenceTransformer(sapbert_model_name)
            except Exception as e:
                print("Failed to load sapbert encoder (needed for concept expansion):", e)
                sap_model = None

    # --- Helper functions for filtering title-like snippets / recovering longer text
    def _longest_text_from_meta(meta):
        """Return the longest plausible text field from a metadata dict."""
        candidates = []
        for k in ('text', 'full_text', 'article_text', 'content', 'body'):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())
        sec = meta.get('sections') or meta.get('paragraphs') or meta.get('chunks')
        if isinstance(sec, (list, tuple)):
            joined = " ".join(str(x) for x in sec if isinstance(x, str) and x.strip())
            if joined:
                candidates.append(joined)
        if not candidates:
            return meta.get('text', '') or ''
        return max(candidates, key=len)

    def _is_title_like(snippet: str, title: str, min_words=6, min_chars=80):
        """
        Heuristic to determine if a snippet is just a title/heading.
        Returns True if snippet is likely a title (so we will try to recover or drop it).
        """
        if not snippet:
            return True
        s = snippet.strip()
        t = (title or "").strip()

        words = s.split()
        # very short -> title-like
        if len(words) < min_words:
            return True

        # if snippet and title strongly overlap
        ls = s.lower()
        lt = t.lower()
        if ls == lt or ls in lt or lt in ls:
            return True

        # no sentence punctuation and short char length -> likely heading
        if all(p not in s for p in ".!?") and len(s) < min_chars:
            return True

        return False

    def _find_longer_from_doc(metadata_list, meta):
        """
        Try to find other chunks from the same document (by common doc id keys)
        and return the longest matching text if any.
        """
        doc_keys = ['document_id', 'doc_id', 'article_id', 'source_id', 'parent_id']
        for k in doc_keys:
            docid = meta.get(k)
            if docid:
                cand_texts = [
                    _longest_text_from_meta(m)
                    for m in metadata_list
                    if m.get(k) == docid and _longest_text_from_meta(m)
                ]
                if cand_texts:
                    return max(cand_texts, key=len)
        return ""

    print('Interactive search ready — type your query (or Ctrl+C to exit)')

    while True:
        try:
            query = input('\nQuery> ').strip()
            if not query:
                print('Empty query; try again')
                continue

            # qvec variable will hold the vector we pass to FAISS
            qvec = None

            # ---------- Option A: If concept-mode, find top concepts and build hybrid query vector ----------
            if sapbert_embeddings is not None and sapbert_mode == 'concept' and sap_model is not None:
                try:
                    q_sap = sap_model.encode([query], convert_to_numpy=True)[0]
                    nq = np.linalg.norm(q_sap)
                    if nq != 0:
                        q_sap = q_sap / nq
                    sims = sapbert_embeddings.dot(q_sap)  # shape (n_concepts,)
                    top_idxs = np.argsort(-sims)[:sapbert_concepts_topk * 2]
                    top_labels = []
                    for i in top_idxs:
                        label = None
                        if concept_meta and i < len(concept_meta):
                            cand = concept_meta[i]
                            if isinstance(cand, dict):
                                label = cand.get('canonical_label') or cand.get('label') or cand.get('name')
                            else:
                                label = str(cand)
                        if label:
                            top_labels.append(label)
                    seen = set()
                    unique_top_labels = []
                    for t in top_labels:
                        tt = t.strip().lower()
                        if tt and tt not in seen:
                            seen.add(tt)
                            unique_top_labels.append(t)
                    unique_top_labels = unique_top_labels[:sapbert_concepts_topk]

                    if unique_top_labels:
                        print(f"[SapBERT concepts] top unique concepts: {unique_top_labels}")
                        qvec_raw = embed_model.encode([query], convert_to_numpy=True)[0]
                        try:
                            label_vecs = embed_model.encode(unique_top_labels, convert_to_numpy=True)
                            label_vec_mean = label_vecs.mean(axis=0)
                        except Exception as e:
                            print("Warning: failed to encode concept labels with embed_model:", e)
                            label_vec_mean = np.zeros_like(qvec_raw)
                        alpha = 0.7
                        beta = 0.3
                        combined = alpha * qvec_raw + beta * label_vec_mean
                        nrm = np.linalg.norm(combined)
                        if nrm > 0:
                            combined = combined / nrm
                        qvec = combined.astype(np.float32)
                    else:
                        qvec = embed_query(query, embed_model, normalize=True)
                except Exception as e:
                    print("Concept expansion failed:", e)
                    qvec = embed_query(query, embed_model, normalize=True)
            else:
                # not concept-mode: use plain query embedding
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

            # Runtime filter: replace title-like snippets with recovered longer text or drop them
            filtered_candidates = []
            for c in candidates:
                m = c['meta']
                title = m.get('title', '') or ''
                best = _longest_text_from_meta(m)
                if _is_title_like(best, title):
                    # try to recover longer text from other chunks with the same doc id
                    recovered = _find_longer_from_doc(metadata, m)
                    if recovered and len(recovered) > len(best):
                        c['display_snippet'] = recovered
                        filtered_candidates.append(c)
                    else:
                        # skip this candidate entirely (title-only and no recovery)
                        # (If you'd rather keep it but de-prioritize, change to filtered_candidates.append(c) with marker)
                        continue
                else:
                    c['display_snippet'] = best
                    filtered_candidates.append(c)

            candidates = filtered_candidates

            if not candidates:
                print('No candidates remain after filtering title-like passages.')
                continue

            # 3) Optional SapBERT scoring/ filtering (doc-level only)
            if sapbert_embeddings is not None and sapbert_mode == 'doc':
                if sap_model is None:
                    print("SapBERT doc-mode requested but sap_model failed to load; skipping sapbert scoring")
                else:
                    try:
                        query_sap_vec = sap_model.encode([query], convert_to_numpy=True)[0]
                        n = np.linalg.norm(query_sap_vec)
                        if n != 0:
                            query_sap_vec = query_sap_vec / n
                        for c in candidates:
                            idx = c['idx']
                            if idx < sapbert_embeddings.shape[0]:
                                s = float(np.dot(query_sap_vec, sapbert_embeddings[idx]))
                            else:
                                s = 0.0
                            c['sapbert_score'] = s
                        if sapbert_threshold is not None:
                            candidates = [c for c in candidates if c.get('sapbert_score', 0.0) >= sapbert_threshold]
                            if not candidates:
                                print('No candidates remain after SapBERT filtering')
                                continue
                    except Exception as e:
                        print('SapBERT scoring failed:', e)

            # 4) Rerank top rerank_k with cross-encoder
            rerank_candidates = candidates[:max(1, rerank_k)]
            # use the recovered/longer snippet for reranking if available
            texts = [c.get('display_snippet', '') for c in rerank_candidates]
            # drop any empty texts (they won't rerank well)
            non_empty = [(c, t) for c, t in zip(rerank_candidates, texts) if t and t.strip()]
            if not non_empty:
                print('No candidates with non-empty text to rerank')
                continue
            rerank_candidates, texts = zip(*non_empty)
            rerank_candidates = list(rerank_candidates)
            texts = list(texts)

            print('Running cross-encoder rerank (this may take a little time)...')
            try:
                if cross_encoder is not None:
                    pairs = [[query, t] for t in texts]
                    ce_scores = cross_encoder.predict(pairs)
                else:
                    ce_scores = rerank_with_cross_encoder(query, texts, cross_encoder_model_name)
            except Exception as e:
                print("Cross-encoder failed:", e)
                ce_scores = [0.0] * len(texts)

            # attach raw cross scores and other signals
            for c, ce in zip(rerank_candidates, ce_scores):
                c['cross_score'] = float(ce)
                c['trust_score'] = compute_trust_score(c['meta'])
                if 'sapbert_score' not in c:
                    c['sapbert_score'] = 0.0

            # Normalize cross_score and faiss_score across rerank_candidates (min-max)
            cross_vals = [c['cross_score'] for c in rerank_candidates]
            faiss_vals = [c['faiss_score'] for c in rerank_candidates]

            def min_max_norm(arr):
                if not arr:
                    return []
                mn = min(arr)
                mx = max(arr)
                if mx - mn == 0:
                    return [0.0 for _ in arr]
                return [(x - mn) / (mx - mn) for x in arr]

            cross_norm = min_max_norm(cross_vals)
            faiss_norm = min_max_norm(faiss_vals)

            for c, ce_norm, faiss_norm_v in zip(rerank_candidates, cross_norm, faiss_norm):
                c['cross_score_norm'] = float(ce_norm)
                c['faiss_score_norm'] = float(faiss_norm_v)

            # compute final combined score (normalized mixture)
            w_cross = 0.65
            # w_trust = 0.25
            w_faiss = 0.35
            # w_sap = 0.05  # only applies in doc-mode where we compute sapbert_score
            for c in rerank_candidates:
                cross_s = c.get('cross_score_norm', 0.0)
                trust_s = c.get('trust_score', 0.0)
                faiss_s = c.get('faiss_score_norm', 0.0)
                sap = c.get('sapbert_score', 0.0)
                # c['final_score'] = w_cross * cross_s + w_trust * trust_s + w_faiss * faiss_s
                c['final_score'] = w_cross * cross_s + w_faiss * faiss_s

            # --- DEBUG PRINT: show raw, normalized, and final scores ---
            print("\nCandidate rerank debug:")
            for i, c in enumerate(rerank_candidates, 1):
                print(f"{i}: cross_raw={c['cross_score']:.4f}, faiss_raw={c['faiss_score']:.4f}, "
                      f"cross_norm={c['cross_score_norm']:.4f}, faiss_norm={c['faiss_score_norm']:.4f}, "
                      f"final_score={c['final_score']:.4f}, trust={c['trust_score']:.4f}, sapbert={c['sapbert_score']:.4f}")

            # sort by final_score
            rerank_candidates.sort(key=lambda x: x['final_score'], reverse=True)

            # Display results
            console = Console()

            console.print("\n[bold cyan]Top results (after rerank):[/bold cyan]\n")

            for rank, c in enumerate(rerank_candidates[:20], 1):
                m = c["meta"]

                # Panel title
                header = Text(f"{rank}. {m.get('title', '')}", style="bold magenta")

                # Colored scores
                scores = Text()
                scores.append(f"final={c['final_score']:.4f}", style="bold green")
                scores.append(" | ")
                scores.append(f"cross={c['cross_score']:.4f}", style="bold yellow")
                scores.append(" | ")
                scores.append(f"faiss={c['faiss_score']:.4f}", style="bold blue")
                scores.append(" | ")
                scores.append(f"trust={c['trust_score']:.3f}", style="bold cyan")

                # Snippet
                snippet = c.get("display_snippet") or _longest_text_from_meta(m)
                snippet = " ".join(snippet.split())
                if len(snippet) > 350:
                    snippet = snippet[:350] + "..."

                # Sources list
                sources = m.get("sources") or []
                if isinstance(sources, str):
                    sources = [sources]

                sources_text = Text()
                for s in sources:
                    sources_text.append("• ", style="dim")
                    sources_text.append(s.strip())
                    sources_text.append("\n")

                # Body as a Text object
                body = Text()
                body.append_text(scores)
                body.append("\n\n")

                body.append("URL: ", style="bold")
                body.append(f"{m.get('url')}\n")

                body.append("Published: ", style="bold")
                body.append(f"{m.get('published_date', 'unknown')}\n")

                body.append("Author: ", style="bold")
                body.append(f"{m.get('author', 'none')}\n")

                body.append("Reviewed by: ", style="bold")
                body.append(f"{m.get('medically_reviewed_by', 'none')}\n\n")

                body.append("Snippet:\n", style="bold")
                body.append(snippet)
                body.append("\n\n")

                body.append("Sources:\n", style="bold")
                body.append_text(sources_text)

                # Render panel (this is the key: pass a Text object, not a string)
                console.print(Panel(body, title=header, expand=False))

        except KeyboardInterrupt:
            print('\nExiting.')
            break
        except Exception as e:
            print('Search failed:', e)



INDEX_DIR="../storage/outputs/webmd/faiss"
SAPBERT_EMBEDDING="../storage/seeds/embeddings/sapbert_embeddings.npy"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index-dir', type=str, default=INDEX_DIR,
                   help='Directory containing index.faiss, metadata.json, embeddings.npy')
    p.add_argument('--embedding-model', type=str, default='all-mpnet-base-v2',
                   help='SentenceTransformer model name for query embedding (used for FAISS index)')
    p.add_argument('--cross-encoder-model', type=str, default='neuml/bioclinical-modernbert-base-embeddings',
                   help='Cross-Encoder model name for reranking')
    p.add_argument('--sapbert-embeddings', type=str, default=SAPBERT_EMBEDDING,
                   help='Optional path to sapbert_embeddings.npy (concept embeddings or doc-level embeddings)')
    p.add_argument('--sapbert-model', type=str, default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
                   help='SapBERT model name used to encode queries into the concept embedding space')
    p.add_argument('--sapbert-concepts-topk', type=int, default=5,
                   help='Number of top concepts to use for query expansion when sapbert_embeddings are concept vectors')
    p.add_argument('--sapbert-threshold', type=float, default=None,
                   help='Optional hard threshold for sapbert cosine filter (doc-level mode only)')
    p.add_argument('--sapbert-boost', type=float, default=0.2,
                   help='Amount to include sapbert score in final score (doc-level mode only)')
    p.add_argument('--topk', type=int, default=100,
                   help='Number of passages to retrieve from FAISS')
    p.add_argument('--rerank', type=int, default=5,
                   help='Number of top FAISS results to rerank with cross-encoder')
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
        sapbert_model_name=args.sapbert_model,
        sapbert_concepts_topk=args.sapbert_concepts_topk,
    )
