"""
Medical Fact Retrieval with Embeddings + Cross-Encoder Reranking

This script uses the FAISS index and embeddings created by build_webmd_faiss_biomed.py
to answer medical fact-checking queries.

Inputs (files created by build_webmd_faiss_biomed.py):
 - index.faiss: FAISS search index
 - embeddings.npy: Primary passage embeddings (biomedical model)
 - sapbert_embeddings.npy: Medical entity embeddings for concept matching
 - metadata.json: Passage metadata (title, URL, author, sources, etc.)

Retrieval Flow:
 1. Load FAISS index, embeddings, metadata, and SapBERT embeddings
 2. Embed user query using the same embedding model that built the index
 3. Search FAISS index to retrieve top-K relevant passages
 4. Compute SapBERT similarity scores for medical concept matching
 5. Rerank top results using Cross-Encoder model
 6. Combine scores: FAISS (35%) + Cross-Encoder (65%) + SapBERT bonus
 7. Display ranked results with explanations and source information

How to use:
 1. Edit the CONFIGURATION section below with your paths and models
 2. Run directly: python build_webmd_indexing_biomed.py
 3. Type your medical question at the Query> prompt
 4. Press Ctrl+C to exit

Requirements:
 pip install sentence-transformers faiss-cpu numpy tqdm python-dateutil rich

"""

import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from dateutil import parser as date_parser
from datetime import datetime
import math

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ============================================================================
# CONFIGURATION - Edit these values before running
# ============================================================================

# Index directory - must contain files from build_webmd_faiss_biomed.py
INDEX_DIR = "../../../storage/outputs/webmd/faiss"

# Models - MUST MATCH the models used in build_webmd_faiss_biomed.py
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # Primary biomedical model
CROSS_ENCODER_MODEL = "cross-encoder/nli-deberta-v3-large"  # NLI reranker
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # For medical entities

# Retrieval parameters
TOPK_FAISS = 100           # Number of passages to retrieve from FAISS
RERANK_K = 20              # Number of top results to rerank with cross-encoder
SAPBERT_CONCEPTS_TOPK = 5  # Top concepts for expansion if SapBERT is concept-mode

# Scoring weights - how to combine different signals
W_CROSS_ENCODER = 0.70  # Cross-encoder importance (65%)
W_FAISS = 0.25          # FAISS similarity importance (35%)
W_SAPBERT = 0.05        # SapBERT boost for entity matching (optional)

# Display parameters
DISPLAY_TOP_RESULTS = 20      # Show top N results to user
SNIPPET_MAX_CHARS = 350       # Max characters in preview snippet
SAPBERT_THRESHOLD = None      # Optional: only show results with SapBERT score >= threshold

# ============================================================================

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


def nli_scores_to_relevance(raw_scores):
    """Map NLI logits/probs (entailment, neutral, contradiction) to a relevance score."""
    try:
        arr = np.asarray(raw_scores)
    except Exception:
        return [float(x) for x in raw_scores] if raw_scores is not None else []

    if arr.ndim == 2 and arr.shape[1] == 3:
        return [float(max(row[1], row[2])) for row in arr]

    # Fallback for 1D scores
    try:
        return [float(x) for x in arr.tolist()]
    except Exception:
        return [float(x) for x in raw_scores] if raw_scores is not None else []


def interactive_loop():
    """Main interactive search loop using configuration variables."""

    # Setup paths
    index_dir = Path(INDEX_DIR)
    index_path = index_dir / 'index.faiss'
    metadata_path = index_dir / 'metadata.json'
    sapbert_path = index_dir / 'sapbert_embeddings.npy'
    
    # Validate required files exist
    print("=" * 60)
    print("Medical Fact Retrieval System")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Index directory: {index_dir}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  Cross-encoder: {CROSS_ENCODER_MODEL}")
    print(f"  SapBERT model: {SAPBERT_MODEL}")
    print(f"  Scoring: {W_CROSS_ENCODER:.0%} cross-encoder + {W_FAISS:.0%} FAISS")
    print()
    
    if not index_path.exists():
        print(f"Error: index.faiss not found at {index_path}")
        print(f"Did you run build_webmd_faiss_biomed.py first?")
        return
    if not metadata_path.exists():
        print(f"Error: metadata.json not found at {metadata_path}")
        return

    # Load core files
    print('Step 1: Loading metadata and index...')
    metadata = load_metadata(metadata_path)
    print(f'  ✓ Loaded {len(metadata)} passages')

    index = load_faiss_index(index_path)
    print(f'  ✓ Loaded FAISS index')

    # Load models
    print('\nStep 2: Loading embedding models...')
    print(f'  Loading: {EMBEDDING_MODEL}')
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f'  ✓ Embedding model loaded')

    print(f'  Loading: {CROSS_ENCODER_MODEL}')
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        print(f'  ✓ Cross-encoder loaded')
    except Exception as e:
        print(f"  ✗ Warning: Failed to load cross-encoder: {e}")
        cross_encoder = None

    # Load SapBERT assets
    print('\nStep 3: Loading SapBERT embeddings...')
    sapbert_embeddings = None
    sapbert_mode = None
    sap_model = None
    concept_meta = None

    if sapbert_path.exists():
        print(f'  Loading: {sapbert_path.name}')
        sapbert_embeddings = np.load(str(sapbert_path))
        print(f'  ✓ SapBERT embeddings shape: {sapbert_embeddings.shape}')
        
        if sapbert_embeddings.shape[0] == len(metadata):
            sapbert_mode = 'doc'
            print(f'  ✓ Detected document-level SapBERT embeddings')
            # Load SapBERT encoder to encode queries in the same space:
            try:
                print(f'  Loading: {SAPBERT_MODEL}')
                sap_model = SentenceTransformer(SAPBERT_MODEL)
                print(f'  ✓ SapBERT encoder loaded')
            except Exception as e:
                print(f"  ✗ Failed to load SapBERT encoder: {e}")
                sap_model = None
        else:
            sapbert_mode = 'concept'
            print(f'  ✓ Detected concept-level SapBERT embeddings (n={sapbert_embeddings.shape[0]} concepts)')
            # attempt to find concept metadata (labels) in same dir as sapbert_path
            cand_meta_names = ['mesh_concepts.json', 'concepts.json', 'combined_metadata.json']
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
                    print(f"  ✓ Loaded concept metadata ({len(concept_meta)} concepts)")
                except Exception as e:
                    print(f"  ✗ Failed to load concept metadata: {e}")
                    concept_meta = None
            else:
                print(f"  ⚠ No concept metadata found; labels unavailable for expansion")
                concept_meta = None

            # load sapbert encoder for queries (required for mapping query -> concept space)
            try:
                print(f'  Loading: {SAPBERT_MODEL}')
                sap_model = SentenceTransformer(SAPBERT_MODEL)
                print(f'  ✓ SapBERT encoder loaded')
            except Exception as e:
                print(f"  ✗ Failed to load SapBERT encoder: {e}")
                sap_model = None
    else:
        print('  ⚠ SapBERT embeddings not found (optional feature disabled)')

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

    print("\n" + "=" * 60)
    print("✓ System ready! Type your medical question (or press Ctrl+C to exit)")
    print("=" * 60)

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
                    top_idxs = np.argsort(-sims)[:SAPBERT_CONCEPTS_TOPK * 2]
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
                    unique_top_labels = unique_top_labels[:SAPBERT_CONCEPTS_TOPK]

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
            faiss_scores, faiss_idxs = faiss_search(index, qvec, topk=TOPK_FAISS)

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
                        
                        # Show top SapBERT-matched concepts/passages
                        top_sapbert = sorted(candidates, key=lambda x: x.get('sapbert_score', 0.0), reverse=True)[:5]
                        print(f"[SapBERT doc-mode] Top medical concept matches:")
                        for i, c in enumerate(top_sapbert, 1):
                            title = c['meta'].get('title', 'Untitled')[:60]
                            score = c['sapbert_score']
                            print(f"  {i}. {title}... (score: {score:.4f})")
                        
                        if SAPBERT_THRESHOLD is not None:
                            candidates = [c for c in candidates if c.get('sapbert_score', 0.0) >= SAPBERT_THRESHOLD]
                            if not candidates:
                                print(f'No candidates above SapBERT threshold ({SAPBERT_THRESHOLD:.3f})')
                                continue
                    except Exception as e:
                        print('SapBERT scoring failed:', e)

            # 4) Rerank top RERANK_K with cross-encoder
            rerank_candidates = candidates[:max(1, RERANK_K)]
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
                    ce_scores = nli_scores_to_relevance(ce_scores)
                else:
                    ce_scores = rerank_with_cross_encoder(query, texts, CROSS_ENCODER_MODEL)
                    ce_scores = nli_scores_to_relevance(ce_scores)
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

            # compute final combined score using configured weights
            for c in rerank_candidates:
                cross_s = c.get('cross_score_norm', 0.0)
                faiss_s = c.get('faiss_score_norm', 0.0)
                sap = c.get('sapbert_score', 0.0)
                
                # Combine: cross-encoder (65%) + FAISS (35%) + optional SapBERT boost
                c['final_score'] = W_CROSS_ENCODER * cross_s + W_FAISS * faiss_s
                
                # Optional: boost score based on SapBERT if available
                if sapbert_mode == 'doc' and sap > 0:
                    c['final_score'] += W_SAPBERT * sap

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

            for rank, c in enumerate(rerank_candidates[:DISPLAY_TOP_RESULTS], 1):
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
                scores.append(f"sapbert={c['sapbert_score']:.4f}", style="bold blue")
                scores.append(" | ")
                scores.append(f"trust={c['trust_score']:.3f}", style="bold cyan")

                # Snippet
                snippet = c.get("display_snippet") or _longest_text_from_meta(m)
                snippet = " ".join(snippet.split())
                if len(snippet) > SNIPPET_MAX_CHARS:
                    snippet = snippet[:SNIPPET_MAX_CHARS] + "..."

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


if __name__ == '__main__':
    try:
        interactive_loop()
    except KeyboardInterrupt:
        print('\n\nExiting. Goodbye!')
    except Exception as e:
        print(f'\n\nFatal error: {e}')
        import traceback
        traceback.print_exc()
