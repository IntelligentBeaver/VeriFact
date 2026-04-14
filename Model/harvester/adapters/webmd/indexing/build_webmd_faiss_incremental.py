"""
Incremental FAISS Indexing for Medical Articles

This script adds new articles to an existing FAISS index without re-embedding all passages.
It detects new articles, embeds only those passages, and rebuilds the index incrementally.

What it does:
- Loads existing index files (embeddings.npy, metadata.json, etc.)
- Identifies new articles not yet indexed
- Embeds only new passages (avoiding duplicate work)
- Appends new embeddings to existing files
- Rebuilds FAISS index with merged embeddings

When to use:
- Weekly article refreshes
- Adding batches of new content without full rebuild
- Reduces time from ~50-100 min to ~5-10 min for small updates

Output:
- Updated embeddings.npy, metadata.json, index.faiss files
- Status report showing new passages added

How to use:
1. Edit the CONFIGURATION section below with paths
2. Place new WebMD JSON files in INPUT_DIR
3. Run: python build_webmd_faiss_incremental.py

Requirements:
    pip install sentence-transformers faiss-cpu numpy tqdm

"""

import json
from pathlib import Path
from hashlib import sha1
from tqdm import tqdm
import numpy as np

# ============================================================================
# CONFIGURATION - Edit these values before running
# ============================================================================

# Input configuration - Directory with NEW articles (only these will be embedded)
INPUT_DIR = "../../storage/webmd/articles_new"

# Existing index directory - must contain files from build_webmd_faiss_biomed.py
INDEX_DIR = "../../storage/outputs/webmd"

# Model configuration - MUST MATCH models used in build_webmd_faiss_biomed.py
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Processing configuration
BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True

# ============================================================================


def sha1_hex(s: str) -> str:
    return sha1(s.encode('utf-8')).hexdigest()


def load_json_files(input_path: Path):
    """Load JSON files from a directory."""
    if not input_path.exists():
        print(f"Warning: Input path does not exist: {input_path}")
        return
    
    json_files = list(input_path.glob('*.json'))
    print(f"Found {len(json_files)} JSON files in {input_path}")
    
    for p in tqdm(sorted(json_files), desc="Loading files"):
        try:
            with p.open('r', encoding='utf-8') as f:
                yield json.load(f)
        except Exception as e:
            print(f"\nWarning: Failed to load {p.name}: {e}")


def extract_passages_from_doc(doc: dict):
    """Break a health article into smaller text passages (same as build_webmd_faiss_biomed.py)."""
    doc_fields = {}
    for k in [
        'url', 'title', 'published_date', 'scrape_timestamp_utc', 'author',
        'medically_reviewed_by', 'sources', 'meta_description', 'tags'
    ]:
        doc_fields[k] = doc.get(k)

    doc_id = sha1_hex(str(doc_fields.get('url') or doc_fields.get('title') or ''))

    concat = []
    passages = []
    cursor = 0

    sections = doc.get('sections') or []
    for si, sec in enumerate(sections):
        heading = sec.get('heading')
        content_blocks = sec.get('content_blocks') or []
        for bi, blk in enumerate(content_blocks):
            text = (blk.get('text') or '').strip()
            bullets = blk.get('associated_bullets')
            if bullets:
                bullets_text = '\n'.join([b.strip() for b in bullets if b])
                full_text = text + '\n' + bullets_text if text else bullets_text
            else:
                full_text = text

            if not full_text:
                continue

            char_start = cursor
            concat.append(full_text)
            cursor += len(full_text)
            char_end = cursor

            passage_id = f"{doc_id}_s{si}_b{bi}"

            p = {
                'passage_id': passage_id,
                'doc_id': doc_id,
                'section_heading': heading,
                'block_index': bi,
                'text': full_text,
                'char_start': char_start,
                'char_end': char_end,
            }
            p.update(doc_fields)
            passages.append(p)

    return passages


def load_existing_metadata(metadata_path: Path):
    """Load existing metadata to track already-indexed passages."""
    if not metadata_path.exists():
        return [], set()
    
    try:
        with metadata_path.open('r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load metadata: {e}")
        return [], set()
    
    # Track existing passage texts to avoid duplicates
    existing_texts = {m.get('text', '') for m in metadata if m.get('text')}
    return metadata, existing_texts


def load_new_articles(input_dir: Path, existing_texts: set):
    """Load only new articles not yet in metadata."""
    new_passages = []
    
    if not input_dir.exists():
        print(f"Warning: Input directory does not exist: {input_dir}")
        return new_passages
    
    for doc in load_json_files(input_dir):
        try:
            passages = extract_passages_from_doc(doc)
            # Filter: only keep passages not already indexed
            for p in passages:
                if p['text'] not in existing_texts:
                    new_passages.append(p)
                    existing_texts.add(p['text'])
        except Exception as e:
            print(f"\nWarning: Failed to extract passages: {e}")
    
    return new_passages


def compute_embeddings(passages, model_name, batch_size=32, normalize=True):
    """Create vector embeddings for passages."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    texts = [p['text'] for p in passages]
    model = SentenceTransformer(model_name)
    
    print(f"Computing embeddings for {len(texts)} passages...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings = embeddings / norms
        print("Embeddings normalized for cosine similarity")

    return embeddings


def compute_sapbert_embeddings(passages, sapbert_model_name, batch_size=32, normalize=True):
    """Create SapBERT embeddings for medical entity matching."""
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading SapBERT model: {sapbert_model_name}")
    texts = [p['text'] for p in passages]
    model = SentenceTransformer(sapbert_model_name)
    
    print(f"Computing SapBERT embeddings for {len(texts)} passages...")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        emb = emb / norms
        print("SapBERT embeddings normalized")
    
    return emb


def append_to_npy(old_embeddings_path: Path, new_embeddings):
    """Append new embeddings to existing .npy file or create new one."""
    if old_embeddings_path.exists():
        try:
            old_emb = np.load(str(old_embeddings_path))
            combined = np.vstack([old_emb, new_embeddings])
            print(f"  Appended to existing: {old_emb.shape[0]} + {new_embeddings.shape[0]} = {combined.shape[0]}")
        except Exception as e:
            print(f"  Warning: Failed to load existing embeddings: {e}")
            combined = new_embeddings
    else:
        combined = new_embeddings
        print(f"  Created new: {new_embeddings.shape[0]} passages")
    
    np.save(str(old_embeddings_path), combined.astype(np.float32))
    return combined


def rebuild_faiss_index(embeddings, index_path: Path):
    """Rebuild FAISS index from all embeddings."""
    try:
        import faiss
    except ImportError:
        raise RuntimeError('FAISS is required. Install via: pip install faiss-cpu')

    dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dim}...")
    
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    
    faiss.write_index(index, str(index_path))
    print(f"  ✓ Index rebuilt: {embeddings.shape[0]} passages")


def save_metadata(metadata_list: list, metadata_path: Path):
    """Save metadata to JSON file."""
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Metadata saved: {len(metadata_list)} passages")


def main():
    """Incremental indexing workflow."""
    
    print("=" * 60)
    print("Incremental FAISS Indexing")
    print("=" * 60)
    
    input_dir = Path(INPUT_DIR)
    index_dir = Path(INDEX_DIR)
    
    metadata_path = index_dir / 'metadata.json'
    embeddings_path = index_dir / 'embeddings.npy'
    sapbert_path = index_dir / 'sapbert_embeddings.npy'
    index_path = index_dir / 'index.faiss'
    
    print(f"\nConfiguration:")
    print(f"  New articles: {input_dir}")
    print(f"  Index directory: {index_dir}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  SapBERT model: {SAPBERT_MODEL or 'Disabled'}")
    print()

    # Load existing metadata
    print("Step 1: Loading existing metadata...")
    existing_metadata, existing_texts = load_existing_metadata(metadata_path)
    print(f"  ✓ Loaded {len(existing_metadata)} existing passages")
    
    # Load new articles
    print("\nStep 2: Loading new articles...")
    new_passages = load_new_articles(input_dir, existing_texts)
    
    if not new_passages:
        print("\n  ⚠ No new passages found. Index is up to date.")
        return
    
    print(f"  ✓ Found {len(new_passages)} new passages")
    
    # Embed new passages
    print("\nStep 3: Computing primary embeddings...")
    new_embeddings = compute_embeddings(new_passages, EMBEDDING_MODEL, batch_size=BATCH_SIZE, normalize=NORMALIZE_EMBEDDINGS)
    print(f"  Shape: {new_embeddings.shape}")
    
    # Embed with SapBERT
    sapbert_emb = None
    if SAPBERT_MODEL:
        print("\nStep 4: Computing SapBERT embeddings...")
        try:
            sapbert_emb = compute_sapbert_embeddings(new_passages, SAPBERT_MODEL, batch_size=BATCH_SIZE, normalize=NORMALIZE_EMBEDDINGS)
            print(f"  Shape: {sapbert_emb.shape}")
        except Exception as e:
            print(f"  Warning: Failed to compute SapBERT embeddings: {e}")
    else:
        print("\nStep 4: Skipping SapBERT embeddings (disabled)")
    
    # Append to existing embeddings
    print("\nStep 5: Merging embeddings...")
    all_embeddings = append_to_npy(embeddings_path, new_embeddings)
    
    if sapbert_emb is not None:
        all_sapbert = append_to_npy(sapbert_path, sapbert_emb)
    
    # Create metadata for new passages
    print("\nStep 6: Merging metadata...")
    new_metadata = []
    for p in new_passages:
        meta = {
            'passage_id': p['passage_id'],
            'doc_id': p['doc_id'],
            'section_heading': p.get('section_heading'),
            'block_index': p.get('block_index'),
            'text': p['text'],
            'char_start': p.get('char_start'),
            'char_end': p.get('char_end'),
            'url': p.get('url'),
            'title': p.get('title'),
            'published_date': p.get('published_date'),
            'scrape_timestamp_utc': p.get('scrape_timestamp_utc'),
            'author': p.get('author'),
            'medically_reviewed_by': p.get('medically_reviewed_by'),
            'sources': p.get('sources'),
            'meta_description': p.get('meta_description'),
            'tags': p.get('tags'),
        }
        new_metadata.append(meta)
    
    all_metadata = existing_metadata + new_metadata
    save_metadata(all_metadata, metadata_path)
    
    # Rebuild FAISS index
    print("\nStep 7: Rebuilding FAISS index...")
    rebuild_faiss_index(all_embeddings, index_path)
    
    print("\n" + "=" * 60)
    print("✓ Incremental indexing complete!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  New passages added: {len(new_passages)}")
    print(f"  Total passages: {len(all_metadata)}")
    print(f"  Primary embeddings: {all_embeddings.shape}")
    if sapbert_emb is not None:
        print(f"  SapBERT embeddings: {all_sapbert.shape}")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
