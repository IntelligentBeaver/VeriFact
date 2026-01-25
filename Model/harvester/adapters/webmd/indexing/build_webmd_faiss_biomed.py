"""
Build FAISS Index for Medical Fact Retrieval (One Time Setup)

This script processes WebMD health articles from JSON files and creates:
1. Passage embeddings using biomedical language models
2. FAISS index for fast similarity search
3. Metadata for each passage (article info, sections, etc.)

What it does:
- Reads JSON files containing health articles
- Breaks articles into smaller chunks (passages) from sections
- Creates two types of embeddings:
  * Primary embeddings: For finding similar medical content
  * SapBERT embeddings: For matching medical terms and concepts
- Builds a searchable FAISS index

Output files:
- embeddings.npy: Primary passage vectors for retrieval
- sapbert_embeddings.npy: Medical entity vectors for term matching
- index.faiss: Fast search index
- metadata.json: Article details for each passage

How to use:
1. Edit the CONFIGURATION section below to set your input/output paths and models
2. Run the script directly: python build_webmd_faiss_biomed.py

Requirements:
    pip install sentence-transformers faiss-cpu numpy tqdm

"""

import os
import json
from pathlib import Path
from hashlib import sha1
from tqdm import tqdm
import numpy as np

# ============================================================================
# CONFIGURATION - Edit these values before running
# ============================================================================

# Input configuration - Choose one:
INPUT_FILE = None  # Set to a specific JSON file path, or leave as None to use INPUT_DIR
INPUT_DIR = "../../storage/webmd/articles"  # Directory containing JSON article files

# Output configuration
OUTPUT_DIR = "../../storage/outputs/webmd"

# Model configuration
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # Primary biomedical retrieval model
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # Medical entity matching model (set to None to disable)

# Processing configuration
BATCH_SIZE = 32  # Number of passages to process at once
NORMALIZE_EMBEDDINGS = True  # Normalize vectors for cosine similarity (recommended: True)

# ============================================================================


def sha1_hex(s: str) -> str:
    return sha1(s.encode('utf-8')).hexdigest()


def load_json_files(input_path: Path):
    """Load JSON files from a single file or directory.
    
    Args:
        input_path: Path to a JSON file or directory containing JSON files
        
    Yields:
        dict: Parsed JSON document
    """
    if input_path.is_file():
        print(f"Loading single file: {input_path}")
        with input_path.open('r', encoding='utf-8') as f:
            yield json.load(f)
    else:
        json_files = list(input_path.glob('*.json'))
        print(f"Found {len(json_files)} JSON files in {input_path}")
        for p in tqdm(sorted(json_files), desc="Loading files"):
            try:
                with p.open('r', encoding='utf-8') as f:
                    yield json.load(f)
            except Exception as e:
                print(f"\nWarning: Failed to load {p.name}: {e}")


def extract_passages_from_doc(doc: dict):
    """Break a health article into smaller text passages.
    
    Each passage is a paragraph or content block from the article, with:
    - The passage text (paragraph + any bullet points)
    - Article metadata (title, URL, author, etc.)
    - Section heading it belongs to
    - Position information (for reference back to source)
    
    Args:
        doc: JSON document containing article content
        
    Returns:
        list: List of passage dictionaries
    """
    # Document-level fields to copy
    doc_fields = {}
    for k in [
        'url', 'title', 'published_date', 'scrape_timestamp_utc', 'author',
        'medically_reviewed_by', 'sources', 'meta_description', 'tags'
    ]:
        doc_fields[k] = doc.get(k)

    doc_id = sha1_hex(str(doc_fields.get('url') or doc_fields.get('title') or ''))

    # Build a concatenated doc text to compute char offsets
    concat = []
    passages = []
    cursor = 0

    sections = doc.get('sections') or []
    for si, sec in enumerate(sections):
        heading = sec.get('heading')
        content_blocks = sec.get('content_blocks') or []
        for bi, blk in enumerate(content_blocks):
            # Compose passage text: paragraph text + associated bullets (if any)
            text = (blk.get('text') or '').strip()
            bullets = blk.get('associated_bullets')
            if bullets:
                # Join bullets as lines; keep them attached to the paragraph for context
                bullets_text = '\n'.join([b.strip() for b in bullets if b])
                # separator ensures sentences don't merge when concatenated
                full_text = text + '\n' + bullets_text if text else bullets_text
            else:
                full_text = text

            if not full_text:
                continue

            # Compute offsets
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
            # copy doc-level fields
            p.update(doc_fields)
            passages.append(p)

    return passages


def dedupe_passages(passages):
    """Remove duplicate passages with identical text.
    
    Keeps the first occurrence of each unique text passage.
    
    Args:
        passages: List of passage dictionaries
        
    Returns:
        list: Deduplicated list of passages
    """
    seen = set()
    out = []
    for p in passages:
        t = p['text']
        if t in seen:
            continue
        seen.add(t)
        out.append(p)
    
    duplicates_removed = len(passages) - len(out)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate passages")
    
    return out


def compute_embeddings(passages, model_name, batch_size=32, normalize=True):
    """Create vector embeddings for passages using a biomedical language model.
    
    Args:
        passages: List of passage dictionaries with 'text' field
        model_name: HuggingFace model identifier (e.g., 'pritamdeka/S-PubMedBert-MS-MARCO')
        batch_size: Number of passages to process at once
        normalize: Whether to normalize vectors (recommended for similarity search)
        
    Returns:
        tuple: (embeddings array, loaded model)
    """
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
        # Normalize vectors to unit length for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings = embeddings / norms
        print("Embeddings normalized for cosine similarity")

    return embeddings, model


def build_faiss_index(embeddings, out_path: Path):
    """Build a FAISS index for fast similarity search.
    
    Creates a flat inner-product index (works with normalized vectors for cosine similarity).
    
    Args:
        embeddings: Numpy array of embeddings (N x dimension)
        out_path: Directory to save the index file
        
    Returns:
        faiss.Index: Built FAISS index
    """
    try:
        import faiss
    except ImportError:
        raise RuntimeError('FAISS is required. Install via: pip install faiss-cpu')

    dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dim}...")
    
    # Inner-product index (equivalent to cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    index_path = out_path / 'index.faiss'
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved: {index_path}")
    
    return index


def save_outputs(embeddings, passages, out_dir: Path, sapbert_embeddings=None):
    """Save embeddings and metadata to disk.
    
    Args:
        embeddings: Primary passage embeddings array
        passages: List of passage dictionaries
        out_dir: Output directory
        sapbert_embeddings: Optional SapBERT embeddings array
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving outputs to: {out_dir}")
    np.save(out_dir / 'embeddings.npy', embeddings.astype(np.float32))
    print(f"  ✓ Saved embeddings.npy ({embeddings.shape})")

    # Create metadata list (same order as embeddings)
    metadata = []
    for p in passages:
        # Keep only the fields you said you'd like to keep (exclude language, isBoilerplate, cui_list)
        meta = {
            'passage_id': p['passage_id'],
            'doc_id': p['doc_id'],
            'section_heading': p.get('section_heading'),
            'block_index': p.get('block_index'),
            'text': p['text'],
            'char_start': p.get('char_start'),
            'char_end': p.get('char_end'),
            # document-level fields copied down
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
        metadata.append(meta)

    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved metadata.json ({len(metadata)} passages)")

    if sapbert_embeddings is not None:
        np.save(out_dir / 'sapbert_embeddings.npy', sapbert_embeddings.astype(np.float32))
        print(f"  ✓ Saved sapbert_embeddings.npy ({sapbert_embeddings.shape})")


def compute_sapbert_embeddings(passages, sapbert_model_name, batch_size=32, normalize=True):
    """Create SapBERT embeddings for medical entity/concept matching.
    
    SapBERT embeddings help match medical terms even when different terminology is used
    (e.g., "heart attack" vs "myocardial infarction").
    
    Args:
        passages: List of passage dictionaries with 'text' field
        sapbert_model_name: HuggingFace SapBERT model identifier
        batch_size: Number of passages to process at once
        normalize: Whether to normalize vectors
        
    Returns:
        numpy.ndarray: SapBERT embeddings array
    """
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


def main():
    """Main execution function - uses configuration variables defined at the top of the file."""
    
    print("=" * 60)
    print("Building FAISS Index for Medical Fact Retrieval")
    print("=" * 60)
    
    input_path = Path(INPUT_FILE) if INPUT_FILE else Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    
    # Validate input path
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    print(f"\nConfiguration:")
    print(f"  Input: {input_path}")
    print(f"  Output: {out_dir}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  SapBERT model: {SAPBERT_MODEL or 'Disabled'}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Normalize: {NORMALIZE_EMBEDDINGS}")
    print()

    all_passages = []

    # Load and parse JSON files
    print("Step 1: Loading and parsing articles...")
    for doc in load_json_files(input_path):
        try:
            passages = extract_passages_from_doc(doc)
            all_passages.extend(passages)
        except Exception as e:
            print(f"\nError parsing document: {e}")

    print(f"\nExtracted {len(all_passages)} raw passages from articles")

    # Deduplicate passages
    print("\nStep 2: Removing duplicate passages...")
    passages = dedupe_passages(all_passages)
    print(f"Final passage count: {len(passages)} passages\n")

    if not passages:
        print('Error: No passages found to process. Check your input files.')
        return

    # Compute primary embeddings
    print("Step 3: Computing primary embeddings...")
    embeddings, model = compute_embeddings(passages, EMBEDDING_MODEL, batch_size=BATCH_SIZE, normalize=NORMALIZE_EMBEDDINGS)
    print(f"Created embeddings with shape: {embeddings.shape}\n")

    # Optional SapBERT embeddings
    sapbert_emb = None
    if SAPBERT_MODEL:
        print("Step 4: Computing SapBERT embeddings for entity matching...")
        try:
            sapbert_emb = compute_sapbert_embeddings(passages, SAPBERT_MODEL, batch_size=BATCH_SIZE, normalize=NORMALIZE_EMBEDDINGS)
            print(f"Created SapBERT embeddings with shape: {sapbert_emb.shape}\n")
        except Exception as e:
            print(f"Warning: Failed to compute SapBERT embeddings: {e}")
            print("Continuing without SapBERT embeddings...\n")
    else:
        print("Step 4: Skipping SapBERT embeddings (disabled)\n")

    # Save embeddings and metadata
    print("Step 5: Saving embeddings and metadata...")
    save_outputs(embeddings, passages, out_dir, sapbert_embeddings=sapbert_emb)
    print()

    # Build and write FAISS index
    print("Step 6: Building FAISS index...")
    try:
        idx = build_faiss_index(embeddings, out_dir)
    except Exception as e:
        print(f"Error: Failed to build FAISS index: {e}")
        return

    print("\n" + "=" * 60)
    print("✓ Successfully completed!")
    print("=" * 60)
    print(f"\nOutput directory: {out_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  • embeddings.npy - Primary passage vectors ({embeddings.shape[0]} passages)")
    print(f"  • index.faiss - Fast similarity search index")
    print(f"  • metadata.json - Article and passage metadata")
    if sapbert_emb is not None:
        print(f"  • sapbert_embeddings.npy - Entity matching vectors ({sapbert_emb.shape[0]} passages)")
    print("\nYou can now use these files for medical fact retrieval!\n")


if __name__ == '__main__':
    main()
