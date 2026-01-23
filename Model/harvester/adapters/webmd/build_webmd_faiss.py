"""
build_faiss_index.py

Create embeddings.npy, index.faiss and metadata.json from WebMD-like JSON files.

Features:
- Parses each input JSON file and extracts passage-level chunks from `sections -> content_blocks`.
- Attaches doc-level metadata fields you requested: url, title, published_date, scrape_timestamp_utc, author,
  medically_reviewed_by, sources, meta_description, tags.
- Optional SapBERT-style embedding step (provide a SentenceTransformer model name if you already
  have SapBERT packaged as a sentence-transformers model). This is optional and controlled
  by the --sapbert-model CLI arg.
- Saves:
    - embeddings.npy        (float32 matrix: N x dim)
    - index.faiss           (FAISS index file for the embeddings)
    - metadata.json         (list of passage metadata objects in same order as embeddings)

Usage example:
python build_faiss_index.py \
    --input_dir ./webmd_jsons \
    --out_dir ./index_output \
    --embedding-model all-mpnet-base-v2 \
    --batch-size 64 \
    --normalize

Optional SapBERT embeddings (will also save sapbert_embeddings.npy if used):
python build_faiss_index.py \
    --input_dir ./webmd_jsons \
    --out_dir ./index_output \
    --embedding-model all-mpnet-base-v2 \
    --sapbert-model pritamdeka/sapbert-sentence-transformers \
    --batch-size 64

Requirements:
pip install sentence-transformers faiss-cpu tqdm numpy

"""

import os
import json
import argparse
from pathlib import Path
from hashlib import sha1
from tqdm import tqdm
import numpy as np

INPUT_FILE= "webmd_healthtopics_sample.json"
INPUT_DIR="../../storage/webmd/healthtopics"
# INPUT_DIR="../webmd"

OUTPUT_DIR = "../../storage/outputs/webmd"


def sha1_hex(s: str) -> str:
    return sha1(s.encode('utf-8')).hexdigest()


def load_json_files(input_path: Path):
    """Yield loaded JSON objects from file or directory."""
    if input_path.is_file():
        with input_path.open('r', encoding='utf-8') as f:
            yield json.load(f)
    else:
        for p in sorted(input_path.glob('*.json')):
            try:
                with p.open('r', encoding='utf-8') as f:
                    yield json.load(f)
            except Exception as e:
                print(f"Warning: failed to load {p}: {e}")


def extract_passages_from_doc(doc: dict):
    """Return list of passage dicts extracted from a single document JSON.

    Passage schema returned (minimal):
      - passage_id
      - doc_id
      - url, title, published_date, scrape_timestamp_utc, author, medically_reviewed_by,
        sources, meta_description, tags
      - section_heading
      - block_index
      - text
      - char_start, char_end (offsets within the concatenated document text)

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
    """Simple exact-text dedupe preserving first occurrence order."""
    seen = set()
    out = []
    for p in passages:
        t = p['text']
        if t in seen:
            continue
        seen.add(t)
        out.append(p)
    return out


def compute_embeddings(passages, model_name, batch_size=32, normalize=True):
    from sentence_transformers import SentenceTransformer

    texts = [p['text'] for p in passages]
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if normalize:
        # L2 normalize for cosine-similarity via inner-product.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings = embeddings / norms

    return embeddings, model


def build_faiss_index(embeddings, out_path: Path):
    try:
        import faiss
    except Exception as e:
        raise RuntimeError('faiss is required (faiss-cpu). Install via `pip install faiss-cpu`')

    dim = embeddings.shape[1]
    # Use a simple inner-product (cosine because vectors are normalized) flat index.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out_path / 'index.faiss'))
    return index


def save_outputs(embeddings, passages, out_dir: Path, sapbert_embeddings=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy', embeddings.astype(np.float32))

    # metadata.json will be a list of passage metadata objects in the same order as embeddings
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

    if sapbert_embeddings is not None:
        np.save(out_dir / 'sapbert_embeddings.npy', sapbert_embeddings.astype(np.float32))


def compute_sapbert_embeddings(passages, sapbert_model_name, batch_size=32, normalize=True):
    """Optional: compute SapBERT-like embeddings using a sentence-transformers model name.

    Note: If SapBERT for entity linking is produced elsewhere based on MeSH seeds, you can
    instead provide a path to precomputed sapbert vectors. This function will attempt to
    load a SentenceTransformer by the provided name and encode the passage texts.
    """
    from sentence_transformers import SentenceTransformer
    texts = [p['text'] for p in passages]
    model = SentenceTransformer(sapbert_model_name)
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
    return emb


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build FAISS index + embeddings + metadata from JSON files')
    parser.add_argument('--input-file', type=str,default=None, help='Single JSON file to process')
    parser.add_argument('--input-dir', type=str, default=INPUT_DIR, help='Directory containing JSON files (default: .)')
    parser.add_argument('--out-dir', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--embedding-model', type=str, default='all-mpnet-base-v2', help='SentenceTransformer model name for passage embeddings')
    parser.add_argument('--sapbert-model', type=str, default=None, help='Optional sentence-transformers SapBERT model name to compute sapbert embeddings')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--normalize', action='store_true', help='L2 normalize embeddings (recommended for cosine similarity)')
    args = parser.parse_args(argv)

    input_path = Path(args.input_file) if args.input_file else Path(args.input_dir)
    out_dir = Path(args.out_dir)

    all_passages = []

    # Load and parse
    for doc in load_json_files(input_path):
        try:
            passages = extract_passages_from_doc(doc)
            all_passages.extend(passages)
        except Exception as e:
            print(f"Error parsing document: {e}")

    print(f"Extracted {len(all_passages)} raw passages")

    # Dedupe
    passages = dedupe_passages(all_passages)
    print(f"After dedupe: {len(passages)} passages")

    if not passages:
        print('No passages found; exiting')
        return

    # Compute primary embeddings
    embeddings, model = compute_embeddings(passages, args.embedding_model, batch_size=args.batch_size, normalize=args.normalize)

    # Optional SapBERT embeddings
    sapbert_emb = None
    if args.sapbert_model:
        print('Computing optional SapBERT embeddings...')
        sapbert_emb = compute_sapbert_embeddings(passages, args.sapbert_model, batch_size=args.batch_size, normalize=args.normalize)

    # Save embeddings and metadata
    save_outputs(embeddings, passages, out_dir, sapbert_embeddings=sapbert_emb)

    # Build and write FAISS index
    try:
        idx = build_faiss_index(embeddings, out_dir)
        print(f"FAISS index written to {out_dir / 'index.faiss'}")
    except Exception as e:
        print(f"Failed to build/write FAISS index: {e}")

    print('Done. Outputs:')
    print(' -', out_dir / 'embeddings.npy')
    print(' -', out_dir / 'index.faiss')
    print(' -', out_dir / 'metadata.json')
    if sapbert_emb is not None:
        print(' -', out_dir / 'sapbert_embeddings.npy')


if __name__ == '__main__':
    main()
