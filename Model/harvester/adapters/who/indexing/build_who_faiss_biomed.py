#!/usr/bin/env python3
"""
Build FAISS Index for WHO News/Fact Sheets (One Time Setup)

This script mirrors build_webmd_faiss_biomed.py:
- Loads WHO JSON files (news or fact sheets)
- Splits each section into passage blocks
- Builds embeddings and a FAISS index
- Saves embeddings.npy, index.faiss, and metadata.json

Requirements:
    pip install sentence-transformers faiss-cpu numpy tqdm
"""

import json
import os
from pathlib import Path
from hashlib import sha1
from tqdm import tqdm
import numpy as np

# ============================================================================
# CONFIGURATION - Edit these values before running
# ============================================================================

# Input configuration - Choose one:
INPUT_FILE = None  # Set to a specific JSON file path, or leave as None to use INPUT_DIR
INPUT_DIR = "../../../storage/who/news"  # Directory containing JSON files

# Output configuration
OUTPUT_DIR = "../../../storage/outputs/who/faiss"

# Model configuration
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # Primary biomedical retrieval model
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # Set to None to disable

# Processing configuration
BATCH_SIZE = 32  # Number of passages to process at once
NORMALIZE_EMBEDDINGS = True  # Normalize vectors for cosine similarity (recommended: True)

# ============================================================================


def sha1_hex(s: str) -> str:
    return sha1(s.encode("utf-8")).hexdigest()


def normalize_sources(references):
    """Normalize WHO `references` into a comma-separated string of `(text,url)` items.

    Supports references as:
    - list of {text, url} dicts (news)
    - list of strings (fact sheets)
    - single dict or string
    """
    if not references:
        return None
    if isinstance(references, (str, dict)):
        references = [references]
    if not isinstance(references, list):
        return None

    items = []
    for ref in references:
        if isinstance(ref, dict):
            text = (ref.get("text") or "").strip()
            url = (ref.get("url") or "").strip()
            if not text and not url:
                continue
            items.append(f"({text},{url})")
        elif isinstance(ref, str):
            text = ref.strip()
            if text:
                items.append(f"({text},)")

    return ", ".join(items) if items else None


def load_json_files(input_path: Path):
    if input_path.is_file():
        print(f"Loading single file: {input_path}")
        with input_path.open("r", encoding="utf-8") as f:
            yield json.load(f)
    else:
        json_files = list(input_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in {input_path}")
        for p in tqdm(sorted(json_files), desc="Loading files"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    yield json.load(f)
            except Exception as e:
                print(f"\nWarning: Failed to load {p.name}: {e}")


def normalize_to_article_list(doc):
    if isinstance(doc, list):
        return [d for d in doc if isinstance(d, dict)]
    if isinstance(doc, dict):
        if any(k in doc for k in ("url", "title", "sections", "content", "published_date")):
            return [doc]
        return [v for v in doc.values() if isinstance(v, dict)]
    return []


def extract_passages_from_doc(doc: dict):
    doc_fields = {}
    for k in [
        "url",
        "title",
        "published_date",
        "scrape_timestamp_utc",
        'author',
        "medically_reviewed_by",
        "tags",
    ]:
        doc_fields[k] = doc.get(k)

    # WHO stores document references as a list of {text, url}; normalize them onto each passage.
    doc_fields["sources"] = normalize_sources(doc.get("references"))

    doc_id = sha1_hex(str(doc_fields.get("url") or doc_fields.get("title") or ""))
    passages = []

    sections = doc.get("content") or doc.get("sections") or []
    for si, sec in enumerate(sections):
        heading = sec.get("heading")
        section_heading = heading if heading not in (None, "") else doc_fields.get("title")
        content_blocks = sec.get("content") or []
        if isinstance(content_blocks, str):
            content_blocks = [content_blocks]

        for bi, blk in enumerate(content_blocks):
            if isinstance(blk, str):
                text = blk.strip()
            elif isinstance(blk, dict):
                text = (blk.get("text") or "").strip()
                bullets = blk.get("bullets") or blk.get("associated_bullets")
                if isinstance(bullets, str):
                    bullets = [bullets]
                elif not isinstance(bullets, list):
                    bullets = []

                if bullets:
                    bullets_text = "\n".join([str(b).strip() for b in bullets if b])
                    text = text + "\n" + bullets_text if text else bullets_text
            else:
                text = ""

            if not text:
                continue

            passage_id = f"{doc_id}_s{si}_b{bi}"
            p = {
                "passage_id": passage_id,
                "doc_id": doc_id,
                "section_heading": section_heading,
                "block_index": bi,
                "text": text,
            }
            p.update(doc_fields)
            passages.append(p)

    return passages


def dedupe_passages(passages):
    seen = set()
    out = []
    for p in passages:
        t = p["text"]
        if t in seen:
            continue
        seen.add(t)
        out.append(p)
    return out


def compute_embeddings(passages, model_name, batch_size=32, normalize=True):
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    texts = [p["text"] for p in passages]
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
    from sentence_transformers import SentenceTransformer

    print(f"Loading SapBERT model: {sapbert_model_name}")
    texts = [p["text"] for p in passages]
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


def build_faiss_index(embeddings, out_path: Path):
    try:
        import faiss
    except ImportError:
        raise RuntimeError("FAISS is required. Install via: pip install faiss-cpu")

    dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dim}...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = out_path / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved: {index_path}")
    return index


def save_outputs(embeddings, passages, out_dir: Path, sapbert_embeddings=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving outputs to: {out_dir}")
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    print(f"  ✓ Saved embeddings.npy ({embeddings.shape})")

    metadata = []
    for p in passages:
        meta = {
            "passage_id": p["passage_id"],
            "doc_id": p["doc_id"],
            "section_heading": p.get("section_heading"),
            "block_index": p.get("block_index"),
            "text": p["text"],
            "url": p.get("url"),
            "title": p.get("title"),
            "published_date": p.get("published_date"),
            'scrape_timestamp_utc': p.get('scrape_timestamp_utc'),
            "author": p.get("author"),
            "medically_reviewed_by": p.get("medically_reviewed_by") or 'N/A',
            "sources": p.get("sources"),
            "location": p.get("location"),
            "tags": p.get("tags"),
        }
        metadata.append(meta)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved metadata.json ({len(metadata)} passages)")

    if sapbert_embeddings is not None:
        np.save(out_dir / "sapbert_embeddings.npy", sapbert_embeddings.astype(np.float32))
        print(f"  ✓ Saved sapbert_embeddings.npy ({sapbert_embeddings.shape})")


def main():
    print("=" * 60)
    print("Building FAISS Index for WHO Retrieval")
    print("=" * 60)

    input_path = Path(INPUT_FILE) if INPUT_FILE else Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    print("\nConfiguration:")
    print(f"  Input: {input_path}")
    print(f"  Output: {out_dir}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  SapBERT model: {SAPBERT_MODEL or 'Disabled'}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Normalize: {NORMALIZE_EMBEDDINGS}")
    print()

    all_passages = []

    print("Step 1: Loading and parsing articles...")
    for doc in load_json_files(input_path):
        for article in normalize_to_article_list(doc):
            try:
                passages = extract_passages_from_doc(article)
                all_passages.extend(passages)
            except Exception as e:
                print(f"\nError parsing document: {e}")

    print(f"\nExtracted {len(all_passages)} raw passages from articles")

    print("\nStep 2: Removing duplicate passages...")
    passages = dedupe_passages(all_passages)
    print(f"Final passage count: {len(passages)} passages\n")

    if not passages:
        print("Error: No passages found to process. Check your input files.")
        return

    print("Step 3: Computing primary embeddings...")
    embeddings = compute_embeddings(passages, EMBEDDING_MODEL, batch_size=BATCH_SIZE, normalize=NORMALIZE_EMBEDDINGS)
    print(f"Created embeddings with shape: {embeddings.shape}\n")

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

    print("Step 5: Saving embeddings and metadata...")
    save_outputs(embeddings, passages, out_dir, sapbert_embeddings=sapbert_emb)
    print()

    print("Step 6: Building FAISS index...")
    try:
        build_faiss_index(embeddings, out_dir)
    except Exception as e:
        print(f"Error: Failed to build FAISS index: {e}")
        return

    print("\n" + "=" * 60)
    print("✓ Successfully completed!")
    print("=" * 60)
    print(f"\nOutput directory: {out_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  • embeddings.npy - Passage vectors ({embeddings.shape[0]} passages)")
    print("  • index.faiss - Fast similarity search index")
    print("  • metadata.json - Passage metadata")
    if sapbert_emb is not None:
        print(f"  • sapbert_embeddings.npy - Entity matching vectors ({sapbert_emb.shape[0]} passages)")


if __name__ == "__main__":
    main()
