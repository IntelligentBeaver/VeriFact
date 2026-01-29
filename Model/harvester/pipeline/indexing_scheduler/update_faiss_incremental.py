#!/usr/bin/env python3
"""
Incremental FAISS Index Updater

This script processes ONLY new articles scraped since the last run:
- Tracks processed passage_ids in checkpoint.json
- Computes embeddings only for new passages
- Merges new embeddings with existing ones
- Updates FAISS index incrementally
- Appends to metadata.json

Usage:
    python update_faiss_incremental.py           # Process new articles only
    python update_faiss_incremental.py --rebuild # Force full rebuild
"""

import json
from pathlib import Path
from hashlib import sha1
from typing import List, Dict, Any, Set
from datetime import datetime
import numpy as np
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]

SOURCES = [
    {
        "name": "webmd_articles",
        "type": "webmd",
        "input_dir": ROOT / "storage" / "webmd" / "articles",
    },
    {
        "name": "webmd_healthtopics",
        "type": "webmd",
        "input_dir": ROOT / "storage" / "webmd" / "healthtopics",
    },
    {
        "name": "who_news",
        "type": "who",
        "input_dir": ROOT / "storage" / "who" / "news",
    },
    {
        "name": "who_fact_sheets",
        "type": "who",
        "input_dir": ROOT / "storage" / "who" / "fact_sheets",
    },
    {
        "name": "who_disease_outbreak",
        "type": "who",
        "input_dir": ROOT / "adapters" / "who" / "scraping" / "storage" / "disease_outbreak_news",
    },
    {
        "name": "who_feature_stories",
        "type": "who",
        "input_dir": ROOT / "adapters" / "who" / "scraping" / "storage" / "feature_stories",
    },
]

OUTPUT_DIR = ROOT / "storage" / "outputs" / "combined"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # Set to None to disable
BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True

# ============================================================================


def sha1_hex(s: str) -> str:
    return sha1(s.encode("utf-8")).hexdigest()


def normalize_sources(references):
    """Normalize references into comma-separated "(text,url)" or "(text,)" entries."""
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
            if text or url:
                items.append(f"({text},{url})")
        elif isinstance(ref, str):
            text = ref.strip()
            if text:
                items.append(f"({text},)")
    return ", ".join(items) if items else None


def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint containing processed passage_ids."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "processed_passage_ids": [],
        "last_update": None,
        "total_passages": 0,
    }


def save_checkpoint(checkpoint: Dict[str, Any]):
    """Save checkpoint after successful update."""
    checkpoint["last_update"] = datetime.utcnow().isoformat() + "Z"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def load_json_files(input_path: Path):
    """Generator that yields JSON documents from file or directory."""
    if input_path.is_file():
        with input_path.open("r", encoding="utf-8") as f:
            yield json.load(f)
    elif input_path.is_dir():
        json_files = list(input_path.glob("*.json"))
        for p in sorted(json_files):
            try:
                with p.open("r", encoding="utf-8") as f:
                    yield json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")


def normalize_to_article_list(doc: Any) -> List[Dict[str, Any]]:
    """Normalize various JSON structures to list of article dicts."""
    if isinstance(doc, list):
        return [d for d in doc if isinstance(d, dict)]
    if isinstance(doc, dict):
        if any(k in doc for k in ("url", "title", "sections", "content", "published_date")):
            return [doc]
        return [v for v in doc.values() if isinstance(v, dict)]
    return []


def extract_passages_webmd(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract passages from WebMD document."""
    doc_fields = {}
    for k in [
        "url",
        "title",
        "published_date",
        "scrape_timestamp_utc",
        "author",
        "medically_reviewed_by",
        "sources",
        "tags",
    ]:
        doc_fields[k] = doc.get(k)

    doc_id = sha1_hex(str(doc_fields.get("url") or doc_fields.get("title") or ""))
    passages = []

    sections = doc.get("sections") or []
    for si, sec in enumerate(sections):
        heading = sec.get("heading")
        content_blocks = sec.get("content_blocks") or []
        for bi, blk in enumerate(content_blocks):
            text = (blk.get("text") or "").strip()
            bullets = blk.get("associated_bullets")
            if bullets:
                bullets_text = "\n".join([b.strip() for b in bullets if b])
                full_text = text + "\n" + bullets_text if text else bullets_text
            else:
                full_text = text
            if not full_text:
                continue
            passage_id = f"{doc_id}_s{si}_b{bi}"
            p = {
                "passage_id": passage_id,
                "doc_id": doc_id,
                "section_heading": heading,
                "block_index": bi,
                "text": full_text,
            }
            p.update(doc_fields)
            passages.append(p)
    return passages


def extract_passages_who(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract passages from WHO document."""
    doc_fields = {}
    for k in [
        "url",
        "title",
        "published_date",
        "scrape_timestamp_utc",
        "author",
        "medically_reviewed_by",
        "tags",
        "location",
    ]:
        doc_fields[k] = doc.get(k)
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


def dedupe_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate passages based on text content."""
    seen = set()
    out = []
    for p in passages:
        t = p.get("text")
        if t in seen:
            continue
        seen.add(t)
        out.append(p)
    return out


def filter_new_passages(
    passages: List[Dict[str, Any]], 
    processed_passage_ids: Set[str]
) -> List[Dict[str, Any]]:
    """Filter out passages that have already been processed."""
    new_passages = [
        p for p in passages 
        if p["passage_id"] not in processed_passage_ids
    ]
    return new_passages


def compute_embeddings(passages, model_name, batch_size=32, normalize=True):
    """Compute embeddings for passages using specified model."""
    from sentence_transformers import SentenceTransformer

    if not passages:
        return np.array([]).reshape(0, 768)  # Empty array with correct shape
    
    texts = [p["text"] for p in passages]
    model = SentenceTransformer(model_name)
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


def load_existing_data():
    """Load existing embeddings, metadata, and FAISS index."""
    embeddings_path = OUTPUT_DIR / "embeddings.npy"
    metadata_path = OUTPUT_DIR / "metadata.json"
    sapbert_path = OUTPUT_DIR / "sapbert_embeddings.npy"
    index_path = OUTPUT_DIR / "index.faiss"
    
    existing_embeddings = None
    existing_metadata = []
    existing_sapbert = None
    existing_index = None
    
    if embeddings_path.exists():
        existing_embeddings = np.load(embeddings_path)
        print(f"✓ Loaded existing embeddings: {existing_embeddings.shape}")
    
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
        print(f"✓ Loaded existing metadata: {len(existing_metadata)} passages")
    
    if sapbert_path.exists():
        existing_sapbert = np.load(sapbert_path)
        print(f"✓ Loaded existing SapBERT embeddings: {existing_sapbert.shape}")
    
    if index_path.exists():
        try:
            import faiss
            existing_index = faiss.read_index(str(index_path))
            print(f"✓ Loaded existing FAISS index: {existing_index.ntotal} vectors")
        except ImportError:
            print("⚠️  FAISS not installed, cannot load index")
        except Exception as e:
            print(f"⚠️  Failed to load FAISS index: {e}")
    
    return existing_embeddings, existing_metadata, existing_sapbert, existing_index


def merge_and_save(
    new_passages: List[Dict[str, Any]],
    new_embeddings: np.ndarray,
    new_sapbert: np.ndarray = None,
):
    """Merge new data with existing and save."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing data
    existing_embeddings, existing_metadata, existing_sapbert, existing_index = load_existing_data()
    
    # Merge embeddings
    if existing_embeddings is not None and existing_embeddings.shape[0] > 0:
        merged_embeddings = np.vstack([existing_embeddings, new_embeddings])
        print(f"✓ Merged embeddings: {existing_embeddings.shape[0]} old + {new_embeddings.shape[0]} new = {merged_embeddings.shape[0]} total")
    else:
        merged_embeddings = new_embeddings
        print(f"✓ Created new embeddings array: {merged_embeddings.shape[0]} passages")
    
    # Merge metadata
    new_metadata = []
    for p in new_passages:
        meta = {
            "passage_id": p.get("passage_id"),
            "doc_id": p.get("doc_id"),
            "section_heading": p.get("section_heading"),
            "block_index": p.get("block_index"),
            "text": p.get("text"),
            "url": p.get("url"),
            "title": p.get("title"),
            "published_date": p.get("published_date"),
            "scrape_timestamp_utc": p.get("scrape_timestamp_utc"),
            "author": p.get("author"),
            "medically_reviewed_by": p.get("medically_reviewed_by") or "N/A",
            "sources": p.get("sources"),
            "location": p.get("location"),
            "tags": p.get("tags"),
        }
        new_metadata.append(meta)
    
    merged_metadata = existing_metadata + new_metadata
    
    # Save merged data
    print(f"\n💾 Saving merged data...")
    np.save(OUTPUT_DIR / "embeddings.npy", merged_embeddings.astype(np.float32))
    print(f"✓ Saved embeddings.npy ({merged_embeddings.shape[0]} passages)")
    
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(merged_metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved metadata.json ({len(merged_metadata)} passages)")
    
    # Handle SapBERT embeddings
    if new_sapbert is not None:
        if existing_sapbert is not None and existing_sapbert.shape[0] > 0:
            merged_sapbert = np.vstack([existing_sapbert, new_sapbert])
        else:
            merged_sapbert = new_sapbert
        np.save(OUTPUT_DIR / "sapbert_embeddings.npy", merged_sapbert.astype(np.float32))
        print(f"✓ Saved sapbert_embeddings.npy ({merged_sapbert.shape[0]} passages)")
    
    # Update FAISS index
    print("\n🔍 Updating FAISS index...")
    try:
        import faiss
        
        if existing_index is not None and new_embeddings.shape[0] > 0:
            # Add new vectors to existing index
            existing_index.add(new_embeddings.astype(np.float32))
            faiss.write_index(existing_index, str(OUTPUT_DIR / "index.faiss"))
            print(f"✓ Updated FAISS index: {existing_index.ntotal} total vectors")
        else:
            # Create new index from scratch
            dim = merged_embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(merged_embeddings.astype(np.float32))
            faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
            print(f"✓ Created new FAISS index: {index.ntotal} vectors")
    except ImportError:
        print("⚠️  FAISS not installed, skipping index creation")
        print("   Install with: pip install faiss-cpu")
    except Exception as e:
        print(f"⚠️  Failed to update FAISS index: {e}")


def collect_all_passages() -> List[Dict[str, Any]]:
    """Collect passages from all sources."""
    all_passages = []
    
    for src in SOURCES:
        input_dir = Path(src["input_dir"])
        if not input_dir.exists():
            print(f"⊘ Skipping {src['name']}: directory not found")
            continue
        
        print(f"📂 Loading {src['name']} from {input_dir.name}/")
        
        if src["type"] == "webmd":
            for doc in load_json_files(input_dir):
                all_passages.extend(extract_passages_webmd(doc))
        elif src["type"] == "who":
            for doc in load_json_files(input_dir):
                for article in normalize_to_article_list(doc):
                    all_passages.extend(extract_passages_who(article))
    
    return all_passages


def main():
    parser = argparse.ArgumentParser(description="Incrementally update FAISS index with new articles")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force full rebuild (ignore checkpoint)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Incremental FAISS Index Updater")
    print("=" * 70)
    
    # Load checkpoint
    if args.rebuild:
        print("\n🔄 REBUILD MODE: Processing all articles from scratch\n")
        checkpoint = {
            "processed_passage_ids": [],
            "last_update": None,
            "total_passages": 0,
        }
    else:
        checkpoint = load_checkpoint()
        if checkpoint["last_update"]:
            print(f"\n📋 Last update: {checkpoint['last_update']}")
            print(f"📊 Previously processed: {checkpoint['total_passages']} passages\n")
        else:
            print("\n📋 No previous checkpoint found - first run\n")
    
    processed_passage_ids = set(checkpoint["processed_passage_ids"])
    
    # Collect all passages
    print("📥 Scanning all sources for articles...")
    print("-" * 70)
    all_passages = collect_all_passages()
    print("-" * 70)
    print(f"📚 Found {len(all_passages)} total passages across all sources")
    
    # Deduplicate
    print("\n🔍 Deduplicating by text content...")
    all_passages = dedupe_passages(all_passages)
    print(f"✓ After deduplication: {len(all_passages)} unique passages")
    
    # Filter for new passages only
    new_passages = filter_new_passages(all_passages, processed_passage_ids)
    print(f"\n✨ NEW passages to process: {len(new_passages)}")
    
    if not new_passages:
        print("\n" + "=" * 70)
        print("✅ No new articles found. Index is up to date!")
        print("=" * 70)
        return
    
    # Compute embeddings for new passages
    print(f"\n🧠 Computing embeddings...")
    print("-" * 70)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Normalize: {NORMALIZE_EMBEDDINGS}")
    print("-" * 70)
    
    new_embeddings = compute_embeddings(
        new_passages, 
        EMBEDDING_MODEL, 
        batch_size=BATCH_SIZE, 
        normalize=NORMALIZE_EMBEDDINGS
    )
    print(f"✓ Primary embeddings computed: {new_embeddings.shape}")
    
    # Compute SapBERT embeddings if enabled
    new_sapbert = None
    if SAPBERT_MODEL:
        try:
            print(f"\n🧬 Computing SapBERT embeddings...")
            print(f"Model: {SAPBERT_MODEL}")
            new_sapbert = compute_embeddings(
                new_passages, 
                SAPBERT_MODEL, 
                batch_size=BATCH_SIZE, 
                normalize=NORMALIZE_EMBEDDINGS
            )
            print(f"✓ SapBERT embeddings computed: {new_sapbert.shape}")
        except Exception as e:
            print(f"⚠️  SapBERT computation failed: {e}")
            print("   Continuing without SapBERT embeddings")
    
    # Merge and save
    merge_and_save(new_passages, new_embeddings, new_sapbert)
    
    # Update checkpoint
    print(f"\n📝 Updating checkpoint...")
    new_passage_ids = [p["passage_id"] for p in new_passages]
    checkpoint["processed_passage_ids"].extend(new_passage_ids)
    checkpoint["total_passages"] += len(new_passages)
    save_checkpoint(checkpoint)
    print(f"✓ Checkpoint saved")
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully processed {len(new_passages)} new passages")
    print(f"📊 Total passages in index: {checkpoint['total_passages']}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
