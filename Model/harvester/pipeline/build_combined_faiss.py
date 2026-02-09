#!/usr/bin/env python3
"""
Build a unified FAISS index across WebMD and WHO content (news + fact sheets).

Outputs in one directory:
- embeddings.npy (primary model)
- sapbert_embeddings.npy (optional)
- index.faiss
- metadata.json (aligned with embeddings order)

Configure input/output paths below. Defaults are relative to the project root
(<repo>/harvester).
"""

import json
from pathlib import Path
from hashlib import sha1
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    {
        "name": "webmd",
        "type": "webmd",
        "input_dir": ROOT / "storage" / "webmd" / "articles",
    },
    {
        "name": "webmd",
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
]

OUTPUT_DIR = ROOT / "storage" / "outputs" / "combined"

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


def load_json_files(input_path: Path):
    if input_path.is_file():
        with input_path.open("r", encoding="utf-8") as f:
            yield json.load(f)
    elif input_path.is_dir():
        json_files = list(input_path.glob("*.json"))
        for p in tqdm(sorted(json_files), desc=f"Loading {input_path.name}"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    yield json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")


def normalize_to_article_list(doc: Any) -> List[Dict[str, Any]]:
    if isinstance(doc, list):
        return [d for d in doc if isinstance(d, dict)]
    if isinstance(doc, dict):
        if any(k in doc for k in ("url", "title", "sections", "content", "published_date")):
            return [doc]
        return [v for v in doc.values() if isinstance(v, dict)]
    return []


def extract_passages_webmd(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    seen = set()
    out = []
    for p in passages:
        t = p.get("text")
        if t in seen:
            continue
        seen.add(t)
        out.append(p)
    return out


def compute_embeddings(passages, model_name, batch_size=32, normalize=True):
    from sentence_transformers import SentenceTransformer

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


def build_faiss_index(embeddings, out_path: Path):
    try:
        import faiss
    except ImportError:
        raise RuntimeError("FAISS is required. Install via: pip install faiss-cpu")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    index_path = out_path / "index.faiss"
    faiss.write_index(index, str(index_path))
    return index


def save_outputs(embeddings, passages, out_dir: Path, sapbert_embeddings=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))

    metadata = []
    for p in passages:
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
        metadata.append(meta)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if sapbert_embeddings is not None:
        np.save(out_dir / "sapbert_embeddings.npy", sapbert_embeddings.astype(np.float32))


def collect_passages_from_source(source_cfg) -> List[Dict[str, Any]]:
    input_dir = Path(source_cfg["input_dir"])
    if not input_dir.exists():
        print(f"Skipping {source_cfg['name']}: missing {input_dir}")
        return []

    print(f"Loading {source_cfg['name']} from {input_dir}")
    passages = []
    if source_cfg["type"] == "webmd":
        for doc in load_json_files(input_dir):
            passages.extend(extract_passages_webmd(doc))
    elif source_cfg["type"] == "who":
        for doc in load_json_files(input_dir):
            for article in normalize_to_article_list(doc):
                passages.extend(extract_passages_who(article))
    else:
        print(f"Unknown source type: {source_cfg['type']}")
    return passages


def main():
    print("=" * 60)
    print("Building unified FAISS index (WebMD + WHO)")
    print("=" * 60)

    all_passages: List[Dict[str, Any]] = []
    for src in SOURCES:
        all_passages.extend(collect_passages_from_source(src))

    print(f"Collected {len(all_passages)} raw passages")

    print("Deduplicating passages...")
    passages = dedupe_passages(all_passages)
    print(f"Passages after dedupe: {len(passages)}")

    if not passages:
        print("No passages found. Check input paths.")
        return

    print("Computing primary embeddings...")
    embeddings = compute_embeddings(passages, EMBEDDING_MODEL, batch_size=BATCH_SIZE, normalize=NORMALIZE_EMBEDDINGS)
    print(f"Primary embeddings: {embeddings.shape}")

    sapbert_emb = None
    if SAPBERT_MODEL:
        try:
            print("Computing SapBERT embeddings...")
            sapbert_emb = compute_embeddings(passages, SAPBERT_MODEL, batch_size=BATCH_SIZE, normalize=NORMALIZE_EMBEDDINGS)
            print(f"SapBERT embeddings: {sapbert_emb.shape}")
        except Exception as e:
            print(f"SapBERT failed ({e}); continuing without.")
            sapbert_emb = None

    print(f"Saving outputs to {OUTPUT_DIR}")
    save_outputs(embeddings, passages, OUTPUT_DIR, sapbert_embeddings=sapbert_emb)

    print("Building FAISS index...")
    build_faiss_index(embeddings, OUTPUT_DIR)

    print("Done. Files written:")
    print(f"  - {OUTPUT_DIR / 'embeddings.npy'}")
    print(f"  - {OUTPUT_DIR / 'index.faiss'}")
    print(f"  - {OUTPUT_DIR / 'metadata.json'}")
    if sapbert_emb is not None:
        print(f"  - {OUTPUT_DIR / 'sapbert_embeddings.npy'}")


if __name__ == "__main__":
    main()
