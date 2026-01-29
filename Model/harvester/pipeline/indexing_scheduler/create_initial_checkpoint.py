#!/usr/bin/env python3
"""
Create initial checkpoint from existing metadata.json
Run this ONCE after your first build_combined_faiss.py run.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "storage" / "outputs" / "combined"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

def create_initial_checkpoint():
    if CHECKPOINT_FILE.exists():
        print(f"❌ Checkpoint already exists at {CHECKPOINT_FILE}")
        print("   Delete it first if you want to recreate it.")
        return
    
    if not METADATA_FILE.exists():
        print(f"❌ Metadata file not found at {METADATA_FILE}")
        print("   Run build_combined_faiss.py first.")
        return
    
    print(f"📖 Reading metadata from {METADATA_FILE}...")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    passage_ids = [entry["passage_id"] for entry in metadata]
    
    checkpoint = {
        "processed_passage_ids": passage_ids,
        "last_update": None,  # Set to None since it was initial build
        "total_passages": len(passage_ids),
    }
    
    print(f"💾 Creating checkpoint with {len(passage_ids)} passages...")
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"✅ Checkpoint created successfully!")
    print(f"   File: {CHECKPOINT_FILE}")
    print(f"   Passages tracked: {len(passage_ids)}")

if __name__ == "__main__":
    create_initial_checkpoint()
