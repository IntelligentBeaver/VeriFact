"""
Admin Configuration - Centralized settings for all admin tools
"""

from pathlib import Path

# Paths (admin is now in harvester directory)
ADMIN_DIR = Path(__file__).parent
HARVESTER_DIR = ADMIN_DIR.parent
PROJECT_ROOT = HARVESTER_DIR.parent

INDEXING_DIR = HARVESTER_DIR / "adapters" / "webmd" / "indexing"
INDEX_DIR = HARVESTER_DIR / "storage" / "outputs" / "webmd" / "faiss"
OUTPUT_DIR = ADMIN_DIR / "output"

# Indexing configuration (must match build_webmd_indexing_biomed.py)
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Default retrieval parameters
DEFAULT_TOPK_FAISS = 100
DEFAULT_RERANK_K = 20
DEFAULT_DISPLAY_RESULTS = 10  # Show top 10 by default

# Auto-labeling thresholds (0.0-1.0 scale)
AUTO_RELEVANT_THRESHOLD = 0.75  # Auto-label as relevant if score >= this
AUTO_UNRELATED_THRESHOLD = 0.35  # Auto-label as unrelated if score <= this
# Scores between thresholds require human review

# Labeling output files
RELEVANT_PASSAGES_FILE = OUTPUT_DIR / "labeled_relevant_passages.json"
UNRELATED_PASSAGES_FILE = OUTPUT_DIR / "labeled_unrelated_passages.json"
QUESTION_PASSAGES_FILE = OUTPUT_DIR / "labeled_question_passages.json"
LABELING_SESSION_FILE = OUTPUT_DIR / "labeling_session.json"

# Create output directory if not exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
