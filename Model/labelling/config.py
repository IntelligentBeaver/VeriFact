"""
Labelling Configuration - Centralized settings for labelling tools
"""

from pathlib import Path

# Paths (labelling system lives at project root)
ADMIN_DIR = Path(__file__).parent
PROJECT_ROOT = ADMIN_DIR.parent
HARVESTER_DIR = PROJECT_ROOT / "harvester"

# Retrieval storage (used by retrieval/simple_retriever.py)
INDEXING_DIR = HARVESTER_DIR / "adapters" / "webmd" / "indexing"
INDEX_DIR = PROJECT_ROOT / "storage"
OUTPUT_DIR = ADMIN_DIR / "output"
CLAIM_INPUT_DIR = ADMIN_DIR / "input"
VERIFIED_CLAIMS_FILE = CLAIM_INPUT_DIR / "claims_unlabeled.json"

# Unlabeled-claim stance labeling
UNLABELED_CLAIMS_OUTPUT_FILE = OUTPUT_DIR / "labeled_unlabeled_claims.json"
UNLABELED_TOPK = 10
NLI_CROSS_ENCODER_MODEL = "cross-encoder/nli-deberta-v3-base"

# Default retrieval parameters
DEFAULT_DISPLAY_RESULTS = 5  # Show top 5 by default

# Auto-labeling thresholds (0.0-1.0 scale)
# Tip: After substantive passage filtering + FAISS capping + CE normalization,
# if too many passages require manual review, lower AUTO_RELEVANT_THRESHOLD to 0.70
AUTO_RELEVANT_THRESHOLD = 0.66  # Auto-label as relevant if score >= this
AUTO_UNRELATED_THRESHOLD = 0.52  # Auto-label as unrelated if score <= this
# Scores between thresholds require human review

# Fully automated labeling (no human prompts)
AUTO_LABELING_MODE = False

# Extra guardrails: auto-unrelated when both CE and lexical overlap are very low
AUTO_UNRELATED_CE_MAX = 0.10
AUTO_UNRELATED_LEX_MAX = 0.40

# Labeling output files
RELEVANT_PASSAGES_FILE = OUTPUT_DIR / "labeled_relevant_passages.json"
UNRELATED_PASSAGES_FILE = OUTPUT_DIR / "labeled_unrelated_passages.json"
QUESTION_PASSAGES_FILE = OUTPUT_DIR / "labeled_question_passages.json"
LABELING_SESSION_FILE = OUTPUT_DIR / "labeling_session.json"
CLAIMS_OUTPUT_FILE = OUTPUT_DIR / "generated_claims.json"
CLAIMS_EXPORT_STATE_FILE = OUTPUT_DIR / "claims_export_state.json"
QUERIES_OUTPUT_FILE = OUTPUT_DIR / "extracted_queries.json"
QUERIES_NEGATED_OUTPUT_FILE = OUTPUT_DIR / "extracted_queries_negated.json"

# Create output directory if not exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
