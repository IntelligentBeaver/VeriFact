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
CLAIM_INPUT_DIR = ADMIN_DIR / "input"
VERIFIED_CLAIMS_FILE = CLAIM_INPUT_DIR / "claims_mixture_unproven.json"
FAKE_CLAIMS_FILE = CLAIM_INPUT_DIR / "fake_claims.json"  # Intentionally false claims for refute labeling

# Indexing configuration (must match build_webmd_indexing_biomed.py)
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # For general relevance ranking
NLI_CROSS_ENCODER_MODEL = "cross-encoder/nli-deberta-v3-large"  # For stance detection only (entailment/contradiction)
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
STANCE_MODEL = "roberta-large-mnli"  # NLI model for stance

# Concept seed embeddings (SapBERT label vectors)
CONCEPT_LABELS_DIR = HARVESTER_DIR / "storage" / "seeds" / "embeddings"
CONCEPT_LABELS_VECTORS = CONCEPT_LABELS_DIR / "sapbert_embeddings.npy"
CONCEPT_LABELS_METADATA = CONCEPT_LABELS_DIR / "metadata.json"
CONCEPT_TOPK = 5

# Default retrieval parameters
DEFAULT_TOPK_FAISS = 100
DEFAULT_RERANK_K = 20
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

# Stance auto-labeling threshold (0.0-1.0). Above this, stance is auto-assigned.
STANCE_AUTO_THRESHOLD = 0.66

# Labeling output files
RELEVANT_PASSAGES_FILE = OUTPUT_DIR / "labeled_relevant_passages.json"
UNRELATED_PASSAGES_FILE = OUTPUT_DIR / "labeled_unrelated_passages.json"
QUESTION_PASSAGES_FILE = OUTPUT_DIR / "labeled_question_passages.json"
LABELING_SESSION_FILE = OUTPUT_DIR / "labeling_session.json"
FAKE_CLAIMS_LABELED_FILE = OUTPUT_DIR / "labeled_fake_claims_refutes.json"  # Passages that refute fake claims

# Create output directory if not exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
