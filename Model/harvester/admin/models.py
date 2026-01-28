"""
Model Management Module
Handles loading and initialization of all ML models used in the labeling system.
"""

from pathlib import Path
from typing import Optional, Tuple
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline


class ModelManager:
    """Manages loading and access to all ML models for passage labeling."""
    
    def __init__(self, index_dir: Path, config: dict):
        """
        Initialize model manager with configuration.
        
        Args:
            index_dir: Directory containing FAISS index and metadata
            config: Configuration dictionary with model paths and settings
        """
        self.index_dir = index_dir
        self.config = config
        
        # Models - initialized on demand
        self.metadata: Optional[list] = None
        self.index: Optional[faiss.Index] = None
        self.embed_model: Optional[SentenceTransformer] = None
        self.cross_encoder: Optional[CrossEncoder] = None
        self.nli_cross_encoder: Optional[CrossEncoder] = None
        self.sapbert_model: Optional[SentenceTransformer] = None
        self.sapbert_embeddings: Optional[np.ndarray] = None
        self.stance_classifier: Optional[any] = None
        self.concept_label_vectors: Optional[np.ndarray] = None
        self.concept_label_meta: Optional[list] = None
        
        self.initialized = False
        
    def initialize_all(self, skip_stance: bool = False) -> bool:
        """
        Load all required models and indices.
        
        Args:
            skip_stance: If True, skip loading stance detection models
            
        Returns:
            True if initialization successful, False otherwise
        """
        print("\n" + "="*60)
        print("Initializing Models and Indices")
        print("="*60)
        
        success = True
        success &= self._load_metadata()
        success &= self._load_faiss_index()
        success &= self._load_embedding_model()
        success &= self._load_cross_encoder()
        
        if not skip_stance:
            self._load_nli_cross_encoder()
            self._load_stance_classifier()
        else:
            print("\nSkipping stance detection models (not needed for workflow)")
            
        self._load_sapbert()
        self._load_concept_embeddings()
        
        if success:
            print("\n" + "="*60)
            print("✓ All models loaded successfully!")
            print("="*60)
            self.initialized = True
        else:
            print("\n" + "="*60)
            print("✗ Model initialization failed")
            print("="*60)
            
        return success
    
    def _load_metadata(self) -> bool:
        """Load passage metadata."""
        metadata_path = self.index_dir / 'metadata.json'
        if not metadata_path.exists():
            print(f"\nError: metadata.json not found at {metadata_path}")
            return False
        
        print("\nStep 1: Loading metadata...")
        with metadata_path.open('r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"  ✓ Loaded {len(self.metadata)} passages")
        return True
    
    def _load_faiss_index(self) -> bool:
        """Load FAISS vector index."""
        index_path = self.index_dir / 'index.faiss'
        if not index_path.exists():
            print(f"\nError: index.faiss not found at {index_path}")
            return False
        
        print("\nStep 2: Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))
        print(f"  ✓ FAISS index loaded")
        return True
    
    def _load_embedding_model(self) -> bool:
        """Load sentence embedding model."""
        print("\nStep 3: Loading embedding model...")
        model_name = self.config.get('EMBEDDING_MODEL')
        print(f"  Model: {model_name}")
        
        try:
            self.embed_model = SentenceTransformer(model_name)
            print(f"  ✓ Embedding model loaded")
            return True
        except Exception as e:
            print(f"  ✗ Failed to load embedding model: {e}")
            return False
    
    def _load_cross_encoder(self) -> bool:
        """Load relevance ranking cross-encoder."""
        print("\nStep 4: Loading cross-encoder for relevance ranking...")
        model_name = self.config.get('CROSS_ENCODER_MODEL')
        print(f"  Model: {model_name}")
        
        try:
            self.cross_encoder = CrossEncoder(model_name)
            print(f"  ✓ Cross-encoder loaded")
            return True
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load cross-encoder: {e}")
            return True  # Non-critical
    
    def _load_nli_cross_encoder(self):
        """Load NLI cross-encoder for stance detection."""
        print("\nStep 4b: Loading NLI cross-encoder for stance detection...")
        model_name = self.config.get('NLI_CROSS_ENCODER_MODEL')
        print(f"  Model: {model_name}")
        
        try:
            self.nli_cross_encoder = CrossEncoder(model_name)
            print(f"  ✓ NLI cross-encoder loaded")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load NLI cross-encoder: {e}")
    
    def _load_sapbert(self):
        """Load SapBERT medical entity embeddings."""
        sapbert_path = self.index_dir / 'sapbert_embeddings.npy'
        
        if sapbert_path.exists():
            print("\nStep 5: Loading SapBERT embeddings...")
            try:
                self.sapbert_embeddings = np.load(str(sapbert_path))
                print(f"  ✓ SapBERT embeddings loaded: {self.sapbert_embeddings.shape}")
                
                model_name = self.config.get('SAPBERT_MODEL')
                self.sapbert_model = SentenceTransformer(model_name)
                print(f"  ✓ SapBERT model loaded")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to load SapBERT: {e}")
        else:
            print("\nStep 5: SapBERT embeddings not found (optional)")
    
    def _load_stance_classifier(self):
        """Load MNLI stance classifier."""
        print("\nStep 6: Loading stance classifier (MNLI)...")
        model_name = self.config.get('STANCE_MODEL')
        print(f"  Model: {model_name}")
        
        try:
            self.stance_classifier = pipeline(
                task="text-classification",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True
            )
            print(f"  ✓ Stance classifier loaded")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load stance classifier: {e}")
            self.stance_classifier = None
        
        if self.stance_classifier is None:
            print("  ⚠ No stance classifier available; stance will require manual input.")
    
    def _load_concept_embeddings(self):
        """Load SapBERT concept label embeddings for query expansion."""
        labels_vec_path = Path(self.config.get('CONCEPT_LABELS_VECTORS'))
        labels_meta_path = Path(self.config.get('CONCEPT_LABELS_METADATA'))
        
        if self.sapbert_model and labels_vec_path.exists() and labels_meta_path.exists():
            try:
                self.concept_label_vectors = np.load(str(labels_vec_path))
                with labels_meta_path.open('r', encoding='utf-8') as f:
                    self.concept_label_meta = json.load(f)
                count = len(self.concept_label_meta) if self.concept_label_meta else 0
                print(f"  ✓ Concept label embeddings loaded: {self.concept_label_vectors.shape} (labels={count})")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to load concept embeddings: {e}")
        else:
            print("  ℹ Concept label embeddings not found (optional)")
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed text using the main embedding model.
        
        Args:
            text: Text to embed
            normalize: Whether to L2-normalize the vector
            
        Returns:
            Embedding vector
        """
        if self.embed_model is None:
            raise RuntimeError("Embedding model not initialized")
        
        vector = self.embed_model.encode([text], convert_to_numpy=True)[0]
        
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
        return vector
    
    def embed_with_sapbert(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Embed text using SapBERT medical model.
        
        Args:
            text: Text to embed
            normalize: Whether to L2-normalize the vector
            
        Returns:
            Embedding vector or None if SapBERT not available
        """
        if self.sapbert_model is None:
            return None
        
        vector = self.sapbert_model.encode([text], convert_to_numpy=True)[0]
        
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
        return vector
    
    def get_passage_count(self) -> int:
        """Get total number of passages in metadata."""
        return len(self.metadata) if self.metadata else 0
    
    def get_passage(self, index: int) -> Optional[dict]:
        """
        Get passage by index.
        
        Args:
            index: Passage index
            
        Returns:
            Passage dictionary or None if invalid index
        """
        if self.metadata and 0 <= index < len(self.metadata):
            return self.metadata[index]
        return None
