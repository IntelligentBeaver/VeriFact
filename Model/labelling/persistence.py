"""
Data Persistence Module
Handles loading and saving of labeled data, claims, and session information.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime


class LabeledDataStore:
    """Manages storage and retrieval of labeled passages."""
    
    def __init__(
        self,
        relevant_file: Path,
        unrelated_file: Path,
        question_file: Path,
        session_file: Path,
        output_dir: Path
    ):
        """
        Initialize data store with file paths.
        
        Args:
            relevant_file: Path to relevant passages JSON
            unrelated_file: Path to unrelated passages JSON
            question_file: Path to question passages JSON
            session_file: Path to session log JSON
            output_dir: Output directory for exports
        """
        self.relevant_file = relevant_file
        self.unrelated_file = unrelated_file
        self.question_file = question_file
        self.session_file = session_file
        self.output_dir = output_dir
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cached data
        self.relevant_passages: List[Dict] = []
        self.unrelated_passages: List[Dict] = []
        self.question_passages: List[Dict] = []
        
        # Load existing data
        self.load_all()
    
    def load_all(self):
        """Load all labeled data from disk."""
        self.relevant_passages = self._load_json(self.relevant_file)
        self.unrelated_passages = self._load_json(self.unrelated_file)
        self.question_passages = self._load_json(self.question_file)
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """
        Load JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data or empty list if file doesn't exist
        """
        if file_path.exists():
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        return []
    
    def save_labeled_passage(
        self,
        passage: Dict,
        query: str,
        label: str,
        faiss_score: float,
        rerank_score: float,
        stance: Optional[str] = None,
        stance_confidence: Optional[float] = None
    ) -> Dict:
        """
        Save labeled passage to appropriate file.
        
        Args:
            passage: Passage dictionary with metadata
            query: Query/claim text
            label: Label ('relevant', 'unrelated', or 'question')
            faiss_score: FAISS similarity score
            rerank_score: Cross-encoder score
            stance: Stance label (for relevant passages)
            stance_confidence: Stance confidence (for relevant passages)
            
        Returns:
            Labeled item dictionary
        """
        labeled_item = {
            'passage_id': passage.get('passage_id'),
            'query': query,
            'label': label,
            'stance': stance if label == 'relevant' else None,
            'stance_confidence': float(stance_confidence) if (
                stance_confidence is not None and label == 'relevant'
            ) else None,
            'text': passage.get('text'),
            'section_heading': passage.get('section_heading'),
            'title': passage.get('title'),
            'url': passage.get('url'),
            'author': passage.get('author'),
            'medically_reviewed_by': passage.get('medically_reviewed_by'),
            'sources': passage.get('sources', []),
            'published_date': passage.get('published_date'),
            'scores': {
                'faiss_similarity': float(faiss_score),
                'cross_encoder_relevance': float(rerank_score)
            },
            'labeled_at': datetime.now().isoformat()
        }
        
        # Append to appropriate list and save
        if label == 'relevant':
            self.relevant_passages.append(labeled_item)
            self._save_json(self.relevant_passages, self.relevant_file)
        elif label == 'unrelated':
            self.unrelated_passages.append(labeled_item)
            self._save_json(self.unrelated_passages, self.unrelated_file)
        elif label == 'question':
            self.question_passages.append(labeled_item)
            self._save_json(self.question_passages, self.question_file)
        
        return labeled_item
    
    def save_session(self, session_data: Dict):
        """
        Save labeling session data.
        
        Args:
            session_data: Session dictionary with metadata and labeled passages
        """
        self._save_json(session_data, self.session_file)
    
    def _save_json(self, data: any, file_path: Path):
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            file_path: Path to save to
        """
        try:
            with self._open_output_file(file_path) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")
    
    def _open_output_file(self, path_obj: Path):
        """
        Open output file safely on Windows paths.
        
        Args:
            path_obj: Path object
            
        Returns:
            File handle
        """
        try:
            return path_obj.open('w', encoding='utf-8')
        except OSError as e:
            if os.name == "nt":
                # Try with Windows long path prefix
                long_path = r"\\?\\" + str(path_obj)
                return open(long_path, 'w', encoding='utf-8')
            raise e
    
    def get_all_labeled(self, subset: str = "all") -> List[Dict]:
        """
        Get labeled passages by subset.
        
        Args:
            subset: One of 'all', 'relevant', 'unrelated', 'question', 'rel+unrel'
            
        Returns:
            List of labeled passage dictionaries
        """
        subset = (subset or "all").strip().lower()
        
        if subset == "relevant":
            return list(self.relevant_passages)
        if subset == "unrelated":
            return list(self.unrelated_passages)
        if subset == "question":
            return list(self.question_passages)
        if subset in {"rel+unrel", "relevant+unrelated", "relevant_unrelated"}:
            return list(self.relevant_passages) + list(self.unrelated_passages)
        
        # Default: all
        return (
            list(self.relevant_passages) +
            list(self.unrelated_passages) +
            list(self.question_passages)
        )
    
    def get_statistics(self) -> Dict:
        """
        Get labeling statistics.
        
        Returns:
            Dictionary with counts and stance breakdown
        """
        stats = {
            'relevant_count': len(self.relevant_passages),
            'unrelated_count': len(self.unrelated_passages),
            'question_count': len(self.question_passages),
            'total_count': (
                len(self.relevant_passages) +
                len(self.unrelated_passages) +
                len(self.question_passages)
            ),
            'stance_breakdown': self._get_stance_breakdown()
        }
        return stats
    
    def _get_stance_breakdown(self) -> Dict:
        """Get breakdown of stance labels for relevant passages."""
        supports = sum(
            1 for p in self.relevant_passages
            if p.get('stance') == 'supports'
        )
        refutes = sum(
            1 for p in self.relevant_passages
            if p.get('stance') == 'refutes'
        )
        neutral = sum(
            1 for p in self.relevant_passages
            if p.get('stance') == 'neutral'
        )
        
        return {
            'supports': supports,
            'refutes': refutes,
            'neutral': neutral
        }


class ClaimsLoader:
    """Loads and manages verified and fake claims."""
    
    @staticmethod
    def load_claims(file_path: Path) -> List[Dict]:
        """
        Load claims from JSON file.
        
        Args:
            file_path: Path to claims JSON file
            
        Returns:
            List of claim dictionaries with 'id' and 'claim' keys
        """
        if not file_path.exists():
            print(f"Claims file not found at {file_path}")
            return []
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            claims = []
            for item in data:
                if isinstance(item, dict):
                    text = item.get('claim') or item.get('text')
                    if text:
                        claims.append({
                            "id": item.get("id"),
                            "claim": text.strip()
                        })
                elif isinstance(item, str):
                    claims.append({"id": None, "claim": item.strip()})
            
            return [c for c in claims if c.get("claim")]
        
        except Exception as e:
            print(f"Failed to read claims from {file_path}: {e}")
            return []


class ProcessedClaimTracker:
    """Tracks which claims have been processed to avoid duplication."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize tracker.
        
        Args:
            output_dir: Directory to store tracking file
        """
        self.tracking_file = output_dir / "processed_claim_ids.json"
        self.processed_ids: Set[str] = set()
        self.load()
    
    def load(self):
        """Load processed claim IDs from disk."""
        if not self.tracking_file.exists():
            return
        
        try:
            with self.tracking_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.processed_ids = set(data) if isinstance(data, list) else set()
        except Exception:
            self.processed_ids = set()
    
    def save(self):
        """Save processed claim IDs to disk."""
        try:
            with self.tracking_file.open("w", encoding="utf-8") as f:
                json.dump(sorted(self.processed_ids), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save processed claim IDs: {e}")
    
    def is_processed(self, claim_id: Optional[str]) -> bool:
        """
        Check if claim has been processed.
        
        Args:
            claim_id: Claim ID to check
            
        Returns:
            True if already processed, False otherwise
        """
        return claim_id is not None and claim_id in self.processed_ids
    
    def mark_processed(self, claim_id: Optional[str]):
        """
        Mark claim as processed.
        
        Args:
            claim_id: Claim ID to mark
        """
        if claim_id is not None:
            self.processed_ids.add(claim_id)
