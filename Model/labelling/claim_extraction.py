"""
Claim extraction utilities for generating atomic claims from passages.
"""

from __future__ import annotations

import re
from typing import List


_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP

    try:
        import spacy
    except Exception as exc:
        raise RuntimeError("spaCy is required for sentence splitting.") from exc

    try:
        _NLP = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    except Exception:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        )

    if "sentencizer" not in _NLP.pipe_names:
        _NLP.add_pipe("sentencizer")

    return _NLP


class SentenceSplitter:
    """Split text into sentences using spaCy."""

    def split(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        nlp = _get_nlp()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


class ClaimSentenceFilter:
    """Heuristic filter for claim-worthy sentences."""

    SKIP_PREFIXES = (
        "according to ",
        "as reported by ",
        "as noted by ",
        "for example ",
        "for instance ",
        "in summary ",
        "in conclusion ",
        "see ",
        "table ",
        "figure ",
    )

    PRONOUN_PREFIXES = (
        "this ",
        "these ",
        "it ",
        "they ",
        "such ",
        "those ",
        "that ",
    )

    MEDICAL_HINTS = (
        "disease",
        "syndrome",
        "infection",
        "cancer",
        "tumor",
        "virus",
        "bacteria",
        "symptom",
        "sign",
        "treatment",
        "therapy",
        "dose",
        "dosage",
        "risk",
        "causes",
        "caused",
        "leads",
        "results",
        "associated",
        "affects",
        "prevents",
        "vaccine",
        "drug",
        "medication",
        "side effect",
    )

    def is_claim_worthy(self, sentence: str) -> bool:
        text = (sentence or "").strip()
        if not text:
            return False

        lower = text.lower()
        if lower.endswith(":"):
            return False

        if lower.startswith(self.SKIP_PREFIXES):
            return False

        if lower.startswith(self.PRONOUN_PREFIXES):
            return False

        if len(text.split()) < 6:
            return False

        if any(hint in lower for hint in self.MEDICAL_HINTS):
            return True

        return False


class ClaimExtractor:
    """Extract atomic claims from a sentence using rule patterns."""

    PATTERNS = [
        (r"(?P<x>.+?) is caused by (?P<y>.+)", "{x} is caused by {y}."),
        (r"(?P<x>.+?) causes (?P<y>.+)", "{x} causes {y}."),
        (r"(?P<x>.+?) results in (?P<y>.+)", "{x} results in {y}."),
        (r"(?P<x>.+?) leads to (?P<y>.+)", "{x} leads to {y}."),
        (r"(?P<x>.+?) is associated with (?P<y>.+)", "{x} is associated with {y}."),
        (r"(?P<x>.+?) treats (?P<y>.+)", "{x} treats {y}."),
        (r"(?P<x>.+?) is used to treat (?P<y>.+)", "{x} treats {y}."),
        (r"(?P<x>.+?) affects (?P<y>.+)", "{x} affects {y}."),
        (r"(?P<x>.+?) is an? (?P<y>.+)", "{x} is a {y}."),
    ]

    CLAUSE_SPLITS = (" that ", " which ", " who ", " because ", " due to ", " while ")

    def extract(self, sentence: str) -> List[str]:
        text = (sentence or "").strip().rstrip(".")
        if not text:
            return []

        claims: List[str] = []

        for pattern, template in self.PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue

            subj = self._clean_clause(match.group("x"))
            obj = self._clean_clause(match.group("y"))
            if not subj or not obj:
                continue

            claim = template.format(x=subj, y=obj)
            claim = self._normalize_claim(claim)
            claims.append(claim)

        if not claims:
            claims.append(self._normalize_claim(text + "."))

        return self._dedupe(claims)

    def _clean_clause(self, text: str) -> str:
        cleaned = text.strip().strip(" ,;:")
        for splitter in self.CLAUSE_SPLITS:
            if splitter in cleaned:
                cleaned = cleaned.split(splitter, 1)[0].strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _normalize_claim(self, text: str) -> str:
        claim = re.sub(r"\s+", " ", text).strip()
        if not claim.endswith("."):
            claim += "."
        return claim

    def _dedupe(self, claims: List[str]) -> List[str]:
        seen = set()
        out = []
        for claim in claims:
            key = claim.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(claim)
        return out


class RefuteGenerator:
    """Generate simple REFUTES edits for a claim."""

    RULES = [
        (r" is caused by ", " is not caused by "),
        (r" causes ", " does not cause "),
        (r" results in ", " does not result in "),
        (r" leads to ", " does not lead to "),
        (r" is associated with ", " is not associated with "),
        (r" treats ", " does not treat "),
        (r" affects ", " does not affect "),
        (r" is a ", " is not a "),
        (r" is an ", " is not an "),
    ]

    def generate(self, claim: str) -> List[str]:
        claim = (claim or "").strip()
        if not claim:
            return []

        lower = claim.lower()
        refutes = []

        for needle, replacement in self.RULES:
            if needle in lower:
                refute = self._apply_replacement(claim, needle, replacement)
                if refute:
                    refutes.append(refute)
                break

        return refutes

    def _apply_replacement(self, text: str, needle: str, replacement: str) -> str:
        pattern = re.compile(re.escape(needle), flags=re.IGNORECASE)
        updated = pattern.sub(replacement, text, count=1)
        updated = re.sub(r"\s+", " ", updated).strip()
        if not updated.endswith("."):
            updated += "."
        return updated
