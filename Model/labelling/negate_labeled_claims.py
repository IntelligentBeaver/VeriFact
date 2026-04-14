import json
from pathlib import Path
from typing import Dict, List

from claim_extraction import RefuteGenerator, set_negator_model
from config import (
    LABELED_CLAIMS_FILE,
    LABELED_CLAIMS_NEGATED_FILE,
    NEGATION_MODEL_LARGE,
    OUTPUT_DIR
)

SAVE_EVERY = 10
LOG_EVERY = 25

LABEL_MAP = {
    "true": "false",
    "false": "true",
    "supports": "refutes",
    "support": "refute",
    "refutes": "supports",
    "refute": "support",
}


def load_claims(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        print(f"Missing input file: {path}")
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"Failed to read {path}: {exc}")
        return []


def save_items(path: Path, items: List[Dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def main() -> None:
    claims = load_claims(LABELED_CLAIMS_FILE)
    if not claims:
        print("No labeled claims to process.")
        return

    set_negator_model(NEGATION_MODEL_LARGE)
    negator = RefuteGenerator(use_model=True)

    items: List[Dict[str, str]] = []
    seen = set()
    total = len(claims)
    processed = 0

    for claim in claims:
        text = (claim.get("claim") or "").strip()
        if not text:
            continue
        processed += 1
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)

        raw_label = (claim.get("label") or "").strip()
        if not raw_label:
            print(f"Warning: Missing label for claim id={claim.get('id')}")
        negated_label = LABEL_MAP.get(raw_label.lower()) if raw_label else None

        negated = negator.negate_text(text, allow_rules=True)
        if not negated:
            continue

        items.append({
            "id": claim.get("id"),
            "claim": text,
            "negated_claim": negated,
            "label_original": raw_label or None,
            "label_negated": negated_label,
        })

        if len(items) % SAVE_EVERY == 0:
            save_items(LABELED_CLAIMS_NEGATED_FILE, items)
            print(f"Saved {len(items)} items to {LABELED_CLAIMS_NEGATED_FILE}")

        if processed % LOG_EVERY == 0 or processed == total:
            print(f"Processed {processed}/{total} | negated: {len(items)}")

    save_items(LABELED_CLAIMS_NEGATED_FILE, items)
    print(f"\nDone. Saved {len(items)} items to {LABELED_CLAIMS_NEGATED_FILE}")


if __name__ == "__main__":
    main()
