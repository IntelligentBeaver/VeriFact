import csv
import json
import os
from typing import List, Dict

# Configuration (edit these variables instead of using CLI arguments)
# Input files may be TSV or CSV and are resolved relative to this script.
INPUT_FILES = [
    "dev.tsv",
    "test.tsv",
    "train.tsv",
    "healthver_dev.csv",
    "healthver_test.csv",
    "healthver_train.csv",
]
# Output file name (written into the labelling `input` directory by default)
OUTPUT_FILE = "claims_unlabeled.json"
# Output file for verified/labeled claims
OUTPUT_FILE_LABELED = "claims_labeled.json"
# Labels to exclude (we only keep rows whose label is NOT one of these)
EXCLUDE_LABELS = {"supports", "refutes"}
# Labels to include for verified/labeled extraction
INCLUDE_LABELS = {"supports", "support", "refutes", "refute", "true", "false"}

def clean_claim_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith("claim:"):
        cleaned = cleaned[len("claim:"):].strip()
    return cleaned


def read_claims_from_file(file_path: str, exclude_labels: set) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    # Choose delimiter based on extension
    delimiter = "\t" if file_path.lower().endswith(".tsv") else ","
    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            raw_label = (row.get("label") or "").strip()
            label = raw_label.lower()
            # keep rows whose label is NOT one of the excluded (Supports/Refutes)
            if label in exclude_labels:
                continue
            claim = clean_claim_text(row.get("claim") or "")
            if claim and "?" not in claim:
                items.append({"claim": claim, "label": raw_label})
    return items


def read_labeled_claims_from_file(file_path: str, include_labels: set) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    delimiter = "\t" if file_path.lower().endswith(".tsv") else ","
    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            raw_label = (row.get("label") or "").strip()
            label = raw_label.lower()
            if label not in include_labels:
                continue
            claim = clean_claim_text(row.get("claim") or "")
            if claim and "?" not in claim:
                items.append({"claim": claim, "label": raw_label})
    return items


def main() -> None:
    # Use top-level configuration variables defined above
    base_dir = os.path.dirname(os.path.abspath(__file__))
    labelling_dir = os.path.dirname(base_dir)
    default_output_dir = os.path.join(labelling_dir, "input")

    all_items: List[Dict[str, str]] = []
    labeled_items: List[Dict[str, str]] = []
    for input_name in INPUT_FILES:
        input_path = input_name
        if not os.path.isabs(input_path):
            input_path = os.path.join(base_dir, input_name)
        if not os.path.exists(input_path):
            continue
        all_items.extend(read_claims_from_file(input_path, EXCLUDE_LABELS))
        labeled_items.extend(read_labeled_claims_from_file(input_path, INCLUDE_LABELS))

    # Generate deterministic IDs for combined claims: UL000001, UL000002, ...
    output_items: List[Dict[str, object]] = []
    for idx, item in enumerate(all_items, start=1):
        uid = f"UL{idx:06d}"
        output_items.append({"id": uid, "claim": item["claim"], "label": item["label"]})

    # Generate deterministic IDs for verified claims: VL000001, VL000002, ...
    labeled_output_items: List[Dict[str, object]] = []
    for idx, item in enumerate(labeled_items, start=1):
        uid = f"VL{idx:06d}"
        labeled_output_items.append({"id": uid, "claim": item["claim"], "label": item["label"]})

    output_name = OUTPUT_FILE
    if not os.path.isabs(output_name):
        os.makedirs(default_output_dir, exist_ok=True)
        output_path = os.path.join(default_output_dir, output_name)
    else:
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        output_path = output_name

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_items, handle, ensure_ascii=False, indent=2)

    labeled_output_name = OUTPUT_FILE_LABELED
    if not os.path.isabs(labeled_output_name):
        os.makedirs(default_output_dir, exist_ok=True)
        labeled_output_path = os.path.join(default_output_dir, labeled_output_name)
    else:
        os.makedirs(os.path.dirname(labeled_output_name), exist_ok=True)
        labeled_output_path = labeled_output_name

    with open(labeled_output_path, "w", encoding="utf-8") as handle:
        json.dump(labeled_output_items, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
