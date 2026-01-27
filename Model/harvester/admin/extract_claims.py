import argparse
import csv
import json
import os
from typing import List, Dict


def read_claims_from_tsv(file_path: str, labels: List[str]) -> List[str]:
    claims: List[str] = []
    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label = (row.get("label") or "").strip().lower()
            if label in labels:
                claim = (row.get("claim") or "").strip()
                if claim:
                    claims.append(claim)
    return claims


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract claims with label 'mixture' or 'unproven' from TSV files."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["dev.tsv", "test.tsv", "train.tsv"],
        help="Input TSV files (default: dev.tsv test.tsv train.tsv)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("output", "claims_mixture_unproven.json"),
        help="Output JSON file path",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    labels = {"mixture", "unproven"}

    all_claims: List[str] = []
    for input_name in args.inputs:
        input_path = input_name
        if not os.path.isabs(input_path):
            input_path = os.path.join(base_dir, input_name)
        if not os.path.exists(input_path):
            continue
        all_claims.extend(read_claims_from_tsv(input_path, labels))

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_items: List[Dict[str, object]] = [
        {"id": idx, "claim": claim} for idx, claim in enumerate(all_claims, start=1)
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_items, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
