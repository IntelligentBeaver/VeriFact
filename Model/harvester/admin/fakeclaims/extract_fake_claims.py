"""
Extract fake claims from CSV and JSON files and store in fake_claims.json
"""

import csv
import json
from pathlib import Path

def extract_fake_claims_from_csvs(csv_folder, output_file):
    """
    Extract titles from all CSV files in the specified folder and save to JSON.
    
    Args:
        csv_folder: Path to folder containing CSV files with fake claims
        output_file: Path to output JSON file
    """
    csv_folder = Path(csv_folder)
    output_file = Path(output_file)
    
    # Load existing claims if file exists
    fake_claims = []
    claim_id_counter = 1
    
    if output_file.exists():
        try:
            with output_file.open('r', encoding='utf-8') as f:
                fake_claims = json.load(f)
            # Find the next ID to use
            if fake_claims:
                last_id = max(int(claim['id'].split('_')[1]) for claim in fake_claims if 'id' in claim and claim['id'].startswith('fake_'))
                claim_id_counter = last_id + 1
            print(f"Loaded {len(fake_claims)} existing claims, starting from ID {claim_id_counter}")
        except Exception as e:
            print(f"Could not load existing claims: {e}")
            fake_claims = []
            claim_id_counter = 1
    
    # Get all CSV files in the folder
    csv_files = sorted(csv_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        
        try:
            with csv_file.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Extract title, handling both 'title' and potential variations
                    title = row.get('title', '').strip()
                    
                    # Skip empty titles
                    if not title:
                        continue
                    
                    # Remove extra quotes if present
                    title = title.strip('"').strip()
                    
                    # Skip if still empty after cleaning
                    if not title:
                        continue
                    
                    fake_claims.append({
                        "id": f"fake_{claim_id_counter:03d}",
                        "claim": title
                    })
                    claim_id_counter += 1
            
            print(f"  ✓ Extracted {claim_id_counter - 1} claims so far")
                    
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}")
            continue
    
    # Save to JSON
    if fake_claims:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(fake_claims, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully saved {len(fake_claims)} total fake claims")
        print(f"✓ Saved to: {output_file}")
        print(f"{'='*60}")
    else:
        print("\n✗ No claims extracted")


def extract_fake_claims_from_jsons(json_folder, output_file):
    """
    Extract titles from all JSON files in the specified folder and append to output JSON.
    
    Args:
        json_folder: Path to folder containing JSON files with fake claims
        output_file: Path to output JSON file
    """
    json_folder = Path(json_folder)
    output_file = Path(output_file)
    
    # Load existing claims
    fake_claims = []
    claim_id_counter = 1
    
    if output_file.exists():
        try:
            with output_file.open('r', encoding='utf-8') as f:
                fake_claims = json.load(f)
            # Find the next ID to use
            if fake_claims:
                last_id = max(int(claim['id'].split('_')[1]) for claim in fake_claims if 'id' in claim and claim['id'].startswith('fake_'))
                claim_id_counter = last_id + 1
            print(f"Loaded {len(fake_claims)} existing claims, starting from ID {claim_id_counter}")
        except Exception as e:
            print(f"Could not load existing claims: {e}")
    
    # Get all JSON files in the folder
    json_files = sorted(json_folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {json_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        
        try:
            with json_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract title
            title = data.get('title', '').strip()
            
            # Skip empty titles
            if not title:
                print(f"  ⚠ No title found in {json_file.name}")
                continue
            
            fake_claims.append({
                "id": f"fake_{claim_id_counter:03d}",
                "claim": title
            })
            claim_id_counter += 1
            print(f"  ✓ Extracted: {title[:60]}...")
                    
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
            continue
    
    # Save to JSON
    if fake_claims:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(fake_claims, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully saved {len(fake_claims)} total fake claims")
        print(f"✓ Saved to: {output_file}")
        print(f"{'='*60}")
    else:
        print("\n✗ No claims found")


if __name__ == "__main__":
    # Paths relative to this script location
    SCRIPT_DIR = Path(__file__)
    OUTPUT_FILE = SCRIPT_DIR.parent / "input" / "fake_claims.json"
    
    # Extract from CSV files (dataset1)
    CSV_FOLDER = SCRIPT_DIR / "fakeclaims" / "dataset1"
    extract_fake_claims_from_csvs(CSV_FOLDER, OUTPUT_FILE)
    
    # Extract from JSON files (dataset2) - appends to existing
    JSON_FOLDER = SCRIPT_DIR / "fakeclaims" / "dataset2"
    extract_fake_claims_from_jsons(JSON_FOLDER, OUTPUT_FILE)
