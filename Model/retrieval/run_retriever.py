"""
Setup and run the retrieval system.

Step 2: Index passages to ElasticSearch (run once)
Step 3: Search with queries (run anytime)
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT_DIR / "storage"

# ============================================================================
# STEP 2: INDEX PASSAGES TO ELASTICSEARCH (Run this once)
# ============================================================================

def setup_elasticsearch():
    """Create ElasticSearch index and load all passages."""
    from simple_retriever import (
        create_elasticsearch_index, 
        index_passages_to_elasticsearch
    )
    
    print("\n" + "="*80)
    print("STEP 2: INDEX PASSAGES TO ELASTICSEARCH")
    print("="*80)
    
    # Create index
    print("\nCreating ElasticSearch index...")
    create_elasticsearch_index()
    
    # Load passages from your data
    print("\nLoading passages from storage...")
    passages_file = STORAGE_DIR / "metadata.json"
    
    if not passages_file.exists():
        print(f"✗ File not found: {passages_file}")
        return False
    
    try:
        with open(passages_file, 'r', encoding='utf-8') as f:
            passages = json.load(f)
    except UnicodeDecodeError:
        print(f"✗ File encoding error. Try with different encoding...")
        with open(passages_file, 'r', encoding='utf-8-sig') as f:
            passages = json.load(f)
    
    print(f"✓ Loaded {len(passages)} passages")
    
    # Index them
    print("\nIndexing passages to ElasticSearch...")
    index_passages_to_elasticsearch(passages)
    
    print("\n✓ Setup complete! ElasticSearch is ready.")
    return True


# ============================================================================
# STEP 3: SEARCH WITH QUERIES (Run this anytime)
# ============================================================================

def search_and_display(query: str, retriever):
    """Search and display results using pre-initialized retriever."""
    print("\n" + "="*80)
    print("STEP 3: SEARCH")
    print("="*80)
    
    # Search
    results = retriever.search(query)
    
    # Display results nicely
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    if not results:
        print("\nNo results found.")
        return
    
    for i, result in enumerate(results, 1):
        p = result['passage']
        scores = result.get('scores', {})
        
        print(f"\n{i}. SCORE: {result['final_score']:.3f}")
        print(f"   Domain Tier: {result.get('domain_tier', 'N/A')}")
        print(f"   Title: {p.get('title', 'N/A')}")
        print(f"   URL: {p.get('url', 'N/A')}")
        print(f"   Author: {p.get('author', 'N/A')}")
        print(f"   Reviewed: {p.get('medically_reviewed_by', 'Not reviewed')}")
        print(f"   Published: {p.get('published_date', 'Unknown')}")
        print(f"   ---")
        print(f"   Score Breakdown:")
        print(f"     - FAISS (semantic):    {scores.get('faiss', 0):.3f}")
        print(f"     - Cross-Encoder:       {scores.get('cross_encoder', 0):.3f}")
        print(f"     - MeSH Entities:       {scores.get('entity_match', 0):.3f}")
        print(f"     - Lexical (exact):     {scores.get('lexical', 0):.3f}")
        print(f"     - Domain authority:    {scores.get('domain', 0):.3f}")
        print(f"     - Freshness:           {scores.get('freshness', 0):.3f}")
        
        # Show text preview (up to 250 words)
        text = p.get('text', '')
        if text:
            words = text.split()
            preview = ' '.join(words[:250])
            if len(words) > 250:
                preview += "..."
            print(f"   ---")
            print(f"   Preview:")
            print(f"   {preview}")


# ============================================================================
# MAIN - Choose what to run
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("RETRIEVAL SYSTEM - SETUP AND SEARCH")
    print("="*80)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            # Setup: Index passages to ElasticSearch
            setup_elasticsearch()
        
        elif command == "search":
            # Search: Run a query
            if len(sys.argv) < 3:
                print("\nUsage: python run_retriever.py search '<query>'")
                print("Example: python run_retriever.py search 'vaccine safety during pregnancy'")
                sys.exit(1)
            
            from simple_retriever import SimpleRetriever, MinimalModelManager

            query = " ".join(sys.argv[2:])
            model_manager = MinimalModelManager(str(STORAGE_DIR))
            retriever = SimpleRetriever(model_manager, str(STORAGE_DIR))
            search_and_display(query, retriever)
        
        else:
            print(f"\nUnknown command: {command}")
            print("\nUsage:")
            print("  Setup (one-time):    python run_retriever.py setup")
            print("  Search:              python run_retriever.py search '<query>'")
            sys.exit(1)
    
    else:
        # No arguments - show interactive menu that keeps running
        print("\n" + "="*80)
        print("RETRIEVAL SYSTEM - INTERACTIVE MODE")
        print("="*80)
        
        # Initialize models once at startup
        from simple_retriever import SimpleRetriever, MinimalModelManager
        
        print("\nInitializing models (one-time)...")
        model_manager = MinimalModelManager(str(STORAGE_DIR))
        retriever = SimpleRetriever(model_manager, str(STORAGE_DIR))
        print("Models loaded! Ready to search.\n")
        
        while True:
            print("\nChoose an option:")
            print("  1. Setup (index passages to ElasticSearch)")
            print("  2. Search")
            print("  3. Exit")
            
            choice = input("\nEnter 1, 2, or 3: ").strip().lower()
            
            if choice == '1':
                setup_elasticsearch()
            
            elif choice == '2':
                query = input("\nEnter search query: ").strip()
                if query:
                    search_and_display(query, retriever)
                else:
                    print("Query cannot be empty.")
            
            elif choice == '3':
                print("\nGoodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")