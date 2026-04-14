#!/usr/bin/env python3
"""
WHO Content Scraping Scheduler - Central orchestrator for all WHO scraping tasks.

This script runs all WHO scrapers in sequence:
1. News headlines + details (last 3 days)
2. Disease Outbreak headlines + details (last 3 days)
3. Feature Stories headlines + details (last 3 days)

Usage:
    python who_scheduler.py --mode full           # Run all scrapers
    python who_scheduler.py --mode news           # Only news
    python who_scheduler.py --mode outbreak       # Only disease outbreak
    python who_scheduler.py --mode features       # Only feature stories
    python who_scheduler.py --days 3              # Custom days (default: 3)
    python who_scheduler.py --mode full --days 14 # Full scrape for last 14 days

For scheduling (every 3 days):
    - Windows Task Scheduler: Run this script weekly
    - Linux cron: 0 2 * * 0 /path/to/python who_scheduler.py --mode full
    - APScheduler in FastAPI: scheduler.add_job(run_scraper, 'interval', days=3)
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Base paths
SCRIPT_DIR = Path(__file__).parent
ADAPTERS_DIR = SCRIPT_DIR.parent.parent / "adapters" / "who" / "scraping"

# Scraper scripts
SCRAPERS = {
    "news": {
        "headlines": ADAPTERS_DIR / "adapter_who_news_headlines.py",
        "details": ADAPTERS_DIR / "adapter_who_news_details.py",
    },
    "outbreak": {
        "headlines": ADAPTERS_DIR / "adapter_who_disease_outbreak_headlines.py",
        "details": ADAPTERS_DIR / "adapter_who_disease_outbreak_details.py",
    },
    "features": {
        "headlines": ADAPTERS_DIR / "adapter_who_feature_stories_headlines.py",
        "details": ADAPTERS_DIR / "adapter_who_feature_stories_details.py",
    }
}


def log(message, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def run_script(script_path, args=None, description=""):
    """Run a Python script with given arguments."""
    if not script_path.exists():
        log(f"Script not found: {script_path}", "ERROR")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    log(f"Running: {description or script_path.name}")
    log(f"Command: {' '.join(cmd)}", "DEBUG")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=script_path.parent,
            capture_output=False,
            text=True,
            check=True
        )
        log(f"✅ Completed: {description or script_path.name}", "SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        log(f"❌ Failed: {description or script_path.name}", "ERROR")
        log(f"Error: {e}", "ERROR")
        return False
    except Exception as e:
        log(f"❌ Unexpected error: {e}", "ERROR")
        return False


def scrape_news(days=7):
    """Scrape WHO News (headlines + details)."""
    log("=" * 60)
    log("Starting WHO News scraping")
    log("=" * 60)
    
    # Step 1: Headlines (last N days)
    success = run_script(
        SCRAPERS["news"]["headlines"],
        args=["--max-age-days", str(days)],
        description=f"WHO News Headlines (last {days} days)"
    )
    
    if not success:
        log("Headlines scraping failed, skipping details", "WARNING")
        return False
    
    # Step 2: Details (only recent articles)
    success = run_script(
        SCRAPERS["news"]["details"],
        args=["--recent-days", str(days)],
        description=f"WHO News Details (last {days} days)"
    )
    
    return success


def scrape_disease_outbreak(days=3):
    """Scrape WHO Disease Outbreak News (headlines + details)."""
    log("=" * 60)
    log("Starting WHO Disease Outbreak scraping")
    log("=" * 60)
    
    # Step 1: Headlines
    success = run_script(
        SCRAPERS["outbreak"]["headlines"],
        args=["--max-age-days", str(days)],
        description=f"Disease Outbreak Headlines (last {days} days)"
    )
    
    if not success:
        log("Headlines scraping failed, skipping details", "WARNING")
        return False
    
    # Step 2: Details
    success = run_script(
        SCRAPERS["outbreak"]["details"],
        args=["--min-year", "2020"],  # Only process recent years to save time
        description="Disease Outbreak Details (recent articles only)"
    )
    
    return success


def scrape_feature_stories(days=7):
    """Scrape WHO Feature Stories (headlines + details)."""
    log("=" * 60)
    log("Starting WHO Feature Stories scraping")
    log("=" * 60)
    
    # Step 1: Headlines
    success = run_script(
        SCRAPERS["features"]["headlines"],
        args=["--max-age-days", str(days)],
        description=f"Feature Stories Headlines (last {days} days)"
    )
    
    if not success:
        log("Headlines scraping failed, skipping details", "WARNING")
        return False
    
    # Step 2: Details
    success = run_script(
        SCRAPERS["features"]["details"],
        args=["--recent-days", str(days)],
        description=f"Feature Stories Details (last {days} days)"
    )
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="WHO Content Scraping Scheduler - Orchestrates all WHO scraping tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python who_scheduler.py --mode full           # Run all scrapers (3 days)
  python who_scheduler.py --mode news           # Only news
  python who_scheduler.py --mode outbreak       # Only disease outbreak
  python who_scheduler.py --mode features       # Only feature stories
  python who_scheduler.py --mode full --days 14 # Full scrape (14 days)
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "news", "outbreak", "features"],
        default="full",
        help="Which scrapers to run (default: full)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days to look back (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Start
    start_time = datetime.now()
    log("=" * 60)
    log(f"WHO Scraping Scheduler Started")
    log(f"Mode: {args.mode}")
    log(f"Days: {args.days}")
    log("=" * 60)
    
    results = {}
    
    # Run selected scrapers
    if args.mode in ["full", "news"]:
        results["news"] = scrape_news(args.days)
    
    if args.mode in ["full", "outbreak"]:
        results["outbreak"] = scrape_disease_outbreak(args.days)
    
    if args.mode in ["full", "features"]:
        results["features"] = scrape_feature_stories(args.days)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    log("=" * 60)
    log("WHO Scraping Scheduler Summary")
    log("=" * 60)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        log(f"{name.upper()}: {status}")
    
    log(f"Duration: {duration}")
    log(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)
    
    # Exit with error code if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
