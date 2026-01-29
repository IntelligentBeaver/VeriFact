"""
WHO Scraping Scheduler Package

Central orchestration for all WHO content scraping tasks.
Supports periodic scheduling and manual runs.
"""

from pathlib import Path

__version__ = "1.0.0"
__author__ = "VeriFact"

# Package root
PACKAGE_ROOT = Path(__file__).parent

# Re-export main functions for convenience
try:
    from .fastapi_integration import setup_who_scheduler, run_who_scraper
    __all__ = ["setup_who_scheduler", "run_who_scraper"]
except ImportError:
    # APScheduler not installed
    __all__ = []
