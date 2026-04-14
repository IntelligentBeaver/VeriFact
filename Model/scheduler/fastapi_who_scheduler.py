#!/usr/bin/env python3
"""
FastAPI integration for WHO scraping scheduler.
Add this to your FastAPI application to enable automatic periodic WHO scraping.

Usage:
    from scheduler.fastapi_who_scheduler import setup_who_scheduler
    
    app = FastAPI()
    
    @app.on_event("startup")
    def startup():
        setup_who_scheduler(app)
"""

import subprocess
import sys
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import logging

from .config import get_who_config, get_logging_config, get_paths

logger = logging.getLogger(__name__)

# Get project root (one level up from this file's parent's parent)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS = get_paths()
SCHEDULER_SCRIPT = PROJECT_ROOT / PATHS["who_scraper"] / "who_scheduler.py"


def run_who_scraper(mode="full", days=None):
    """
    Run the WHO scraper script.
    
    Args:
        mode: Which scrapers to run (full, news, outbreak, features)
        days: Number of days to look back (default from config)
    """
    config = get_who_config()
    if days is None:
        days = config["default_days"]
    
    logger.info(f"Starting WHO scraper: mode={mode}, days={days}")
    
    try:
        cmd = [
            sys.executable,
            str(SCHEDULER_SCRIPT),
            "--mode", mode,
            "--days", str(days)
        ]
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("WHO scraper completed successfully")
            logger.debug(f"Output: {result.stdout}")
        else:
            logger.error(f"WHO scraper failed with code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("WHO scraper timed out after 1 hour")
    except Exception as e:
        logger.error(f"Failed to run WHO scraper: {e}")


def setup_who_scheduler(app=None, interval_days=None, schedule_time=None):
    """
    Setup APScheduler for periodic WHO scraping.
    
    Args:
        app: FastAPI app instance (optional, for lifecycle management)
        interval_days: Run every N days (default: from config)
        schedule_time: Specific time to run daily (format: "HH:MM", overrides interval_days)
    
    Returns:
        BackgroundScheduler instance
    """
    config = get_who_config()
    
    if interval_days is None:
        interval_days = config["schedule_interval_days"]
    
    if schedule_time is None:
        schedule_time = config["schedule_time"]
    
    scheduler = BackgroundScheduler()
    
    # Choose trigger type
    if schedule_time:
        # Run daily at specific time
        hour, minute = map(int, schedule_time.split(":"))
        trigger = CronTrigger(hour=hour, minute=minute)
        logger.info(f"WHO scraper scheduled daily at {schedule_time}")
    else:
        # Run every N days
        trigger = IntervalTrigger(days=interval_days)
        logger.info(f"WHO scraper scheduled every {interval_days} days")
    
    # Add job
    scheduler.add_job(
        run_who_scraper,
        trigger=trigger,
        id="who_scraper",
        replace_existing=True,
        kwargs={"mode": "full", "days": config["default_days"]}
    )
    
    # Start scheduler
    scheduler.start()
    logger.info("WHO scraping scheduler started")
    
    # Register shutdown hook if app provided
    if app:
        @app.on_event("shutdown")
        def shutdown_scheduler():
            scheduler.shutdown()
            logger.info("WHO scraping scheduler stopped")
    
    return scheduler


# Example FastAPI app integration
if __name__ == "__main__":
    print("""
Example FastAPI integration for WHO Scraping:

from fastapi import FastAPI
from scheduler.fastapi_who_scheduler import setup_who_scheduler

app = FastAPI()

@app.on_event("startup")
def startup():
    # Schedule WHO scraping every 3 days at 2 AM
    setup_who_scheduler(app)

@app.get("/")
def root():
    return {"message": "WHO scraper is running in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    """)
