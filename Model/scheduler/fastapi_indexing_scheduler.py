#!/usr/bin/env python3
"""
FastAPI integration for FAISS incremental indexing scheduler.
Add this to your FastAPI application to enable automatic periodic FAISS index updates.

Usage:
    from scheduler.fastapi_indexing_scheduler import setup_indexing_scheduler
    
    app = FastAPI()
    
    @app.on_event("startup")
    def startup():
        setup_indexing_scheduler(app)
"""

import subprocess
import sys
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import logging

from .config import get_indexing_config, get_logging_config, get_paths

logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS = get_paths()
INDEXING_SCRIPT = PROJECT_ROOT / PATHS["indexing_scheduler"] / "update_faiss_incremental.py"


def run_indexing_updater(rebuild=False):
    """
    Run the incremental FAISS indexing script.
    
    Args:
        rebuild: Force full rebuild instead of incremental (default: False)
    """
    logger.info(f"Starting FAISS indexing: rebuild={rebuild}")
    
    try:
        cmd = [sys.executable, str(INDEXING_SCRIPT)]
        
        if rebuild:
            cmd.append("--rebuild")
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("FAISS indexing completed successfully")
            logger.debug(f"Output: {result.stdout}")
        else:
            logger.error(f"FAISS indexing failed with code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("FAISS indexing timed out after 2 hours")
    except Exception as e:
        logger.error(f"Failed to run FAISS indexing: {e}")


def setup_indexing_scheduler(app=None, interval_days=None, schedule_time=None, rebuild_on_startup=False):
    """
    Setup APScheduler for periodic FAISS index updates.
    
    Args:
        app: FastAPI app instance (optional, for lifecycle management)
        interval_days: Run every N days (default: from config)
        schedule_time: Specific time to run daily (format: "HH:MM", overrides interval_days)
        rebuild_on_startup: Run full rebuild immediately on startup (default: False)
    
    Returns:
        BackgroundScheduler instance
    """
    config = get_indexing_config()
    
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
        logger.info(f"FAISS indexing scheduled daily at {schedule_time}")
    else:
        # Run every N days
        trigger = IntervalTrigger(days=interval_days)
        logger.info(f"FAISS indexing scheduled every {interval_days} days")
    
    # Add job
    scheduler.add_job(
        run_indexing_updater,
        trigger=trigger,
        id="faiss_indexing",
        replace_existing=True,
        kwargs={"rebuild": False}
    )
    
    # Start scheduler
    scheduler.start()
    logger.info("FAISS indexing scheduler started")
    
    # Run immediately on startup if requested
    if rebuild_on_startup:
        logger.info("Running full FAISS rebuild on startup...")
        run_indexing_updater(rebuild=True)
    
    # Register shutdown hook if app provided
    if app:
        @app.on_event("shutdown")
        def shutdown_scheduler():
            scheduler.shutdown()
            logger.info("FAISS indexing scheduler stopped")
    
    return scheduler


# Example FastAPI app integration
if __name__ == "__main__":
    print("""
Example FastAPI integration for FAISS Indexing:

from fastapi import FastAPI
from scheduler.fastapi_indexing_scheduler import setup_indexing_scheduler

app = FastAPI()

@app.on_event("startup")
def startup():
    # Schedule FAISS indexing every 2 weeks at 3 AM
    setup_indexing_scheduler(app)
    
    # OR run full rebuild immediately then schedule updates:
    # setup_indexing_scheduler(app, rebuild_on_startup=True)

@app.get("/")
def root():
    return {"message": "FAISS indexing is running in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    """)
