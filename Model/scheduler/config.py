#!/usr/bin/env python3
"""
Central scheduler configuration for WHO scraping and FAISS indexing.
All scheduler components read from this central config.
"""

# ============================================================================
# WHO SCRAPING SCHEDULER CONFIG
# ============================================================================

WHO_SCRAPING = {
    "enabled": True,
    "default_days": 3,  # Look back 3 days by default
    "schedule_interval_days": 3,  # Run every 3 days
    "schedule_time": "02:00",  # Run at 2 AM (24-hour format)
    "max_threads": 4,  # Parallel threads for detail scraping
    "request_delay": 0.5,  # Seconds between requests
    "min_year": 2020,  # Don't process articles older than this
    "max_retries": 5,
    "backoff_base": 2.0,
    "backoff_min": 2.0,
    "backoff_max": 60.0,
}

# ============================================================================
# FAISS INDEXING SCHEDULER CONFIG
# ============================================================================

FAISS_INDEXING = {
    "enabled": True,
    "default_batch_size": 32,  # Embedding batch size
    "schedule_interval_days": 14,  # Run every 2 weeks
    "schedule_time": "03:00",  # Run at 3 AM (24-hour format)
    "normalize_embeddings": True,
    
    # Embedding models
    "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
    "sapbert_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    
    # Processing
    "skip_sapbert": False,  # Set to True to skip SapBERT computation (faster)
}

# ============================================================================
# GLOBAL LOGGING CONFIG
# ============================================================================

LOGGING = {
    "enabled": True,
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "log_file": None,  # Set to file path to log to file
}

# ============================================================================
# GLOBAL NOTIFICATION CONFIG (Future use)
# ============================================================================

NOTIFICATIONS = {
    "email_enabled": False,
    "email_on_completion": False,
    "email_on_error": True,
    "email_recipients": [],
    "slack_enabled": False,
    "slack_webhook": None,
}

# ============================================================================
# PATHS (Relative to project root)
# ============================================================================

PATHS = {
    "harvester_root": "harvester",
    "pipeline_root": "harvester/pipeline",
    "storage_root": "harvester/storage",
    "indexing_scheduler": "harvester/pipeline/indexing_scheduler",
    "who_scraper": "harvester/pipeline/who_scraping_scheduler",
}


def get_who_config():
    """Get WHO scraping configuration."""
    return WHO_SCRAPING.copy()


def get_indexing_config():
    """Get FAISS indexing configuration."""
    return FAISS_INDEXING.copy()


def get_logging_config():
    """Get logging configuration."""
    return LOGGING.copy()


def get_notifications_config():
    """Get notifications configuration."""
    return NOTIFICATIONS.copy()


def get_paths():
    """Get path configuration."""
    return PATHS.copy()


if __name__ == "__main__":
    print("WHO Scraping Config:")
    print(WHO_SCRAPING)
    print("\nFAISS Indexing Config:")
    print(FAISS_INDEXING)
    print("\nLogging Config:")
    print(LOGGING)
    print("\nPaths:")
    print(PATHS)
