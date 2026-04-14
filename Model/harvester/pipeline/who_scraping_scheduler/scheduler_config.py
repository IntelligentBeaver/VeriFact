#!/usr/bin/env python3
"""
Configuration for WHO scraping scheduler.
Modify these settings to control scraping behavior.
"""

# Scraping intervals (in days)
DEFAULT_SCRAPE_DAYS = 3  # Look back 3 days by default

# Scheduling configuration (for APScheduler integration)
SCHEDULE_INTERVAL_DAYS = 3  # Run every 3 days
SCHEDULE_TIME = "02:00"  # Run at 2 AM

# Performance settings
MAX_THREADS = 4  # Parallel threads for detail scraping
REQUEST_DELAY = 0.5  # Seconds between requests

# Filtering
MIN_YEAR = 2020  # Don't process articles older than this

# Output settings
ENABLE_LOGGING = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Retry settings
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
BACKOFF_MIN = 2.0
BACKOFF_MAX = 60.0

# Notification settings (for future use)
SEND_EMAIL_ON_COMPLETION = False
EMAIL_RECIPIENTS = []

# Storage paths (relative to script)
STORAGE_BASE = "../../storage/who"
