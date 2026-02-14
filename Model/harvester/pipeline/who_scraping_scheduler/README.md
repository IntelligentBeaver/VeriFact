# WHO Scraping Scheduler

Central orchestration system for all WHO content scraping tasks. Automatically fetches new content from WHO sources every 7 days (configurable).

## Features

- **Automated Scheduling**: Run scrapers periodically (every 7 days by default)
- **Smart Caching**: Skips existing articles to save time
- **Three Content Types**:
  - WHO News (press releases, statements)
  - Disease Outbreak News
  - Feature Stories
- **Flexible Modes**: Run all scrapers or select specific ones
- **FastAPI Integration**: Easy integration with your web server

## Quick Start

### 1. Manual Run

```bash
# Run all scrapers (last 7 days)
python who_scheduler.py --mode full

# Run specific scraper
python who_scheduler.py --mode news
python who_scheduler.py --mode outbreak
python who_scheduler.py --mode features

# Custom time range
python who_scheduler.py --mode full --days 14
```

### 2. Windows Task Scheduler

Create a scheduled task that runs every 7 days:

```powershell
schtasks /Create /SC WEEKLY /D SUN /ST 02:00 /TN "WHO_Scraper" /TR "python C:\path\to\who_scheduler.py --mode full"
```

Or use the GUI:
1. Open Task Scheduler
2. Create Task → Triggers → New → Weekly
3. Actions → Start a program → Browse to `python.exe`
4. Add arguments: `C:\path\to\who_scheduler.py --mode full`

### 3. Linux Cron

Add to crontab (`crontab -e`):

```cron
# Run every Sunday at 2 AM
0 2 * * 0 /usr/bin/python3 /path/to/who_scheduler.py --mode full
```

### 4. FastAPI Integration

```python
from fastapi import FastAPI
from harvester.pipeline.who_scraping_scheduler.fastapi_integration import setup_who_scheduler

app = FastAPI()

@app.on_event("startup")
def startup():
    # Schedule WHO scraping every 7 days
    setup_who_scheduler(app)
    
    # OR schedule daily at 2 AM:
    # setup_who_scheduler(app, schedule_time="02:00")

@app.get("/")
def root():
    return {"message": "WHO scraper is running in background"}
```

Install required package:
```bash
pip install apscheduler
```

## Configuration

Edit `scheduler_config.py` to customize:

```python
# Scraping intervals
DEFAULT_SCRAPE_DAYS = 7  # Look back 7 days

# Scheduling
SCHEDULE_INTERVAL_DAYS = 7  # Run every 7 days
SCHEDULE_TIME = "02:00"  # Run at 2 AM

# Performance
MAX_THREADS = 4
REQUEST_DELAY = 0.5

# Filtering
MIN_YEAR = 2020  # Don't process very old articles
```

## How It Works

### Two-Stage Process

Each content type uses a two-stage scraping process:

1. **Headlines Stage**: Scrapes article lists (fast)
   - Fetches only articles from last N days
   - Saves to JSON files grouped by year
   - Skips existing URLs

2. **Details Stage**: Fetches full article content (slower)
   - Processes only new articles from headlines
   - Extracts full text, images, references
   - Saves detailed JSON for each article

### Smart Caching

- Checks existing files before scraping
- Skips already-downloaded articles
- Only fetches new content
- Significantly reduces run time on subsequent runs

### Example Flow

```
WHO News Scraping:
├── Stage 1: Headlines (adapter_who_news_headlines.py)
│   ├── Fetch articles from last 7 days
│   ├── Save to storage/who/headlines/2026.json
│   └── Skip URLs already in file
│
└── Stage 2: Details (adapter_who_news_details.py)
    ├── Read headlines from 2026.json
    ├── Check existing files in storage/who/news/
    ├── Scrape only new articles
    └── Save full content

Same process for Disease Outbreak and Feature Stories
```

## Output Structure

```
storage/who/
├── headlines/              # News headlines by year
│   ├── 2026.json
│   └── 2025.json
├── news/                   # Full news articles
│   ├── article1.json
│   └── article2.json
├── disease_outbreak_headlines/
│   └── 2026.json
├── disease_outbreak_news/
│   └── 2026.json
├── feature_stories_headlines/
│   └── 2026.json
└── feature_stories_news/
    └── 2026.json
```

## Monitoring

Check logs for status:

```bash
# View scheduler logs
tail -f scheduler.log

# Check last run status
python who_scheduler.py --mode full
```

Output includes:
- Timestamp for each operation
- Success/failure status
- Number of new articles found
- Number of articles skipped
- Total duration

## Troubleshooting

### No new articles found

This is normal if you run frequently. The scraper only fetches articles from the last 7 days.

### Rate limiting (429 errors)

The scripts include retry logic with exponential backoff. If you still see errors:
- Increase `REQUEST_DELAY` in config
- Reduce `MAX_THREADS`

### Script fails to start

Check paths in `who_scheduler.py`:
```python
ADAPTERS_DIR = SCRIPT_DIR.parent.parent / "adapters" / "who" / "scraping"
```

### APScheduler import error

Install dependencies:
```bash
pip install apscheduler
```

## Manual Scraper Usage

You can still run individual scrapers:

```bash
# News headlines (last 7 days)
cd adapters/who/scraping
python adapter_who_news_headlines.py --max-age-days 7

# News details (recent only)
python adapter_who_news_details.py --recent-days 7

# Disease outbreak
python adapter_who_disease_outbreak_headlines.py --max-age-days 7
python adapter_who_disease_outbreak_details.py --min-year 2020

# Feature stories
python adapter_who_feature_stories_headlines.py --max-age-days 7
python adapter_who_feature_stories_details.py --recent-days 7
```

## Best Practices

1. **Run regularly**: Schedule for every 7 days to avoid missing content
2. **Off-peak hours**: Schedule for 2-4 AM to minimize server load
3. **Monitor logs**: Check for errors after each run
4. **Storage cleanup**: Periodically archive old JSON files
5. **Rate limiting**: Respect WHO servers, don't reduce delays too much

## Integration with Your Pipeline

This scheduler is designed to work with your existing VeriFact pipeline:

```python
# In your main pipeline
from harvester.pipeline.who_scraping_scheduler import run_who_scraper

# Manually trigger scraping
run_who_scraper(mode="full", days=7)

# Or integrate with your scheduler
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(
    run_who_scraper,
    'interval',
    days=7,
    kwargs={'mode': 'full', 'days': 7}
)
scheduler.start()
```

## Support

For issues or questions:
1. Check logs for error messages
2. Verify all paths in configuration
3. Test individual scrapers manually
4. Check WHO website structure hasn't changed
