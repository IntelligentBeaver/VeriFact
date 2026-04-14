"""Converted from adapter_pubmed_esearch.ipynb"""

# %% Imports & constants
import requests
import time
import json
import threading
import logging
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "health-harvester"
REQUEST_TIMEOUT = 30
RETRY_TOTAL = 3
RETRY_BACKOFF = 0.5
RETRY_STATUS = (429, 500, 502, 503, 504)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

# %% Storage directory configurations
STORAGE_DIR = Path.cwd().parent / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

SEED_DIR = STORAGE_DIR / "seeds"

# If set, load seeds from this file or directory. If None, use SEED_DIR.
SEED_PATH: Optional[str] = str(SEED_DIR / "mesh_seed_Z.json")

# Run configuration
WORKERS = 5
TOP_K_TERMS = 3
RETMAX = 25
PERSIST = True
VERBOSE = True

_thread_local = threading.local()

def _create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=RETRY_STATUS,
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _create_session()
        _thread_local.session = session
    return session

# %% Stats class
class SearchStats:
    def __init__(self, total_seeds: int):
        self.start = time.time()
        self.total_seeds = total_seeds
        self.seeds_done = 0
        self.terms_searched = 0
        self.total_pmids = 0
        self.failed_terms = 0
        self.lock = threading.Lock()

    def log_term(self, pmid_count: int, success: bool = True):
        """Call for each term searched."""
        with self.lock:
            self.terms_searched += 1
            if success:
                self.total_pmids += int(pmid_count or 0)
            else:
                self.failed_terms += 1

    def log_seed(self):
        """Call once per finished seed."""
        with self.lock:
            self.seeds_done += 1
            # print brief periodic summary
            if (self.seeds_done % 10 == 0) or (self.seeds_done == self.total_seeds):
                self.print_summary()

    def print_summary(self):
        elapsed = time.time() - self.start
        rate = self.terms_searched / max(elapsed, 1.0)
        avg_pmids = (self.total_pmids / self.terms_searched) if self.terms_searched else 0
        logger.info(
            "[SEARCH STATS] seeds=%s/%s | terms=%s | pmids_total=%s | "
            "avg_pmids/term=%.1f | failed_terms=%s | %.2f terms/sec | %.1fs elapsed",
            self.seeds_done,
            self.total_seeds,
            self.terms_searched,
            self.total_pmids,
            avg_pmids,
            self.failed_terms,
            rate,
            elapsed,
        )

# %% API key + email configuration
PUBMED_KEYS = [
    ("ec74621abe110994f710510d05aa0780d607", "abamsheikh@gmail.com"),
    ("fc54b7ce2e97e3cd4506c9a780d8c5e3c208", "prashant.211528@ncit.edu.np"),
    ("7441cfd3f83858837e75dd0d746419db1408", "prashantchhetrii465@gmail.com"),
    ("8e301d66b6848df9d80acf3082c910a1f308", "aman.21506@ncit.edu.np"),
    ("8e301d66b6848df9d80acf3082c910a1f308", "abamsheikh1@gmail.com"),
]

# %% PubMed search function
def pubmed_search(
    query: str,
    retstart: int = 0,
    retmax: int = 20,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: int = REQUEST_TIMEOUT,
):
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retstart": retstart,
        "retmax": retmax,
        "tool": TOOL,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    sess = session or _get_session()
    r = sess.get(f"{ENTREZ_BASE}/esearch.fcgi", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# %% API key rate-limiter
class ApiKeyRateLimiter:
    def __init__(self, api_key, email, delay=0.11):
        self.api_key = api_key
        self.email = email
        self.delay = delay
        self.lock = threading.Lock()
        self.last_call = 0.0

    def call(self, fn, *args, **kwargs):
        with self.lock:
            now = time.time()
            wait = max(0, self.delay - (now - self.last_call))
            if wait:
                time.sleep(wait)
            result = fn(*args, api_key=self.api_key, email=self.email, **kwargs)
            self.last_call = time.time()
            return result

def normalize_key_pairs(key_email_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    valid = []
    for key, email in key_email_pairs:
        if isinstance(key, str) and key and isinstance(email, str) and email:
            valid.append((key, email))
    if not valid:
        logger.warning("No valid API keys/emails provided.")
    return valid


def build_api_key_pool(key_email_pairs: List[Tuple[str, str]], delay_per_key: float = 0.11) -> Queue:
    q = Queue()
    for key, email in normalize_key_pairs(key_email_pairs):
        q.put(ApiKeyRateLimiter(key, email, delay_per_key))
    return q

# %% Load seeds
def load_seed_files_from_dir(seeds_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    seeds_dir = Path(seeds_dir)
    if not seeds_dir.exists():
        logger.warning("Seeds directory does not exist: %s", seeds_dir)
        return []
    seeds = []
    for p in seeds_dir.glob("*.json"):
        seeds.extend(load_seed_file(p))
    if not seeds:
        logger.warning("No seed JSON files found under %s", seeds_dir)
    return seeds


def load_seed_file(seed_path: Union[str, Path]) -> List[Dict[str, Any]]:
    seed_path = Path(seed_path)
    if not seed_path.exists():
        logger.warning("Seed file does not exist: %s", seed_path)
        return []
    try:
        with open(seed_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        seeds = []
        # attach a helper path so we can save the file back later if desired
        if isinstance(data, dict):
            data.setdefault("_seed_file_path", str(seed_path))
            seeds.append(data)
        elif isinstance(data, list):
            for s in data:
                if isinstance(s, dict):
                    s.setdefault("_seed_file_path", str(seed_path))
                    seeds.append(s)
        else:
            logger.warning("Seed file %s does not contain dict or list", seed_path)
        return seeds
    except Exception as e:
        logger.warning("Failed to load seed file %s: %s", seed_path, e)
        return []


def load_seeds_from_path(path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    if path is None:
        return load_seed_files_from_dir(SEED_DIR)
    path = Path(path)
    if path.is_dir():
        return load_seed_files_from_dir(path)
    return load_seed_file(path)

# %% MeSH-aware query builder
def build_mesh_aware_query(term: str) -> str:
    safe_term = term.replace('"', "\\\"")
    return f'(\"{safe_term}\"[MeSH Terms] OR \"{safe_term}\"[Title/Abstract])'

# %% Single-term PubMed search worker
def search_term_with_pool(term, retmax, key_pool, stats: SearchStats = None):
    """
    Borrow a limiter from key_pool, run the query, return the log dict.
    Updates stats via stats.log_term(...)
    """
    if key_pool.empty():
        entry = {
            "term": term,
            "query": None,
            "count": 0,
            "pmids": [],
            "error": "no API keys available",
            "api_key_used": None,
            "email_used": None,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if stats:
            stats.log_term(0, success=False)
        return entry

    limiter = key_pool.get()
    try:
        if not term:
            entry = {
                "term": term,
                "query": None,
                "count": 0,
                "pmids": [],
                "error": "empty term",
                "api_key_used": limiter.api_key[-6:],
                "email_used": limiter.email,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if stats:
                stats.log_term(0, success=False)
            return entry
        query = build_mesh_aware_query(term)
        try:
            res = limiter.call(pubmed_search, query, 0, retmax, session=_get_session())
            es = res.get("esearchresult", {}) if isinstance(res, dict) else {}
            count = int(es.get("count", "0") or 0)
            pmids = es.get("idlist", []) or []
            entry = {
                "term": term,
                "query": query,
                "count": count,
                "pmids": pmids,
                "api_key_used": limiter.api_key[-6:],
                "email_used": limiter.email,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if stats:
                stats.log_term(count, success=True)
            return entry
        except Exception as e:
            # record failure entry
            entry = {
                "term": term,
                "query": query,
                "count": 0,
                "pmids": [],
                "error": str(e),
                "api_key_used": limiter.api_key[-6:],
                "email_used": limiter.email,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if stats:
                stats.log_term(0, success=False)
            return entry
    finally:
        key_pool.put(limiter)

# %% Seed-level PubMed search
def search_seed_pubmed(seed, key_pool, top_k=3, retmax=200, stats: SearchStats = None, verbose: bool = True):
    """
    Search preferred terms for a seed using the key_pool.
    Appends per-term log entries to seed['pubmed_search_log'].
    Calls stats.log_seed() once finished.
    """
    existing_log = seed.get("pubmed_search_log")
    if isinstance(existing_log, list) and existing_log:
        if verbose:
            logger.info("[%s] skipping: pubmed_search_log already populated", seed.get("seed_id"))
        if stats:
            stats.log_seed()
        return seed

    seed.setdefault("pubmed_search_log", [])
    terms = seed.get("preferred_search_terms", [])
    if not isinstance(terms, list):
        terms = []
    terms = terms[:top_k]
    if not terms:
        # fallback to candidates
        candidates = seed.get("keyword_candidates", [])
        if isinstance(candidates, list):
            terms = [c.get("term") for c in candidates if isinstance(c, dict)][:top_k]
        else:
            terms = []

    for term in terms:
        entry = search_term_with_pool(term, retmax, key_pool, stats=stats)
        seed["pubmed_search_log"].append(entry)
        if verbose:
            if entry.get("error"):
                logger.warning(
                    "[%s] term='%s' FAILED -> %s",
                    seed.get("seed_id"),
                    term,
                    entry.get("error"),
                )
            else:
                logger.info(
                    "[%s] term='%s' -> %s pmids (key=*%s)",
                    seed.get("seed_id"),
                    term,
                    entry.get("count"),
                    entry.get("api_key_used"),
                )

    if stats:
        stats.log_seed()

    return seed

# %% Parallel execution across all seeds
def run_pubmed_search_parallel(seeds, key_pool, workers=8, top_k_terms=3, retmax=10, persist=False, stats: SearchStats = None, verbose=True):
    """
    Parallel search across seeds. Prints per-seed progress and periodic summaries via stats.
    """
    results = []
    total = len(seeds)
    if total == 0:
        logger.warning("No seeds to process.")
        return results
    if stats is None:
        stats = SearchStats(total_seeds=total)
    workers = max(1, min(workers, total))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(search_seed_pubmed, s, key_pool, top_k_terms, retmax, stats, verbose): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                updated_seed = fut.result()
            except Exception as e:
                logger.error("Seed %s worker failed: %s", seed.get("seed_id"), e)
                results.append(seed)
                continue

            # optionally persist updated seed to file if _seed_file_path is present
            if persist and updated_seed.get("_seed_file_path"):
                try:
                    p = Path(updated_seed["_seed_file_path"])
                    with open(p, "w", encoding="utf-8") as fh:
                        json.dump(updated_seed, fh, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning("Failed to persist seed %s: %s", updated_seed.get("seed_id"), e)

            results.append(updated_seed)

    # final stats print
    stats.print_summary()
    return results

# %% Usage
if __name__ == "__main__":
    # build key pool from (key,email) pairs you defined earlier
    key_pool = build_api_key_pool(PUBMED_KEYS, delay_per_key=0.11)

    if key_pool.empty():
        logger.error("No API keys available. Add PUBMED_KEYS before running.")
        raise SystemExit(1)

    # load seeds
    seeds = load_seeds_from_path(SEED_PATH)

    # create stats
    stats = SearchStats(total_seeds=len(seeds))

    # run searches (prints progress per-term and periodic summary)
    updated_seeds = run_pubmed_search_parallel(
        seeds,
        key_pool,
        workers=WORKERS,          # concurrency level (can be larger than number of keys)
        top_k_terms=TOP_K_TERMS,
        retmax=RETMAX,
        persist=PERSIST,      # set True to write seed files back
        stats=stats,
        verbose=VERBOSE
    )
