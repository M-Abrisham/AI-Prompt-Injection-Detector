#!/usr/bin/env python3
"""Parallel social media scraper for prompt injection and jailbreak content.

Runs a team of parallel scraping agents every 3 hours via GitHub Actions:

    Agent 1  RedditAgent   -- searches Reddit public JSON API
    Agent 2  TwitterAgent  -- searches Twitter/X API v2
    Agent 3  DatasetAgent  -- downloads static dataset sources

All agents run concurrently via ``ThreadPoolExecutor``.  Results are
deduplicated, classified, and merged into standard JSONL format.

Output schema::

    {"text": "...", "label": 0|1, "source": "...", "category": "...",
     "confidence": 0.0-1.0, "scraped_at": "..."}

Usage::

    python scripts/social_scraper.py
    python scripts/social_scraper.py --since-hours 3 --sources reddit,twitter
    python scripts/social_scraper.py --dry-run --verbose
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDDIT_SUBREDDITS = [
    "ChatGPT",
    "LocalLLaMA",
    "PromptEngineering",
    "MachineLearning",
    "artificial",
    "cybersecurity",
    "netsec",
    "ChatGPTPromptGenius",
    "generativeAI",
    "datasets",
    "OpenAI",
    "ClaudeAI",
    "ArtificialIntelligence",
]

SEARCH_KEYWORDS = [
    "prompt injection",
    "jailbreak prompt",
    "DAN prompt",
    "ignore previous instructions",
    "system prompt bypass",
    "prompt leak",
    "prompt hacking",
    "LLM jailbreak",
    "chatgpt jailbreak",
    "AI safety bypass",
    "prompt injection attack",
    "indirect prompt injection",
]

# Regex patterns that strongly indicate prompt injection content
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?prior\s+(instructions|rules|guidelines)", re.I),
    re.compile(r"you\s+are\s+now\s+(DAN|evil|unrestricted|jailbroken)", re.I),
    re.compile(r"do\s+anything\s+now", re.I),
    re.compile(r"developer\s+mode\s+(enabled|activated)", re.I),
    re.compile(r"(act|behave)\s+as\s+(a\s+)?(evil|unrestricted|unfiltered)", re.I),
    re.compile(r"bypass\s+(your\s+)?(safety|restrictions|guidelines|rules)", re.I),
    re.compile(r"override\s+(your\s+)?(restrictions|safety|rules)", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|instructions|rules)", re.I),
    re.compile(
        r"what\s+(is|are)\s+your\s+(system\s+prompt|initial\s+instructions)", re.I
    ),
    re.compile(r"pretend\s+(you|that)\s+(are|have)\s+no\s+(restrictions|rules)", re.I),
    re.compile(r"\bSTAN\s+mode\b", re.I),
    re.compile(r"\bopposite\s+mode\b", re.I),
    re.compile(r"(new|special)\s+instructions?\s*:", re.I),
    re.compile(r"\[system\]|\[INST\]|<\|system\|>", re.I),
]

# Weaker signals -- injection-adjacent but not conclusive alone
WEAK_SIGNALS = [
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bprompt\s+injection\b", re.I),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"\binitial\s+prompt\b", re.I),
    re.compile(r"\brole\s*play\b", re.I),
    re.compile(r"\bprompt\s+leak(age)?\b", re.I),
]

REDDIT_USER_AGENT = (
    "Na0S-Scraper/1.0 (prompt injection research;"
    " https://github.com/M-Abrisham/Na0S)"
)
REDDIT_RATE_LIMIT_SECONDS = 2.0
TWITTER_RATE_LIMIT_SECONDS = 1.0
HTTP_RETRY_WAIT = 5

# Static dataset sources derived from user-provided URLs
STATIC_DATASETS = [
    {
        "name": "kaggle_injection_wild",
        "type": "kaggle",
        "dataset_id": "arielzilber/prompt-injection-in-the-wild",
        "text_column": "Prompt",
        "label": 1,
        "description": "Prompt injection examples in the wild (Kaggle, MIT)",
    },
    {
        "name": "cyberseceval_prompt_injection",
        "type": "github_json",
        "urls": [
            (
                "https://raw.githubusercontent.com/meta-llama/PurpleLlama"
                "/main/CyberSecEval/prompt_injection/test_cases"
                "/prompt_injection.json"
            ),
        ],
        "text_key": "test_case_prompt",
        "label": 1,
        "description": "Meta CyberSecEval prompt injection test cases",
    },
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _http_get(url, headers=None, timeout=30):
    """Perform an HTTP GET and return the response body as a string.

    Handles gzip encoding and retries once on 429 (rate limit).
    Returns ``None`` on failure.
    """
    hdrs = {"Accept-Encoding": "identity"}
    if headers:
        hdrs.update(headers)
    req = Request(url, headers=hdrs)
    for attempt in range(2):
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                encoding = resp.headers.get("Content-Encoding", "")
                if encoding == "gzip":
                    data = gzip.decompress(data)
                return data.decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                wait = int(exc.headers.get("Retry-After", HTTP_RETRY_WAIT))
                time.sleep(min(wait, 30))
                continue
            return None
        except (URLError, OSError):
            return None
    return None


def _http_get_json(url, headers=None, timeout=30):
    """GET *url* and return parsed JSON, or ``None`` on failure."""
    body = _http_get(url, headers=headers, timeout=timeout)
    if body is None:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None


def _normalize_text(text):
    """Normalize text for deduplication: lowercase, collapse whitespace."""
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _text_hash(text):
    """SHA-256 hash of normalized text for deduplication."""
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _classify_injection(text):
    """Heuristic classifier for prompt injection content.

    Returns ``(label, confidence)`` where label is 1 (injection) or
    0 (benign) and confidence is a float between 0.0 and 1.0.
    """
    if not text or len(text.strip()) < 10:
        return 0, 0.0

    strong_hits = sum(1 for p in INJECTION_PATTERNS if p.search(text))
    weak_hits = sum(1 for p in WEAK_SIGNALS if p.search(text))

    if strong_hits >= 2:
        return 1, 0.95
    if strong_hits == 1:
        return 1, 0.80
    if weak_hits >= 3:
        return 1, 0.60
    if weak_hits >= 1:
        return 1, 0.40
    return 0, 0.10


def _load_known_hashes(path):
    """Load set of known text hashes from file."""
    if not os.path.isfile(path):
        return set()
    with open(path, "r") as fh:
        return {
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("#")
        }


def _save_known_hashes(hashes, path):
    """Save set of text hashes to file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# Known text hashes for deduplication\n")
        for h in sorted(hashes):
            fh.write(h + "\n")


def _write_jsonl(records, path):
    """Write *records* as JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    """Read JSONL file and return list of dicts."""
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _log(msg, verbose=True):
    """Print timestamped log message when *verbose* is truthy."""
    if verbose:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [{ts}] {msg}")


# ---------------------------------------------------------------------------
# Agent 1: RedditAgent
# ---------------------------------------------------------------------------


class RedditAgent:
    """Scrapes Reddit for prompt injection and jailbreak content
    using the public JSON API (no authentication required).
    """

    def __init__(self, since_hours=3, verbose=False):
        self.since_hours = since_hours
        self.verbose = verbose
        self._last_request = 0.0

    def _rate_limit(self):
        """Respect Reddit's rate limits (1 req per 2 s unauthenticated)."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < REDDIT_RATE_LIMIT_SECONDS:
            time.sleep(REDDIT_RATE_LIMIT_SECONDS - elapsed)
        self._last_request = time.monotonic()

    def _get_json(self, url):
        """Fetch JSON from Reddit API with rate limiting."""
        self._rate_limit()
        headers = {"User-Agent": REDDIT_USER_AGENT}
        return _http_get_json(url, headers=headers, timeout=30)

    def search_subreddit(self, subreddit, keyword):
        """Search a single subreddit for *keyword*."""
        query = quote_plus(keyword)
        url = (
            f"https://www.reddit.com/r/{subreddit}/search.json"
            f"?q={query}&sort=new&restrict_sr=on&t=week&limit=50"
        )
        data = self._get_json(url)
        if not data or "data" not in data:
            return []
        return self._extract_posts(data, subreddit, keyword)

    def search_global(self, keyword):
        """Search all of Reddit for *keyword*."""
        query = quote_plus(keyword)
        url = (
            f"https://www.reddit.com/search.json"
            f"?q={query}&sort=new&t=week&limit=100"
        )
        data = self._get_json(url)
        if not data or "data" not in data:
            return []
        return self._extract_posts(data, "global", keyword)

    def _extract_posts(self, data, subreddit, keyword):
        """Pull structured records out of a Reddit API response."""
        records = []
        cutoff = time.time() - (self.since_hours * 3600)
        children = data.get("data", {}).get("children", [])
        for child in children:
            post = child.get("data", {})
            created = post.get("created_utc", 0)
            if created < cutoff:
                continue

            sub = post.get("subreddit", subreddit)
            post_id = post.get("id", "")

            # --- title ---
            title = str(post.get("title", "")).strip()
            if title and len(title) >= 20:
                label, conf = _classify_injection(title)
                records.append({
                    "text": title,
                    "label": label,
                    "source": f"reddit/r/{sub}",
                    "category": "injection" if label == 1 else "discussion",
                    "confidence": conf,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "meta_id": f"reddit_{post_id}_title",
                    "search_keyword": keyword,
                })

            # --- body ---
            selftext = str(post.get("selftext", "")).strip()
            if selftext and len(selftext) >= 30 and selftext != "[removed]":
                label, conf = _classify_injection(selftext)
                records.append({
                    "text": selftext,
                    "label": label,
                    "source": f"reddit/r/{sub}",
                    "category": "injection" if label == 1 else "discussion",
                    "confidence": conf,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "meta_id": f"reddit_{post_id}_body",
                    "search_keyword": keyword,
                })

        return records

    def scrape_thread(self, permalink):
        """Scrape comments from a specific Reddit thread."""
        url = f"https://www.reddit.com{permalink}.json?limit=200"
        data = self._get_json(url)
        if not data or not isinstance(data, list) or len(data) < 2:
            return []

        records = []
        comments = data[1].get("data", {}).get("children", [])
        for child in comments:
            if child.get("kind") != "t1":
                continue
            body = str(child.get("data", {}).get("body", "")).strip()
            if body and len(body) >= 30 and body != "[removed]":
                label, conf = _classify_injection(body)
                if label == 1:
                    records.append({
                        "text": body,
                        "label": label,
                        "source": "reddit/comment",
                        "category": "injection",
                        "confidence": conf,
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        "meta_id": f"reddit_{child['data'].get('id', '')}",
                    })
        return records

    def search(self):
        """Run all Reddit searches and return collected records."""
        _log("RedditAgent: starting search ...", self.verbose)
        all_records = []

        # Global searches for top keywords
        for kw in SEARCH_KEYWORDS[:6]:
            _log(f"  Reddit global: {kw!r}", self.verbose)
            try:
                all_records.extend(self.search_global(kw))
            except Exception as exc:
                _log(f"  Reddit error (global/{kw}): {exc}", self.verbose)

        # Targeted subreddit searches
        targets = [
            ("ChatGPT", "jailbreak"),
            ("ChatGPT", "prompt injection"),
            ("LocalLLaMA", "jailbreak"),
            ("ChatGPTPromptGenius", "prompt injection"),
            ("cybersecurity", "prompt injection"),
            ("netsec", "LLM injection"),
            ("OpenAI", "jailbreak"),
            ("ClaudeAI", "prompt injection"),
            ("datasets", "prompt injection dataset"),
            ("generativeAI", "jailbreak"),
        ]
        for sub, kw in targets:
            _log(f"  Reddit r/{sub}: {kw!r}", self.verbose)
            try:
                all_records.extend(self.search_subreddit(sub, kw))
            except Exception as exc:
                _log(f"  Reddit error (r/{sub}/{kw}): {exc}", self.verbose)

        _log(f"RedditAgent: {len(all_records)} raw records", self.verbose)
        return all_records


# ---------------------------------------------------------------------------
# Agent 2: TwitterAgent
# ---------------------------------------------------------------------------


class TwitterAgent:
    """Scrapes Twitter/X for prompt injection and jailbreak content
    using the v2 API.  Requires ``TWITTER_BEARER_TOKEN`` env var.
    """

    API_URL = "https://api.twitter.com/2/tweets/search/recent"

    def __init__(self, bearer_token=None, since_hours=3, verbose=False):
        self.bearer_token = (
            bearer_token
            or os.environ.get("TWITTER_BEARER_TOKEN", "")
        )
        self.since_hours = since_hours
        self.verbose = verbose
        self._last_request = 0.0

    def _rate_limit(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < TWITTER_RATE_LIMIT_SECONDS:
            time.sleep(TWITTER_RATE_LIMIT_SECONDS - elapsed)
        self._last_request = time.monotonic()

    def _search_tweets(self, query):
        """Search Twitter API v2 for recent tweets matching *query*."""
        if not self.bearer_token:
            return []

        self._rate_limit()

        start_time = (
            datetime.now(timezone.utc) - timedelta(hours=self.since_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = urlencode({
            "query": f"{query} -is:retweet lang:en",
            "max_results": 100,
            "start_time": start_time,
            "tweet.fields": "created_at,text,author_id",
        })
        url = f"{self.API_URL}?{params}"

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": REDDIT_USER_AGENT,
        }
        data = _http_get_json(url, headers=headers, timeout=30)
        if not data or "data" not in data:
            return []

        records = []
        for tweet in data.get("data", []):
            text = str(tweet.get("text", "")).strip()
            if not text or len(text) < 20:
                continue

            label, conf = _classify_injection(text)
            records.append({
                "text": text,
                "label": label,
                "source": "twitter",
                "category": "injection" if label == 1 else "discussion",
                "confidence": conf,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "meta_id": f"tweet_{tweet.get('id', '')}",
                "search_keyword": query,
            })

        return records

    def search(self):
        """Run all Twitter searches and return collected records."""
        if not self.bearer_token:
            _log(
                "TwitterAgent: TWITTER_BEARER_TOKEN not set, skipping",
                self.verbose,
            )
            return []

        _log("TwitterAgent: starting search ...", self.verbose)
        all_records = []

        queries = [
            '"prompt injection"',
            '"jailbreak" "LLM"',
            '"ignore previous instructions"',
            '"DAN prompt"',
            '"system prompt bypass"',
            '"prompt hacking"',
        ]

        for q in queries:
            _log(f"  Twitter: {q}", self.verbose)
            try:
                all_records.extend(self._search_tweets(q))
            except Exception as exc:
                _log(f"  Twitter error ({q}): {exc}", self.verbose)

        _log(f"TwitterAgent: {len(all_records)} raw records", self.verbose)
        return all_records


# ---------------------------------------------------------------------------
# Agent 3: DatasetAgent
# ---------------------------------------------------------------------------


class DatasetAgent:
    """Downloads and processes static dataset sources from Kaggle,
    GitHub repos, and direct CSV/JSON URLs.
    """

    def __init__(self, output_dir="data/scraped/static", verbose=False):
        self.output_dir = output_dir
        self.verbose = verbose

    # -- Kaggle ---------------------------------------------------------------

    def _download_kaggle(self, dataset_id, text_column, default_label):
        """Download a Kaggle dataset via the ``kaggle`` CLI."""
        import shutil
        import subprocess
        import tempfile

        if not shutil.which("kaggle"):
            _log(
                f"  kaggle CLI not found, skipping {dataset_id}",
                self.verbose,
            )
            return []

        tmpdir = tempfile.mkdtemp(prefix="na0s_kaggle_")
        try:
            cmd = [
                "kaggle", "datasets", "download",
                "-d", dataset_id, "-p", tmpdir, "--unzip",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                _log(
                    f"  Kaggle error: {result.stderr[:200]}", self.verbose,
                )
                return []

            records = []
            for root, _dirs, files in os.walk(tmpdir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if fname.endswith(".csv"):
                        records.extend(
                            self._parse_csv(fpath, text_column, default_label)
                        )
                    elif fname.endswith((".json", ".jsonl")):
                        records.extend(
                            self._parse_json_file(
                                fpath, text_column, default_label,
                            )
                        )
            return records

        except subprocess.TimeoutExpired:
            _log(f"  Kaggle download timed out: {dataset_id}", self.verbose)
            return []
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # -- CSV ------------------------------------------------------------------

    def _parse_csv(self, path, text_column, default_label):
        """Parse a CSV file into standard records."""
        records = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    text = str(row.get(text_column, "")).strip()
                    if not text or len(text) < 10:
                        continue
                    label = default_label
                    if "label" in row:
                        try:
                            label = int(row["label"])
                        except (ValueError, TypeError):
                            pass
                    records.append({
                        "text": text,
                        "label": label,
                        "source": os.path.basename(path),
                        "category": "injection" if label == 1 else "benign",
                    })
        except Exception as exc:
            _log(f"  CSV parse error ({path}): {exc}", self.verbose)
        return records

    # -- JSON / JSONL ---------------------------------------------------------

    def _parse_json_file(self, path, text_key, default_label):
        """Parse a JSON or JSONL file into standard records."""
        records = []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
            if not content:
                return []
            if content.startswith("["):
                data = json.loads(content)
            else:
                data = [
                    json.loads(line)
                    for line in content.splitlines()
                    if line.strip()
                ]

            for item in data:
                if isinstance(item, str):
                    text = item.strip()
                elif isinstance(item, dict):
                    text = str(item.get(text_key, "")).strip()
                else:
                    continue
                if not text or len(text) < 10:
                    continue
                label = default_label
                if isinstance(item, dict) and "label" in item:
                    try:
                        label = int(item["label"])
                    except (ValueError, TypeError):
                        pass
                records.append({
                    "text": text,
                    "label": label,
                    "source": os.path.basename(path),
                    "category": "injection" if label == 1 else "benign",
                })
        except Exception as exc:
            _log(f"  JSON parse error ({path}): {exc}", self.verbose)
        return records

    # -- GitHub JSON ----------------------------------------------------------

    def _download_github_json(self, urls, text_key, label_val):
        """Download JSON files from GitHub raw URLs."""
        records = []
        for url in urls:
            _log(f"  Downloading {url.split('/')[-1]} ...", self.verbose)
            body = _http_get(url, timeout=60)
            if not body:
                _log(f"  Failed: {url}", self.verbose)
                continue
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    # Might be {"test_cases": [...]} or similar
                    for key in ("test_cases", "data", "items", "prompts"):
                        if key in data and isinstance(data[key], list):
                            data = data[key]
                            break
                    else:
                        data = list(data.values())
                if not isinstance(data, list):
                    continue
                for item in data:
                    if isinstance(item, str):
                        text = item.strip()
                    elif isinstance(item, dict):
                        text = str(item.get(text_key, "")).strip()
                    else:
                        continue
                    if text and len(text) >= 10:
                        records.append({
                            "text": text,
                            "label": label_val,
                            "source": url.split("/")[-1],
                            "category": (
                                "injection" if label_val == 1 else "benign"
                            ),
                        })
            except (json.JSONDecodeError, ValueError) as exc:
                _log(f"  JSON decode error: {exc}", self.verbose)
        return records

    # -- Orchestrator ---------------------------------------------------------

    def download_all(self):
        """Download all registered static dataset sources."""
        _log("DatasetAgent: starting downloads ...", self.verbose)
        all_records = []

        for ds in STATIC_DATASETS:
            _log(f"  Source: {ds['name']}", self.verbose)
            try:
                if ds["type"] == "kaggle":
                    recs = self._download_kaggle(
                        ds["dataset_id"],
                        ds["text_column"],
                        ds.get("label", 1),
                    )
                elif ds["type"] == "github_json":
                    recs = self._download_github_json(
                        ds.get("urls", []),
                        ds.get("text_key", "text"),
                        ds.get("label", 1),
                    )
                else:
                    _log(f"  Unknown type: {ds['type']}", self.verbose)
                    recs = []

                if recs:
                    out_path = os.path.join(
                        self.output_dir, f"{ds['name']}.jsonl",
                    )
                    _write_jsonl(recs, out_path)
                    _log(f"  {len(recs)} records -> {out_path}", self.verbose)

                all_records.extend(recs)

            except Exception as exc:
                _log(f"  DatasetAgent error ({ds['name']}): {exc}", self.verbose)

        _log(f"DatasetAgent: {len(all_records)} total records", self.verbose)
        return all_records


# ---------------------------------------------------------------------------
# Coordinator -- parallel agent execution
# ---------------------------------------------------------------------------


def run_scrape(
    sources=("reddit", "twitter", "datasets"),
    since_hours=3,
    output_dir="data/scraped",
    dry_run=False,
    verbose=False,
):
    """Run all scraping agents in parallel and merge results.

    Returns a stats dict with counts and timing information.
    """
    _log(f"Coordinator: sources={sources}, window={since_hours}h", verbose)
    start = time.monotonic()

    # Instantiate requested agents
    agents = {}
    if "reddit" in sources:
        agents["reddit"] = RedditAgent(
            since_hours=since_hours, verbose=verbose,
        )
    if "twitter" in sources:
        agents["twitter"] = TwitterAgent(
            since_hours=since_hours, verbose=verbose,
        )
    if "datasets" in sources:
        agents["datasets"] = DatasetAgent(
            output_dir=os.path.join(output_dir, "static"),
            verbose=verbose,
        )

    if not agents:
        _log("No agents to run.", verbose)
        return {"new_count": 0, "total_raw": 0, "sources": {}}

    # Run agents in parallel via ThreadPoolExecutor
    results = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        future_map = {}
        for name, agent in agents.items():
            if name == "datasets":
                future = pool.submit(agent.download_all)
            else:
                future = pool.submit(agent.search)
            future_map[future] = name

        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                _log(f"Agent '{name}' crashed: {exc}", verbose)
                results[name] = []

    # Merge all records
    all_records = []
    for recs in results.values():
        all_records.extend(recs)

    _log(f"Collected {len(all_records)} raw records total", verbose)

    # Deduplicate against known hashes
    hashes_path = os.path.join(output_dir, "known_hashes.txt")
    known_hashes = _load_known_hashes(hashes_path)
    _log(f"Loaded {len(known_hashes)} known hashes", verbose)

    new_records = []
    new_hashes = set()
    for rec in all_records:
        h = _text_hash(rec.get("text", ""))
        if h not in known_hashes and h not in new_hashes:
            new_records.append(rec)
            new_hashes.add(h)

    _log(f"After dedup: {len(new_records)} new records", verbose)

    if dry_run:
        _log("Dry run -- skipping writes", verbose)
        elapsed = time.monotonic() - start
        return {
            "new_count": len(new_records),
            "total_raw": len(all_records),
            "duplicates_removed": len(all_records) - len(new_records),
            "sources": {k: len(v) for k, v in results.items()},
            "elapsed_seconds": round(elapsed, 1),
            "dry_run": True,
        }

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)

    if new_records:
        # Per-run snapshot
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(output_dir, f"scrape_{ts}.jsonl")
        _write_jsonl(new_records, snapshot_path)
        _log(f"Wrote {len(new_records)} records -> {snapshot_path}", verbose)

        # Append to running merged file
        merged_path = os.path.join(output_dir, "merged_scrape.jsonl")
        with open(merged_path, "a", encoding="utf-8") as fh:
            for rec in new_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _log(f"Appended to {merged_path}", verbose)

        # Update known hashes
        known_hashes.update(new_hashes)
        _save_known_hashes(known_hashes, hashes_path)
        _log(f"Known hashes now: {len(known_hashes)}", verbose)

    # Scrape history
    elapsed = time.monotonic() - start
    label_dist = {
        "injection": sum(1 for r in new_records if r.get("label") == 1),
        "benign": sum(1 for r in new_records if r.get("label") == 0),
    }
    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "new_count": len(new_records),
        "total_raw": len(all_records),
        "duplicates_removed": len(all_records) - len(new_records),
        "sources": {k: len(v) for k, v in results.items()},
        "elapsed_seconds": round(elapsed, 1),
        "label_distribution": label_dist,
    }

    history_path = os.path.join(output_dir, "scrape_history.json")
    history = []
    if os.path.isfile(history_path):
        try:
            with open(history_path, "r") as fh:
                history = json.load(fh)
        except (json.JSONDecodeError, ValueError):
            pass
    history.append(stats)
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)

    _log(f"Done in {elapsed:.1f}s -- {len(new_records)} new records", verbose)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Parallel social media scraper for prompt injection content.  "
            "Runs Reddit, Twitter/X, and static dataset agents concurrently."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/scraped",
        help="Output directory (default: data/scraped/).",
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=3,
        help="Scrape content from the last N hours (default: 3).",
    )
    parser.add_argument(
        "--sources",
        default="reddit,twitter,datasets",
        help=(
            "Comma-separated sources to scrape.  "
            "Options: reddit, twitter, datasets (default: all)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search but do not write output files.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress messages.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    valid_sources = {"reddit", "twitter", "datasets"}
    for s in sources:
        if s not in valid_sources:
            print(
                f"ERROR: unknown source '{s}'.  "
                f"Valid: {', '.join(sorted(valid_sources))}",
                file=sys.stderr,
            )
            return 1

    stats = run_scrape(
        sources=sources,
        since_hours=args.since_hours,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # Summary
    print(f"\n{'=' * 55}")
    print("Scrape complete:")
    print(f"  New records:        {stats['new_count']}")
    print(f"  Total raw:          {stats['total_raw']}")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    print(f"  Sources:            {stats['sources']}")
    print(f"  Elapsed:            {stats['elapsed_seconds']}s")
    if not stats.get("dry_run"):
        dist = stats.get("label_distribution", {})
        print(f"  Injection:          {dist.get('injection', 0)}")
        print(f"  Benign/discussion:  {dist.get('benign', 0)}")
    print(f"{'=' * 55}")

    # GitHub Actions output
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as fh:
            fh.write(f"new_count={stats['new_count']}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
