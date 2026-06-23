#!/usr/bin/env python3
"""Discover and harvest new prompt-injection datasets from public sources.

Searches HuggingFace Hub, arXiv, and GitHub for recently published datasets,
papers, and repositories related to prompt injection.  Designed to run as a
weekly cron job or GitHub Actions schedule.

Output files (written to ``data/harvest/`` by default)::

    latest_scan.json    -- full scan results from the most recent run
    new_datasets.jsonl  -- append-only log of every newly discovered entry
    scan_history.json   -- history of all scans with dates and counts
    known_datasets.txt  -- registry of already-known dataset IDs (one per line)

Usage
-----
    python scripts/weekly_harvest.py
    python scripts/weekly_harvest.py --since-days 30
    python scripts/weekly_harvest.py --sources hf,arxiv
    python scripts/weekly_harvest.py --output-dir /tmp/harvest
    python scripts/weekly_harvest.py --dry-run
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("weekly_harvest")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_SEARCH_API = (
    "https://huggingface.co/api/datasets"
    "?search={query}&sort=lastModified&direction=-1&limit=50"
)

ARXIV_API = (
    "http://export.arxiv.org/api/query"
    "?search_query={query}&sortBy=submittedDate"
    "&sortOrder=descending&max_results=20"
)

GITHUB_SEARCH_API = (
    "https://api.github.com/search/repositories"
    "?q={query}+pushed:>{since_date}&sort=updated&per_page=20"
)

HF_QUERIES = [
    "prompt injection",
    "jailbreak",
    "adversarial prompt",
    "red teaming LLM",
    "prompt security",
    "guardrail bypass",
    "LLM safety",
    # Adversarial-suffix / gradient-optimized jailbreak family (D7.5/A1.1) — the
    # coverage gap the static list missed; needed to discover datasets for the
    # GCG/AdvBench retrain path.
    "GCG adversarial suffix",
    "AdvBench harmful behaviors",
    "AutoDAN jailbreak",
    "HarmBench",
    "universal adversarial trigger",
    # Indirect / RAG-poisoning ingestion attacks (I1/IG).
    "indirect prompt injection",
    "RAG poisoning",
    "retrieval augmented generation attack",
]

ARXIV_QUERIES = [
    "all:prompt+injection+LLM",
    "all:jailbreak+large+language+model",
    "all:adversarial+prompt+attack",
    "all:GCG+adversarial+suffix+language+model",
    "all:AutoDAN+jailbreak",
    "all:universal+adversarial+trigger+LLM",
    "all:retrieval+augmented+generation+poisoning",
]

GITHUB_QUERIES = [
    "prompt+injection+dataset",
    "jailbreak+LLM+dataset",
    "adversarial+prompt+dataset",
    "GCG+adversarial+suffix",
    "llm-attacks",
    "AutoDAN+jailbreak",
    "HarmBench+behaviors",
    "RAG+poisoning+dataset",
]

# Dataset IDs already used in Na0S (pre-populated for known_datasets.txt)
SEED_KNOWN_DATASETS = [
    "imoxto/prompt_injection_cleaned_dataset-v2",
    "geekyrakshit/prompt-injection-dataset",
    "Lakera/gandalf-rct",
    "microsoft/llmail-inject-challenge",
    "hackaprompt/hackaprompt-dataset",
    "deepset/prompt-injections",
    "tatsu-lab/alpaca",
    "databricks/databricks-dolly-15k",
    "xTRam1/safe-guard-prompt-injection",
    "reshabhs/SPML_Chatbot_Prompt_Injection",
    "ethz-spylab/ctf-satml24",
    "OpenAssistant/oasst1",
]

DEFAULT_OUTPUT_DIR = "data/harvest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_requests():
    """Raise a clear error if the ``requests`` library is not installed."""
    if requests is None:
        print(
            "ERROR: the 'requests' package is required. "
            "Install it with:  pip install requests",
            file=sys.stderr,
        )
        sys.exit(1)


def _utcnow():
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def _iso_now():
    """Return the current UTC time as an ISO-8601 string."""
    return _utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _date_str(dt):
    """Return *dt* as a YYYY-MM-DD string."""
    return dt.strftime("%Y-%m-%d")


def _get_hf_token():
    """Return a HuggingFace API token if available, else None."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(token_path):
        try:
            with open(token_path, "r", encoding="utf-8") as fh:
                token = fh.read().strip()
            if token:
                return token
        except OSError:
            pass
    return None


def _get_github_token():
    """Return a GitHub API token if available, else None."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token.strip()
    return None


def _hf_headers():
    """Return headers for HuggingFace API requests."""
    token = _get_hf_token()
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_headers():
    """Return headers for GitHub API requests."""
    token = _get_github_token()
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _http_get(url, headers=None, timeout=60):
    """GET *url* with retry and rate-limit handling.  Returns a Response."""
    _ensure_requests()
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            # Respect rate limits
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 30))
                log.warning("Rate limited. Sleeping %ds ...", retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp
        except (requests.RequestException, IOError) as exc:
            last_exc = exc
            if attempt < 2:
                wait = 2 ** attempt
                log.warning("Retry %d/2 after %ds: %s", attempt + 1, wait, exc)
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _parse_iso_date(date_str):
    """Parse an ISO-8601 date string to a timezone-aware datetime.

    Handles common formats from HuggingFace, arXiv, and GitHub.
    """
    date_str = date_str.strip()
    # Remove trailing 'Z' and treat as UTC
    if date_str.endswith("Z"):
        date_str = date_str[:-1]
    # Remove fractional seconds for simpler parsing
    date_str = re.sub(r"\.\d+", "", date_str)
    # Remove timezone offset (we treat everything as UTC)
    date_str = re.sub(r"[+-]\d{2}:\d{2}$", "", date_str)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {date_str!r}")


# ---------------------------------------------------------------------------
# Known-datasets registry
# ---------------------------------------------------------------------------

def load_known_datasets(path):
    """Load known dataset IDs from registry file.

    Returns a set of dataset ID strings.  If the file does not exist,
    returns an empty set.
    """
    if not os.path.isfile(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids


def save_known_datasets(path, ids):
    """Save known dataset IDs to registry file (sorted, one per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Known dataset IDs -- managed by weekly_harvest.py\n")
        for dataset_id in sorted(ids):
            fh.write(dataset_id + "\n")


def _ensure_known_datasets(path):
    """Ensure the known-datasets file exists with seed entries."""
    if os.path.isfile(path):
        return
    log.info("Initializing known-datasets registry: %s", path)
    save_known_datasets(path, set(SEED_KNOWN_DATASETS))


# ---------------------------------------------------------------------------
# Source A: HuggingFace Hub
# ---------------------------------------------------------------------------

def scan_huggingface(queries, since_days=7, known_ids=None):
    """Search HF Hub API for new datasets.  Returns list of discovery dicts."""
    if known_ids is None:
        known_ids = set()

    cutoff = _utcnow() - timedelta(days=since_days)
    seen_ids = set()
    results = []

    for query in queries:
        url = HF_SEARCH_API.format(query=requests.utils.quote(query))
        log.info("[HF] Searching: %s", query)

        try:
            resp = _http_get(url, headers=_hf_headers())
            data = resp.json()
        except Exception as exc:
            log.warning("[HF] Search failed for %r: %s", query, exc)
            continue

        if not isinstance(data, list):
            log.warning("[HF] Unexpected response type for %r: %s",
                        query, type(data).__name__)
            continue

        for item in data:
            ds_id = item.get("id", "")
            if not ds_id or ds_id in seen_ids or ds_id in known_ids:
                continue
            seen_ids.add(ds_id)

            # Check modification date
            last_modified = item.get("lastModified", "")
            if last_modified:
                try:
                    mod_dt = _parse_iso_date(last_modified)
                    if mod_dt < cutoff:
                        continue
                except ValueError:
                    pass  # keep it if we can't parse the date

            tags = item.get("tags", [])
            description = item.get("description", "") or ""
            downloads = item.get("downloads", 0)

            # Estimate size from download count
            if downloads > 100000:
                size_est = "100K+"
            elif downloads > 10000:
                size_est = "10K+"
            elif downloads > 1000:
                size_est = "1K+"
            else:
                size_est = "unknown"

            # Record which query matched
            matched_keywords = [
                q for q in queries
                if q.lower() in (description + " " + " ".join(tags)).lower()
            ]

            results.append({
                "id": ds_id,
                "source": "huggingface",
                "discovered_at": _iso_now(),
                "url": f"https://huggingface.co/datasets/{ds_id}",
                "description": description[:500],
                "size_estimate": size_est,
                "tags": tags[:20],
                "relevance_keywords": matched_keywords or [query],
                "scan_date": _date_str(_utcnow()),
                "last_modified": last_modified,
                "downloads": downloads,
            })

        # Be polite to the API
        time.sleep(0.5)

    log.info("[HF] Found %d new dataset(s)", len(results))
    return results


# ---------------------------------------------------------------------------
# Source B: arXiv
# ---------------------------------------------------------------------------

def scan_arxiv(queries, since_days=7, known_ids=None):
    """Search arXiv API for recent papers.  Returns list of paper dicts.

    *known_ids* is the shared known-corpus registry (arXiv IDs, GitHub
    full_names, and HF IDs all live in one set). Papers whose arXiv ID is
    already known are skipped so weekly runs don't re-surface the same papers.
    """
    if known_ids is None:
        known_ids = set()

    cutoff = _utcnow() - timedelta(days=since_days)
    seen_ids = set()
    results = []

    # arXiv Atom namespace
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for query in queries:
        url = ARXIV_API.format(query=query)
        log.info("[arXiv] Searching: %s", query)

        try:
            resp = _http_get(url, timeout=30)
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            log.warning("[arXiv] XML parse error for %r: %s", query, exc)
            continue
        except Exception as exc:
            log.warning("[arXiv] Search failed for %r: %s", query, exc)
            continue

        for entry in root.findall("atom:entry", ns):
            arxiv_id_el = entry.find("atom:id", ns)
            if arxiv_id_el is None or arxiv_id_el.text is None:
                continue
            arxiv_id = arxiv_id_el.text.strip()
            if arxiv_id in seen_ids or arxiv_id in known_ids:
                continue
            seen_ids.add(arxiv_id)

            # Check published date
            published_el = entry.find("atom:published", ns)
            published_str = ""
            if published_el is not None and published_el.text:
                published_str = published_el.text.strip()
                try:
                    pub_dt = _parse_iso_date(published_str)
                    if pub_dt < cutoff:
                        continue
                except ValueError:
                    pass

            title_el = entry.find("atom:title", ns)
            title = (title_el.text.strip() if title_el is not None
                     and title_el.text else "")
            # Collapse newlines in title
            title = re.sub(r"\s+", " ", title)

            summary_el = entry.find("atom:summary", ns)
            summary = (summary_el.text.strip() if summary_el is not None
                       and summary_el.text else "")

            # Extract links
            links = []
            for link_el in entry.findall("atom:link", ns):
                href = link_el.get("href", "")
                if href:
                    links.append(href)

            # Extract GitHub / HuggingFace URLs from summary
            combined_text = f"{title} {summary}"
            github_urls = re.findall(
                r"https?://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+",
                combined_text,
            )
            hf_urls = re.findall(
                r"https?://huggingface\.co/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+",
                combined_text,
            )

            results.append({
                "id": arxiv_id,
                "source": "arxiv",
                "discovered_at": _iso_now(),
                "url": arxiv_id,
                "description": title,
                "size_estimate": "paper",
                "tags": [],
                "relevance_keywords": [query.replace("+", " ")],
                "scan_date": _date_str(_utcnow()),
                "published": published_str,
                "summary": summary[:500],
                "links": links,
                "github_urls": github_urls,
                "hf_urls": hf_urls,
            })

        # arXiv asks for 3s between requests
        time.sleep(3)

    log.info("[arXiv] Found %d recent paper(s)", len(results))
    return results


# ---------------------------------------------------------------------------
# Source C: GitHub
# ---------------------------------------------------------------------------

def scan_github(queries, since_days=7, known_ids=None):
    """Search GitHub API for new repos.  Returns list of repo dicts.

    *known_ids* is the shared known-corpus registry. Repos whose ``full_name``
    is already known are skipped so weekly runs don't re-surface the same repos.
    """
    if known_ids is None:
        known_ids = set()

    since_date = _date_str(_utcnow() - timedelta(days=since_days))
    seen_ids = set()
    results = []

    for query in queries:
        url = GITHUB_SEARCH_API.format(
            query=query,
            since_date=since_date,
        )
        log.info("[GitHub] Searching: %s", query)

        try:
            resp = _http_get(url, headers=_github_headers())
            data = resp.json()
        except Exception as exc:
            log.warning("[GitHub] Search failed for %r: %s", query, exc)
            continue

        items = data.get("items", [])
        if not isinstance(items, list):
            log.warning("[GitHub] Unexpected items type for %r", query)
            continue

        for repo in items:
            full_name = repo.get("full_name", "")
            if not full_name or full_name in seen_ids or full_name in known_ids:
                continue
            seen_ids.add(full_name)

            description = repo.get("description", "") or ""
            html_url = repo.get("html_url", "")
            stars = repo.get("stargazers_count", 0)
            updated_at = repo.get("updated_at", "")
            topics = repo.get("topics", []) or []

            # Estimate size from stars
            if stars > 1000:
                size_est = "1K+ stars"
            elif stars > 100:
                size_est = "100+ stars"
            else:
                size_est = f"{stars} stars"

            results.append({
                "id": full_name,
                "source": "github",
                "discovered_at": _iso_now(),
                "url": html_url,
                "description": description[:500],
                "size_estimate": size_est,
                "tags": topics[:20],
                "relevance_keywords": [query.replace("+", " ")],
                "scan_date": _date_str(_utcnow()),
                "stars": stars,
                "updated_at": updated_at,
            })

        # Respect GitHub rate limits
        time.sleep(2)

    log.info("[GitHub] Found %d repo(s)", len(results))
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_latest_scan(output_dir, scan_result):
    """Write full scan results to latest_scan.json."""
    path = os.path.join(output_dir, "latest_scan.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scan_result, fh, indent=2, ensure_ascii=False)
    log.info("Wrote %s", path)


def _append_new_datasets(output_dir, entries):
    """Append newly discovered entries to new_datasets.jsonl."""
    path = os.path.join(output_dir, "new_datasets.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("Appended %d entries to %s", len(entries), path)


def _update_scan_history(output_dir, summary):
    """Update scan_history.json with this scan's summary."""
    path = os.path.join(output_dir, "scan_history.json")
    os.makedirs(output_dir, exist_ok=True)

    history = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                history = json.load(fh)
        except (json.JSONDecodeError, OSError):
            log.warning("Could not read existing scan history; starting fresh.")
            history = []

    history.append(summary)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)
    log.info("Updated %s (%d total scans)", path, len(history))


# ---------------------------------------------------------------------------
# Taxonomy-aware tagging (additive, offline, keyless)
# ---------------------------------------------------------------------------

def _tag_discoveries(discoveries):
    """Attach a canonical Na0S ``attack_category`` to each discovery in place.

    Pure/offline: routes each record's relevance signals through
    ``na0s.eval.harvest.tag_discovery`` (ATLAS id -> mapped code, else curated
    keyword -> canonical code). Records with no confident, canonical match are
    left untagged (never guessed, never dropped). The na0s import is guarded so
    the standalone harvester still runs if the package is not importable.
    """
    try:
        from na0s.eval.harvest import tag_discovery
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        log.warning("Taxonomy tagging unavailable (%s); leaving records untagged", exc)
        return
    tagged = 0
    for entry in discoveries:
        try:
            category = tag_discovery(entry)
        except Exception as exc:  # pragma: no cover - never let tagging crash a scan
            log.warning("Tagging failed for %r: %s", entry.get("id"), exc)
            continue
        if category:
            entry["attack_category"] = category
            tagged += 1
    log.info("Tagged %d/%d discoveries with a canonical attack_category",
             tagged, len(discoveries))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_harvest(output_dir, since_days=7, sources=None, dry_run=False):
    """Main orchestrator.  Runs all scanners, deduplicates, writes output.

    Parameters
    ----------
    output_dir : str
        Directory for all output files.
    since_days : int
        Only include items modified/published in the last N days.
    sources : set[str] or None
        Which sources to scan.  Default: all (``{"hf", "arxiv", "github"}``).
    dry_run : bool
        If True, scan and print but do not write any files.

    Returns
    -------
    dict
        Scan summary with counts.
    """
    if sources is None:
        sources = {"hf", "arxiv", "github"}

    known_path = os.path.join(output_dir, "known_datasets.txt")
    if not dry_run:
        _ensure_known_datasets(known_path)
    known_ids = load_known_datasets(known_path)

    scan_time = _iso_now()
    all_discoveries = []
    source_counts = {}
    errors = {}

    # --- Source A: HuggingFace ---
    if "hf" in sources:
        try:
            hf_results = scan_huggingface(
                HF_QUERIES, since_days=since_days, known_ids=known_ids,
            )
            all_discoveries.extend(hf_results)
            source_counts["huggingface"] = len(hf_results)
        except Exception as exc:
            log.error("[HF] Scanner failed: %s", exc)
            errors["huggingface"] = str(exc)
            source_counts["huggingface"] = 0

    # --- Source B: arXiv ---
    if "arxiv" in sources:
        try:
            arxiv_results = scan_arxiv(
                ARXIV_QUERIES, since_days=since_days, known_ids=known_ids,
            )
            all_discoveries.extend(arxiv_results)
            source_counts["arxiv"] = len(arxiv_results)
        except Exception as exc:
            log.error("[arXiv] Scanner failed: %s", exc)
            errors["arxiv"] = str(exc)
            source_counts["arxiv"] = 0

    # --- Source C: GitHub ---
    if "github" in sources:
        try:
            gh_results = scan_github(
                GITHUB_QUERIES, since_days=since_days, known_ids=known_ids,
            )
            all_discoveries.extend(gh_results)
            source_counts["github"] = len(gh_results)
        except Exception as exc:
            log.error("[GitHub] Scanner failed: %s", exc)
            errors["github"] = str(exc)
            source_counts["github"] = 0

    # Deduplicate by (source, id)
    seen = set()
    unique = []
    for entry in all_discoveries:
        key = (entry["source"], entry["id"])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    all_discoveries = unique

    # Taxonomy-aware tagging (additive, offline). Attach a CANONICAL Na0S
    # attack_category to each discovery when one can be confidently resolved
    # from its relevance signals (ATLAS id or curated keyword); records with no
    # confident match are left untagged for manual mapping, never dropped.
    _tag_discoveries(all_discoveries)

    total = len(all_discoveries)

    # Build scan result
    scan_result = {
        "scan_time": scan_time,
        "since_days": since_days,
        "sources_scanned": sorted(sources),
        "total_discovered": total,
        "total_new": total,  # alias for backward compat with CI workflows
        "per_source_counts": source_counts,
        "errors": errors,
        "discoveries": all_discoveries,
    }

    # Build scan summary (for history)
    scan_summary = {
        "scan_time": scan_time,
        "since_days": since_days,
        "sources_scanned": sorted(sources),
        "total_discovered": total,
        "total_new": total,  # alias for backward compat with CI workflows
        "per_source_counts": source_counts,
        "errors": errors,
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Harvest Scan Summary")
    print(f"{'=' * 60}")
    print(f"  Scan time:    {scan_time}")
    print(f"  Since days:   {since_days}")
    print(f"  Sources:      {', '.join(sorted(sources))}")
    print(f"  Total found:  {total}")
    for src, count in sorted(source_counts.items()):
        print(f"    {src}: {count}")
    if errors:
        print(f"  Errors:       {len(errors)}")
        for src, msg in sorted(errors.items()):
            print(f"    {src}: {msg}")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        if all_discoveries:
            print(f"\nDiscoveries ({total}):")
            for entry in all_discoveries:
                print(f"  [{entry['source']}] {entry['id']}")
                print(f"    URL: {entry['url']}")
                desc = entry.get("description", "")
                if desc:
                    print(f"    Description: {desc[:120]}")
        return scan_result

    # Write outputs
    _write_latest_scan(output_dir, scan_result)
    if all_discoveries:
        _append_new_datasets(output_dir, all_discoveries)
    _update_scan_history(output_dir, scan_summary)

    # Update the shared known-corpus registry with ALL newly discovered IDs
    # (HuggingFace IDs, arXiv IDs, and GitHub full_names alike).  Previously
    # only HF IDs were persisted, so every weekly run re-surfaced the same
    # arXiv papers and GitHub repos.  One flat registry serves all three
    # sources because their ID namespaces are disjoint (HF "org/ds", arXiv
    # "http://arxiv.org/abs/...", GitHub "owner/repo").
    new_ids = {
        entry["id"] for entry in all_discoveries if entry.get("id")
    }
    new_ids -= known_ids
    if new_ids:
        updated_known = known_ids | new_ids
        save_known_datasets(known_path, updated_known)
        log.info("Added %d new ID(s) to known-datasets registry", len(new_ids))

    # Print discovered items
    if all_discoveries:
        print(f"\nNewly discovered ({total}):")
        for entry in all_discoveries:
            print(f"  [{entry['source']}] {entry['id']}")
            print(f"    URL: {entry['url']}")
    else:
        print("\nNo new items discovered.")

    print(f"\nOutput directory: {output_dir}")
    return scan_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Discover and harvest new prompt-injection datasets from "
            "HuggingFace, arXiv, and GitHub."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=7,
        help="Look back N days for new items (default: 7).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="hf,arxiv,github",
        help=(
            "Comma-separated list of sources to scan "
            "(choices: hf, arxiv, github; default: hf,arxiv,github)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and print results but do not write any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse sources
    valid_sources = {"hf", "arxiv", "github"}
    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    unknown = sources - valid_sources
    if unknown:
        print(
            f"WARNING: unknown source(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(sorted(valid_sources))}",
            file=sys.stderr,
        )
        sources = sources & valid_sources

    if not sources:
        print("ERROR: no valid sources specified.", file=sys.stderr)
        return 1

    print("Na0S Weekly Harvest")
    print(f"Output directory: {args.output_dir}")
    print(f"Since days:       {args.since_days}")
    print(f"Sources:          {', '.join(sorted(sources))}")
    print(f"Dry run:          {args.dry_run}")

    result = run_harvest(
        output_dir=args.output_dir,
        since_days=args.since_days,
        sources=sources,
        dry_run=args.dry_run,
    )

    error_count = len(result.get("errors", {}))
    if error_count:
        print(f"\nCompleted with {error_count} error(s).", file=sys.stderr)

    return 1 if error_count == len(sources) else 0


if __name__ == "__main__":
    sys.exit(main())
