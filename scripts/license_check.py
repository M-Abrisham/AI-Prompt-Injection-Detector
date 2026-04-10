#!/usr/bin/env python3
"""License compliance checking for dataset sources.

Reads data/datasets.yaml, checks license fields for HuggingFace datasets,
and reports ALLOWED/BLOCKED/REVIEW status for each source.

Caches license info in data/license_cache.yaml for offline operation.

Usage::

    python scripts/license_check.py
    python scripts/license_check.py --strict   # exit non-zero if any BLOCKED/REVIEW
    python scripts/license_check.py --refresh   # force re-fetch from HuggingFace
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS_PATH = os.path.join(ROOT, "data", "datasets.yaml")
CACHE_PATH = os.path.join(ROOT, "data", "license_cache.yaml")

# ── License classification ──────────────────────────────────────────
ALLOWED_LICENSES = {
    "mit", "apache-2.0", "cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0",
    "odc-by-1.0", "openrail", "openrail++", "bigscience-openrail-m",
    "cc-by-3.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense",
}

BLOCKED_PREFIXES = [
    "cc-by-nc", "gpl", "agpl", "lgpl", "proprietary",
    "cc-by-nd",
]


def _classify_license(license_id: str | None) -> str:
    """Return ALLOWED, BLOCKED, or REVIEW for a license identifier."""
    if not license_id:
        return "REVIEW"

    normalized = license_id.lower().strip()

    if normalized in ALLOWED_LICENSES:
        return "ALLOWED"

    for prefix in BLOCKED_PREFIXES:
        if normalized.startswith(prefix):
            return "BLOCKED"

    return "REVIEW"


def _load_cache() -> dict:
    """Load cached license info."""
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_cache(cache: dict):
    """Save license cache."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cache, f, default_flow_style=False)


def _fetch_hf_license(repo: str) -> str | None:
    """Fetch license from HuggingFace dataset card."""
    try:
        from huggingface_hub import dataset_info
        info = dataset_info(repo)
        return getattr(info, "license", None) or getattr(info, "cardData", {}).get("license")
    except Exception:
        return None


def check_licenses(
    refresh: bool = False,
    strict: bool = False,
) -> list[dict]:
    """Check licenses for all dataset sources.

    Returns a list of dicts with keys: source, repo, license, status.
    """
    if not os.path.isfile(DATASETS_PATH):
        print(f"ERROR: datasets.yaml not found: {DATASETS_PATH}")
        sys.exit(1)

    with open(DATASETS_PATH) as f:
        config = yaml.safe_load(f)

    sources = config.get("sources", {})
    cache = _load_cache() if not refresh else {}
    results = []
    cache_updated = False

    for name, spec in sorted(sources.items()):
        repo = spec.get("repo", "")
        source_type = spec.get("type", "")

        if source_type == "huggingface" and repo:
            # Check cache first
            if repo in cache and not refresh:
                license_id = cache[repo]
            else:
                print(f"  Fetching license for {repo}...")
                license_id = _fetch_hf_license(repo)
                cache[repo] = license_id
                cache_updated = True

            status = _classify_license(license_id)
            results.append({
                "source": name,
                "repo": repo,
                "license": license_id or "unknown",
                "status": status,
            })
        elif source_type == "github_csv":
            # GitHub CSVs — check URL for known license patterns
            url = spec.get("url", "")
            results.append({
                "source": name,
                "repo": url[:60] + "..." if len(url) > 60 else url,
                "license": "check-repo",
                "status": "REVIEW",
            })
        else:
            results.append({
                "source": name,
                "repo": repo or "n/a",
                "license": "unknown",
                "status": "REVIEW",
            })

    # Save updated cache
    if cache_updated:
        _save_cache(cache)

    # Print table
    print(f"\n{'Source':<30} {'License':<25} {'Status':<10}")
    print("-" * 65)
    for r in results:
        marker = ""
        if r["status"] == "BLOCKED":
            marker = " <-- BLOCKED"
        elif r["status"] == "REVIEW":
            marker = " <-- REVIEW"
        print(f"{r['source']:<30} {r['license']:<25} {r['status']:<10}{marker}")

    # Summary
    allowed = sum(1 for r in results if r["status"] == "ALLOWED")
    blocked = sum(1 for r in results if r["status"] == "BLOCKED")
    review = sum(1 for r in results if r["status"] == "REVIEW")

    print(f"\n{'=' * 50}")
    print(f"License Compliance Summary")
    print(f"{'=' * 50}")
    print(f"  ALLOWED: {allowed}")
    print(f"  BLOCKED: {blocked}")
    print(f"  REVIEW:  {review}")
    print(f"  Total:   {len(results)}")
    print(f"{'=' * 50}")

    if strict and (blocked > 0 or review > 0):
        print("\nFAIL: Strict mode — blocked or review-required licenses found.")
        sys.exit(1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Check dataset license compliance.")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any BLOCKED or REVIEW licenses found.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-fetch licenses from HuggingFace (ignore cache).",
    )
    args = parser.parse_args()
    check_licenses(refresh=args.refresh, strict=args.strict)


if __name__ == "__main__":
    main()
