#!/usr/bin/env python3
"""Check license compliance for all HuggingFace datasets in the registry.

Reads data/datasets.yaml, fetches license metadata from the HuggingFace Hub
API for each dataset, classifies licenses as permissive / restrictive /
unknown, and flags potential conflicts with the project's own license.

Usage:
    python scripts/license_checker.py
    python scripts/license_checker.py --datasets data/datasets.yaml
    python scripts/license_checker.py --output data/license_report.json
    python scripts/license_checker.py --strict   # exit 1 on restrictive/unknown
"""

import argparse
import json
import os
import sys

from scripts.safe_yaml import safe_load_yaml

# Optional — gracefully degrade if not installed
try:
    from huggingface_hub import dataset_info as hf_dataset_info

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASETS_PATH = os.path.join(ROOT, "data", "datasets.yaml")
DEFAULT_OUTPUT_PATH = os.path.join(ROOT, "data", "license_report.json")

# ---------------------------------------------------------------------------
# License classification
# ---------------------------------------------------------------------------

PERMISSIVE_LICENSES = frozenset({
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-2.0",
    "cc0-1.0",
    "unlicense",
    "openrail",
    "openrail++",
    "bigscience-openrail-m",
    "bigscience-bloom-rail-1.0",
    "creativeml-openrail-m",
    "odc-by",
    "pddl",
    "cdla-permissive-1.0",
    "cdla-permissive-2.0",
    "wtfpl",
    "artistic-2.0",
    "zlib",
    "ecl-2.0",
})

RESTRICTIVE_LICENSES = frozenset({
    "cc-by-nc-4.0",
    "cc-by-nc-3.0",
    "cc-by-nc-2.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nc-sa-3.0",
    "cc-by-nc-sa-2.0",
    "cc-by-nc-nd-4.0",
    "cc-by-nc-nd-3.0",
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "cc-by-nd-4.0",
    "cc-by-nd-3.0",
    "gpl-2.0",
    "gpl-3.0",
    "agpl-3.0",
    "lgpl-2.1",
    "lgpl-3.0",
    "odc-odbl",
    "c-uda",
})


def classify_license(license_id):
    """Classify a license string as 'permissive', 'restrictive', or 'unknown'.

    Args:
        license_id: SPDX-style license identifier (case-insensitive), or None.

    Returns:
        One of: 'permissive', 'restrictive', 'unknown'.
    """
    if not license_id:
        return "unknown"

    normalised = license_id.strip().lower()

    if normalised in PERMISSIVE_LICENSES:
        return "permissive"
    if normalised in RESTRICTIVE_LICENSES:
        return "restrictive"
    return "unknown"


def has_nc_conflict(license_id):
    """Return True if the license contains a non-commercial (NC) restriction.

    NC-licensed data cannot be used in a commercially-licensed project.
    """
    if not license_id:
        return False
    normalised = license_id.strip().lower()
    return "-nc" in normalised


def has_sa_conflict(license_id):
    """Return True if the license contains a share-alike (SA) restriction.

    SA-licensed data may require derivative works to use the same license.
    """
    if not license_id:
        return False
    normalised = license_id.strip().lower()
    # Match CC-BY-SA but not CC-BY-NC-SA (NC is the primary concern there).
    return "-sa" in normalised


def has_nd_conflict(license_id):
    """Return True if the license contains a no-derivatives (ND) restriction."""
    if not license_id:
        return False
    normalised = license_id.strip().lower()
    return "-nd" in normalised


def detect_conflicts(license_id):
    """Return a list of conflict strings for a given license.

    Possible conflicts:
    - 'non-commercial': NC license in a potentially commercial project
    - 'share-alike': SA license may force project license change
    - 'no-derivatives': ND license may prohibit model training
    """
    conflicts = []
    if has_nc_conflict(license_id):
        conflicts.append("non-commercial")
    if has_sa_conflict(license_id):
        conflicts.append("share-alike")
    if has_nd_conflict(license_id):
        conflicts.append("no-derivatives")
    return conflicts


# ---------------------------------------------------------------------------
# HuggingFace Hub API
# ---------------------------------------------------------------------------

def fetch_license_from_hub(repo_id):
    """Fetch the license field for a HuggingFace dataset repo.

    Returns the license string, or None if unavailable.
    """
    if not HF_HUB_AVAILABLE:
        return None
    try:
        info = hf_dataset_info(repo_id)
        # The cardData field may contain the license; the top-level
        # attribute is set by HF Hub from the dataset card metadata.
        license_id = getattr(info, "license", None)
        if license_id:
            return license_id
        # Fallback: check card_data / tags
        tags = getattr(info, "tags", []) or []
        for tag in tags:
            if tag.startswith("license:"):
                return tag.split(":", 1)[1]
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Registry parsing
# ---------------------------------------------------------------------------

def parse_datasets(registry_path):
    """Parse datasets.yaml and return a list of (name, repo_id, cfg) for HF datasets.

    Non-HuggingFace sources (e.g. github_csv) are skipped.
    """
    registry = safe_load_yaml(registry_path)
    sources = registry.get("sources", {})

    datasets = []
    for name, cfg in sources.items():
        if cfg.get("type") != "huggingface":
            continue
        repo = cfg.get("repo")
        if repo:
            datasets.append((name, repo, cfg))
    return datasets


def parse_hf_registry():
    """Parse the HFDatasetSpec registry and return (name, hf_id, spec) tuples.

    Returns an empty list if the registry module is not importable.
    """
    try:
        from scripts.data.hf_dataset_registry import get_registry
    except ImportError:
        return []

    datasets = []
    for spec in get_registry():
        datasets.append((spec.hf_id.replace("/", "_"), spec.hf_id, spec))
    return datasets


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(registry_path, cached_licenses=None, include_hf_registry=True):
    """Build the full license compliance report.

    Args:
        registry_path: Path to datasets.yaml.
        cached_licenses: Optional dict mapping repo_id -> license string,
            used as fallback when the HF API is unreachable.
        include_hf_registry: If True, also include datasets from the
            HFDatasetSpec registry (scripts/data/hf_dataset_registry.py).

    Returns:
        A list of dicts, one per HuggingFace dataset, with keys:
            name, repo, license, classification, conflicts, compliant
    """
    if cached_licenses is None:
        cached_licenses = {}

    datasets = parse_datasets(registry_path)

    # Collect repo IDs already seen to avoid duplicates
    seen_repos = {repo for _, repo, _ in datasets}

    # Optionally merge in the HF dataset registry
    if include_hf_registry:
        for name, hf_id, spec in parse_hf_registry():
            if hf_id not in seen_repos:
                seen_repos.add(hf_id)
                # Build a pseudo-cfg with license from the spec
                pseudo_cfg = {"license": getattr(spec, "license", None)}
                datasets.append((name, hf_id, pseudo_cfg))

    report = []

    for name, repo, cfg in datasets:
        # Try live API first, then fall back to cache, then to YAML/spec field
        license_id = fetch_license_from_hub(repo)
        source = "hub"
        if license_id is None:
            license_id = cached_licenses.get(repo)
            source = "cache" if license_id else "unavailable"
        # Also check the cfg-level license as last fallback
        if license_id is None:
            cfg_license = cfg.get("license") if isinstance(cfg, dict) else getattr(cfg, "license", None)
            if cfg_license:
                license_id = cfg_license
                source = "cache"

        classification = classify_license(license_id)
        conflicts = detect_conflicts(license_id)
        compliant = classification == "permissive" and len(conflicts) == 0

        report.append({
            "name": name,
            "repo": repo,
            "license": license_id,
            "license_source": source,
            "classification": classification,
            "conflicts": conflicts,
            "compliant": compliant,
        })

    return report


def save_report(report, output_path):
    """Write the report as pretty-printed JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def print_report(report):
    """Print a human-readable summary to stdout."""
    print("\n{:<30s} {:<20s} {:<14s} {:<10s} {}".format(
        "DATASET", "LICENSE", "CLASS", "COMPLIANT", "CONFLICTS",
    ))
    print("-" * 90)

    for entry in report:
        conflicts_str = ", ".join(entry["conflicts"]) if entry["conflicts"] else "-"
        print("{:<30s} {:<20s} {:<14s} {:<10s} {}".format(
            entry["name"],
            entry["license"] or "(unknown)",
            entry["classification"],
            "yes" if entry["compliant"] else "NO",
            conflicts_str,
        ))

    # Summary
    total = len(report)
    compliant_count = sum(1 for e in report if e["compliant"])
    restrictive_count = sum(1 for e in report if e["classification"] == "restrictive")
    unknown_count = sum(1 for e in report if e["classification"] == "unknown")

    print("\n--- Summary ---")
    print("Total HF datasets: {}".format(total))
    print("Compliant:         {}".format(compliant_count))
    print("Restrictive:       {}".format(restrictive_count))
    print("Unknown:           {}".format(unknown_count))


def load_cached_licenses(registry_path):
    """Extract any license hints already present in datasets.yaml.

    If a source entry has a 'license' field, use it as a cached value.
    Returns a dict mapping repo_id -> license string.
    """
    registry = safe_load_yaml(registry_path)
    sources = registry.get("sources", {})
    cache = {}
    for _name, cfg in sources.items():
        if cfg.get("type") != "huggingface":
            continue
        repo = cfg.get("repo")
        license_id = cfg.get("license")
        if repo and license_id:
            cache[repo] = license_id
    return cache


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Check license compliance for HuggingFace datasets.",
    )
    parser.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS_PATH,
        help="Path to datasets.yaml registry (default: data/datasets.yaml).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path for JSON report output (default: data/license_report.json).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any restrictive or unknown licenses are found.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    print("Loading dataset registry from {} ...".format(args.datasets))

    # Load cached licenses from the registry itself (offline fallback)
    cached_licenses = load_cached_licenses(args.datasets)

    report = generate_report(args.datasets, cached_licenses=cached_licenses)

    if not report:
        print("No HuggingFace datasets found in registry.")
        return 0

    print_report(report)
    save_report(report, args.output)
    print("\nReport saved to {}".format(args.output))

    if args.strict:
        non_compliant = [
            e for e in report
            if e["classification"] in ("restrictive", "unknown")
        ]
        if non_compliant:
            print(
                "\nSTRICT MODE: {} dataset(s) with restrictive/unknown licenses.".format(
                    len(non_compliant)
                ),
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
