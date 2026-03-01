#!/usr/bin/env python3
"""Per-technique-tag performance analysis for Na0S prompt injection detector.

Loads the malicious holdout dataset and adversarial evasion dataset, runs
scan() on each sample, and computes per-technique-category and per-evasion-type
recall tables.

Usage
-----
    python scripts/technique_analysis.py
    python scripts/technique_analysis.py --max-samples 20
    python scripts/technique_analysis.py --threshold 0.50 --max-samples 50
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure Na0S package is importable when running from the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------
_DEFAULT_MALICIOUS_PATH = str(_REPO_ROOT / "data" / "holdout" / "malicious_holdout.jsonl")
_DEFAULT_EVASION_PATH = str(_REPO_ROOT / "data" / "benchmark" / "adversarial_evasion.jsonl")
_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "benchmarks" / "results")
_DEFAULT_OUTPUT_PATH = os.path.join(_DEFAULT_OUTPUT_DIR, "technique_analysis.json")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_malicious_holdout(path, max_samples=None):
    """Load malicious holdout JSONL and return list of dicts with text+category."""
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping line {lineno} (bad JSON): {exc}",
                      file=sys.stderr)
                continue
            samples.append({
                "text": obj.get("text", ""),
                "category": obj.get("category", "unknown"),
                "label": int(obj.get("label", 1)),
            })
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def load_evasion_dataset(path, max_samples=None):
    """Load adversarial evasion JSONL and return list of dicts with text+evasion_type."""
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping line {lineno} (bad JSON): {exc}",
                      file=sys.stderr)
                continue
            samples.append({
                "text": obj.get("text", ""),
                "evasion_type": obj.get("evasion_type", "unknown"),
                "label": int(obj.get("label", 1)),
            })
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# Per-category analysis
# ---------------------------------------------------------------------------

def analyze_by_category(samples, scan_fn, threshold):
    """Run scan on each sample and aggregate results by category.

    Parameters
    ----------
    samples : list[dict]
        Each dict has keys: text, category, label.
    scan_fn : callable
        A function(text, threshold) -> ScanResult.
    threshold : float
        Decision threshold for scan.

    Returns
    -------
    dict
        Keys are category codes (e.g. "D1", "D3"); values are dicts with
        total, detected, missed, recall, technique_tags_seen, avg_latency_ms,
        total_time_ms.
    """
    category_data = defaultdict(lambda: {
        "total": 0,
        "detected": 0,
        "missed": 0,
        "technique_tags_seen": defaultdict(int),
        "latencies_ms": [],
        "missed_texts": [],
    })

    for i, sample in enumerate(samples):
        text = sample["text"]
        category = sample["category"]
        entry = category_data[category]
        entry["total"] += 1

        t0 = time.perf_counter()
        result = scan_fn(text, threshold=threshold)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        entry["latencies_ms"].append(elapsed_ms)

        if result.is_malicious:
            entry["detected"] += 1
        else:
            entry["missed"] += 1
            entry["missed_texts"].append(text[:100])

        for tag in result.technique_tags:
            entry["technique_tags_seen"][tag] += 1

        # Progress
        status = "DET" if result.is_malicious else "MISS"
        print(f"  [{i+1:>4}/{len(samples)}] cat={category:>4} {status} "
              f"score={result.risk_score:.4f} "
              f"latency={elapsed_ms:.1f}ms "
              f"tags={result.technique_tags}")

    # Build final results
    results = {}
    for category in sorted(category_data.keys()):
        d = category_data[category]
        recall = d["detected"] / d["total"] if d["total"] > 0 else 0.0
        avg_lat = sum(d["latencies_ms"]) / len(d["latencies_ms"]) if d["latencies_ms"] else 0.0
        total_time = sum(d["latencies_ms"])
        results[category] = {
            "total": d["total"],
            "detected": d["detected"],
            "missed": d["missed"],
            "recall": round(recall, 4),
            "technique_tags_seen": dict(d["technique_tags_seen"]),
            "avg_latency_ms": round(avg_lat, 2),
            "total_time_ms": round(total_time, 2),
            "missed_samples_preview": d["missed_texts"][:5],
        }
    return results


def analyze_by_evasion_type(samples, scan_fn, threshold):
    """Run scan on each sample and aggregate results by evasion_type.

    Parameters
    ----------
    samples : list[dict]
        Each dict has keys: text, evasion_type, label.
    scan_fn : callable
        A function(text, threshold) -> ScanResult.
    threshold : float
        Decision threshold for scan.

    Returns
    -------
    dict
        Keys are evasion types (e.g. "base64", "rot13"); values are dicts with
        total, detected, missed, detection_rate, technique_tags_seen,
        avg_latency_ms, total_time_ms.
    """
    evasion_data = defaultdict(lambda: {
        "total": 0,
        "detected": 0,
        "missed": 0,
        "technique_tags_seen": defaultdict(int),
        "latencies_ms": [],
        "missed_texts": [],
    })

    for i, sample in enumerate(samples):
        text = sample["text"]
        etype = sample["evasion_type"]
        entry = evasion_data[etype]
        entry["total"] += 1

        t0 = time.perf_counter()
        result = scan_fn(text, threshold=threshold)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        entry["latencies_ms"].append(elapsed_ms)

        if result.is_malicious:
            entry["detected"] += 1
        else:
            entry["missed"] += 1
            entry["missed_texts"].append(text[:100])

        for tag in result.technique_tags:
            entry["technique_tags_seen"][tag] += 1

        # Progress
        status = "DET" if result.is_malicious else "MISS"
        print(f"  [{i+1:>4}/{len(samples)}] evasion={etype:<20} {status} "
              f"score={result.risk_score:.4f} "
              f"latency={elapsed_ms:.1f}ms")

    # Build final results
    results = {}
    for etype in sorted(evasion_data.keys()):
        d = evasion_data[etype]
        rate = d["detected"] / d["total"] if d["total"] > 0 else 0.0
        avg_lat = sum(d["latencies_ms"]) / len(d["latencies_ms"]) if d["latencies_ms"] else 0.0
        total_time = sum(d["latencies_ms"])
        results[etype] = {
            "total": d["total"],
            "detected": d["detected"],
            "missed": d["missed"],
            "detection_rate": round(rate, 4),
            "technique_tags_seen": dict(d["technique_tags_seen"]),
            "avg_latency_ms": round(avg_lat, 2),
            "total_time_ms": round(total_time, 2),
            "missed_samples_preview": d["missed_texts"][:5],
        }
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_category_table(results):
    """Print a markdown-formatted per-technique recall table."""
    print()
    print("## Per-Technique Category Recall (Malicious Holdout)")
    print()
    print("| Category | Total | Detected | Missed | Recall | Avg Latency (ms) | Top Tags |")
    print("|----------|-------|----------|--------|--------|------------------|----------|")

    total_all = 0
    detected_all = 0
    for cat in sorted(results.keys()):
        r = results[cat]
        total_all += r["total"]
        detected_all += r["detected"]
        # Top 3 most common technique tags
        tags_sorted = sorted(r["technique_tags_seen"].items(),
                             key=lambda x: x[1], reverse=True)
        top_tags = ", ".join(f"{t}({c})" for t, c in tags_sorted[:3])
        print(f"| {cat:<8} | {r['total']:>5} | {r['detected']:>8} | {r['missed']:>6} "
              f"| {r['recall']:>6.2%} | {r['avg_latency_ms']:>16.1f} | {top_tags} |")

    overall = detected_all / total_all if total_all > 0 else 0.0
    print(f"| {'TOTAL':<8} | {total_all:>5} | {detected_all:>8} | "
          f"{total_all - detected_all:>6} | {overall:>6.2%} | {'':>16} | |")
    print()


def print_evasion_table(results):
    """Print a markdown-formatted per-evasion-type detection rate table."""
    print()
    print("## Per-Evasion-Type Detection Rate (Adversarial Evasion)")
    print()
    print("| Evasion Type         | Total | Detected | Missed | Det. Rate | Avg Latency (ms) |")
    print("|----------------------|-------|----------|--------|-----------|------------------|")

    total_all = 0
    detected_all = 0
    for etype in sorted(results.keys()):
        r = results[etype]
        total_all += r["total"]
        detected_all += r["detected"]
        print(f"| {etype:<20} | {r['total']:>5} | {r['detected']:>8} | {r['missed']:>6} "
              f"| {r['detection_rate']:>9.2%} | {r['avg_latency_ms']:>16.1f} |")

    overall = detected_all / total_all if total_all > 0 else 0.0
    print(f"| {'TOTAL':<20} | {total_all:>5} | {detected_all:>8} | "
          f"{total_all - detected_all:>6} | {overall:>9.2%} | {'':>16} |")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-technique-tag performance analysis for Na0S.",
    )
    parser.add_argument(
        "--malicious-dataset", default=_DEFAULT_MALICIOUS_PATH,
        help=f"Path to malicious holdout JSONL (default: {_DEFAULT_MALICIOUS_PATH}).",
    )
    parser.add_argument(
        "--evasion-dataset", default=_DEFAULT_EVASION_PATH,
        help=f"Path to adversarial evasion JSONL (default: {_DEFAULT_EVASION_PATH}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="Decision threshold for scan (default: 0.55).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples per dataset (for quick testing).",
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {_DEFAULT_OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    # Validate dataset paths
    for path, label in [(args.malicious_dataset, "malicious holdout"),
                        (args.evasion_dataset, "adversarial evasion")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} dataset not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Deferred import so --help is fast
    from na0s.predict import scan

    # -----------------------------------------------------------------------
    # Part 1: Malicious holdout — per-technique category recall
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("Part 1: Malicious Holdout — Per-Technique Category Recall")
    print("=" * 70)

    mal_samples = load_malicious_holdout(args.malicious_dataset,
                                         max_samples=args.max_samples)
    print(f"Loaded {len(mal_samples)} malicious holdout samples.")

    t0_mal = time.perf_counter()
    category_results = analyze_by_category(mal_samples, scan, args.threshold)
    mal_wall = time.perf_counter() - t0_mal

    print_category_table(category_results)
    print(f"Wall-clock time (malicious holdout): {mal_wall:.2f}s")

    # -----------------------------------------------------------------------
    # Part 2: Adversarial evasion — per-evasion-type detection rate
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Part 2: Adversarial Evasion — Per-Evasion-Type Detection Rate")
    print("=" * 70)

    ev_samples = load_evasion_dataset(args.evasion_dataset,
                                       max_samples=args.max_samples)
    print(f"Loaded {len(ev_samples)} adversarial evasion samples.")

    t0_ev = time.perf_counter()
    evasion_results = analyze_by_evasion_type(ev_samples, scan, args.threshold)
    ev_wall = time.perf_counter() - t0_ev

    print_evasion_table(evasion_results)
    print(f"Wall-clock time (adversarial evasion): {ev_wall:.2f}s")

    # -----------------------------------------------------------------------
    # Compute overall summary
    # -----------------------------------------------------------------------
    mal_total = sum(r["total"] for r in category_results.values())
    mal_detected = sum(r["detected"] for r in category_results.values())
    ev_total = sum(r["total"] for r in evasion_results.values())
    ev_detected = sum(r["detected"] for r in evasion_results.values())

    summary = {
        "overall_malicious_recall": round(mal_detected / mal_total, 4) if mal_total > 0 else 0.0,
        "overall_evasion_detection_rate": round(ev_detected / ev_total, 4) if ev_total > 0 else 0.0,
        "malicious_total": mal_total,
        "malicious_detected": mal_detected,
        "evasion_total": ev_total,
        "evasion_detected": ev_detected,
        "threshold": args.threshold,
        "wall_time_malicious_s": round(mal_wall, 2),
        "wall_time_evasion_s": round(ev_wall, 2),
    }

    # -----------------------------------------------------------------------
    # Write JSON output
    # -----------------------------------------------------------------------
    try:
        from na0s._version import __version__ as na0s_version
    except ImportError:
        na0s_version = "unknown"

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": na0s_version,
        "summary": summary,
        "per_category": category_results,
        "per_evasion_type": evasion_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(f"\nResults written to: {args.output}")

    # Print overall summary
    print()
    print("=" * 70)
    print("Overall Summary")
    print("=" * 70)
    print(f"  Malicious holdout recall:      {summary['overall_malicious_recall']:.2%} "
          f"({mal_detected}/{mal_total})")
    print(f"  Evasion detection rate:         {summary['overall_evasion_detection_rate']:.2%} "
          f"({ev_detected}/{ev_total})")
    print(f"  Threshold:                      {args.threshold}")
    print()


if __name__ == "__main__":
    main()
