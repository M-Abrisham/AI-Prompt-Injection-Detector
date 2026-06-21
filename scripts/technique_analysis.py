#!/usr/bin/env python3
"""Per-technique performance analysis for Na0S — the single source of truth.

Loads the malicious holdout, the benign (safe) holdout, and the adversarial
evasion dataset; runs scan() on each sample; and computes a TWO-SIDED report:

  * per-technique-category **recall** (malicious holdout)
  * per-evasion-type **detection rate** (adversarial evasion)
  * per-category benign **false-positive rate** (safe holdout)
  * aggregate **precision / F1** and Wilson 95% confidence intervals on every rate

A recall-only harness is misleading: a detector that flags everything scores
100% recall. This harness measures recall AND benign FPR together, attaches a
confidence interval (Wilson score — valid for the small per-technique n and for
proportions near 0/1) to every rate, and can act as a CI gate that fails on the
*worst* slice using the CI bound (not the noisy point estimate).

Usage
-----
    python scripts/technique_analysis.py
    python scripts/technique_analysis.py --max-samples 50
    python scripts/technique_analysis.py --gate --recall-floor 0.50 --fpr-ceiling 0.10

If the datasets are missing (they are gitignored), they are regenerated
deterministically (seed=42) before the run.
"""

import argparse
import json
import math
import os
import subprocess
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
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------
_DEFAULT_MALICIOUS_PATH = str(_REPO_ROOT / "data" / "holdout" / "malicious_holdout.jsonl")
_DEFAULT_SAFE_PATH = str(_REPO_ROOT / "data" / "holdout" / "safe_holdout.jsonl")
_DEFAULT_EVASION_PATH = str(_REPO_ROOT / "data" / "benchmark" / "adversarial_evasion.jsonl")
_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "benchmarks" / "results")
_DEFAULT_OUTPUT_PATH = os.path.join(_DEFAULT_OUTPUT_DIR, "technique_analysis.json")

# Schema version for the emitted artifact. Bump on breaking field changes.
SCHEMA_VERSION = 2

# Gate defaults. A slice with fewer than _DEFAULT_MIN_SLICE samples is reported
# but never gated (too few samples for a reliable CI bound).
_DEFAULT_RECALL_FLOOR = 0.50
_DEFAULT_FPR_CEILING = 0.10
_DEFAULT_MIN_SLICE = 5


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for a binomial proportion k/n.

    Preferred over the normal-approximation (Wald) interval because it stays
    valid for small n and for proportions near 0 or 1 — exactly the regime of
    per-technique slices (e.g. 20 samples at ~100% recall). Returns (lo, hi)
    clamped to [0, 1]. For n == 0 returns (0.0, 0.0).
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (round(lo, 4), round(hi, 4))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_jsonl(path, fields, max_samples=None):
    """Load a JSONL file into a list of dicts projecting the requested fields.

    ``fields`` maps output-key -> (json-key, default). Lines that are blank or
    fail to parse are skipped with a warning.
    """
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
                out_key: obj.get(json_key, default)
                for out_key, (json_key, default) in fields.items()
            })
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def load_malicious_holdout(path, max_samples=None):
    """Load malicious holdout JSONL -> list of {text, category, label} dicts."""
    samples = _load_jsonl(path, {
        "text": ("text", ""),
        "category": ("category", "unknown"),
        "label": ("label", 1),
    }, max_samples=max_samples)
    for s in samples:
        s["label"] = int(s["label"])
    return samples


def load_safe_holdout(path, max_samples=None):
    """Load benign/safe holdout JSONL -> list of {text, category, label} dicts."""
    samples = _load_jsonl(path, {
        "text": ("text", ""),
        "category": ("category", "unknown"),
        "label": ("label", 0),
    }, max_samples=max_samples)
    for s in samples:
        s["label"] = int(s["label"])
    return samples


def load_evasion_dataset(path, max_samples=None):
    """Load adversarial evasion JSONL -> list of {text, evasion_type, label} dicts."""
    samples = _load_jsonl(path, {
        "text": ("text", ""),
        "evasion_type": ("evasion_type", "unknown"),
        "label": ("label", 1),
    }, max_samples=max_samples)
    for s in samples:
        s["label"] = int(s["label"])
    return samples


# ---------------------------------------------------------------------------
# Reproducibility — regenerate datasets if missing (they are gitignored).
# ---------------------------------------------------------------------------

def ensure_datasets(malicious_path, safe_path, evasion_path):
    """Regenerate any missing default dataset so the harness always runs.

    Holdout (malicious + safe) is regenerated together via gen_all_datasets
    (the only generator that includes D6); the canonical 9-type evasion file is
    regenerated via generate_adversarial.py. Custom paths that are missing are
    left to the caller's validation (clear error in main()).
    """
    # Malicious holdout — gen_all_datasets is the only generator that includes D6.
    if malicious_path == _DEFAULT_MALICIOUS_PATH and not os.path.isfile(malicious_path):
        print("Regenerating missing malicious holdout...", file=sys.stderr)
        import gen_all_datasets as gad
        with open(_DEFAULT_MALICIOUS_PATH, "w", encoding="utf-8") as fh:
            for s in gad.generate_malicious_holdout():
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Safe holdout — generate_safe_holdout.py is canonical (500+ samples, the
    # set tests/test_holdout_safe.py validates), NOT gen_all_datasets' 100-row
    # version. Regenerate via the canonical generator to keep the FPR set stable.
    if safe_path == _DEFAULT_SAFE_PATH and not os.path.isfile(safe_path):
        print("Regenerating missing safe holdout...", file=sys.stderr)
        subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "generate_safe_holdout.py")],
            check=True,
        )

    if evasion_path == _DEFAULT_EVASION_PATH and not os.path.isfile(evasion_path):
        print("Regenerating adversarial evasion dataset (missing on disk)...", file=sys.stderr)
        subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "generate_adversarial.py")],
            check=True,
        )


# ---------------------------------------------------------------------------
# Per-slice analysis
# ---------------------------------------------------------------------------

def _new_bucket():
    return {
        "total": 0,
        "detected": 0,
        "missed": 0,
        "technique_tags_seen": defaultdict(int),
        "latencies_ms": [],
        "missed_texts": [],
    }


def analyze_by_category(samples, scan_fn, threshold):
    """Run scan on each malicious sample and aggregate recall by category.

    Returns a dict keyed by category code (e.g. "D4") whose values carry
    total, detected, missed, recall, recall_ci (Wilson 95%), n,
    technique_tags_seen, latency, and a preview of missed samples.
    """
    category_data = defaultdict(_new_bucket)

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

        status = "DET" if result.is_malicious else "MISS"
        print(f"  [{i+1:>4}/{len(samples)}] cat={category:>5} {status} "
              f"score={result.risk_score:.4f} latency={elapsed_ms:.1f}ms "
              f"tags={result.technique_tags}")

    results = {}
    for category in sorted(category_data.keys()):
        d = category_data[category]
        recall = d["detected"] / d["total"] if d["total"] > 0 else 0.0
        avg_lat = sum(d["latencies_ms"]) / len(d["latencies_ms"]) if d["latencies_ms"] else 0.0
        results[category] = {
            "total": d["total"],
            "n": d["total"],
            "detected": d["detected"],
            "missed": d["missed"],
            "recall": round(recall, 4),
            "recall_ci": list(wilson_ci(d["detected"], d["total"])),
            "technique_tags_seen": dict(d["technique_tags_seen"]),
            "avg_latency_ms": round(avg_lat, 2),
            "total_time_ms": round(sum(d["latencies_ms"]), 2),
            "missed_samples_preview": d["missed_texts"][:5],
        }
    return results


def analyze_by_evasion_type(samples, scan_fn, threshold):
    """Run scan on each evasion sample and aggregate detection rate by type.

    Returns a dict keyed by evasion type (e.g. "hex_encoding") whose values
    carry total, detected, missed, detection_rate, detection_rate_ci, n,
    technique_tags_seen, latency, and a preview of missed samples.
    """
    evasion_data = defaultdict(_new_bucket)

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

        status = "DET" if result.is_malicious else "MISS"
        print(f"  [{i+1:>4}/{len(samples)}] evasion={etype:<20} {status} "
              f"score={result.risk_score:.4f} latency={elapsed_ms:.1f}ms")

    results = {}
    for etype in sorted(evasion_data.keys()):
        d = evasion_data[etype]
        rate = d["detected"] / d["total"] if d["total"] > 0 else 0.0
        avg_lat = sum(d["latencies_ms"]) / len(d["latencies_ms"]) if d["latencies_ms"] else 0.0
        results[etype] = {
            "total": d["total"],
            "n": d["total"],
            "detected": d["detected"],
            "missed": d["missed"],
            "detection_rate": round(rate, 4),
            "detection_rate_ci": list(wilson_ci(d["detected"], d["total"])),
            "technique_tags_seen": dict(d["technique_tags_seen"]),
            "avg_latency_ms": round(avg_lat, 2),
            "total_time_ms": round(sum(d["latencies_ms"]), 2),
            "missed_samples_preview": d["missed_texts"][:5],
        }
    return results


def analyze_benign(samples, scan_fn, threshold):
    """Run scan on each benign sample and aggregate false positives by category.

    A "false positive" is a benign sample flagged malicious. Returns a dict
    keyed by benign category (e.g. "S1") with total, false_positives,
    true_negatives, false_positive_rate, fpr_ci (Wilson 95%), n, latency, and a
    preview of the flagged (false-positive) samples.
    """
    benign_data = defaultdict(lambda: {
        "total": 0,
        "false_positives": 0,
        "latencies_ms": [],
        "fp_texts": [],
    })

    for i, sample in enumerate(samples):
        text = sample["text"]
        category = sample["category"]
        entry = benign_data[category]
        entry["total"] += 1

        t0 = time.perf_counter()
        result = scan_fn(text, threshold=threshold)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        entry["latencies_ms"].append(elapsed_ms)

        if result.is_malicious:
            entry["false_positives"] += 1
            entry["fp_texts"].append(text[:100])

        status = "FP!" if result.is_malicious else "ok"
        print(f"  [{i+1:>4}/{len(samples)}] benign={category:>5} {status} "
              f"score={result.risk_score:.4f} latency={elapsed_ms:.1f}ms")

    results = {}
    for category in sorted(benign_data.keys()):
        d = benign_data[category]
        fp = d["false_positives"]
        total = d["total"]
        fpr = fp / total if total > 0 else 0.0
        avg_lat = sum(d["latencies_ms"]) / len(d["latencies_ms"]) if d["latencies_ms"] else 0.0
        results[category] = {
            "total": total,
            "n": total,
            "false_positives": fp,
            "true_negatives": total - fp,
            "false_positive_rate": round(fpr, 4),
            "fpr_ci": list(wilson_ci(fp, total)),
            "avg_latency_ms": round(avg_lat, 2),
            "total_time_ms": round(sum(d["latencies_ms"]), 2),
            "false_positive_preview": d["fp_texts"][:5],
        }
    return results


# ---------------------------------------------------------------------------
# CI gate
# ---------------------------------------------------------------------------

def evaluate_gate(category_results, benign_results, recall_floor,
                  fpr_ceiling, min_slice):
    """Two-sided pass/fail gate using CI bounds.

    Recall is gated **per malicious category** (the worst slice fails the
    build) because recall holes are technique-specific: a category fails if the
    *lower* bound of its recall CI is below ``recall_floor``. Categories with
    n < ``min_slice`` are skipped (too few samples for a reliable bound).

    False positives are gated on the **pooled** benign set, not per benign
    category: a benign category typically has too few samples for a tight CI
    (0/25 still has a ~13% Wilson upper bound), so per-category gating would
    fail a perfectly clean detector. Pooling (n≈100) yields a meaningful bound
    and matches the "merge tiny slices" guidance. The build fails if the upper
    bound of the pooled-FPR CI exceeds ``fpr_ceiling``.

    Gating on the CI bound (not the point estimate) prevents a single-sample
    flip on a small slice from flaking CI.
    """
    failures = []
    skipped = []

    for cat in sorted(category_results.keys()):
        r = category_results[cat]
        if r["n"] < min_slice:
            skipped.append({"slice": cat, "kind": "recall", "n": r["n"]})
            continue
        recall_lo = r["recall_ci"][0]
        if recall_lo < recall_floor:
            failures.append({
                "slice": cat, "kind": "recall", "n": r["n"],
                "recall": r["recall"], "recall_ci_low": recall_lo,
                "floor": recall_floor,
            })

    fp = sum(r["false_positives"] for r in benign_results.values())
    benign_n = sum(r["total"] for r in benign_results.values())
    fpr_ci = wilson_ci(fp, benign_n)
    if benign_n >= min_slice and fpr_ci[1] > fpr_ceiling:
        failures.append({
            "slice": "OVERALL_BENIGN", "kind": "fpr", "n": benign_n,
            "fpr": round(fp / benign_n, 4) if benign_n else 0.0,
            "fpr_ci_high": fpr_ci[1], "ceiling": fpr_ceiling,
        })

    # Fail CLOSED on missing coverage: a gate that passes because it evaluated
    # nothing (empty/all-tiny datasets, e.g. --max-samples 2) is worse than no
    # gate — it green-lights a broken run. Require that at least one malicious
    # category and the pooled benign set were actually assessable.
    evaluated_recall_n = sum(r["n"] for c, r in category_results.items()
                             if r["n"] >= min_slice)
    if evaluated_recall_n == 0:
        failures.append({
            "slice": "COVERAGE", "kind": "coverage", "n": evaluated_recall_n,
            "detail": "no malicious category had n >= min_slice; recall not assessable",
        })
    if benign_n < min_slice:
        failures.append({
            "slice": "COVERAGE", "kind": "coverage", "n": benign_n,
            "detail": "benign pooled n < min_slice; FPR not assessable",
        })

    return {
        "passed": not failures,
        "recall_floor": recall_floor,
        "fpr_ceiling": fpr_ceiling,
        "min_slice": min_slice,
        "pooled_benign_fpr_ci": list(fpr_ci),
        "evaluated_recall_n": evaluated_recall_n,
        "benign_n": benign_n,
        "failures": failures,
        "skipped_small_slices": skipped,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_category_table(results):
    """Print a markdown per-technique recall table (with 95% CI)."""
    print()
    print("## Per-Technique Category Recall (Malicious Holdout)")
    print()
    print("| Category | n | Detected | Missed | Recall | 95% CI | Top Tags |")
    print("|----------|---|----------|--------|--------|--------|----------|")

    total_all = detected_all = 0
    for cat in sorted(results.keys()):
        r = results[cat]
        total_all += r["total"]
        detected_all += r["detected"]
        tags_sorted = sorted(r["technique_tags_seen"].items(),
                             key=lambda x: x[1], reverse=True)
        top_tags = ", ".join(f"{t}({c})" for t, c in tags_sorted[:3])
        ci = r.get("recall_ci", [0.0, 0.0])
        print(f"| {cat:<8} | {r['total']:>3} | {r['detected']:>8} | {r['missed']:>6} "
              f"| {r['recall']:>6.2%} | [{ci[0]:.2f}, {ci[1]:.2f}] | {top_tags} |")

    overall = detected_all / total_all if total_all > 0 else 0.0
    print(f"| {'TOTAL':<8} | {total_all:>3} | {detected_all:>8} | "
          f"{total_all - detected_all:>6} | {overall:>6.2%} | | |")
    print()


def print_evasion_table(results):
    """Print a markdown per-evasion-type detection rate table (with 95% CI)."""
    print()
    print("## Per-Evasion-Type Detection Rate (Adversarial Evasion)")
    print()
    print("| Evasion Type         | n | Detected | Missed | Det. Rate | 95% CI |")
    print("|----------------------|---|----------|--------|-----------|--------|")

    total_all = detected_all = 0
    for etype in sorted(results.keys()):
        r = results[etype]
        total_all += r["total"]
        detected_all += r["detected"]
        ci = r.get("detection_rate_ci", [0.0, 0.0])
        print(f"| {etype:<20} | {r['total']:>3} | {r['detected']:>8} | {r['missed']:>6} "
              f"| {r['detection_rate']:>9.2%} | [{ci[0]:.2f}, {ci[1]:.2f}] |")

    overall = detected_all / total_all if total_all > 0 else 0.0
    print(f"| {'TOTAL':<20} | {total_all:>3} | {detected_all:>8} | "
          f"{total_all - detected_all:>6} | {overall:>9.2%} | |")
    print()


def print_benign_table(results):
    """Print a markdown per-category benign false-positive table (with 95% CI)."""
    print()
    print("## Per-Category Benign False-Positive Rate (Safe Holdout)")
    print()
    print("| Benign Cat | n | False Pos | FPR | 95% CI |")
    print("|------------|---|-----------|-----|--------|")

    total_all = fp_all = 0
    for cat in sorted(results.keys()):
        r = results[cat]
        total_all += r["total"]
        fp_all += r["false_positives"]
        ci = r.get("fpr_ci", [0.0, 0.0])
        print(f"| {cat:<10} | {r['total']:>3} | {r['false_positives']:>9} "
              f"| {r['false_positive_rate']:>5.2%} | [{ci[0]:.2f}, {ci[1]:.2f}] |")

    overall = fp_all / total_all if total_all > 0 else 0.0
    print(f"| {'TOTAL':<10} | {total_all:>3} | {fp_all:>9} | {overall:>5.2%} | |")
    print()


def print_gate(gate):
    """Print the gate pass/fail summary."""
    print()
    print("=" * 70)
    print(f"CI Gate: {'PASS' if gate['passed'] else 'FAIL'}  "
          f"(recall_floor={gate['recall_floor']}, fpr_ceiling={gate['fpr_ceiling']}, "
          f"min_slice={gate['min_slice']})")
    print("=" * 70)
    for f in gate["failures"]:
        if f["kind"] == "recall":
            print(f"  FAIL recall  {f['slice']:<8} n={f['n']:<3} "
                  f"recall={f['recall']:.2%} ci_low={f['recall_ci_low']:.2%} "
                  f"< floor {f['floor']:.2%}")
        elif f["kind"] == "fpr":
            print(f"  FAIL fpr     {f['slice']:<8} n={f['n']:<3} "
                  f"fpr={f['fpr']:.2%} ci_high={f['fpr_ci_high']:.2%} "
                  f"> ceiling {f['ceiling']:.2%}")
        else:  # coverage
            print(f"  FAIL coverage {f['slice']:<8} n={f['n']:<3} {f['detail']}")
    if gate["skipped_small_slices"]:
        small = ", ".join(f"{s['slice']}(n={s['n']})" for s in gate["skipped_small_slices"])
        print(f"  skipped (n < {gate['min_slice']}): {small}")
    print()


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------

def _git_sha():
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-technique two-sided (recall + benign FPR) analysis for Na0S.",
    )
    parser.add_argument("--malicious-dataset", default=_DEFAULT_MALICIOUS_PATH,
                        help="Path to malicious holdout JSONL.")
    parser.add_argument("--safe-dataset", default=_DEFAULT_SAFE_PATH,
                        help="Path to benign/safe holdout JSONL.")
    parser.add_argument("--evasion-dataset", default=_DEFAULT_EVASION_PATH,
                        help="Path to adversarial evasion JSONL.")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Decision threshold for scan (default: 0.55).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per dataset (for quick testing).")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT_PATH,
                        help="Output JSON path.")
    parser.add_argument("--gate", action="store_true",
                        help="Run as a CI gate: exit non-zero if any slice "
                             "breaches the recall floor / FPR ceiling.")
    parser.add_argument("--recall-floor", type=float, default=_DEFAULT_RECALL_FLOOR,
                        help=f"Min recall CI-low per category (default: {_DEFAULT_RECALL_FLOOR}).")
    parser.add_argument("--fpr-ceiling", type=float, default=_DEFAULT_FPR_CEILING,
                        help=f"Max FPR CI-high per benign category (default: {_DEFAULT_FPR_CEILING}).")
    parser.add_argument("--min-slice", type=int, default=_DEFAULT_MIN_SLICE,
                        help=f"Skip gating slices below this n (default: {_DEFAULT_MIN_SLICE}).")
    args = parser.parse_args()

    if args.max_samples is not None:
        print(f"WARNING: --max-samples {args.max_samples} truncates each dataset by "
              f"FILE ORDER; per-slice n is NOT representative and the result is "
              f"unsuitable for gating or for comparison with a full run.",
              file=sys.stderr)

    # Reproducibility: regenerate gitignored default datasets if missing.
    ensure_datasets(args.malicious_dataset, args.safe_dataset, args.evasion_dataset)

    # Validate dataset paths (clear error for custom paths that don't exist).
    for path, label in [(args.malicious_dataset, "malicious holdout"),
                        (args.safe_dataset, "safe holdout"),
                        (args.evasion_dataset, "adversarial evasion")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} dataset not found: {path}", file=sys.stderr)
            sys.exit(1)

    from na0s.predict import scan  # deferred so --help stays fast

    # --- Part 1: malicious holdout — per-technique category recall ---
    print("=" * 70)
    print("Part 1: Malicious Holdout — Per-Technique Category Recall")
    print("=" * 70)
    mal_samples = load_malicious_holdout(args.malicious_dataset, max_samples=args.max_samples)
    print(f"Loaded {len(mal_samples)} malicious holdout samples.")
    if not mal_samples:
        print(f"ERROR: malicious holdout loaded 0 samples: {args.malicious_dataset}",
              file=sys.stderr)
        sys.exit(1)
    t0 = time.perf_counter()
    category_results = analyze_by_category(mal_samples, scan, args.threshold)
    mal_wall = time.perf_counter() - t0
    print_category_table(category_results)
    print(f"Wall-clock time (malicious holdout): {mal_wall:.2f}s")

    # --- Part 2: adversarial evasion — per-evasion-type detection rate ---
    print()
    print("=" * 70)
    print("Part 2: Adversarial Evasion — Per-Evasion-Type Detection Rate")
    print("=" * 70)
    ev_samples = load_evasion_dataset(args.evasion_dataset, max_samples=args.max_samples)
    print(f"Loaded {len(ev_samples)} adversarial evasion samples.")
    if not ev_samples:
        print(f"ERROR: adversarial evasion loaded 0 samples: {args.evasion_dataset}",
              file=sys.stderr)
        sys.exit(1)
    t0 = time.perf_counter()
    evasion_results = analyze_by_evasion_type(ev_samples, scan, args.threshold)
    ev_wall = time.perf_counter() - t0
    print_evasion_table(evasion_results)
    print(f"Wall-clock time (adversarial evasion): {ev_wall:.2f}s")

    # --- Part 3: safe holdout — per-category benign false-positive rate ---
    print()
    print("=" * 70)
    print("Part 3: Safe Holdout — Per-Category Benign False-Positive Rate")
    print("=" * 70)
    safe_samples = load_safe_holdout(args.safe_dataset, max_samples=args.max_samples)
    print(f"Loaded {len(safe_samples)} benign holdout samples.")
    if not safe_samples:
        print(f"ERROR: safe holdout loaded 0 samples: {args.safe_dataset}",
              file=sys.stderr)
        sys.exit(1)
    t0 = time.perf_counter()
    benign_results = analyze_benign(safe_samples, scan, args.threshold)
    benign_wall = time.perf_counter() - t0
    print_benign_table(benign_results)
    print(f"Wall-clock time (safe holdout): {benign_wall:.2f}s")

    # --- Aggregate: the two operationally-valid, independently-sourced rates ---
    # We deliberately do NOT report precision/F1: TP comes from the malicious
    # holdout and FP from a separate benign holdout with no shared real-world
    # prevalence, so a blended precision would be an artifact of the 340:100
    # split, not a meaningful operating point. Recall and benign FPR are each
    # valid on their own pool and are reported with Wilson CIs.
    tp = sum(r["detected"] for r in category_results.values())
    fn = sum(r["missed"] for r in category_results.values())
    mal_total = tp + fn
    fp = sum(r["false_positives"] for r in benign_results.values())
    benign_total = sum(r["total"] for r in benign_results.values())
    ev_total = sum(r["total"] for r in evasion_results.values())
    ev_detected = sum(r["detected"] for r in evasion_results.values())

    recall = tp / mal_total if mal_total > 0 else 0.0
    benign_fpr = fp / benign_total if benign_total > 0 else 0.0

    summary = {
        "overall_malicious_recall": round(recall, 4),
        "overall_malicious_recall_ci": list(wilson_ci(tp, mal_total)),
        "overall_evasion_detection_rate": round(ev_detected / ev_total, 4) if ev_total > 0 else 0.0,
        "overall_benign_fpr": round(benign_fpr, 4),
        "overall_benign_fpr_ci": list(wilson_ci(fp, benign_total)),
        "metrics_note": ("recall and benign_fpr are measured on independent pools; "
                         "precision/F1 are intentionally omitted (no shared prevalence)"),
        "truncated": args.max_samples is not None,
        "max_samples": args.max_samples,
        "malicious_total": mal_total,
        "malicious_detected": tp,
        "benign_total": benign_total,
        "benign_false_positives": fp,
        "evasion_total": ev_total,
        "evasion_detected": ev_detected,
        "threshold": args.threshold,
        "wall_time_malicious_s": round(mal_wall, 2),
        "wall_time_evasion_s": round(ev_wall, 2),
        "wall_time_benign_s": round(benign_wall, 2),
    }

    # --- Optional CI gate ---
    gate = None
    if args.gate:
        gate = evaluate_gate(category_results, benign_results,
                             args.recall_floor, args.fpr_ceiling, args.min_slice)
        print_gate(gate)

    # --- Write JSON artifact ---
    try:
        from na0s._version import __version__ as na0s_version
    except ImportError:
        na0s_version = "unknown"

    output = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": na0s_version,
        "git_sha": _git_sha(),
        "summary": summary,
        "gate": gate,
        "per_category": category_results,
        "per_evasion_type": evasion_results,
        "per_benign": benign_results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(f"\nResults written to: {args.output}")

    # --- Overall summary ---
    print()
    print("=" * 70)
    print("Overall Summary")
    print("=" * 70)
    print(f"  Malicious holdout recall:  {summary['overall_malicious_recall']:.2%} "
          f"({tp}/{mal_total})  CI={summary['overall_malicious_recall_ci']}")
    print(f"  Benign false-positive rate:{summary['overall_benign_fpr']:.2%} "
          f"({fp}/{benign_total})  CI={summary['overall_benign_fpr_ci']}")
    print(f"  Evasion detection rate:    {summary['overall_evasion_detection_rate']:.2%} "
          f"({ev_detected}/{ev_total})")
    print(f"  Threshold:                 {args.threshold}")
    if args.max_samples is not None:
        print("  NOTE: truncated run (--max-samples) — not representative.")
    print()

    if gate is not None and not gate["passed"]:
        print("CI gate FAILED — see breaches above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
