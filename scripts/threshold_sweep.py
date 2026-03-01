#!/usr/bin/env python3
"""Threshold sweep benchmark for the Na0S prompt injection detector.

Evaluates the scan() function across multiple decision thresholds on holdout
datasets and an adversarial evasion dataset.  Produces a human-readable table
and a machine-readable JSON report.

Usage
-----
    # Full sweep (all samples)
    python scripts/threshold_sweep.py

    # Quick smoke test (50 samples per dataset)
    python scripts/threshold_sweep.py --max-samples 50

    # Custom thresholds
    python scripts/threshold_sweep.py --thresholds 0.40 0.50 0.60

    # Skip adversarial dataset
    python scripts/threshold_sweep.py --skip-adversarial
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# Ensure the project root is on sys.path so `na0s` can be imported when
# running the script directly (e.g. `python scripts/threshold_sweep.py`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from na0s.predict import scan  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ThresholdMetrics:
    """Metrics for a single threshold on a single dataset."""
    threshold: float = 0.0
    dataset: str = ""
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fpr: float = 0.0
    accuracy: float = 0.0
    total_samples: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[dict]:
    """Load samples from a JSONL file.

    Each line must have at least ``text`` and ``label`` keys.

    Parameters
    ----------
    path : str
        Path to the JSONL file.
    max_samples : int or None
        If set, load at most this many samples.

    Returns
    -------
    list[dict]
        List of sample dicts.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: List[bool],
    labels: List[int],
    threshold: float,
    dataset_name: str,
    latencies_ms: List[float],
) -> ThresholdMetrics:
    """Compute classification metrics from predictions and ground-truth labels.

    Parameters
    ----------
    predictions : list[bool]
        Model predictions (True = malicious, False = safe).
    labels : list[int]
        Ground-truth labels (1 = malicious, 0 = safe).
    threshold : float
        The threshold that was used for these predictions.
    dataset_name : str
        Human-readable name for the dataset.
    latencies_ms : list[float]
        Per-sample latencies in milliseconds.

    Returns
    -------
    ThresholdMetrics
    """
    tp = tn = fp = fn = 0
    for pred, label in zip(predictions, labels):
        if pred and label == 1:
            tp += 1
        elif not pred and label == 0:
            tn += 1
        elif pred and label == 0:
            fp += 1
        elif not pred and label == 1:
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    return ThresholdMetrics(
        threshold=threshold,
        dataset=dataset_name,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        fpr=round(fpr, 4),
        accuracy=round(accuracy, 4),
        total_samples=total,
        avg_latency_ms=round(avg_latency, 2),
    )


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

def run_sweep_on_dataset(
    samples: List[dict],
    thresholds: List[float],
    dataset_name: str,
) -> List[ThresholdMetrics]:
    """Run the threshold sweep on a list of samples.

    For each threshold, iterates over all samples, calls ``scan()``, and
    collects predictions.  Returns a list of ThresholdMetrics (one per
    threshold).

    Parameters
    ----------
    samples : list[dict]
        Dataset samples (must have ``text`` and ``label`` keys).
    thresholds : list[float]
        Thresholds to evaluate.
    dataset_name : str
        Human-readable dataset name for reporting.

    Returns
    -------
    list[ThresholdMetrics]
    """
    # Pre-load the model once to avoid repeated cold-start overhead.
    # We call scan() once with a dummy string to trigger model loading.
    scan("warmup", threshold=0.55)

    results = []

    for t in thresholds:
        predictions: List[bool] = []
        labels: List[int] = []
        latencies_ms: List[float] = []

        for i, sample in enumerate(samples):
            text = sample["text"]
            label = sample["label"]

            t0 = time.perf_counter()
            result = scan(text, threshold=t)
            elapsed = (time.perf_counter() - t0) * 1000

            predictions.append(result.is_malicious)
            labels.append(label)
            latencies_ms.append(elapsed)

            # Progress indicator every 100 samples
            if (i + 1) % 100 == 0:
                print(f"  [{dataset_name}] threshold={t:.2f}: {i+1}/{len(samples)} samples...")

        metrics = compute_metrics(predictions, labels, t, dataset_name, latencies_ms)
        results.append(metrics)
        print(f"  [{dataset_name}] threshold={t:.2f}: done ({metrics.total_samples} samples, "
              f"F1={metrics.f1:.4f}, FPR={metrics.fpr:.4f})")

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(results: List[ThresholdMetrics], title: str) -> None:
    """Print a formatted table of threshold sweep results."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    header = (
        f"{'Thresh':>7s} | {'TP':>5s} {'TN':>5s} {'FP':>5s} {'FN':>5s} | "
        f"{'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'FPR':>7s} {'Acc':>7s} | "
        f"{'Latency':>9s} {'N':>5s}"
    )
    print(header)
    print("-" * 100)
    for m in results:
        row = (
            f"{m.threshold:>7.2f} | {m.tp:>5d} {m.tn:>5d} {m.fp:>5d} {m.fn:>5d} | "
            f"{m.precision:>7.4f} {m.recall:>7.4f} {m.f1:>7.4f} {m.fpr:>7.4f} {m.accuracy:>7.4f} | "
            f"{m.avg_latency_ms:>7.2f}ms {m.total_samples:>5d}"
        )
        print(row)
    print("-" * 100)


def find_optimal(results: List[ThresholdMetrics]) -> ThresholdMetrics:
    """Find the threshold with the highest F1 score."""
    return max(results, key=lambda m: m.f1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

DEFAULT_SAFE_PATH = str(_PROJECT_ROOT / "data" / "holdout" / "safe_holdout.jsonl")
DEFAULT_MALICIOUS_PATH = str(_PROJECT_ROOT / "data" / "holdout" / "malicious_holdout.jsonl")
DEFAULT_ADVERSARIAL_PATH = str(_PROJECT_ROOT / "data" / "benchmark" / "adversarial_evasion.jsonl")
DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "benchmarks" / "results")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Threshold sweep benchmark for Na0S prompt injection detector.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Thresholds to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to load per dataset (default: all).",
    )
    parser.add_argument(
        "--safe-path",
        type=str,
        default=DEFAULT_SAFE_PATH,
        help="Path to safe holdout JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--malicious-path",
        type=str,
        default=DEFAULT_MALICIOUS_PATH,
        help="Path to malicious holdout JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--adversarial-path",
        type=str,
        default=DEFAULT_ADVERSARIAL_PATH,
        help="Path to adversarial evasion JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON output (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-adversarial",
        action="store_true",
        help="Skip the adversarial evasion dataset.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> dict:
    """Run the threshold sweep and return the results dict.

    Parameters
    ----------
    argv : list[str] or None
        CLI arguments.  If None, reads from sys.argv.

    Returns
    -------
    dict
        The full results dict (also written to JSON).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    thresholds = sorted(args.thresholds)

    # --- Load datasets ---
    print("\n=== Na0S Threshold Sweep Benchmark ===\n")

    print(f"Loading safe holdout from: {args.safe_path}")
    safe_samples = load_jsonl(args.safe_path, args.max_samples)
    print(f"  Loaded {len(safe_samples)} safe samples")

    print(f"Loading malicious holdout from: {args.malicious_path}")
    malicious_samples = load_jsonl(args.malicious_path, args.max_samples)
    print(f"  Loaded {len(malicious_samples)} malicious samples")

    # Combine safe + malicious for holdout sweep
    holdout_samples = safe_samples + malicious_samples
    print(f"  Combined holdout: {len(holdout_samples)} samples "
          f"({len(safe_samples)} safe + {len(malicious_samples)} malicious)")

    # --- Run holdout sweep ---
    print(f"\nRunning holdout sweep with thresholds: {thresholds}")
    t_start = time.perf_counter()
    holdout_results = run_sweep_on_dataset(holdout_samples, thresholds, "holdout")
    holdout_elapsed = time.perf_counter() - t_start

    print_table(holdout_results, "Holdout Dataset Results")
    best_holdout = find_optimal(holdout_results)
    print(f"\n  >> Best holdout threshold: {best_holdout.threshold:.2f} "
          f"(F1={best_holdout.f1:.4f}, FPR={best_holdout.fpr:.4f})")

    # --- Run adversarial sweep ---
    adversarial_results = []
    adversarial_elapsed = 0.0
    if not args.skip_adversarial and os.path.isfile(args.adversarial_path):
        print(f"\nLoading adversarial evasion from: {args.adversarial_path}")
        adversarial_samples = load_jsonl(args.adversarial_path, args.max_samples)
        print(f"  Loaded {len(adversarial_samples)} adversarial samples")

        print(f"\nRunning adversarial sweep with thresholds: {thresholds}")
        t_start = time.perf_counter()
        adversarial_results = run_sweep_on_dataset(
            adversarial_samples, thresholds, "adversarial"
        )
        adversarial_elapsed = time.perf_counter() - t_start

        print_table(adversarial_results, "Adversarial Evasion Dataset Results")
        best_adv = find_optimal(adversarial_results)
        print(f"\n  >> Best adversarial threshold: {best_adv.threshold:.2f} "
              f"(F1={best_adv.f1:.4f}, Recall={best_adv.recall:.4f})")
    elif args.skip_adversarial:
        print("\nSkipping adversarial dataset (--skip-adversarial).")
    else:
        print(f"\nAdversarial dataset not found at {args.adversarial_path}, skipping.")

    # --- Assemble output ---
    output = {
        "meta": {
            "thresholds": thresholds,
            "max_samples": args.max_samples,
            "safe_path": args.safe_path,
            "malicious_path": args.malicious_path,
            "adversarial_path": args.adversarial_path,
            "holdout_elapsed_s": round(holdout_elapsed, 2),
            "adversarial_elapsed_s": round(adversarial_elapsed, 2),
        },
        "holdout": [m.to_dict() for m in holdout_results],
        "adversarial": [m.to_dict() for m in adversarial_results],
        "best_holdout": best_holdout.to_dict(),
    }
    if adversarial_results:
        output["best_adversarial"] = find_optimal(adversarial_results).to_dict()

    # --- Write JSON ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "threshold_sweep.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults written to: {output_path}")

    # --- Summary ---
    print(f"\nTotal holdout time: {holdout_elapsed:.1f}s")
    if adversarial_elapsed > 0:
        print(f"Total adversarial time: {adversarial_elapsed:.1f}s")
    print("Done.\n")

    return output


if __name__ == "__main__":
    main()
