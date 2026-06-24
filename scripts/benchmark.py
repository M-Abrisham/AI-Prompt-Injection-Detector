#!/usr/bin/env python3
"""Unified benchmark harness for Na0S prompt-injection detection.

Loads a JSONL dataset (each line: {"text": "...", "label": 0|1}), runs the
selected detection tool on every sample, and computes trustworthy ML metrics.

Metrics posture (imbalanced security task):
  * Leads with TPR/TNR/precision/recall, each with a 95% bootstrap CI
    (``na0s.judge.calibration``), so a small sample's noise is visible.
  * Accuracy is DEMOTED to a footnote shown next to prevalence — on a
    benign-heavy slice it is dominated by true negatives and hides a weak
    detector.
  * The optional sklearn AUC, when skipped, emits an explicit WARNING rather
    than being silently swallowed.

Usage
-----
    python scripts/benchmark.py --dataset data/benchmark/test_sample.jsonl
    python scripts/benchmark.py --dataset data/benchmark/test_sample.jsonl \
        --threshold 0.50 --max-samples 100 \
        --output /tmp/bench_per_sample.jsonl \
        --summary /tmp/bench_summary.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure Na0S package is importable when running from the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Trustworthy-metrics primitives (bootstrap CIs, no-accuracy-headline posture).
# Import is best-effort: the harness must still run if numpy/the judge package
# is unavailable, so we degrade to "no CIs" rather than crash.
try:
    from na0s.judge import calibration as _calibration
    _HAS_CALIBRATION = True
except ImportError:  # pragma: no cover - exercised only without numpy/judge pkg
    _calibration = None
    _HAS_CALIBRATION = False


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _run_na0s(text, threshold):
    """Run Na0S scan() and return a normalised prediction dict."""
    from na0s import scan  # deferred import so argparse --help is fast
    result = scan(text, threshold=threshold)
    return {
        "prediction": 1 if result.is_malicious else 0,
        "score": result.risk_score,
        "latency_ms": result.elapsed_ms,
        "label": result.label,
    }


def _run_llm_guard(text, threshold):
    """Run LLM Guard scan and return a normalised prediction dict."""
    from scripts.wrappers.llm_guard import LLMGuardWrapper

    if not hasattr(_run_llm_guard, "_wrapper"):
        _run_llm_guard._wrapper = LLMGuardWrapper()

    result = _run_llm_guard._wrapper.predict(text)
    return {
        "prediction": result["label"],
        "score": result["score"],
        "latency_ms": result["latency_ms"],
        "label": "MALICIOUS" if result["label"] == 1 else "SAFE",
    }


def _run_prompt_guard(text, threshold):
    """Run Prompt Guard 2 scan and return a normalised prediction dict."""
    from scripts.wrappers.prompt_guard import PromptGuardWrapper

    if not hasattr(_run_prompt_guard, "_wrapper"):
        _run_prompt_guard._wrapper = PromptGuardWrapper()

    result = _run_prompt_guard._wrapper.predict(text)
    return {
        "prediction": result["label"],
        "score": result["score"],
        "latency_ms": result["latency_ms"],
        "label": "MALICIOUS" if result["label"] == 1 else "SAFE",
    }


_TOOL_RUNNERS = {
    "na0s": _run_na0s,
    "llm_guard": _run_llm_guard,
    "prompt_guard": _run_prompt_guard,
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path, max_samples=None):
    """Load a JSONL dataset and return a list of (text, label) tuples."""
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
            text = obj.get("text", "")
            label = int(obj.get("label", 0))
            samples.append((text, label))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _percentile(sorted_vals, pct):
    """Return the p-th percentile from an already-sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def compute_metrics(records, threshold):
    """Compute classification and latency metrics from per-sample records.

    Parameters
    ----------
    records : list[dict]
        Each dict has keys: ground_truth, prediction, score, latency_ms.
    threshold : float
        The decision threshold used.

    Returns
    -------
    dict  — the summary metrics.
    """
    tp = tn = fp = fn = 0
    latencies = []
    ground_truths = []
    predictions = []
    scores = []

    for rec in records:
        gt = rec["ground_truth"]
        pred = rec["prediction"]
        ground_truths.append(gt)
        predictions.append(pred)
        scores.append(rec["score"])
        latencies.append(rec["latency_ms"])

        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    n = len(records)

    # True-positive / true-negative rates (sensitivity / specificity). These are
    # the honest headline metrics on an imbalanced security task — see the
    # accuracy-demotion note below.
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # == recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # == specificity

    # Precision, Recall, F1 — handle zero-division gracefully
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0
    # prevalence = fraction of samples that are actually positive (malicious).
    # Reported alongside accuracy so a TN-dominated accuracy on a benign-heavy
    # slice is not misread as a strong detector.
    prevalence = (tp + fn) / n if n > 0 else 0.0

    # Bootstrap 95% CIs for the rate metrics, when the calibration primitives
    # and raw labels/predictions are available. alpha=0.05 -> a 95% percentile
    # interval (the conventional reporting level for a security benchmark).
    # n_boot defaults to calibration.bootstrap_ci's 2000 (the standard floor for
    # a stable 95% percentile interval); not overridden here.
    ci_alpha = 0.05
    tpr_ci = tnr_ci = precision_ci = recall_ci = None
    if _HAS_CALIBRATION and predictions is not None and n > 0:
        y_true = ground_truths
        y_pred = predictions

        def _ci(stat_fn):
            lo, hi = _calibration.bootstrap_ci(
                y_true, y_pred, stat_fn, alpha=ci_alpha,
            )
            # round for stable JSON/printing; None when the resample never had
            # the relevant class (NaN) so callers don't print a bogus interval.
            import math
            if math.isnan(lo) or math.isnan(hi):
                return None
            return [round(lo, 4), round(hi, 4)]

        tpr_ci = _ci(_calibration.tpr_stat)
        tnr_ci = _ci(_calibration.tnr_stat)
        precision_ci = _ci(_calibration.precision_stat)
        recall_ci = _ci(_calibration.recall_stat)

    # AUC-ROC and AUC-PR — optional sklearn dependency. When skipped, emit an
    # explicit WARNING (do NOT silently swallow): a missing AUC quietly omitted
    # from a report can be mistaken for a clean run.
    auc_roc = None
    auc_pr = None
    single_class = len(set(ground_truths)) <= 1
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if not single_class:
            auc_roc = round(roc_auc_score(ground_truths, scores), 4)
            auc_pr = round(average_precision_score(ground_truths, scores), 4)
        else:
            print(
                "WARNING: AUC-ROC/AUC-PR skipped — dataset is single-class "
                "(all samples share one label); AUC is undefined.",
                file=sys.stderr,
            )
    except ImportError:
        print(
            "WARNING: AUC-ROC/AUC-PR skipped — sklearn is not installed. "
            "Install scikit-learn to report threshold-independent ranking "
            "metrics.",
            file=sys.stderr,
        )
    except ValueError as exc:
        # Defensive: sklearn raised for some other degenerate input.
        print(
            f"WARNING: AUC-ROC/AUC-PR skipped — {exc}",
            file=sys.stderr,
        )

    # Latency stats
    sorted_lat = sorted(latencies)
    total_time_sec = sum(latencies) / 1000.0 if latencies else 0.0

    n_malicious = sum(1 for r in records if r["ground_truth"] == 1)
    n_safe = n - n_malicious

    return {
        "n_samples": n,
        "n_malicious": n_malicious,
        "n_safe": n_safe,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        # Headline rate metrics (honest on imbalanced data) + their 95% CIs.
        "tpr": round(tpr, 4),
        "tnr": round(tnr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tpr_ci": tpr_ci,
        "tnr_ci": tnr_ci,
        "precision_ci": precision_ci,
        "recall_ci": recall_ci,
        "ci_level": round(1.0 - ci_alpha, 4),  # e.g. 0.95 for alpha=0.05
        # Accuracy DEMOTED to a footnote: paired with prevalence so a
        # TN-dominated accuracy on a benign-heavy slice is not misread.
        "accuracy": round(accuracy, 4),
        "prevalence": round(prevalence, 4),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(_percentile(sorted_lat, 50), 2),
        "p95_latency_ms": round(_percentile(sorted_lat, 95), 2),
        "p99_latency_ms": round(_percentile(sorted_lat, 99), 2),
        "throughput_per_sec": round(n / total_time_sec, 2) if total_time_sec > 0 else 0.0,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_ci(ci):
    """Render a [lo, hi] CI as a compact '[lo, hi]' string, or 'N/A'."""
    if not ci:
        return "N/A"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def print_summary_table(metrics, tool, dataset_name):
    """Print a markdown-formatted summary table to stdout.

    Metric ordering is deliberate: the imbalance-honest rate metrics
    (TPR/TNR/precision/recall, each with a 95% bootstrap CI) lead. Accuracy is
    DEMOTED to a footnote below the table and shown next to prevalence, because
    on a benign-heavy slice accuracy is dominated by true negatives and hides a
    weak detector.
    """
    ci_pct = int(round(metrics.get("ci_level", 0.95) * 100))
    print()
    print(f"## Benchmark Results: {tool} on {dataset_name}")
    print()
    print("| Metric               | Value        | "
          f"{ci_pct}% CI            |")
    print("|----------------------|--------------|----------------------|")
    print(f"| Samples              | {metrics['n_samples']:>12} | "
          f"{'':>20} |")
    print(f"| Malicious            | {metrics['n_malicious']:>12} | "
          f"{'':>20} |")
    print(f"| Safe                 | {metrics['n_safe']:>12} | "
          f"{'':>20} |")
    print(f"| TP                   | {metrics['tp']:>12} | {'':>20} |")
    print(f"| TN                   | {metrics['tn']:>12} | {'':>20} |")
    print(f"| FP                   | {metrics['fp']:>12} | {'':>20} |")
    print(f"| FN                   | {metrics['fn']:>12} | {'':>20} |")
    # --- HEADLINE rate metrics (lead the table), each with its CI ---
    print(f"| TPR (recall)         | {metrics['tpr']:>12.4f} | "
          f"{_fmt_ci(metrics.get('tpr_ci')):>20} |")
    print(f"| TNR (specificity)    | {metrics['tnr']:>12.4f} | "
          f"{_fmt_ci(metrics.get('tnr_ci')):>20} |")
    print(f"| Precision            | {metrics['precision']:>12.4f} | "
          f"{_fmt_ci(metrics.get('precision_ci')):>20} |")
    print(f"| Recall               | {metrics['recall']:>12.4f} | "
          f"{_fmt_ci(metrics.get('recall_ci')):>20} |")
    print(f"| F1                   | {metrics['f1']:>12.4f} | {'':>20} |")
    print(f"| FPR                  | {metrics['fpr']:>12.4f} | {'':>20} |")
    auc_roc_str = f"{metrics['auc_roc']:.4f}" if metrics['auc_roc'] is not None else "N/A"
    auc_pr_str = f"{metrics['auc_pr']:.4f}" if metrics['auc_pr'] is not None else "N/A"
    print(f"| AUC-ROC              | {auc_roc_str:>12} | {'':>20} |")
    print(f"| AUC-PR               | {auc_pr_str:>12} | {'':>20} |")
    print(f"| Avg Latency (ms)     | {metrics['avg_latency_ms']:>12.2f} | "
          f"{'':>20} |")
    print(f"| P50 Latency (ms)     | {metrics['p50_latency_ms']:>12.2f} | "
          f"{'':>20} |")
    print(f"| P95 Latency (ms)     | {metrics['p95_latency_ms']:>12.2f} | "
          f"{'':>20} |")
    print(f"| P99 Latency (ms)     | {metrics['p99_latency_ms']:>12.2f} | "
          f"{'':>20} |")
    print(f"| Throughput (samp/s)  | {metrics['throughput_per_sec']:>12.2f} | "
          f"{'':>20} |")
    print(f"| Threshold            | {metrics['threshold']:>12.2f} | "
          f"{'':>20} |")
    print()
    # --- Accuracy demoted to a footnote, shown with prevalence ---
    print(
        f"_Footnote — accuracy: {metrics['accuracy']:.4f} "
        f"(prevalence={metrics['prevalence']:.4f}). "
        "Accuracy is NOT a headline metric: on an imbalanced slice it is "
        "dominated by true negatives; read TPR/TNR/precision/recall above._"
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Na0S prompt-injection detection against "
                    "labelled JSONL datasets.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to JSONL file (each line: {\"text\": ..., \"label\": 0|1}).",
    )
    parser.add_argument(
        "--tool", default="na0s", choices=sorted(_TOOL_RUNNERS),
        help="Detection tool to benchmark (default: na0s).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="Decision threshold for Na0S (default: 0.55).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples (for quick iteration).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write per-sample results as JSONL to this path.",
    )
    parser.add_argument(
        "--summary", default=None,
        help="Write summary metrics as JSON to this path.",
    )
    args = parser.parse_args()

    # Validate dataset path
    if not os.path.isfile(args.dataset):
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    runner = _TOOL_RUNNERS[args.tool]

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    samples = load_dataset(args.dataset, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples.")

    if not samples:
        print("ERROR: no samples loaded.", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    records = []
    wall_start = time.perf_counter()

    for i, (text, gt_label) in enumerate(samples):
        result = runner(text, args.threshold)
        record = {
            "text": text,
            "ground_truth": gt_label,
            "prediction": result["prediction"],
            "score": result["score"],
            "latency_ms": result["latency_ms"],
            "tool": args.tool,
        }
        records.append(record)

        # Progress indicator
        status = "MAL" if result["prediction"] == 1 else "safe"
        gt_str = "MAL" if gt_label == 1 else "safe"
        match = "OK" if result["prediction"] == gt_label else "MISS"
        print(f"  [{i+1:>4}/{len(samples)}] gt={gt_str} pred={status} "
              f"score={result['score']:.4f} "
              f"latency={result['latency_ms']:.1f}ms  {match}")

    wall_elapsed = time.perf_counter() - wall_start

    # Compute metrics
    metrics = compute_metrics(records, args.threshold)
    dataset_name = os.path.basename(args.dataset)

    # Print summary
    print_summary_table(metrics, args.tool, dataset_name)
    print(f"Wall-clock time: {wall_elapsed:.2f}s")

    # Build full summary object
    try:
        from na0s._version import __version__ as na0s_version
    except ImportError:
        na0s_version = "unknown"

    summary = {
        "dataset": dataset_name,
        "tool": args.tool,
        **metrics,
        "version": na0s_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write per-sample JSONL
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        print(f"Per-sample results written to: {args.output}")

    # Write summary JSON
    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
            fh.write("\n")
        print(f"Summary written to: {args.summary}")


if __name__ == "__main__":
    main()
