#!/usr/bin/env python3
"""Canary evaluation set -- verify model quality after retraining.

This script loads the hand-verified canary evaluation CSV (never used for
training) and runs Na0S prediction on each sample.  It computes detailed
metrics and exits with code 1 if quality thresholds are not met.

Usage:
    python scripts/canary_eval.py
    python scripts/canary_eval.py --csv data/canary/canary_eval.csv
    python scripts/canary_eval.py --verbose

Thresholds (exit code 1 if violated):
    Injection accuracy (TPR) >= 95%
    Benign accuracy (TNR)    >= 90%
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure the project src/ is importable when running as a standalone script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from na0s.safe_pickle import safe_load
from na0s.models import get_model_path
from na0s.predict import classify_prompt, DECISION_THRESHOLD


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_CSV = str(_PROJECT_ROOT / "data" / "canary" / "canary_eval.csv")
_INJ_ACCURACY_THRESHOLD = 0.95   # TPR -- injection recall
_BEN_ACCURACY_THRESHOLD = 0.90   # TNR -- benign recall (1 - FPR)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute classification metrics from true/predicted label lists."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    total = tp + tn + fp + fn
    accuracy = _safe_div(tp + tn, total)
    tpr = _safe_div(tp, tp + fn)        # recall / sensitivity
    tnr = _safe_div(tn, tn + fp)        # specificity
    fpr = _safe_div(fp, fp + tn)        # false positive rate
    fnr = _safe_div(fn, fn + tp)        # false negative rate
    precision = _safe_div(tp, tp + fp)
    recall = tpr
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def load_canary_csv(csv_path: str) -> list[dict]:
    """Load the canary CSV and return a list of row dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            rows.append(row)
    return rows


def evaluate(csv_path: str, verbose: bool = False) -> dict:
    """Run Na0S prediction on the canary set and return metrics."""

    print("=" * 70)
    print("  Na0S Canary Evaluation Set")
    print("=" * 70)
    print(f"  CSV:       {csv_path}")
    print(f"  Threshold: {DECISION_THRESHOLD}")
    print()

    # Load model
    print("  Loading TF-IDF model...")
    vectorizer = safe_load(get_model_path("tfidf_vectorizer.pkl"))
    model = safe_load(get_model_path("model.pkl"))
    print("  Model loaded.\n")

    # Load data
    rows = load_canary_csv(csv_path)
    n_inj = sum(1 for r in rows if r["label"] == 1)
    n_ben = sum(1 for r in rows if r["label"] == 0)
    print(f"  Samples:   {len(rows)} total ({n_inj} injection, {n_ben} benign)")
    print()

    # Run predictions
    y_true = []
    y_pred = []
    errors = []          # (row_index, row, predicted_label, score)
    technique_results = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})

    t0 = time.perf_counter()

    for i, row in enumerate(rows):
        text = row["text"]
        true_label = row["label"]
        technique = row.get("technique", "")

        try:
            pred_label_str, score, hits, l0, _detailed, _emb = classify_prompt(
                text, vectorizer, model
            )
        except Exception as exc:
            # If classification fails, treat as prediction = 0 (safe)
            pred_label_str = "SAFE"
            score = 0.0
            hits = []
            print(f"  [WARN] Row {i}: classification error: {exc}")

        pred_label = 1 if "MALICIOUS" in pred_label_str or "BLOCKED" in pred_label_str else 0

        y_true.append(true_label)
        y_pred.append(pred_label)

        # Track per-technique results
        if true_label == 1 and pred_label == 1:
            technique_results[technique]["tp"] += 1
        elif true_label == 0 and pred_label == 0:
            technique_results[technique]["tn"] += 1
        elif true_label == 0 and pred_label == 1:
            technique_results[technique]["fp"] += 1
        elif true_label == 1 and pred_label == 0:
            technique_results[technique]["fn"] += 1

        # Track errors
        is_correct = (true_label == pred_label)
        if not is_correct:
            errors.append((i, row, pred_label, score))

        if verbose:
            status = "OK" if is_correct else "MISS"
            print(
                f"  [{status}] #{i:03d}  true={true_label}  pred={pred_label}  "
                f"score={score:.4f}  tech={technique}  "
                f"text={text[:60]}..."
            )

    elapsed = time.perf_counter() - t0

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)

    # Print report
    print()
    print("-" * 70)
    print("  RESULTS")
    print("-" * 70)
    print(f"  Overall accuracy:    {metrics['accuracy']:.4f}  ({metrics['tp'] + metrics['tn']}/{len(rows)})")
    print()
    print(f"  Injection (label=1):")
    print(f"    TPR (recall):      {metrics['tpr']:.4f}  ({metrics['tp']}/{metrics['tp'] + metrics['fn']})")
    print(f"    FNR (miss rate):   {metrics['fnr']:.4f}  ({metrics['fn']}/{metrics['tp'] + metrics['fn']})")
    print()
    print(f"  Benign (label=0):")
    print(f"    TNR (specificity): {metrics['tnr']:.4f}  ({metrics['tn']}/{metrics['tn'] + metrics['fp']})")
    print(f"    FPR:               {metrics['fpr']:.4f}  ({metrics['fp']}/{metrics['tn'] + metrics['fp']})")
    print()
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  F1 score:            {metrics['f1']:.4f}")
    print()
    print(f"  Confusion matrix:")
    print(f"                    Predicted Safe  Predicted Malicious")
    print(f"    Actual Safe       {metrics['tn']:>5d}           {metrics['fp']:>5d}")
    print(f"    Actual Malicious  {metrics['fn']:>5d}           {metrics['tp']:>5d}")
    print()
    print(f"  Elapsed:             {elapsed:.2f}s  ({elapsed / len(rows) * 1000:.1f}ms/sample)")

    # Per-technique breakdown
    print()
    print("-" * 70)
    print("  PER-TECHNIQUE BREAKDOWN")
    print("-" * 70)
    print(f"  {'Technique':<12} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'Acc':>8}")
    print(f"  {'-'*12} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*8}")
    for tech in sorted(technique_results.keys()):
        tr = technique_results[tech]
        total = tr["tp"] + tr["tn"] + tr["fp"] + tr["fn"]
        correct = tr["tp"] + tr["tn"]
        acc = _safe_div(correct, total)
        print(f"  {tech:<12} {tr['tp']:>4} {tr['tn']:>4} {tr['fp']:>4} {tr['fn']:>4} {acc:>8.4f}")

    # Error details
    if errors:
        print()
        print("-" * 70)
        print(f"  MISCLASSIFIED SAMPLES ({len(errors)} errors)")
        print("-" * 70)
        for idx, row, pred, score in errors:
            true = row["label"]
            kind = "FALSE NEGATIVE" if true == 1 else "FALSE POSITIVE"
            print(f"  [{kind}] #{idx:03d}  score={score:.4f}  tech={row.get('technique', '')}")
            print(f"    {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}")
            if row.get("notes"):
                print(f"    Note: {row['notes']}")
            print()

    # Quality gate
    print("-" * 70)
    print("  QUALITY GATE")
    print("-" * 70)

    inj_pass = metrics["tpr"] >= _INJ_ACCURACY_THRESHOLD
    ben_pass = metrics["tnr"] >= _BEN_ACCURACY_THRESHOLD

    print(f"  Injection TPR >= {_INJ_ACCURACY_THRESHOLD:.0%}:  "
          f"{'PASS' if inj_pass else 'FAIL'}  ({metrics['tpr']:.2%})")
    print(f"  Benign TNR    >= {_BEN_ACCURACY_THRESHOLD:.0%}:  "
          f"{'PASS' if ben_pass else 'FAIL'}  ({metrics['tnr']:.2%})")

    passed = inj_pass and ben_pass
    print()
    if passed:
        print("  OVERALL: PASS")
    else:
        print("  OVERALL: FAIL")
    print("=" * 70)

    return {
        "metrics": metrics,
        "errors": errors,
        "technique_results": dict(technique_results),
        "passed": passed,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Na0S canary evaluation set",
    )
    parser.add_argument(
        "--csv",
        default=_DEFAULT_CSV,
        help=f"Path to canary evaluation CSV (default: {_DEFAULT_CSV})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample results",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: Canary CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    result = evaluate(args.csv, verbose=args.verbose)

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
