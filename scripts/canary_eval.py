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
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
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
_DEFAULT_JSON = str(_PROJECT_ROOT / "data" / "canary" / "canary_results.json")
_DEFAULT_JSONL = str(_PROJECT_ROOT / "data" / "canary" / "canary_history.jsonl")
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


def evaluate(csv_path: str, verbose: bool = False, model_dir: str = None) -> dict:
    """Run Na0S prediction on the canary set and return metrics.

    When *model_dir* is given (e.g. ``data/processed`` during a retrain), the
    CANDIDATE model triple is scored instead of the shipped package model — so
    the deploy gate validates the freshly-trained model, not the old one.  This
    repoints predict.py's structural-scaler + char-vectorizer caches at the
    candidate too; otherwise classify_prompt would build features with the
    SHIPPED scaler/char-vectorizer (mismatched) and the gate would be invalid.
    """

    print("=" * 70)
    print("  Na0S Canary Evaluation Set")
    print("=" * 70)
    print(f"  CSV:       {csv_path}")
    print(f"  Threshold: {DECISION_THRESHOLD}")
    print()

    # Load model
    if model_dir:
        import os as _os
        import na0s.predict as _p
        # Repoint ALL artifact paths predict.py reads from its module globals so
        # classify_prompt scores the candidate consistently (model + tfidf +
        # char-tfidf + structural scaler all from the candidate dir).
        _p.SCALER_PATH = _os.path.join(model_dir, "structural_scaler.pkl")
        _p.CHAR_VECTORIZER_PATH = _os.path.join(model_dir, "char_tfidf_vectorizer.pkl")
        _p._cached_scaler = None
        _p._cached_char_vectorizer = None
        print(f"  Loading CANDIDATE model from {model_dir} ...")
        vectorizer = safe_load(_os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        model = safe_load(_os.path.join(model_dir, "model.pkl"))
    else:
        print("  Loading shipped package model ...")
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
    classification_error_count = 0
    technique_results = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})

    t0 = time.perf_counter()

    for i, row in enumerate(rows):
        text = row["text"]
        true_label = row["label"]
        technique = row.get("technique", "")

        try:
            pred_label_str, score, hits, l0, _detailed, _emb, _perp = classify_prompt(
                text, vectorizer, model
            )
        except Exception as exc:
            # If classification fails, treat as prediction = 0 (safe)
            # for backward compatibility, but track the error.
            pred_label_str = "SAFE"
            score = 0.0
            hits = []
            classification_error_count += 1
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

    # Classification error summary
    if classification_error_count > 0:
        print()
        print("-" * 70)
        print(f"  CLASSIFICATION ERRORS: {classification_error_count}")
        print("-" * 70)
        print(f"  WARNING: {classification_error_count} sample(s) raised exceptions during "
              f"classify_prompt().")
        print(f"  These were coerced to SAFE predictions for backward compatibility,")
        print(f"  but indicate a potentially broken evaluator path.")

    # Quality gate
    print()
    print("-" * 70)
    print("  QUALITY GATE")
    print("-" * 70)

    inj_pass = metrics["tpr"] >= _INJ_ACCURACY_THRESHOLD
    ben_pass = metrics["tnr"] >= _BEN_ACCURACY_THRESHOLD
    no_classification_errors = classification_error_count == 0

    print(f"  Injection TPR >= {_INJ_ACCURACY_THRESHOLD:.0%}:  "
          f"{'PASS' if inj_pass else 'FAIL'}  ({metrics['tpr']:.2%})")
    print(f"  Benign TNR    >= {_BEN_ACCURACY_THRESHOLD:.0%}:  "
          f"{'PASS' if ben_pass else 'FAIL'}  ({metrics['tnr']:.2%})")
    print(f"  Classification errors == 0: "
          f"{'PASS' if no_classification_errors else 'FAIL'}  ({classification_error_count} errors)")

    passed = inj_pass and ben_pass and no_classification_errors
    print()
    if passed:
        print("  OVERALL: PASS")
    else:
        print("  OVERALL: FAIL")
    print("=" * 70)

    return {
        "metrics": metrics,
        "errors": errors,
        "classification_errors": classification_error_count,
        "technique_results": dict(technique_results),
        "passed": passed,
        "elapsed_s": elapsed,
        "_rows": rows,          # kept for JSON export; not part of public API
    }


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(
    result: dict,
    csv_path: str,
    json_path: str = _DEFAULT_JSON,
    jsonl_path: str = _DEFAULT_JSONL,
) -> None:
    """Write canary_results.json and append a summary line to canary_history.jsonl."""

    rows = result["_rows"]          # injected by evaluate()
    n_inj = sum(1 for r in rows if r["label"] == 1)
    n_ben = sum(1 for r in rows if r["label"] == 0)
    metrics = result["metrics"]
    elapsed = result["elapsed_s"]
    passed = result["passed"]
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build the errors list in the documented format.
    errors_export = []
    for idx, row, pred_label, score in result["errors"]:
        errors_export.append({
            "index": idx,
            "true_label": row["label"],
            "predicted_label": pred_label,
            "score": score,
            "technique": row.get("technique", ""),
            "text_preview": row["text"][:80],
        })

    payload = {
        "timestamp": timestamp,
        "csv_path": csv_path,
        "sample_count": {
            "total": len(rows),
            "injection": n_inj,
            "benign": n_ben,
        },
        "metrics": metrics,
        "per_technique": result["technique_results"],
        "passed": passed,
        "classification_errors": result.get("classification_errors", 0),
        "thresholds": {
            "inj_accuracy": _INJ_ACCURACY_THRESHOLD,
            "ben_accuracy": _BEN_ACCURACY_THRESHOLD,
        },
        "errors": errors_export,
        "elapsed_s": elapsed,
    }

    # Write full results file (overwrite).
    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  JSON results written to: {out_path}")

    # Append one-line summary to history log.
    history_path = Path(jsonl_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": timestamp,
        "passed": passed,
        "accuracy": metrics["accuracy"],
        "tpr": metrics["tpr"],
        "tnr": metrics["tnr"],
        "f1": metrics["f1"],
        "sample_count": len(rows),
        "classification_errors": result.get("classification_errors", 0),
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"  History appended to:     {history_path}")


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
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory holding the CANDIDATE model.pkl/tfidf_vectorizer.pkl/"
             "char_tfidf_vectorizer.pkl/structural_scaler.pkl to score instead of "
             "the shipped package model (e.g. data/processed during a retrain).",
    )

    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument(
        "--json",
        dest="json_path",
        default=_DEFAULT_JSON,
        metavar="PATH",
        help=(
            f"Path for JSON results output "
            f"(default: {_DEFAULT_JSON})"
        ),
    )
    json_group.add_argument(
        "--no-json",
        dest="json_path",
        action="store_const",
        const=None,
        help="Disable JSON export entirely",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: Canary CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    result = evaluate(args.csv, verbose=args.verbose, model_dir=args.model_dir)

    if args.json_path is not None:
        export_json(result, csv_path=args.csv, json_path=args.json_path)

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
