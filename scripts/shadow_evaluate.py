#!/usr/bin/env python3
"""Shadow evaluation — train a candidate model and compare against production.

Trains a candidate model on new data, evaluates it against holdout and canary
sets, and compares metrics against the current production model.  Auto-rejects
if quality drops below configurable thresholds.

Exit codes:
    0 = PASS — candidate model is safe to deploy
    1 = FAIL — candidate model regresses on one or more gates

Usage::

    python scripts/shadow_evaluate.py --candidate-data data/processed/combined_data.csv
    python scripts/shadow_evaluate.py --candidate-data new_data.csv --holdout data/holdout/ --canary data/canary/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_HOLDOUT = os.path.join(ROOT, "data", "holdout")
DEFAULT_CANARY = os.path.join(ROOT, "data", "canary")
DEFAULT_REPORT = os.path.join(ROOT, "data", "staging", "shadow_report.json")

# ── Rejection gates ─────────────────────────────────────────────────
MAX_F1_DROP = 0.02          # reject if F1 drops > 2%
MIN_CANARY_ACCURACY = 0.95  # reject if canary accuracy < 95%
MAX_FPR_INCREASE = 0.01     # reject if FPR increases > 1%


def _load_eval_set(directory: str) -> pd.DataFrame | None:
    """Load all CSVs/JSONLs from a directory into a single DataFrame."""
    if not os.path.isdir(directory):
        return None

    frames = []
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        frames.append(pd.read_csv(path))
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        frames.append(pd.read_json(path, lines=True))

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def _load_production_model():
    """Load the current production model and vectorizer."""
    from na0s.safe_pickle import safe_load

    model_path = os.path.join(ROOT, "data", "processed", "model.pkl")
    vec_path = os.path.join(ROOT, "data", "processed", "tfidf_vectorizer.pkl")

    if not os.path.isfile(model_path) or not os.path.isfile(vec_path):
        return None, None

    return safe_load(model_path), safe_load(vec_path)


def _compute_metrics(model, vectorizer, df: pd.DataFrame) -> dict:
    """Compute classification metrics for a model on a dataset."""
    X = vectorizer.transform(df["text"])
    y_true = df["label"].values.astype(int)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    safe_mask = y_true == 0
    mal_mask = y_true == 1

    fpr = y_pred[safe_mask].sum() / safe_mask.sum() if safe_mask.sum() > 0 else 0.0
    tpr = y_pred[mal_mask].sum() / mal_mask.sum() if mal_mask.sum() > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "fpr": float(fpr),
        "tpr": float(tpr),
        "n_samples": int(len(df)),
    }


def shadow_evaluate(
    candidate_data_path: str,
    holdout_dir: str = DEFAULT_HOLDOUT,
    canary_dir: str = DEFAULT_CANARY,
    report_path: str = DEFAULT_REPORT,
) -> dict:
    """Train candidate model and compare against production.

    Returns a dict with metrics comparison and PASS/FAIL verdict.
    """
    # Load candidate training data
    if not os.path.isfile(candidate_data_path):
        print(f"ERROR: Candidate data not found: {candidate_data_path}")
        sys.exit(1)

    print(f"Loading candidate data: {candidate_data_path}")
    train_df = pd.read_csv(candidate_data_path)
    train_df["text"] = train_df["text"].fillna("").astype(str)

    if len(train_df) < 100:
        print(f"ERROR: Only {len(train_df)} training samples — need at least 100.")
        sys.exit(1)

    # Train candidate model
    print("Training candidate model...")
    candidate_vec = TfidfVectorizer(
        lowercase=True, max_features=10000, ngram_range=(1, 3), sublinear_tf=True,
    )
    X_train = candidate_vec.fit_transform(train_df["text"])
    y_train = train_df["label"].values.astype(int)

    base_clf = LogisticRegression(max_iter=10000, random_state=0, class_weight="balanced")
    base_clf.fit(X_train, y_train)

    candidate_model = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
    candidate_model.fit(X_train, y_train)

    # Load production model
    print("Loading production model...")
    prod_model, prod_vec = _load_production_model()

    # Load evaluation sets
    holdout_df = _load_eval_set(holdout_dir)
    canary_df = _load_eval_set(canary_dir)

    report = {"gates": [], "verdict": "PASS"}
    failures = []

    # Evaluate on holdout set
    if holdout_df is not None and len(holdout_df) > 0:
        print(f"Evaluating on holdout set ({len(holdout_df)} samples)...")
        candidate_holdout = _compute_metrics(candidate_model, candidate_vec, holdout_df)
        report["candidate_holdout"] = candidate_holdout

        if prod_model is not None and prod_vec is not None:
            prod_holdout = _compute_metrics(prod_model, prod_vec, holdout_df)
            report["production_holdout"] = prod_holdout

            f1_drop = prod_holdout["f1"] - candidate_holdout["f1"]
            fpr_increase = candidate_holdout["fpr"] - prod_holdout["fpr"]

            gate_f1 = {"gate": "F1 drop", "threshold": MAX_F1_DROP,
                       "actual": round(f1_drop, 4), "passed": f1_drop <= MAX_F1_DROP}
            gate_fpr = {"gate": "FPR increase", "threshold": MAX_FPR_INCREASE,
                        "actual": round(fpr_increase, 4), "passed": fpr_increase <= MAX_FPR_INCREASE}

            report["gates"].extend([gate_f1, gate_fpr])
            if not gate_f1["passed"]:
                failures.append(f"F1 dropped {f1_drop:.4f} (max {MAX_F1_DROP})")
            if not gate_fpr["passed"]:
                failures.append(f"FPR increased {fpr_increase:.4f} (max {MAX_FPR_INCREASE})")
    else:
        print("WARNING: No holdout set found — skipping holdout evaluation.")

    # Evaluate on canary set
    if canary_df is not None and len(canary_df) > 0:
        print(f"Evaluating on canary set ({len(canary_df)} samples)...")
        candidate_canary = _compute_metrics(candidate_model, candidate_vec, canary_df)
        report["candidate_canary"] = candidate_canary

        gate_canary = {
            "gate": "Canary accuracy",
            "threshold": MIN_CANARY_ACCURACY,
            "actual": round(candidate_canary["accuracy"], 4),
            "passed": candidate_canary["accuracy"] >= MIN_CANARY_ACCURACY,
        }
        report["gates"].append(gate_canary)
        if not gate_canary["passed"]:
            failures.append(
                f"Canary accuracy {candidate_canary['accuracy']:.4f} "
                f"(min {MIN_CANARY_ACCURACY})"
            )
    else:
        print("WARNING: No canary set found — skipping canary evaluation.")

    # Verdict
    if failures:
        report["verdict"] = "FAIL"
        report["failures"] = failures

    # Save report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'=' * 55}")
    print(f"Shadow Evaluation Summary")
    print(f"{'=' * 55}")
    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  [{status}] {gate['gate']}: {gate['actual']} (threshold: {gate['threshold']})")
    print(f"\n  Verdict: {report['verdict']}")
    print(f"  Report:  {report_path}")
    print(f"{'=' * 55}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Shadow evaluation of candidate model.")
    parser.add_argument(
        "--candidate-data", required=True,
        help="Path to candidate training data CSV.",
    )
    parser.add_argument(
        "--holdout", default=DEFAULT_HOLDOUT,
        help=f"Holdout evaluation directory (default: {DEFAULT_HOLDOUT})",
    )
    parser.add_argument(
        "--canary", default=DEFAULT_CANARY,
        help=f"Canary evaluation directory (default: {DEFAULT_CANARY})",
    )
    parser.add_argument(
        "--report", default=DEFAULT_REPORT,
        help=f"JSON report output path (default: {DEFAULT_REPORT})",
    )
    args = parser.parse_args()

    report = shadow_evaluate(args.candidate_data, args.holdout, args.canary, args.report)
    sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
