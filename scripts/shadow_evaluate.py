#!/usr/bin/env python3
"""Shadow evaluation — compare a candidate model against production before promotion.

Loads both the current production model and a candidate model, runs them on the
same holdout/evaluation dataset, and compares metrics side-by-side.  A gate
check determines whether the candidate is safe to promote.

Exit codes:
    0 = PASS — candidate model meets all promotion gates
    1 = FAIL — candidate model regresses on one or more gates

Usage::

    python scripts/shadow_evaluate.py --candidate models/candidate/
    python scripts/shadow_evaluate.py --candidate models/candidate/ --holdout data/canary/canary_eval.csv
    python scripts/shadow_evaluate.py --candidate models/candidate/ --holdout data/holdout/ --output models/shadow_results.json
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_HOLDOUT = os.path.join(ROOT, "data", "canary", "canary_eval.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "models", "shadow_results.json")

# ── Promotion gates ──────────────────────────────────────────────────
# Candidate must meet ALL criteria to pass:
GATE_MAX_FPR_INCREASE = 0.01   # max 1% FPR increase over production
GATE_MAX_RECALL_DROP = 0.005   # max 0.5% recall drop from production


# ── Data loading ─────────────────────────────────────────────────────

def load_eval_dataset(path: str) -> pd.DataFrame:
    """Load evaluation data from a CSV file or a directory of CSVs/JSONLs.

    Returns a DataFrame with at least ``text`` and ``label`` columns.
    Raises ``FileNotFoundError`` if nothing loadable is found.
    """
    if os.path.isfile(path):
        if path.endswith(".jsonl"):
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_csv(path)
    elif os.path.isdir(path):
        frames: list[pd.DataFrame] = []
        for p in sorted(globmod.glob(os.path.join(path, "*.csv"))):
            frames.append(pd.read_csv(p))
        for p in sorted(globmod.glob(os.path.join(path, "*.jsonl"))):
            frames.append(pd.read_json(p, lines=True))
        if not frames:
            raise FileNotFoundError(f"No CSV/JSONL files found in {path}")
        df = pd.concat(frames, ignore_index=True)
    else:
        raise FileNotFoundError(f"Holdout path not found: {path}")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset must contain 'text' and 'label' columns, got: {list(df.columns)}")

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


# ── Model loading ────────────────────────────────────────────────────

def _load_model_pair(directory: str):
    """Load a (model, vectorizer) pair from a directory.

    Looks for ``model.pkl`` and ``tfidf_vectorizer.pkl`` inside *directory*.
    Returns ``(model, vectorizer)`` or raises ``FileNotFoundError``.
    """
    from na0s.safe_pickle import safe_load

    model_path = os.path.join(directory, "model.pkl")
    vec_path = os.path.join(directory, "tfidf_vectorizer.pkl")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model.pkl not found in {directory}")
    if not os.path.isfile(vec_path):
        raise FileNotFoundError(f"tfidf_vectorizer.pkl not found in {directory}")

    return safe_load(model_path), safe_load(vec_path)


def load_production_model():
    """Load the current production model from ``data/processed/``."""
    prod_dir = os.path.join(ROOT, "data", "processed")
    return _load_model_pair(prod_dir)


def load_candidate_model(candidate_path: str):
    """Load a candidate model from the given directory."""
    return _load_model_pair(candidate_path)


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics from ground truth and predictions.

    Returns a dict with accuracy, precision, recall, f1, fpr, fnr, and
    sample counts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    safe_mask = y_true == 0
    mal_mask = y_true == 1

    n_safe = int(safe_mask.sum())
    n_mal = int(mal_mask.sum())

    # FPR = false positives / actual negatives
    fpr = float(y_pred[safe_mask].sum() / n_safe) if n_safe > 0 else 0.0
    # FNR = false negatives / actual positives
    fnr = float((1 - y_pred[mal_mask]).sum() / n_mal) if n_mal > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "n_samples": int(len(y_true)),
        "n_safe": n_safe,
        "n_malicious": n_mal,
    }


def predict_with_model(model, vectorizer, texts: pd.Series) -> np.ndarray:
    """Run model prediction on a Series of text strings."""
    X = vectorizer.transform(texts)
    return model.predict(X)


# ── Gate logic ───────────────────────────────────────────────────────

def check_gates(prod_metrics: dict, cand_metrics: dict) -> list[dict]:
    """Evaluate promotion gates.  Returns a list of gate result dicts.

    Each dict has keys: gate, threshold, actual, passed.
    """
    gates = []

    # Gate 1: F1 must not regress
    f1_delta = cand_metrics["f1"] - prod_metrics["f1"]
    gates.append({
        "gate": "F1 no regression",
        "threshold": 0.0,
        "actual": round(f1_delta, 6),
        "passed": cand_metrics["f1"] >= prod_metrics["f1"],
    })

    # Gate 2: FPR increase <= 1%
    fpr_increase = cand_metrics["fpr"] - prod_metrics["fpr"]
    gates.append({
        "gate": "FPR increase",
        "threshold": GATE_MAX_FPR_INCREASE,
        "actual": round(fpr_increase, 6),
        "passed": round(fpr_increase, 9) <= GATE_MAX_FPR_INCREASE,
    })

    # Gate 3: Recall drop <= 0.5%
    recall_drop = prod_metrics["recall"] - cand_metrics["recall"]
    gates.append({
        "gate": "Recall drop",
        "threshold": GATE_MAX_RECALL_DROP,
        "actual": round(recall_drop, 6),
        "passed": round(recall_drop, 9) <= GATE_MAX_RECALL_DROP,
    })

    return gates


# ── Disagreement analysis ────────────────────────────────────────────

def find_disagreements(
    texts: pd.Series,
    y_true: np.ndarray,
    y_prod: np.ndarray,
    y_cand: np.ndarray,
    max_samples: int = 50,
) -> list[dict]:
    """Find samples where production and candidate models disagree.

    Returns a list of dicts with text snippet, true label, and both
    predictions, capped at *max_samples*.
    """
    disagree_mask = y_prod != y_cand
    indices = np.where(disagree_mask)[0]

    samples = []
    for idx in indices[:max_samples]:
        samples.append({
            "text_snippet": str(texts.iloc[idx])[:200],
            "true_label": int(y_true[idx]),
            "production_pred": int(y_prod[idx]),
            "candidate_pred": int(y_cand[idx]),
        })
    return samples


# ── Comparison table ─────────────────────────────────────────────────

def format_comparison_table(prod_metrics: dict, cand_metrics: dict) -> str:
    """Build a side-by-side ASCII comparison table."""
    lines = []
    lines.append(f"{'Metric':<15s} {'Production':>12s} {'Candidate':>12s} {'Delta':>12s}")
    lines.append("-" * 55)
    for key in ("accuracy", "precision", "recall", "f1", "fpr", "fnr"):
        pv = prod_metrics[key]
        cv = cand_metrics[key]
        delta = cv - pv
        sign = "+" if delta >= 0 else ""
        lines.append(f"{key:<15s} {pv:>12.6f} {cv:>12.6f} {sign}{delta:>11.6f}")
    lines.append("-" * 55)
    lines.append(f"{'n_samples':<15s} {prod_metrics['n_samples']:>12d} {cand_metrics['n_samples']:>12d}")
    return "\n".join(lines)


# ── Main evaluation ──────────────────────────────────────────────────

def shadow_evaluate(
    candidate_path: str,
    holdout_path: str = DEFAULT_HOLDOUT,
    output_path: str = DEFAULT_OUTPUT,
) -> dict:
    """Run shadow evaluation comparing candidate against production.

    Returns a dict with metrics, gates, verdict, and disagreement samples.
    """
    # Load evaluation data
    print(f"Loading holdout data: {holdout_path}")
    eval_df = load_eval_dataset(holdout_path)
    y_true = eval_df["label"].values.astype(int)
    print(f"  {len(eval_df)} samples ({(y_true == 1).sum()} malicious, {(y_true == 0).sum()} safe)")

    # Load models
    print(f"Loading production model...")
    prod_model, prod_vec = load_production_model()

    print(f"Loading candidate model: {candidate_path}")
    cand_model, cand_vec = load_candidate_model(candidate_path)

    # Run predictions
    print("Running production model predictions...")
    y_prod = predict_with_model(prod_model, prod_vec, eval_df["text"])

    print("Running candidate model predictions...")
    y_cand = predict_with_model(cand_model, cand_vec, eval_df["text"])

    # Compute metrics
    prod_metrics = compute_metrics(y_true, y_prod)
    cand_metrics = compute_metrics(y_true, y_cand)

    # Gate check
    gates = check_gates(prod_metrics, cand_metrics)
    failures = [g for g in gates if not g["passed"]]
    verdict = "FAIL" if failures else "PASS"

    # Disagreement analysis
    disagreements = find_disagreements(eval_df["text"], y_true, y_prod, y_cand)

    # Build report
    report = {
        "verdict": verdict,
        "production_metrics": prod_metrics,
        "candidate_metrics": cand_metrics,
        "gates": gates,
        "failures": [g["gate"] for g in failures],
        "disagreements": disagreements,
        "n_disagreements": int((y_prod != y_cand).sum()),
        "holdout_path": holdout_path,
        "candidate_path": candidate_path,
    }

    # Save report
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    table = format_comparison_table(prod_metrics, cand_metrics)
    print(f"\n{'=' * 55}")
    print("Shadow Evaluation — Side-by-Side Comparison")
    print(f"{'=' * 55}")
    print(table)
    print(f"\n{'=' * 55}")
    print("Gate Results")
    print(f"{'=' * 55}")
    for g in gates:
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  [{status}] {g['gate']}: {g['actual']} (threshold: {g['threshold']})")
    print(f"\n  Verdict: {verdict}")
    if disagreements:
        print(f"  Disagreements: {report['n_disagreements']} total ({len(disagreements)} shown)")
    print(f"  Report saved: {output_path}")
    print(f"{'=' * 55}")

    return report


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shadow evaluation: compare candidate model against production before promotion.",
    )
    parser.add_argument(
        "--candidate", required=True,
        help="Path to directory containing candidate model.pkl and tfidf_vectorizer.pkl.",
    )
    parser.add_argument(
        "--holdout", default=DEFAULT_HOLDOUT,
        help=f"Path to holdout CSV/JSONL file or directory (default: {DEFAULT_HOLDOUT})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"JSON report output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    report = shadow_evaluate(args.candidate, args.holdout, args.output)
    sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
