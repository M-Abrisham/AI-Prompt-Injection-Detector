#!/usr/bin/env python3
"""Cleanlab label quality audit for Na0S training data.

Uses Confident Learning to detect potentially mislabeled samples in the
combined training dataset.  Outputs a CSV of flagged rows with quality
scores so they can be reviewed and optionally quarantined.

Requirements:
    pip install cleanlab>=2.6.0

Usage::

    python scripts/cleanlab_audit.py
    python scripts/cleanlab_audit.py --threshold 0.4 --output data/staging/label_issues.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    from cleanlab.rank import get_label_quality_scores
except ImportError:
    get_label_quality_scores = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUT = os.path.join(ROOT, "data", "processed", "combined_data.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "staging", "label_issues.csv")
DEFAULT_THRESHOLD = 0.5

# Minimum samples required to run the audit meaningfully
_MIN_SAMPLES = 100


def _load_model():
    """Load the trained classifier and TF-IDF vectorizer."""
    from na0s.safe_pickle import safe_load

    model_path = os.path.join(ROOT, "data", "processed", "model.pkl")
    vec_path = os.path.join(ROOT, "data", "processed", "tfidf_vectorizer.pkl")

    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found: {model_path}")
        print("       Run scripts/model.py first.")
        sys.exit(1)
    if not os.path.isfile(vec_path):
        print(f"ERROR: Vectorizer not found: {vec_path}")
        print("       Run scripts/features.py first.")
        sys.exit(1)

    return safe_load(model_path), safe_load(vec_path)


def audit(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Run Cleanlab label quality audit.

    Returns a DataFrame of flagged samples (quality_score < *threshold*).
    """
    if get_label_quality_scores is None:
        print("ERROR: cleanlab is not installed.")
        print("       Install it with: pip install 'cleanlab>=2.6.0'")
        sys.exit(1)

    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Load data
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    df["text"] = df["text"].fillna("").astype(str)

    if len(df) < _MIN_SAMPLES:
        print(f"ERROR: Only {len(df)} samples — need at least {_MIN_SAMPLES}.")
        sys.exit(1)

    labels = df["label"].values.astype(int)

    # Load model and compute predicted probabilities
    print("Loading model and computing predictions...")
    model, vectorizer = _load_model()
    X = vectorizer.transform(df["text"])
    pred_probs = model.predict_proba(X)

    # Compute label quality scores
    print("Computing label quality scores...")
    quality_scores = get_label_quality_scores(
        labels=labels,
        pred_probs=pred_probs,
    )

    df["quality_score"] = quality_scores

    # Suggest flipped labels for low-quality rows
    suggested = pred_probs.argmax(axis=1)
    df["suggested_label"] = suggested

    # Filter to flagged rows
    flagged = df[df["quality_score"] < threshold].copy()
    flagged = flagged.sort_values("quality_score").reset_index(drop=True)

    # Select output columns
    out_cols = ["text", "label", "suggested_label", "quality_score"]
    if "technique_id" in flagged.columns:
        out_cols.append("technique_id")
    flagged = flagged[out_cols]
    flagged = flagged.rename(columns={"label": "current_label"})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    flagged.to_csv(output_path, index=False)

    # Report
    n_flagged = len(flagged)
    n_flipped = (flagged["current_label"] != flagged["suggested_label"]).sum()
    print(f"\n{'=' * 50}")
    print(f"Cleanlab Audit Summary")
    print(f"{'=' * 50}")
    print(f"  Total samples:    {len(df)}")
    print(f"  Threshold:        {threshold}")
    print(f"  Flagged:          {n_flagged} ({n_flagged / len(df) * 100:.1f}%)")
    print(f"  Label flips:      {n_flipped}")
    print(f"  Mean quality:     {quality_scores.mean():.4f}")
    print(f"  Median quality:   {np.median(quality_scores):.4f}")
    print(f"  Output:           {output_path}")
    print(f"{'=' * 50}")

    return flagged


def main():
    parser = argparse.ArgumentParser(
        description="Audit training data for label quality issues using Cleanlab."
    )
    parser.add_argument(
        "--input", "-i", default=DEFAULT_INPUT,
        help=f"Path to combined CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"Path to write flagged rows (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
        help=f"Quality score threshold for flagging (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()
    audit(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
