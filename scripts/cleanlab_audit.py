#!/usr/bin/env python3
"""Cleanlab label quality audit for Na0S training data.

Uses Confident Learning to detect potentially mislabeled samples in the
combined training dataset.  Supports two modes:

  1. Cross-validated mode (default): trains a LogisticRegression + TF-IDF
     pipeline with cross-validation to obtain out-of-sample predicted
     probabilities, then uses cleanlab to find label issues.
  2. Pre-trained mode (--use-pretrained): uses the existing Na0S L4
     model/vectorizer to compute predicted probabilities.

Outputs a CSV of flagged rows with quality scores so they can be reviewed
and optionally quarantined.

Requirements:
    pip install cleanlab>=2.6.0

Usage::

    python scripts/cleanlab_audit.py
    python scripts/cleanlab_audit.py --data data/processed/combined_data.csv
    python scripts/cleanlab_audit.py --top 50 --output data/label_issues.csv
    python scripts/cleanlab_audit.py --use-pretrained --threshold 0.4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cleanlab availability -- graceful degradation
# ---------------------------------------------------------------------------
_CLEANLAB_AVAILABLE = False
_CLEANLAB_IMPORT_ERROR: str | None = None

try:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    _CLEANLAB_AVAILABLE = True
except ImportError as exc:
    find_label_issues = None  # type: ignore[assignment]
    get_label_quality_scores = None  # type: ignore[assignment]
    _CLEANLAB_IMPORT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA = os.path.join(ROOT, "data", "processed", "combined_data.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "label_issues.csv")
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_N = 20
DEFAULT_CV_FOLDS = 5

# Minimum samples required to run the audit meaningfully
_MIN_SAMPLES = 100

logger = logging.getLogger("cleanlab_audit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pretrained_model():
    """Load the trained Na0S classifier and TF-IDF vectorizer."""
    from na0s.safe_pickle import safe_load

    model_path = os.path.join(ROOT, "data", "processed", "model.pkl")
    vec_path = os.path.join(ROOT, "data", "processed", "tfidf_vectorizer.pkl")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run scripts/model.py first."
        )
    if not os.path.isfile(vec_path):
        raise FileNotFoundError(
            f"Vectorizer not found: {vec_path}. Run scripts/features.py first."
        )

    return safe_load(model_path), safe_load(vec_path)


def _cross_val_predict_proba(texts: pd.Series, labels: np.ndarray,
                             n_folds: int = DEFAULT_CV_FOLDS) -> np.ndarray:
    """Train LogisticRegression + TF-IDF with CV, return out-of-sample probs.

    This mirrors Na0S's L4 layer approach: TF-IDF features fed into a
    LogisticRegression classifier with balanced class weights.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    logger.info("Building TF-IDF features for cross-validation...")
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(texts)

    pred_probs = np.zeros((len(labels), 2), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, labels), 1):
        logger.info("  Fold %d/%d ...", fold_idx, n_folds)
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        clf.fit(X[train_idx], labels[train_idx])
        pred_probs[val_idx] = clf.predict_proba(X[val_idx])

    return pred_probs


def _truncate_text(text: str, max_len: int = 120) -> str:
    """Truncate text for display, appending ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def load_data(data_path: str) -> pd.DataFrame:
    """Load and minimally clean the dataset CSV.

    Expected columns: text, label (0=safe, 1=malicious).
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Input file not found: {data_path}")

    df = pd.read_csv(data_path)

    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    return df


def find_issues(
    df: pd.DataFrame,
    *,
    use_pretrained: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    n_folds: int = DEFAULT_CV_FOLDS,
) -> pd.DataFrame:
    """Identify label issues in *df* using cleanlab.

    Returns a DataFrame with columns:
        index, text, given_label, suggested_label, confidence

    sorted by confidence ascending (worst offenders first).
    """
    if not _CLEANLAB_AVAILABLE:
        raise ImportError(
            f"cleanlab is not installed ({_CLEANLAB_IMPORT_ERROR}). "
            "Install it with: pip install 'cleanlab>=2.6.0'"
        )

    if len(df) < _MIN_SAMPLES:
        raise ValueError(
            f"Only {len(df)} samples -- need at least {_MIN_SAMPLES}."
        )

    labels = df["label"].values.astype(int)
    texts = df["text"]

    # Obtain predicted probabilities
    if use_pretrained:
        logger.info("Using pre-trained model for predictions...")
        model, vectorizer = _load_pretrained_model()
        X = vectorizer.transform(texts)
        pred_probs = model.predict_proba(X)
    else:
        logger.info(
            "Training cross-validated classifier (%d folds)...", n_folds,
        )
        pred_probs = _cross_val_predict_proba(texts, labels, n_folds=n_folds)

    # Use cleanlab to find label issues
    logger.info("Running cleanlab label issue detection...")
    issue_mask = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    # Also compute per-sample quality scores
    quality_scores = get_label_quality_scores(
        labels=labels,
        pred_probs=pred_probs,
    )

    # Build result dataframe preserving original index
    suggested_labels = pred_probs.argmax(axis=1)

    result = pd.DataFrame({
        "index": df.index,
        "text": df["text"].values,
        "given_label": labels,
        "suggested_label": suggested_labels,
        "confidence": quality_scores,
    })

    # Filter: keep rows flagged by cleanlab OR below threshold
    if isinstance(issue_mask, np.ndarray) and issue_mask.dtype == np.int_:
        # find_label_issues returned ranked indices
        flagged_indices = set(issue_mask)
    else:
        # find_label_issues returned a boolean mask
        flagged_indices = set(np.where(issue_mask)[0])

    below_threshold = set(np.where(quality_scores < threshold)[0])
    all_flagged = flagged_indices | below_threshold

    result = result.loc[result.index.isin(all_flagged)].copy()
    result = result.sort_values("confidence", ascending=True).reset_index(drop=True)

    logger.info("Found %d label issues.", len(result))
    return result


def print_report(
    issues: pd.DataFrame,
    total_samples: int,
    top_n: int = DEFAULT_TOP_N,
) -> str:
    """Format and print a human-readable report. Returns the report string."""
    n_issues = len(issues)
    n_flips = int((issues["given_label"] != issues["suggested_label"]).sum()) if n_issues else 0

    lines = [
        "",
        "=" * 60,
        "Cleanlab Label Quality Audit Report",
        "=" * 60,
        f"  Total samples:       {total_samples}",
        f"  Issues found:        {n_issues} ({n_issues / total_samples * 100:.1f}%)" if total_samples else f"  Issues found:        {n_issues}",
        f"  Suggested flips:     {n_flips}",
    ]

    if n_issues > 0:
        lines.append(f"  Mean confidence:     {issues['confidence'].mean():.4f}")
        lines.append(f"  Min confidence:      {issues['confidence'].min():.4f}")

    lines.append("=" * 60)

    if n_issues > 0:
        display_n = min(top_n, n_issues)
        lines.append(f"\nTop-{display_n} worst offenders:")
        lines.append("-" * 60)
        for i, row in issues.head(display_n).iterrows():
            preview = _truncate_text(row["text"])
            lines.append(
                f"  [{i}] conf={row['confidence']:.4f}  "
                f"given={int(row['given_label'])} -> suggested={int(row['suggested_label'])}"
            )
            lines.append(f"      {preview}")
        lines.append("-" * 60)

    report = "\n".join(lines)
    print(report)
    return report


def audit(
    data_path: str = DEFAULT_DATA,
    output_path: str = DEFAULT_OUTPUT,
    threshold: float = DEFAULT_THRESHOLD,
    top_n: int = DEFAULT_TOP_N,
    use_pretrained: bool = False,
    n_folds: int = DEFAULT_CV_FOLDS,
) -> pd.DataFrame:
    """Run the full Cleanlab label quality audit.

    Returns the issues DataFrame.
    """
    logger.info("Loading data from %s", data_path)
    df = load_data(data_path)
    logger.info("Loaded %d samples.", len(df))

    issues = find_issues(
        df,
        use_pretrained=use_pretrained,
        threshold=threshold,
        n_folds=n_folds,
    )

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    issues.to_csv(output_path, index=False)
    logger.info("Saved %d issues to %s", len(issues), output_path)

    # Print report
    print_report(issues, total_samples=len(df), top_n=top_n)

    return issues


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Audit training data for label quality issues using Cleanlab.",
    )
    parser.add_argument(
        "--data", "-d",
        default=DEFAULT_DATA,
        help=f"Path to combined CSV (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Path to write label issues CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Quality score threshold for flagging (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of worst offenders to display (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use pre-trained Na0S model instead of cross-validated classifier.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help=f"Number of CV folds for cross-validated mode (default: {DEFAULT_CV_FOLDS})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 1 on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _CLEANLAB_AVAILABLE:
        logger.error(
            "cleanlab is not installed (%s). "
            "Install it with: pip install 'cleanlab>=2.6.0'",
            _CLEANLAB_IMPORT_ERROR,
        )
        return 1

    try:
        audit(
            data_path=args.data,
            output_path=args.output,
            threshold=args.threshold,
            top_n=args.top,
            use_pretrained=args.use_pretrained,
            n_folds=args.folds,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
