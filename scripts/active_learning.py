#!/usr/bin/env python3
"""Active learning sample selection for human review.

Loads the current trained model and an unlabeled data pool, then selects
the most informative samples for human labeling.  Two selection strategies
are combined:

1. **Uncertainty sampling** — samples where the model is least certain
   (predicted probability closest to 0.5, i.e., in the 0.4-0.6 band).
2. **Rule-model disagreement** — samples where the model is confident
   but simple structural rules disagree with the model's prediction.

Output is a CSV with columns: text, model_score, model_label,
suggested_priority, selection_reason.

Usage::

    python scripts/active_learning.py
    python scripts/active_learning.py --pool data/unlabeled.csv --top-n 100
    python scripts/active_learning.py --model-dir data/processed --output review.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_POOL = os.path.join(ROOT, "data", "staging", "unlabeled_pool.csv")
DEFAULT_MODEL_DIR = os.path.join(ROOT, "data", "processed")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "staging", "active_learning_review.csv")
DEFAULT_TOP_N = 50

# Uncertainty band — samples with confidence in [0.4, 0.6] are most uncertain
UNCERTAINTY_LOW = 0.4
UNCERTAINTY_HIGH = 0.6

# Confidence threshold for "model is confident" in disagreement check
CONFIDENT_THRESHOLD = 0.75

# Simple rule patterns that suggest injection
INJECTION_PHRASES = [
    "ignore previous", "ignore all", "disregard", "bypass",
    "new instructions", "system prompt", "you are now",
    "forget everything", "override", "jailbreak",
    "do not follow", "pretend you are", "act as",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based heuristic
# ---------------------------------------------------------------------------

def rule_predicts_injection(text: str) -> bool:
    """Return True if simple rules suggest the text is an injection."""
    lower = text.lower()
    return any(phrase in lower for phrase in INJECTION_PHRASES)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: str) -> Any:
    """Load the trained model from *model_dir*.

    Returns the model object (sklearn estimator with predict_proba).
    """
    from na0s.safe_pickle import safe_load

    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = safe_load(model_path)
    logger.info("Loaded model from %s", model_path)
    return model


def load_features_pipeline(model_dir: str) -> Any:
    """Load the feature extraction pipeline.

    Returns the features object (sklearn transformer with transform).
    """
    from na0s.safe_pickle import safe_load

    features_path = os.path.join(model_dir, "features.pkl")
    if not os.path.isfile(features_path):
        raise FileNotFoundError(f"Features pipeline not found: {features_path}")

    pipeline = safe_load(features_path)
    logger.info("Loaded features pipeline from %s", features_path)
    return pipeline


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------

def load_pool(pool_path: str) -> list[str]:
    """Load unlabeled text samples from a CSV file.

    Expects a CSV with at least a 'text' column.  Returns a list of
    text strings.
    """
    if not os.path.isfile(pool_path):
        raise FileNotFoundError(f"Unlabeled pool not found: {pool_path}")

    texts: list[str] = []
    with open(pool_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"Pool CSV must have a 'text' column, found: {reader.fieldnames}")
        for row in reader:
            text = (row.get("text") or "").strip()
            if text:
                texts.append(text)

    logger.info("Loaded %d samples from %s", len(texts), pool_path)
    return texts


# ---------------------------------------------------------------------------
# Scoring & selection
# ---------------------------------------------------------------------------

def score_samples(
    texts: list[str],
    model: Any,
    features_pipeline: Any | None = None,
) -> np.ndarray:
    """Score each text with the model's predicted probability of injection.

    Returns an array of shape (n_samples,) with P(injection).
    """
    if features_pipeline is not None:
        X = features_pipeline.transform(texts)
    else:
        # Assume model can handle raw text (e.g., pipeline model)
        X = texts

    probas = model.predict_proba(X)
    # Column 1 is P(injection) for binary classifiers
    if probas.ndim == 2 and probas.shape[1] >= 2:
        return probas[:, 1]
    return probas.ravel()


def select_uncertain(
    texts: list[str],
    scores: np.ndarray,
    top_n: int,
    low: float = UNCERTAINTY_LOW,
    high: float = UNCERTAINTY_HIGH,
) -> list[dict]:
    """Select samples in the uncertainty band [low, high].

    Returns up to *top_n* samples sorted by proximity to 0.5.
    """
    mask = (scores >= low) & (scores <= high)
    indices = np.where(mask)[0]

    # Sort by distance to 0.5 (ascending — most uncertain first)
    distances = np.abs(scores[indices] - 0.5)
    sorted_order = np.argsort(distances)
    selected_indices = indices[sorted_order][:top_n]

    results = []
    for idx in selected_indices:
        results.append({
            "text": texts[idx],
            "model_score": float(scores[idx]),
            "model_label": int(scores[idx] >= 0.5),
            "suggested_priority": "high",
            "selection_reason": "uncertainty",
        })
    return results


def select_disagreements(
    texts: list[str],
    scores: np.ndarray,
    top_n: int,
    confident_threshold: float = CONFIDENT_THRESHOLD,
) -> list[dict]:
    """Select samples where model is confident but rules disagree.

    Returns up to *top_n* samples sorted by model confidence (descending).
    """
    results = []
    for idx, (text, score) in enumerate(zip(texts, scores)):
        model_label = int(score >= 0.5)
        confidence = score if model_label == 1 else (1.0 - score)

        if confidence < confident_threshold:
            continue

        rule_label = 1 if rule_predicts_injection(text) else 0
        if rule_label == model_label:
            continue

        results.append({
            "text": text,
            "model_score": float(score),
            "model_label": model_label,
            "suggested_priority": "critical",
            "selection_reason": "rule_disagreement",
            "_confidence": confidence,
        })

    # Sort by confidence descending (most confident disagreements first)
    results.sort(key=lambda r: r["_confidence"], reverse=True)
    for r in results:
        del r["_confidence"]

    return results[:top_n]


def select_samples(
    texts: list[str],
    scores: np.ndarray,
    top_n: int,
) -> list[dict]:
    """Combine uncertainty and disagreement selection.

    Returns up to 2 * top_n samples (top_n from each strategy), deduplicated.
    """
    uncertain = select_uncertain(texts, scores, top_n)
    disagreements = select_disagreements(texts, scores, top_n)

    # Merge, dedup by text
    seen: set[str] = set()
    combined: list[dict] = []

    # Disagreements first (higher priority)
    for sample in disagreements + uncertain:
        if sample["text"] not in seen:
            seen.add(sample["text"])
            combined.append(sample)

    return combined


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_review_csv(samples: list[dict], output_path: str) -> None:
    """Write selected samples to CSV for human review."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = ["text", "model_score", "model_label", "suggested_priority", "selection_reason"]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({k: sample[k] for k in fieldnames})

    logger.info("Wrote %d samples to %s", len(samples), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select most informative samples for human review (active learning).",
    )
    parser.add_argument(
        "--pool",
        default=DEFAULT_POOL,
        help=f"Path to unlabeled pool CSV (default: {DEFAULT_POOL}).",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing model.pkl and features.pkl (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of samples to select per strategy (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to write review CSV (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        texts = load_pool(args.pool)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        model = load_model(args.model_dir)
        features_pipeline = load_features_pipeline(args.model_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    scores = score_samples(texts, model, features_pipeline)
    selected = select_samples(texts, scores, args.top_n)

    if not selected:
        logger.warning("No samples selected — pool may be too small or model too confident.")
        return 0

    write_review_csv(selected, args.output)
    print(f"Selected {len(selected)} samples for review → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
