#!/usr/bin/env python3
"""Train the Na0S worm corpus classifier on Morris II and existing data.

Loads the combined JSONL produced by ``ingest_morris2.py``, optionally merges
existing Na0S HF training data, then trains ``_WormCorpusClassifier`` and
evaluates both the corpus classifier and the full ``WormSignatureDetector``
pipeline.

Usage::

    python -m scripts.train_worm_classifier [--data data/raw/morris2/morris2_combined.jsonl] [--test-split 0.2]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

# Ensure src/ is on the path so na0s imports work when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from na0s.worm_detector import WormSignatureDetector, _WormCorpusClassifier

DEFAULT_DATA = os.path.join(ROOT, "data", "raw", "morris2", "morris2_combined.jsonl")
DEFAULT_HF_DIR = os.path.join(ROOT, "data", "raw", "hf")
DEFAULT_TEST_SPLIT = 0.2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file, returning list of dicts with text + label."""
    records = []
    if not os.path.isfile(path):
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (obj.get("text") or "").strip()
            label = obj.get("label")
            if text and label is not None:
                records.append({"text": text, "label": int(label)})
    return records


def _load_hf_data(hf_dir: str) -> List[Dict]:
    """Load all JSONL files from the HF data directory."""
    records = []
    if not os.path.isdir(hf_dir):
        return records
    for fname in sorted(os.listdir(hf_dir)):
        if fname.endswith(".jsonl"):
            records.extend(_load_jsonl(os.path.join(hf_dir, fname)))
    return records


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def _stratified_split(
    texts: List[str],
    labels: List[int],
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Stratified train/test split.

    Uses sklearn if available, otherwise falls back to manual stratification.
    """
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_ratio, random_state=seed, stratify=labels,
        )
        return X_train, y_train, X_test, y_test
    except ImportError:
        pass

    # Manual stratified split
    rng = random.Random(seed)
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for t, l in zip(texts, labels):
        by_class.setdefault(l, []).append((t, l))

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for cls, items in by_class.items():
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio))
        for t, l in items[:n_test]:
            test_texts.append(t)
            test_labels.append(l)
        for t, l in items[n_test:]:
            train_texts.append(t)
            train_labels.append(l)

    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute precision, recall, F1 for class 1 (worm)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _print_metrics(name: str, metrics: Dict[str, float]) -> None:
    """Print evaluation results in a table."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-' * 30}")
    print(f"  {'Precision':<20} {metrics['precision']:>10.4f}")
    print(f"  {'Recall':<20} {metrics['recall']:>10.4f}")
    print(f"  {'F1':<20} {metrics['f1']:>10.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Neg    Pos")
    print(f"  Actual Neg  {metrics['tn']:>6}  {metrics['fp']:>6}")
    print(f"  Actual Pos  {metrics['fn']:>6}  {metrics['tp']:>6}")


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    data_path: str = DEFAULT_DATA,
    hf_dir: str = DEFAULT_HF_DIR,
    test_split: float = DEFAULT_TEST_SPLIT,
) -> Dict:
    """Load data, train classifier, evaluate, return metrics."""
    # Load data
    print("Loading Morris II data ...")
    records = _load_jsonl(data_path)
    print(f"  {len(records)} records from {data_path}")

    hf_records = _load_hf_data(hf_dir)
    if hf_records:
        print(f"  {len(hf_records)} records from HF data at {hf_dir}")
        records.extend(hf_records)

    if not records:
        print("ERROR: No training data found.", file=sys.stderr)
        return {}

    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    label_counts = Counter(labels)
    print(f"\n  Total: {len(texts)} samples  |  Worm: {label_counts.get(1, 0)}  |  Benign: {label_counts.get(0, 0)}")

    if len(set(labels)) < 2:
        print("ERROR: Need at least 2 classes (worm + benign) for training.", file=sys.stderr)
        return {}

    # Split
    X_train, y_train, X_test, y_test = _stratified_split(texts, labels, test_split)
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Train corpus classifier
    print("\nTraining _WormCorpusClassifier ...")
    classifier = _WormCorpusClassifier()
    classifier.train(X_train, y_train)
    print("  Training complete.")

    # Evaluate corpus classifier
    print("\nEvaluating corpus classifier on test set ...")
    y_pred_corpus = []
    for text in X_test:
        prob = classifier.predict_proba(text)
        y_pred_corpus.append(1 if prob >= 0.5 else 0)

    corpus_metrics = _compute_metrics(y_test, y_pred_corpus)
    _print_metrics("Corpus Classifier (TF-IDF + LR)", corpus_metrics)

    # Evaluate full WormSignatureDetector pipeline
    print("\nEvaluating full WormSignatureDetector pipeline on test set ...")
    detector = WormSignatureDetector()
    y_pred_pipeline = []
    for text in X_test:
        result = detector.scan(text)
        y_pred_pipeline.append(1 if result.get("is_worm", False) else 0)
        detector.reset_history()

    pipeline_metrics = _compute_metrics(y_test, y_pred_pipeline)
    _print_metrics("Full WormSignatureDetector Pipeline", pipeline_metrics)

    return {
        "corpus_classifier": corpus_metrics,
        "pipeline": pipeline_metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Na0S worm corpus classifier on Morris II data.",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"Path to combined JSONL file (default: {DEFAULT_DATA}).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=DEFAULT_TEST_SPLIT,
        help=f"Test split ratio (default: {DEFAULT_TEST_SPLIT}).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        results = train_and_evaluate(data_path=args.data, test_split=args.test_split)
        if not results:
            return 1
        print("\nDone.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
