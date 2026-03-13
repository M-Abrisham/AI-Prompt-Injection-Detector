#!/usr/bin/env python3
"""Benchmark embedding models for prompt-injection classification.

Compares candidate sentence-transformer models by training a LogisticRegression
classifier on each model's embeddings and evaluating accuracy, F1, and AUC.

Usage:
    PYTHONPATH=src:. python scripts/benchmark_embeddings.py
    PYTHONPATH=src:. python scripts/benchmark_embeddings.py --sample-size 5000

Results are printed as a comparison table and saved to
``data/processed/embedding_benchmark.json``.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure Na0S package is importable when running from the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Candidate models to benchmark
# ---------------------------------------------------------------------------
CANDIDATE_MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
]

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
RESULTS_PATH = os.path.join(
    str(_REPO_ROOT), "data", "processed", "embedding_benchmark.json",
)

# ---------------------------------------------------------------------------
# Graceful degradation for sentence-transformers
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False


def _load_data(sample_size=None):
    """Load combined_data.csv and optionally subsample (stratified).

    Parameters
    ----------
    sample_size : int or None
        Maximum number of samples.  If None or >= dataset size, use all.

    Returns
    -------
    tuple[list[str], numpy.ndarray]
        ``(texts, labels)`` where labels are 0 (safe) or 1 (malicious).
    """
    import pandas as pd
    data_path = os.path.join(str(_REPO_ROOT), "data", "processed", "combined_data.csv")
    df = pd.read_csv(data_path)
    df["text"] = df["text"].fillna("").astype(str)

    if sample_size is not None and len(df) > sample_size:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df, train_size=sample_size, random_state=42,
            stratify=df["label"],
        )
        df = df.reset_index(drop=True)

    texts = df["text"].tolist()
    labels = df["label"].values
    return texts, labels


def benchmark_model(model_name, texts, labels, random_state=42):
    """Encode texts with a model, train LogReg, return metrics.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for sentence-transformers.
    texts : list[str]
        Input texts to encode.
    labels : numpy.ndarray
        Binary labels (0=safe, 1=malicious).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Benchmark result with keys: model, accuracy, f1, auc, embed_dim,
        encode_time_s, train_time_s.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    # Encode
    t0 = time.time()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    encode_time = time.time() - t0

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=random_state,
        stratify=labels,
    )

    # Train LogisticRegression
    t1 = time.time()
    clf = LogisticRegression(
        max_iter=10000,
        class_weight="balanced",
        C=1.0,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - t1

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        auc = 0.0  # single-class edge case

    return {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "embed_dim": int(embeddings.shape[1]),
        "encode_time_s": round(encode_time, 2),
        "train_time_s": round(train_time, 2),
        "n_samples": len(texts),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def run_benchmark(sample_size=None, models=None):
    """Run full benchmark across candidate models.

    Parameters
    ----------
    sample_size : int or None
        Maximum number of samples to use.
    models : list[str] or None
        Model list to benchmark. Defaults to CANDIDATE_MODELS.

    Returns
    -------
    list[dict]
        List of benchmark results, one per model.
    """
    if not _HAS_SENTENCE_TRANSFORMERS:
        print("ERROR: sentence-transformers is not installed.")
        print("Install with: pip install sentence-transformers")
        return []

    if models is None:
        models = CANDIDATE_MODELS

    print("Loading data...")
    texts, labels = _load_data(sample_size=sample_size)
    n_safe = int((labels == 0).sum())
    n_mal = int((labels == 1).sum())
    print("  Samples: {0} (safe={1}, malicious={2})".format(
        len(texts), n_safe, n_mal,
    ))

    results = []
    for i, model_name in enumerate(models):
        print("\n{'='*60}")
        print("[{0}/{1}] Benchmarking: {2}".format(
            i + 1, len(models), model_name,
        ))
        print("=" * 60)

        try:
            result = benchmark_model(model_name, texts, labels)
            results.append(result)
            print("  Accuracy: {0:.2%}  F1: {1:.2%}  AUC: {2:.4f}".format(
                result["accuracy"], result["f1"], result["auc"],
            ))
        except Exception as exc:
            print("  FAILED: {0}".format(exc))
            results.append({
                "model": model_name,
                "error": str(exc),
            })

    return results


def print_comparison_table(results):
    """Print a formatted comparison table of benchmark results.

    Parameters
    ----------
    results : list[dict]
        Benchmark results from ``run_benchmark()``.
    """
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 80)
    print("EMBEDDING MODEL BENCHMARK COMPARISON")
    print("=" * 80)
    header = "{0:<30} {1:<10} {2:<10} {3:<10} {4:<8} {5:<10}".format(
        "Model", "Accuracy", "F1", "AUC", "Dim", "Enc(s)",
    )
    print(header)
    print("-" * 80)

    for r in results:
        if "error" in r:
            print("{0:<30} FAILED: {1}".format(r["model"], r["error"]))
            continue
        print("{0:<30} {1:<10.2%} {2:<10.2%} {3:<10.4f} {4:<8} {5:<10.1f}".format(
            r["model"],
            r["accuracy"],
            r["f1"],
            r["auc"],
            r["embed_dim"],
            r["encode_time_s"],
        ))

    # Highlight winner
    successful = [r for r in results if "error" not in r]
    if successful:
        best = max(successful, key=lambda r: r["f1"])
        print("\nBest model by F1: {0} (F1={1:.2%})".format(
            best["model"], best["f1"],
        ))


def save_results(results, path=None):
    """Save benchmark results to JSON.

    Parameters
    ----------
    results : list[dict]
        Benchmark results from ``run_benchmark()``.
    path : str or None
        Output path. Defaults to RESULTS_PATH.
    """
    if path is None:
        path = RESULTS_PATH

    os.makedirs(os.path.dirname(path), exist_ok=True)

    output = {
        "benchmark": "embedding_model_comparison",
        "candidate_models": CANDIDATE_MODELS,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to {0}".format(path))


def get_results_schema():
    """Return the expected JSON schema for benchmark results.

    Returns
    -------
    dict
        JSON-schema-like dictionary describing the output format.
    """
    return {
        "type": "object",
        "required": ["benchmark", "candidate_models", "results", "timestamp"],
        "properties": {
            "benchmark": {"type": "string"},
            "candidate_models": {
                "type": "array",
                "items": {"type": "string"},
            },
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["model"],
                    "properties": {
                        "model": {"type": "string"},
                        "accuracy": {"type": "number"},
                        "f1": {"type": "number"},
                        "auc": {"type": "number"},
                        "embed_dim": {"type": "integer"},
                        "encode_time_s": {"type": "number"},
                        "train_time_s": {"type": "number"},
                        "n_samples": {"type": "integer"},
                        "n_train": {"type": "integer"},
                        "n_test": {"type": "integer"},
                    },
                },
            },
            "timestamp": {"type": "string"},
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models for prompt-injection detection",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Maximum number of samples to use (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: data/processed/embedding_benchmark.json)",
    )
    args = parser.parse_args()

    results = run_benchmark(sample_size=args.sample_size)

    if results:
        print_comparison_table(results)
        save_results(results, path=args.output)
