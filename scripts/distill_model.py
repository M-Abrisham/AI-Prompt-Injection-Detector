"""Knowledge distillation: train a small fast model to mimic ensemble output.

Trains a LogisticRegression on TF-IDF features using soft labels (probability
outputs) from the teacher ensemble (TF-IDF classifier + embedding classifier).
The distilled model is smaller and faster for deployment while retaining most
of the ensemble's accuracy.

Usage:
    PYTHONPATH=src:. python scripts/distill_model.py \
        --tfidf-features data/processed/features.pkl \
        --teacher-predictions data/processed/teacher_preds.npy \
        --output data/processed/distilled_model.pkl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from na0s.integrity.safe_pickle import safe_load, safe_dump

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------
_HAS_SKLEARN = False
_import_error: Optional[str] = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.model_selection import train_test_split

    _HAS_SKLEARN = True
except ImportError as exc:
    _import_error = str(exc)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT = "data/processed/distilled_model.pkl"


# ---------------------------------------------------------------------------
# Core distillation API
# ---------------------------------------------------------------------------

def distill(
    teacher_predictions: np.ndarray,
    tfidf_features: np.ndarray,
    temperature: float = 2.0,
    C: float = 1.0,
    max_iter: int = 5000,
) -> object:
    """Train a student model on soft labels from the teacher ensemble.

    Parameters
    ----------
    teacher_predictions : np.ndarray
        Soft-label probabilities from the teacher (shape: (n_samples,) or
        (n_samples, 2)).  Values in [0, 1] representing P(malicious).
    tfidf_features : np.ndarray
        TF-IDF feature matrix (shape: (n_samples, n_features)).
    temperature : float
        Temperature for softening teacher probabilities.  Higher values
        produce softer distributions (more knowledge transfer).
    C : float
        Regularization parameter for LogisticRegression.
    max_iter : int
        Maximum iterations for LogisticRegression solver.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Fitted student model.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    ValueError
        If inputs have incompatible shapes.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for knowledge distillation. "
            "Install with: pip install scikit-learn"
        )

    # Normalize teacher predictions to 1-D probabilities
    if teacher_predictions.ndim == 2:
        teacher_probs = teacher_predictions[:, 1]
    else:
        teacher_probs = teacher_predictions.ravel()

    if len(teacher_probs) != tfidf_features.shape[0]:
        raise ValueError(
            "Shape mismatch: teacher_predictions has {0} samples but "
            "tfidf_features has {1} rows".format(
                len(teacher_probs), tfidf_features.shape[0]
            )
        )

    # Apply temperature softening
    # For binary classification: soften probabilities toward 0.5
    softened = _soften_probabilities(teacher_probs, temperature)

    # Convert soft labels to hard labels for LogisticRegression
    # but use sample_weight to encode the teacher's confidence
    hard_labels = (softened >= 0.5).astype(int)
    # Confidence = distance from decision boundary (0.5)
    confidence = np.abs(softened - 0.5) * 2  # scale to [0, 1]
    # Minimum weight to avoid zero-weight samples
    sample_weights = np.clip(confidence, 0.1, 1.0)

    logger.info(
        "Distilling: %d samples, %d features, temperature=%.1f, C=%.1f",
        tfidf_features.shape[0], tfidf_features.shape[1], temperature, C,
    )
    logger.info(
        "  Soft label distribution: mean=%.3f, std=%.3f",
        softened.mean(), softened.std(),
    )
    logger.info(
        "  Hard labels: safe=%d, malicious=%d",
        int((hard_labels == 0).sum()), int((hard_labels == 1).sum()),
    )

    student = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    student.fit(tfidf_features, hard_labels, sample_weight=sample_weights)

    return student


def _soften_probabilities(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Soften binary probabilities using temperature scaling.

    Maps probabilities through logit space with temperature division:
        softened = sigmoid(logit(p) / temperature)

    Parameters
    ----------
    probs : np.ndarray
        Probabilities in [0, 1].
    temperature : float
        Temperature > 1 softens (pushes toward 0.5), < 1 sharpens.

    Returns
    -------
    np.ndarray
        Softened probabilities.
    """
    # Clip to avoid log(0) or log(1)
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    logits = np.log(p / (1 - p))  # logit transform
    scaled_logits = logits / temperature
    softened = 1.0 / (1.0 + np.exp(-scaled_logits))  # sigmoid
    return softened


def evaluate_distilled(
    student,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate the distilled model on a test set.

    Parameters
    ----------
    student : fitted sklearn estimator
    X_test : np.ndarray
        TF-IDF test features.
    y_test : np.ndarray
        Ground-truth binary labels.

    Returns
    -------
    dict
        Metrics including accuracy, f1, and classification_report string.
    """
    y_pred = student.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(
        y_test, y_pred, target_names=["Safe", "Malicious"], zero_division=0,
    )
    return {"accuracy": acc, "f1": f1, "report": report}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: train fast student from ensemble teacher",
    )
    parser.add_argument(
        "--tfidf-features", default="data/processed/features.pkl",
        help="Path to pickled TF-IDF features (X, y) tuple",
    )
    parser.add_argument(
        "--teacher-predictions", default="data/processed/teacher_preds.npy",
        help="Path to .npy file with teacher soft labels",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output path for distilled model pickle (default: {0})".format(
            DEFAULT_OUTPUT
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=2.0,
        help="Temperature for softening teacher predictions (default: 2.0)",
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="Regularization parameter (default: 1.0)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation on a held-out split",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)

    if not _HAS_SKLEARN:
        print(
            "ERROR: scikit-learn is required for knowledge distillation.\n"
            "Install with: pip install scikit-learn\n"
            "Import error: {0}".format(_import_error),
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load TF-IDF features
    print("Loading TF-IDF features from {0}".format(args.tfidf_features))
    X, y = safe_load(args.tfidf_features)
    print("  Shape: {0}".format(X.shape))

    # Load teacher predictions
    print("Loading teacher predictions from {0}".format(args.teacher_predictions))
    teacher_preds = np.load(args.teacher_predictions)
    print("  Shape: {0}".format(teacher_preds.shape))

    if args.eval:
        # Split for evaluation
        X_train, X_test, y_train, y_test, tp_train, tp_test = train_test_split(
            X, y, teacher_preds, test_size=0.2, random_state=42, stratify=y,
        )
        student = distill(
            tp_train, X_train,
            temperature=args.temperature, C=args.C,
        )
        metrics = evaluate_distilled(student, X_test, y_test)
        print("\n--- Distilled Model Evaluation ---")
        print("Accuracy: {0:.2%}".format(metrics["accuracy"]))
        print("F1:       {0:.2%}".format(metrics["f1"]))
        print(metrics["report"])
    else:
        student = distill(
            teacher_preds, X,
            temperature=args.temperature, C=args.C,
        )

    # Save distilled model
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    safe_dump(student, args.output)
    print("Distilled model saved to {0}".format(args.output))


if __name__ == "__main__":
    main()
