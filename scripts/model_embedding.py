"""Train a classifier on sentence-transformer embeddings.

Supports LogisticRegression (fast) and a simple MLP (better accuracy).
Uses CalibratedClassifierCV for well-calibrated probabilities.

Usage:
    PYTHONPATH=src:. python src/model_embedding.py
    PYTHONPATH=src:. python src/model_embedding.py mlp        # MLP variant
    PYTHONPATH=src:. python src/model_embedding.py logistic 0  # no calibration
"""

import os
import sys

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from na0s.safe_pickle import safe_dump, safe_load

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_PATH = "data/processed/features_embedding.pkl"
MODEL_PATH = "data/processed/model_embedding.pkl"

# ---------------------------------------------------------------------------
# TF-IDF baseline numbers (for comparison printout)
# ---------------------------------------------------------------------------
TFIDF_ACCURACY = 91.4
TFIDF_FPR = 82.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verify_stratified_split(y_full, y_train, y_test, tolerance=0.02):
    """Verify that train/test splits preserve class distribution.

    Parameters
    ----------
    y_full : array-like
        Full label array before splitting.
    y_train : array-like
        Training labels after splitting.
    y_test : array-like
        Test labels after splitting.
    tolerance : float
        Maximum allowed absolute difference in class proportions
        between the full dataset and each split.

    Returns
    -------
    bool
        True if stratification is maintained within tolerance.

    Raises
    ------
    ValueError
        If the class distribution deviates beyond tolerance.
    """
    full_ratio = np.mean(y_full == 1)
    train_ratio = np.mean(y_train == 1)
    test_ratio = np.mean(y_test == 1)

    train_diff = abs(full_ratio - train_ratio)
    test_diff = abs(full_ratio - test_ratio)

    ok = True
    if train_diff > tolerance:
        ok = False
    if test_diff > tolerance:
        ok = False

    return ok


def _false_positive_rate(y_true, y_pred):
    """FPR = FP / (FP + TN).  Safe=0, Malicious=1."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (fp + tn) == 0:
        return 0.0
    return fp / (fp + tn)


def _evaluate_at_threshold(y_true, y_proba, threshold):
    """Apply a decision threshold and return (accuracy, FPR)."""
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    fpr = _false_positive_rate(y_true, y_pred)
    return acc, fpr


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_embedding_model(model_type="logistic", calibrate=True):
    """Train and evaluate an embedding-based classifier.

    Parameters
    ----------
    model_type : str
        ``"logistic"`` for LogisticRegression (fast, interpretable) or
        ``"mlp"`` for a two-layer MLP (higher capacity).
    calibrate : bool
        If *True*, wrap the base estimator in ``CalibratedClassifierCV``
        with 5-fold isotonic calibration for reliable probabilities.

    Returns
    -------
    clf : fitted estimator (saved to ``data/processed/model_embedding.pkl``)
    """
    # ------------------------------------------------------------------
    # Load embedding features
    # ------------------------------------------------------------------
    print("Loading embedding features from {0}".format(FEATURES_PATH))
    X, y = safe_load(FEATURES_PATH)
    print("  Shape: {0}   Labels: safe={1}, malicious={2}".format(
        X.shape, int((y == 0).sum()), int((y == 1).sum()),
    ))

    # ------------------------------------------------------------------
    # Train / test split (80/20, stratified)
    # ------------------------------------------------------------------
    # Class distribution before split
    total_safe = int((y == 0).sum())
    total_mal = int((y == 1).sum())
    total_n = len(y)
    print("\n  Class distribution (full dataset):")
    print("    Safe:      {0} ({1:.1%})".format(total_safe, total_safe / total_n))
    print("    Malicious: {0} ({1:.1%})".format(total_mal, total_mal / total_n))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Class distribution after split
    train_safe = int((y_train == 0).sum())
    train_mal = int((y_train == 1).sum())
    test_safe = int((y_test == 0).sum())
    test_mal = int((y_test == 1).sum())
    print("\n  Class distribution (train split, n={0}):".format(len(y_train)))
    print("    Safe:      {0} ({1:.1%})".format(train_safe, train_safe / len(y_train)))
    print("    Malicious: {0} ({1:.1%})".format(train_mal, train_mal / len(y_train)))
    print("  Class distribution (test split, n={0}):".format(len(y_test)))
    print("    Safe:      {0} ({1:.1%})".format(test_safe, test_safe / len(y_test)))
    print("    Malicious: {0} ({1:.1%})".format(test_mal, test_mal / len(y_test)))
    print("  Train: {0}  Test: {1}".format(len(y_train), len(y_test)))

    # ------------------------------------------------------------------
    # Build base estimator
    # ------------------------------------------------------------------
    if model_type == "mlp":
        print("\nTraining MLPClassifier (256, 128) ...")
        base = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )
    else:
        print("\nTraining LogisticRegression ...")
        base = LogisticRegression(
            max_iter=10000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        )

    # ------------------------------------------------------------------
    # Optional calibration
    # ------------------------------------------------------------------
    if calibrate:
        print("Wrapping in CalibratedClassifierCV (5-fold isotonic) ...")
        clf = CalibratedClassifierCV(base, cv=5, method="isotonic")
    else:
        clf = base

    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate at default threshold (0.5)
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # P(malicious)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr = _false_positive_rate(y_test, y_pred)

    print("\n--- Evaluation (threshold=0.5) ---")
    print("Accuracy:   {0:.2%}".format(acc))
    print("Precision:  {0:.2%}".format(prec))
    print("Recall:     {0:.2%}".format(rec))
    print("F1:         {0:.2%}".format(f1))
    print("FPR:        {0:.2%}".format(fpr))
    print()
    print(classification_report(
        y_test, y_pred, target_names=["Safe", "Malicious"],
    ))

    # ------------------------------------------------------------------
    # FPR at various thresholds
    # ------------------------------------------------------------------
    print("--- FPR at various thresholds ---")
    print("{0:<12} {1:<12} {2:<12}".format("Threshold", "Accuracy", "FPR"))
    print("-" * 36)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        t_acc, t_fpr = _evaluate_at_threshold(y_test, y_proba, t)
        print("{0:<12.1f} {1:<12.2%} {2:<12.2%}".format(t, t_acc, t_fpr))

    # ------------------------------------------------------------------
    # Comparison with TF-IDF baseline
    # ------------------------------------------------------------------
    print("\n--- Comparison vs TF-IDF baseline ---")
    print("{0:<20} {1:<15} {2:<15}".format("Metric", "TF-IDF", "Embedding"))
    print("-" * 50)
    print("{0:<20} {1:<15} {2:<15}".format(
        "Accuracy", "{0:.1f}%".format(TFIDF_ACCURACY), "{0:.1f}%".format(acc * 100),
    ))
    print("{0:<20} {1:<15} {2:<15}".format(
        "FPR", "{0:.1f}%".format(TFIDF_FPR), "{0:.1f}%".format(fpr * 100),
    ))
    delta_fpr = TFIDF_FPR - (fpr * 100)
    if delta_fpr > 0:
        print("\nFPR reduced by {0:.1f} percentage points.".format(delta_fpr))
    else:
        print("\nFPR changed by {0:+.1f} percentage points.".format(-delta_fpr))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print("\nSaving model to {0}".format(MODEL_PATH))
    safe_dump(clf, MODEL_PATH)
    print("Done.")

    return clf


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow CLI overrides:  python model_embedding.py [model_type] [calibrate]
    mtype = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    cal = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
    train_embedding_model(model_type=mtype, calibrate=cal)
