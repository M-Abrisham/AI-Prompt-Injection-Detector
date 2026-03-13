"""Model training script — logistic regression with isotonic calibration.

Reads the feature matrix produced by ``scripts/features.py``, trains a
``LogisticRegression`` model, applies ``CalibratedClassifierCV`` for
probability calibration, and writes the result to
``data/processed/model.pkl``.
"""

import json
import sys
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix,
)
from na0s.safe_pickle import safe_load, safe_dump
import numpy as np

__all__ = ["train_model", "compute_ece"]

FEATURES_PATH = "data/processed/features.pkl"
MODEL_PATH = "data/processed/model.pkl"
METRICS_PATH = "data/processed/training_metrics.json"

# Minimum number of samples required for meaningful training
_MIN_SAMPLES = 100

# Default decision threshold used by the detector
_DEFAULT_THRESHOLD = 0.55


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error.

    Bins predicted probabilities into *n_bins* equal-width bins and returns
    the weighted average of ``|accuracy - confidence|`` per bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (count / len(y_true)) * abs(avg_acc - avg_conf)
    return float(ece)


def train_model():
    # --- Guard: verify input file exists ---
    if not os.path.isfile(FEATURES_PATH):
        print(f"ERROR: Features file not found: {FEATURES_PATH}")
        print("       Run scripts/features.py first to generate it.")
        sys.exit(1)

    try:
        print(" Training...")

        # load Binary data (integrity-checked)
        loaded = safe_load(FEATURES_PATH)

        # --- Guard: features file must contain a (matrix, array) tuple ---
        if (
            not isinstance(loaded, tuple)
            or len(loaded) != 2
        ):
            print(
                f"ERROR: Features file has unexpected format. "
                f"Expected a 2-tuple (X, y), got {type(loaded).__name__}."
            )
            sys.exit(1)

        X, y = loaded

        # --- Guard: feature matrix must not be empty ---
        if X.shape[0] == 0:
            print("ERROR: Feature matrix has 0 samples. Cannot train.")
            sys.exit(1)

        # --- Guard: minimum sample count ---
        n_samples = X.shape[0]
        if n_samples < _MIN_SAMPLES:
            print(
                f"ERROR: Feature matrix has only {n_samples} sample(s). "
                f"At least {_MIN_SAMPLES} are required for reliable training."
            )
            sys.exit(1)

        # --- Guard: both class labels must be present ---
        y_arr = np.asarray(y)
        unique_labels = set(y_arr)
        if not {0, 1}.issubset(unique_labels):
            print(
                f"ERROR: Labels must contain both class 0 (Safe) and class 1 "
                f"(Malicious). Found only: {sorted(unique_labels)}"
            )
            sys.exit(1)

        # --- Stratified k-fold cross-validation (base model selection) ---
        print("\n Stratified 5-fold cross-validation (base LogisticRegression)...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_model = LogisticRegression(max_iter=10000, random_state=0, class_weight='balanced')
        cv_results = cross_validate(
            cv_model, X, y, cv=skf,
            scoring=["accuracy", "roc_auc"],
            return_train_score=False,
        )
        cv_acc_mean = cv_results["test_accuracy"].mean()
        cv_acc_std = cv_results["test_accuracy"].std()
        cv_auc_mean = cv_results["test_roc_auc"].mean()
        cv_auc_std = cv_results["test_roc_auc"].std()
        print(f"   CV Accuracy : {cv_acc_mean:.4f} +/- {cv_acc_std:.4f}")
        print(f"   CV ROC-AUC  : {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")

        # Split data: Test + Training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model (balanced class weights to handle class imbalance)
        clf = LogisticRegression(max_iter=10000, random_state=0, class_weight='balanced')
        clf.fit(X_train, y_train)

        # Raw model evaluation
        y_pred_raw = clf.predict(X_test)
        acc_raw = accuracy_score(y_test, y_pred_raw)
        print(f" Raw model accuracy: {acc_raw * 100:.2f}%")
        print(classification_report(y_test, y_pred_raw, target_names=["Safe", "Malicious"]))

        # Probability calibration (isotonic regression, 5-fold CV)
        print(" Calibrating probabilities...")
        calibrated = CalibratedClassifierCV(clf, cv=5, method='isotonic')
        calibrated.fit(X_train, y_train)

        # Calibrated model evaluation
        y_pred_cal = calibrated.predict(X_test)
        acc_cal = accuracy_score(y_test, y_pred_cal)
        print(f" Calibrated model accuracy: {acc_cal * 100:.2f}%")
        print(classification_report(y_test, y_pred_cal, target_names=["Safe", "Malicious"]))

        # Before/after comparison
        print(f" Accuracy comparison: raw={acc_raw * 100:.2f}% vs calibrated={acc_cal * 100:.2f}%")

        # FPR/TPR at various thresholds
        probs = calibrated.predict_proba(X_test)[:, 1]
        safe_mask = (y_test == 0)
        malicious_mask = (y_test == 1)
        print(f"\n {'Threshold':<12}{'TPR':<10}{'FPR':<10}")
        print(f" {'-' * 30}")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            predicted_positive = (probs >= t)
            tpr = predicted_positive[malicious_mask].sum() / malicious_mask.sum() if malicious_mask.sum() > 0 else 0.0
            fpr = predicted_positive[safe_mask].sum() / safe_mask.sum() if safe_mask.sum() > 0 else 0.0
            print(f" {t:<12.2f}{tpr:<10.4f}{fpr:<10.4f}")
        print()

        # --- Comprehensive metrics ---
        print(" === Comprehensive Metrics ===")

        # ROC-AUC and PR-AUC
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        print(f"   ROC-AUC  : {roc_auc:.4f}")
        print(f"   PR-AUC   : {pr_auc:.4f}")

        # Brier score
        brier = brier_score_loss(y_test, probs)
        print(f"   Brier    : {brier:.4f}")

        # Expected Calibration Error
        ece = compute_ece(y_test, probs)
        print(f"   ECE      : {ece:.4f}")

        # FNR at default threshold
        pred_at_thresh = (probs >= _DEFAULT_THRESHOLD).astype(int)
        fn = ((pred_at_thresh == 0) & (np.asarray(y_test) == 1)).sum()
        tp = ((pred_at_thresh == 1) & (np.asarray(y_test) == 1)).sum()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        print(f"   FNR@{_DEFAULT_THRESHOLD} : {fnr:.4f}")

        # Confusion matrix at default threshold
        cm = confusion_matrix(y_test, pred_at_thresh)
        print(f"\n   Confusion matrix (threshold={_DEFAULT_THRESHOLD}):")
        print(f"                  Predicted Safe  Predicted Malicious")
        print(f"   Actual Safe       {cm[0, 0]:<15d}{cm[0, 1]:<15d}")
        print(f"   Actual Malicious  {cm[1, 0]:<15d}{cm[1, 1]:<15d}")
        print()

        # --- Save metrics JSON ---
        metrics = {
            "cv_accuracy_mean": round(cv_acc_mean, 4),
            "cv_accuracy_std": round(cv_acc_std, 4),
            "cv_roc_auc_mean": round(cv_auc_mean, 4),
            "cv_roc_auc_std": round(cv_auc_std, 4),
            "raw_accuracy": round(acc_raw, 4),
            "calibrated_accuracy": round(acc_cal, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "fnr_at_default_threshold": round(fnr, 4),
            "default_threshold": _DEFAULT_THRESHOLD,
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f" Metrics saved to {METRICS_PATH}")

        # Save calibrated model
        safe_dump(calibrated, MODEL_PATH)

    except (np.linalg.LinAlgError, ValueError, TypeError) as e:
        print(f"ERROR: Training failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected failure during model training: {e}")
        sys.exit(1)

    # --- Verify output file exists and exit non-zero if missing ---
    if os.path.isfile(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        print(f"Verified: {MODEL_PATH} ({size:,} bytes)")
    else:
        print(f"ERROR: Expected model file was not created: {MODEL_PATH}")
        sys.exit(1)

    print(" Classifier is successfully trained, calibrated, and saved")

if __name__ == "__main__":
    train_model()