    #Train the Model
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from na0s.safe_pickle import safe_load, safe_dump
import numpy as np

FEATURES_PATH = "data/processed/features.pkl"
MODEL_PATH = "data/processed/model.pkl"

def train_model():
    # --- Guard: verify input file exists ---
    if not os.path.isfile(FEATURES_PATH):
        print(f"ERROR: Features file not found: {FEATURES_PATH}")
        print("       Run scripts/features.py first to generate it.")
        sys.exit(1)

    try:
        print(" Training...")

        # load Binary data (integrity-checked)
        X, y = safe_load(FEATURES_PATH)

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

        # Save calibrated model
        safe_dump(calibrated, MODEL_PATH)

    except (np.linalg.LinAlgError, ValueError, TypeError) as e:
        print(f"ERROR: Training failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected failure during model training: {e}")
        sys.exit(1)

    # --- Verify output file exists and report size ---
    if os.path.isfile(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        print(f"Verified: {MODEL_PATH} ({size:,} bytes)")
    else:
        print(f"WARNING: Expected model file was not created: {MODEL_PATH}")

    print(" Classifier is successfully trained, calibrated, and saved")

if __name__ == "__main__":
    train_model()