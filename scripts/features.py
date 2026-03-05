"""Feature extraction script — TF-IDF vectorizer over the combined dataset.

Reads ``data/processed/combined_data.csv``, fits a TF-IDF vectorizer, and
writes the sparse feature matrix plus label vector to
``data/processed/features.pkl``.  The fitted vectorizer is saved separately
so it can be reused at inference time.
"""

import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from na0s.safe_pickle import safe_dump

__all__ = ["load_training_data"]

# Paths
INPUT_PATH = "data/processed/combined_data.csv"
VECTORIZER_PATH = "data/processed/tfidf_vectorizer.pkl"
FEATURES_PATH = "data/processed/features.pkl"

# Minimum number of samples required for reliable training
_MIN_SAMPLES = 10
# TF-IDF vocabulary ceiling
_MAX_FEATURES = 10000


def load_training_data():
    # --- Guard: verify input file exists ---
    if not os.path.isfile(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        sys.exit(1)

    try:
        print("Loading...")
        dataset = pd.read_csv(INPUT_PATH)  #import the dataset

        # --- Guard: dataset must not be empty ---
        if dataset.empty or len(dataset) == 0:
            print("ERROR: Dataset is empty (0 rows). Cannot train.")
            sys.exit(1)

        # --- Guard: required columns must be present ---
        missing_cols = [c for c in ("text", "label") if c not in dataset.columns]
        if missing_cols:
            print(f"ERROR: Dataset is missing required column(s): {missing_cols}")
            sys.exit(1)

        dataset['text'] = dataset['text'].fillna('').astype(str)

        # --- Guard: both class labels must be present ---
        unique_labels = set(dataset['label'].dropna().unique())
        if not {0, 1}.issubset(unique_labels):
            print(
                f"ERROR: Dataset must contain both label 0 (Safe) and label 1 "
                f"(Malicious). Found only: {sorted(unique_labels)}"
            )
            sys.exit(1)

        # --- Guard: minimum sample count ---
        n_samples = len(dataset)
        if n_samples < _MIN_SAMPLES:
            print(
                f"ERROR: Dataset has only {n_samples} sample(s). "
                f"At least {_MIN_SAMPLES} are required for meaningful training."
            )
            sys.exit(1)

        # Warn when vocabulary ceiling exceeds the dataset size
        if _MAX_FEATURES > n_samples:
            print(
                f"WARNING: max_features={_MAX_FEATURES} exceeds dataset size "
                f"({n_samples}). Vocabulary will be limited to unique terms."
            )

    # Create Vectorizer with top 10K words
        vec = TfidfVectorizer(
            lowercase=True,
            max_features=_MAX_FEATURES
            )

        X = vec.fit_transform(dataset['text'])
        y = dataset['label'] # 0 = Safe, 1 = Malicious

    # Save VECTORIZER_Path data
        print(f"Saving vectorizer to {VECTORIZER_PATH}...")
        safe_dump(vec, VECTORIZER_PATH)

     # Save FEATURES_PATH data
        print(f"Saving features to {FEATURES_PATH}...")
        safe_dump((X, y), FEATURES_PATH)

    except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError, ValueError) as e:
        print(f"ERROR: Failed to process training data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected failure during feature extraction: {e}")
        sys.exit(1)

    # --- Verify output files exist and exit non-zero if missing ---
    all_ok = True
    for path in (VECTORIZER_PATH, FEATURES_PATH):
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"Verified: {path} ({size:,} bytes)")
        else:
            print(f"ERROR: Expected output file was not created: {path}")
            all_ok = False
    if not all_ok:
        sys.exit(1)

    print(f"Successfully created! Shape: {X.shape}")

# Test
if __name__ == "__main__":
    load_training_data()