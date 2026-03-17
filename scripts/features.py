"""Feature extraction script — TF-IDF + structural features.

Reads ``data/processed/combined_data.csv``, fits a TF-IDF vectorizer and a
StandardScaler for Layer 3 structural features, then writes the combined
sparse feature matrix (TF-IDF columns + scaled structural columns) plus
label vector to ``data/processed/features.pkl``.

The fitted vectorizer and scaler are saved separately so they can be reused
at inference time.
"""

import sys
import os
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from na0s.safe_pickle import safe_dump
from na0s.structural_features import extract_structural_features_batch

__all__ = ["load_training_data"]

# Paths
INPUT_PATH = "data/processed/combined_data.csv"
VECTORIZER_PATH = "data/processed/tfidf_vectorizer.pkl"
CHAR_VECTORIZER_PATH = "data/processed/char_tfidf_vectorizer.pkl"
SCALER_PATH = "data/processed/structural_scaler.pkl"
FEATURES_PATH = "data/processed/features.pkl"

# Minimum number of samples required for reliable training
_MIN_SAMPLES = 10
# TF-IDF vocabulary ceiling
_MAX_FEATURES = 10000
# Character-level TF-IDF vocabulary ceiling (supplementary, smaller)
_CHAR_MAX_FEATURES = 5000


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

        # --- Dataset rebalancing (max 3:1 majority:minority ratio) ---
        # Activated when minority class is < 20% of total.
        # Set SKIP_REBALANCE=1 to disable.
        if not os.environ.get("SKIP_REBALANCE", "") == "1":
            label_counts = dataset['label'].value_counts()
            total = len(dataset)
            minority_count = label_counts.min()
            majority_label = label_counts.idxmax()
            minority_label = label_counts.idxmin()

            print(f"Class distribution before rebalancing:")
            for lbl in sorted(label_counts.index):
                cnt = label_counts[lbl]
                pct = cnt / total * 100
                print(f"  Label {lbl}: {cnt} ({pct:.1f}%)")

            if minority_count < 0.20 * total:
                max_majority = minority_count * 3
                majority_df = dataset[dataset['label'] == majority_label]
                minority_df = dataset[dataset['label'] == minority_label]

                if len(majority_df) > max_majority:
                    majority_df = majority_df.sample(
                        n=max_majority, random_state=42
                    )
                    dataset = pd.concat(
                        [majority_df, minority_df], ignore_index=True
                    )
                    print(f"Rebalanced: undersampled label {majority_label} "
                          f"from {label_counts[majority_label]} to {max_majority} "
                          f"(3:1 ratio). Total samples: {len(dataset)}")
                else:
                    print("No rebalancing needed (majority already within 3:1).")
            else:
                print("No rebalancing needed (minority class >= 20% of total).")
        else:
            print("Rebalancing skipped (SKIP_REBALANCE=1).")

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

        # --- TF-IDF features ---
        vec = TfidfVectorizer(
            lowercase=True,
            max_features=_MAX_FEATURES,
            ngram_range=(1, 3),
            sublinear_tf=True,
        )
        X_tfidf = vec.fit_transform(dataset['text'])
        print(f"TF-IDF features: {X_tfidf.shape}")

        # --- Layer 4: Character-level TF-IDF features ---
        char_vec = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=_CHAR_MAX_FEATURES,
            sublinear_tf=True,
        )
        X_char_tfidf = char_vec.fit_transform(dataset['text'])
        print(f"Char TF-IDF features: {X_char_tfidf.shape}")

        # --- Layer 3: Structural features ---
        texts = dataset['text'].tolist()
        print(f"Extracting structural features for {len(texts)} samples...")
        X_structural_raw = extract_structural_features_batch(texts)
        print(f"Structural features: {X_structural_raw.shape}")

        # Fit StandardScaler on structural features (zero mean, unit variance)
        scaler = StandardScaler()
        X_structural_scaled = scaler.fit_transform(X_structural_raw)

        # Combine: sparse word TF-IDF + char TF-IDF + dense structural (converted to sparse)
        X_structural_sparse = scipy.sparse.csr_matrix(X_structural_scaled)
        X = scipy.sparse.hstack([X_tfidf, X_char_tfidf, X_structural_sparse], format="csr")
        y = dataset['label']  # 0 = Safe, 1 = Malicious
        print(f"Combined features: {X.shape} "
              f"({X_tfidf.shape[1]} word TF-IDF + {X_char_tfidf.shape[1]} char TF-IDF"
              f" + {X_structural_raw.shape[1]} structural)")

        # Save word vectorizer
        print(f"Saving vectorizer to {VECTORIZER_PATH}...")
        safe_dump(vec, VECTORIZER_PATH)

        # Save char vectorizer
        print(f"Saving char vectorizer to {CHAR_VECTORIZER_PATH}...")
        safe_dump(char_vec, CHAR_VECTORIZER_PATH)

        # Save structural scaler
        print(f"Saving structural scaler to {SCALER_PATH}...")
        safe_dump(scaler, SCALER_PATH)

        # Save combined features
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
    for path in (VECTORIZER_PATH, CHAR_VECTORIZER_PATH, SCALER_PATH, FEATURES_PATH):
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
