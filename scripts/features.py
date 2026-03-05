import re
import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from na0s.safe_pickle import safe_dump

# Paths
INPUT_PATH = "data/processed/combined_data.csv"
VECTORIZER_PATH = "data/processed/tfidf_vectorizer.pkl"
FEATURES_PATH = "data/processed/features.pkl"

def load_training_data():
    # --- Guard: verify input file exists ---
    if not os.path.isfile(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        sys.exit(1)

    try:
        print("Loading...")
        dataset = pd.read_csv(INPUT_PATH)  #import the dataset

        dataset['text'] = dataset['text'].fillna('').astype(str)

    # Create Vectorizer with top 10K words
        vec = TfidfVectorizer(
            lowercase=True,
            max_features=10000
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

    # --- Verify output files exist and report sizes ---
    for path in (VECTORIZER_PATH, FEATURES_PATH):
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"Verified: {path} ({size:,} bytes)")
        else:
            print(f"WARNING: Expected output file was not created: {path}")

    print(f"Successfully created! Shape: {X.shape}")

# Test
if __name__ == "__main__":
    load_training_data()