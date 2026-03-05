"""Embedding-based feature extraction using sentence-transformers.

Replaces TF-IDF with semantic embeddings that understand context.
Model: all-MiniLM-L6-v2 (22M params, 384-dim, ~20ms/sample on CPU)

Usage:
    PYTHONPATH=src:. python src/features_embedding.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required for embedding features.\n"
        "Install it with:  pip install sentence-transformers\n"
        "This will also install torch and transformers as dependencies."
    )

from na0s.safe_pickle import safe_dump

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH = "data/processed/combined_data.csv"
FEATURES_PATH = "data/processed/features_embedding.pkl"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_embedding_model(model_name=DEFAULT_MODEL):
    """Load and return a SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to ``all-MiniLM-L6-v2``
        (384-dim embeddings, fast on CPU).

    Returns
    -------
    SentenceTransformer
    """
    print("Loading sentence-transformer model: {0}".format(model_name))
    model = SentenceTransformer(model_name)
    return model


def extract_embeddings(texts, model=None):
    """Encode a list of texts into dense embeddings.

    Parameters
    ----------
    texts : list[str]
        Raw text strings to embed.
    model : SentenceTransformer or None
        If *None*, the default model is loaded automatically.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(texts), embedding_dim)`` — 384 for the
        default model.
    """
    if model is None:
        model = load_embedding_model()

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    return embeddings


def build_embedding_features():
    """End-to-end feature extraction pipeline.

    1. Loads ``data/processed/combined_data.csv``
    2. Encodes every text with the sentence-transformer
    3. Saves ``(X_embeddings, y)`` to ``data/processed/features_embedding.pkl``
       using SHA-256 verified pickle I/O.

    Set the ``EMBEDDING_MAX_SAMPLES`` environment variable to an integer
    to stratified-sample the dataset before encoding (useful for large
    datasets that would take too long to encode in full).
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading training data from {0}".format(INPUT_PATH))
    df = pd.read_csv(INPUT_PATH)
    df["text"] = df["text"].fillna("").astype(str)

    # Optional stratified subsampling for large datasets
    max_samples = os.environ.get("EMBEDDING_MAX_SAMPLES")
    if max_samples is not None:
        max_samples = int(max_samples)
        if len(df) > max_samples:
            from sklearn.model_selection import train_test_split
            print("  Subsampling {0} -> {1} (stratified)".format(
                len(df), max_samples))
            df, _ = train_test_split(
                df, train_size=max_samples, random_state=42,
                stratify=df["label"],
            )
            df = df.reset_index(drop=True)

    texts = df["text"].tolist()
    y = df["label"].values  # 0 = safe, 1 = malicious

    print("  Samples: {0}  (safe={1}, malicious={2})".format(
        len(texts),
        int((y == 0).sum()),
        int((y == 1).sum()),
    ))

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    model = load_embedding_model()
    print("Encoding {0} texts...".format(len(texts)))
    X = extract_embeddings(texts, model=model)

    elapsed = time.time() - t0
    print("Embedding shape: {0}".format(X.shape))
    print("Time: {0:.1f}s  ({1:.1f} ms/sample)".format(
        elapsed, (elapsed / len(texts)) * 1000
    ))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print("Saving features to {0}".format(FEATURES_PATH))
    safe_dump((X, y), FEATURES_PATH)
    print("Done.")


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_embedding_features()
