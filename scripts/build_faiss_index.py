#!/usr/bin/env python
"""Build a FAISS index of malicious-only embeddings for KNN lookup.

Reads embedding features from the training pipeline output, filters to
malicious samples only, L2-normalizes, and builds a FAISS IndexFlatIP
index for cosine-similarity search.

Usage:
    PYTHONPATH=src:. python scripts/build_faiss_index.py

    # Custom paths:
    PYTHONPATH=src:. python scripts/build_faiss_index.py \
        --embedding-path data/processed/features_embedding.pkl \
        --output data/processed/faiss_injection_index.bin

Requires:
    - faiss-cpu  (pip install faiss-cpu)
    - numpy
    - na0s.safe_pickle  (for loading verified pkl files)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure src/ is on the path for na0s imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_project_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_project_root, "src"))

import numpy as np

try:
    import faiss
except ImportError:
    print(
        "ERROR: faiss-cpu is required to build the FAISS index.\n"
        "Install it with: pip install faiss-cpu"
    )
    sys.exit(1)

from na0s.safe_pickle import safe_load
from na0s.faiss_classifier import FAISSClassifier


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_PATH = os.path.join("data", "processed", "features_embedding.pkl")
DEFAULT_OUTPUT_PATH = os.path.join("data", "processed", "faiss_injection_index.bin")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index of malicious embeddings for KNN lookup."
    )
    parser.add_argument(
        "--embedding-path",
        default=DEFAULT_EMBEDDING_PATH,
        help="Path to features_embedding.pkl (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for FAISS index binary (default: %(default)s)",
    )
    parser.add_argument(
        "--include-safe",
        action="store_true",
        default=False,
        help="Include safe samples in the index (default: malicious only)",
    )
    args = parser.parse_args()

    t0 = time.time()

    # ------------------------------------------------------------------
    # Load embedding features
    # ------------------------------------------------------------------
    print("Loading embeddings from {}".format(args.embedding_path))
    X, y = safe_load(args.embedding_path)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print("  Total samples: {}  (safe={}, malicious={})".format(
        len(y), int((y == 0).sum()), int((y == 1).sum())
    ))

    # ------------------------------------------------------------------
    # Filter to malicious only (unless --include-safe)
    # ------------------------------------------------------------------
    if not args.include_safe:
        mask = y == 1
        X_index = X[mask]
        y_index = y[mask]
        print("  Indexing malicious samples only: {}".format(len(y_index)))
    else:
        X_index = X
        y_index = y
        print("  Indexing all samples: {}".format(len(y_index)))

    if len(y_index) == 0:
        print("ERROR: No samples to index.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build FAISS index
    # ------------------------------------------------------------------
    print("Building FAISS IndexFlatIP (dim={})...".format(X_index.shape[1]))
    classifier = FAISSClassifier()
    classifier.build_index(X_index, y_index)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Saving index to {}".format(args.output))
    classifier.save(args.output)

    elapsed = time.time() - t0
    print("Done in {:.1f}s.".format(elapsed))
    print("  Index vectors: {}".format(classifier._index.ntotal))
    print("  Index file: {}".format(args.output))
    print("  Labels file: {}".format(args.output + ".labels.pkl"))


if __name__ == "__main__":
    main()
