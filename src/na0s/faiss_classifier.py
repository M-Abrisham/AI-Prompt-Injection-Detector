"""Layer 5: FAISS KNN classifier for nearest-known-attack lookup.

This module provides a K-nearest-neighbor classifier backed by a FAISS
IndexFlatIP (inner product) index. Embeddings are L2-normalized before
indexing so that inner product equals cosine similarity.

The index stores only *malicious* embeddings. At query time the classifier
finds the K nearest known attacks and returns a weighted vote score:

    score = (number of neighbors within threshold) / k

This gives a "how close is this input to known attacks?" signal that
complements the centroid-based embedding_classifier and the trained
LogisticRegression head in predict_embedding.

Architecture:
  - Build-time: L2-normalize embeddings, add to IndexFlatIP with labels
  - Query-time: L2-normalize query, search k neighbors, vote
  - Thread-safe singleton with double-checked locking
  - Graceful degradation when ``faiss`` (faiss-cpu) is not installed

Integration:
  - predict_embedding.py optionally queries this after main classification
  - Controlled by ``NA0S_FAISS_ENABLED=1`` env var (default: disabled)
  - If FAISS score > threshold and main classifier says safe, flag for review
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import: faiss-cpu is an optional dependency
# ---------------------------------------------------------------------------
# Defer faiss import to avoid SWIG side-effects at module load time
# (SWIG shared libs can interfere with SQLite thread-safety).
import numpy as np

_faiss = None  # lazy-loaded module reference
_HAS_FAISS = None  # tri-state: None=unchecked, True/False=resolved


def _get_faiss():
    """Lazily import faiss on first use."""
    global _faiss, _HAS_FAISS
    if _HAS_FAISS is None:
        try:
            import faiss as _f
            _faiss = _f
            _HAS_FAISS = True
        except ImportError:
            _HAS_FAISS = False
    return _faiss


def _faiss_available() -> bool:
    """Check if faiss-cpu is installed (triggers lazy import)."""
    _get_faiss()
    return bool(_HAS_FAISS)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_INDEX_PATH = os.path.join("data", "processed", "faiss_injection_index.bin")
DEFAULT_K = int(os.environ.get("NA0S_FAISS_K", "5"))
DEFAULT_THRESHOLD = float(os.environ.get("NA0S_FAISS_THRESHOLD", "0.6"))


def _l2_normalize(vectors: "np.ndarray") -> "np.ndarray":
    """L2-normalize each row so inner product == cosine similarity.

    Parameters
    ----------
    vectors : np.ndarray
        2-D array of shape ``(n, dim)``.

    Returns
    -------
    np.ndarray
        Row-normalized copy (or the original if norms are already 1).
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class FAISSClassifier:
    """KNN classifier using a FAISS inner-product index of known attacks.

    Thread-safe: index loading uses double-checked locking, queries are
    read-only on the FAISS index (thread-safe by design in faiss).

    Parameters
    ----------
    index_path : str or None
        Path to the serialized index file. If *None*, uses the default.
    k : int
        Number of nearest neighbors for voting.
    threshold : float
        Minimum cosine similarity for a neighbor to count as a positive
        vote. Neighbors below this threshold are ignored in the vote.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        k: int = DEFAULT_K,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self._index_path = index_path or DEFAULT_INDEX_PATH
        self._k = k
        self._threshold = threshold

        # Lazy-loaded state
        self._index: Optional[Any] = None
        self._labels: Optional["np.ndarray"] = None
        self._lock = threading.Lock()
        self._init_failed = False
        self._loaded = False

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(
        self,
        embeddings: "np.ndarray",
        labels: "np.ndarray",
    ) -> None:
        """Build a FAISS IndexFlatIP from training embeddings and labels.

        Embeddings are L2-normalized before indexing so that inner product
        equals cosine similarity.

        Parameters
        ----------
        embeddings : np.ndarray
            2-D array of shape ``(n_samples, embedding_dim)``.
        labels : np.ndarray
            1-D array of integer labels (one per embedding). Typically
            all 1 (malicious) when building an attack-only index.
        """
        if not _faiss_available():
            raise RuntimeError(
                "faiss-cpu is required to build a FAISS index. "
                "Install it with: pip install faiss-cpu"
            )

        embeddings = _l2_normalize(embeddings)
        dim = embeddings.shape[1]

        index = _get_faiss().IndexFlatIP(dim)
        index.add(embeddings)

        self._index = index
        self._labels = np.asarray(labels, dtype=np.int64)
        self._loaded = True
        self._init_failed = False

        logger.info(
            "FAISS index built: %d vectors, dim=%d", index.ntotal, dim
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the FAISS index and labels to disk.

        Creates two files:
        - ``path`` — the FAISS index binary
        - ``path + '.labels.pkl'`` — pickled label array

        Parameters
        ----------
        path : str
            Output path for the FAISS index file.
        """
        if not _faiss_available():
            raise RuntimeError("faiss-cpu is required to save a FAISS index.")
        if self._index is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        _get_faiss().write_index(self._index, path)

        labels_path = path + ".labels.pkl"
        with open(labels_path, "wb") as f:
            pickle.dump(self._labels, f)

        logger.info("FAISS index saved to %s (%d vectors)", path, self._index.ntotal)

    def load(self, path: str) -> None:
        """Load a FAISS index and labels from disk.

        Parameters
        ----------
        path : str
            Path to the FAISS index file. Labels are expected at
            ``path + '.labels.pkl'``.
        """
        if not _faiss_available():
            raise RuntimeError("faiss-cpu is required to load a FAISS index.")

        self._index = _get_faiss().read_index(path)

        labels_path = path + ".labels.pkl"
        with open(labels_path, "rb") as f:
            self._labels = pickle.load(f)

        self._loaded = True
        self._init_failed = False
        self._index_path = path

        logger.info(
            "FAISS index loaded from %s (%d vectors)", path, self._index.ntotal
        )

    # ------------------------------------------------------------------
    # Lazy loading (for singleton pattern)
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Lazy-load index from disk. Thread-safe with double-checked locking.

        Returns True if the index is ready, False otherwise.
        """
        if self._loaded:
            return True
        if self._init_failed:
            return False

        with self._lock:
            if self._loaded:
                return True
            if self._init_failed:
                return False

            try:
                if not os.path.exists(self._index_path):
                    logger.debug(
                        "FAISS index not found at %s — classifier disabled",
                        self._index_path,
                    )
                    self._init_failed = True
                    return False

                self.load(self._index_path)
                return True

            except Exception as exc:
                logger.warning("Failed to load FAISS index: %s", exc)
                self._init_failed = True
                return False

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, embedding: "np.ndarray") -> Dict[str, Any]:
        """Query the k nearest neighbors and return a weighted vote.

        Parameters
        ----------
        embedding : np.ndarray
            1-D or 2-D array (single embedding vector).

        Returns
        -------
        dict
            ``{"score": float, "label": str, "k": int,
              "neighbors_within_threshold": int,
              "max_similarity": float, "mean_similarity": float}``

            - ``score``: fraction of k neighbors within threshold
              (0.0 = no nearby attacks, 1.0 = all k neighbors are attacks
              above threshold)
            - ``label``: ``"MALICIOUS"`` if score > 0.5, else ``"SAFE"``
        """
        if not self._ensure_loaded():
            return {
                "score": 0.0,
                "label": "SAFE",
                "k": self._k,
                "neighbors_within_threshold": 0,
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
            }

        embedding = _l2_normalize(embedding)

        # Clamp k to index size
        effective_k = min(self._k, self._index.ntotal)
        if effective_k == 0:
            return {
                "score": 0.0,
                "label": "SAFE",
                "k": self._k,
                "neighbors_within_threshold": 0,
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
            }

        # Search — distances are cosine similarities (inner product on
        # L2-normalized vectors), higher = more similar
        distances, indices = self._index.search(embedding, effective_k)
        distances = distances[0]  # single query
        indices = indices[0]

        # Count neighbors above threshold
        within_threshold = int(np.sum(distances >= self._threshold))

        # Weighted vote: fraction of neighbors within threshold
        score = within_threshold / self._k

        max_sim = float(np.max(distances)) if len(distances) > 0 else 0.0
        mean_sim = float(np.mean(distances)) if len(distances) > 0 else 0.0

        label = "MALICIOUS" if score > 0.5 else "SAFE"

        return {
            "score": score,
            "label": label,
            "k": self._k,
            "neighbors_within_threshold": within_threshold,
            "max_similarity": max_sim,
            "mean_similarity": mean_sim,
        }

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls, index_path: Optional[str] = None) -> bool:
        """Check whether faiss-cpu is importable and an index file exists.

        Parameters
        ----------
        index_path : str or None
            Path to check. Defaults to ``DEFAULT_INDEX_PATH``.

        Returns
        -------
        bool
        """
        if not _faiss_available():
            return False
        path = index_path or DEFAULT_INDEX_PATH
        return os.path.exists(path)


# ---------------------------------------------------------------------------
# Module-level singleton — thread-safe lazy initialization
# ---------------------------------------------------------------------------
_singleton: Optional[FAISSClassifier] = None
_singleton_lock = threading.Lock()


def get_faiss_classifier(
    index_path: Optional[str] = None,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> FAISSClassifier:
    """Return the module-level FAISS classifier singleton.

    Thread-safe via double-checked locking.

    Parameters
    ----------
    index_path : str or None
        Path to the FAISS index file.
    k : int
        Number of nearest neighbors.
    threshold : float
        Cosine similarity threshold for voting.

    Returns
    -------
    FAISSClassifier
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    with _singleton_lock:
        if _singleton is not None:
            return _singleton

        _singleton = FAISSClassifier(
            index_path=index_path, k=k, threshold=threshold
        )

    return _singleton


def reset_singleton() -> None:
    """Reset the module-level singleton. Used in tests only."""
    global _singleton
    with _singleton_lock:
        _singleton = None
