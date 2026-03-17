"""Stacking meta-learner for combining Stage 2 feature signals.

Instead of a hand-tuned linear combination, this module trains a logistic
regression (or other sklearn estimator) on the intermediate features
produced by Stage 2: [ml_score, rule_weight, obf_weight, structural_score,
embedding_score].

When no trained model is available, the meta-learner gracefully degrades
to the current linear combination in ``_voting.py``.
"""

from __future__ import annotations

import os
import pickle
import threading
from typing import Any

import numpy as np


class StackingMetaLearner:
    """Meta-learner that sits on top of Stage 2 feature outputs.

    Parameters
    ----------
    base_model : sklearn estimator or None
        The meta-learner model.  Defaults to ``LogisticRegression`` if None.
    """

    # Feature names in the expected order.
    FEATURE_NAMES = [
        "ml_score",
        "rule_weight",
        "obf_weight",
        "structural_score",
        "embedding_score",
    ]

    def __init__(self, base_model: Any = None):
        self._lock = threading.Lock()
        if base_model is not None:
            self._model = base_model
        else:
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )
        self._trained = False

    # -- Training ------------------------------------------------------------

    def train(self, stage2_features: np.ndarray, labels: np.ndarray) -> None:
        """Train the meta-learner on Stage 2 outputs.

        Parameters
        ----------
        stage2_features : np.ndarray
            Array of shape ``(n_samples, 5)`` with columns
            ``[ml_score, rule_weight, obf_weight, structural_score, embedding_score]``.
        labels : np.ndarray
            Binary labels (1 = malicious, 0 = safe).
        """
        if stage2_features.ndim != 2 or stage2_features.shape[1] != len(self.FEATURE_NAMES):
            raise ValueError(
                "stage2_features must have shape (n, {}), got {}".format(
                    len(self.FEATURE_NAMES), stage2_features.shape,
                )
            )
        with self._lock:
            self._model.fit(stage2_features, labels)
            self._trained = True

    # -- Prediction ----------------------------------------------------------

    def predict(self, stage2_features: np.ndarray) -> tuple[str, float]:
        """Predict using the trained meta-learner.

        Parameters
        ----------
        stage2_features : np.ndarray
            Feature vector of shape ``(1, 5)`` or ``(5,)``.

        Returns
        -------
        tuple[str, float]
            ``(label, confidence)`` where label is ``"MALICIOUS"`` or ``"SAFE"``.

        Raises
        ------
        RuntimeError
            If no trained model is available (use :meth:`is_available` first).
        """
        if not self._trained:
            raise RuntimeError("No trained stacking model available")
        features = np.atleast_2d(stage2_features)
        with self._lock:
            proba = self._model.predict_proba(features)[0]
            # proba is [P(safe), P(malicious)] for classes [0, 1]
            classes = list(self._model.classes_)
            if 1 in classes:
                mal_idx = classes.index(1)
            else:
                mal_idx = -1

            if mal_idx >= 0:
                p_mal = proba[mal_idx]
            else:
                p_mal = proba[-1]

        if p_mal >= 0.5:
            return "MALICIOUS", float(p_mal)
        return "SAFE", float(1.0 - p_mal)

    # -- Persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained model to *path*."""
        with self._lock:
            with open(path, "wb") as fh:
                pickle.dump(
                    {"model": self._model, "trained": self._trained},
                    fh,
                )

    def load(self, path: str) -> None:
        """Load a trained model from *path*."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)  # noqa: S301
        with self._lock:
            self._model = data["model"]
            self._trained = data["trained"]

    # -- Status --------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if a trained model is loaded and ready for prediction."""
        return self._trained
