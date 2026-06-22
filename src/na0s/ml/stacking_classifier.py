"""Stacking meta-learner for combining Stage 2 feature signals.

Instead of a hand-tuned linear combination, this module trains a logistic
regression (or other sklearn estimator) on the intermediate features
produced by Stage 2: [ml_score, rule_weight, obf_weight, structural_score,
embedding_score].

When no trained model is available, the meta-learner gracefully degrades
to the current linear combination in ``_voting.py``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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
        """Save the trained model to *path* with an integrity sidecar.

        Uses ``safe_dump`` so the meta-learner pickle ships with a verifiable
        ``.hmac``/``.sha256`` sidecar; :meth:`load` refuses a tampered or
        sidecar-less file rather than executing an attacker-controlled pickle.
        """
        from na0s.integrity.safe_pickle import safe_dump

        with self._lock:
            safe_dump({"model": self._model, "trained": self._trained}, path)

    def load(self, path: str) -> None:
        """Load a trained model from *path*, verifying integrity first.

        ``safe_load`` verifies the digest sidecar BEFORE unpickling. On a
        tampered file (``ValueError``) or a missing sidecar / KNOWN_HASHES
        entry (``FileNotFoundError``) the load is refused: we log and leave the
        meta-learner untrained so :meth:`is_available` returns ``False`` and the
        ensemble degrades to the linear combiner — we do NOT silently load an
        unverified pickle.
        """
        from na0s.integrity.safe_pickle import safe_load

        try:
            data = safe_load(path)
        except (ValueError, FileNotFoundError, OSError) as exc:
            logger.warning(
                "Refusing to load stacking model from %s: %s. "
                "Meta-learner stays unavailable (ensemble degrades).",
                path, exc,
            )
            with self._lock:
                self._trained = False
            return
        with self._lock:
            self._model = data["model"]
            self._trained = data["trained"]

    # -- Status --------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if a trained model is loaded and ready for prediction."""
        return self._trained
