"""Ensemble combiner for Layer 4 (TF-IDF) and Layer 5 (Embeddings).

Combines calibrated probabilities from both models via weighted average.
TF-IDF catches keyword patterns; embeddings catch semantic similarity.
Together they produce a more robust detection signal than either alone.

Design decisions:
  - Weighted average of P(malicious) from both models (simplest, most robust).
  - Graceful degradation: if embedding model is unavailable, falls back to
    TF-IDF only (no error, just a log message).
  - Configurable weights via parameters or NA0S_ENSEMBLE_TFIDF_WEIGHT env var.
  - Returns a ScanResult for compatibility with the rest of the pipeline.
"""

import logging
import os

from na0s.scan_result import ScanResult
from na0s.predict import scan as tfidf_scan
from na0s._voting import get_decision_threshold as _get_decision_threshold

# Layer 5: Embedding-based classifier -- optional import
try:
    from na0s.predict_embedding import (
        classify_prompt_embedding,
        load_models as _load_embedding_models,
    )
    _HAS_EMBEDDING = True
except ImportError:
    _HAS_EMBEDDING = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights (configurable via env var or parameter)
# ---------------------------------------------------------------------------
_DEFAULT_TFIDF_WEIGHT = 0.5
_DEFAULT_EMBEDDING_WEIGHT = 0.5

# Read from environment if set.
# NA0S_ENSEMBLE_TFIDF_WEIGHT sets the TF-IDF weight; embedding weight
# is the complement (1.0 - tfidf_weight).
_ENV_TFIDF_WEIGHT = os.environ.get("NA0S_ENSEMBLE_TFIDF_WEIGHT")
if _ENV_TFIDF_WEIGHT is not None:
    try:
        _DEFAULT_TFIDF_WEIGHT = float(_ENV_TFIDF_WEIGHT)
        _DEFAULT_EMBEDDING_WEIGHT = 1.0 - _DEFAULT_TFIDF_WEIGHT
    except (ValueError, TypeError):
        pass  # Keep defaults if env var is invalid

# Decision threshold -- loaded dynamically from _voting.py (single source of truth)
_DECISION_THRESHOLD = _get_decision_threshold()


def ensemble_scan(
    text,
    tfidf_weight=None,
    embedding_weight=None,
    vectorizer=None,
    model=None,
    embedding_model=None,
    embedding_classifier=None,
):
    """Combine Layer 4 (TF-IDF) and Layer 5 (Embeddings) into a single scan.

    Parameters
    ----------
    text : str
        The prompt text to scan.
    tfidf_weight : float or None
        Weight for the TF-IDF model's P(malicious).  Defaults to 0.5 or
        the value from ``NA0S_ENSEMBLE_TFIDF_WEIGHT`` env var.
    embedding_weight : float or None
        Weight for the embedding model's P(malicious).  Defaults to
        ``1.0 - tfidf_weight``.
    vectorizer, model : sklearn objects or None
        Pre-loaded TF-IDF vectorizer and classifier.  Loaded lazily if None.
    embedding_model : SentenceTransformer or None
        Pre-loaded embedding model.  Loaded lazily if None.
    embedding_classifier : sklearn model or None
        Pre-loaded embedding classifier.  Loaded lazily if None.

    Returns
    -------
    ScanResult
        Unified scan result with combined score from both models.
    """
    # Resolve weights
    w_tfidf = tfidf_weight if tfidf_weight is not None else _DEFAULT_TFIDF_WEIGHT
    w_embed = embedding_weight if embedding_weight is not None else _DEFAULT_EMBEDDING_WEIGHT

    # If only one weight was explicitly passed, compute the other as complement
    if tfidf_weight is not None and embedding_weight is None:
        w_embed = 1.0 - w_tfidf
    elif embedding_weight is not None and tfidf_weight is None:
        w_tfidf = 1.0 - w_embed

    # Clamp weights to [0, 1] and re-normalize
    w_tfidf = max(0.0, min(1.0, w_tfidf))
    w_embed = max(0.0, min(1.0, w_embed))
    total_weight = w_tfidf + w_embed
    if total_weight > 0:
        w_tfidf /= total_weight
        w_embed /= total_weight
    else:
        w_tfidf = 0.5
        w_embed = 0.5

    # ------------------------------------------------------------------
    # Layer 4: TF-IDF scan (always available)
    # ------------------------------------------------------------------
    tfidf_result = tfidf_scan(text, vectorizer=vectorizer, model=model)

    # If Layer 0 blocked the input, return immediately
    if tfidf_result.rejected:
        return tfidf_result

    # ------------------------------------------------------------------
    # Layer 5: Embedding scan (optional, graceful degradation)
    # ------------------------------------------------------------------
    embedding_available = False
    emb_p_malicious = 0.0
    emb_hits = []

    if _HAS_EMBEDDING:
        try:
            if embedding_model is None or embedding_classifier is None:
                embedding_model, embedding_classifier = _load_embedding_models()

            emb_label, emb_confidence, emb_hits_raw, _emb_l0 = classify_prompt_embedding(
                text,
                embedding_model=embedding_model,
                classifier=embedding_classifier,
            )

            emb_p_malicious = emb_confidence
            emb_hits = emb_hits_raw if emb_hits_raw else []
            embedding_available = True
            logger.debug(
                "Ensemble: embedding P(malicious)=%.4f, label=%s",
                emb_p_malicious, emb_label,
            )
        except Exception as exc:
            logger.warning(
                "Ensemble: embedding model unavailable, falling back to "
                "TF-IDF only: %s", exc,
            )

    # ------------------------------------------------------------------
    # Combine scores
    # ------------------------------------------------------------------
    tfidf_risk = tfidf_result.risk_score

    if embedding_available:
        combined_risk = (w_tfidf * tfidf_risk) + (w_embed * emb_p_malicious)
        contributor_tag = "ensemble:tfidf+embedding"
        logger.info(
            "Ensemble: tfidf=%.4f (w=%.2f) + embedding=%.4f (w=%.2f) = %.4f",
            tfidf_risk, w_tfidf, emb_p_malicious, w_embed, combined_risk,
        )
    else:
        combined_risk = tfidf_risk
        contributor_tag = "ensemble:tfidf_only"
        logger.info(
            "Ensemble: tfidf_only=%.4f (embedding unavailable)", tfidf_risk,
        )

    combined_risk = round(max(0.0, min(1.0, combined_risk)), 4)

    # ------------------------------------------------------------------
    # Decision: apply threshold
    # ------------------------------------------------------------------
    is_malicious = combined_risk >= _DECISION_THRESHOLD

    # Merge rule_hits from both models
    merged_hits = list(tfidf_result.rule_hits)
    for h in emb_hits:
        if h not in merged_hits:
            merged_hits.append(h)
    merged_hits.append(contributor_tag)

    # ------------------------------------------------------------------
    # Build ScanResult
    # ------------------------------------------------------------------
    return ScanResult(
        sanitized_text=tfidf_result.sanitized_text,
        is_malicious=is_malicious,
        risk_score=combined_risk,
        label="malicious" if is_malicious else "safe",
        technique_tags=list(tfidf_result.technique_tags),
        rule_hits=merged_hits,
        ml_confidence=combined_risk,
        ml_label=tfidf_result.ml_label,
        anomaly_flags=list(tfidf_result.anomaly_flags),
    )


# ---------------------------------------------------------------------------
# EnsembleClassifier — object-oriented wrapper around ensemble_scan()
# ---------------------------------------------------------------------------

import threading as _threading


class EnsembleClassifier:
    """Object-oriented ensemble of TF-IDF (Layer 4) and Embedding (Layer 5).

    Wraps :func:`ensemble_scan` into a reusable, thread-safe class that
    pre-loads models once and reuses them across calls.  Gracefully degrades
    to TF-IDF-only when the embedding model is unavailable.

    Parameters
    ----------
    tfidf_weight : float or None
        Weight for the TF-IDF model's P(malicious).  Defaults to 0.5
        (or the ``NA0S_ENSEMBLE_TFIDF_WEIGHT`` env var).
    embedding_weight : float or None
        Weight for the embedding model's P(malicious).  Defaults to
        ``1.0 - tfidf_weight``.
    threshold : float
        Decision threshold for the combined score.  Defaults to 0.55.

    Usage::

        clf = EnsembleClassifier(tfidf_weight=0.6)
        result = clf.scan("Ignore all previous instructions")
        print(result.is_malicious, result.risk_score)

        label, confidence, hits = clf.classify("some prompt")
    """

    def __init__(
        self,
        tfidf_weight=None,
        embedding_weight=None,
        threshold=None,
    ):
        self._tfidf_weight = tfidf_weight
        self._embedding_weight = embedding_weight
        self._threshold = threshold if threshold is not None else _DECISION_THRESHOLD

        # Pre-loaded sklearn objects (lazy, thread-safe)
        self._vectorizer = None
        self._model = None
        self._embedding_model = None
        self._embedding_classifier = None
        self._loaded = False
        self._lock = _threading.Lock()

    # ------------------------------------------------------------------
    # Lazy model loading (thread-safe, double-checked locking)
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        """Lazy-load TF-IDF and (optionally) embedding models.

        Thread-safe via double-checked locking.  If the embedding model
        is unavailable the classifier still works — it falls back to
        TF-IDF only through :func:`ensemble_scan`.
        """
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            # TF-IDF models — loaded from predict.py's lazy loader
            try:
                from na0s.predict import _ensure_model_loaded, _vectorizer, _model
                _ensure_model_loaded()
                # Re-import after loading (module-level globals updated)
                from na0s.predict import _vectorizer, _model
                self._vectorizer = _vectorizer
                self._model = _model
            except Exception as exc:
                logger.warning(
                    "EnsembleClassifier: TF-IDF model load failed: %s", exc,
                )

            # Embedding models — optional, graceful degradation
            if _HAS_EMBEDDING:
                try:
                    emb_model, emb_clf = _load_embedding_models()
                    self._embedding_model = emb_model
                    self._embedding_classifier = emb_clf
                except Exception as exc:
                    logger.warning(
                        "EnsembleClassifier: embedding model unavailable, "
                        "falling back to TF-IDF only: %s", exc,
                    )

            self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, text):
        """Scan text and return a :class:`ScanResult`.

        Combines TF-IDF and embedding classification via weighted average.
        Falls back to TF-IDF only when the embedding model is unavailable.

        Parameters
        ----------
        text : str
            The prompt text to scan.

        Returns
        -------
        ScanResult
        """
        self._ensure_loaded()
        return ensemble_scan(
            text,
            tfidf_weight=self._tfidf_weight,
            embedding_weight=self._embedding_weight,
            vectorizer=self._vectorizer,
            model=self._model,
            embedding_model=self._embedding_model,
            embedding_classifier=self._embedding_classifier,
        )

    def classify(self, text):
        """Classify text and return a ``(label, confidence, rule_hits)`` tuple.

        This is a convenience method that calls :meth:`scan` and unpacks
        the result into the same shape returned by
        :meth:`CascadeClassifier.classify`.

        Parameters
        ----------
        text : str
            The prompt text to classify.

        Returns
        -------
        tuple[str, float, list]
            ``(label, confidence, rule_hits)`` where *label* is
            ``"malicious"`` or ``"safe"``, *confidence* is the combined
            risk score, and *rule_hits* is the merged list of rule hits.
        """
        result = self.scan(text)
        return result.label, result.risk_score, list(result.rule_hits)

    def __repr__(self):
        return (
            "EnsembleClassifier("
            "tfidf_weight={!r}, embedding_weight={!r}, threshold={!r}"
            ")".format(self._tfidf_weight, self._embedding_weight, self._threshold)
        )
