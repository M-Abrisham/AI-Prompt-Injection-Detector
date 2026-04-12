# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.worm.detector"""
from na0s.worm.detector import *  # noqa: F401,F403
from na0s.worm.detector import (  # noqa: F401 — private names needed by downstream
    _BayesWormScorer,
    _EmbeddingSimilarity,
    _HAS_SENTENCE_TRANSFORMERS,
    _LightweightSemanticClassifier,
    _PCASignatureExtractor,
    _WormCorpusClassifier,
    _BENIGN_TRAINING_TEXTS,
    _WORM_TRAINING_TEXTS,
)
import warnings as _warnings
_warnings.warn("na0s.worm_detector is deprecated; use na0s.worm.detector instead", DeprecationWarning, stacklevel=2)
