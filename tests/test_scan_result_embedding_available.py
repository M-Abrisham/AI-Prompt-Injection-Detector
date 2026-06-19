"""Observability: ScanResult.embedding_available reflects the degraded state.

The resilient loader (g1) exposes ``get_embedding_classifier().is_degraded`` /
``.available``, but ScanResult did not surface whether the embedding signal was
degraded for a given scan, so callers/telemetry could not see when detection ran
on a fallback backend.  ``ScanResult.embedding_available`` (default True,
observability-safe) is populated in predict.scan() from the classifier's
``is_degraded`` flag.

TEETH: mocking ``is_degraded=True`` must flip ``embedding_available`` to False.
If the field is dropped or hard-coded True, the degraded-backend assertion fails.
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SCAN_TIMEOUT_SEC", "0")

from na0s.models import get_model_path
from na0s.scan_result import ScanResult

_MODEL_PATH = get_model_path("model.pkl")
_VECTORIZER_PATH = get_model_path("tfidf_vectorizer.pkl")
_MODELS_AVAILABLE = os.path.isfile(_MODEL_PATH) and os.path.isfile(_VECTORIZER_PATH)

from na0s import predict

_SKIP_REASON = "" if _MODELS_AVAILABLE else "Model files not found"


class _FakeEmbedding:
    def __init__(self, is_degraded):
        self.is_degraded = is_degraded

    def classify(self, text):
        return 0.0, []


class TestScanResultFieldDefault(unittest.TestCase):
    """The new field is backward-compatible and observability-safe."""

    def test_field_defaults_true(self):
        self.assertTrue(ScanResult().embedding_available)

    def test_field_round_trips_through_serialization(self):
        d = ScanResult(embedding_available=False).to_dict()
        self.assertIn("embedding_available", d)
        self.assertFalse(d["embedding_available"])
        # to_json must not raise with the new field present.
        self.assertIn("embedding_available", ScanResult().to_json())

    def test_existing_construction_without_field_still_works(self):
        # No embedding_available kwarg -> defaults True (backward compatible).
        r = ScanResult(sanitized_text="hi", risk_score=0.1)
        self.assertTrue(r.embedding_available)


@unittest.skipUnless(_MODELS_AVAILABLE, _SKIP_REASON)
class TestScanResultEmbeddingAvailable(unittest.TestCase):
    """scan() populates embedding_available from is_degraded."""

    def test_live_backend_reports_available(self):
        fake = _FakeEmbedding(is_degraded=False)
        with patch.object(predict, "_HAS_EMBEDDING_CLASSIFIER", True), \
             patch.object(predict, "get_embedding_classifier", return_value=fake), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NA0S_EMBEDDING_ENABLED", None)
            result = predict.scan("What time does the museum open?")
        self.assertTrue(result.embedding_available)

    def test_degraded_backend_reports_unavailable(self):
        """TEETH: is_degraded=True -> embedding_available False."""
        fake = _FakeEmbedding(is_degraded=True)
        with patch.object(predict, "_HAS_EMBEDDING_CLASSIFIER", True), \
             patch.object(predict, "get_embedding_classifier", return_value=fake), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NA0S_EMBEDDING_ENABLED", None)
            result = predict.scan("What time does the museum open?")
        self.assertFalse(result.embedding_available)

    def test_runtime_disabled_reports_unavailable(self):
        """Env-disabled at runtime -> embedding ran on no backend -> unavailable."""
        fake = _FakeEmbedding(is_degraded=False)
        with patch.object(predict, "_HAS_EMBEDDING_CLASSIFIER", True), \
             patch.object(predict, "get_embedding_classifier", return_value=fake), \
             patch.dict(os.environ, {}, clear=False):
            os.environ["NA0S_EMBEDDING_ENABLED"] = "0"
            result = predict.scan("What time does the museum open?")
        self.assertFalse(result.embedding_available)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
