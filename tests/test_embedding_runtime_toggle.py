"""Runtime determinism: NA0S_EMBEDDING_ENABLED must be honored at SCAN TIME.

The Layer 5 embedding signal is gated by ``NA0S_EMBEDDING_ENABLED``.  Historically
the env was read ONCE at import into module-level flags
(``predict._HAS_EMBEDDING_CLASSIFIER`` / ``cascade._HAS_EMBEDDING_CENTROID``), so
flipping the env AFTER import — a common test/app pattern — was silently ignored.

``predict._embedding_enabled()`` / ``cascade._embedding_enabled()`` now combine the
import-time availability flag with a RUNTIME read of the env, so the toggle is
honored at the call site while preserving predict/cascade parity.

TEETH: each runtime-toggle test mocks the embedding classifier to a POSITIVE
score and proves the contribution appears when the env is unset and DISAPPEARS
when ``NA0S_EMBEDDING_ENABLED=0`` is set after import.  If the env read is moved
back to import time (or the helper is reverted to the raw module flag), the
"disabled" run still contributes and these tests fail.
"""

import os
import unittest
from unittest.mock import patch

# Disable thread-based scan timeout so signal.SIGALRM works in the main thread.
# Must be set BEFORE importing predict/cascade (timeout.py reads it at import).
os.environ.setdefault("SCAN_TIMEOUT_SEC", "0")

from na0s.models import get_model_path

_MODEL_PATH = get_model_path("model.pkl")
_VECTORIZER_PATH = get_model_path("tfidf_vectorizer.pkl")
_MODELS_AVAILABLE = os.path.isfile(_MODEL_PATH) and os.path.isfile(_VECTORIZER_PATH)

from na0s import cascade, predict
from na0s.cascade import WeightedClassifier

_SKIP_REASON = ""
_OK = False
if _MODELS_AVAILABLE:
    try:
        from na0s.safe_pickle import safe_load

        _vectorizer = safe_load(_VECTORIZER_PATH)
        _model = safe_load(_MODEL_PATH)
        _OK = True
    except Exception as _err:  # pragma: no cover
        _SKIP_REASON = "Model loading failed: {}".format(_err)
else:  # pragma: no cover
    _SKIP_REASON = "Model files not found at {}".format(_MODEL_PATH)


class _FakeEmbedding:
    """Stand-in for get_embedding_classifier() with a controllable score."""

    def __init__(self, score, matches, is_degraded=False):
        self._score = score
        self._matches = matches
        self.is_degraded = is_degraded
        self.calls = 0

    def classify(self, text):
        self.calls += 1
        return self._score, self._matches


def _mal_composite(label, confidence):
    """Recover the underlying malicious composite from cascade's (label,
    confidence) API: confidence is P(label correct) — composite when MALICIOUS,
    1 - composite when SAFE."""
    return confidence if label == "MALICIOUS" else (1.0 - confidence)


# ---------------------------------------------------------------------------
# Helper-level tests (no model files required) — the toggle mechanism itself.
# ---------------------------------------------------------------------------
class TestEmbeddingEnabledHelper(unittest.TestCase):
    """predict/cascade._embedding_enabled() read the env at CALL time."""

    def test_predict_helper_honors_runtime_disable(self):
        with patch.object(predict, "_HAS_EMBEDDING_CLASSIFIER", True):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NA0S_EMBEDDING_ENABLED", None)
                self.assertTrue(predict._embedding_enabled())
                # Flip the env AFTER import — must be honored immediately.
                os.environ["NA0S_EMBEDDING_ENABLED"] = "0"
                self.assertFalse(predict._embedding_enabled())
                os.environ["NA0S_EMBEDDING_ENABLED"] = "false"
                self.assertFalse(predict._embedding_enabled())

    def test_cascade_helper_honors_runtime_disable(self):
        with patch.object(cascade, "_HAS_EMBEDDING_CENTROID", True):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NA0S_EMBEDDING_ENABLED", None)
                self.assertTrue(cascade._embedding_enabled())
                os.environ["NA0S_EMBEDDING_ENABLED"] = "0"
                self.assertFalse(cascade._embedding_enabled())

    def test_helper_false_when_import_unavailable(self):
        """Env unset but import failed -> still disabled (default preserved)."""
        with patch.object(predict, "_HAS_EMBEDDING_CLASSIFIER", False):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NA0S_EMBEDDING_ENABLED", None)
                self.assertFalse(predict._embedding_enabled())

    def test_predict_and_cascade_helpers_agree(self):
        """Parity: identical inputs -> identical enabled verdict."""
        for env in (None, "0", "false", "1", "yes", ""):
            with patch.object(predict, "_HAS_EMBEDDING_CLASSIFIER", True), \
                 patch.object(cascade, "_HAS_EMBEDDING_CENTROID", True), \
                 patch.dict(os.environ, {}, clear=False):
                if env is None:
                    os.environ.pop("NA0S_EMBEDDING_ENABLED", None)
                else:
                    os.environ["NA0S_EMBEDDING_ENABLED"] = env
                self.assertEqual(
                    predict._embedding_enabled(),
                    cascade._embedding_enabled(),
                    "predict/cascade disagree on enabled for env={!r}".format(env),
                )


# ---------------------------------------------------------------------------
# End-to-end scan-time tests (require model files).
# ---------------------------------------------------------------------------
@unittest.skipUnless(_OK, _SKIP_REASON)
class TestPredictRuntimeToggle(unittest.TestCase):
    """predict.classify_prompt honors the env flipped AFTER import."""

    def _classify(self, text):
        return predict.classify_prompt(text, _vectorizer, _model)

    def test_runtime_disable_drops_embedding_contribution(self):
        """TEETH: a positive embedding score raises the composite when enabled,
        and contributes NOTHING once NA0S_EMBEDDING_ENABLED=0 is set after
        import.  Fails if the env read moves back to import time."""
        text = "What time does the museum open on Saturday afternoon?"
        fake = _FakeEmbedding(0.20, ["D1.1"])

        with patch.object(predict, "_HAS_EMBEDDING_CLASSIFIER", True), \
             patch.object(predict, "get_embedding_classifier", return_value=fake), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NA0S_EMBEDDING_ENABLED", None)
            _, comp_on, _, _, _, info_on, _ = self._classify(text)

            os.environ["NA0S_EMBEDDING_ENABLED"] = "0"
            fake.calls = 0
            _, comp_off, _, _, _, info_off, _ = self._classify(text)

        self.assertEqual(
            fake.calls, 0,
            "embedding classifier was still called with the env disabled",
        )
        self.assertEqual(info_off.get("score", 0.0), 0.0)
        self.assertGreater(info_on.get("score", 0.0), 0.0)
        self.assertGreater(
            comp_on, comp_off,
            "embedding boost survived a runtime disable — env read not at call site?",
        )


@unittest.skipUnless(_OK, _SKIP_REASON)
class TestCascadeRuntimeToggle(unittest.TestCase):
    """CascadeClassifier honors the env flipped AFTER import (parity)."""

    def _classify(self, text):
        return WeightedClassifier().classify(text, _vectorizer, _model)

    def test_runtime_disable_drops_embedding_contribution(self):
        text = "What time does the museum open on Saturday afternoon?"
        fake = _FakeEmbedding(0.20, ["D1.1"])

        with patch.object(cascade, "_HAS_EMBEDDING_CENTROID", True), \
             patch.object(cascade, "_get_centroid_classifier", return_value=fake), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NA0S_EMBEDDING_ENABLED", None)
            label_on, conf_on, hits_on = self._classify(text)

            os.environ["NA0S_EMBEDDING_ENABLED"] = "0"
            fake.calls = 0
            label_off, conf_off, hits_off = self._classify(text)

        self.assertEqual(
            fake.calls, 0,
            "centroid classifier was still called with the env disabled",
        )
        self.assertIn("embedding:D1.1", hits_on)
        self.assertFalse(any(h.startswith("embedding:") for h in hits_off))
        self.assertGreater(
            _mal_composite(label_on, conf_on), _mal_composite(label_off, conf_off),
            "centroid boost survived a runtime disable — env read not at call site?",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
