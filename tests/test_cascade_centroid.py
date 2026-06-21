"""Layer 5 parity: the centroid embedding classifier must feed the cascade.

predict.scan()/classify_prompt mixes a bounded semantic-similarity boost from
get_embedding_classifier() into its composite, but CascadeClassifier's
WeightedClassifier historically omitted it — so the two public entry points
could return different verdicts for the same input (the "L5 split").

These tests pin the wiring added in cascade.WeightedClassifier.classify(), with
a TEETH check that fails if the centroid fold is removed.
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

from na0s import cascade
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


class _FakeCentroid:
    """Stand-in for get_embedding_classifier() with a controllable score."""

    def __init__(self, score, matches):
        self._score = score
        self._matches = matches
        self.calls = 0

    def classify(self, text):
        self.calls += 1
        return self._score, self._matches


def _mal_composite(label, confidence):
    """Recover the underlying malicious composite from the (label, confidence)
    API.  confidence is P(label correct): composite when MALICIOUS, 1 - composite
    when SAFE — so this returns the raw composite either way."""
    return confidence if label == "MALICIOUS" else (1.0 - confidence)


@unittest.skipUnless(_OK, _SKIP_REASON)
class TestCascadeCentroidParity(unittest.TestCase):
    """The centroid embedding signal must reach CascadeClassifier."""

    def _classify(self, text):
        return WeightedClassifier().classify(text, _vectorizer, _model)

    def test_centroid_is_invoked(self):
        """WeightedClassifier.classify must actually call the centroid."""
        fake = _FakeCentroid(0.0, [])
        # Pin the runtime kill-switch ON: this test asserts the ENABLED-path
        # wiring, so it must not be subverted by an ambient NA0S_EMBEDDING_ENABLED=0
        # (the runtime flag is honored at call time as of the g7 hardening).
        with patch.dict(os.environ, {"NA0S_EMBEDDING_ENABLED": "1"}), \
             patch.object(cascade, "_HAS_EMBEDDING_CENTROID", True), \
             patch.object(cascade, "_get_centroid_classifier", return_value=fake):
            self._classify("hello, how are you doing today?")
        self.assertEqual(
            fake.calls, 1,
            "centroid classifier was not called by CascadeClassifier",
        )

    def test_centroid_boost_raises_composite_and_adds_hit(self):
        """TEETH: a positive centroid score must raise the malicious composite
        and surface an ``embedding:<technique>`` hit.  If the cascade fold is
        removed, the boosted and unboosted runs become identical and this
        fails."""
        text = "What time does the museum open on Saturday afternoon?"

        # Pin the runtime kill-switch ON for both legs (see test_centroid_is_invoked).
        fake_zero = _FakeCentroid(0.0, [])
        with patch.dict(os.environ, {"NA0S_EMBEDDING_ENABLED": "1"}), \
             patch.object(cascade, "_HAS_EMBEDDING_CENTROID", True), \
             patch.object(cascade, "_get_centroid_classifier", return_value=fake_zero):
            label0, conf0, hits0 = self._classify(text)

        fake_pos = _FakeCentroid(0.20, ["D1.1"])
        with patch.dict(os.environ, {"NA0S_EMBEDDING_ENABLED": "1"}), \
             patch.object(cascade, "_HAS_EMBEDDING_CENTROID", True), \
             patch.object(cascade, "_get_centroid_classifier", return_value=fake_pos):
            label1, conf1, hits1 = self._classify(text)

        self.assertIn("embedding:D1.1", hits1)
        self.assertNotIn("embedding:D1.1", hits0)
        self.assertGreater(
            _mal_composite(label1, conf1), _mal_composite(label0, conf0),
            "centroid boost did not raise the composite — cascade wiring missing?",
        )

    def test_disabled_contributes_nothing(self):
        """With the flag off (NA0S_EMBEDDING_ENABLED=0 parity), no call and no
        embedding hit."""
        fake = _FakeCentroid(0.20, ["D1.1"])
        with patch.object(cascade, "_HAS_EMBEDDING_CENTROID", False), \
             patch.object(cascade, "_get_centroid_classifier", return_value=fake):
            _, _, hits = self._classify(
                "Could you walk me through the configuration steps?"
            )
        self.assertEqual(fake.calls, 0)
        self.assertFalse(any(h.startswith("embedding:") for h in hits))

    def test_classifier_failure_is_non_fatal(self):
        """A raising centroid classifier must not break classification."""
        boom = patch.object(
            cascade, "_get_centroid_classifier",
            side_effect=RuntimeError("embedding backend exploded"),
        )
        with patch.object(cascade, "_HAS_EMBEDDING_CENTROID", True), boom:
            label, conf, hits = self._classify("What is the capital of France?")
        self.assertIn(label, ("SAFE", "MALICIOUS"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
