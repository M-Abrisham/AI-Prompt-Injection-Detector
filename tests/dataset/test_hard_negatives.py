"""Regression tests for hard-negative mining model resolution.

Guards the auto-retrain cold-start fix: on a fresh CI runner the pipeline
mines hard negatives BEFORE any model is trained, so there is no
feature-compatible model to run the FP-diagnostic phases against. The
resolver must return (None, None) (so those phases are skipped) and must
NOT fall back to the bundled na0s.models/ production model — that model is
TF-IDF + structural (10029 dims) and is feature-incompatible with phase1's
TF-IDF-only (10000) matrix (``ValueError: X has 10000 features ... 10029``).
"""

import unittest
from unittest import mock

from na0s.dataset import hard_negatives


class TestResolveModelArtifacts(unittest.TestCase):
    def test_cold_start_returns_none_with_no_production_fallback(self):
        # No freshly-trained data/processed/ model present -> (None, None),
        # never the incompatible bundled production model.
        with mock.patch.object(hard_negatives.os.path, "isfile", return_value=False):
            model, vectorizer = hard_negatives._resolve_model_artifacts()
        self.assertIsNone(model)
        self.assertIsNone(vectorizer)

    def test_warm_loads_only_the_freshly_trained_pair(self):
        sentinel_model, sentinel_vec = object(), object()
        loaded_paths = []

        def fake_safe_load(path):
            loaded_paths.append(path)
            return sentinel_model if path == hard_negatives.MODEL_PATH else sentinel_vec

        with mock.patch.object(hard_negatives.os.path, "isfile", return_value=True), \
                mock.patch.object(hard_negatives, "safe_load", side_effect=fake_safe_load):
            model, vectorizer = hard_negatives._resolve_model_artifacts()

        self.assertIs(model, sentinel_model)
        self.assertIs(vectorizer, sentinel_vec)
        # Only the data/processed/ pair is touched — no na0s.models/ fallback.
        self.assertEqual(
            set(loaded_paths),
            {hard_negatives.MODEL_PATH, hard_negatives.VECTORIZER_PATH},
        )


if __name__ == "__main__":
    unittest.main()
