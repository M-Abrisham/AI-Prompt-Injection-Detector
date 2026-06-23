"""F-AR8 Finding A: load-time feature-width reconciliation.

`_validate_feature_dimensions` cross-checks the assembled bundle width
(word vocab + optional char vocab + optional structural count) against
`model.n_features_in_` exactly once at load, and fails loud naming the
offending component instead of letting a mismatch surface as a cryptic
per-request `ValueError: X has N features` deep in `model.predict`.

These tests assert REAL load-time behavior — none mock away
`_validate_feature_dimensions` itself.

Run: python3 -m pytest tests/structural/test_dimension_guard.py -v
"""

import unittest
from unittest.mock import patch, MagicMock

import na0s.predict as predict
from na0s.structural.features import FEATURE_NAMES


def _save_caches():
    return (
        predict._cached_vectorizer,
        predict._cached_model,
        predict._cached_scaler,
        predict._cached_char_vectorizer,
        predict._dimensions_validated,
    )


def _restore_caches(saved):
    (
        predict._cached_vectorizer,
        predict._cached_model,
        predict._cached_scaler,
        predict._cached_char_vectorizer,
        predict._dimensions_validated,
    ) = saved


def _reset_caches():
    """Drop all cached artifacts + the once-only flag so the next load
    re-runs the reconciliation (the established reset pattern)."""
    predict._cached_vectorizer = None
    predict._cached_model = None
    predict._cached_scaler = None
    predict._cached_char_vectorizer = None
    predict._dimensions_validated = False


class TestDimensionGuard(unittest.TestCase):
    """Load-time width reconciliation (Finding A)."""

    def setUp(self):
        self._saved = _save_caches()

    def tearDown(self):
        # Strict save/restore so the process-wide _dimensions_validated and
        # _cached_* globals never leak into other tests.
        _restore_caches(self._saved)

    def test_valid_shipped_bundle_passes(self):
        """The real shipped bundle (10000 word + 0 char + 29 structural =
        10029) reconciles cleanly — guards against false positives."""
        _reset_caches()
        # preload_models() funnels through _get_cached_models(), which runs the
        # load-time guard. Must NOT raise on the valid bundle.
        predict.preload_models()
        vec, model = predict._get_cached_models()
        # Sanity: confirm the decomposition the guard reconciled.
        n = getattr(model, "n_features_in_", None)
        if n is not None:
            self.assertEqual(n, 10029)
        self.assertTrue(predict._dimensions_validated)

    def test_missing_scaler_against_structural_model_raises_named(self):
        """A dropped structural_scaler.pkl against a 10029-feature model is
        caught at load, and the error names the structural-scaler component."""
        _reset_caches()
        # Load the real word vectorizer + model, but simulate the scaler being
        # absent (dropped artifact). char vectorizer stays None (shipped state).
        vec, model = predict._get_cached_models()
        # Re-arm for a fresh validation against the patched loaders.
        predict._dimensions_validated = False
        with patch.object(predict, "_get_cached_scaler", return_value=None), \
             patch.object(predict, "_get_cached_char_vectorizer", return_value=None):
            with self.assertRaises(ValueError) as ctx:
                predict._validate_feature_dimensions(vec, model)
        msg = str(ctx.exception).lower()
        self.assertIn("structural scaler", msg)

    def test_wrong_vocab_word_vectorizer_raises(self):
        """A word vectorizer with a wrong vocab against the real model width
        fails loud naming a word-vocab width mismatch.

        Char and structural blocks are present-and-correct here (char vec
        supplies the remainder, structural scaler supplies 29), so the only
        thing off is the word vocab — the guard must attribute the mismatch to
        the word vocab, not to a missing char/structural block."""
        _reset_caches()
        _, model = predict._get_cached_models()
        # n_features_in_ == 10029 = word(10000) + char(0) + structural(29).
        # Make word short (9000) AND supply a char vec that fills 0 -> the
        # leftover 1000 delta is purely word-side, not a whole char/structural
        # block, so it classifies as a word-vocab mismatch.
        stub_vec = MagicMock()
        stub_vec.get_feature_names_out.return_value = list(range(9000))
        stub_char = MagicMock()
        stub_char.get_feature_names_out.return_value = []  # char present, 0 wide
        predict._dimensions_validated = False
        with patch.object(predict, "_get_cached_char_vectorizer", return_value=stub_char):
            with self.assertRaises(ValueError) as ctx:
                predict._validate_feature_dimensions(stub_vec, model)
        self.assertIn("word vocab", str(ctx.exception).lower())

    def test_non_sklearn_model_skips(self):
        """A model with n_features_in_=None (non-sklearn / mock) is skipped —
        backward compat, no raise."""
        _reset_caches()
        vec = MagicMock()
        vec.get_feature_names_out.return_value = list(range(10))
        model = MagicMock()
        model.n_features_in_ = None
        # Must not raise.
        predict._validate_feature_dimensions(vec, model)

    def test_structural_count_is_feature_names_len(self):
        """Guard against a magic number: the structural block the guard adds is
        len(FEATURE_NAMES), the canonical structural-feature ordering."""
        self.assertEqual(len(FEATURE_NAMES), 29)


if __name__ == "__main__":
    unittest.main()
