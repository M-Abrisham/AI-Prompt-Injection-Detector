"""Tests for Layer 0 externalized configuration constants.

Verifies that:
- Named constants exist with expected default values.
- Environment variables override defaults when set.
- Invalid environment variable values fall back to defaults gracefully.
"""

import importlib
import os
import unittest
from unittest.mock import patch


class TestNormalizationConstants(unittest.TestCase):
    """Tests for normalization.py configurable thresholds."""

    def _reload_module(self):
        """Force-reload the module so module-level env reads re-execute."""
        import na0s.layer0.normalization as mod
        importlib.reload(mod)
        return mod

    # --- _NFKC_CHANGE_THRESHOLD ---

    def test_nfkc_threshold_default(self):
        mod = self._reload_module()
        self.assertEqual(mod._NFKC_CHANGE_THRESHOLD, 0.25)

    def test_nfkc_threshold_env_override(self):
        with patch.dict(os.environ, {"L0_NFKC_CHANGE_THRESHOLD": "0.4"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._NFKC_CHANGE_THRESHOLD, 0.4)

    def test_nfkc_threshold_env_invalid_string(self):
        with patch.dict(os.environ, {"L0_NFKC_CHANGE_THRESHOLD": "not_a_number"}):
            mod = self._reload_module()
            self.assertEqual(mod._NFKC_CHANGE_THRESHOLD, 0.25)

    def test_nfkc_threshold_env_out_of_range_high(self):
        with patch.dict(os.environ, {"L0_NFKC_CHANGE_THRESHOLD": "1.5"}):
            mod = self._reload_module()
            self.assertEqual(mod._NFKC_CHANGE_THRESHOLD, 0.25)

    def test_nfkc_threshold_env_out_of_range_negative(self):
        with patch.dict(os.environ, {"L0_NFKC_CHANGE_THRESHOLD": "-0.1"}):
            mod = self._reload_module()
            self.assertEqual(mod._NFKC_CHANGE_THRESHOLD, 0.25)

    def test_nfkc_threshold_env_boundary_zero(self):
        with patch.dict(os.environ, {"L0_NFKC_CHANGE_THRESHOLD": "0.0"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._NFKC_CHANGE_THRESHOLD, 0.0)

    def test_nfkc_threshold_env_boundary_one(self):
        with patch.dict(os.environ, {"L0_NFKC_CHANGE_THRESHOLD": "1.0"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._NFKC_CHANGE_THRESHOLD, 1.0)

    def test_nfkc_threshold_env_nan(self):
        """NaN must fall back to default — NaN bypasses all comparisons."""
        with patch.dict(os.environ, {"L0_NFKC_CHANGE_THRESHOLD": "nan"}):
            mod = self._reload_module()
            self.assertEqual(mod._NFKC_CHANGE_THRESHOLD, 0.25)

    def test_nfkc_threshold_env_inf(self):
        """Infinity must fall back to default."""
        with patch.dict(os.environ, {"L0_NFKC_CHANGE_THRESHOLD": "inf"}):
            mod = self._reload_module()
            self.assertEqual(mod._NFKC_CHANGE_THRESHOLD, 0.25)

    def test_nfkc_threshold_env_negative_inf(self):
        """Negative infinity must fall back to default."""
        with patch.dict(os.environ, {"L0_NFKC_CHANGE_THRESHOLD": "-inf"}):
            mod = self._reload_module()
            self.assertEqual(mod._NFKC_CHANGE_THRESHOLD, 0.25)

    # --- _INVISIBLE_CHARS_THRESHOLD ---

    def test_invisible_threshold_default(self):
        mod = self._reload_module()
        self.assertEqual(mod._INVISIBLE_CHARS_THRESHOLD, 2)

    def test_invisible_threshold_env_override(self):
        with patch.dict(os.environ, {"L0_INVISIBLE_CHARS_THRESHOLD": "5"}):
            mod = self._reload_module()
            self.assertEqual(mod._INVISIBLE_CHARS_THRESHOLD, 5)

    def test_invisible_threshold_env_invalid_string(self):
        with patch.dict(os.environ, {"L0_INVISIBLE_CHARS_THRESHOLD": "abc"}):
            mod = self._reload_module()
            self.assertEqual(mod._INVISIBLE_CHARS_THRESHOLD, 2)

    def test_invisible_threshold_env_negative(self):
        with patch.dict(os.environ, {"L0_INVISIBLE_CHARS_THRESHOLD": "-1"}):
            mod = self._reload_module()
            self.assertEqual(mod._INVISIBLE_CHARS_THRESHOLD, 2)

    def test_invisible_threshold_env_zero(self):
        """Zero is valid -- flag all invisible chars."""
        with patch.dict(os.environ, {"L0_INVISIBLE_CHARS_THRESHOLD": "0"}):
            mod = self._reload_module()
            self.assertEqual(mod._INVISIBLE_CHARS_THRESHOLD, 0)


class TestTokenizationConstants(unittest.TestCase):
    """Tests for tokenization.py configurable thresholds."""

    def _reload_module(self):
        import na0s.layer0.tokenization as mod
        importlib.reload(mod)
        return mod

    # --- GLOBAL_RATIO_THRESHOLD ---

    def test_global_ratio_default(self):
        mod = self._reload_module()
        self.assertAlmostEqual(mod.GLOBAL_RATIO_THRESHOLD, 0.75)

    def test_global_ratio_env_override(self):
        with patch.dict(os.environ, {"L0_GLOBAL_RATIO_THRESHOLD": "0.6"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod.GLOBAL_RATIO_THRESHOLD, 0.6)

    def test_global_ratio_env_invalid(self):
        with patch.dict(os.environ, {"L0_GLOBAL_RATIO_THRESHOLD": "nope"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod.GLOBAL_RATIO_THRESHOLD, 0.75)

    def test_global_ratio_env_out_of_range(self):
        with patch.dict(os.environ, {"L0_GLOBAL_RATIO_THRESHOLD": "2.0"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod.GLOBAL_RATIO_THRESHOLD, 0.75)

    def test_global_ratio_env_nan(self):
        """NaN must fall back to default."""
        with patch.dict(os.environ, {"L0_GLOBAL_RATIO_THRESHOLD": "nan"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod.GLOBAL_RATIO_THRESHOLD, 0.75)

    def test_global_ratio_env_inf(self):
        """Infinity must fall back to default."""
        with patch.dict(os.environ, {"L0_GLOBAL_RATIO_THRESHOLD": "inf"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod.GLOBAL_RATIO_THRESHOLD, 0.75)

    # --- WINDOW_RATIO_THRESHOLD ---

    def test_window_ratio_default(self):
        mod = self._reload_module()
        self.assertAlmostEqual(mod.WINDOW_RATIO_THRESHOLD, 0.85)

    def test_window_ratio_env_override(self):
        with patch.dict(os.environ, {"L0_WINDOW_RATIO_THRESHOLD": "0.9"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod.WINDOW_RATIO_THRESHOLD, 0.9)

    def test_window_ratio_env_invalid(self):
        with patch.dict(os.environ, {"L0_WINDOW_RATIO_THRESHOLD": "bad"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod.WINDOW_RATIO_THRESHOLD, 0.85)

    # --- WINDOW_SIZE ---

    def test_window_size_default(self):
        mod = self._reload_module()
        self.assertEqual(mod.WINDOW_SIZE, 50)

    def test_window_size_env_override(self):
        with patch.dict(os.environ, {"L0_WINDOW_SIZE": "100"}):
            mod = self._reload_module()
            self.assertEqual(mod.WINDOW_SIZE, 100)

    def test_window_size_env_too_small(self):
        """Window size below minimum (10) falls back to default."""
        with patch.dict(os.environ, {"L0_WINDOW_SIZE": "5"}):
            mod = self._reload_module()
            self.assertEqual(mod.WINDOW_SIZE, 50)

    def test_window_size_env_invalid(self):
        with patch.dict(os.environ, {"L0_WINDOW_SIZE": "xyz"}):
            mod = self._reload_module()
            self.assertEqual(mod.WINDOW_SIZE, 50)

    # --- _CJK_FRACTION_THRESHOLD ---

    def test_cjk_fraction_default(self):
        mod = self._reload_module()
        self.assertAlmostEqual(mod._CJK_FRACTION_THRESHOLD, 0.3)

    def test_cjk_fraction_env_override(self):
        with patch.dict(os.environ, {"L0_CJK_FRACTION_THRESHOLD": "0.5"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._CJK_FRACTION_THRESHOLD, 0.5)

    def test_cjk_fraction_env_invalid(self):
        with patch.dict(os.environ, {"L0_CJK_FRACTION_THRESHOLD": "invalid"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._CJK_FRACTION_THRESHOLD, 0.3)

    # --- _MIN_TEXT_LENGTH_FOR_TOKENIZATION ---

    def test_min_text_length_default(self):
        mod = self._reload_module()
        self.assertEqual(mod._MIN_TEXT_LENGTH_FOR_TOKENIZATION, 10)

    def test_min_text_length_env_override(self):
        with patch.dict(os.environ, {"L0_MIN_TEXT_LENGTH_FOR_TOKENIZATION": "20"}):
            mod = self._reload_module()
            self.assertEqual(mod._MIN_TEXT_LENGTH_FOR_TOKENIZATION, 20)

    def test_min_text_length_env_zero(self):
        """Zero is below minimum (1), falls back to default."""
        with patch.dict(os.environ, {"L0_MIN_TEXT_LENGTH_FOR_TOKENIZATION": "0"}):
            mod = self._reload_module()
            self.assertEqual(mod._MIN_TEXT_LENGTH_FOR_TOKENIZATION, 10)

    def test_min_text_length_env_invalid(self):
        with patch.dict(os.environ, {"L0_MIN_TEXT_LENGTH_FOR_TOKENIZATION": "nah"}):
            mod = self._reload_module()
            self.assertEqual(mod._MIN_TEXT_LENGTH_FOR_TOKENIZATION, 10)


class TestEncodingConstants(unittest.TestCase):
    """Tests for encoding.py configurable thresholds."""

    def _reload_module(self):
        import na0s.layer0.encoding as mod
        importlib.reload(mod)
        return mod

    # --- _MIN_CONFIDENCE ---

    def test_min_confidence_default(self):
        mod = self._reload_module()
        self.assertAlmostEqual(mod._MIN_CONFIDENCE, 0.5)

    def test_min_confidence_env_override(self):
        with patch.dict(os.environ, {"L0_MIN_ENCODING_CONFIDENCE": "0.8"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._MIN_CONFIDENCE, 0.8)

    def test_min_confidence_env_invalid(self):
        with patch.dict(os.environ, {"L0_MIN_ENCODING_CONFIDENCE": "broken"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._MIN_CONFIDENCE, 0.5)

    def test_min_confidence_env_out_of_range(self):
        with patch.dict(os.environ, {"L0_MIN_ENCODING_CONFIDENCE": "1.5"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._MIN_CONFIDENCE, 0.5)

    def test_min_confidence_env_negative(self):
        with patch.dict(os.environ, {"L0_MIN_ENCODING_CONFIDENCE": "-0.1"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._MIN_CONFIDENCE, 0.5)

    def test_min_confidence_env_nan(self):
        """NaN must fall back to default."""
        with patch.dict(os.environ, {"L0_MIN_ENCODING_CONFIDENCE": "nan"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._MIN_CONFIDENCE, 0.5)

    def test_min_confidence_env_inf(self):
        """Infinity must fall back to default."""
        with patch.dict(os.environ, {"L0_MIN_ENCODING_CONFIDENCE": "inf"}):
            mod = self._reload_module()
            self.assertAlmostEqual(mod._MIN_CONFIDENCE, 0.5)


if __name__ == "__main__":
    unittest.main()
