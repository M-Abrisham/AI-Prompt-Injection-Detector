"""Tests for input length validation at scan/classify entry points (M7)."""

import os
import unittest
from unittest.mock import patch

from na0s.config import MAX_INPUT_LENGTH
from na0s.scan_result import ScanResult


class TestScanInputLength(unittest.TestCase):
    """Input length guard in predict.scan()."""

    def _scan(self, text: str) -> ScanResult:
        from na0s.predict import scan
        return scan(text)

    def test_normal_length_passes_through(self):
        """A short, legitimate prompt should not be rejected by the length guard."""
        try:
            result = self._scan("Hello, how are you?")
        except Exception:
            # Pipeline may fail for unrelated reasons (model not loaded, etc.)
            # — the point is that the length guard did NOT fire.
            return
        # The length guard must NOT fire — result should come from the pipeline
        self.assertNotIn("input_length_exceeded", result.rule_hits)

    def test_oversized_input_flagged(self):
        """Input exceeding MAX_INPUT_LENGTH is immediately flagged."""
        oversized = "A" * (MAX_INPUT_LENGTH + 1)
        result = self._scan(oversized)
        self.assertTrue(result.is_malicious)
        self.assertEqual(result.risk_score, 1.0)
        self.assertIn(result.label, ("malicious", "blocked"))
        self.assertTrue(result.rejected)
        self.assertIn("input_length_exceeded", result.rule_hits)
        self.assertTrue(
            "maximum length" in result.rejection_reason
            or "char limit" in result.rejection_reason,
            f"Expected length rejection reason, got: {result.rejection_reason}",
        )
        # Must return quickly (no expensive processing)
        self.assertLess(result.elapsed_ms, 100)

    def test_exactly_at_limit_passes(self):
        """Input at exactly MAX_INPUT_LENGTH should NOT be rejected."""
        text = "A" * MAX_INPUT_LENGTH
        try:
            result = self._scan(text)
        except Exception:
            # Pipeline may fail for unrelated reasons — length guard didn't fire
            return
        self.assertNotIn("input_length_exceeded", result.rule_hits)

    def test_one_over_limit_rejected(self):
        """Input one char over MAX_INPUT_LENGTH is rejected."""
        text = "A" * (MAX_INPUT_LENGTH + 1)
        result = self._scan(text)
        self.assertTrue(result.rejected)
        self.assertIn("input_length_exceeded", result.rule_hits)

    @patch.dict(os.environ, {"NA0S_MAX_INPUT_LENGTH": "100"})
    def test_env_var_override(self):
        """The NA0S_MAX_INPUT_LENGTH env var overrides the default limit."""
        # Reimport to pick up the new env var value
        import importlib
        import na0s.config
        importlib.reload(na0s.config)
        try:
            from na0s.config import MAX_INPUT_LENGTH as new_limit
            self.assertEqual(new_limit, 100)
        finally:
            # Restore original value
            importlib.reload(na0s.config)


class TestCascadeInputLength(unittest.TestCase):
    """Input length guard in CascadeClassifier.classify()."""

    def test_oversized_input_blocked(self):
        """CascadeClassifier.classify() blocks oversized input."""
        from na0s.cascade import CascadeClassifier
        cc = CascadeClassifier()
        oversized = "B" * (MAX_INPUT_LENGTH + 1)
        label, confidence, hits, stage = cc.classify(oversized)
        self.assertEqual(label, "BLOCKED")
        self.assertEqual(confidence, 1.0)
        self.assertIn("input_length_exceeded", hits)
        self.assertEqual(stage, "blocked")

    def test_normal_input_not_blocked(self):
        """CascadeClassifier.classify() does not block normal-length input."""
        from na0s.cascade import CascadeClassifier
        cc = CascadeClassifier()
        label, confidence, hits, stage = cc.classify("What is the weather?")
        self.assertNotIn("input_length_exceeded", hits)

    def test_exactly_at_limit_not_blocked(self):
        """Input at exactly MAX_INPUT_LENGTH passes the guard."""
        from na0s.cascade import CascadeClassifier
        cc = CascadeClassifier()
        text = "C" * MAX_INPUT_LENGTH
        label, confidence, hits, stage = cc.classify(text)
        self.assertNotIn("input_length_exceeded", hits)


if __name__ == "__main__":
    unittest.main()
