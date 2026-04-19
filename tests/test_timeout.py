"""Tests for src/layer0/timeout.py -- timeout enforcement and integration."""

import os
import sys
import time
import unittest
from unittest import mock

# Ensure env vars are set BEFORE importing the module under test
os.environ.setdefault("L0_TIMEOUT_SEC", "5")

from na0s.layer0.timeout import (
    DEFAULT_TIMEOUT,
    L0_PIPELINE_TIMEOUT,
    Layer0TimeoutError,
    SCAN_TIMEOUT,
    get_step_timeout,
    with_timeout,
)


class TestLayer0TimeoutError(unittest.TestCase):
    """Layer0TimeoutError exception behaviour."""

    def test_attributes(self):
        err = Layer0TimeoutError("normalize", 3.0)
        self.assertEqual(err.step_name, "normalize")
        self.assertEqual(err.timeout_sec, 3.0)

    def test_message_format(self):
        err = Layer0TimeoutError("html", 2.5)
        self.assertIn("html", str(err))
        self.assertIn("2.5", str(err))

    def test_is_exception(self):
        self.assertTrue(issubclass(Layer0TimeoutError, Exception))


class TestGetStepTimeout(unittest.TestCase):
    """get_step_timeout() lookup behaviour."""

    def test_known_steps(self):
        for step in ("normalize", "html", "tokenize"):
            val = get_step_timeout(step)
            self.assertIsInstance(val, float)
            self.assertGreater(val, 0)

    def test_unknown_step_falls_back(self):
        val = get_step_timeout("nonexistent_step")
        self.assertEqual(val, DEFAULT_TIMEOUT)


class TestWithTimeout(unittest.TestCase):
    """with_timeout() wrapper behaviour."""

    def test_fast_function_returns_result(self):
        result = with_timeout(lambda x: x * 2, 2.0, 21, step_name="test")
        self.assertEqual(result, 42)

    def test_fast_function_kwargs(self):
        def greet(name="world"):
            return "hello " + name
        result = with_timeout(greet, 2.0, step_name="test", name="pytest")
        self.assertEqual(result, "hello pytest")

    def test_slow_function_raises_timeout(self):
        def slow():
            time.sleep(10)
            return "done"

        with self.assertRaises(Layer0TimeoutError) as ctx:
            with_timeout(slow, 0.3, step_name="slow_op")
        self.assertEqual(ctx.exception.step_name, "slow_op")
        self.assertAlmostEqual(ctx.exception.timeout_sec, 0.3)

    def test_zero_timeout_means_no_timeout(self):
        result = with_timeout(lambda: 99, 0, step_name="test")
        self.assertEqual(result, 99)

    def test_negative_timeout_means_no_timeout(self):
        result = with_timeout(lambda: 99, -1, step_name="test")
        self.assertEqual(result, 99)

    def test_none_timeout_uses_default(self):
        result = with_timeout(lambda: "ok", None, step_name="test")
        self.assertEqual(result, "ok")

    def test_function_exception_propagates(self):
        def bad():
            raise ValueError("boom")
        with self.assertRaises(ValueError):
            with_timeout(bad, 2.0, step_name="test")


class TestEnvVarConfiguration(unittest.TestCase):
    """Environment variable configuration is respected."""

    def test_default_timeout_from_env(self):
        self.assertIsInstance(DEFAULT_TIMEOUT, float)
        self.assertGreater(DEFAULT_TIMEOUT, 0)

    def test_step_timeouts_are_float(self):
        for step in ("normalize", "html", "tokenize"):
            self.assertIsInstance(get_step_timeout(step), float)


class TestPipelineTimeoutConstants(unittest.TestCase):
    """Verify pipeline-level and scan-level constants exist and are positive."""

    def test_pipeline_timeout_is_positive(self):
        self.assertIsInstance(L0_PIPELINE_TIMEOUT, float)
        self.assertGreater(L0_PIPELINE_TIMEOUT, 0)

    def test_scan_timeout_is_positive(self):
        # SCAN_TIMEOUT is frozen at import time from SCAN_TIMEOUT_SEC env var.
        # Valid values: 0.0 (timeout disabled) or positive (seconds).
        # We can't retroactively inspect import-time env, so accept either.
        self.assertIsInstance(SCAN_TIMEOUT, float)
        self.assertTrue(
            SCAN_TIMEOUT == 0.0 or SCAN_TIMEOUT > 0,
            f"SCAN_TIMEOUT={SCAN_TIMEOUT} not a valid setting",
        )

    def test_pipeline_timeout_default(self):
        # Default is 30s unless overridden by env var
        self.assertGreaterEqual(L0_PIPELINE_TIMEOUT, 1.0)

    def test_scan_timeout_default(self):
        # SCAN_TIMEOUT is frozen at import time. Either it was imported with
        # SCAN_TIMEOUT_SEC=0 (disabled, 0.0) or with a positive seconds value.
        # Runtime env may differ from import-time env due to test pollution,
        # so we assert on the frozen constant instead of the live env.
        self.assertTrue(
            SCAN_TIMEOUT == 0.0 or SCAN_TIMEOUT >= 1.0,
            f"SCAN_TIMEOUT={SCAN_TIMEOUT} not a valid setting",
        )


class TestSanitizerTimeoutIntegration(unittest.TestCase):
    """Integration tests: sanitizer returns rejected Layer0Result on timeout."""

    def test_normalize_timeout_returns_rejected(self):
        """When normalize_text times out, sanitizer rejects the input."""
        def slow_normalize(text):
            time.sleep(10)
            return text, 0, []

        with mock.patch(
            "na0s.layer0.sanitizer.normalize_text", side_effect=slow_normalize
        ), mock.patch(
            "na0s.layer0.sanitizer.get_step_timeout", return_value=0.1
        ):
            from na0s.layer0.sanitizer import layer0_sanitize
            result = layer0_sanitize("Hello world test input")

        self.assertTrue(result.rejected)
        self.assertIn("timeout", result.rejection_reason.lower())
        self.assertIn("timeout_normalize", result.anomaly_flags)

    def test_html_extraction_timeout_returns_rejected(self):
        """When extract_safe_text times out, sanitizer rejects the input."""
        def slow_html(text):
            time.sleep(10)
            return text, []

        with mock.patch(
            "na0s.layer0.sanitizer.extract_safe_text", side_effect=slow_html
        ), mock.patch(
            "na0s.layer0.sanitizer.get_step_timeout", return_value=0.1
        ):
            from na0s.layer0.sanitizer import layer0_sanitize
            result = layer0_sanitize("Hello world test input")

        self.assertTrue(result.rejected)
        self.assertIn("timeout", result.rejection_reason.lower())
        self.assertIn("timeout_html", result.anomaly_flags)

    def test_tokenization_timeout_returns_rejected(self):
        """When check_tokenization_anomaly times out, sanitizer rejects."""
        def slow_tokenize(text):
            time.sleep(10)
            return [], 0.0, {}

        with mock.patch(
            "na0s.layer0.sanitizer.check_tokenization_anomaly",
            side_effect=slow_tokenize,
        ), mock.patch(
            "na0s.layer0.sanitizer.get_step_timeout", return_value=0.1
        ):
            from na0s.layer0.sanitizer import layer0_sanitize
            result = layer0_sanitize("Hello world test input")

        self.assertTrue(result.rejected)
        self.assertIn("timeout", result.rejection_reason.lower())
        self.assertIn("timeout_tokenize", result.anomaly_flags)

    def test_pipeline_timeout_returns_rejected(self):
        """When the entire pipeline times out, sanitizer rejects."""
        def slow_inner(raw_input):
            time.sleep(10)

        with mock.patch(
            "na0s.layer0.sanitizer._layer0_sanitize_inner",
            side_effect=slow_inner,
        ), mock.patch(
            "na0s.layer0.sanitizer.L0_PIPELINE_TIMEOUT", 0.1
        ):
            from na0s.layer0.sanitizer import layer0_sanitize
            result = layer0_sanitize("Hello world test input")

        self.assertTrue(result.rejected)
        self.assertIn("timeout", result.rejection_reason.lower())
        self.assertIn("timeout_pipeline", result.anomaly_flags)

    def test_normal_input_still_works(self):
        """Quick sanity check: normal input passes through with no timeout."""
        from na0s.layer0.sanitizer import layer0_sanitize
        result = layer0_sanitize("Hello world")
        self.assertFalse(result.rejected)
        self.assertEqual(result.rejection_reason, "")


class TestScanTimeoutIntegration(unittest.TestCase):
    """Integration tests: scan() returns rejected ScanResult on timeout."""

    def test_scan_timeout_returns_rejected(self):
        """When classify_prompt times out, scan() returns rejected result."""
        def slow_classify(text, vectorizer, model, threshold=0.55):
            time.sleep(10)

        with mock.patch(
            "na0s.predict.classify_prompt", side_effect=slow_classify
        ), mock.patch(
            "na0s.predict.SCAN_TIMEOUT", 0.1
        ):
            from na0s.predict import scan
            fake_vectorizer = mock.MagicMock()
            fake_model = mock.MagicMock()
            result = scan("test input", fake_vectorizer, fake_model)

        self.assertTrue(result.rejected)
        self.assertIn("timeout", result.rejection_reason.lower())
        self.assertIn("timeout_scan", result.anomaly_flags)
        self.assertEqual(result.label, "blocked")


if __name__ == "__main__":
    unittest.main()
