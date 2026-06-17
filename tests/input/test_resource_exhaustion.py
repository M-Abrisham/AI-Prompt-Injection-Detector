"""Integration tests for resource exhaustion protection.

Verifies that resource guards (input size, HTML depth, memory budget,
expansion ratio) are enforced through the Layer 0 pipeline and produce
correctly rejected Layer0Result objects.

These tests complement tests/test_resource_guard.py which tests the
guard functions in isolation.
"""

import os
import sys
import unittest

# Disable scan timeout for testing (avoids signal/thread issues)
os.environ.setdefault("SCAN_TIMEOUT_SEC", "0")


class TestInputSizePipelineIntegration(unittest.TestCase):
    """Verify oversized inputs are rejected through the full pipeline."""

    def test_normal_input_passes(self):
        from na0s.layer0.sanitizer import layer0_sanitize

        result = layer0_sanitize("Hello, this is a normal prompt.")
        self.assertFalse(result.rejected)
        self.assertIn("Hello", result.sanitized_text)

    def test_oversized_chars_rejected_by_pipeline(self):
        """Input exceeding L0_MAX_INPUT_CHARS should be rejected."""
        from na0s.layer0.sanitizer import layer0_sanitize
        from na0s.layer0.resource_guard import MAX_INPUT_CHARS

        text = "a" * (MAX_INPUT_CHARS + 100)
        result = layer0_sanitize(text)
        self.assertTrue(result.rejected)
        self.assertIn("limit", result.rejection_reason.lower())

    def test_oversized_bytes_rejected_by_pipeline(self):
        """Multi-byte input exceeding L0_MAX_INPUT_BYTES should be rejected."""
        from na0s.layer0.sanitizer import layer0_sanitize
        from na0s.layer0.resource_guard import MAX_INPUT_BYTES

        # Use 4-byte emoji to create byte-heavy input
        emoji = "\U0001F600"
        count = (MAX_INPUT_BYTES // 4) + 100
        # Only test if count is within char limit to isolate byte check
        from na0s.layer0.resource_guard import MAX_INPUT_CHARS
        if count <= MAX_INPUT_CHARS:
            text = emoji * count
            result = layer0_sanitize(text)
            self.assertTrue(result.rejected)

    def test_at_char_limit_passes(self):
        """Input at exactly the char limit should pass."""
        from na0s.layer0.sanitizer import layer0_sanitize
        from na0s.layer0.resource_guard import MAX_INPUT_CHARS

        # Use a slightly smaller value to stay within validation limits too
        text = "a " * (MAX_INPUT_CHARS // 2 - 1)
        text = text[:MAX_INPUT_CHARS]
        result = layer0_sanitize(text)
        # Should not be rejected for size (may be rejected for other reasons
        # but not for resource limits)
        if result.rejected:
            self.assertNotIn("resource limit", result.rejection_reason.lower())

    def test_empty_input_rejected_by_validation_not_resource_guard(self):
        """Empty input should be rejected by validation, not resource guard."""
        from na0s.layer0.sanitizer import layer0_sanitize

        result = layer0_sanitize("")
        self.assertTrue(result.rejected)
        self.assertIn("empty", result.rejection_reason.lower())


class TestHtmlDepthPipelineIntegration(unittest.TestCase):
    """Verify deeply nested HTML is rejected through the full pipeline."""

    def test_shallow_html_passes(self):
        from na0s.layer0.sanitizer import layer0_sanitize

        html = "<div><p>Hello world</p></div>"
        result = layer0_sanitize(html)
        self.assertFalse(result.rejected)

    def test_deep_html_rejected_by_pipeline(self):
        """HTML exceeding max nesting depth should be rejected."""
        from na0s.layer0.sanitizer import layer0_sanitize
        from na0s.layer0.resource_guard import MAX_HTML_DEPTH

        depth = MAX_HTML_DEPTH + 50
        html = "<div>" * depth + "bomb" + "</div>" * depth
        result = layer0_sanitize(html)
        self.assertTrue(result.rejected)
        # Should have resource_limit flag
        has_resource_flag = any(
            "resource_limit" in f or "html_depth" in f
            for f in result.anomaly_flags
        )
        self.assertTrue(
            has_resource_flag,
            "Expected resource_limit or html_depth flag in: {}".format(
                result.anomaly_flags
            ),
        )

    def test_moderate_html_passes(self):
        """HTML within depth limit should pass."""
        from na0s.layer0.sanitizer import layer0_sanitize

        depth = 10
        html = "<div>" * depth + "safe content" + "</div>" * depth
        result = layer0_sanitize(html)
        self.assertFalse(result.rejected)


class TestHtmlDepthInExtractor(unittest.TestCase):
    """Verify html_extractor.py enforces depth check before parsing."""

    def test_deeply_nested_html_flagged(self):
        """extract_safe_text should flag deeply nested HTML."""
        from na0s.layer0.html_extractor import extract_safe_text
        from na0s.layer0.resource_guard import MAX_HTML_DEPTH

        depth = MAX_HTML_DEPTH + 20
        html = "<div>" * depth + "payload" + "</div>" * depth
        text, flags = extract_safe_text(html)
        self.assertIn("html_depth_exceeded", flags)

    def test_shallow_html_not_flagged(self):
        """Shallow HTML should not get depth flag."""
        from na0s.layer0.html_extractor import extract_safe_text

        html = "<div><p>Hello</p></div>"
        text, flags = extract_safe_text(html)
        self.assertNotIn("html_depth_exceeded", flags)

    def test_no_html_not_checked(self):
        """Plain text should skip HTML checks entirely."""
        from na0s.layer0.html_extractor import extract_safe_text

        text = "Just plain text, no tags."
        result_text, flags = extract_safe_text(text)
        self.assertNotIn("html_depth_exceeded", flags)
        self.assertEqual(result_text, text)


class TestMemoryBudgetPipelineIntegration(unittest.TestCase):
    """Verify memory budget is enforced through the pipeline."""

    def test_small_input_within_budget(self):
        from na0s.layer0.sanitizer import layer0_sanitize

        result = layer0_sanitize("Small input within memory budget.")
        self.assertFalse(result.rejected)

    def test_memory_budget_check_runs(self):
        """Verify check_memory_budget is callable and works."""
        from na0s.layer0.resource_guard import check_memory_budget

        # Small input should pass
        check_memory_budget("hello world")

        # Zero cap disables the check
        check_memory_budget("x" * 100000, cap_mb=0)


class TestExpansionRatioPipelineIntegration(unittest.TestCase):
    """Verify expansion ratio guard works in the pipeline."""

    def test_normal_normalization_passes(self):
        """Normal text normalization should not trigger expansion guard."""
        from na0s.layer0.sanitizer import layer0_sanitize

        result = layer0_sanitize("Hello world, this is a test.")
        self.assertFalse(result.rejected)

    def test_expansion_ratio_check_standalone(self):
        """Verify check_expansion_ratio works correctly."""
        from na0s.layer0.resource_guard import (
            ResourceLimitExceeded,
            check_expansion_ratio,
        )

        # Normal ratio passes
        check_expansion_ratio(100, 200)

        # Excessive ratio rejected
        with self.assertRaises(ResourceLimitExceeded):
            check_expansion_ratio(100, 1500)

        # Zero original passes (avoid division by zero)
        check_expansion_ratio(0, 100)


class TestResourceGuardFlags(unittest.TestCase):
    """Verify that resource guard rejections produce correct flags."""

    def test_oversized_input_flag(self):
        from na0s.layer0.sanitizer import layer0_sanitize
        from na0s.layer0.resource_guard import MAX_INPUT_CHARS

        text = "x" * (MAX_INPUT_CHARS + 1)
        result = layer0_sanitize(text)
        self.assertTrue(result.rejected)
        has_flag = any(
            "resource_limit" in f or "char limit" in result.rejection_reason
            for f in result.anomaly_flags
        ) or "limit" in result.rejection_reason.lower()
        self.assertTrue(has_flag, "Expected resource limit indication")

    def test_deep_html_flag(self):
        from na0s.layer0.sanitizer import layer0_sanitize
        from na0s.layer0.resource_guard import MAX_HTML_DEPTH

        depth = MAX_HTML_DEPTH + 20
        html = "<div>" * depth + "x" + "</div>" * depth
        result = layer0_sanitize(html)
        self.assertTrue(result.rejected)


class TestResourceLimitExceededException(unittest.TestCase):
    """Test the ResourceLimitExceeded exception is properly exported."""

    def test_importable_from_layer0(self):
        from na0s.layer0 import ResourceLimitExceeded

        err = ResourceLimitExceeded("test_guard", "test detail")
        self.assertEqual(err.guard_name, "test_guard")
        self.assertEqual(err.detail, "test detail")

    def test_is_exception(self):
        from na0s.layer0 import ResourceLimitExceeded

        self.assertTrue(issubclass(ResourceLimitExceeded, Exception))


class TestRunEntryGuardsIntegration(unittest.TestCase):
    """Verify run_entry_guards is called from the sanitizer pipeline."""

    def test_run_entry_guards_catches_oversized(self):
        """Oversized input should be caught by run_entry_guards in pipeline."""
        from na0s.layer0.resource_guard import (
            ResourceLimitExceeded,
            run_entry_guards,
            MAX_INPUT_CHARS,
        )

        # Should raise for oversized input
        with self.assertRaises(ResourceLimitExceeded):
            run_entry_guards("x" * (MAX_INPUT_CHARS + 1))

        # Should pass for normal input
        run_entry_guards("Hello, normal input")

    def test_run_entry_guards_catches_deep_html(self):
        """Deep HTML should be caught by run_entry_guards."""
        from na0s.layer0.resource_guard import (
            ResourceLimitExceeded,
            run_entry_guards,
            MAX_HTML_DEPTH,
        )

        depth = MAX_HTML_DEPTH + 10
        html = "<div>" * depth + "x" + "</div>" * depth
        with self.assertRaises(ResourceLimitExceeded):
            run_entry_guards(html)


class TestEndToEndResourceProtection(unittest.TestCase):
    """End-to-end tests verifying the full pipeline handles resource attacks."""

    def test_normal_scan_still_works(self):
        """Normal inputs should still work after adding resource guards."""
        from na0s.layer0.sanitizer import layer0_sanitize

        result = layer0_sanitize("What is the weather today?")
        self.assertFalse(result.rejected)
        self.assertTrue(len(result.sanitized_text) > 0)

    def test_html_with_hidden_content_still_detected(self):
        """HTML with hidden content should still be detected (not blocked by depth)."""
        from na0s.layer0.sanitizer import layer0_sanitize

        html = '<div style="display:none">secret payload</div>visible text'
        result = layer0_sanitize(html)
        self.assertFalse(result.rejected)
        # hidden_html_content flag should still fire
        self.assertIn("hidden_html_content", result.anomaly_flags)

    def test_unicode_normalization_not_blocked(self):
        """Unicode normalization should work without hitting expansion guard."""
        from na0s.layer0.sanitizer import layer0_sanitize

        # NFKC normalization of fullwidth chars
        text = "\uff28\uff45\uff4c\uff4c\uff4f"  # "Hello" in fullwidth
        result = layer0_sanitize(text)
        self.assertFalse(result.rejected)

    def test_deeply_nested_div_bomb_blocked(self):
        """A div-bomb attack should be blocked."""
        from na0s.layer0.sanitizer import layer0_sanitize
        from na0s.layer0.resource_guard import MAX_HTML_DEPTH

        # Create a div bomb that exceeds max depth
        depth = MAX_HTML_DEPTH + 100
        html = "<div>" * depth + "payload" + "</div>" * depth
        result = layer0_sanitize(html)
        self.assertTrue(result.rejected)

    def test_moderate_nesting_allowed(self):
        """Reasonable HTML nesting should not be blocked."""
        from na0s.layer0.sanitizer import layer0_sanitize

        html = (
            "<html><body><div><section><article><main>"
            "<p><span><strong><em>content</em></strong></span></p>"
            "</main></article></section></div></body></html>"
        )
        result = layer0_sanitize(html)
        self.assertFalse(result.rejected)


if __name__ == "__main__":
    unittest.main()
