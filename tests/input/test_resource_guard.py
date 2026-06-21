"""Tests for src/layer0/resource_guard.py -- resource exhaustion protection."""

import os
import unittest

# Set env vars BEFORE importing the module under test
os.environ["L0_MAX_INPUT_CHARS"] = "50000"
os.environ["L0_MAX_INPUT_BYTES"] = "200000"
os.environ["L0_MAX_HTML_DEPTH"] = "100"
os.environ["L0_MAX_EXPANSION_RATIO"] = "10.0"
os.environ["L0_MEMORY_CAP_MB"] = "50"
os.environ["L0_RATE_LIMIT_ENABLED"] = "0"

from na0s.layer0.resource_guard import (
    MAX_HTML_DEPTH,
    MAX_INPUT_BYTES,
    MAX_INPUT_CHARS,
    RateLimiter,
    ResourceLimitExceeded,
    check_expansion_ratio,
    check_html_depth,
    check_input_size,
    check_memory_budget,
    check_rate_limit,
    run_entry_guards,
)


class TestResourceLimitExceeded(unittest.TestCase):
    """ResourceLimitExceeded exception behaviour."""

    def test_attributes(self):
        err = ResourceLimitExceeded("input_size", "too big")
        self.assertEqual(err.guard_name, "input_size")
        self.assertEqual(err.detail, "too big")

    def test_message_format(self):
        err = ResourceLimitExceeded("html_depth", "depth 200 > 100")
        self.assertIn("html_depth", str(err))
        self.assertIn("depth 200 > 100", str(err))

    def test_is_exception(self):
        self.assertTrue(issubclass(ResourceLimitExceeded, Exception))


class TestCheckInputSize(unittest.TestCase):
    """check_input_size() validates character and byte limits."""

    def test_normal_input_passes(self):
        check_input_size("hello world")

    def test_empty_input_passes(self):
        check_input_size("")

    def test_oversized_chars_rejected(self):
        text = "a" * (MAX_INPUT_CHARS + 1)
        with self.assertRaises(ResourceLimitExceeded) as ctx:
            check_input_size(text)
        self.assertEqual(ctx.exception.guard_name, "input_size")
        self.assertIn("char limit", ctx.exception.detail)

    def test_oversized_bytes_rejected(self):
        emoji = "\U0001F600"  # 4 bytes in UTF-8
        count = (MAX_INPUT_BYTES // 4) + 1
        if count <= MAX_INPUT_CHARS:
            text = emoji * count
            with self.assertRaises(ResourceLimitExceeded) as ctx:
                check_input_size(text)
            self.assertEqual(ctx.exception.guard_name, "input_size")
            self.assertIn("byte limit", ctx.exception.detail)

    def test_at_limit_passes(self):
        text = "a" * MAX_INPUT_CHARS
        check_input_size(text)


class TestCheckHtmlDepth(unittest.TestCase):
    """check_html_depth() limits HTML nesting."""

    def test_no_html_passes(self):
        check_html_depth("Just plain text, no angle brackets.")

    def test_shallow_html_passes(self):
        html = "<div><p><span>hello</span></p></div>"
        check_html_depth(html)

    def test_deep_nesting_rejected(self):
        depth = MAX_HTML_DEPTH + 10
        html = "<div>" * depth + "bomb" + "</div>" * depth
        with self.assertRaises(ResourceLimitExceeded) as ctx:
            check_html_depth(html)
        self.assertEqual(ctx.exception.guard_name, "html_depth")
        self.assertIn("depth", ctx.exception.detail)

    def test_custom_max_depth(self):
        html = "<div>" * 6 + "x" + "</div>" * 6
        with self.assertRaises(ResourceLimitExceeded):
            check_html_depth(html, max_depth=5)
        check_html_depth(html, max_depth=10)

    def test_malformed_html_does_not_crash(self):
        html = "<div><div><div>" * 5
        check_html_depth(html)


class TestCheckExpansionRatio(unittest.TestCase):
    """check_expansion_ratio() prevents zip-bomb style expansion."""

    def test_normal_ratio_passes(self):
        check_expansion_ratio(100, 100)
        check_expansion_ratio(100, 200)

    def test_excessive_ratio_rejected(self):
        with self.assertRaises(ResourceLimitExceeded) as ctx:
            check_expansion_ratio(100, 1500)
        self.assertEqual(ctx.exception.guard_name, "expansion_ratio")
        self.assertIn("ratio", ctx.exception.detail)

    def test_zero_original_passes(self):
        check_expansion_ratio(0, 100)

    def test_custom_max_ratio(self):
        with self.assertRaises(ResourceLimitExceeded):
            check_expansion_ratio(100, 400, max_ratio=3.0)
        check_expansion_ratio(100, 200, max_ratio=3.0)


class TestCheckMemoryBudget(unittest.TestCase):
    """check_memory_budget() rejects inputs that would blow memory."""

    def test_small_input_passes(self):
        check_memory_budget("hello world")

    def test_zero_cap_disables(self):
        big = "x" * 1_000_000
        check_memory_budget(big, cap_mb=0)

    def test_negative_cap_disables(self):
        big = "x" * 1_000_000
        check_memory_budget(big, cap_mb=-1)


class TestRateLimiter(unittest.TestCase):
    """RateLimiter sliding-window behaviour."""

    def test_disabled_by_default(self):
        rl = RateLimiter(enabled=False)
        for _ in range(200):
            rl.check("user1")

    def test_allows_within_limit(self):
        rl = RateLimiter(max_requests=5, window_sec=60, enabled=True)
        for _ in range(5):
            rl.check("user1")

    def test_rejects_over_limit(self):
        rl = RateLimiter(max_requests=3, window_sec=60, enabled=True)
        rl.check("user1")
        rl.check("user1")
        rl.check("user1")
        with self.assertRaises(ResourceLimitExceeded) as ctx:
            rl.check("user1")
        self.assertEqual(ctx.exception.guard_name, "rate_limit")

    def test_separate_callers(self):
        rl = RateLimiter(max_requests=2, window_sec=60, enabled=True)
        rl.check("alice")
        rl.check("alice")
        rl.check("bob")
        rl.check("bob")
        with self.assertRaises(ResourceLimitExceeded):
            rl.check("alice")


class TestModuleLevelRateLimit(unittest.TestCase):
    """Module-level check_rate_limit() uses the singleton."""

    def test_no_op_when_disabled(self):
        check_rate_limit("test_user")


class TestRunEntryGuards(unittest.TestCase):
    """run_entry_guards() is the combined entry point."""

    def test_normal_input_passes(self):
        run_entry_guards("Hello, this is a normal prompt.")

    def test_oversized_input_rejected(self):
        text = "x" * (MAX_INPUT_CHARS + 1)
        with self.assertRaises(ResourceLimitExceeded):
            run_entry_guards(text)

    def test_deep_html_rejected(self):
        depth = MAX_HTML_DEPTH + 10
        html = "<div>" * depth + "x" + "</div>" * depth
        with self.assertRaises(ResourceLimitExceeded):
            run_entry_guards(html)


class TestEnvVarConfiguration(unittest.TestCase):
    """Verify that module-level constants read from env vars."""

    def test_max_input_chars(self):
        self.assertEqual(MAX_INPUT_CHARS, 50000)

    def test_max_input_bytes(self):
        self.assertEqual(MAX_INPUT_BYTES, 200000)

    def test_max_html_depth(self):
        self.assertEqual(MAX_HTML_DEPTH, 100)


if __name__ == "__main__":
    unittest.main()
