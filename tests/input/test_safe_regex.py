"""Tests for src/layer0/safe_regex.py -- ReDoS protection module.

Covers:
    - check_pattern_safety() detects known-bad patterns
    - check_pattern_safety() passes known-safe patterns
    - safe_compile() rejects unsafe patterns by default
    - safe_compile() compiles safe patterns successfully
    - safe_match() returns correct results for normal patterns
    - safe_search() returns correct results for normal patterns
    - safe_sub() and safe_findall() work correctly
    - safe_match()/safe_search() raise RegexTimeoutError on catastrophic
      backtracking (timeout protection)
    - Fallback works when re2 is not installed
    - All existing rules.py patterns pass safety audit
    - All existing cascade.py WhitelistFilter patterns pass safety audit

Run: python -m unittest tests/test_safe_regex.py -v
"""

import os
import re
import signal
import sys
import time
import unittest


from na0s.layer0.safe_regex import (
    check_pattern_safety,
    safe_compile,
    safe_match,
    safe_search,
    safe_sub,
    safe_findall,
    re2_available,
    RegexTimeoutError,
    _RE2_AVAILABLE,
)


# ---------------------------------------------------------------------------
# 1. Pattern safety checker tests
# ---------------------------------------------------------------------------

class TestCheckPatternSafety(unittest.TestCase):
    """Tests for check_pattern_safety() static auditor."""

    # --- Known-bad patterns (must produce warnings) ---

    def test_nested_quantifier_plus_plus(self):
        """(a+)+ is classic ReDoS."""
        warnings = check_pattern_safety(r"(a+)+")
        self.assertTrue(len(warnings) > 0, "Should detect nested quantifier")
        self.assertTrue(
            any("quantifier" in w.lower() for w in warnings),
            "Warning should mention 'quantifier'"
        )

    def test_nested_quantifier_star_star(self):
        """(a*)*  is ReDoS-vulnerable."""
        warnings = check_pattern_safety(r"(a*)*")
        self.assertTrue(len(warnings) > 0)

    def test_nested_quantifier_plus_star(self):
        """(a+)* is ReDoS-vulnerable."""
        warnings = check_pattern_safety(r"(a+)*")
        self.assertTrue(len(warnings) > 0)

    def test_nested_quantifier_brace_plus(self):
        """(a{1,})+  has a nested quantifier."""
        warnings = check_pattern_safety(r"(a{1,})+")
        self.assertTrue(len(warnings) > 0)

    def test_overlapping_alternatives_in_group(self):
        """(a|ab)+ has overlapping alternatives."""
        warnings = check_pattern_safety(r"(a|ab)+")
        self.assertTrue(len(warnings) > 0)
        self.assertTrue(
            any("overlap" in w.lower() for w in warnings),
            "Warning should mention 'overlap'"
        )

    def test_backreference_with_quantifier(self):
        r"""(a)\1+ has a backreference with quantifier."""
        warnings = check_pattern_safety(r"(a)\1+")
        self.assertTrue(len(warnings) > 0)
        self.assertTrue(
            any("backreference" in w.lower() for w in warnings),
            "Warning should mention 'backreference'"
        )

    # --- Known-safe patterns (must produce NO warnings) ---

    def test_safe_simple_literal(self):
        warnings = check_pattern_safety(r"hello world")
        self.assertEqual(warnings, [])

    def test_safe_bounded_quantifier(self):
        warnings = check_pattern_safety(r".{0,40}")
        self.assertEqual(warnings, [])

    def test_safe_char_class_plus(self):
        """[A-Za-z0-9]+ -- single quantifier on a char class is safe."""
        warnings = check_pattern_safety(r"[A-Za-z0-9]+")
        self.assertEqual(warnings, [])

    def test_safe_alternation_no_group_quantifier(self):
        """(foo|bar) without + or * on the group is safe."""
        warnings = check_pattern_safety(r"(foo|bar)")
        self.assertEqual(warnings, [])

    def test_safe_word_boundary_alternation(self):
        warnings = check_pattern_safety(r"\byou are now\b|\bpretend to be\b")
        self.assertEqual(warnings, [])

    def test_safe_bounded_gap(self):
        warnings = check_pattern_safety(r"(reveal|show).{0,40}(system prompt)")
        self.assertEqual(warnings, [])


# ---------------------------------------------------------------------------
# 2. safe_compile() tests
# ---------------------------------------------------------------------------

class TestSafeCompile(unittest.TestCase):
    """Tests for safe_compile()."""

    def test_rejects_unsafe_pattern(self):
        """safe_compile should raise ValueError for an unsafe pattern."""
        with self.assertRaises(ValueError) as ctx:
            safe_compile(r"(a+)+b")
        self.assertIn("Unsafe regex", str(ctx.exception))

    def test_accepts_safe_pattern(self):
        """safe_compile should return a compiled pattern for safe input."""
        pat = safe_compile(r"hello\s+world", re.IGNORECASE)
        self.assertTrue(hasattr(pat, "search"))
        m = pat.search("Hello  World")
        self.assertIsNotNone(m)

    def test_skip_safety_check(self):
        """check_safety=False should allow unsafe patterns through."""
        pat = safe_compile(r"(a+)+b", check_safety=False)
        self.assertTrue(hasattr(pat, "search"))

    def test_invalid_syntax_raises_re_error(self):
        """Invalid regex syntax should raise re.error."""
        with self.assertRaises(re.error):
            safe_compile(r"[unclosed")

    def test_flags_are_honoured(self):
        """Compile flags like IGNORECASE should be applied."""
        pat = safe_compile(r"abc", re.IGNORECASE)
        self.assertIsNotNone(pat.search("ABC"))


# ---------------------------------------------------------------------------
# 3. safe_match / safe_search basic functionality
# ---------------------------------------------------------------------------

class TestSafeMatchSearch(unittest.TestCase):
    """Tests for safe_match() and safe_search() normal operation."""

    def test_safe_search_finds_match(self):
        result = safe_search(r"world", "hello world")
        self.assertIsNotNone(result)
        self.assertEqual(result.group(), "world")

    def test_safe_search_no_match(self):
        result = safe_search(r"xyz", "hello world")
        self.assertIsNone(result)

    def test_safe_match_anchored(self):
        result = safe_match(r"hello", "hello world")
        self.assertIsNotNone(result)
        self.assertEqual(result.group(), "hello")

    def test_safe_match_no_match(self):
        result = safe_match(r"world", "hello world")
        self.assertIsNone(result)

    def test_safe_search_with_compiled_pattern(self):
        pat = re.compile(r"(\d+)")
        result = safe_search(pat, "abc 123 def")
        self.assertIsNotNone(result)
        self.assertEqual(result.group(1), "123")

    def test_safe_match_with_flags(self):
        result = safe_match(r"hello", "HELLO world", flags=re.IGNORECASE)
        self.assertIsNotNone(result)

    def test_safe_sub_works(self):
        result = safe_sub(r"\d+", "NUM", "abc 123 def 456")
        self.assertEqual(result, "abc NUM def NUM")

    def test_safe_findall_works(self):
        result = safe_findall(r"\d+", "abc 123 def 456")
        self.assertEqual(result, ["123", "456"])


# ---------------------------------------------------------------------------
# 4. Timeout protection tests
# ---------------------------------------------------------------------------

class TestTimeoutProtection(unittest.TestCase):
    """Verify that safe_search/safe_match raise RegexTimeoutError on
    catastrophic backtracking.

    We use a known-bad pattern + evil input that triggers exponential
    backtracking in the Python re engine.

    On Unix (macOS/Linux) we use signal.SIGALRM which has 1-second
    granularity, so the timeout fires within ~1-2 seconds.  On Windows
    (no SIGALRM) the static checker + optional RE2 are the defences,
    and this test is skipped.
    """

    # Classic ReDoS: (a+)+$ matched against "aaa...a!"
    EVIL_PATTERN = re.compile(r"(a+)+$")
    EVIL_INPUT = "a" * 28 + "!"  # 28 a's + non-matching char

    @unittest.skipUnless(
        hasattr(signal, "SIGALRM") and os.name != "nt",
        "SIGALRM-based timeout only available on Unix",
    )
    def test_timeout_on_catastrophic_backtracking(self):
        """safe_search must raise RegexTimeoutError, not hang."""
        start = time.monotonic()
        with self.assertRaises(RegexTimeoutError):
            # SIGALRM rounds up to 1s minimum
            safe_search(self.EVIL_PATTERN, self.EVIL_INPUT, timeout_ms=500)
        elapsed = time.monotonic() - start
        # Should return within ~2s (alarm fires at 1s + overhead), not minutes
        self.assertLess(elapsed, 5.0, "Should not hang for more than 5s")

    @unittest.skipUnless(
        hasattr(signal, "SIGALRM") and os.name != "nt",
        "SIGALRM-based timeout only available on Unix",
    )
    def test_timeout_on_safe_match(self):
        """safe_match must also respect timeout."""
        start = time.monotonic()
        with self.assertRaises(RegexTimeoutError):
            safe_match(self.EVIL_PATTERN, self.EVIL_INPUT, timeout_ms=500)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 5.0)


# ---------------------------------------------------------------------------
# 5. Fallback / RE2 availability tests
# ---------------------------------------------------------------------------

class TestRE2Fallback(unittest.TestCase):
    """Test that the system works regardless of whether re2 is installed."""

    def test_re2_available_returns_bool(self):
        result = re2_available()
        self.assertIsInstance(result, bool)

    def test_compile_works_without_re2(self):
        """Even if re2 is not available, safe_compile should work via stdlib."""
        # Force use_re2=False to simulate missing re2
        pat = safe_compile(r"test\s+pattern", use_re2=False)
        self.assertTrue(hasattr(pat, "search"))
        self.assertIsNotNone(pat.search("test  pattern"))

    def test_search_works_without_re2(self):
        """safe_search uses stdlib re and should always work."""
        result = safe_search(r"hello", "hello world", timeout_ms=1000)
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# 6. Existing rules.py patterns are all safe
# ---------------------------------------------------------------------------

class TestExistingRulesAreSafe(unittest.TestCase):
    """Verify every pattern in rules.RULES passes the safety auditor."""

    def test_all_rule_patterns_pass_safety_check(self):
        from na0s.rules import RULES
        for rule in RULES:
            with self.subTest(rule=rule.name):
                warnings = check_pattern_safety(rule.pattern)
                self.assertEqual(
                    warnings, [],
                    "Rule '{}' pattern '{}' failed safety check: {}".format(
                        rule.name, rule.pattern, warnings,
                    )
                )

    def test_all_rule_patterns_compile_via_safe_compile(self):
        """Every rule pattern should compile through safe_compile without error."""
        from na0s.rules import RULES
        for rule in RULES:
            with self.subTest(rule=rule.name):
                pat = safe_compile(rule.pattern, re.IGNORECASE)
                self.assertTrue(hasattr(pat, "search"))


# ---------------------------------------------------------------------------
# 7. Existing cascade.py WhitelistFilter patterns are all safe
# ---------------------------------------------------------------------------

class TestCascadePatternsAreSafe(unittest.TestCase):
    """Verify WhitelistFilter patterns pass safety audit."""

    def test_whitelist_filter_patterns_safe(self):
        # These are the raw pattern strings used in WhitelistFilter.
        # We test them directly rather than importing cascade.py (which
        # has heavy dependencies) to keep the test lightweight.
        patterns = [
            # QUESTION_WORDS
            (r"^\s*(what|how|why|when|where|who|which|can|could|would|should"
             r"|is|are|do|does|will|did)\b"),
            # BOUNDARY_MARKERS
            r"---|===|\*\*\*|\[SYSTEM\]|\[INST\]|<<SYS>>|</s>",
            # _BASE64_HEURISTIC
            r"(?<!\w)[A-Za-z0-9+/]{20,}={0,2}(?!\w)",
            # _HEX_HEURISTIC
            r"(?<!\w)[0-9a-fA-F]{16,}(?!\w)",
            # _URLENCODE_HEURISTIC (rewritten safe version)
            r"%[0-9a-fA-F]{2}.{0,200}%[0-9a-fA-F]{2}",
            # ROLE_ASSIGNMENT -- imported from canonical source in rules.py;
            # inline copy kept here only for safe_compile audit coverage.
            (r"\byou\s+are\s+now\b"
             r"|\bpretend\s+to\s+be\b"
             r"|\bact\s+as\s+(?:(?:a|an|the|my|if)\s+)?\w"
             r"|\bfrom\s+now\s+on\b"
             r"|\b(?:your\s+)?new\s+role\b"),
            # SAFE_TOPIC_INDICATORS
            (r"\b(explain|what is|how does|teach me|help me understand"
             r"|learn about|definition of)\b"),
        ]
        for pattern in patterns:
            with self.subTest(pattern=pattern[:50]):
                warnings = check_pattern_safety(pattern)
                self.assertEqual(
                    warnings, [],
                    "Cascade pattern '{}...' failed: {}".format(
                        pattern[:50], warnings,
                    )
                )


if __name__ == "__main__":
    unittest.main()
