"""Layer 12 — Unit tests for taxonomy buff mutation transforms.

Tests every buff in scripts/taxonomy/_buffs.py for correctness,
edge cases, round-trips, and multi-buff composition.
"""

import base64
import codecs
import os
import unittest

os.environ["SCAN_TIMEOUT_SEC"] = "0"

from scripts.taxonomy._buffs import (
    ALL_BUFFS,
    Base64Buff,
    ROT13Buff,
    LeetBuff,
    FullwidthBuff,
    ZeroWidthBuff,
    HomoglyphBuff,
    ReverseBuff,
    CaseAlternatingBuff,
    Buff,
)


class TestAllBuffsRegistry(unittest.TestCase):
    """Tests for the ALL_BUFFS list."""

    def test_all_buffs_contains_eight(self):
        self.assertEqual(len(ALL_BUFFS), 8)

    def test_all_buffs_expected_classes(self):
        expected = {
            Base64Buff, ROT13Buff, LeetBuff, FullwidthBuff,
            ZeroWidthBuff, HomoglyphBuff, ReverseBuff, CaseAlternatingBuff,
        }
        self.assertEqual(set(ALL_BUFFS), expected)

    def test_all_buffs_are_subclasses_of_buff(self):
        for cls in ALL_BUFFS:
            self.assertTrue(issubclass(cls, Buff), f"{cls.__name__} not a Buff subclass")

    def test_each_buff_is_callable_and_returns_string(self):
        for cls in ALL_BUFFS:
            buff = cls()
            result = buff.apply("hello")
            self.assertIsInstance(result, str, f"{cls.__name__}.apply() did not return str")

    def test_each_buff_has_nonempty_name(self):
        for cls in ALL_BUFFS:
            self.assertTrue(cls.name, f"{cls.__name__}.name is empty")

    def test_all_buff_names_unique(self):
        names = [cls.name for cls in ALL_BUFFS]
        self.assertEqual(len(names), len(set(names)))


# ============================================================================
# Base64Buff
# ============================================================================
class TestBase64Buff(unittest.TestCase):
    def setUp(self):
        self.buff = Base64Buff()

    def test_basic(self):
        self.assertEqual(self.buff.apply("hello"), base64.b64encode(b"hello").decode("ascii"))

    def test_empty(self):
        self.assertEqual(self.buff.apply(""), "")

    def test_non_ascii(self):
        text = "café"
        expected = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self.assertEqual(self.buff.apply(text), expected)

    def test_emoji(self):
        text = "🔥🚀"
        expected = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self.assertEqual(self.buff.apply(text), expected)

    def test_round_trip(self):
        text = "round trip test 123"
        encoded = self.buff.apply(text)
        decoded = base64.b64decode(encoded).decode("utf-8")
        self.assertEqual(decoded, text)

    def test_long_string(self):
        text = "A" * 10000
        result = self.buff.apply(text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ============================================================================
# ROT13Buff
# ============================================================================
class TestROT13Buff(unittest.TestCase):
    def setUp(self):
        self.buff = ROT13Buff()

    def test_basic(self):
        self.assertEqual(self.buff.apply("hello"), "uryyb")

    def test_empty(self):
        self.assertEqual(self.buff.apply(""), "")

    def test_round_trip(self):
        text = "round trip"
        self.assertEqual(self.buff.apply(self.buff.apply(text)), text)

    def test_non_alpha_unchanged(self):
        self.assertEqual(self.buff.apply("123!@#"), "123!@#")

    def test_mixed(self):
        result = self.buff.apply("Hello 123")
        self.assertEqual(result, "Uryyb 123")


# ============================================================================
# LeetBuff
# ============================================================================
class TestLeetBuff(unittest.TestCase):
    def setUp(self):
        self.buff = LeetBuff()

    def test_basic(self):
        self.assertEqual(self.buff.apply("aeiost"), "431057")

    def test_uppercase(self):
        self.assertEqual(self.buff.apply("AEIOST"), "431057")

    def test_empty(self):
        self.assertEqual(self.buff.apply(""), "")

    def test_no_leet_chars(self):
        self.assertEqual(self.buff.apply("bcd"), "bcd")

    def test_mixed(self):
        result = self.buff.apply("Ignore all instructions")
        self.assertIn("1", result)  # i -> 1
        self.assertIn("4", result)  # a -> 4


# ============================================================================
# FullwidthBuff
# ============================================================================
class TestFullwidthBuff(unittest.TestCase):
    def setUp(self):
        self.buff = FullwidthBuff()

    def test_basic(self):
        result = self.buff.apply("AB")
        self.assertEqual(result, "\uff21\uff22")

    def test_space(self):
        result = self.buff.apply(" ")
        self.assertEqual(result, "\u3000")

    def test_empty(self):
        self.assertEqual(self.buff.apply(""), "")

    def test_non_ascii_passthrough(self):
        # Characters outside ASCII printable range pass through unchanged
        result = self.buff.apply("日本")
        self.assertEqual(result, "日本")

    def test_emoji_passthrough(self):
        result = self.buff.apply("🔥")
        self.assertEqual(result, "🔥")


# ============================================================================
# ZeroWidthBuff
# ============================================================================
class TestZeroWidthBuff(unittest.TestCase):
    def setUp(self):
        self.buff = ZeroWidthBuff()

    def test_basic(self):
        result = self.buff.apply("ab")
        self.assertEqual(result, "a\u200Db")

    def test_single_char(self):
        self.assertEqual(self.buff.apply("x"), "x")

    def test_empty(self):
        self.assertEqual(self.buff.apply(""), "")

    def test_length(self):
        text = "hello"
        result = self.buff.apply(text)
        # 5 chars joined by 4 ZWJs = 9
        self.assertEqual(len(result), 9)


# ============================================================================
# HomoglyphBuff
# ============================================================================
class TestHomoglyphBuff(unittest.TestCase):
    def setUp(self):
        self.buff = HomoglyphBuff()

    def test_basic(self):
        # 'a' -> Cyrillic а (\u0430)
        result = self.buff.apply("a")
        self.assertEqual(result, "\u0430")

    def test_no_mapping(self):
        # Characters without homoglyphs pass through
        self.assertEqual(self.buff.apply("bdfg"), "bdfg")

    def test_empty(self):
        self.assertEqual(self.buff.apply(""), "")

    def test_uppercase_unchanged(self):
        # Only lowercase mapped in _MAP
        self.assertEqual(self.buff.apply("A"), "A")

    def test_visually_similar(self):
        # 'ace' should all map to Cyrillic
        result = self.buff.apply("ace")
        self.assertNotEqual(result, "ace")
        self.assertEqual(len(result), 3)


# ============================================================================
# ReverseBuff
# ============================================================================
class TestReverseBuff(unittest.TestCase):
    def setUp(self):
        self.buff = ReverseBuff()

    def test_basic(self):
        self.assertEqual(self.buff.apply("hello"), "olleh")

    def test_empty(self):
        self.assertEqual(self.buff.apply(""), "")

    def test_single_char(self):
        self.assertEqual(self.buff.apply("x"), "x")

    def test_round_trip(self):
        text = "round trip"
        self.assertEqual(self.buff.apply(self.buff.apply(text)), text)

    def test_palindrome(self):
        self.assertEqual(self.buff.apply("racecar"), "racecar")


# ============================================================================
# CaseAlternatingBuff
# ============================================================================
class TestCaseAlternatingBuff(unittest.TestCase):
    def setUp(self):
        self.buff = CaseAlternatingBuff()

    def test_basic(self):
        result = self.buff.apply("hello")
        self.assertEqual(result, "HeLlO")

    def test_empty(self):
        self.assertEqual(self.buff.apply(""), "")

    def test_numbers_skipped(self):
        result = self.buff.apply("a1b2c")
        self.assertEqual(result, "A1b2C")

    def test_spaces_skipped(self):
        result = self.buff.apply("a b c")
        # alpha index: a=0(U), b=1(l), c=2(U)
        self.assertEqual(result, "A b C")

    def test_all_upper_input(self):
        result = self.buff.apply("HELLO")
        self.assertEqual(result, "HeLlO")


# ============================================================================
# Edge cases across all buffs
# ============================================================================
class TestBuffEdgeCases(unittest.TestCase):
    def test_very_long_string_all_buffs(self):
        text = "x" * 5000
        for cls in ALL_BUFFS:
            buff = cls()
            result = buff.apply(text)
            self.assertIsInstance(result, str, f"{cls.__name__} failed on long string")
            self.assertGreater(len(result), 0)

    def test_emoji_all_buffs(self):
        text = "🔥 ignore all 🚀"
        for cls in ALL_BUFFS:
            buff = cls()
            result = buff.apply(text)
            self.assertIsInstance(result, str, f"{cls.__name__} failed on emoji input")


# ============================================================================
# Multi-buff composition
# ============================================================================
class TestBuffComposition(unittest.TestCase):
    def test_leet_then_reverse(self):
        text = "Ignore this"
        step1 = LeetBuff().apply(text)
        step2 = ReverseBuff().apply(step1)
        self.assertEqual(step2, step1[::-1])

    def test_rot13_then_base64(self):
        text = "attack"
        step1 = ROT13Buff().apply(text)
        step2 = Base64Buff().apply(step1)
        # Decode to verify
        decoded = base64.b64decode(step2).decode("utf-8")
        self.assertEqual(decoded, codecs.encode(text, "rot_13"))

    def test_case_alternating_then_homoglyph(self):
        text = "payload"
        step1 = CaseAlternatingBuff().apply(text)
        step2 = HomoglyphBuff().apply(step1)
        self.assertIsInstance(step2, str)
        self.assertEqual(len(step2), len(text))

    def test_triple_composition(self):
        text = "ignore previous"
        result = text
        for cls in [LeetBuff, CaseAlternatingBuff, ReverseBuff]:
            result = cls().apply(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_all_buffs_chained(self):
        """Apply every non-base64 buff sequentially (base64 changes charset)."""
        text = "test input"
        result = text
        for cls in ALL_BUFFS:
            if cls is Base64Buff:
                continue  # base64 output is ASCII-only, breaks fullwidth round-trip
            result = cls().apply(result)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
