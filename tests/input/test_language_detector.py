"""Tests for src/layer0/language_detector.py — language detection module.

Run: python3 -m unittest tests/test_language_detector.py -v
"""

import os
import sys
import unittest


from na0s.layer0.language_detector import (
    detect_language,
    _has_mixed_scripts,
    _heuristic_detect,
    _HAS_LANGDETECT,
)


class TestEnglishDetection(unittest.TestCase):
    """English text should NOT flag anomalies."""

    def test_simple_english(self):
        result = detect_language(
            "The quick brown fox jumps over the lazy dog. "
            "This is a perfectly normal English sentence."
        )
        self.assertFalse(result["is_non_english"])
        self.assertEqual(result["anomaly_flags"], [])

    def test_english_with_numbers(self):
        result = detect_language(
            "Order 12345 was placed on 2024-01-15 for $99.99 shipping."
        )
        self.assertFalse(result["is_non_english"])


@unittest.skipUnless(_HAS_LANGDETECT, "langdetect not installed")
class TestNonEnglishDetection(unittest.TestCase):
    """Non-English text should flag non_english_input."""

    def test_chinese_text(self):
        result = detect_language("这是一个中文句子，用来测试语言检测功能是否正常工作")
        self.assertTrue(result["is_non_english"])
        self.assertIn("non_english_input", result["anomaly_flags"])

    def test_arabic_text(self):
        result = detect_language("هذا نص باللغة العربية لاختبار كشف اللغة والتوجيه متعدد اللغات")
        self.assertTrue(result["is_non_english"])
        self.assertIn("non_english_input", result["anomaly_flags"])

    def test_spanish_text(self):
        result = detect_language(
            "Esta es una oración en español para probar la detección de idiomas."
        )
        self.assertTrue(result["is_non_english"])
        self.assertIn("non_english_input", result["anomaly_flags"])

    def test_russian_text(self):
        result = detect_language("Это предложение на русском языке для тестирования обнаружения языка")
        self.assertTrue(result["is_non_english"])
        self.assertIn("non_english_input", result["anomaly_flags"])


@unittest.skipUnless(_HAS_LANGDETECT, "langdetect not installed")
class TestMixedLanguageDetection(unittest.TestCase):
    """Mixed-language text should flag mixed_language_input."""

    def test_english_with_chinese(self):
        result = detect_language(
            "Hello world 这是中文 this is English 还有更多中文内容在这里"
        )
        self.assertIn("mixed_language_input", result["anomaly_flags"])

    def test_english_with_arabic(self):
        result = detect_language(
            "Hello world هذا نص عربي and this is English مرحبا بكم"
        )
        self.assertIn("mixed_language_input", result["anomaly_flags"])


class TestMixedScriptsHeuristic(unittest.TestCase):
    """Test the _has_mixed_scripts helper directly."""

    def test_latin_only(self):
        self.assertFalse(_has_mixed_scripts("Hello World"))

    def test_cjk_only(self):
        self.assertFalse(_has_mixed_scripts("这是中文"))

    def test_latin_and_cjk(self):
        self.assertTrue(_has_mixed_scripts("Hello 你好"))

    def test_latin_and_cyrillic(self):
        self.assertTrue(_has_mixed_scripts("Hello Привет"))

    def test_latin_and_arabic(self):
        self.assertTrue(_has_mixed_scripts("Hello مرحبا"))


class TestShortTextHeuristic(unittest.TestCase):
    """Short text (< 20 chars) uses script heuristic fallback."""

    def test_short_latin(self):
        result = detect_language("Hello")
        self.assertFalse(result["is_non_english"])

    def test_short_cjk(self):
        result = detect_language("你好世界")
        self.assertTrue(result["is_non_english"])
        self.assertIn("non_english_input", result["anomaly_flags"])

    def test_short_mixed(self):
        result = detect_language("Hi 你好")
        self.assertTrue(result["is_non_english"])
        self.assertIn("mixed_language_input", result["anomaly_flags"])


class TestEdgeCases(unittest.TestCase):
    """Empty, whitespace, None-like inputs."""

    def test_empty_string(self):
        result = detect_language("")
        self.assertEqual(result["detected_language"], "unknown")
        self.assertFalse(result["is_non_english"])
        self.assertEqual(result["anomaly_flags"], [])

    def test_whitespace_only(self):
        result = detect_language("   \t\n  ")
        self.assertEqual(result["detected_language"], "unknown")
        self.assertFalse(result["is_non_english"])

    def test_none_input(self):
        result = detect_language(None)
        self.assertEqual(result["detected_language"], "unknown")
        self.assertFalse(result["is_non_english"])

    def test_numbers_only(self):
        result = detect_language("1234567890 9876543210")
        self.assertFalse(result["is_non_english"])


class TestResultStructure(unittest.TestCase):
    """Verify the returned dict has all expected keys."""

    def test_all_keys_present(self):
        result = detect_language("Test sentence for structure check.")
        self.assertIn("detected_language", result)
        self.assertIn("language_confidence", result)
        self.assertIn("is_non_english", result)
        self.assertIn("anomaly_flags", result)

    def test_confidence_range(self):
        result = detect_language(
            "The quick brown fox jumps over the lazy dog."
        )
        self.assertGreaterEqual(result["language_confidence"], 0.0)
        self.assertLessEqual(result["language_confidence"], 1.0)


@unittest.skipUnless(_HAS_LANGDETECT, "langdetect not installed")
class TestSanitizerIntegration(unittest.TestCase):
    """Language detection results should appear in layer0_sanitize() output."""

    def test_english_in_sanitizer(self):
        from na0s.layer0.sanitizer import layer0_sanitize
        result = layer0_sanitize("What is the weather today in San Francisco?")
        self.assertFalse(result.rejected)
        self.assertIn("language", result.source_metadata)
        self.assertFalse(result.source_metadata["language"]["is_non_english"])

    def test_chinese_in_sanitizer(self):
        from na0s.layer0.sanitizer import layer0_sanitize
        result = layer0_sanitize("这是一个测试句子用来验证语言检测是否正常工作的功能")
        self.assertFalse(result.rejected)
        self.assertIn("non_english_input", result.anomaly_flags)
        self.assertTrue(result.source_metadata["language"]["is_non_english"])


if __name__ == "__main__":
    unittest.main()
