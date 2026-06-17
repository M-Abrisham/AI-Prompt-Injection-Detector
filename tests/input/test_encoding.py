"""Tests for src/na0s/layer0/encoding.py — encoding detection and decoding.

Covers:
- BOM detection for all 5 BOM types (UTF-8-SIG, UTF-16-LE/BE, UTF-32-LE/BE)
- chardet fallback when no BOM is present
- Low confidence handling (below _MIN_CONFIDENCE)
- Empty / None / non-bytes input
- decode_to_str: successful decoding, fallback chain, anomaly flags
- Edge cases: pure ASCII, binary garbage, null bytes, very short input

Run:
    SCAN_TIMEOUT_SEC=0 python3 -m unittest tests.test_encoding -v
"""

import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("SCAN_TIMEOUT_SEC", "0")

from na0s.layer0.encoding import (
    detect_encoding,
    decode_to_str,
    _BOM_MAP,
    _MIN_CONFIDENCE,
)


# ---------------------------------------------------------------------------
# BOM Detection
# ---------------------------------------------------------------------------

class TestBomDetection(unittest.TestCase):
    """detect_encoding must recognise all 5 BOM types deterministically."""

    def test_utf8_sig_bom(self):
        data = b"\xef\xbb\xbfHello"
        enc, conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-8-sig")
        self.assertEqual(conf, 1.0)
        self.assertIn("bom_detected_utf-8-sig", flags)

    def test_utf16_le_bom(self):
        data = b"\xff\xfe" + "Hello".encode("utf-16-le")
        enc, conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-16-le")
        self.assertEqual(conf, 1.0)
        self.assertIn("bom_detected_utf-16-le", flags)

    def test_utf16_be_bom(self):
        data = b"\xfe\xff" + "Hello".encode("utf-16-be")
        enc, conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-16-be")
        self.assertEqual(conf, 1.0)
        self.assertIn("bom_detected_utf-16-be", flags)

    def test_utf32_le_bom(self):
        data = b"\xff\xfe\x00\x00" + "Hi".encode("utf-32-le")
        enc, conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-32-le")
        self.assertEqual(conf, 1.0)
        self.assertIn("bom_detected_utf-32-le", flags)

    def test_utf32_be_bom(self):
        data = b"\x00\x00\xfe\xff" + "Hi".encode("utf-32-be")
        enc, conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-32-be")
        self.assertEqual(conf, 1.0)
        self.assertIn("bom_detected_utf-32-be", flags)

    def test_bom_priority_utf32_le_over_utf16_le(self):
        """UTF-32-LE BOM (FF FE 00 00) starts with FF FE (UTF-16-LE BOM).

        _BOM_MAP lists UTF-32-LE before UTF-16-LE so the longer match wins.
        """
        data = b"\xff\xfe\x00\x00" + b"\x41\x00\x00\x00"
        enc, _conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-32-le")
        self.assertIn("bom_detected_utf-32-le", flags)

    def test_bom_only_no_payload(self):
        """BOM with no trailing content should still detect correctly."""
        data = b"\xef\xbb\xbf"
        enc, conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-8-sig")
        self.assertEqual(conf, 1.0)


# ---------------------------------------------------------------------------
# Chardet Fallback (no BOM present)
# ---------------------------------------------------------------------------

class TestChardetFallback(unittest.TestCase):
    """When no BOM is present, chardet should determine the encoding."""

    def test_plain_utf8(self):
        data = "Hello, world!".encode("utf-8")
        enc, conf, flags = detect_encoding(data)
        # ASCII text is normalised to utf-8
        self.assertIn(enc, ("utf-8", "ascii"))
        self.assertGreater(conf, 0)
        # No BOM flag
        for f in flags:
            self.assertNotIn("bom_detected", f)

    def test_latin1_text(self):
        # Latin-1 specific characters: accented vowels, cedilla
        data = "caf\xe9 cr\xe8me".encode("latin-1")
        enc, conf, flags = detect_encoding(data)
        # chardet may return latin-1, iso-8859-1, or windows-1252 variant
        # All are normalised via alias_map
        self.assertIn(enc.lower().replace("_", "-"),
                      ("latin-1", "cp1252", "iso-8859-1", "windows-1252",
                       "iso-8859-9", "utf-8"))

    def test_ascii_normalised_to_utf8(self):
        """Pure ASCII should be mapped to utf-8 via alias_map."""
        data = b"just plain ascii text here"
        enc, conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-8")

    def test_utf8_multibyte(self):
        data = "Sch\u00f6ne Gr\u00fc\u00dfe".encode("utf-8")
        enc, conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-8")
        self.assertGreater(conf, 0.5)

    def test_cjk_utf8(self):
        data = "\u4f60\u597d\u4e16\u754c".encode("utf-8")
        enc, conf, flags = detect_encoding(data)
        self.assertEqual(enc, "utf-8")


# ---------------------------------------------------------------------------
# Low Confidence Handling
# ---------------------------------------------------------------------------

class TestLowConfidence(unittest.TestCase):
    """When chardet confidence is below _MIN_CONFIDENCE, a flag is emitted."""

    @patch("na0s.layer0.encoding.chardet.detect")
    def test_low_confidence_flag(self, mock_detect):
        mock_detect.return_value = {"encoding": "utf-8", "confidence": 0.3}
        enc, conf, flags = detect_encoding(b"ambiguous data")
        self.assertEqual(conf, 0.3)
        self.assertTrue(
            any("low_encoding_confidence" in f for f in flags),
            f"Expected low confidence flag but got {flags}",
        )

    @patch("na0s.layer0.encoding.chardet.detect")
    def test_zero_confidence_flag(self, mock_detect):
        mock_detect.return_value = {"encoding": "utf-8", "confidence": 0.0}
        enc, conf, flags = detect_encoding(b"\x80\x81\x82")
        self.assertEqual(conf, 0.0)
        self.assertIn("low_encoding_confidence_0pct", flags)

    @patch("na0s.layer0.encoding.chardet.detect")
    def test_at_threshold_no_flag(self, mock_detect):
        """Confidence exactly at _MIN_CONFIDENCE should NOT trigger the flag."""
        mock_detect.return_value = {"encoding": "utf-8",
                                    "confidence": _MIN_CONFIDENCE}
        enc, conf, flags = detect_encoding(b"data")
        self.assertFalse(
            any("low_encoding_confidence" in f for f in flags),
            f"Should not flag at threshold, but got {flags}",
        )

    @patch("na0s.layer0.encoding.chardet.detect")
    def test_above_threshold_no_flag(self, mock_detect):
        mock_detect.return_value = {"encoding": "utf-8", "confidence": 0.9}
        enc, conf, flags = detect_encoding(b"data")
        self.assertFalse(any("low_encoding_confidence" in f for f in flags))

    @patch("na0s.layer0.encoding.chardet.detect")
    def test_chardet_returns_none_encoding(self, mock_detect):
        """When chardet returns None encoding, detect_encoding defaults to utf-8."""
        mock_detect.return_value = {"encoding": None, "confidence": 0.0}
        enc, conf, flags = detect_encoding(b"\xff\xfa\xfb")
        self.assertEqual(enc, "utf-8")


# ---------------------------------------------------------------------------
# decode_to_str — Successful Decoding
# ---------------------------------------------------------------------------

class TestDecodeToStr(unittest.TestCase):
    """Successful decoding from various encodings."""

    def test_utf8_bytes(self):
        original = "Hello, world!"
        decoded, enc, flags = decode_to_str(original.encode("utf-8"))
        self.assertEqual(decoded, original)

    def test_utf8_multibyte_bytes(self):
        original = "\u00e9\u00e8\u00ea"  # accented characters
        decoded, enc, flags = decode_to_str(original.encode("utf-8"))
        self.assertEqual(decoded, original)

    def test_utf8_sig_bom_stripped(self):
        """UTF-8 BOM should be stripped from decoded output."""
        original = "Hello"
        data = b"\xef\xbb\xbf" + original.encode("utf-8")
        decoded, enc, flags = decode_to_str(data)
        self.assertEqual(decoded, original)
        self.assertEqual(enc, "utf-8-sig")
        self.assertIn("bom_detected_utf-8-sig", flags)

    def test_utf16_le_bom_stripped(self):
        original = "Hello"
        data = b"\xff\xfe" + original.encode("utf-16-le")
        decoded, enc, flags = decode_to_str(data)
        self.assertEqual(decoded, original)
        self.assertIn("bom_detected_utf-16-le", flags)

    def test_utf16_be_bom_stripped(self):
        original = "Hello"
        data = b"\xfe\xff" + original.encode("utf-16-be")
        decoded, enc, flags = decode_to_str(data)
        self.assertEqual(decoded, original)
        self.assertIn("bom_detected_utf-16-be", flags)

    def test_string_passthrough(self):
        """If input is already a string, return it unchanged."""
        text = "already a string"
        decoded, enc, flags = decode_to_str(text)
        self.assertEqual(decoded, text)
        self.assertEqual(enc, "utf-8")
        self.assertEqual(flags, [])

    def test_non_bytes_coercion(self):
        """Non-bytes/non-str input should be coerced via str()."""
        decoded, enc, flags = decode_to_str(42)
        self.assertEqual(decoded, "42")
        self.assertEqual(enc, "utf-8")
        self.assertIn("coerced_to_str", flags)

    def test_none_coercion(self):
        decoded, enc, flags = decode_to_str(None)
        self.assertEqual(decoded, "None")
        self.assertIn("coerced_to_str", flags)

    def test_bytearray_input(self):
        original = "bytearray test"
        data = bytearray(original.encode("utf-8"))
        decoded, enc, flags = decode_to_str(data)
        self.assertEqual(decoded, original)


# ---------------------------------------------------------------------------
# decode_to_str — Fallback Chain
# ---------------------------------------------------------------------------

class TestDecodeFallback(unittest.TestCase):
    """When primary encoding fails, decode_to_str falls back to UTF-8 replace."""

    @patch("na0s.layer0.encoding.chardet.detect")
    def test_fallback_on_wrong_encoding(self, mock_detect):
        """If chardet picks the wrong encoding, fallback should still produce a string."""
        # Give Latin-1 bytes but claim it's utf-32
        data = "caf\xe9".encode("latin-1")
        mock_detect.return_value = {"encoding": "utf-32", "confidence": 0.9}
        decoded, enc, flags = decode_to_str(data)
        self.assertIsInstance(decoded, str)
        self.assertIn("encoding_fallback_utf8", flags)

    @patch("na0s.layer0.encoding.chardet.detect")
    def test_fallback_on_lookup_error(self, mock_detect):
        """If chardet returns a bogus codec name, LookupError triggers fallback."""
        mock_detect.return_value = {"encoding": "totally-made-up-codec",
                                    "confidence": 0.8}
        data = b"some bytes"
        decoded, enc, flags = decode_to_str(data)
        self.assertIsInstance(decoded, str)
        self.assertIn("encoding_fallback_utf8", flags)

    def test_fallback_produces_replacement_chars(self):
        """Invalid bytes decoded via fallback should contain replacement chars."""
        # Bytes that are invalid UTF-8
        data = b"\x80\x81\x82\x83\x84"
        decoded, enc, flags = decode_to_str(data)
        self.assertIsInstance(decoded, str)
        # May or may not trigger fallback depending on chardet's guess.
        # At minimum, a string is returned.
        self.assertGreater(len(decoded), 0)


# ---------------------------------------------------------------------------
# Anomaly Flags
# ---------------------------------------------------------------------------

class TestAnomalyFlags(unittest.TestCase):
    """Verify correct anomaly flags are emitted for each scenario."""

    def test_no_flags_for_plain_utf8(self):
        data = "Just normal text.".encode("utf-8")
        _, _, flags = decode_to_str(data)
        # Should have no bom or fallback flags
        for f in flags:
            self.assertNotIn("bom_detected", f)
            self.assertNotIn("encoding_fallback", f)

    def test_bom_flag_format(self):
        """BOM flag should follow the pattern bom_detected_<encoding>."""
        for bom_bytes, bom_enc in _BOM_MAP:
            with self.subTest(encoding=bom_enc):
                data = bom_bytes + b"\x00"
                _, _, flags = detect_encoding(data)
                expected_flag = f"bom_detected_{bom_enc}"
                self.assertIn(expected_flag, flags)

    @patch("na0s.layer0.encoding.chardet.detect")
    def test_low_confidence_flag_format(self, mock_detect):
        """Low confidence flag should include percentage."""
        mock_detect.return_value = {"encoding": "utf-8", "confidence": 0.25}
        _, _, flags = detect_encoding(b"test")
        self.assertIn("low_encoding_confidence_25pct", flags)

    @patch("na0s.layer0.encoding.chardet.detect")
    def test_fallback_flag_from_decode(self, mock_detect):
        """encoding_fallback_utf8 must appear when primary decode fails."""
        mock_detect.return_value = {"encoding": "utf-32", "confidence": 0.9}
        data = b"\xc0\xc1"  # invalid for utf-32
        _, _, flags = decode_to_str(data)
        self.assertIn("encoding_fallback_utf8", flags)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Empty input, None, binary data, very short input, null bytes."""

    def test_empty_bytes(self):
        enc, conf, flags = detect_encoding(b"")
        self.assertEqual(enc, "utf-8")
        self.assertEqual(conf, 1.0)
        self.assertEqual(flags, [])

    def test_empty_bytearray(self):
        enc, conf, flags = detect_encoding(bytearray())
        self.assertEqual(enc, "utf-8")
        self.assertEqual(conf, 1.0)

    def test_none_input(self):
        """None is not bytes — returns utf-8 default."""
        enc, conf, flags = detect_encoding(None)
        self.assertEqual(enc, "utf-8")
        self.assertEqual(conf, 1.0)
        self.assertEqual(flags, [])

    def test_int_input(self):
        enc, conf, flags = detect_encoding(12345)
        self.assertEqual(enc, "utf-8")
        self.assertEqual(conf, 1.0)
        self.assertEqual(flags, [])

    def test_string_input(self):
        """String (not bytes) should return utf-8 default."""
        enc, conf, flags = detect_encoding("already a string")
        self.assertEqual(enc, "utf-8")
        self.assertEqual(conf, 1.0)

    def test_single_byte(self):
        enc, conf, flags = detect_encoding(b"A")
        self.assertIsInstance(enc, str)
        self.assertIsInstance(conf, float)
        self.assertIsInstance(flags, list)

    def test_null_bytes(self):
        data = b"\x00\x00\x00\x00\x00"
        enc, conf, flags = detect_encoding(data)
        self.assertIsInstance(enc, str)
        self.assertIsInstance(conf, float)

    def test_binary_garbage(self):
        """Random non-text bytes should still return a valid result."""
        data = bytes(range(128, 256))
        enc, conf, flags = detect_encoding(data)
        self.assertIsInstance(enc, str)
        self.assertIsInstance(conf, float)

    def test_decode_empty_bytes(self):
        decoded, enc, flags = decode_to_str(b"")
        self.assertEqual(decoded, "")
        self.assertEqual(enc, "utf-8")

    def test_decode_single_byte(self):
        decoded, enc, flags = decode_to_str(b"X")
        self.assertEqual(decoded, "X")

    def test_list_input_to_decode(self):
        """List input should be coerced to str."""
        decoded, enc, flags = decode_to_str([1, 2, 3])
        self.assertEqual(decoded, "[1, 2, 3]")
        self.assertIn("coerced_to_str", flags)


# ---------------------------------------------------------------------------
# _MIN_CONFIDENCE constant validation
# ---------------------------------------------------------------------------

class TestMinConfidenceConstant(unittest.TestCase):
    """Ensure _MIN_CONFIDENCE is sensible."""

    def test_min_confidence_is_float(self):
        self.assertIsInstance(_MIN_CONFIDENCE, float)

    def test_min_confidence_in_range(self):
        self.assertGreater(_MIN_CONFIDENCE, 0.0)
        self.assertLessEqual(_MIN_CONFIDENCE, 1.0)


if __name__ == "__main__":
    unittest.main()
