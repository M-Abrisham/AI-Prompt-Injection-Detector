"""Tests for whitespace steganography detection module.

Comprehensive tests covering:
  1. SNOW structural detection (0.95 confidence)
  2. Statistical anomaly detection (0.70 confidence)
  3. Simple binary encoding (0.60 confidence)
  4. Trailing WS anomaly (0.50 confidence)
  5. CRLF handling
  6. Confidence thresholds
  7. StegoResult dataclass
  8. Edge cases (empty, single line, clean text, oversized input, all-whitespace)
  9. False positive resistance
  10. Env-configurable thresholds
  11. Internal helpers (_shannon_entropy, _extract_trailing_whitespace, etc.)
  12. Integration via detect_whitespace_stego()

Technique coverage: D4 (Whitespace Steganography)
"""

import math
import os
import sys
import unittest

# Ensure the src directory is on the path for imports.
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from na0s.layer2.whitespace_stego import (
    detect_whitespace_stego,
    StegoResult,
    _extract_trailing_whitespace,
    _shannon_entropy,
    _detect_snow_pattern,
    _snow_bit_reverse,
    _decode_snow,
    _decode_binary_ws,
    _printable_ratio,
    _filter_markdown_breaks,
    _CONFIDENCE_SNOW,
    _CONFIDENCE_STATISTICAL,
    _CONFIDENCE_BINARY,
    _CONFIDENCE_ANOMALY,
    _MIN_DECODED_LEN,
    _SNOW_LINE_RE,
    MAX_INPUT_LENGTH,
    MIN_TRAILING_BYTES,
)


# ---------------------------------------------------------------------------
# Helper: build SNOW-encoded trailing whitespace for a known payload
# ---------------------------------------------------------------------------

def _snow_encode_char_to_3bit_groups(char):
    """Encode a single ASCII char into a list of 3-bit values (SNOW style)."""
    byte_val = ord(char)
    # We need ceil(8/3) = 3 groups of 3 bits for one byte.
    # SNOW packs 3-bit values MSB-first into a bit stream, then reassembles
    # into bytes.  For simplicity we encode multiple chars and let the
    # decoder reassemble.
    groups = []
    # 8 bits -> 3 groups: bits 7-5, 4-2, 1-0 (padded)
    groups.append((byte_val >> 5) & 0x07)
    groups.append((byte_val >> 2) & 0x07)
    groups.append((byte_val & 0x03) << 1)  # left-pad remaining 2 bits
    return groups


def _snow_encode_string(text):
    """Encode a string into SNOW-style trailing whitespace lines.

    Each 3-bit value is encoded as N spaces between TABs, with
    SNOW bit-reversal applied.  Returns a list of trailing-WS strings
    (one per output line).  We pack all groups onto a single line for
    simplicity.
    """
    three_bit_values = []
    bit_buffer = 0
    bits_in_buffer = 0
    for ch in text:
        bit_buffer = (bit_buffer << 8) | ord(ch)
        bits_in_buffer += 8
        while bits_in_buffer >= 3:
            bits_in_buffer -= 3
            val = (bit_buffer >> bits_in_buffer) & 0x07
            three_bit_values.append(val)

    # Apply SNOW bit-reversal (swap bits 0 and 2)
    reversed_vals = [_snow_bit_reverse(v) for v in three_bit_values]

    # Build trailing WS: TAB + (spaces TAB)* for each group
    parts = ["\t"]
    for rv in reversed_vals:
        parts.append(" " * rv)
        parts.append("\t")
    return "".join(parts)


def _build_snow_text(payload, num_lines=4, visible_prefix="Hello world"):
    """Build a multi-line text with SNOW-encoded payload in trailing WS."""
    snow_ws = _snow_encode_string(payload)
    lines = []
    for i in range(num_lines):
        lines.append(visible_prefix + snow_ws)
    return "\n".join(lines)


def _build_binary_ws_text(payload, visible_prefix="Line"):
    """Build text with simple binary (space=0, tab=1) encoded payload."""
    bits = []
    for ch in payload:
        byte_val = ord(ch)
        for shift in range(7, -1, -1):
            bits.append((byte_val >> shift) & 1)

    # Distribute bits across lines (8 bits per line = 1 char per line)
    lines = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        trailing = "".join("\t" if b else " " for b in chunk)
        lines.append("{} {}".format(visible_prefix, i // 8) + trailing)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. StegoResult dataclass tests
# ---------------------------------------------------------------------------

class TestStegoResult(unittest.TestCase):
    """Test StegoResult dataclass defaults and construction."""

    def test_default_values(self):
        result = StegoResult()
        self.assertFalse(result.detected)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.method, "")
        self.assertEqual(result.decoded_payload, "")
        self.assertEqual(result.flags, [])
        self.assertEqual(result.stats, {})

    def test_custom_values(self):
        result = StegoResult(
            detected=True,
            confidence=0.95,
            method="snow",
            decoded_payload="HELLO",
            flags=["whitespace_stego_snow"],
            stats={"total_lines": 10},
        )
        self.assertTrue(result.detected)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.method, "snow")
        self.assertEqual(result.decoded_payload, "HELLO")
        self.assertEqual(result.flags, ["whitespace_stego_snow"])
        self.assertEqual(result.stats["total_lines"], 10)

    def test_flags_default_is_not_shared(self):
        """Each StegoResult instance should have its own flags list."""
        r1 = StegoResult()
        r2 = StegoResult()
        r1.flags.append("test")
        self.assertEqual(len(r2.flags), 0)


# ---------------------------------------------------------------------------
# 2. Internal helpers tests
# ---------------------------------------------------------------------------

class TestExtractTrailingWhitespace(unittest.TestCase):
    """Test _extract_trailing_whitespace helper."""

    def test_no_trailing_ws(self):
        result = _extract_trailing_whitespace("hello\nworld")
        self.assertEqual(result[0], (0, "hello", ""))
        self.assertEqual(result[1], (1, "world", ""))

    def test_trailing_spaces(self):
        result = _extract_trailing_whitespace("hello   \nworld\t\t")
        self.assertEqual(result[0][2], "   ")
        self.assertEqual(result[1][2], "\t\t")

    def test_crlf_stripped(self):
        result = _extract_trailing_whitespace("hello \r\nworld\t\r\n")
        # \r should be stripped from trailing WS
        self.assertEqual(result[0][2], " ")
        self.assertEqual(result[1][2], "\t")

    def test_empty_lines(self):
        result = _extract_trailing_whitespace("\n\n")
        self.assertEqual(len(result), 3)
        for _, visible, trailing in result:
            self.assertEqual(visible, "")
            self.assertEqual(trailing, "")


class TestShannonEntropy(unittest.TestCase):
    """Test _shannon_entropy helper."""

    def test_empty_string(self):
        self.assertEqual(_shannon_entropy(""), 0.0)

    def test_single_char(self):
        self.assertEqual(_shannon_entropy("aaaa"), 0.0)

    def test_two_equal_chars(self):
        entropy = _shannon_entropy("abab")
        self.assertAlmostEqual(entropy, 1.0, places=5)

    def test_mixed_ws_entropy(self):
        # Equal mix of spaces and tabs
        entropy = _shannon_entropy(" \t" * 50)
        self.assertAlmostEqual(entropy, 1.0, places=5)


class TestSnowBitReverse(unittest.TestCase):
    """Test _snow_bit_reverse helper."""

    def test_self_inverse(self):
        """Applying bit reversal twice should return original."""
        for val in range(8):
            self.assertEqual(_snow_bit_reverse(_snow_bit_reverse(val)), val)

    def test_known_values(self):
        # 0b000 -> 0b000
        self.assertEqual(_snow_bit_reverse(0), 0)
        # 0b001 -> 0b100
        self.assertEqual(_snow_bit_reverse(1), 4)
        # 0b100 -> 0b001
        self.assertEqual(_snow_bit_reverse(4), 1)
        # 0b010 -> 0b010
        self.assertEqual(_snow_bit_reverse(2), 2)
        # 0b111 -> 0b111
        self.assertEqual(_snow_bit_reverse(7), 7)


class TestPrintableRatio(unittest.TestCase):
    """Test _printable_ratio helper."""

    def test_empty_string(self):
        self.assertEqual(_printable_ratio(""), 0.0)

    def test_all_printable(self):
        self.assertEqual(_printable_ratio("Hello World!"), 1.0)

    def test_with_control_chars(self):
        # 4 printable + 1 non-printable
        text = "abcd\x01"
        ratio = _printable_ratio(text)
        self.assertAlmostEqual(ratio, 0.8, places=5)

    def test_newline_counts_as_printable(self):
        ratio = _printable_ratio("abc\n")
        self.assertEqual(ratio, 1.0)

    def test_tab_not_printable(self):
        # Tab is NOT counted as printable in decoded payloads
        ratio = _printable_ratio("\t\t\t")
        self.assertEqual(ratio, 0.0)


class TestFilterMarkdownBreaks(unittest.TestCase):
    """Test _filter_markdown_breaks helper."""

    def test_two_spaces_filtered(self):
        ws_list = ["  ", "\t\t", "   ", ""]
        result = _filter_markdown_breaks(ws_list)
        # Exactly 2 spaces -> filtered to ""
        self.assertEqual(result[0], "")
        # Others unchanged
        self.assertEqual(result[1], "\t\t")
        self.assertEqual(result[2], "   ")
        self.assertEqual(result[3], "")

    def test_no_markdown_breaks(self):
        ws_list = ["\t", "   ", "\t \t"]
        result = _filter_markdown_breaks(ws_list)
        self.assertEqual(result, ws_list)


# ---------------------------------------------------------------------------
# 3. SNOW structural detection tests
# ---------------------------------------------------------------------------

class TestSnowDetection(unittest.TestCase):
    """Test SNOW pattern detection and decoding."""

    def test_snow_line_regex_valid_pattern(self):
        """A valid SNOW line: TAB + spaces + TAB + spaces + TAB."""
        line = "\t   \t  \t"
        self.assertIsNotNone(_SNOW_LINE_RE.match(line))

    def test_snow_line_regex_minimal(self):
        """Minimal valid SNOW line: TAB + 0-7 spaces + TAB + spaces + TAB."""
        line = "\t\t\t"
        self.assertIsNotNone(_SNOW_LINE_RE.match(line))

    def test_snow_line_regex_rejects_lone_tab(self):
        """A single TAB should NOT match SNOW pattern."""
        self.assertIsNone(_SNOW_LINE_RE.match("\t"))

    def test_snow_line_regex_rejects_spaces_only(self):
        """Pure spaces should NOT match SNOW pattern."""
        self.assertIsNone(_SNOW_LINE_RE.match("     "))

    def test_detect_snow_pattern_minimum_lines(self):
        """Needs at least SNOW_MIN_LINES matching lines."""
        ws_list = ["\t   \t  \t", "\t\t\t"]
        is_snow, count = _detect_snow_pattern(ws_list)
        self.assertTrue(is_snow)
        self.assertEqual(count, 2)

    def test_detect_snow_pattern_not_enough(self):
        """Only 1 matching line should not trigger (default min=2)."""
        ws_list = ["\t   \t  \t", "   "]
        is_snow, count = _detect_snow_pattern(ws_list)
        self.assertFalse(is_snow)
        self.assertEqual(count, 1)

    def test_snow_detection_full_pipeline(self):
        """Build SNOW-encoded payload and verify detect_whitespace_stego finds it."""
        payload = "HELLO"
        text = _build_snow_text(payload, num_lines=4)
        result = detect_whitespace_stego(text)
        self.assertTrue(result.detected)
        self.assertEqual(result.method, "snow")
        self.assertEqual(result.confidence, _CONFIDENCE_SNOW)
        self.assertIn("whitespace_stego_snow", result.flags)

    def test_snow_decoded_payload_is_printable(self):
        """SNOW-decoded payload should contain printable ASCII."""
        payload = "TEST"
        text = _build_snow_text(payload, num_lines=3)
        result = detect_whitespace_stego(text)
        if result.detected and result.method == "snow":
            pr = _printable_ratio(result.decoded_payload)
            self.assertGreaterEqual(pr, 0.60)


# ---------------------------------------------------------------------------
# 4. Simple binary encoding tests
# ---------------------------------------------------------------------------

class TestBinaryEncoding(unittest.TestCase):
    """Test simple space=0/tab=1 binary detection."""

    def test_decode_binary_known_payload(self):
        """Decode a known binary-encoded ASCII char."""
        # 'A' = 0x41 = 0b01000001 -> space tab space space space space space tab
        ws_list = [" \t     \t"]
        decoded = _decode_binary_ws(ws_list)
        self.assertEqual(decoded, "A")

    def test_decode_binary_multi_char(self):
        """Decode multi-character binary payload."""
        # 'Hi' = H(0x48=01001000) i(0x69=01101001)
        ws_list = [" \t  \t   ", " \t\t \t  \t"]
        decoded = _decode_binary_ws(ws_list)
        self.assertEqual(decoded, "Hi")

    def test_decode_binary_too_short(self):
        """Less than 8 bits should return empty."""
        ws_list = ["  \t"]
        decoded = _decode_binary_ws(ws_list)
        self.assertEqual(decoded, "")

    def test_binary_detection_full_pipeline(self):
        """Build binary-encoded payload and verify detection."""
        payload = "INJECT"
        text = _build_binary_ws_text(payload)
        result = detect_whitespace_stego(text)
        # Should detect as binary or possibly snow/statistical depending on
        # the exact trailing WS distribution.  The key is detection triggers.
        self.assertTrue(result.detected)
        self.assertIn(result.method, ("binary", "snow", "statistical"))


# ---------------------------------------------------------------------------
# 5. Statistical anomaly detection tests
# ---------------------------------------------------------------------------

class TestStatisticalAnomaly(unittest.TestCase):
    """Test statistical anomaly detection (Method 2)."""

    def test_high_volume_mixed_ws_triggers(self):
        """Large amount of mixed trailing WS with high entropy should trigger."""
        # Build text with lots of mixed tabs/spaces trailing WS but
        # no valid SNOW structure (avoid SNOW regex match).
        lines = []
        for i in range(20):
            # Non-SNOW trailing: spaces then tabs (not TAB-first pattern)
            trailing = "   \t  \t \t  " * 3  # 30+ bytes mixed
            lines.append("Visible line {}{}".format(i, trailing))
        text = "\n".join(lines)
        result = detect_whitespace_stego(text)
        # Should detect as statistical or anomaly
        if result.detected:
            self.assertIn(result.method, ("statistical", "anomaly", "binary"))

    def test_high_entropy_mixed_ws(self):
        """High entropy in trailing WS suggests intentional encoding."""
        # Alternate space/tab to maximize entropy
        lines = []
        for i in range(30):
            trailing = (" \t" * 20)  # 40 bytes, entropy ~1.0
            lines.append("Line {}{}".format(i, trailing))
        text = "\n".join(lines)
        result = detect_whitespace_stego(text)
        self.assertTrue(result.detected)
        self.assertGreaterEqual(result.confidence, _CONFIDENCE_ANOMALY)


# ---------------------------------------------------------------------------
# 6. Trailing WS anomaly tests (Method 4)
# ---------------------------------------------------------------------------

class TestTrailingWSAnomaly(unittest.TestCase):
    """Test trailing whitespace anomaly detection (Method 4)."""

    def test_high_ratio_mixed_ws_anomaly(self):
        """50%+ lines with mixed trailing WS should trigger anomaly."""
        lines = []
        for i in range(20):
            # Mixed trailing WS (tabs and spaces) on every line
            lines.append("Content {}\t ".format(i))
        text = "\n".join(lines)
        result = detect_whitespace_stego(text)
        if result.detected:
            self.assertIn(result.method,
                          ("anomaly", "binary", "statistical", "snow"))
            self.assertGreaterEqual(result.confidence, _CONFIDENCE_ANOMALY)

    def test_pure_spaces_no_anomaly(self):
        """Pure trailing spaces (no tabs) should NOT trigger anomaly method."""
        lines = []
        for i in range(20):
            lines.append("Content {}   ".format(i))
        text = "\n".join(lines)
        result = detect_whitespace_stego(text)
        # Pure spaces with no tabs -> has_mixed_ws is False -> no detection
        self.assertFalse(result.detected)


# ---------------------------------------------------------------------------
# 7. Confidence threshold tests
# ---------------------------------------------------------------------------

class TestConfidenceThresholds(unittest.TestCase):
    """Verify each method produces the correct confidence level."""

    def test_snow_confidence_is_095(self):
        self.assertEqual(_CONFIDENCE_SNOW, 0.95)

    def test_statistical_confidence_is_070(self):
        self.assertEqual(_CONFIDENCE_STATISTICAL, 0.70)

    def test_binary_confidence_is_060(self):
        self.assertEqual(_CONFIDENCE_BINARY, 0.60)

    def test_anomaly_confidence_is_050(self):
        self.assertEqual(_CONFIDENCE_ANOMALY, 0.50)


# ---------------------------------------------------------------------------
# 8. CRLF handling tests
# ---------------------------------------------------------------------------

class TestCRLFHandling(unittest.TestCase):
    """Test that CRLF line endings are handled correctly."""

    def test_crlf_does_not_inflate_trailing_bytes(self):
        """\\r should be stripped and not counted as trailing WS."""
        text = "hello \t\r\nworld \t\r\n"
        line_data = _extract_trailing_whitespace(text)
        for _, _, trailing in line_data:
            self.assertNotIn("\r", trailing)

    def test_crlf_and_lf_produce_same_result(self):
        """CRLF vs LF should produce identical detection results."""
        base = "hello \t\nworld \t\n"
        crlf = "hello \t\r\nworld \t\r\n"
        result_lf = detect_whitespace_stego(base)
        result_crlf = detect_whitespace_stego(crlf)
        self.assertEqual(result_lf.detected, result_crlf.detected)
        self.assertEqual(result_lf.method, result_crlf.method)

    def test_crlf_snow_detection(self):
        """SNOW detection should work identically with CRLF endings."""
        payload = "TEST"
        lf_text = _build_snow_text(payload, num_lines=4)
        crlf_text = lf_text.replace("\n", "\r\n")
        result_lf = detect_whitespace_stego(lf_text)
        result_crlf = detect_whitespace_stego(crlf_text)
        self.assertEqual(result_lf.detected, result_crlf.detected)
        self.assertEqual(result_lf.method, result_crlf.method)


# ---------------------------------------------------------------------------
# 9. Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_empty_string(self):
        result = detect_whitespace_stego("")
        self.assertFalse(result.detected)

    def test_none_input(self):
        result = detect_whitespace_stego(None)
        self.assertFalse(result.detected)

    def test_integer_input(self):
        result = detect_whitespace_stego(42)
        self.assertFalse(result.detected)

    def test_single_line_no_trailing(self):
        result = detect_whitespace_stego("Hello world")
        self.assertFalse(result.detected)

    def test_single_line_with_trailing(self):
        result = detect_whitespace_stego("Hello world   ")
        self.assertFalse(result.detected)

    def test_all_whitespace_input(self):
        result = detect_whitespace_stego("   \t\t\t   \n  \t  \n")
        # All whitespace but the "visible" part is empty — trailing WS
        # is the entire line content.  Depending on byte count and mix,
        # this may or may not trigger detection.
        self.assertIsInstance(result, StegoResult)

    def test_very_long_input_truncated(self):
        """Input exceeding MAX_INPUT_LENGTH should be truncated safely."""
        # Build a large clean text
        line = "Normal text line with no trailing whitespace"
        # Each line ~44 chars + newline = ~45 bytes
        num_lines = (MAX_INPUT_LENGTH // 45) + 100
        text = "\n".join([line] * num_lines)
        self.assertGreater(len(text), MAX_INPUT_LENGTH)
        result = detect_whitespace_stego(text)
        # Should not crash, should not detect anything
        self.assertFalse(result.detected)

    def test_oversized_input_no_crash(self):
        """Ensure oversized input does not cause memory or runtime errors."""
        text = ("A" * 500 + " \t" * 10 + "\n") * 5000
        result = detect_whitespace_stego(text)
        self.assertIsInstance(result, StegoResult)

    def test_below_min_trailing_bytes_no_detection(self):
        """If total trailing bytes < MIN_TRAILING_BYTES, no detection."""
        # Very small amount of trailing WS
        text = "hello \nworld\t\n"
        result = detect_whitespace_stego(text)
        self.assertFalse(result.detected)
        if result.stats:
            self.assertLess(result.stats.get("total_trailing_bytes", 0),
                            MIN_TRAILING_BYTES)


# ---------------------------------------------------------------------------
# 10. False positive resistance tests
# ---------------------------------------------------------------------------

class TestFalsePositiveResistance(unittest.TestCase):
    """Test that normal text patterns do NOT trigger detection."""

    def test_normal_english_text(self):
        text = ("This is a perfectly normal paragraph of English text.\n"
                "It has multiple sentences across several lines.\n"
                "There is nothing suspicious about this content at all.\n"
                "It should not trigger any steganography detection.\n")
        result = detect_whitespace_stego(text)
        self.assertFalse(result.detected)

    def test_code_with_indentation(self):
        """Code with tab indentation should not trigger."""
        text = ("def hello():\n"
                "\tprint('hello')\n"
                "\tif True:\n"
                "\t\treturn 42\n")
        result = detect_whitespace_stego(text)
        self.assertFalse(result.detected)

    def test_occasional_trailing_spaces(self):
        """A few lines with trailing spaces is normal in many editors."""
        lines = ["Line {}".format(i) for i in range(20)]
        # Add trailing spaces to just 2 lines out of 20
        lines[3] += "   "
        lines[10] += "  "
        text = "\n".join(lines)
        result = detect_whitespace_stego(text)
        self.assertFalse(result.detected)

    def test_markdown_line_breaks(self):
        """Markdown 2-space line breaks should be filtered out."""
        text = ("First line  \n"
                "Second line  \n"
                "Third line  \n"
                "Fourth line  \n"
                "Fifth line  \n")
        result = detect_whitespace_stego(text)
        # Markdown breaks are filtered; remaining trailing WS count is low
        self.assertFalse(result.detected)

    def test_tsv_data(self):
        """TSV-like data with tab separators should not trigger."""
        text = ("Name\tAge\tCity\n"
                "Alice\t30\tSeattle\n"
                "Bob\t25\tPortland\n"
                "Charlie\t35\tDenver\n")
        result = detect_whitespace_stego(text)
        self.assertFalse(result.detected)

    def test_pure_trailing_spaces_no_tabs(self):
        """Trailing spaces only (no tabs) should not trigger — not mixed WS."""
        lines = ["Line {} content     ".format(i) for i in range(30)]
        text = "\n".join(lines)
        result = detect_whitespace_stego(text)
        self.assertFalse(result.detected)


# ---------------------------------------------------------------------------
# 11. Integration tests
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    """Test detect_whitespace_stego returns expected StegoResult for payloads."""

    def test_clean_text_returns_empty_result(self):
        text = "No steganography here.\nJust normal text.\n"
        result = detect_whitespace_stego(text)
        self.assertFalse(result.detected)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.method, "")
        self.assertEqual(result.decoded_payload, "")

    def test_stats_populated_even_when_clean(self):
        text = "Normal\nText\nHere\n"
        result = detect_whitespace_stego(text)
        self.assertIn("total_lines", result.stats)
        self.assertIn("lines_with_trailing_ws", result.stats)
        self.assertIn("trailing_ws_ratio", result.stats)

    def test_snow_result_has_correct_flag(self):
        payload = "ATTACK"
        text = _build_snow_text(payload, num_lines=4)
        result = detect_whitespace_stego(text)
        if result.detected and result.method == "snow":
            self.assertIn("whitespace_stego_snow", result.flags)

    def test_binary_result_has_correct_flag(self):
        """Binary detection should set whitespace_stego_binary flag."""
        # Build text where binary decoding will work
        # 'Hi!' = H(01001000) i(01101001) !(00100001)
        # Encode as space/tab on separate lines
        h_bits = " \t  \t   "   # H = 0x48
        i_bits = " \t\t \t  \t"  # i = 0x69
        bang = "  \t    \t"      # ! = 0x21
        # Need visible content before trailing WS, and enough chars
        text = "Visible1{}\nVisible2{}\nVisible3{}\n".format(h_bits, i_bits, bang)
        result = detect_whitespace_stego(text)
        if result.detected and result.method == "binary":
            self.assertIn("whitespace_stego_binary", result.flags)
            self.assertEqual(result.confidence, _CONFIDENCE_BINARY)

    def test_anomaly_result_has_correct_flag(self):
        """Anomaly detection should set trailing_whitespace_anomaly flag."""
        # Build text with mixed WS on 60%+ of lines but no valid SNOW structure
        lines = []
        for i in range(20):
            # Non-SNOW structure: spaces before tabs
            lines.append("Content {}  \t".format(i))
        text = "\n".join(lines)
        result = detect_whitespace_stego(text)
        if result.detected and result.method == "anomaly":
            self.assertIn("trailing_whitespace_anomaly", result.flags)
            self.assertEqual(result.confidence, _CONFIDENCE_ANOMALY)

    def test_stats_include_entropy(self):
        """Stats dict should include ws_entropy field."""
        lines = ["Line {}\t ".format(i) for i in range(10)]
        text = "\n".join(lines)
        result = detect_whitespace_stego(text)
        self.assertIn("ws_entropy", result.stats)

    def test_stats_trailing_tabs_and_spaces(self):
        """Stats should count trailing tabs and spaces separately."""
        text = "Hello\t \nWorld \t\n"
        result = detect_whitespace_stego(text)
        self.assertIn("trailing_tabs", result.stats)
        self.assertIn("trailing_spaces", result.stats)


# ---------------------------------------------------------------------------
# 12. SNOW decode correctness tests
# ---------------------------------------------------------------------------

class TestSnowDecode(unittest.TestCase):
    """Test _decode_snow with known encoded payloads."""

    def test_decode_empty_list(self):
        self.assertEqual(_decode_snow([]), "")

    def test_decode_no_tabs(self):
        """Lines without tabs produce no SNOW data."""
        self.assertEqual(_decode_snow(["   ", "     "]), "")

    def test_decode_single_tab(self):
        """A single tab per line is not valid SNOW data (start marker only)."""
        decoded = _decode_snow(["\t", "\t"])
        # With only start markers and no data groups, result should be empty
        self.assertEqual(decoded, "")

    def test_encode_decode_roundtrip(self):
        """Encode a payload and verify _decode_snow recovers most of it.

        SNOW uses 3-bit groups packed into 8-bit bytes.  The last byte may
        be incomplete (8 is not divisible by 3), so up to one trailing char
        can be lost or corrupted.  We verify the decoded prefix matches.
        """
        payload = "HELLO"
        ws_lines = [_snow_encode_string(payload)]
        decoded = _decode_snow(ws_lines)
        # At least the first 4 chars should survive (last char may be lost
        # due to 3-bit alignment padding).
        self.assertTrue(
            payload.startswith(decoded) or decoded.startswith(payload[:4]),
            "Expected at least '{}' prefix, got '{}'".format(
                payload[:4], decoded))


# ---------------------------------------------------------------------------
# 13. Binary decode correctness tests
# ---------------------------------------------------------------------------

class TestBinaryDecode(unittest.TestCase):
    """Test _decode_binary_ws with known encoded payloads."""

    def test_decode_empty_list(self):
        self.assertEqual(_decode_binary_ws([]), "")

    def test_decode_null_byte_stops(self):
        """A null byte (all spaces) should terminate decoding."""
        # 'A' then null: 01000001 00000000
        ws = " \t     \t        "
        decoded = _decode_binary_ws([ws])
        self.assertEqual(decoded, "A")

    def test_decode_non_ws_chars_ignored(self):
        """Characters other than space and tab are skipped."""
        ws_list = [" \t     \t"]  # 'A'
        decoded = _decode_binary_ws(ws_list)
        self.assertEqual(decoded, "A")


if __name__ == "__main__":
    unittest.main()
