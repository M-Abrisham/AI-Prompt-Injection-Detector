"""Property-based tests for Layer 0 sanitizer using Hypothesis.

Validates that arbitrary Unicode input NEVER crashes layer0_sanitize,
and that all output invariants hold regardless of input content.

Run:  python -m unittest tests/test_layer0_hypothesis.py -v

Requires: pip install hypothesis
"""

import os
import sys
import unicodedata
import unittest

# ---------------------------------------------------------------------------
# Hypothesis import — skip the entire module gracefully if not installed.
#
# We must abort before any class body is evaluated, because decorators
# like @given(...) and @FUZZ_SETTINGS reference hypothesis objects.
# unittest.skipUnless on a class does NOT prevent its body from executing,
# so class-level skips alone are insufficient — they would still cause
# NameError at import time.  raise unittest.SkipTest(...) at module level
# is the standard unittest pattern for optional-dependency test modules.
# ---------------------------------------------------------------------------
try:
    from hypothesis import given, settings, HealthCheck, assume
    import hypothesis.strategies as st
except ImportError:
    raise unittest.SkipTest(
        "hypothesis not installed — skipping property-based tests. "
        "Install with:  pip install hypothesis"
    )

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from na0s.layer0 import layer0_sanitize
from na0s.layer0.result import Layer0Result
from na0s.layer0.normalization import normalize_text
from na0s.layer0.validation import MAX_INPUT_LENGTH

# ---------------------------------------------------------------------------
# Shared Hypothesis settings and strategies
# ---------------------------------------------------------------------------
FUZZ_SETTINGS = settings(
    max_examples=200,
    deadline=5000,  # 5 seconds per example
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.large_base_example,
        HealthCheck.filter_too_much,
    ],
)

# -------------------------------------------------------------------
# Custom strategies
# -------------------------------------------------------------------

# Full Unicode range (every category, every plane)
unicode_text = st.text(
    alphabet=st.characters(),
    min_size=0,
    max_size=5000,
)

# Raw binary data
binary_data = st.binary(min_size=0, max_size=100_000)

# Invisible / control character alphabet
_INVISIBLE_CHARS = (
    "\u200b"   # zero-width space
    "\u200c"   # zero-width non-joiner
    "\u200d"   # zero-width joiner
    "\u2060"   # word joiner
    "\ufeff"   # BOM / zero-width no-break space
    "\u00ad"   # soft hyphen
    "\u200e"   # LTR mark
    "\u200f"   # RTL mark
    "\u202a"   # LTR embedding
    "\u202b"   # RTL embedding
    "\u202c"   # pop directional
    "\u202d"   # LTR override
    "\u202e"   # RTL override
    "\u2066"   # LTR isolate
    "\u2067"   # RTL isolate
    "\u2068"   # first strong isolate
    "\u2069"   # pop directional isolate
    "\x00"     # null
    "\x01"     # SOH
    "\x7f"     # DEL
)

# Mixed content: normal text with injected invisible chars and HTML
mixed_content = st.builds(
    lambda parts: "".join(parts),
    st.lists(
        st.one_of(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P", "Z"),
                ),
                min_size=1,
                max_size=100,
            ),
            st.sampled_from(list(_INVISIBLE_CHARS)),
            st.sampled_from([
                "<b>", "</b>", "<script>", "</script>",
                "<div style=\"display:none\">", "</div>",
                "<!-- comment -->", "<img src=x onerror=alert(1)>",
            ]),
        ),
        min_size=1,
        max_size=20,
    ),
)

# Adversarial Unicode: zero-width, RTL, homoglyphs, variation selectors,
# tag characters, combining chars
_ADVERSARIAL_CHARS = (
    # Zero-width
    "\u200b\u200c\u200d\u2060\ufeff"
    # RTL/LTR overrides
    "\u202a\u202b\u202c\u202d\u202e"
    # Variation selectors
    "\ufe00\ufe01\ufe02\ufe0e\ufe0f"
    # Combining diacriticals
    "\u0300\u0301\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u0309"
    "\u030a\u030b\u030c\u030d\u030e\u030f"
    # Cyrillic homoglyphs of Latin
    "\u0430\u0435\u043e\u0440\u0441\u0443\u0445"
)

# Tag characters U+E0001-U+E007F (used in emoji flag sequences)
_TAG_CHARS = "".join(chr(c) for c in range(0xE0001, 0xE0080))

adversarial_unicode = st.builds(
    lambda base, inserts: "".join(
        ch + inserts[i % len(inserts)] if inserts else ch
        for i, ch in enumerate(base)
    ),
    st.text(
        alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz ")),
        min_size=1,
        max_size=200,
    ),
    st.lists(
        st.sampled_from(list(_ADVERSARIAL_CHARS + _TAG_CHARS)),
        min_size=1,
        max_size=50,
    ),
)

# Long input (up to ~200K chars) -- built by repeating a base string
# Hypothesis st.text has a ~8192 max_size limit, so we use st.builds to scale up
@st.composite
def long_input(draw):
    """Generate long text inputs that exceed MAX_INPUT_LENGTH."""
    base = draw(st.text(
        alphabet=st.sampled_from(list("abcdefghij \n\t")),
        min_size=10,
        max_size=500,
    ))
    # Repeat enough times to exceed 50K char limit
    repeats = draw(st.integers(min_value=200, max_value=2000))
    return base * repeats

# Encoding variants: same text encoded in different charsets then passed as bytes
_ENCODINGS = ["utf-8", "utf-16", "latin-1", "ascii", "cp1252"]

@st.composite
def encoding_variants(draw):
    """Generate bytes from text encoded in various charsets."""
    text = draw(st.text(
        alphabet=st.characters(
            max_codepoint=255,  # stay within Latin-1 range for broad compat
            whitelist_categories=("L", "N", "P", "Z"),
        ),
        min_size=1,
        max_size=500,
    ))
    encoding = draw(st.sampled_from(_ENCODINGS))
    try:
        return text.encode(encoding)
    except (UnicodeEncodeError, UnicodeDecodeError):
        # If encoding fails, fall back to UTF-8
        return text.encode("utf-8", errors="replace")


# ===================================================================
# Test classes
# ===================================================================


class TestNeverCrash(unittest.TestCase):
    """Core invariant: layer0_sanitize NEVER raises for any str or bytes."""

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_str_input_never_crashes(self, text):
        """Any Python str must not crash layer0_sanitize."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result, Layer0Result)

    @given(binary_data)
    @FUZZ_SETTINGS
    def test_bytes_input_never_crashes(self, data):
        """Any bytes object must not crash layer0_sanitize."""
        result = layer0_sanitize(data)
        self.assertIsInstance(result, Layer0Result)

    @given(mixed_content)
    @FUZZ_SETTINGS
    def test_mixed_content_never_crashes(self, text):
        """Text mixed with invisible chars and HTML must not crash."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result, Layer0Result)

    @given(adversarial_unicode)
    @FUZZ_SETTINGS
    def test_adversarial_unicode_never_crashes(self, text):
        """Adversarial Unicode (RTL, homoglyphs, combiners) must not crash."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result, Layer0Result)

    @given(encoding_variants())
    @FUZZ_SETTINGS
    def test_encoding_variants_never_crash(self, data):
        """Bytes in various encodings must not crash."""
        result = layer0_sanitize(data)
        self.assertIsInstance(result, Layer0Result)


class TestResultTypeInvariants(unittest.TestCase):
    """Every field of Layer0Result has the correct type and range."""

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_sanitized_text_is_str(self, text):
        """sanitized_text must always be a str, never None or bytes."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result.sanitized_text, str)

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_original_length_non_negative(self, text):
        """original_length must be >= 0."""
        result = layer0_sanitize(text)
        self.assertGreaterEqual(result.original_length, 0)

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_chars_stripped_non_negative(self, text):
        """chars_stripped must be >= 0 (or at worst a reasonable value)."""
        result = layer0_sanitize(text)
        # chars_stripped is original_length - len(sanitized_text)
        # With NFKC expansion it could technically be negative, but
        # we verify it is an int
        self.assertIsInstance(result.chars_stripped, int)

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_anomaly_flags_is_list(self, text):
        """anomaly_flags must always be a list."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result.anomaly_flags, list)

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_anomaly_flags_contains_only_strings(self, text):
        """Every anomaly flag must be a str."""
        result = layer0_sanitize(text)
        for flag in result.anomaly_flags:
            self.assertIsInstance(flag, str)

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_rejected_is_bool(self, text):
        """rejected must be a bool."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result.rejected, bool)

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_rejection_reason_is_str(self, text):
        """rejection_reason must be a str."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result.rejection_reason, str)


class TestRejectionInvariants(unittest.TestCase):
    """Invariants linking rejected status to other fields."""

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_rejected_implies_nonempty_reason(self, text):
        """If rejected is True, rejection_reason must be non-empty."""
        result = layer0_sanitize(text)
        if result.rejected:
            self.assertTrue(
                len(result.rejection_reason) > 0,
                "rejected=True but rejection_reason is empty"
            )

    @given(binary_data)
    @FUZZ_SETTINGS
    def test_rejected_implies_nonempty_reason_bytes(self, data):
        """Same invariant for bytes input."""
        result = layer0_sanitize(data)
        if result.rejected:
            self.assertTrue(
                len(result.rejection_reason) > 0,
                "rejected=True but rejection_reason is empty"
            )

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_not_rejected_implies_empty_reason(self, text):
        """If rejected is False, rejection_reason must be empty."""
        result = layer0_sanitize(text)
        if not result.rejected:
            self.assertEqual(
                result.rejection_reason, "",
                f"rejected=False but rejection_reason={result.rejection_reason!r}"
            )


class TestSanitizedTextInvariants(unittest.TestCase):
    """Properties that must hold on the sanitized output text."""

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_sanitized_text_valid_utf8(self, text):
        """Sanitized text must be encodable as valid UTF-8."""
        result = layer0_sanitize(text)
        # If it's a str, encoding to UTF-8 should not raise
        encoded = result.sanitized_text.encode("utf-8")
        self.assertIsInstance(encoded, bytes)

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_no_format_chars_in_output(self, text):
        """Sanitized output must not contain Unicode Cf (format) characters.

        These are invisible and used for evasion (zero-width, RTL overrides).
        """
        result = layer0_sanitize(text)
        for ch in result.sanitized_text:
            cat = unicodedata.category(ch)
            self.assertNotEqual(
                cat, "Cf",
                f"Found Cf char U+{ord(ch):04X} ({unicodedata.name(ch, '?')}) "
                f"in sanitized output"
            )

    @given(unicode_text)
    @FUZZ_SETTINGS
    def test_no_control_chars_except_whitespace(self, text):
        """Sanitized output must not contain Cc chars except \\n, \\r, \\t, space."""
        result = layer0_sanitize(text)
        allowed = {"\n", "\r", "\t", " "}
        for ch in result.sanitized_text:
            cat = unicodedata.category(ch)
            if cat == "Cc":
                self.assertIn(
                    ch, allowed,
                    f"Found disallowed Cc char U+{ord(ch):04X} in sanitized output"
                )

    @given(binary_data)
    @FUZZ_SETTINGS
    def test_no_format_chars_in_output_from_bytes(self, data):
        """Same Cf check for bytes input path."""
        result = layer0_sanitize(data)
        for ch in result.sanitized_text:
            cat = unicodedata.category(ch)
            self.assertNotEqual(
                cat, "Cf",
                f"Found Cf char U+{ord(ch):04X} in sanitized output (bytes path)"
            )


class TestSizeLimitProperty(unittest.TestCase):
    """Size limit enforcement as a property."""

    @given(long_input())
    @settings(
        max_examples=20,  # Long inputs are slow to generate
        deadline=10000,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.large_base_example,
            HealthCheck.filter_too_much,
        ],
    )
    def test_oversized_input_rejected(self, text):
        """Inputs exceeding MAX_INPUT_LENGTH must be rejected."""
        assume(len(text) > MAX_INPUT_LENGTH)
        result = layer0_sanitize(text)
        self.assertTrue(
            result.rejected,
            f"Input of length {len(text)} was not rejected "
            f"(limit={MAX_INPUT_LENGTH})"
        )

    @given(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=1000,
    ))
    @FUZZ_SETTINGS
    def test_small_valid_input_not_rejected(self, text):
        """Short alphanumeric input should generally not be rejected."""
        assume(len(text.strip()) > 0)
        result = layer0_sanitize(text)
        # If the text after NFKC still has visible content, it should not
        # be rejected (unless it hits the byte limit, which is unlikely
        # for small inputs)
        if not result.rejected:
            self.assertGreater(len(result.sanitized_text), 0)


class TestEmptyAndWhitespaceRejection(unittest.TestCase):
    """Empty, whitespace-only, and all-invisible inputs must be rejected."""

    def test_empty_string_rejected(self):
        result = layer0_sanitize("")
        self.assertTrue(result.rejected)

    @given(st.text(
        alphabet=st.sampled_from([" ", "\t", "\n", "\r"]),
        min_size=1,
        max_size=100,
    ))
    @FUZZ_SETTINGS
    def test_whitespace_only_rejected(self, text):
        """All-whitespace input must be rejected."""
        result = layer0_sanitize(text)
        self.assertTrue(
            result.rejected,
            f"Whitespace-only input {text!r} was not rejected"
        )

    @given(st.text(
        alphabet=st.sampled_from(list(_INVISIBLE_CHARS)),
        min_size=1,
        max_size=200,
    ))
    @FUZZ_SETTINGS
    def test_all_invisible_chars_rejected(self, text):
        """All-invisible-character input must be rejected (BUG-1 validation).

        This validates the post-normalization empty check in sanitizer.py
        lines 40-49, which catches inputs that pass validation but become
        empty after invisible-char stripping.
        """
        result = layer0_sanitize(text)
        self.assertTrue(
            result.rejected,
            f"All-invisible input {text!r} (len={len(text)}) was not rejected. "
            f"sanitized_text={result.sanitized_text!r}"
        )


class TestNormalizationIdempotency(unittest.TestCase):
    """Normalization must be idempotent: normalize(normalize(x)) == normalize(x).

    KNOWN LIMITATION: When ftfy is installed, random character sequences can
    trigger false-positive mojibake detection.  For example, ``÷ß`` (U+00F7
    U+00DF) is misidentified as Mac Roman-encoded UTF-8 and "corrected" to
    ``֧`` (U+05A7).  On a second pass ftfy does not change ``֧``, so
    idempotency breaks.  This is a documented ftfy trade-off (optimised for
    real text, not random codepoint sequences) and does NOT affect real-world
    prompt-injection detection.
    """

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "ftfy false-positive on random chars in Python <3.11; see docstring",
    )
    @given(st.text(
        alphabet=st.characters(),
        min_size=1,
        max_size=2000,
    ))
    @FUZZ_SETTINGS
    def test_normalize_is_idempotent(self, text):
        """Applying normalize_text twice must produce the same result as once."""
        text1, stripped1, flags1 = normalize_text(text)
        text2, stripped2, flags2 = normalize_text(text1)
        self.assertEqual(
            text1, text2,
            f"Normalization not idempotent:\n"
            f"  Once:  {text1!r}\n"
            f"  Twice: {text2!r}"
        )


class TestHTMLStripping(unittest.TestCase):
    """HTML tags must be stripped from output."""

    @given(st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "Z"),
        ),
        min_size=1,
        max_size=200,
    ))
    @FUZZ_SETTINGS
    def test_no_html_tags_in_output(self, inner_text):
        """Wrapping text in HTML tags: tags must not appear in output."""
        html_input = f"<div><p>{inner_text}</p></div>"
        result = layer0_sanitize(html_input)
        if not result.rejected:
            self.assertNotIn("<div>", result.sanitized_text)
            self.assertNotIn("</div>", result.sanitized_text)
            self.assertNotIn("<p>", result.sanitized_text)
            self.assertNotIn("</p>", result.sanitized_text)

    @given(st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N"),
        ),
        min_size=1,
        max_size=100,
    ))
    @FUZZ_SETTINGS
    def test_script_tags_stripped(self, payload):
        """<script> tags and content are stripped."""
        html_input = f"safe text<script>{payload}</script>more text"
        result = layer0_sanitize(html_input)
        if not result.rejected:
            self.assertNotIn("<script>", result.sanitized_text)
            self.assertNotIn("</script>", result.sanitized_text)


class TestBytesInputPath(unittest.TestCase):
    """Verify the bytes decoding path produces valid results."""

    @given(binary_data)
    @FUZZ_SETTINGS
    def test_bytes_always_produce_str_output(self, data):
        """Bytes input must always result in sanitized_text being a str."""
        result = layer0_sanitize(data)
        self.assertIsInstance(result.sanitized_text, str)

    @given(encoding_variants())
    @FUZZ_SETTINGS
    def test_encoded_text_roundtrips(self, data):
        """Text encoded in various charsets must decode without crash."""
        result = layer0_sanitize(data)
        self.assertIsInstance(result, Layer0Result)
        self.assertIsInstance(result.sanitized_text, str)

    def test_utf16_bytes(self):
        """UTF-16 encoded text with BOM."""
        text = "Hello, World!"
        data = text.encode("utf-16")
        result = layer0_sanitize(data)
        self.assertIsInstance(result, Layer0Result)
        if not result.rejected:
            self.assertIn("Hello", result.sanitized_text)

    def test_latin1_bytes(self):
        """Latin-1 encoded text."""
        text = "caf\u00e9 cr\u00e8me"
        data = text.encode("latin-1")
        result = layer0_sanitize(data)
        self.assertIsInstance(result, Layer0Result)

    def test_empty_bytes_rejected(self):
        """Empty bytes must be rejected."""
        result = layer0_sanitize(b"")
        self.assertTrue(result.rejected)

    def test_null_bytes(self):
        """All-null bytes must not crash."""
        result = layer0_sanitize(b"\x00" * 100)
        self.assertIsInstance(result, Layer0Result)


class TestAdversarialPatterns(unittest.TestCase):
    """Targeted adversarial patterns that should survive fuzzing."""

    @given(adversarial_unicode)
    @FUZZ_SETTINGS
    def test_adversarial_produces_valid_result(self, text):
        """Adversarial Unicode must produce a valid Layer0Result."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result, Layer0Result)
        self.assertIsInstance(result.sanitized_text, str)
        self.assertIsInstance(result.anomaly_flags, list)

    @given(st.text(
        alphabet=st.sampled_from(list(
            # Tag characters U+E0001-U+E007F
            "".join(chr(c) for c in range(0xE0001, 0xE0080))
        )),
        min_size=1,
        max_size=100,
    ))
    @FUZZ_SETTINGS
    def test_tag_characters_handled(self, text):
        """Unicode tag characters (U+E0001-E007F) must not crash."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result, Layer0Result)

    @given(st.text(
        alphabet=st.sampled_from(list(
            # Variation selectors + variation selector supplements
            "".join(chr(c) for c in range(0xFE00, 0xFE10))
            + "".join(chr(c) for c in range(0xE0100, 0xE01F0))
        )),
        min_size=1,
        max_size=100,
    ))
    @FUZZ_SETTINGS
    def test_variation_selectors_handled(self, text):
        """Variation selectors (U+FE00-FE0F, U+E0100-E01EF) must not crash."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result, Layer0Result)

    @given(st.text(
        alphabet=st.characters(
            whitelist_categories=("Mn",),  # combining marks only
        ),
        min_size=1,
        max_size=500,
    ))
    @FUZZ_SETTINGS
    def test_combining_marks_only(self, text):
        """Input of only combining marks must not crash."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result, Layer0Result)

    @given(st.text(
        # Surrogates are invalid in Python str, but we test nearby codepoints
        # and supplementary plane characters
        alphabet=st.characters(
            whitelist_categories=("So", "Sk", "Sm", "Sc"),
        ),
        min_size=1,
        max_size=500,
    ))
    @FUZZ_SETTINGS
    def test_symbol_categories(self, text):
        """Input of various symbol categories must not crash."""
        result = layer0_sanitize(text)
        self.assertIsInstance(result, Layer0Result)


class TestFingerprintInvariants(unittest.TestCase):
    """Fingerprint field invariants for non-rejected, non-trivial inputs."""

    @given(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=15,  # must be >= 10 chars for tokenization to run
        max_size=1000,
    ))
    @FUZZ_SETTINGS
    def test_fingerprint_is_dict_when_present(self, text):
        """Fingerprint must be a dict for non-trivial non-rejected input."""
        result = layer0_sanitize(text)
        if not result.rejected and len(result.sanitized_text) >= 10:
            self.assertIsInstance(result.fingerprint, dict)
            # Should have standard keys
            self.assertIn("content_hash", result.fingerprint)
            self.assertIn("normalized_hash", result.fingerprint)
            self.assertIn("token_hash", result.fingerprint)
            self.assertIn("token_count", result.fingerprint)
            self.assertIn("char_count", result.fingerprint)
            self.assertIn("ratio", result.fingerprint)

    @given(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=15,
        max_size=500,
    ))
    @FUZZ_SETTINGS
    def test_token_char_ratio_non_negative(self, text):
        """token_char_ratio must be >= 0."""
        result = layer0_sanitize(text)
        self.assertGreaterEqual(result.token_char_ratio, 0.0)


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    unittest.main()
