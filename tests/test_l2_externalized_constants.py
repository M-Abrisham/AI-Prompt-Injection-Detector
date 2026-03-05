"""Tests for externalized named constants in Layer 2 obfuscation module.

Verifies that:
1. Default values match expected defaults
2. Environment variable overrides work correctly (via monkeypatch)
3. Edge cases: negative values, zero, very large values, non-numeric strings
4. The obfuscation_scan function uses the named constants correctly
"""

import importlib
import pytest

from na0s.layer2 import obfuscation as obs_module
from na0s.layer2.obfuscation import (
    PUNCTUATION_FLOOD_RATIO,
    CASING_TRANSITION_THRESHOLD,
    CASING_TRANSITION_RATIO,
    DEFAULT_MAX_DECODES,
    MIN_BASE64_LENGTH,
    MIN_HEX_LENGTH,
    MIN_PRINTABLE_CHARS,
    MIN_PRINTABLE_RATIO,
    MIN_CANDIDATE_ALPHA,
    MIN_ENTROPY_TEXT_LENGTH,
    MIN_KL_LETTERS,
    MIN_DECODED_STRIP_LENGTH,
    ZLIB_COMPRESSION_LEVEL,
    _env_float,
    _env_int,
    obfuscation_scan,
)


# ---------------------------------------------------------------------------
# 1. Default values match expected
# ---------------------------------------------------------------------------

class TestDefaultValues:
    """Each named constant must have the documented default."""

    def test_punctuation_flood_ratio_default(self):
        assert PUNCTUATION_FLOOD_RATIO == 0.40

    def test_casing_transition_threshold_default(self):
        assert CASING_TRANSITION_THRESHOLD == 6

    def test_casing_transition_ratio_default(self):
        assert CASING_TRANSITION_RATIO == 0.12

    def test_default_max_decodes(self):
        assert DEFAULT_MAX_DECODES == 5

    def test_min_base64_length_default(self):
        assert MIN_BASE64_LENGTH == 16

    def test_min_hex_length_default(self):
        assert MIN_HEX_LENGTH == 8

    def test_min_printable_chars_default(self):
        assert MIN_PRINTABLE_CHARS == 3

    def test_min_printable_ratio_default(self):
        assert MIN_PRINTABLE_RATIO == 0.7

    def test_min_candidate_alpha_default(self):
        assert MIN_CANDIDATE_ALPHA == 10

    def test_min_entropy_text_length_default(self):
        assert MIN_ENTROPY_TEXT_LENGTH == 10

    def test_min_kl_letters_default(self):
        assert MIN_KL_LETTERS == 5

    def test_min_decoded_strip_length_default(self):
        assert MIN_DECODED_STRIP_LENGTH == 2

    def test_zlib_compression_level_default(self):
        assert ZLIB_COMPRESSION_LEVEL == 6


# ---------------------------------------------------------------------------
# 2. _env_float / _env_int helper functions
# ---------------------------------------------------------------------------

class TestEnvFloat:
    """Unit tests for the _env_float helper."""

    def test_returns_default_when_absent(self, monkeypatch):
        monkeypatch.delenv("NA0S_TEST_FLOAT", raising=False)
        assert _env_float("NA0S_TEST_FLOAT", 3.14) == 3.14

    def test_reads_valid_float(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_FLOAT", "0.55")
        assert _env_float("NA0S_TEST_FLOAT", 3.14) == 0.55

    def test_reads_negative_float(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_FLOAT", "-1.5")
        assert _env_float("NA0S_TEST_FLOAT", 3.14) == -1.5

    def test_reads_zero(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_FLOAT", "0")
        assert _env_float("NA0S_TEST_FLOAT", 3.14) == 0.0

    def test_reads_very_large_float(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_FLOAT", "1e10")
        assert _env_float("NA0S_TEST_FLOAT", 3.14) == 1e10

    def test_returns_default_for_garbage(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_FLOAT", "not_a_number")
        assert _env_float("NA0S_TEST_FLOAT", 3.14) == 3.14

    def test_returns_default_for_empty(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_FLOAT", "")
        assert _env_float("NA0S_TEST_FLOAT", 3.14) == 3.14


class TestEnvInt:
    """Unit tests for the _env_int helper."""

    def test_returns_default_when_absent(self, monkeypatch):
        monkeypatch.delenv("NA0S_TEST_INT", raising=False)
        assert _env_int("NA0S_TEST_INT", 42) == 42

    def test_reads_valid_int(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_INT", "10")
        assert _env_int("NA0S_TEST_INT", 42) == 10

    def test_reads_negative_int(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_INT", "-5")
        assert _env_int("NA0S_TEST_INT", 42) == -5

    def test_reads_zero(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_INT", "0")
        assert _env_int("NA0S_TEST_INT", 42) == 0

    def test_reads_very_large_int(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_INT", "999999999")
        assert _env_int("NA0S_TEST_INT", 42) == 999999999

    def test_returns_default_for_garbage(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_INT", "abc")
        assert _env_int("NA0S_TEST_INT", 42) == 42

    def test_returns_default_for_float_string(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_INT", "3.14")
        assert _env_int("NA0S_TEST_INT", 42) == 42

    def test_returns_default_for_empty(self, monkeypatch):
        monkeypatch.setenv("NA0S_TEST_INT", "")
        assert _env_int("NA0S_TEST_INT", 42) == 42


# ---------------------------------------------------------------------------
# 3. Env var overrides are picked up on module reload
# ---------------------------------------------------------------------------

class TestEnvVarOverridesOnReload:
    """Verify that setting NA0S_* env vars and reloading the module
    changes the constants at module level."""

    def test_punctuation_flood_ratio_override(self, monkeypatch):
        monkeypatch.setenv("NA0S_PUNCTUATION_FLOOD_RATIO", "0.55")
        importlib.reload(obs_module)
        assert obs_module.PUNCTUATION_FLOOD_RATIO == 0.55
        # Restore
        monkeypatch.delenv("NA0S_PUNCTUATION_FLOOD_RATIO", raising=False)
        importlib.reload(obs_module)

    def test_casing_transition_threshold_override(self, monkeypatch):
        monkeypatch.setenv("NA0S_CASING_TRANSITION_THRESHOLD", "12")
        importlib.reload(obs_module)
        assert obs_module.CASING_TRANSITION_THRESHOLD == 12
        monkeypatch.delenv("NA0S_CASING_TRANSITION_THRESHOLD", raising=False)
        importlib.reload(obs_module)

    def test_casing_transition_ratio_override(self, monkeypatch):
        monkeypatch.setenv("NA0S_CASING_TRANSITION_RATIO", "0.25")
        importlib.reload(obs_module)
        assert obs_module.CASING_TRANSITION_RATIO == 0.25
        monkeypatch.delenv("NA0S_CASING_TRANSITION_RATIO", raising=False)
        importlib.reload(obs_module)

    def test_max_decodes_override(self, monkeypatch):
        monkeypatch.setenv("NA0S_MAX_DECODES", "10")
        importlib.reload(obs_module)
        assert obs_module.DEFAULT_MAX_DECODES == 10
        monkeypatch.delenv("NA0S_MAX_DECODES", raising=False)
        importlib.reload(obs_module)

    def test_min_base64_length_override(self, monkeypatch):
        monkeypatch.setenv("NA0S_MIN_BASE64_LENGTH", "32")
        importlib.reload(obs_module)
        assert obs_module.MIN_BASE64_LENGTH == 32
        monkeypatch.delenv("NA0S_MIN_BASE64_LENGTH", raising=False)
        importlib.reload(obs_module)

    def test_invalid_override_uses_default(self, monkeypatch):
        monkeypatch.setenv("NA0S_PUNCTUATION_FLOOD_RATIO", "not_a_float")
        importlib.reload(obs_module)
        assert obs_module.PUNCTUATION_FLOOD_RATIO == 0.40
        monkeypatch.delenv("NA0S_PUNCTUATION_FLOOD_RATIO", raising=False)
        importlib.reload(obs_module)


# ---------------------------------------------------------------------------
# 4. Edge cases: negative, zero, very large values
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Ensure constants handle extreme values gracefully at the
    function level (not just parsing)."""

    def test_zero_punctuation_ratio_flags_everything(self, monkeypatch):
        """If PUNCTUATION_FLOOD_RATIO is 0.0, even minimal punctuation triggers."""
        monkeypatch.setattr(obs_module, "PUNCTUATION_FLOOD_RATIO", 0.0)
        # Text with any punctuation at all
        result = obs_module._scan_single_layer("Hello, world!")
        flags = result[0]
        assert "punctuation_flood" in flags

    def test_high_punctuation_ratio_disables_flag(self, monkeypatch):
        """If PUNCTUATION_FLOOD_RATIO is very high, normal punctuated text won't trigger."""
        monkeypatch.setattr(obs_module, "PUNCTUATION_FLOOD_RATIO", 0.99)
        # This has a high ratio but not 99%
        result = obs_module._scan_single_layer("Hello! World? Yes!!!")
        flags = result[0]
        assert "punctuation_flood" not in flags

    def test_zero_casing_threshold_triggers_easily(self, monkeypatch):
        """If CASING_TRANSITION_THRESHOLD is 0, almost any mixed-case text triggers."""
        monkeypatch.setattr(obs_module, "CASING_TRANSITION_THRESHOLD", 0)
        monkeypatch.setattr(obs_module, "CASING_TRANSITION_RATIO", 0.0)
        # Simple text with at least one transition
        result = obs_module._scan_single_layer("Hello World How Are You Today")
        flags = result[0]
        assert "weird_casing" in flags

    def test_very_large_casing_threshold_disables_flag(self, monkeypatch):
        """If CASING_TRANSITION_THRESHOLD is huge, nothing triggers."""
        monkeypatch.setattr(obs_module, "CASING_TRANSITION_THRESHOLD", 99999)
        result = obs_module._scan_single_layer("aLtErNaTiNg CaSe TeXt HeRe NoW")
        flags = result[0]
        assert "weird_casing" not in flags

    def test_zero_min_entropy_text_length(self, monkeypatch):
        """MIN_ENTROPY_TEXT_LENGTH=0 allows very short text to be evaluated."""
        monkeypatch.setattr(obs_module, "MIN_ENTROPY_TEXT_LENGTH", 0)
        # Should not crash; short text simply won't have high entropy
        result = obs_module._composite_entropy_check("abc")
        assert isinstance(result, bool)

    def test_negative_min_kl_letters_still_works(self, monkeypatch):
        """Negative MIN_KL_LETTERS should not crash."""
        monkeypatch.setattr(obs_module, "MIN_KL_LETTERS", -5)
        result = obs_module._kl_divergence_from_english("ab")
        assert isinstance(result, float)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# 5. obfuscation_scan uses DEFAULT_MAX_DECODES
# ---------------------------------------------------------------------------

class TestMaxDecodesDefault:
    """The default max_decodes parameter should be DEFAULT_MAX_DECODES (5)."""

    def test_default_signature(self):
        """obfuscation_scan default max_decodes should be 5 (not 2)."""
        import inspect
        sig = inspect.signature(obs_module.obfuscation_scan)
        default = sig.parameters["max_decodes"].default
        assert default == 5

    def test_explicit_max_decodes_still_works(self):
        """Passing max_decodes=1 explicitly should still be accepted."""
        result = obs_module.obfuscation_scan("Hello world", max_decodes=1)
        assert "obfuscation_score" in result

    def test_max_decodes_zero(self):
        """max_decodes=0 should still use _DEFAULT_MAX_TOTAL_DECODES as floor."""
        result = obs_module.obfuscation_scan("Hello world", max_decodes=0)
        assert "obfuscation_score" in result

    def test_backward_compat_result_keys(self):
        """All expected result keys must be present."""
        result = obs_module.obfuscation_scan("Hello world")
        expected_keys = {
            "obfuscation_score", "decoded_views", "evasion_flags",
            "decoded_chain", "max_depth_reached", "encoding_chains",
        }
        assert expected_keys.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# 6. Constants are accessible from the public __init__ re-exports
# ---------------------------------------------------------------------------

class TestPublicImports:
    """Verify constants are re-exported from na0s.layer2."""

    def test_imports_from_layer2_package(self):
        from na0s.layer2 import (
            PUNCTUATION_FLOOD_RATIO,
            CASING_TRANSITION_THRESHOLD,
            CASING_TRANSITION_RATIO,
            DEFAULT_MAX_DECODES,
            MIN_BASE64_LENGTH,
            MIN_HEX_LENGTH,
            MIN_PRINTABLE_CHARS,
            MIN_PRINTABLE_RATIO,
            MIN_CANDIDATE_ALPHA,
            MIN_ENTROPY_TEXT_LENGTH,
            MIN_KL_LETTERS,
            MIN_DECODED_STRIP_LENGTH,
            ZLIB_COMPRESSION_LEVEL,
        )
        # Smoke check: they should all be numeric
        for val in [
            PUNCTUATION_FLOOD_RATIO, CASING_TRANSITION_RATIO,
            MIN_PRINTABLE_RATIO,
        ]:
            assert isinstance(val, float)
        for val in [
            CASING_TRANSITION_THRESHOLD, DEFAULT_MAX_DECODES,
            MIN_BASE64_LENGTH, MIN_HEX_LENGTH, MIN_PRINTABLE_CHARS,
            MIN_CANDIDATE_ALPHA, MIN_ENTROPY_TEXT_LENGTH,
            MIN_KL_LETTERS, MIN_DECODED_STRIP_LENGTH,
            ZLIB_COMPRESSION_LEVEL,
        ]:
            assert isinstance(val, int)
