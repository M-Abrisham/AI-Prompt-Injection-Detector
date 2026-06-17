# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.layer2.obfuscation

All obfuscation logic now lives in ``na0s.layer2.obfuscation``.
This module preserves ``from na0s.obfuscation import ...`` for existing
consumers and test files.
"""

from .layer2.obfuscation import (  # noqa: F401
    obfuscation_scan,
    shannon_entropy,
    DecodedView,
    PUNCTUATION_PATTERN,
    BASE64_PATTERN,
    HEX_PATTERN,
    URLENCODED_PATTERN,
    _kl_divergence_from_english,
    _compression_ratio,
    _composite_entropy_check,
    _scan_single_layer,
    _build_encoding_chains,
    _has_attack_keywords,
    _is_rot13_candidate,
    _is_reversed_candidate,
    _is_leetspeak_candidate,
    _is_morse_candidate,
    _is_numeric_candidate,
    _decode_rot13,
    _decode_base64,
    _decode_hex,
    _decode_url,
    _normalize_leetspeak,
    _leet_density,
    _punctuation_ratio,
    _casing_transitions,
    _casing_transition_ratio,
    _is_structured_data,
    _extract_embedded_base64,
    _extract_embedded_hex,
    _caesar_brute_force,
    _caesar_shift,
    _validate_english,
    _detect_pig_latin,
    _decode_pig_latin_word,
    _ENGLISH_COMMON_WORDS,
    _ENGLISH_AY_WORDS,
)
