"""Layer 2: Obfuscation Detection & Decoding for prompt injection detection.

Recursive multi-layer obfuscation scanner handling 10+ encoding types:
- Base64, hex, URL-encoding (with embedded substring extraction)
- ROT13/Caesar cipher, leetspeak normalization, reversed text
- Morse code (ITU-R M.1677 with Unicode dot/dash normalization)
- Binary/octal/decimal ASCII decoding
- Whitespace steganography (SNOW-style, 4 detection methods)
- Shannon entropy + KL-divergence + compression ratio composite scoring
- Matryoshka recursive unwrapping with encoding chain provenance

Public API:
  obfuscation_scan(text)            -> dict
  detect_morse(text)                -> MorseResult
  detect_numeric(text)              -> NumericDecodeResult
  detect_binary(text)               -> NumericDecodeResult
  detect_octal(text)                -> NumericDecodeResult
  detect_decimal(text)              -> NumericDecodeResult
  detect_whitespace_stego(text)     -> StegoResult
  detect_ascii_art(text)            -> AsciiArtResult
  dehyphenate_suspicious(text)      -> SplittingResult
"""

from .obfuscation import (
    obfuscation_scan,
    shannon_entropy,
    DecodedView,
    # Externalized named constants (for testing and env var override)
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
# Back-compat re-exports folded from the deleted top-level obfuscation.py shim
# (tests rely on `from na0s.obfuscation import _decode_base64`, patterns, etc.).
from .obfuscation import (  # noqa: F401
    PUNCTUATION_PATTERN, BASE64_PATTERN, HEX_PATTERN, URLENCODED_PATTERN,
    _kl_divergence_from_english, _compression_ratio, _composite_entropy_check,
    _scan_single_layer, _build_encoding_chains, _has_attack_keywords,
    _is_rot13_candidate, _is_reversed_candidate, _is_leetspeak_candidate,
    _is_morse_candidate, _is_numeric_candidate, _decode_rot13, _decode_base64,
    _decode_hex, _decode_url, _normalize_leetspeak, _leet_density,
    _punctuation_ratio, _casing_transitions, _casing_transition_ratio,
    _is_structured_data, _extract_embedded_base64, _extract_embedded_hex,
    _caesar_brute_force, _caesar_shift, _validate_english, _detect_pig_latin,
    _decode_pig_latin_word, _ENGLISH_COMMON_WORDS, _ENGLISH_AY_WORDS,
)
from .morse_code import detect_morse, MorseResult
from .numeric_decode import (
    detect_numeric,
    detect_binary,
    detect_octal,
    detect_decimal,
    NumericDecodeResult,
)
from .whitespace_stego import detect_whitespace_stego, StegoResult
from .ascii_art_detector import detect_ascii_art, AsciiArtResult
from .syllable_splitting import dehyphenate_suspicious, SplittingResult
