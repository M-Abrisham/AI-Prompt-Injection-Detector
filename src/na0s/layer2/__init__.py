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
