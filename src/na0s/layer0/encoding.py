import os

try:
    import chardet
    _HAS_CHARDET = True
except ImportError:
    _HAS_CHARDET = False

# BOM markers — checked before chardet for deterministic results
_BOM_MAP = [
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
]


def _safe_float_env(name, default, lo=0.0, hi=1.0):
    """Read a float from env, clamping to [lo, hi]. Falls back to *default*."""
    import math
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(val):
        return default
    if val < lo or val > hi:
        return default
    return val


# Minimum confidence to trust chardet's detection.
# Below this threshold the ``low_encoding_confidence`` flag is raised.
# Default 0.5 (50%).  Override: L0_MIN_ENCODING_CONFIDENCE
_MIN_CONFIDENCE = _safe_float_env(
    "L0_MIN_ENCODING_CONFIDENCE", 0.5, lo=0.0, hi=1.0
)


def detect_encoding(raw_bytes):
    """Detect the encoding of raw bytes.

    Never assumes UTF-8 — always inspects the actual bytes first.

    Returns (encoding_name, confidence, flags).
    encoding_name is a Python codec name (e.g. "utf-8", "latin-1").
    """
    flags = []

    if not isinstance(raw_bytes, (bytes, bytearray)):
        return "utf-8", 1.0, []

    if not raw_bytes:
        return "utf-8", 1.0, []

    # Step 1: Check for BOM — deterministic, highest priority
    for bom, encoding in _BOM_MAP:
        if raw_bytes.startswith(bom):
            flags.append("bom_detected_{}".format(encoding))
            return encoding, 1.0, flags

    # Step 2: Use chardet for statistical detection
    if not _HAS_CHARDET:
        return "utf-8", 0.5, ["chardet_not_available"]
    result = chardet.detect(raw_bytes)
    encoding = result.get("encoding") or "utf-8"
    confidence = result.get("confidence", 0.0)

    # Normalize common aliases
    encoding = encoding.lower().replace("-", "_")
    alias_map = {
        "ascii": "utf-8",        # ASCII is a subset of UTF-8
        "iso_8859_1": "latin-1",
        "windows_1252": "cp1252",
    }
    encoding = alias_map.get(encoding, encoding).replace("_", "-")

    if confidence < _MIN_CONFIDENCE:
        flags.append("low_encoding_confidence_{:.0f}pct".format(confidence * 100))

    return encoding, confidence, flags


def decode_to_str(raw_bytes):
    """Decode raw bytes to a Python string using detected encoding.

    Returns (decoded_string, encoding_used, flags).
    """
    if isinstance(raw_bytes, str):
        return raw_bytes, "utf-8", []

    if not isinstance(raw_bytes, (bytes, bytearray)):
        return str(raw_bytes), "utf-8", ["coerced_to_str"]

    encoding, confidence, flags = detect_encoding(raw_bytes)

    # Strip BOM bytes before decoding
    for bom, bom_enc in _BOM_MAP:
        if raw_bytes.startswith(bom):
            raw_bytes = raw_bytes[len(bom):]
            break

    try:
        decoded = raw_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        # Fallback: decode as UTF-8 with replacement chars
        flags.append("encoding_fallback_utf8")
        decoded = raw_bytes.decode("utf-8", errors="replace")

    return decoded, encoding, flags
