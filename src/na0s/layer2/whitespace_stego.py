"""Whitespace steganography detection for prompt injection.

Detects SNOW-style trailing tab/space encoding, simple binary whitespace
encoding (space=0/tab=1), and statistical anomalies in trailing whitespace.

MUST be called on RAW text before L0 normalize_text() strips trailing WS.

Detection methods (decreasing confidence):
  1. SNOW structural detection -- tab-delimited 3-bit groups
  2. Statistical anomaly -- high trailing WS ratio + entropy
  3. Simple binary encoding -- space=0, tab=1 byte decode
  4. Trailing WS anomaly -- high ratio but low entropy

No external dependencies -- stdlib only.

Public API:
    detect_whitespace_stego(text) -> StegoResult
"""

import math
import re
from dataclasses import dataclass, field

from ._env_utils import safe_float_env, safe_int_env


# ---------------------------------------------------------------------------
# Configurable thresholds (env-overridable)
# ---------------------------------------------------------------------------

SNOW_MIN_LINES = safe_int_env("WS_STEGO_SNOW_MIN_LINES", 2, lo=1)
RATIO_THRESHOLD = safe_float_env("WS_STEGO_RATIO_THRESHOLD", 0.30, lo=0.0, hi=1.0)
ENTROPY_THRESHOLD = safe_float_env("WS_STEGO_ENTROPY_THRESHOLD", 0.50, lo=0.0, hi=1.0)
MIN_TRAILING_BYTES = safe_int_env("WS_STEGO_MIN_TRAILING_BYTES", 8, lo=1)
MAX_INPUT_LENGTH = safe_int_env("WS_STEGO_MAX_INPUT_LENGTH", 1_000_000, lo=1)
MIN_PRINTABLE_RATIO = safe_float_env(
    "WS_STEGO_MIN_PRINTABLE_RATIO", 0.60, lo=0.0, hi=1.0
)

# Named confidence levels — avoids magic numbers scattered across methods.
_CONFIDENCE_SNOW = 0.95
_CONFIDENCE_STATISTICAL = 0.70
_CONFIDENCE_BINARY = 0.60
_CONFIDENCE_ANOMALY = 0.50

# Minimum decoded payload length to accept (suppresses noise from 1-2 char
# accidental decodes on short trailing WS).
_MIN_DECODED_LEN = 3

# Minimum trailing bytes for statistical method (higher than general
# MIN_TRAILING_BYTES to avoid FPs on mixed-indentation code).
_MIN_STATISTICAL_BYTES = 24

# Very high trailing data volume — flag even without a decoded payload.
_HIGH_VOLUME_BYTES = 48


# ---------------------------------------------------------------------------
# Pre-compiled patterns (module-level for performance)
# ---------------------------------------------------------------------------

# SNOW pattern: TAB followed by at LEAST one data group (0-7 spaces + TAB),
# optionally ending with 0-7 trailing spaces.  A lone TAB is NOT a valid
# SNOW data line — it's just a tab character.  Requiring one inner group
# eliminates single-tab false positives.
_SNOW_LINE_RE = re.compile(r"^\t[ ]{0,7}\t([ ]{0,7}\t)*[ ]{0,7}\t?$")

# Markdown 2-space line break: exactly 2 trailing spaces.
_MARKDOWN_BR_RE = re.compile(r"^  $")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StegoResult:
    """Result of whitespace steganography analysis."""

    detected: bool = False
    confidence: float = 0.0
    method: str = ""            # "snow", "binary", "statistical", "anomaly"
    decoded_payload: str = ""
    flags: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_trailing_whitespace(text):
    """Split text into lines and extract trailing whitespace per line.

    Only tabs and spaces are considered trailing whitespace.  Carriage
    returns (\\r) are stripped from both the visible content and the
    trailing WS to prevent CRLF line endings from inflating byte counts
    or confusing tab/space analysis.

    Parameters
    ----------
    text : str
        Raw input text (newlines preserved).

    Returns
    -------
    list[tuple[int, str, str]]
        List of (line_index, visible_content, trailing_whitespace).
        Trailing whitespace contains only space (0x20) and tab (0x09).
    """
    results = []
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        # Strip \r so CRLF endings don't pollute trailing WS analysis.
        clean = line.rstrip("\r")
        visible = clean.rstrip(" \t")
        trailing = clean[len(visible):]
        results.append((idx, visible, trailing))
    return results


def _shannon_entropy(chars):
    """Compute Shannon entropy (base 2) of a character string.

    Parameters
    ----------
    chars : str
        String of characters to analyse.

    Returns
    -------
    float
        Entropy in bits.  Returns 0.0 for empty input.
    """
    if not chars:
        return 0.0
    length = len(chars)
    freq = {}
    for ch in chars:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _detect_snow_pattern(trailing_ws_list):
    """Check for SNOW structural pattern in trailing whitespace.

    Parameters
    ----------
    trailing_ws_list : list[str]
        List of trailing whitespace strings (one per line).

    Returns
    -------
    tuple[bool, int]
        (is_snow_pattern, num_matching_lines)
    """
    matching = 0
    for ws in trailing_ws_list:
        if not ws:
            continue
        if _SNOW_LINE_RE.match(ws):
            matching += 1
    return (matching >= SNOW_MIN_LINES, matching)


def _snow_bit_reverse(val):
    """Apply SNOW bit reversal: swap bits 0 and 2, keep bit 1.

    This is its own inverse: applying it twice returns the original value.

    Parameters
    ----------
    val : int
        3-bit value (0-7).

    Returns
    -------
    int
        Bit-reversed 3-bit value.
    """
    return ((val & 1) << 2) | (val & 2) | ((val & 4) >> 2)


def _decode_snow(trailing_ws_list):
    """Attempt SNOW binary decoding of trailing whitespace.

    Each line's trailing WS is split on TABs.  Each segment's length
    (0-7 spaces) encodes a 3-bit value after bit reversal.

    Parameters
    ----------
    trailing_ws_list : list[str]
        List of trailing whitespace strings (one per line).

    Returns
    -------
    str
        Decoded ASCII string (may be empty or contain unprintable chars).
    """
    three_bit_values = []
    for ws in trailing_ws_list:
        if not ws:
            continue
        # Skip lines that don't start with a TAB (no SNOW start marker)
        if not ws.startswith("\t"):
            continue
        # Split on TABs — first element is always '' (before leading tab,
        # the SNOW start marker) and last element may be '' (after trailing
        # tab).  We skip both and only process the inner data groups.
        groups = ws.split("\t")
        # Skip groups[0] (start marker) — always empty
        data_groups = groups[1:]
        # If the last group is empty, it is the trailing TAB end-marker;
        # an empty group between TABs means "0 spaces encoded" which is
        # a valid 3-bit value of 0, so we only strip the very last one.
        if data_groups and data_groups[-1] == "":
            data_groups = data_groups[:-1]
        for g in data_groups:
            # Each group should be 0-7 spaces only
            if g and not all(c == " " for c in g):
                continue  # Not a valid SNOW group
            n_spaces = len(g)
            if n_spaces > 7:
                continue  # Invalid SNOW encoding
            val = _snow_bit_reverse(n_spaces)
            three_bit_values.append(val)

    if not three_bit_values:
        return ""

    # Assemble 3-bit values into bytes (8 bits each).
    # Null byte (0x00) signals end of message or padding.
    bit_buffer = 0
    bits_in_buffer = 0
    decoded_bytes = []

    for val in three_bit_values:
        bit_buffer = (bit_buffer << 3) | val
        bits_in_buffer += 3
        while bits_in_buffer >= 8:
            bits_in_buffer -= 8
            byte_val = (bit_buffer >> bits_in_buffer) & 0xFF
            if byte_val == 0:
                break
            decoded_bytes.append(byte_val)
        else:
            continue
        break  # Inner break triggered => stop outer loop too

    try:
        return bytes(decoded_bytes).decode("ascii", errors="replace")
    except Exception:
        return ""


def _decode_binary_ws(trailing_ws_list):
    """Attempt simple space=0 / tab=1 binary decoding.

    Concatenates trailing whitespace from all lines, maps space->0 and
    tab->1, groups into 8-bit bytes, and decodes as ASCII.

    Parameters
    ----------
    trailing_ws_list : list[str]
        List of trailing whitespace strings (one per line).

    Returns
    -------
    str
        Decoded ASCII string (may be empty).
    """
    bits = []
    for ws in trailing_ws_list:
        for ch in ws:
            if ch == " ":
                bits.append(0)
            elif ch == "\t":
                bits.append(1)

    if len(bits) < 8:
        return ""

    decoded_bytes = []
    for i in range(0, len(bits) - 7, 8):
        byte_val = 0
        for bit in bits[i : i + 8]:
            byte_val = (byte_val << 1) | bit
        if byte_val == 0:
            break
        decoded_bytes.append(byte_val)

    try:
        return bytes(decoded_bytes).decode("ascii", errors="replace")
    except Exception:
        return ""


def _printable_ratio(text):
    """Return fraction of printable ASCII characters in *text*.

    Printable ASCII is 0x20-0x7E (visible chars + space) plus \\n and \\r.
    Tabs are NOT counted as printable in decoded payloads — a decoded
    message of all tabs should not pass the printability check.
    """
    if not text:
        return 0.0
    printable = sum(
        1 for c in text if 0x20 <= ord(c) <= 0x7E or c in "\n\r"
    )
    return printable / len(text)


def _filter_markdown_breaks(trailing_ws_list):
    """Replace markdown 2-space line breaks with empty strings.

    Returns a new list with the same length — markdown break entries
    are replaced with '' so they don't contribute to WS analysis or
    decoding attempts.
    """
    result = []
    for ws in trailing_ws_list:
        if ws and _MARKDOWN_BR_RE.match(ws):
            result.append("")
        else:
            result.append(ws)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_whitespace_stego(text):
    """Detect whitespace steganography in text.

    Checks for SNOW-style encoding, simple binary encoding,
    and statistical anomalies in trailing whitespace.

    Parameters
    ----------
    text : str
        Raw input text (before normalization).

    Returns
    -------
    StegoResult
        Detection result with confidence, method, decoded payload, and stats.
    """
    # Guard: non-string or empty input
    if not isinstance(text, str) or not text:
        return StegoResult()

    # Truncate oversized inputs at the last newline boundary to avoid
    # splitting a line mid-way (which would create a partial trailing WS
    # that triggers a false positive on the boundary line).
    if len(text) > MAX_INPUT_LENGTH:
        cut = text[:MAX_INPUT_LENGTH]
        last_nl = cut.rfind("\n")
        scan_text = cut[: last_nl + 1] if last_nl >= 0 else cut
    else:
        scan_text = text

    # Step 1: Extract trailing whitespace per line
    line_data = _extract_trailing_whitespace(scan_text)
    total_lines = len(line_data)
    if total_lines == 0:
        return StegoResult()

    trailing_ws_list = [ws for _, _, ws in line_data]

    # Filter out markdown 2-space line breaks — used consistently by all
    # methods below so decoding never interprets line breaks as data.
    filtered_ws = _filter_markdown_breaks(trailing_ws_list)

    # Step 2: Compute statistics (on filtered list)
    lines_with_ws = sum(1 for ws in filtered_ws if ws)
    trailing_ws_ratio = lines_with_ws / max(total_lines, 1)

    all_trailing = "".join(filtered_ws)
    total_trailing_bytes = len(all_trailing)

    # Count tabs and spaces in trailing WS
    total_tabs = all_trailing.count("\t")
    total_spaces = all_trailing.count(" ")

    # Shannon entropy of trailing whitespace characters
    ws_entropy = _shannon_entropy(all_trailing)

    stats = {
        "total_lines": total_lines,
        "lines_with_trailing_ws": lines_with_ws,
        "trailing_ws_ratio": round(trailing_ws_ratio, 4),
        "total_trailing_bytes": total_trailing_bytes,
        "trailing_tabs": total_tabs,
        "trailing_spaces": total_spaces,
        "ws_entropy": round(ws_entropy, 4),
        "decoded_printable_ratio": 0.0,
    }

    # Skip if not enough trailing WS data
    if total_trailing_bytes < MIN_TRAILING_BYTES:
        return StegoResult(stats=stats)

    # Skip SNOW/binary detection if trailing WS is homogeneous (entropy ~= 0)
    # — pure spaces from code indentation, pure tabs from TSV, etc.
    has_mixed_ws = total_tabs > 0 and total_spaces > 0

    # Effective ratio after markdown filtering
    effective_ratio = lines_with_ws / max(total_lines, 1)

    # Cache SNOW decode result so Methods 1 and 2 don't redundantly decode.
    _snow_decoded = None  # sentinel: None = not attempted, str = result

    # -------------------------------------------------------------------
    # Method 1: SNOW structural detection
    # -------------------------------------------------------------------
    if has_mixed_ws:
        is_snow, snow_lines = _detect_snow_pattern(filtered_ws)
        stats["snow_pattern_lines"] = snow_lines

        if is_snow:
            _snow_decoded = _decode_snow(filtered_ws)
            pr = _printable_ratio(_snow_decoded)
            stats["decoded_printable_ratio"] = round(pr, 4)

            if (_snow_decoded
                    and len(_snow_decoded) >= _MIN_DECODED_LEN
                    and pr >= MIN_PRINTABLE_RATIO):
                return StegoResult(
                    detected=True,
                    confidence=_CONFIDENCE_SNOW,
                    method="snow",
                    decoded_payload=_snow_decoded,
                    flags=["whitespace_stego_snow"],
                    stats=stats,
                )

    # -------------------------------------------------------------------
    # Method 2: Statistical anomaly
    # -------------------------------------------------------------------
    if (
        has_mixed_ws
        and effective_ratio >= RATIO_THRESHOLD
        and ws_entropy >= ENTROPY_THRESHOLD
        and total_trailing_bytes >= _MIN_STATISTICAL_BYTES
    ):
        # Re-use cached decode from Method 1 if available.
        if _snow_decoded is None:
            _snow_decoded = _decode_snow(filtered_ws)
        decoded = _snow_decoded
        pr = _printable_ratio(decoded) if decoded else 0.0
        stats["decoded_printable_ratio"] = round(pr, 4)

        has_payload = (decoded
                       and pr >= MIN_PRINTABLE_RATIO
                       and len(decoded) >= _MIN_DECODED_LEN)
        if has_payload or total_trailing_bytes >= _HIGH_VOLUME_BYTES:
            return StegoResult(
                detected=True,
                confidence=_CONFIDENCE_STATISTICAL,
                method="statistical",
                decoded_payload=decoded if has_payload else "",
                flags=["whitespace_stego_suspicious"],
                stats=stats,
            )

    # -------------------------------------------------------------------
    # Method 3: Simple binary encoding
    # -------------------------------------------------------------------
    if has_mixed_ws and total_trailing_bytes >= 8:
        decoded = _decode_binary_ws(filtered_ws)
        pr = _printable_ratio(decoded)
        stats["decoded_printable_ratio"] = round(pr, 4)

        if (decoded
                and len(decoded) >= _MIN_DECODED_LEN
                and pr >= MIN_PRINTABLE_RATIO):
            return StegoResult(
                detected=True,
                confidence=_CONFIDENCE_BINARY,
                method="binary",
                decoded_payload=decoded,
                flags=["whitespace_stego_binary"],
                stats=stats,
            )

    # -------------------------------------------------------------------
    # Method 4: High trailing WS anomaly
    # -------------------------------------------------------------------
    # Require mixed WS (tabs AND spaces) — pure trailing spaces from code
    # editors are not suspicious enough to flag.
    if effective_ratio >= 0.50 and has_mixed_ws:
        return StegoResult(
            detected=True,
            confidence=_CONFIDENCE_ANOMALY,
            method="anomaly",
            decoded_payload="",
            flags=["trailing_whitespace_anomaly"],
            stats=stats,
        )

    # No steganography detected
    return StegoResult(stats=stats)
