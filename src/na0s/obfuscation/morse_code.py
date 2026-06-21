"""Layer 2 Morse code detection -- decode Morse-encoded text to reveal hidden attacks.

Detects International Morse Code (ITU-R M.1677) embedded in text, including
Unicode dot/dash variants (middle dot, bullet, en dash, em dash, etc.).
Attackers can encode prompt injection payloads in Morse to bypass tokenization
and keyword-based detection.

Detection approach:
  1. Normalize Unicode dot/dash alternatives to standard '.' and '-'
  2. Check for explicit Morse labels ("Morse:", "morse code:", etc.)
  3. Calculate Morse density (fraction of valid Morse chars)
  4. Attempt decode and validate (>= 3 printable chars)
  5. Confidence = density * decode success factor

FP mitigation:
  - Exempt markdown headers (# + space)
  - Exempt horizontal rules (--- alone)
  - Exempt ellipsis in prose (... surrounded by words)
  - Exempt IP addresses (dotted quad notation)
  - Require minimum 10 non-whitespace chars of Morse content

No external dependencies -- stdlib only.

Public API:
    detect_morse(text)    -> MorseResult
    decode_morse(text)    -> str
    normalize_morse(text) -> str
    morse_density(text)   -> float
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# International Morse Code mapping (ITU-R M.1677)
# ---------------------------------------------------------------------------

MORSE_TO_CHAR = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9',
}

# Reverse mapping for reference / validation
CHAR_TO_MORSE = {v: k for k, v in MORSE_TO_CHAR.items()}


# ---------------------------------------------------------------------------
# Unicode alternatives for dots and dashes
# ---------------------------------------------------------------------------
# Normalize these to standard '.' and '-' before decoding.

UNICODE_DOT_CHARS = {
    '\u00B7',  # MIDDLE DOT
    '\u2022',  # BULLET
    '\u2219',  # BULLET OPERATOR
    '\u22C5',  # DOT OPERATOR
    '\u25CF',  # BLACK CIRCLE
    '\u25CB',  # WHITE CIRCLE
}

UNICODE_DASH_CHARS = {
    '\u2013',  # EN DASH
    '\u2014',  # EM DASH
    '\u2212',  # MINUS SIGN
    '\u2015',  # HORIZONTAL BAR
}


# ---------------------------------------------------------------------------
# Pre-compiled patterns
# ---------------------------------------------------------------------------

# Explicit Morse label: "Morse:", "morse code:", "decode this morse:", etc.
_MORSE_LABEL_RE = re.compile(
    r"(?:decode\s+(?:this\s+)?)?morse(?:\s+code)?\s*[:;=\-]\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)

# IP address pattern: dotted quad like 192.168.1.1
_IP_ADDR_RE = re.compile(
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
)

# Ellipsis surrounded by word characters: "Wait... let me"
_ELLIPSIS_RE = re.compile(
    r"\w\.{2,4}\s"
)

# Markdown header: line starting with # + space
_MARKDOWN_HEADER_RE = re.compile(
    r"^\s*#{1,6}\s+"
)

# Horizontal rule: line that is only dashes, asterisks, or underscores (3+),
# possibly with spaces — covers all three Markdown HR variants.
_HORIZONTAL_RULE_RE = re.compile(
    r"^\s*(?:[-]{3,}|[*]{3,}|[_]{3,})\s*$"
)

# Valid Morse characters (for density calculation)
_MORSE_CHARS = frozenset('.-/|')

# Minimum non-whitespace length for Morse content
_MIN_MORSE_LENGTH = 10

# Morse density threshold for auto-detection (no explicit label)
_DENSITY_THRESHOLD = 0.80

# Minimum decoded printable characters
_MIN_DECODED_CHARS = 3

# Maximum input length for scanning
_MAX_SCAN_LENGTH = 100_000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MorseResult:
    """Result of Morse code detection analysis."""
    detected: bool = False
    decoded_text: str = ""
    confidence: float = 0.0
    density: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def normalize_morse(text):
    """Replace Unicode dot/dash variants with standard ASCII chars.

    Parameters
    ----------
    text : str
        Input text possibly containing Unicode dot/dash characters.

    Returns
    -------
    str
        Text with Unicode dots replaced by '.' and dashes by '-'.
    """
    result = []
    for ch in text:
        if ch in UNICODE_DOT_CHARS:
            result.append('.')
        elif ch in UNICODE_DASH_CHARS:
            result.append('-')
        else:
            result.append(ch)
    return ''.join(result)


def decode_morse(morse_text):
    """Decode standard Morse code to plaintext.

    Word separators: ' / ' (slash with spaces) or '   ' (3+ spaces).
    Character separators: ' ' (single space) or '|' (pipe).

    Parameters
    ----------
    morse_text : str
        Morse-encoded text using dots (.) and dashes (-).

    Returns
    -------
    str
        Decoded plaintext (uppercase). Empty string if decoding fails.
    """
    if not morse_text or not morse_text.strip():
        return ""

    # First split by word separators: ' / ' or 3+ spaces
    # Normalize: replace ' / ' with a unique word separator token
    text = morse_text.strip()
    text = text.replace(' / ', '\x00')
    # Also treat 3+ consecutive spaces as word separator
    text = re.sub(r' {3,}', '\x00', text)

    words = text.split('\x00')
    decoded_words = []

    for word in words:
        word = word.strip()
        if not word:
            continue

        # Split characters by single space or pipe
        chars = re.split(r'[| ]+', word)
        decoded_chars = []

        for char_code in chars:
            char_code = char_code.strip()
            if not char_code:
                continue
            decoded_char = MORSE_TO_CHAR.get(char_code)
            if decoded_char is not None:
                decoded_chars.append(decoded_char)
            # Skip unrecognized sequences silently

        if decoded_chars:
            decoded_words.append(''.join(decoded_chars))

    return ' '.join(decoded_words)


def morse_density(text):
    """Calculate fraction of non-whitespace chars that are valid Morse characters.

    Valid Morse characters: . - / |

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    float
        Density in [0.0, 1.0]. Returns 0.0 for empty or whitespace-only text.
    """
    non_ws_count = 0
    morse_count = 0
    for ch in text:
        if ch.isspace():
            continue
        non_ws_count += 1
        if ch in _MORSE_CHARS:
            morse_count += 1
    if non_ws_count == 0:
        return 0.0
    return morse_count / non_ws_count


def _is_false_positive(text):
    """Check if text matches known false positive patterns.

    Parameters
    ----------
    text : str
        Input text to check.

    Returns
    -------
    bool
        True if text should be exempted from Morse detection.
    """
    stripped = text.strip()

    # Horizontal rule: line is only dashes
    if _HORIZONTAL_RULE_RE.match(stripped):
        return True

    # Markdown header
    if _MARKDOWN_HEADER_RE.match(stripped):
        return True

    # IP addresses: if the text is primarily an IP address
    if _IP_ADDR_RE.match(stripped):
        return True

    # Ellipsis in prose: if dots are part of ellipsis patterns (word...word)
    # and the text has substantial alphabetic content
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count > len(stripped) * 0.3 and _ELLIPSIS_RE.search(stripped):
        return True

    return False


def _count_non_whitespace(text):
    """Count non-whitespace characters in text."""
    return sum(1 for ch in text if not ch.isspace())


def detect_morse(text):
    """Detect and decode Morse code in text.

    Detection logic:
      1. Normalize Unicode dot/dash alternatives
      2. Check for explicit Morse labels ("Morse:", "morse code:", etc.)
      3. Calculate Morse density -- if >= 0.80, attempt decode
      4. Decode and validate: decoded text must have >= 3 printable chars
      5. Final confidence = density * decode_success_factor

    FP mitigation:
      - Exempt markdown headers, horizontal rules, ellipsis, IP addresses
      - Require minimum 10 non-whitespace chars of Morse content

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    MorseResult
        Detection result with decoded text and confidence.
    """
    if not isinstance(text, str) or not text:
        return MorseResult()

    # Truncate oversized inputs
    scan_text = text[:_MAX_SCAN_LENGTH] if len(text) > _MAX_SCAN_LENGTH else text

    # Normalize Unicode dot/dash variants
    normalized = normalize_morse(scan_text)

    # --- Check for explicit Morse label ---
    label_match = _MORSE_LABEL_RE.search(normalized)
    if label_match:
        payload = label_match.group(1).strip()
        if payload:
            decoded = decode_morse(payload)
            printable_count = sum(1 for c in decoded if c.isprintable())
            if decoded and printable_count >= _MIN_DECODED_CHARS:
                density = morse_density(payload)
                return MorseResult(
                    detected=True,
                    decoded_text=decoded,
                    confidence=max(0.85, density),
                    density=density,
                )

    # --- FP checks (line-by-line for multi-line text) ---
    # For single-line text, check the whole thing.
    # For multi-line, process line by line and accumulate Morse lines.
    lines = normalized.split('\n')
    morse_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_false_positive(stripped):
            continue
        morse_lines.append(stripped)

    if not morse_lines:
        return MorseResult()

    # Rejoin the non-FP lines for analysis
    morse_content = ' / '.join(morse_lines) if len(morse_lines) > 1 else morse_lines[0]

    # --- Minimum length check ---
    non_ws = _count_non_whitespace(morse_content)
    if non_ws < _MIN_MORSE_LENGTH:
        return MorseResult()

    # --- Morse density check ---
    density = morse_density(morse_content)
    if density < _DENSITY_THRESHOLD:
        return MorseResult(density=density)

    # --- Attempt decode ---
    decoded = decode_morse(morse_content)
    printable_count = sum(1 for c in decoded if c.isprintable())

    if not decoded or printable_count < _MIN_DECODED_CHARS:
        return MorseResult(density=density)

    # --- Confidence calculation ---
    # Base confidence from density, boosted by successful decode
    decode_ratio = printable_count / max(len(decoded), 1)
    confidence = density * decode_ratio
    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))

    return MorseResult(
        detected=True,
        decoded_text=decoded,
        confidence=round(confidence, 4),
        density=round(density, 4),
    )
