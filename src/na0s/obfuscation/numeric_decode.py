"""Layer 2 numeric ASCII decoding -- detect binary/octal/decimal encoded text.

Detects three numeric encoding schemes used to obfuscate prompt injection payloads:

  1. **Binary** (8-bit groups): groups of 7-8 binary digits (01001001 01100111...)
  2. **Octal** (3-digit groups): groups of 1-3 octal digits (111 147 156 157...)
  3. **Decimal** (1-3 digit groups): groups of decimal digits in ASCII range (73 71 78...)

Detection approach:
  1. Check for explicit labels ("binary:", "octal:", "decimal ASCII:", etc.)
  2. Try binary detection first (most specific pattern -- 7/8-bit groups of 0/1)
  3. Try octal detection (1-3 digit groups in 0-177 range)
  4. Try decimal detection (1-3 digit groups in 32-126 range)
  5. Return first successful detection

FP mitigation:
  - Minimum 5 groups required for all three types
  - All decoded chars must be printable ASCII (32-126) or common whitespace
  - At least 70% of decoded chars must be printable
  - Skip text that is predominantly alphabetic (> 50% alpha chars)
  - Exempt IP address patterns from decimal detection
  - Exempt Unix permission patterns from octal detection

No external dependencies -- stdlib only.

Public API:
    detect_numeric(text)  -> NumericDecodeResult  (tries all three, returns first)
    detect_binary(text)   -> NumericDecodeResult
    detect_octal(text)    -> NumericDecodeResult
    detect_decimal(text)  -> NumericDecodeResult
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NumericDecodeResult:
    """Result of numeric ASCII detection analysis."""
    detected: bool = False
    decoded_text: str = ""
    encoding_type: str = ""   # "binary", "octal", "decimal"
    confidence: float = 0.0
    group_count: int = 0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of numeric groups required for detection
_MIN_GROUPS = 5

# Minimum fraction of decoded chars that must be printable
_MIN_PRINTABLE_RATIO = 0.70

# Maximum input length for scanning (same as morse_code.py)
_MAX_SCAN_LENGTH = 100_000

# Common Unix permission patterns (false positive exemption for octal)
_UNIX_PERMISSIONS = frozenset({
    '777', '755', '750', '700', '644', '640', '600',
    '444', '400', '666', '775', '770', '664', '660',
    '0777', '0755', '0750', '0700', '0644', '0640', '0600',
    '0444', '0400', '0666', '0775', '0770', '0664', '0660',
})


# ---------------------------------------------------------------------------
# Explicit label patterns
# ---------------------------------------------------------------------------

_BINARY_LABEL_RE = re.compile(
    r'(?:binary|in\s+binary)\s*[:;=\-]\s*(.+)',
    re.IGNORECASE | re.DOTALL,
)

_OCTAL_LABEL_RE = re.compile(
    r'(?:octal|in\s+octal)\s*[:;=\-]\s*(.+)',
    re.IGNORECASE | re.DOTALL,
)

_DECIMAL_LABEL_RE = re.compile(
    r'(?:decimal(?:\s+ascii)?|ascii\s+(?:codes?|values?))\s*[:;=\-]\s*(.+)',
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Group extraction patterns
# ---------------------------------------------------------------------------

# Binary: 7 or 8 binary digits
_BINARY_GROUP_RE = re.compile(r'[01]{7,8}')

# Octal: 1-3 octal digits (standalone, not part of larger number).
# NOTE: Intentionally loose -- valid ASCII octal values only span 0-177
# (decimal 0-127), but the regex also matches e.g. "200"-"377" (128-255).
# Out-of-range values are rejected during the decode loop in detect_octal()
# via per-character range validation, mirroring detect_decimal()'s approach.
_OCTAL_GROUP_RE = re.compile(r'[0-7]{1,3}')

# Decimal: 1-3 decimal digits (standalone)
_DECIMAL_GROUP_RE = re.compile(r'\d{1,3}')

# IP address pattern (false positive exemption for decimal)
_IP_ADDR_RE = re.compile(
    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
)

# Version number pattern: x.y.z (false positive exemption for decimal)
_VERSION_RE = re.compile(
    r'\b\d{1,3}\.\d{1,3}(?:\.\d{1,3}){0,2}\b'
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_predominantly_alpha(text):
    """Return True if > 50% of non-whitespace chars are alphabetic.

    This filters out normal English text that happens to contain some numbers.
    """
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return False
    alpha_count = sum(1 for c in non_ws if c.isalpha())
    return alpha_count / len(non_ws) > 0.50


def _is_printable_ascii(ch):
    """Return True if char is printable ASCII (32-126) or common whitespace."""
    code = ord(ch)
    return (32 <= code <= 126) or ch in ('\n', '\r', '\t')


def _validate_decoded(decoded_text):
    """Validate that decoded text is mostly printable ASCII.

    Returns True if at least 70% of chars are printable.
    """
    if not decoded_text:
        return False
    total = len(decoded_text)
    printable = sum(1 for c in decoded_text if _is_printable_ascii(c))
    return printable / total >= _MIN_PRINTABLE_RATIO


def _extract_groups(text, pattern):
    """Extract numeric groups from text using the given regex pattern.

    Splits text on common separators (space, comma, pipe, dash, newline)
    and returns groups that match the pattern.
    """
    # Split on common separators
    tokens = re.split(r'[\s,|\n]+', text.strip())
    groups = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if pattern.fullmatch(token):
            groups.append(token)
    return groups


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------

def detect_binary(text):
    """Detect and decode binary-encoded ASCII text.

    Looks for groups of 7-8 binary digits (0s and 1s) separated by
    common separators. Each group is decoded as chr(int(group, 2)).

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    NumericDecodeResult
        Detection result with decoded text and confidence.
    """
    if not isinstance(text, str) or not text:
        return NumericDecodeResult()

    scan_text = text[:_MAX_SCAN_LENGTH] if len(text) > _MAX_SCAN_LENGTH else text

    # Check for explicit binary label
    payload = None
    label_match = _BINARY_LABEL_RE.search(scan_text)
    if label_match:
        payload = label_match.group(1).strip()

    target = payload if payload else scan_text

    # Skip predominantly alphabetic text (unless labeled)
    if not payload and _is_predominantly_alpha(target):
        return NumericDecodeResult()

    # Extract binary groups (7 or 8 bits)
    groups = _extract_groups(target, _BINARY_GROUP_RE)

    if len(groups) < _MIN_GROUPS:
        return NumericDecodeResult()

    # Decode each group
    decoded_chars = []
    for group in groups:
        try:
            value = int(group, 2)
            ch = chr(value)
            decoded_chars.append(ch)
        except (ValueError, OverflowError):
            return NumericDecodeResult()

    decoded_text = ''.join(decoded_chars)

    if not _validate_decoded(decoded_text):
        return NumericDecodeResult()

    # Confidence: higher with explicit label, based on group count
    confidence = 0.90 if payload else min(0.85, 0.50 + len(groups) * 0.05)

    return NumericDecodeResult(
        detected=True,
        decoded_text=decoded_text,
        encoding_type="binary",
        confidence=round(confidence, 4),
        group_count=len(groups),
    )


# ---------------------------------------------------------------------------
# Octal detection
# ---------------------------------------------------------------------------

def detect_octal(text):
    """Detect and decode octal-encoded ASCII text.

    Looks for groups of 1-3 octal digits separated by common separators.
    Each group is decoded as chr(int(group, 8)).

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    NumericDecodeResult
        Detection result with decoded text and confidence.
    """
    if not isinstance(text, str) or not text:
        return NumericDecodeResult()

    scan_text = text[:_MAX_SCAN_LENGTH] if len(text) > _MAX_SCAN_LENGTH else text

    # Check for explicit octal label
    payload = None
    label_match = _OCTAL_LABEL_RE.search(scan_text)
    if label_match:
        payload = label_match.group(1).strip()

    target = payload if payload else scan_text

    # Skip predominantly alphabetic text (unless labeled)
    if not payload and _is_predominantly_alpha(target):
        return NumericDecodeResult()

    # Extract octal groups
    groups = _extract_groups(target, _OCTAL_GROUP_RE)

    if len(groups) < _MIN_GROUPS:
        return NumericDecodeResult()

    # FP: check if all groups are Unix permission patterns
    if not payload:
        perm_count = sum(1 for g in groups if g in _UNIX_PERMISSIONS)
        if perm_count == len(groups):
            return NumericDecodeResult()

    # Decode each group -- must be in printable ASCII range (32-126)
    # or common whitespace (9=tab, 10=newline, 13=CR).
    # Valid ASCII octal values are 0-177 (decimal 0-127), but the regex
    # also matches higher values (200-377 = decimal 128-255).  We
    # replicate detect_decimal()'s approach: accept printable/whitespace,
    # record '?' for anything else, and require a 70% valid ratio.
    decoded_chars = []
    valid_count = 0
    for group in groups:
        try:
            value = int(group, 8)
            if 32 <= value <= 126:
                decoded_chars.append(chr(value))
                valid_count += 1
            elif value in (9, 10, 13):  # tab, newline, CR
                decoded_chars.append(chr(value))
                valid_count += 1
            else:
                # Value outside printable ASCII range (e.g. octal 200+ = 128+)
                decoded_chars.append('?')
        except (ValueError, OverflowError):
            return NumericDecodeResult()

    # Require at least 70% of groups to decode to valid ASCII
    if len(groups) > 0 and valid_count / len(groups) < _MIN_PRINTABLE_RATIO:
        return NumericDecodeResult()

    decoded_text = ''.join(decoded_chars)

    if not _validate_decoded(decoded_text):
        return NumericDecodeResult()

    # Confidence: higher with explicit label
    confidence = 0.90 if payload else min(0.80, 0.45 + len(groups) * 0.05)

    return NumericDecodeResult(
        detected=True,
        decoded_text=decoded_text,
        encoding_type="octal",
        confidence=round(confidence, 4),
        group_count=len(groups),
    )


# ---------------------------------------------------------------------------
# Decimal detection
# ---------------------------------------------------------------------------

def detect_decimal(text):
    """Detect and decode decimal-encoded ASCII text.

    Looks for groups of 1-3 decimal digits in the printable ASCII range
    (32-126) separated by common separators.

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    NumericDecodeResult
        Detection result with decoded text and confidence.
    """
    if not isinstance(text, str) or not text:
        return NumericDecodeResult()

    scan_text = text[:_MAX_SCAN_LENGTH] if len(text) > _MAX_SCAN_LENGTH else text

    # Check for explicit decimal label
    payload = None
    label_match = _DECIMAL_LABEL_RE.search(scan_text)
    if label_match:
        payload = label_match.group(1).strip()

    target = payload if payload else scan_text

    # Skip predominantly alphabetic text (unless labeled)
    if not payload and _is_predominantly_alpha(target):
        return NumericDecodeResult()

    # FP: skip if text looks like IP addresses
    if not payload and _IP_ADDR_RE.search(target):
        # If the IP pattern covers most of the text, skip
        ip_matches = _IP_ADDR_RE.findall(target)
        ip_text_len = sum(len(m) for m in ip_matches)
        non_ws_len = sum(1 for c in target if not c.isspace())
        if non_ws_len > 0 and ip_text_len / non_ws_len > 0.5:
            return NumericDecodeResult()

    # FP: skip if text looks like version numbers
    if not payload and _VERSION_RE.search(target):
        version_matches = _VERSION_RE.findall(target)
        ver_text_len = sum(len(m) for m in version_matches)
        non_ws_len = sum(1 for c in target if not c.isspace())
        if non_ws_len > 0 and ver_text_len / non_ws_len > 0.5:
            return NumericDecodeResult()

    # Extract decimal groups
    groups = _extract_groups(target, _DECIMAL_GROUP_RE)

    if len(groups) < _MIN_GROUPS:
        return NumericDecodeResult()

    # Decode each group -- must be in printable ASCII range (32-126)
    # or common whitespace (9=tab, 10=newline, 13=CR)
    decoded_chars = []
    valid_count = 0
    for group in groups:
        try:
            value = int(group)
            if 32 <= value <= 126:
                decoded_chars.append(chr(value))
                valid_count += 1
            elif value in (9, 10, 13):
                decoded_chars.append(chr(value))
                valid_count += 1
            else:
                # Value outside printable range -- not a valid ASCII encoding
                # Allow a few out-of-range values, but track them
                decoded_chars.append('?')
        except (ValueError, OverflowError):
            return NumericDecodeResult()

    # Require at least 70% of groups to decode to valid ASCII
    if len(groups) > 0 and valid_count / len(groups) < _MIN_PRINTABLE_RATIO:
        return NumericDecodeResult()

    decoded_text = ''.join(decoded_chars)

    if not _validate_decoded(decoded_text):
        return NumericDecodeResult()

    # Confidence: higher with explicit label
    confidence = 0.90 if payload else min(0.80, 0.45 + len(groups) * 0.05)

    return NumericDecodeResult(
        detected=True,
        decoded_text=decoded_text,
        encoding_type="decimal",
        confidence=round(confidence, 4),
        group_count=len(groups),
    )


# ---------------------------------------------------------------------------
# Combined detection
# ---------------------------------------------------------------------------

def _decoded_quality(decoded_text):
    """Score decoded text quality: higher = more likely to be real text.

    Returns the fraction of non-whitespace characters that are alphabetic.
    Real encoded messages tend to be mostly alphabetic, while garbage
    decodes have lots of symbols/control characters.
    """
    if not decoded_text:
        return 0.0
    non_ws = [c for c in decoded_text if not c.isspace()]
    if not non_ws:
        return 0.0
    alpha_count = sum(1 for c in non_ws if c.isalpha())
    return alpha_count / len(non_ws)


def detect_numeric(text):
    """Detect and decode numeric-encoded ASCII text.

    Tries all three encoding schemes and returns the best result based
    on decoded text quality (alphabetic ratio).  Binary is given priority
    when its quality ties with others because it is the most specific
    pattern (7/8-bit groups of only 0/1).

    If explicit labels are found ("binary:", "octal:", "decimal ASCII:"),
    the labeled type is returned immediately without quality comparison.

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    NumericDecodeResult
        Detection result with decoded text, encoding type, and confidence.
    """
    if not isinstance(text, str) or not text:
        return NumericDecodeResult()

    # Try binary first (most specific pattern) -- return immediately
    # because binary groups (7-8 bits of 0/1) are unambiguous
    binary_result = detect_binary(text)
    if binary_result.detected:
        return binary_result

    # Try both octal and decimal, then pick the one with higher quality
    # decoded text.  This handles the ambiguous case where groups like
    # "73 71 78" are valid as both octal and decimal but decode to
    # different characters.
    octal_result = detect_octal(text)
    decimal_result = detect_decimal(text)

    candidates = []
    if octal_result.detected:
        candidates.append(octal_result)
    if decimal_result.detected:
        candidates.append(decimal_result)

    if not candidates:
        return NumericDecodeResult()

    if len(candidates) == 1:
        return candidates[0]

    # Both matched -- pick the one with the highest decoded text quality
    best = max(candidates, key=lambda r: _decoded_quality(r.decoded_text))
    return best
