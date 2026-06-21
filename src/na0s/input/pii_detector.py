"""PII and secrets pre-screening for Layer 0.

Pure-regex detection of personally identifiable information (PII) and
secrets (API keys, tokens) in text input.  Designed to flag data
exfiltration attempts where an attacker tries to extract PII/secrets
from the LLM context via prompt injection.

No external dependencies -- stdlib only.  All regexes are pre-compiled
at module level for performance.

IMPORTANT: This module NEVER logs or stores actual PII values.
All values in results are redacted (first 4 chars + "***").
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclass for scan results
# ---------------------------------------------------------------------------

@dataclass
class PiiScanResult:
    """Result of a PII/secrets scan."""
    has_pii: bool = False
    pii_types_found: list = field(default_factory=list)
    pii_count: int = 0
    anomaly_flags: set = field(default_factory=set)
    details: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Luhn checksum validation (for credit card numbers)
# ---------------------------------------------------------------------------

def _luhn_check(number_str):
    """Validate a number string using the Luhn algorithm."""
    digits = [int(d) for d in number_str]
    digits.reverse()
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Redaction helper -- NEVER expose full PII values
# ---------------------------------------------------------------------------

def _redact(value):
    """Redact a PII value, showing at most 25% of the original.

    - 1-3 chars  → show 1 char  + "***"
    - 4-7 chars  → show 2 chars + "***"
    - 8+ chars   → show 4 chars + "***"
    """
    length = len(value)
    if length <= 3:
        return value[:1] + "***"
    if length <= 7:
        return value[:2] + "***"
    return value[:4] + "***"


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

_CREDIT_CARD_RE = re.compile(
    r"\b("
    r"4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"       # Visa 16
    r"|5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"   # MC 51-55
    r"|(?:222[1-9]|22[3-9]\d|2[3-6]\d{2}|27[01]\d|2720)[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"  # MC 2221-2720
    r"|3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}"                # Amex
    r"|6011[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"           # Discover 6011
    r"|65\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"        # Discover 65
    r")\b"
)

_SSN_RE = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
)

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:(?:\+1[\s.-]?)?"
    r"(?:\([2-9]\d{2}\)[\s.-]?\d{3}[\s.-]?\d{4}"
    r"|[2-9]\d{2}[\s.-]?\d{3}[\s.-]?\d{4}))"
    r"(?!\d)"
)

_AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")

_GITHUB_TOKEN_RE = re.compile(r"\bgh[ps]_[A-Za-z0-9_]{36,255}\b")

_GENERIC_HEX_RE = re.compile(r"\b[0-9a-fA-F]{40,}\b")

_GENERIC_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")

_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# Private / loopback / link-local ranges — not exfiltration targets.
_IPV4_PRIVATE_RE = re.compile(
    r"^(?:"
    r"0\.0\.0\.0"                       # Unspecified
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3}" # Loopback
    r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"  # RFC 1918 Class A
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"  # RFC 1918 Class B
    r"|192\.168\.\d{1,3}\.\d{1,3}"     # RFC 1918 Class C
    r"|169\.254\.\d{1,3}\.\d{1,3}"     # Link-local
    r")$"
)

# Maximum input length for PII scanning (100 KB).
_MAX_SCAN_LENGTH = 100_000


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    ("credit_card", _CREDIT_CARD_RE, "credit_card", "pii_credit_card"),
    ("ssn", _SSN_RE, "ssn", "pii_ssn"),
    ("email", _EMAIL_RE, "email", "pii_email"),
    ("phone", _PHONE_RE, "phone", "pii_phone"),
    ("aws_key", _AWS_KEY_RE, "api_key", "pii_api_key"),
    ("github_token", _GITHUB_TOKEN_RE, "api_key", "pii_api_key"),
    ("generic_hex", _GENERIC_HEX_RE, "api_key", "pii_api_key"),
    ("generic_base64", _GENERIC_BASE64_RE, "api_key", "pii_api_key"),
    ("ipv4", _IPV4_RE, "ipv4", "pii_ipv4"),
]


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------

def scan_pii(text):
    """Scan text for PII and secrets.

    Parameters
    ----------
    text : str
        The text to scan (should be post-normalization sanitized text).

    Returns
    -------
    PiiScanResult
        Scan results with detected PII types, counts, and redacted details.
        Actual PII values are NEVER stored -- only redacted versions.
    """
    if not text:
        return PiiScanResult()

    # Truncate oversized inputs to prevent regex resource exhaustion.
    if len(text) > _MAX_SCAN_LENGTH:
        text = text[:_MAX_SCAN_LENGTH]

    types_found = []
    flags = set()
    details = []
    count = 0

    for name, pattern, pii_type, flag_name in _PII_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue

        for match in matches:
            raw_value = match.group()

            # Credit card: strip separators and validate Luhn
            if name == "credit_card":
                digits_only = re.sub(r"[\s-]", "", raw_value)
                if not _luhn_check(digits_only):
                    continue

            # Generic hex/base64: require minimum diversity
            if name == "generic_hex":
                unique_chars = len(set(raw_value.lower()))
                if unique_chars < 6:
                    continue
            if name == "generic_base64":
                unique_chars = len(set(raw_value.lower()))
                if unique_chars < 10:
                    continue
                # Require at least one digit or +/= to avoid English words/paths
                if not re.search(r"[0-9+=]", raw_value):
                    continue

            # IPv4: skip private, loopback, and link-local addresses
            if name == "ipv4" and _IPV4_PRIVATE_RE.match(raw_value):
                continue

            details.append({
                "type": pii_type,
                "subtype": name,
                "redacted_value": _redact(raw_value),
                "position": match.start(),
            })
            flags.add(flag_name)
            if pii_type not in types_found:
                types_found.append(pii_type)
            count += 1

    return PiiScanResult(
        has_pii=count > 0,
        pii_types_found=types_found,
        pii_count=count,
        anomaly_flags=flags,
        details=details,
    )
