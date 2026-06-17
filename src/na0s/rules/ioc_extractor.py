"""Layer 1 IOC extractor -- detect and refang defanged IOCs.

Handles defanged URLs, IPs, emails, domains, and file hashes commonly seen
in prompt injection attempts that use security analyst notation to evade
Layer 0's PII detector.

Key distinction from L0's pii_detector.py:
  - L0 catches *standard* PII (raw emails, raw IPs, etc.)
  - L1 IOC extractor catches *defanged* indicators that analysts use in
    threat reports (hXXp://, [.], [@], etc.).  In the context of prompt
    injection detection, presence of defanged IOCs is suspicious because
    normal users don't use this notation -- attackers use it to evade
    URL/IP blocking rules.

Defanging conventions handled:
  [.]  (.)  {.}  DOT  (dot)  [dot]   -> .
  [:]  (:)                            -> :
  hXXp hxxp HXXP hXXps hxxps         -> http / https
  [@]  (@)  [at]  (at)  AT           -> @
  [://] (://)                         -> ://

No external dependencies -- stdlib only.
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IocResult:
    """Result of IOC extraction."""
    has_iocs: bool = False
    ioc_count: int = 0
    ioc_types_found: list = field(default_factory=list)
    refanged_text: str = ""
    details: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-compiled defanging patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Protocol defanging: hXXp:// hxxps:// hXXps:// etc.
_DEFANG_PROTO_PATTERNS = [
    # http[:]// or https[:]// — must run BEFORE hxxp to handle hxxps[:]//
    (re.compile(r"\bhttps?\[:\]//", re.IGNORECASE),
     lambda m: m.group().lower().replace("[:]", ":")),
    # http[s]://
    (re.compile(r"\bhttp\[s\]://", re.IGNORECASE),
     lambda m: "https://"),
    # http(://)
    (re.compile(r"\bhttp\(://\)", re.IGNORECASE),
     lambda m: "http://"),
    # ftp[:]//
    (re.compile(r"\bftp\[:\]//", re.IGNORECASE),
     lambda m: "ftp://"),
    # hXXp:// hxxp:// HXXP:// — runs LAST so hxxps[:]// is handled above first
    (re.compile(r"\bhxxp(s?)://", re.IGNORECASE),
     lambda m: "http" + m.group(1).lower() + "://"),
]

# Dot defanging: [.] (.) {.} [dot] (dot) " DOT "
_DEFANG_DOT_PATTERNS = [
    (re.compile(r"\[\.\]"), "."),
    (re.compile(r"\(\.\)"), "."),
    (re.compile(r"\{\.\}"), "."),
    (re.compile(r"\[dot\]", re.IGNORECASE), "."),
    (re.compile(r"\(dot\)", re.IGNORECASE), "."),
    # " DOT " with surrounding whitespace -- require uppercase or mixed with spaces
    (re.compile(r"(?<=\w)\s+DOT\s+(?=\w)"), "."),
]

# At-sign defanging: [@] (@) [at] (at) " AT "
_DEFANG_AT_PATTERNS = [
    (re.compile(r"\[@\]"), "@"),
    (re.compile(r"\(@\)"), "@"),
    (re.compile(r"\[at\]", re.IGNORECASE), "@"),
    (re.compile(r"\(at\)", re.IGNORECASE), "@"),
    # " AT " with surrounding word chars
    (re.compile(r"(?<=\w)\s+AT\s+(?=\w)"), "@"),
]

# Colon defanging: [:]  (:)
_DEFANG_COLON_PATTERNS = [
    (re.compile(r"\[:\]"), ":"),
    (re.compile(r"\(:\)"), ":"),
]

# Slash defanging: [/] (/) [://] (://)
_DEFANG_SLASH_PATTERNS = [
    (re.compile(r"\[://\]"), "://"),
    (re.compile(r"\(://\)"), "://"),
    (re.compile(r"\[/\]"), "/"),
    (re.compile(r"\(/\)"), "/"),
]


# ---------------------------------------------------------------------------
# Detection patterns (for identifying defanged IOCs in text)
# ---------------------------------------------------------------------------

# Defanged URL: starts with defanged protocol followed by at least 4 non-space chars
_DEFANGED_URL_RE = re.compile(
    r"\b(?:hxxps?://|https?\[:\]//|http\[s\]://|http\(://\)|ftp\[:\]//)"
    r"[^\s]{4,}",
    re.IGNORECASE,
)

# Defanged IP: at least one octet separator uses [.] or (.) or {.} or [dot]
_DEFANGED_IP_RE = re.compile(
    r"\b\d{1,3}"
    r"(?:\[\.\]|\(\.\)|\{\.\}|\[dot\])"
    r"\d{1,3}"
    r"(?:\[\.\]|\(\.\)|\{\.\}|\[dot\]|\.)"
    r"\d{1,3}"
    r"(?:\[\.\]|\(\.\)|\{\.\}|\[dot\]|\.)"
    r"\d{1,3}\b",
    re.IGNORECASE,
)

# Defanged domain: word chars followed by at least one defanged dot + TLD
_DEFANGED_DOMAIN_RE = re.compile(
    r"\b[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\[\.\]|\(\.\)|\{\.\}|\[dot\])"
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\[\.\]|\(\.\)|\{\.\}|\[dot\]|\.))?"
    r"[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Defanged email: user[@]domain[.]tld or user[at]domain[dot]tld
_DEFANGED_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+"
    r"(?:\[@\]|\(@\)|\[at\]|\(at\))"
    r"[A-Za-z0-9.-]+"
    r"(?:\[\.\]|\(\.\)|\{\.\}|\[dot\])"
    r"[A-Za-z]{2,}\b",
    re.IGNORECASE,
)

# File hashes: MD5 (32 hex), SHA1 (40 hex), SHA256 (64 hex)
_MD5_RE = re.compile(r"\b[0-9a-fA-F]{32}\b")
_SHA1_RE = re.compile(r"\b[0-9a-fA-F]{40}\b")
_SHA256_RE = re.compile(r"\b[0-9a-fA-F]{64}\b")


# ---------------------------------------------------------------------------
# Maximum input length for IOC scanning (100 KB).
# ---------------------------------------------------------------------------
_MAX_SCAN_LENGTH = 100_000


# ---------------------------------------------------------------------------
# Diversity check for hex strings
# ---------------------------------------------------------------------------

def _hex_has_diversity(hex_str, min_unique=8):
    """Check that a hex string has enough character diversity.

    Prevents flagging repetitive hex like "aaaa...aaaa" or common padding
    as file hashes.  Real hashes have high entropy and many unique chars.
    """
    return len(set(hex_str.lower())) >= min_unique


# ---------------------------------------------------------------------------
# Redaction helper -- never expose full IOC values in results
# ---------------------------------------------------------------------------

def _redact_ioc(value, max_show=12):
    """Show truncated IOC value for result details."""
    if len(value) <= max_show:
        return value
    return value[:max_show] + "..."


# ---------------------------------------------------------------------------
# Refanging function
# ---------------------------------------------------------------------------

def refang(text):
    """Restore defanged IOCs to their standard form.

    Converts common defanging conventions back to live indicators:
      hXXp://  -> http://
      [.]      -> .
      [@]      -> @
      [:]      -> :
      etc.

    Parameters
    ----------
    text : str
        Input text that may contain defanged IOCs.

    Returns
    -------
    str
        Text with all defanged IOCs restored to standard form.
    """
    if not text:
        return text

    result = text

    # Apply colon/slash refanging BEFORE protocol patterns so that
    # hxxps[:]// becomes hxxps:// first, then hxxp->http conversion works.
    for pattern, replacement in _DEFANG_COLON_PATTERNS:
        result = pattern.sub(replacement, result)

    for pattern, replacement in _DEFANG_SLASH_PATTERNS:
        result = pattern.sub(replacement, result)

    # Apply protocol refanging (now [:] and (://) are already normalized)
    for pattern, replacement in _DEFANG_PROTO_PATTERNS:
        result = pattern.sub(replacement, result)

    # Apply dot refanging
    for pattern, replacement in _DEFANG_DOT_PATTERNS:
        result = pattern.sub(replacement, result)

    # Apply at-sign refanging
    for pattern, replacement in _DEFANG_AT_PATTERNS:
        result = pattern.sub(replacement, result)

    return result


# ---------------------------------------------------------------------------
# IOC extraction function
# ---------------------------------------------------------------------------

def extract_iocs(text):
    """Extract and classify defanged IOCs from text.

    Scans for defanged URLs, IPs, domains, emails, and file hashes.
    Returns an IocResult with detection details and refanged text.

    IMPORTANT: This function targets DEFANGED IOCs only.  Standard
    (non-defanged) IPs, emails, and URLs are handled by Layer 0's
    pii_detector.py to avoid duplicate detection.

    Parameters
    ----------
    text : str
        Input text to scan for defanged IOCs.

    Returns
    -------
    IocResult
        Extraction results with IOC types, counts, and refanged text.
    """
    if not text:
        return IocResult(refanged_text="")

    # Truncate oversized inputs
    scan_text = text[:_MAX_SCAN_LENGTH] if len(text) > _MAX_SCAN_LENGTH else text

    details = []
    types_found = []

    def _add_type(type_name):
        if type_name not in types_found:
            types_found.append(type_name)

    # --- Defanged URLs ---
    for m in _DEFANGED_URL_RE.finditer(scan_text):
        details.append({
            "type": "defanged_url",
            "value": _redact_ioc(m.group()),
            "position": m.start(),
        })
        _add_type("defanged_url")

    # --- Defanged IPs ---
    for m in _DEFANGED_IP_RE.finditer(scan_text):
        val = m.group()
        # Only flag IPs with actual defanging markers (brackets/parens)
        if "[" in val or "(" in val or "{" in val:
            details.append({
                "type": "defanged_ip",
                "value": _redact_ioc(val),
                "position": m.start(),
            })
            _add_type("defanged_ip")

    # --- Defanged domains (not caught by URL pattern) ---
    for m in _DEFANGED_DOMAIN_RE.finditer(scan_text):
        val = m.group()
        # Skip if already captured as part of a URL
        pos = m.start()
        already_captured = any(
            d["type"] == "defanged_url"
            and d["position"] <= pos
            and pos < d["position"] + 60
            for d in details
        )
        if not already_captured:
            details.append({
                "type": "defanged_domain",
                "value": _redact_ioc(val),
                "position": pos,
            })
            _add_type("defanged_domain")

    # --- Defanged emails ---
    for m in _DEFANGED_EMAIL_RE.finditer(scan_text):
        details.append({
            "type": "defanged_email",
            "value": _redact_ioc(m.group()),
            "position": m.start(),
        })
        _add_type("defanged_email")

    # --- File hashes (SHA-256 only for now -- most distinctive) ---
    for m in _SHA256_RE.finditer(scan_text):
        if _hex_has_diversity(m.group(), min_unique=8):
            details.append({
                "type": "sha256_hash",
                "value": m.group()[:8] + "...",
                "position": m.start(),
            })
            _add_type("file_hash")

    # --- SHA-1 hashes ---
    for m in _SHA1_RE.finditer(scan_text):
        val = m.group()
        # Skip if this is a substring of a longer hash already detected
        already_captured = any(
            d["type"] == "sha256_hash"
            and abs(d["position"] - m.start()) < 5
            for d in details
        )
        if not already_captured and _hex_has_diversity(val, min_unique=7):
            details.append({
                "type": "sha1_hash",
                "value": val[:8] + "...",
                "position": m.start(),
            })
            _add_type("file_hash")

    # --- MD5 hashes ---
    for m in _MD5_RE.finditer(scan_text):
        val = m.group()
        # Skip if substring of a longer hash
        already_captured = any(
            d["type"] in ("sha256_hash", "sha1_hash")
            and abs(d["position"] - m.start()) < 5
            for d in details
        )
        if not already_captured and _hex_has_diversity(val, min_unique=6):
            details.append({
                "type": "md5_hash",
                "value": val[:8] + "...",
                "position": m.start(),
            })
            _add_type("file_hash")

    refanged_text = refang(scan_text)

    return IocResult(
        has_iocs=len(details) > 0,
        ioc_count=len(details),
        ioc_types_found=types_found,
        refanged_text=refanged_text,
        details=details,
    )
