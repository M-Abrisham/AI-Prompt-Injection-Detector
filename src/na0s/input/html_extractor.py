"""HTML text extraction and content-type sniffing for prompt-injection defence.

Threat model
------------
Attackers can embed prompt-injection instructions inside HTML payloads to
bypass surface-level string matching:

* **Hidden content** – ``display:none``, ``opacity:0``, or ``font-size:0``
  CSS that hides text from human readers but is visible to an LLM.
* **Script / style injection** – ``<script>`` or ``<style>`` blocks whose
  text content carries injected instructions.
* **Comment injection** – HTML comments containing override phrases such as
  "ignore previous instructions".
* **Malformed-HTML desync** – deliberately mismatched or surplus close tags
  intended to desync a naïve depth counter and leak hidden content.

Public API
----------
``extract_safe_text(text)`` → :class:`ExtractionResult`
    Strip tags, return visible text and a list of anomaly flags.

``sniff_content_type(text)`` → ``list[str]``
    Return format-detection flags without modifying the input.

Security guarantees
-------------------
This module is a *best-effort* heuristic layer, not a cryptographic control.
Novel obfuscation techniques not covered by the current rule set may evade
detection.  It is designed to raise attacker cost, not to provide absolute
safety guarantees.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import NamedTuple

from .content_type import sniff_binary
from .resource_guard import check_html_depth, ResourceLimitExceeded


class ExtractionResult(NamedTuple):
    """Return value of :func:`extract_safe_text` and :meth:`_TextExtractor.get_result`.

    Supports both attribute access (``result.text``, ``result.flags``) and
    legacy tuple unpacking (``text, flags = extract_safe_text(...)``).
    """

    text: str
    flags: list[str]


# Quick check: does the string contain anything that looks like an HTML tag?
_HTML_TAG_RE = re.compile(r"<[a-zA-Z/!]")

_MAX_INPUT_CHARS: int = 500_000
_MAX_NESTING_DEPTH: int = 200

# --- Magic bytes / content-type sniffing ---
# BOM markers (strip before further checks)
_BOM_PREFIXES: list[tuple[bytes, str]] = [
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
]

# HTML/XML/SVG document signatures (case-insensitive on first bytes)
_HTML_SIGNATURES: list[bytes] = [
    b"<!doctype",
    b"<html",
    b"<head",
    b"<body",
    b"<script",
    b"<iframe",
    b"<svg",
    b"<?xml",
]

# HTML5 void elements — browsers never emit a close tag for these, so
# handle_endtag will never fire.  We must not push them onto the depth
# stack or _skip_depth would permanently over-count.
_VOID_ELEMENTS: frozenset[str] = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})

# Tags whose *text content* must always be suppressed: <script> and
# <style> bodies are never user-visible and can carry injected strings
# that would mislead a downstream LLM.
_SKIP_TAGS: frozenset[str] = frozenset({"script", "style"})

# Hidden-content CSS patterns in inline styles
# opacity: the terminator group uses (?:[;'"]|$) so a style attribute that
# ends with "opacity: 0" (no trailing semicolon) is still caught.
_HIDDEN_STYLE_RE = re.compile(
    r"display\s*:\s*none|"
    r"opacity\s*:\s*0(?:\.0*)?\s*(?:[;'\"]|$)|"
    r"font-size\s*:\s*0",
    re.IGNORECASE,
)

# Suspicious injection PHRASES inside HTML comments.
# Require multi-word patterns — bare "ignore" or "instruction" match too much
# legitimate content (e.g. "ignore older browsers", "see instructions below").
_COMMENT_KEYWORDS_RE = re.compile(
    r"ignore.{0,15}(?:instruction|previous|above|system|prompt)|"
    r"system.{0,10}prompt|"
    r"reveal.{0,10}(?:prompt|secret|password|key|credential)|"
    r"override.{0,10}(?:instruction|rule|policy|filter)|"
    r"bypass.{0,10}(?:filter|safety|security|guard|check)|"
    r"exfiltrate",
    re.IGNORECASE,
)


def sniff_content_type(text: str) -> list[str]:
    """Detect real content type from magic bytes, not file extensions.

    A ".txt" file could contain HTML -- this function inspects the actual
    bytes to determine the true content type.

    Binary format detection (PDF, RTF, images, archives, executables, etc.)
    is delegated to ``content_type.sniff_binary()``.  This function retains
    HTML/SVG signature detection and BOM handling.

    Returns a list of flags. Empty list means plain text.
    """
    flags = []
    # Note: text is a Python str (already decoded).  Re-encoding to UTF-8
    # lets us reuse the bytes-based signature checks below.  UTF-8 BOM
    # detection works because U+FEFF encodes to 0xEF 0xBB 0xBF; UTF-16
    # BOMs (0xFF 0xFE / 0xFE 0xFF) cannot appear in a Python str so those
    # entries in _BOM_PREFIXES will never match (by design).
    raw = text.encode("utf-8", errors="replace")

    # Strip BOM if present
    for bom, encoding in _BOM_PREFIXES:
        if raw.startswith(bom):
            flags.append("bom_detected_{}".format(encoding))
            raw = raw[len(bom):]
            break

    # Check for HTML/XML/SVG signatures
    lower = raw.lstrip()[:50].lower()
    for sig in _HTML_SIGNATURES:
        if lower.startswith(sig):
            flags.append("magic_bytes_html")
            break

    # Delegate binary format detection to content_type module
    binary_flags = sniff_binary(text)
    flags.extend(binary_flags)

    return flags


class _TextExtractor(HTMLParser):
    """Strips HTML tags and collects visible text + hidden-content flags."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._flags: list[str] = []
        self._skip_depth: int = 0  # depth inside hidden/suppressed elements
        self._tag_stack: list[str] = []  # parallel stack for matched close-tag accounting

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Guard: cap tracking depth to prevent unbounded stack growth on
        # adversarially nested HTML.
        if len(self._tag_stack) >= _MAX_NESTING_DEPTH:
            if "nesting_limit_exceeded" not in self._flags:
                self._flags.append("nesting_limit_exceeded")
            return  # content at excessive depth stays suppressed

        # Void elements are self-closing — handle_endtag never fires
        # for them, so we must not push them onto the stack.
        if tag in _VOID_ELEMENTS:
            return

        # Script/style content is never user-visible; always suppress.
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            self._tag_stack.append(tag)
            return

        style = dict(attrs).get("style", "")
        if self._skip_depth > 0:
            # Already inside a hidden element — track nested depth
            self._skip_depth += 1
            self._tag_stack.append(tag)
        elif _HIDDEN_STYLE_RE.search(style):
            self._flags.append("hidden_html_content")
            self._skip_depth += 1
            self._tag_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        # Only decrement when the closing tag matches the most recent
        # opening tag on the stack.  Mismatched / surplus close tags in
        # malformed HTML are ignored so _skip_depth never desyncs.
        if self._skip_depth > 0 and self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(data)

    def handle_comment(self, data: str) -> None:
        if _COMMENT_KEYWORDS_RE.search(data):
            self._flags.append("suspicious_html_comment")

    def get_result(self) -> ExtractionResult:
        text = " ".join(self._pieces)
        # Collapse whitespace left by tag removal
        text = " ".join(text.split())
        return ExtractionResult(text=text, flags=list(set(self._flags)))


def extract_safe_text(text: str) -> ExtractionResult:
    """Extract visible text from input.

    File type is detected by magic bytes, not extensions —
    a ".txt" could contain HTML and will be caught here.

    Returns an :class:`ExtractionResult` with ``text`` and ``flags`` attributes.
    """
    flags = []

    # Guard: truncate pathologically large inputs before any processing.
    if len(text) > _MAX_INPUT_CHARS:
        flags.append("input_size_limit_exceeded")
        text = text[:_MAX_INPUT_CHARS]

    # Step 1: Magic bytes sniffing — detect real content type
    sniff_flags = sniff_content_type(text)
    flags.extend(sniff_flags)

    # Non-HTML binary content detected — flag and return raw text.
    # Any embedded_* category flag means the content is not parseable HTML.
    _BINARY_CATEGORIES = (
        "embedded_document", "embedded_image", "embedded_audio",
        "embedded_video", "embedded_archive", "embedded_executable",
    )
    if any(f in _BINARY_CATEGORIES for f in sniff_flags):
        return ExtractionResult(text=text, flags=flags)

    # Step 2: Decide whether to parse as HTML
    has_html_magic = "magic_bytes_html" in sniff_flags
    has_html_tags = bool(_HTML_TAG_RE.search(text))

    if not has_html_magic and not has_html_tags:
        return ExtractionResult(text=text, flags=flags)

    # Step 2b: HTML depth pre-check — reject deeply nested HTML before
    # feeding it to the parser to prevent stack overflow / excessive
    # recursion.  On violation, flag and return raw text (defense in depth).
    try:
        check_html_depth(text)
    except ResourceLimitExceeded:
        flags.append("html_depth_exceeded")
        return ExtractionResult(text=text, flags=flags)

    # Step 3: Parse HTML and extract visible text
    parser = _TextExtractor()
    try:
        parser.feed(text)
    except Exception:
        flags.append("html_parse_error")
        return ExtractionResult(text=text, flags=flags)

    parsed_text, parse_flags = parser.get_result()
    flags.extend(parse_flags)
    return ExtractionResult(text=parsed_text, flags=flags)
