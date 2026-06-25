"""Shared canary encoded-leak detection.

A single, reusable, FP-safe detector for whether a canary token appears in
some text in *any* of the supported forms (exact, case-insensitive, partial
first-half with word boundary, base64 incl. block-decode, hex incl.
block-decode, reversed, ROT13, unicode-escape, URL-encoded).

This is the ONE source of truth for canary leak detection.  Every wrapper
(``CanaryManager._is_present``, ``RotatingCanaryManager.check_output``,
``SessionCanaryManager.check_session_output``, ``HoneypotManager``) delegates
here so that encoded leaks are caught everywhere identically -- not just by an
exact substring match.

Design notes
------------
* These checks use ``in`` (substring search) rather than
  ``hmac.compare_digest`` because we are scanning untrusted LLM output for any
  *occurrence* of the token.  Timing-safe comparison only applies to
  fixed-length equality checks, not substring search, so ``in`` is the correct
  primitive here.
* Every decoder is wrapped in ``try``/``except`` so malformed/benign input can
  never crash the scanner (fail-open on *decode*, never on detection).
* The partial-match path requires ``len(half) >= 10`` plus a word boundary so
  short or embedded fragments do not produce false positives on benign text.
"""

from __future__ import annotations

import base64
import codecs
import logging
import re
import urllib.parse
from typing import Any, Union

logger = logging.getLogger(__name__)

# A canary may be passed either as a CanaryToken-like object (anything exposing
# a ``.token`` str attribute) or as a raw token string.  We avoid importing
# CanaryToken here to keep this module dependency-free and importable from
# manager.py without a circular import.
CanaryLike = Union[str, Any]

# Minimum length of the partial (first-half) fragment before it is allowed to
# match.  Below this, a half-token is too short to be a reliable leak signal
# and risks colliding with benign text, so we skip the partial path entirely.
_MIN_PARTIAL_LEN = 10

# A base64 block must be at least this many chars before we attempt to decode
# it.  Shorter runs are too likely to be incidental alphanumerics.
_MIN_B64_BLOCK = 16

# A hex block must be at least this many chars before we attempt to decode it.
_MIN_HEX_BLOCK = 20

_B64_CHARSET_RE = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def _has_word_boundary(text: str, substring: str) -> bool:
    """True if *substring* occurs in *text* not flanked by alphanumerics."""
    pattern = r"(?<![A-Za-z0-9])" + re.escape(substring) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, text))


def _coerce_token(canary: CanaryLike) -> str:
    """Accept either a ``CanaryToken`` (anything with a ``.token`` str attr) or a raw token string."""
    tok = getattr(canary, "token", canary)
    return tok


def is_canary_present(canary: CanaryLike, text: str) -> bool:
    """Return True if *canary* appears in *text* in any supported form.

    Parameters
    ----------
    canary:
        A ``CanaryToken`` (or any object exposing a ``.token`` string) or a
        raw token string.
    text:
        The (untrusted) text to scan -- typically LLM output.

    Returns
    -------
    bool
        True if the canary is present (exact or encoded), else False.
        FP-safe: benign text returns False.
    """
    if not text:
        return False

    token = _coerce_token(canary)
    if not token:
        return False

    token_bytes = token.encode("utf-8")
    half = token[: len(token) // 2]

    # 1. Exact match (substring -- see module docstring)
    if token in text:
        return True

    # 2. Case-insensitive match
    if token.lower() in text.lower():
        return True

    # 3. Partial match (first half) -- min length + word boundary
    if len(half) >= _MIN_PARTIAL_LEN and half in text and _has_word_boundary(text, half):
        return True

    # 4. Base64 encoded (direct + block-decode)
    b64 = base64.b64encode(token_bytes).decode("ascii")
    if b64 in text:
        return True
    for b64_block in re.findall(r"[A-Za-z0-9+/]{%d,}={0,2}" % _MIN_B64_BLOCK, text):
        if not _B64_CHARSET_RE.match(b64_block):
            continue
        try:
            decoded = base64.b64decode(b64_block).decode("utf-8")
            if token in decoded or (len(half) >= _MIN_PARTIAL_LEN and half in decoded):
                return True
        except Exception as exc:  # noqa: BLE001 -- malformed block must not crash scan
            logger.debug("base64 decode error for block %r: %s", b64_block[:30], exc)

    # 5. Hex encoded (direct + block-decode)
    hex_token = token_bytes.hex()
    if hex_token in text.lower():
        return True
    for hex_block in re.findall(r"[0-9a-fA-F]{%d,}" % _MIN_HEX_BLOCK, text):
        if len(hex_block) % 2 != 0:
            logger.debug("skipping odd-length hex block: %s", hex_block[:30])
            continue
        try:
            decoded = bytes.fromhex(hex_block).decode("utf-8")
            if token in decoded or (len(half) >= _MIN_PARTIAL_LEN and half in decoded):
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("hex decode error for block %r: %s", hex_block[:30], exc)

    # 6. Reversed
    if token[::-1] in text:
        return True

    # 7. ROT13
    try:
        rot13_decoded = codecs.decode(text, "rot_13")
        if token in rot13_decoded:
            return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("rot13 decode error: %s", exc)

    # 8. Unicode escapes (\\uXXXX sequences)
    if "\\u" in text:
        try:
            unicode_decoded = text.encode("utf-8").decode("unicode_escape")
            if token in unicode_decoded:
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("unicode escape decode error: %s", exc)

    # 9. URL-encoded
    try:
        url_decoded = urllib.parse.unquote(text)
        if url_decoded != text and token in url_decoded:
            return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("url decode error: %s", exc)

    return False
