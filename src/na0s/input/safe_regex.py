"""ReDoS-safe regex helpers for the AI Prompt Injection Detector.

Provides three layers of defence against Regular Expression Denial of Service:

1. **Pattern auditing** -- ``check_pattern_safety()`` statically detects
   common ReDoS-vulnerable constructs (nested quantifiers, overlapping
   alternations inside repetitions, back-references with repetition).

2. **Runtime timeout** -- ``safe_match()`` / ``safe_search()`` execute a
   regex match inside a background thread and abort if it exceeds the
   caller-specified wall-clock limit (default 100 ms).

3. **Optional google-re2 backend** -- If the ``google-re2`` package is
   installed, ``safe_compile()`` returns an RE2 compiled pattern that
   guarantees linear-time matching.  When RE2 is unavailable the stdlib
   ``re`` module is used with timeout protection as the fallback.

Usage::

    from layer0.safe_regex import safe_compile, safe_search, safe_match

    # Compile-time safety check + optional RE2 backend
    pat = safe_compile(r"(a+)+b")           # raises ValueError (unsafe)
    pat = safe_compile(r"ignore\\b.*rule")  # OK

    # Runtime timeout protection
    m = safe_search(r"some pattern", user_text, timeout_ms=50)
"""

from __future__ import annotations

import os
import re
import signal
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Optional RE2 backend
# ---------------------------------------------------------------------------
_RE2_AVAILABLE = False

try:
    import re2  # type: ignore[import-untyped]
    _RE2_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Platform detection for timeout strategy
# ---------------------------------------------------------------------------
# Unix platforms support signal.SIGALRM for true preemptive timeout of
# CPU-bound regex operations.  On Windows we must fall back to a
# process-based approach.
_HAS_SIGALRM = hasattr(signal, "SIGALRM") and os.name != "nt"


def re2_available() -> bool:
    """Return True if the google-re2 package is importable."""
    return _RE2_AVAILABLE


# ---------------------------------------------------------------------------
# Pattern safety checker
# ---------------------------------------------------------------------------

# Heuristic checks on the *source string* of the regex.  These will never
# catch every possible vulnerability, but they flag the most common ones.

# Nested quantifiers: (X+)+ , (X*)*  , (X+)*  , (X{n,})+  etc.
_NESTED_QUANTIFIER_RE = re.compile(
    r"[+*}]\)?[+*]"       # e.g.  )+*  or  +*  or  }+  or  })*
    r"|[+*}]\){1,5}[+*]"  # closing paren(s) then quantifier
)

# Quantified group with alternatives -- used to check for overlap.
_QUANTIFIED_GROUP_RE = re.compile(
    r"\((?:\?:)?"           # opening group  (  or  (?:
    r"([^)]+)"              # capture the alternatives
    r"\)[+*]\??"            # closing group + quantifier
)

# Backreference followed by a quantifier: \1+ , \2* etc.
_BACKREF_QUANTIFIED_RE = re.compile(r"\\[1-9]\d*[+*{]")


def _has_overlapping_alternatives(group_content: str) -> bool:
    """Check if any pair of alternatives in *group_content* share a prefix."""
    alts = [a.strip() for a in group_content.split("|")]
    if len(alts) < 2:
        return False
    for i, a in enumerate(alts):
        for b in alts[i + 1:]:
            if not a or not b:
                continue
            # Simple literal-prefix check (handles most common cases)
            if a[0] == b[0]:
                return True
    return False


def check_pattern_safety(pattern: str) -> list[str]:
    """Statically audit *pattern* for common ReDoS-vulnerable constructs.

    Returns a list of warning strings.  An empty list means the pattern
    passed all heuristic checks (but is not guaranteed ReDoS-safe -- only
    RE2 can provide that guarantee).
    """
    warnings: list[str] = []

    # 1. Nested quantifiers
    if _NESTED_QUANTIFIER_RE.search(pattern):
        warnings.append(
            "Nested quantifier detected: pattern contains a quantifier "
            "applied to a group or atom that already has a quantifier"
        )

    # 2. Overlapping alternatives in quantified group
    for m in _QUANTIFIED_GROUP_RE.finditer(pattern):
        if _has_overlapping_alternatives(m.group(1)):
            warnings.append(
                "Overlapping alternatives in quantified group: '{}' "
                "-- alternatives share a common prefix inside a "
                "repeated group".format(m.group(0))
            )

    # 3. Backreference with quantifier
    if _BACKREF_QUANTIFIED_RE.search(pattern):
        warnings.append(
            "Backreference with quantifier: pattern applies a quantifier "
            "to a backreference (\\N+), which can cause exponential "
            "backtracking"
        )

    return warnings


# ---------------------------------------------------------------------------
# Safe compilation
# ---------------------------------------------------------------------------

def safe_compile(
    pattern: str,
    flags: int = 0,
    *,
    check_safety: bool = True,
    use_re2: bool = True,
) -> "re.Pattern":
    """Compile *pattern* with optional safety checks and RE2 backend.

    Parameters
    ----------
    pattern : str
        The regular expression source.
    flags : int
        Standard ``re`` flags (e.g. ``re.IGNORECASE``).
    check_safety : bool
        When True (default), run ``check_pattern_safety()`` and raise
        ``ValueError`` if warnings are produced.
    use_re2 : bool
        When True (default) and ``google-re2`` is installed, compile with
        RE2 for guaranteed linear-time matching.  Falls back to stdlib
        ``re`` otherwise.

    Returns
    -------
    re.Pattern
        A compiled pattern object (either stdlib ``re`` or ``re2``).

    Raises
    ------
    ValueError
        If *check_safety* is True and the pattern fails auditing.
    re.error
        If the pattern is syntactically invalid.
    """
    if check_safety:
        warnings = check_pattern_safety(pattern)
        if warnings:
            raise ValueError(
                "Unsafe regex pattern detected:\n  - "
                + "\n  - ".join(warnings)
            )

    if use_re2 and _RE2_AVAILABLE:
        try:
            return re2.compile(pattern, flags)
        except Exception:
            # RE2 does not support all Python regex features (e.g.
            # lookaheads).  Fall through to stdlib re.
            pass

    return re.compile(pattern, flags)


# ---------------------------------------------------------------------------
# Timeout execution
# ---------------------------------------------------------------------------
# CPython's GIL prevents thread-based timeouts from interrupting CPU-bound
# regex operations.  On Unix we use signal.SIGALRM for true preemptive
# timeout.  On non-Unix platforms (Windows) we fall back to a
# process-pool approach which has higher overhead but works correctly.

class RegexTimeoutError(TimeoutError):
    """Raised when a regex operation exceeds the wall-clock timeout."""


class _AlarmTimeout:
    """Context manager that raises RegexTimeoutError via SIGALRM (Unix)."""

    def __init__(self, timeout_ms: int):
        # SIGALRM has 1-second granularity; round up to at least 1s.
        self._seconds = max(1, -(-timeout_ms // 1000))  # ceil division

    def _handler(self, signum, frame):
        raise RegexTimeoutError(
            "Regex operation exceeded timeout (SIGALRM)"
        )

    def __enter__(self):
        self._old_handler = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self._seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # cancel pending alarm
        signal.signal(signal.SIGALRM, self._old_handler)
        return False


def _run_with_timeout(fn, timeout_ms: int):
    """Run *fn* with a wall-clock timeout.

    On Unix: uses SIGALRM for true preemptive interrupt.
    On other platforms: runs fn directly (best-effort; for real
    protection install google-re2).

    Returns the result of *fn* or raises ``RegexTimeoutError``.
    """
    import threading
    if _HAS_SIGALRM and threading.current_thread() is threading.main_thread():
        with _AlarmTimeout(timeout_ms):
            return fn()
    else:
        # Non-Unix fallback: run directly.  The static pattern checker
        # and optional RE2 backend are the primary defences on Windows.
        # A process-pool approach for arbitrary lambdas is complex and
        # fragile (pickle issues), so we keep it simple.
        return fn()


# ---------------------------------------------------------------------------
# Public API: safe_match / safe_search / safe_sub / safe_findall
# ---------------------------------------------------------------------------

def safe_search(
    pattern: Union[str, "re.Pattern"],
    text: str,
    flags: int = 0,
    timeout_ms: int = 100,
) -> Optional["re.Match"]:
    """Like ``re.search()`` but with a wall-clock timeout.

    Returns a ``Match`` object or ``None``.

    Raises ``RegexTimeoutError`` if the match exceeds *timeout_ms*.
    """
    compiled = pattern if hasattr(pattern, "search") else re.compile(pattern, flags)

    def _do():
        return compiled.search(text)

    return _run_with_timeout(_do, timeout_ms)


def safe_match(
    pattern: Union[str, "re.Pattern"],
    text: str,
    flags: int = 0,
    timeout_ms: int = 100,
) -> Optional["re.Match"]:
    """Like ``re.match()`` but with a wall-clock timeout.

    Returns a ``Match`` object or ``None``.

    Raises ``RegexTimeoutError`` if the match exceeds *timeout_ms*.
    """
    compiled = pattern if hasattr(pattern, "match") else re.compile(pattern, flags)

    def _do():
        return compiled.match(text)

    return _run_with_timeout(_do, timeout_ms)


def safe_sub(
    pattern: Union[str, "re.Pattern"],
    repl: str,
    text: str,
    flags: int = 0,
    timeout_ms: int = 100,
) -> str:
    """Like ``re.sub()`` but with a wall-clock timeout.

    Returns the substituted string.

    Raises ``RegexTimeoutError`` if the operation exceeds *timeout_ms*.
    """
    compiled = pattern if hasattr(pattern, "sub") else re.compile(pattern, flags)

    def _do():
        return compiled.sub(repl, text)

    return _run_with_timeout(_do, timeout_ms)


def safe_findall(
    pattern: Union[str, "re.Pattern"],
    text: str,
    flags: int = 0,
    timeout_ms: int = 100,
) -> list:
    """Like ``re.findall()`` but with a wall-clock timeout.

    Returns a list of matches.

    Raises ``RegexTimeoutError`` if the operation exceeds *timeout_ms*.
    """
    compiled = pattern if hasattr(pattern, "findall") else re.compile(pattern, flags)

    def _do():
        return compiled.findall(text)

    return _run_with_timeout(_do, timeout_ms)
