"""Layer 0 resource exhaustion protection.

Guards against:
    - Oversized input (char + byte limits, extends validation.py)
    - Deep HTML nesting (limits parser recursion depth)
    - Expansion-ratio attacks (zip-bomb style normalization output)
    - Memory cap enforcement (reject if input would exceed budget)
    - Optional per-caller rate limiting (token-bucket, disabled by default)

All limits are configurable via environment variables with sensible defaults.

Environment variables
---------------------
L0_MAX_INPUT_CHARS           Max character count.            Default: 50000
L0_MAX_INPUT_BYTES           Max byte count (UTF-8).         Default: 200000
L0_MAX_HTML_DEPTH            Max HTML nesting depth.         Default: 100
L0_MAX_EXPANSION_RATIO       Max output/input length ratio.  Default: 10.0
L0_MEMORY_CAP_MB             Max memory budget (MB).         Default: 50
L0_RATE_LIMIT_ENABLED        Enable rate limiting.           Default: 0 (off)
L0_RATE_LIMIT_REQUESTS       Max requests per window.        Default: 100
L0_RATE_LIMIT_WINDOW_SEC     Sliding window seconds.         Default: 60
"""

import logging
import os
import sys
import threading
import time
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (all from env vars with sensible defaults)
# ---------------------------------------------------------------------------
from na0s._env import safe_int_env, safe_float_env

MAX_INPUT_CHARS = safe_int_env("L0_MAX_INPUT_CHARS", 50000, lo=1)
MAX_INPUT_BYTES = safe_int_env("L0_MAX_INPUT_BYTES", 200000, lo=1)
MAX_HTML_DEPTH = safe_int_env("L0_MAX_HTML_DEPTH", 100, lo=1)
MAX_EXPANSION_RATIO = safe_float_env("L0_MAX_EXPANSION_RATIO", 10.0, lo=1.0)
MEMORY_CAP_MB = safe_float_env("L0_MEMORY_CAP_MB", 50.0, lo=1.0)
RATE_LIMIT_ENABLED = os.getenv("L0_RATE_LIMIT_ENABLED", "0") == "1"
RATE_LIMIT_REQUESTS = safe_int_env("L0_RATE_LIMIT_REQUESTS", 100, lo=1)
RATE_LIMIT_WINDOW_SEC = safe_float_env("L0_RATE_LIMIT_WINDOW_SEC", 60.0, lo=1.0)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ResourceLimitExceeded(Exception):
    """Raised when an input exceeds a resource guard limit."""

    def __init__(self, guard_name: str, detail: str):
        self.guard_name = guard_name
        self.detail = detail
        super().__init__(
            "Resource guard '{}': {}".format(guard_name, detail)
        )


# ---------------------------------------------------------------------------
# Input size guard
# ---------------------------------------------------------------------------
def check_input_size(text: str) -> None:
    """Reject input exceeding char or byte limits.

    This complements validation.py (which already checks sizes) by being
    callable independently from the resource-guard entry point.

    Raises ResourceLimitExceeded on violation.
    """
    if len(text) > MAX_INPUT_CHARS:
        raise ResourceLimitExceeded(
            "input_size",
            "input exceeds {} char limit (got {})".format(
                MAX_INPUT_CHARS, len(text)
            ),
        )
    byte_len = len(text.encode("utf-8", errors="replace"))
    if byte_len > MAX_INPUT_BYTES:
        raise ResourceLimitExceeded(
            "input_size",
            "input exceeds {} byte limit (got {})".format(
                MAX_INPUT_BYTES, byte_len
            ),
        )


# ---------------------------------------------------------------------------
# HTML depth guard
# ---------------------------------------------------------------------------
class _DepthChecker(HTMLParser):
    """Lightweight parser that only tracks nesting depth."""

    def __init__(self, max_depth):
        super().__init__()
        self._depth = 0
        self._max_depth = max_depth
        self.exceeded = False

    def handle_starttag(self, tag, attrs):
        self._depth += 1
        if self._depth > self._max_depth:
            self.exceeded = True

    def handle_endtag(self, tag):
        if self._depth > 0:
            self._depth -= 1


def check_html_depth(text: str, max_depth: int = None) -> None:
    """Reject HTML with nesting deeper than *max_depth*.

    Only runs the check if the text looks like it might contain HTML tags.
    Raises ResourceLimitExceeded on violation.
    """
    if max_depth is None:
        max_depth = MAX_HTML_DEPTH

    # Quick bail-out: no angle brackets means no HTML
    if "<" not in text:
        return

    checker = _DepthChecker(max_depth)
    try:
        checker.feed(text)
    except Exception:
        logger.debug("HTML depth check parse error", exc_info=True)

    if checker.exceeded:
        raise ResourceLimitExceeded(
            "html_depth",
            "HTML nesting exceeds max depth of {}".format(max_depth),
        )


# ---------------------------------------------------------------------------
# Expansion ratio guard
# ---------------------------------------------------------------------------
def check_expansion_ratio(
    original_len: int, output_len: int, max_ratio: float = None
) -> None:
    """Reject if output expanded beyond *max_ratio* x original length.

    Prevents zip-bomb style attacks where normalization causes massive
    expansion (e.g. Unicode compatibility decomposition).

    Raises ResourceLimitExceeded on violation.
    """
    if max_ratio is None:
        max_ratio = MAX_EXPANSION_RATIO

    if original_len == 0:
        return

    ratio = output_len / original_len
    if ratio > max_ratio:
        raise ResourceLimitExceeded(
            "expansion_ratio",
            "output/input ratio {:.1f}x exceeds max {:.1f}x (original={}, output={})".format(
                ratio, max_ratio, original_len, output_len
            ),
        )


# ---------------------------------------------------------------------------
# Memory cap guard
# ---------------------------------------------------------------------------
def check_memory_budget(text: str, cap_mb: float = None) -> None:
    """Reject if processing *text* would likely exceed the memory cap.

    Estimates memory as: sys.getsizeof(text) + 3x overhead for intermediate
    copies during normalization / tokenization.

    Raises ResourceLimitExceeded on violation.
    """
    if cap_mb is None:
        cap_mb = MEMORY_CAP_MB

    if cap_mb <= 0:
        return  # disabled

    text_bytes = sys.getsizeof(text)
    # Conservative estimate: original + normalized copy + HTML copy + token list
    estimated_mb = (text_bytes * 4) / (1024 * 1024)

    if estimated_mb > cap_mb:
        raise ResourceLimitExceeded(
            "memory_cap",
            "estimated memory {:.1f} MB exceeds cap {:.1f} MB".format(
                estimated_mb, cap_mb
            ),
        )


# ---------------------------------------------------------------------------
# Rate limiter (sliding-window token bucket)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Tracks request timestamps per caller_id.  Disabled by default
    (``L0_RATE_LIMIT_ENABLED=0``).
    """

    def __init__(
        self,
        max_requests: int = None,
        window_sec: float = None,
        enabled: bool = None,
    ):
        self._max_requests = (
            max_requests if max_requests is not None else RATE_LIMIT_REQUESTS
        )
        self._window_sec = (
            window_sec if window_sec is not None else RATE_LIMIT_WINDOW_SEC
        )
        self._enabled = enabled if enabled is not None else RATE_LIMIT_ENABLED
        self._lock = threading.Lock()
        self._windows: dict = {}  # caller_id -> [timestamp, ...]

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check(self, caller_id: str = "default") -> None:
        """Check rate limit for *caller_id*.

        Raises ResourceLimitExceeded if the caller exceeds the rate limit.
        No-op if rate limiting is disabled.
        """
        if not self._enabled:
            return

        now = time.monotonic()
        cutoff = now - self._window_sec

        with self._lock:
            timestamps = self._windows.get(caller_id, [])
            # Prune expired timestamps
            timestamps = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= self._max_requests:
                raise ResourceLimitExceeded(
                    "rate_limit",
                    "caller '{}' exceeded {} requests in {:.0f}s window".format(
                        caller_id, self._max_requests, self._window_sec
                    ),
                )
            timestamps.append(now)
            self._windows[caller_id] = timestamps


# Module-level singleton rate limiter
_rate_limiter = RateLimiter()


def check_rate_limit(caller_id: str = "default") -> None:
    """Module-level convenience for the default rate limiter."""
    _rate_limiter.check(caller_id)


# ---------------------------------------------------------------------------
# Combined guard (convenience for sanitizer.py)
# ---------------------------------------------------------------------------
def run_entry_guards(text: str, caller_id: str = "default") -> None:
    """Run all resource guards at pipeline entry.

    Call this at the top of ``layer0_sanitize()`` before any processing.
    Raises ResourceLimitExceeded on any violation.
    """
    check_rate_limit(caller_id)
    check_input_size(text)
    check_memory_budget(text)
    check_html_depth(text)
