"""Token-bucket rate limiter for LLM Judge API calls.

Thread-safe implementation that supports both blocking (with timeout)
and non-blocking acquire semantics.
"""

import threading
import time


class TokenBucketRateLimiter:
    """Token-bucket rate limiter.

    Parameters
    ----------
    rate : float
        Tokens added per second.
    burst : int
        Maximum number of tokens the bucket can hold.
    """

    def __init__(self, rate: float = 10.0, burst: int = 20):
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def _refill(self) -> None:
        """Add tokens based on elapsed time (must be called under lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def try_acquire(self) -> bool:
        """Non-blocking: consume one token if available, else return False."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def acquire(self, timeout: float = 5.0) -> bool:
        """Blocking: wait up to *timeout* seconds for a token.

        Returns True if a token was acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                # Wait a short interval then retry
                wait = min(remaining, 1.0 / self._rate if self._rate > 0 else remaining)
                self._condition.wait(timeout=wait)
