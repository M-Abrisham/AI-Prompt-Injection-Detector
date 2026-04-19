"""Streaming output scanner -- scan LLM output as it arrives chunk by chunk.

Wraps :class:`OutputScanner` to support streaming / SSE responses.
Lightweight per-chunk checks (secrets, role-break) run on every chunk,
while the full scan runs only on :meth:`finalize`.

Thread-safe: the internal buffer is protected by a :class:`threading.Lock`.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

from .scanner import (
    OutputScanner,
    OutputScanResult,
    _SECRET_PATTERNS,
    _ROLE_BREAK_PATTERNS,
)


class StreamingOutputScanner:
    """Incrementally scan LLM output delivered as a stream of chunks.

    Parameters
    ----------
    scanner : OutputScanner or None
        The scanner instance to use for the final full scan.
        If ``None``, a default ``OutputScanner("medium")`` is created.
    chunk_size : int
        Minimum number of accumulated characters before running
        per-chunk lightweight checks (default 100).
    """

    def __init__(
        self,
        scanner: Optional[OutputScanner] = None,
        chunk_size: int = 100,
    ) -> None:
        self._scanner = scanner or OutputScanner("medium")
        self._chunk_size = max(1, chunk_size)
        self._buffer: str = ""
        self._lock = threading.Lock()
        self._alerts: list[dict] = []
        # Track how much of the buffer has been checked already
        self._checked_up_to: int = 0

    # ---- public API -------------------------------------------------------

    def process_chunk(self, chunk: str) -> Optional[dict]:
        """Append *chunk* to the buffer and run lightweight checks.

        Returns a dict ``{"alert": ..., "pattern": ...}`` if a
        suspicious pattern is found in the newly accumulated text,
        otherwise ``None``.
        """
        with self._lock:
            self._buffer += chunk

            # Only run lightweight checks when we have enough new text
            if len(self._buffer) - self._checked_up_to < self._chunk_size:
                return None

            # Check the unchecked portion (with some overlap for boundary matches)
            overlap = 40  # chars of overlap so patterns crossing chunk boundaries are caught
            start = max(0, self._checked_up_to - overlap)
            window = self._buffer[start:]
            self._checked_up_to = len(self._buffer)

            alert = self._lightweight_check(window)
            if alert:
                self._alerts.append(alert)
            return alert

    def finalize(
        self,
        original_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> OutputScanResult:
        """Run the full scanner on the accumulated buffer and return the result."""
        with self._lock:
            text = self._buffer
        return self._scanner.scan(
            output_text=text,
            original_prompt=original_prompt,
            system_prompt=system_prompt,
        )

    def reset(self) -> None:
        """Clear the buffer and alerts for the next response."""
        with self._lock:
            self._buffer = ""
            self._checked_up_to = 0
            self._alerts.clear()

    @property
    def buffer(self) -> str:
        """Return a snapshot of the current buffer contents."""
        with self._lock:
            return self._buffer

    @property
    def alerts(self) -> list[dict]:
        """Return all alerts raised during chunk processing."""
        with self._lock:
            return list(self._alerts)

    # ---- internal ---------------------------------------------------------

    @staticmethod
    def _lightweight_check(text: str) -> Optional[dict]:
        """Fast per-chunk check for secrets and role-break indicators."""
        for pat in _SECRET_PATTERNS:
            match = pat.search(text)
            if match:
                return {
                    "alert": "secret_pattern",
                    "pattern": pat.pattern[:50],
                    "match": match.group()[:30],
                }

        for pat in _ROLE_BREAK_PATTERNS:
            match = pat.search(text)
            if match:
                return {
                    "alert": "role_break",
                    "pattern": pat.pattern[:50],
                    "match": match.group()[:60],
                }

        return None
