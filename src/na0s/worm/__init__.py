"""Worm signature detection — self-replicating prompt injection patterns.

Public API
----------
``WormSignatureDetector``
    Full detector.  Supports single-input scanning *and* cross-message /
    output-replication analysis (``source_text`` comparison plus an internal
    cross-turn reconstruction buffer).  The cross-message capability is an
    OUTPUT / multi-message concern — it is used by the output-scan and
    propagation code paths, not by the single-input ``na0s.scan()`` entry point.

``get_worm_detector()``
    Process-wide *stateless* singleton (``reconstruction_window=1`` disables the
    cross-turn buffer) suitable for single-input classification.  This is what
    ``predict.classify_prompt()`` uses to fold a conservative worm-propagation
    signal into ``scan()`` without making scans stateful or order-dependent.
"""

from __future__ import annotations

import threading

from .detector import WormSignatureDetector

__all__ = ["WormSignatureDetector", "get_worm_detector"]

_STATELESS_DETECTOR: WormSignatureDetector | None = None
_STATELESS_LOCK = threading.Lock()


def get_worm_detector() -> WormSignatureDetector:
    """Return a process-wide stateless ``WormSignatureDetector``.

    The instance is created with ``reconstruction_window=1`` so the cross-turn
    reconstruction buffer is disabled (``_history_limit == 0``).  Each
    ``scan(text)`` call is therefore independent and deterministic — no state
    leaks between unrelated single-input scans.  This is the detector wired into
    ``predict.classify_prompt()``.
    """
    global _STATELESS_DETECTOR
    if _STATELESS_DETECTOR is None:
        with _STATELESS_LOCK:
            if _STATELESS_DETECTOR is None:
                _STATELESS_DETECTOR = WormSignatureDetector(reconstruction_window=1)
    return _STATELESS_DETECTOR
