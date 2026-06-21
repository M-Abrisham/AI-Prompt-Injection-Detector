"""Worm signature detection — self-replicating prompt injection patterns."""

from __future__ import annotations

import threading
from typing import Optional

from na0s.worm.detector import WormSignatureDetector

__all__ = ["WormSignatureDetector", "get_worm_detector"]

# ---------------------------------------------------------------------------
# Process-wide STATELESS worm detector for the INPUT path (WD-3)
# ---------------------------------------------------------------------------
# A shared WormSignatureDetector() with the default reconstruction_window=6
# keeps a cross-turn reconstruction buffer.  On the input path that buffer let a
# prior worm turn poison a later BENIGN turn: scan(worm) then scan("What is the
# capital of France?") flagged is_worm=True conf~0.8 because the benign text was
# joined with the buffered worm text and re-scanned.  The input pipeline must
# use a STATELESS detector, so this singleton forces reconstruction_window=1
# (=> _history_limit == 0 => the turn buffer is never populated or consulted).
_worm_detector_singleton: Optional[WormSignatureDetector] = None
_worm_detector_lock = threading.Lock()


def get_worm_detector() -> WormSignatureDetector:
    """Return the process-wide STATELESS worm detector for the input path.

    Thread-safe double-checked singleton.  Always constructed with
    ``reconstruction_window=1`` so the cross-turn reconstruction buffer is
    disabled (``_history_limit == 0``) — a worm turn can never raise the verdict
    of a subsequent benign turn through this instance.  Callers that genuinely
    want stateful cross-turn reconstruction must construct their own
    ``WormSignatureDetector`` explicitly.
    """
    global _worm_detector_singleton
    if _worm_detector_singleton is None:
        with _worm_detector_lock:
            if _worm_detector_singleton is None:
                _worm_detector_singleton = WormSignatureDetector(reconstruction_window=1)
    return _worm_detector_singleton
