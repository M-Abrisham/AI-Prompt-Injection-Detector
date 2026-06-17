"""Bridge for Layer 16 detectors to re-scan text through Na0S WITHOUT recursion.

ARCHITECTURE DECISION: Option D — Stateless Scan (no session_id).

scan() without session_id does NOT trigger Layer 16 (gated by
``if session_id:`` in predict.py:1656). This is safe because Layer 16
only activates when session_id is explicitly provided.

Usage::

    from na0s.conversation.scan_bridge import rescan_text

    result = rescan_text("combined multi-turn text")
    if result.is_malicious:
        # The combined text triggers Na0S detection
        print(f"Risk: {result.risk_score}, Detections: {result.detections}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

logger = logging.getLogger(__name__)

# Threshold for classifying re-scan result as malicious.
# Reads from Na0S config if available, falls back to 0.55 (Na0S default).
try:
    from na0s.predict import DECISION_THRESHOLD
    RESCAN_THRESHOLD = DECISION_THRESHOLD
except ImportError:
    RESCAN_THRESHOLD = 0.55


@dataclass
class RescanResult:
    """Result of a stateless re-scan (no multi-turn context)."""

    risk_score: float = 0.0
    is_malicious: bool = False
    detections: List[str] = field(default_factory=list)
    raw_result: Any = None


def rescan_text(text: str) -> RescanResult:
    """Run Na0S detection on text WITHOUT triggering Layer 16.

    Calls ``na0s.predict.scan(text)`` with NO session_id, so the
    Layer 16 multi-turn pipeline is not invoked. This prevents
    infinite recursion when the payload splitting detector needs
    to re-scan combined text.

    Parameters
    ----------
    text : str
        Combined/assembled text to scan.

    Returns
    -------
    RescanResult
        Detection result with risk_score, is_malicious flag, and
        list of triggered detector names.
    """
    if not text or not text.strip():
        return RescanResult()

    logger.debug("scan_bridge: re-scanning %d chars", len(text))

    from na0s.predict import scan  # lazy import to avoid circular

    # NO session_id → Layer 16 stays dormant → no recursion
    raw = scan(text)

    risk = getattr(raw, "risk_score", 0.0)
    detections = []
    if getattr(raw, "rule_hits", None):
        detections.extend(raw.rule_hits)
    if getattr(raw, "technique_tags", None):
        detections.extend(raw.technique_tags)

    return RescanResult(
        risk_score=risk,
        is_malicious=risk >= RESCAN_THRESHOLD or getattr(raw, "is_malicious", False),
        detections=detections,
        raw_result=raw,
    )
