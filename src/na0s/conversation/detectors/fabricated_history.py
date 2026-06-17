"""Fabricated history detector -- pasted fake conversation transcripts.

Works on SINGLE turns (not accumulated state).  Detects attempts to
inject a fabricated conversation history inside a single message.

Algorithm
---------
1. Count turn markers: ``User:``, ``Human:``, ``Assistant:``, etc.
2. If >= FABRICATED_TURN_MARKER_THRESHOLD (6), flag it.
3. Check for FABRICATED_KEYWORDS ("conversation history", "transcript", ...).
4. Check structural pattern: alternating speaker labels.
5. Length anomaly: if a single turn has 5x the average turn length AND
   contains turn markers.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..config import (
    ENABLE_FABRICATED_HISTORY,
    FABRICATED_KEYWORDS,
    FABRICATED_TURN_MARKER_THRESHOLD,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector

# Speaker label pattern (at start of a line or after a newline)
_TURN_MARKER = re.compile(
    r"(?:^|\n)\s*(?:User|Human|Assistant|AI|Bot|System)\s*:",
    re.IGNORECASE,
)

# Alternating pattern: at least 3 pairs of different speakers
# Use [\s\S]{0,500} (bounded) instead of .*? (unbounded) to prevent ReDoS
_ALTERNATING = re.compile(
    r"(?:(?:User|Human)\s*:[\s\S]{0,500}?\n\s*(?:Assistant|AI|Bot)\s*:)"
    r"|(?:(?:Assistant|AI|Bot)\s*:[\s\S]{0,500}?\n\s*(?:User|Human)\s*:)",
    re.IGNORECASE,
)


class FabricatedHistoryDetector(MultiTurnDetector):
    """Detect pasted fake conversation transcripts within a single turn."""

    @property
    def detector_name(self) -> str:
        return "fabricated_history"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["D1.22"]

    def reset(self) -> None:
        pass

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not ENABLE_FABRICATED_HISTORY:
            return []
        if state is None or state.is_empty:
            return []

        # Compute average turn length for length-anomaly check
        total_len = sum(len(t.text) for t in state.turns if t.text)
        avg_len = total_len / len(state.turns) if state.turns else 0

        alerts: List[Alert] = []
        for idx, turn in enumerate(state.turns):
            alert = self._check_turn(turn.text, idx, avg_len)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def analyze_single(self, text: str, turn_index: int = 0) -> List[Alert]:
        """Convenience: check a single piece of text without state."""
        alert = self._check_turn(text, turn_index, avg_len=0)
        return [alert] if alert else []

    # ----- internals -------------------------------------------------------

    def _check_turn(
        self, text: str, turn_index: int, avg_len: float = 0,
    ) -> Optional[Alert]:
        if not text:
            return None

        signals: List[str] = []
        score = 0.0

        # 1. Count turn markers
        markers = _TURN_MARKER.findall(text)
        marker_count = len(markers)
        if marker_count >= FABRICATED_TURN_MARKER_THRESHOLD:
            score += 0.5
            signals.append(f"turn_markers={marker_count}")

        # 2. Fabricated keywords
        text_lower = text.lower()
        kw_hits = [kw for kw in FABRICATED_KEYWORDS if kw in text_lower]
        if kw_hits:
            score += 0.15 * min(len(kw_hits), 3)
            signals.append(f"keywords={kw_hits}")

        # 3. Alternating speaker pattern
        alternating_count = len(_ALTERNATING.findall(text))
        if alternating_count >= 3:
            score += 0.3
            signals.append(f"alternating_pairs={alternating_count}")

        # 4. Length anomaly: 5x average turn length with turn markers present
        if avg_len > 0 and len(text) > 5 * avg_len and marker_count >= 4:
            score += 0.15
            signals.append(
                f"length_anomaly=5x_avg(len={len(text)}, avg={avg_len:.0f})"
            )
        elif len(text) > 1000 and marker_count >= 4:
            # Fallback: absolute length check when avg is not available
            score += 0.1
            signals.append(f"length_anomaly=len({len(text)})")

        if score < 0.5:
            return None

        confidence = min(1.0, score)
        return Alert(
            alert_type="fabricated_history",
            severity="high" if confidence >= 0.7 else "medium",
            confidence=round(confidence, 4),
            description=(
                f"Fabricated conversation history detected in turn {turn_index} "
                f"({marker_count} speaker markers, {alternating_count} alternating pairs)"
            ),
            turn_range=(turn_index, turn_index),
            evidence=signals,
        )
