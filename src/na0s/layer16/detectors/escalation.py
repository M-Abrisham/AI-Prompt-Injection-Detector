"""Escalation detector -- gradual risk increase across turns (C1.1, C1MT).

Algorithm
---------
1. Extract risk scores from conversation turns.
2. If fewer than ESCALATION_MIN_TURNS, return no alert.
3. Compute slope via least-squares:
     slope = sum((i - mean_i)(r_i - mean_r)) / sum((i - mean_i)^2)
4. Compute R-squared for confidence calibration.
5. If slope > ESCALATION_SLOPE_THRESHOLD AND latest risk > 0.5: alert.
6. Confidence = min(1.0, slope / ESCALATION_SLOPE_THRESHOLD * r_squared).
7. Monotonic boost: if last 3 turns are monotonically increasing, boost
   confidence by 0.1.
"""

from __future__ import annotations

from typing import List

from ..config import (
    ENABLE_ESCALATION,
    ESCALATION_MIN_TURNS,
    ESCALATION_SLOPE_THRESHOLD,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector


def _linear_slope(values: List[float]) -> float:
    """Compute the slope of a least-squares fit, numpy-free.

    Uses the deviation-based formula:
        slope = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
    where x = 0, 1, 2, ... (turn indices) and y = risk scores.
    """
    n = len(values)
    if n < 2:
        return 0.0

    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n

    numer = 0.0
    denom = 0.0
    for i, y in enumerate(values):
        dx = i - mean_x
        numer += dx * (y - mean_y)
        denom += dx * dx

    if denom == 0.0:
        return 0.0

    return numer / denom


def _r_squared(values: List[float], slope: float) -> float:
    """Compute R-squared (coefficient of determination) for the linear fit."""
    n = len(values)
    if n < 2:
        return 0.0

    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n

    ss_tot = 0.0
    ss_res = 0.0
    for i, y in enumerate(values):
        predicted = mean_y + slope * (i - mean_x)
        ss_res += (y - predicted) ** 2
        ss_tot += (y - mean_y) ** 2

    if ss_tot == 0.0:
        return 0.0

    return max(0.0, 1.0 - ss_res / ss_tot)


def _is_monotonically_increasing(values: List[float]) -> bool:
    """Check if every element is strictly greater than the previous."""
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            return False
    return True


class EscalationDetector(MultiTurnDetector):
    """Detect gradual risk escalation across conversation turns."""

    # ----- MultiTurnDetector interface -------------------------------------

    @property
    def detector_name(self) -> str:
        return "escalation"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["C1.1", "C1MT.1", "C1MT.3"]

    def reset(self) -> None:
        pass  # stateless -- all data comes from ConversationState

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not ENABLE_ESCALATION:
            return []
        if state is None or state.is_empty:
            return []

        scores = [t.risk_score for t in state.turns]

        alerts: List[Alert] = []

        # --- trend-based escalation ----------------------------------------
        if len(scores) >= ESCALATION_MIN_TURNS:
            slope = _linear_slope(scores)
            latest = scores[-1]

            if slope > ESCALATION_SLOPE_THRESHOLD and latest > 0.5:
                r2 = _r_squared(scores, slope)
                confidence = min(1.0, slope / ESCALATION_SLOPE_THRESHOLD * r2)

                # Monotonic boost: if last 3 turns are monotonically increasing
                if len(scores) >= 3 and _is_monotonically_increasing(scores[-3:]):
                    confidence = min(1.0, confidence + 0.1)

                alerts.append(
                    Alert(
                        alert_type="escalation",
                        severity="high" if confidence >= 0.7 else "medium",
                        confidence=round(confidence, 4),
                        description=(
                            f"Risk escalation detected: slope={slope:.3f} "
                            f"over {len(scores)} turns (latest={latest:.2f})"
                        ),
                        turn_range=(0, len(scores) - 1),
                        evidence=[
                            f"slope={slope:.4f}",
                            f"r_squared={r2:.4f}",
                            f"threshold={ESCALATION_SLOPE_THRESHOLD}",
                            f"scores={[round(s, 2) for s in scores]}",
                        ],
                    )
                )

        # --- rapid escalation (last 3 turns all above 0.5) ----------------
        if len(scores) >= 3 and all(s > 0.5 for s in scores[-3:]):
            alerts.append(
                Alert(
                    alert_type="escalation",
                    severity="high",
                    confidence=0.85,
                    description=(
                        "Rapid escalation: last 3 turns all above 0.5 risk "
                        f"({[round(s, 2) for s in scores[-3:]]})"
                    ),
                    turn_range=(max(0, len(scores) - 3), len(scores) - 1),
                    evidence=[
                        f"recent_scores={[round(s, 2) for s in scores[-3:]]}",
                        "rapid_escalation=True",
                    ],
                )
            )

        return alerts
