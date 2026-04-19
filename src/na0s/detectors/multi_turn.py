"""Multi-turn context validator for positive validation (Layer 8, P2).

Tracks conversation turns and detects escalation patterns where
validation confidence degrades over successive turns -- a signal
that an attacker is gradually probing or wearing down defenses.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from ..validation import ValidationResult


_DEFAULT_WINDOW = 10
_ESCALATION_STREAK = 3


@dataclass
class _TurnRecord:
    """Internal record of a single conversation turn."""
    text: str
    confidence: float
    is_valid: bool


class MultiTurnValidator:
    """Track rolling window of conversation turns and detect escalation.

    Parameters
    ----------
    window_size : int
        Maximum number of turns to retain (default 10).
    escalation_streak : int
        Number of consecutive declining confidence scores that
        constitutes an escalation (default 3).
    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW,
        escalation_streak: int = _ESCALATION_STREAK,
    ) -> None:
        self.window_size = window_size
        self.escalation_streak = escalation_streak
        self._turns: Deque[_TurnRecord] = deque(maxlen=window_size)

    # ---- public API -------------------------------------------------------

    def record_turn(self, text: str, result: ValidationResult) -> None:
        """Record a conversation turn with its validation result."""
        self._turns.append(
            _TurnRecord(
                text=text,
                confidence=result.confidence,
                is_valid=result.is_valid,
            )
        )

    def detect_escalation(self) -> bool:
        """Return True if the last N turns show declining confidence.

        Escalation is defined as *escalation_streak* or more consecutive
        turns where each confidence is strictly lower than the previous.
        """
        if len(self._turns) < self.escalation_streak:
            return False

        # Check the tail of the window for a declining streak
        turns = list(self._turns)
        declining = 0
        for i in range(len(turns) - 1, 0, -1):
            if turns[i].confidence < turns[i - 1].confidence:
                declining += 1
            else:
                break
        return declining >= self.escalation_streak

    def get_turn_count(self) -> int:
        """Return the number of turns currently tracked."""
        return len(self._turns)

    def reset(self) -> None:
        """Clear all recorded turns (new conversation)."""
        self._turns.clear()

    def get_confidence_history(self) -> List[float]:
        """Return list of confidence scores in chronological order."""
        return [t.confidence for t in self._turns]
