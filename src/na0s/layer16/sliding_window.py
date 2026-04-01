"""Layer 16 SlidingWindow — bounded turn history.

Count-based sliding window backed by collections.deque for O(1)
append and eviction.
"""

from __future__ import annotations

from collections import deque
from typing import List

from na0s.layer16.models import ConversationTurn


class SlidingWindow:
    """A fixed-size sliding window over conversation turns.

    Uses a deque with maxlen so the oldest turn is automatically
    evicted when the window is full.

    Args:
        max_size: Maximum number of turns to retain. Defaults to 10.
    """

    def __init__(self, max_size: int = 10) -> None:
        self._max_size = max_size
        self._turns: deque[ConversationTurn] = deque(maxlen=max_size)

    def add(self, turn: ConversationTurn) -> None:
        """Add a turn, evicting the oldest if the window is full.

        Args:
            turn: The ConversationTurn to append.
        """
        self._turns.append(turn)

    def get_turns(self) -> List[ConversationTurn]:
        """Return all turns in the window, oldest first.

        Returns:
            List of ConversationTurn objects.
        """
        return list(self._turns)

    def get_combined_text(self) -> str:
        """Concatenate all turn texts in the window.

        Returns:
            Single string with turn texts joined by newlines.
        """
        return "\n".join(t.text for t in self._turns)

    def get_risk_scores(self) -> List[float]:
        """Return the risk score for each turn in the window.

        Returns:
            List of float risk scores.
        """
        return [t.risk_score for t in self._turns]

    @property
    def is_full(self) -> bool:
        """True if the window has reached max_size."""
        return len(self._turns) == self._max_size

    @property
    def size(self) -> int:
        """Number of turns currently in the window."""
        return len(self._turns)

    @property
    def max_size(self) -> int:
        """Maximum capacity of the window."""
        return self._max_size
