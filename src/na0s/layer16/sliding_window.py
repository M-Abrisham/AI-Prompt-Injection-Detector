"""Layer 16 SlidingWindow — bounded turn history with weighted eviction.

Suspicious turns (high risk_score) receive higher weight and persist
longer in the window.  Each time a new turn is added, existing weights
decay by a configurable factor.  When the window is full the turn with
the *lowest* weight is evicted, so benign turns are dropped first.
"""

from __future__ import annotations

from typing import List, Tuple

from na0s.layer16.models import ConversationTurn

# Default weight parameters
_DEFAULT_DECAY = 0.9  # per-turn multiplicative decay
_SUSPICIOUS_THRESHOLD = 0.5  # risk_score above this gets a weight boost
_BASE_WEIGHT = 1.0
_SUSPICIOUS_BOOST = 2.0  # extra weight for suspicious turns


class SlidingWindow:
    """A fixed-size sliding window over conversation turns.

    Turns are stored alongside a weight value.  When the window is full
    the lowest-weight turn is evicted first, so suspicious turns persist
    longer than benign ones.  All weights decay each time a new turn is
    added, keeping the window fresh.

    Args:
        max_size: Maximum number of turns to retain. Defaults to 10.
        decay_factor: Multiplicative decay applied to all weights on
            each ``add()``.  Defaults to 0.9.
        suspicious_threshold: ``risk_score`` at or above which a turn
            receives a weight boost.  Defaults to 0.5.
        suspicious_boost: Extra weight added for suspicious turns.
            Defaults to 2.0.
    """

    def __init__(
        self,
        max_size: int = 10,
        decay_factor: float = _DEFAULT_DECAY,
        suspicious_threshold: float = _SUSPICIOUS_THRESHOLD,
        suspicious_boost: float = _SUSPICIOUS_BOOST,
    ) -> None:
        self._max_size = max_size
        self._decay_factor = decay_factor
        self._suspicious_threshold = suspicious_threshold
        self._suspicious_boost = suspicious_boost
        # Parallel lists: turns and their current weights.
        self._entries: List[Tuple[ConversationTurn, float]] = []

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_weight(
        turn: ConversationTurn,
        threshold: float,
        boost: float,
    ) -> float:
        """Compute the initial weight for a turn."""
        w = _BASE_WEIGHT
        if turn.risk_score >= threshold:
            w += boost
        return w

    def _decay_weights(self) -> None:
        """Apply multiplicative decay to all existing weights."""
        self._entries = [
            (t, w * self._decay_factor) for t, w in self._entries
        ]

    def _evict_lowest(self) -> None:
        """Remove the entry with the smallest weight (oldest wins ties)."""
        if not self._entries:
            return
        min_idx = 0
        min_w = self._entries[0][1]
        for i, (_, w) in enumerate(self._entries[1:], start=1):
            if w < min_w:
                min_w = w
                min_idx = i
        del self._entries[min_idx]

    # ------------------------------------------------------------------
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------

    def add(self, turn: ConversationTurn) -> None:
        """Add a turn, evicting the lowest-weight turn if full.

        Steps:
        1. Decay all existing weights.
        2. Evict the lowest-weight turn if at capacity.
        3. Compute initial weight for the new turn and append it.

        Args:
            turn: The ConversationTurn to append.
        """
        self._decay_weights()
        if len(self._entries) >= self._max_size:
            self._evict_lowest()
        w = self._initial_weight(
            turn, self._suspicious_threshold, self._suspicious_boost
        )
        self._entries.append((turn, w))

    def get_turns(self) -> List[ConversationTurn]:
        """Return all turns in the window, oldest first.

        Returns:
            List of ConversationTurn objects.
        """
        return [t for t, _ in self._entries]

    def get_combined_text(self) -> str:
        """Concatenate all turn texts in the window.

        Returns:
            Single string with turn texts joined by newlines.
        """
        return "\n".join(t.text for t, _ in self._entries)

    def get_risk_scores(self) -> List[float]:
        """Return the risk score for each turn in the window.

        Returns:
            List of float risk scores.
        """
        return [t.risk_score for t, _ in self._entries]

    def get_weights(self) -> List[float]:
        """Return the current weight for each turn in the window.

        Returns:
            List of weight floats, one per turn.
        """
        return [w for _, w in self._entries]

    @property
    def is_full(self) -> bool:
        """True if the window has reached max_size."""
        return len(self._entries) == self._max_size

    @property
    def size(self) -> int:
        """Number of turns currently in the window."""
        return len(self._entries)

    @property
    def max_size(self) -> int:
        """Maximum capacity of the window."""
        return self._max_size
