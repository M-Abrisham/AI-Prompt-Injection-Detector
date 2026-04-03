"""Layer 16 SlidingWindow — bounded turn history with weighted eviction.

Suspicious turns (high risk_score) receive higher weight and persist
longer in the window.  Each time a new turn is added, existing weights
decay by a configurable factor.  When the window is full the turn with
the *lowest* weight is evicted, so benign turns are dropped first.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

from na0s.layer16.models import ConversationTurn

# Default weight parameters
_DEFAULT_DECAY = 0.9  # per-turn multiplicative decay
_SUSPICIOUS_THRESHOLD = 0.5  # risk_score above this gets a weight boost
_BASE_WEIGHT = 1.0
_SUSPICIOUS_BOOST = 2.0  # extra weight for suspicious turns

_EVICTION_LOG_CAP = 50


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
        min_weight: Weight floor for suspicious turns after decay.
            Defaults to 0.1.
    """

    def __init__(
        self,
        max_size: int = 10,
        decay_factor: float = _DEFAULT_DECAY,
        suspicious_threshold: float = _SUSPICIOUS_THRESHOLD,
        suspicious_boost: float = _SUSPICIOUS_BOOST,
        min_weight: float = 0.1,
    ) -> None:
        # --- Input validation (T1.6) ---
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError("decay_factor must be in (0.0, 1.0]")
        if suspicious_threshold < 0.0:
            raise ValueError("suspicious_threshold must be >= 0.0")
        if suspicious_boost < 0.0:
            raise ValueError("suspicious_boost must be >= 0.0")

        self._max_size = max_size
        self._decay_factor = decay_factor
        self._suspicious_threshold = suspicious_threshold
        self._suspicious_boost = suspicious_boost
        self._min_weight = min_weight
        # Parallel lists: turns and their current weights.
        self._entries: List[Tuple[ConversationTurn, float]] = []
        # Eviction summary log (T1.2)
        self._eviction_log: List[dict] = []

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
        """Apply multiplicative decay to all existing weights.

        Suspicious turns (risk_score >= suspicious_threshold) are clamped
        to ``min_weight`` so they cannot be evicted before fresh benign
        turns after many decay cycles (T1.7).
        """
        new_entries: List[Tuple[ConversationTurn, float]] = []
        for t, w in self._entries:
            decayed = w * self._decay_factor
            if t.risk_score >= self._suspicious_threshold:
                decayed = max(decayed, self._min_weight)
            new_entries.append((t, decayed))
        self._entries = new_entries

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

        # Save eviction summary before deleting (T1.2)
        evicted_turn, evicted_weight = self._entries[min_idx]
        self._eviction_log.append({
            "turn_id": evicted_turn.turn_id,
            "risk_score": evicted_turn.risk_score,
            "label": evicted_turn.label,
            "flags": list(evicted_turn.flags),
            "evicted_weight": evicted_weight,
        })
        # Cap eviction log at _EVICTION_LOG_CAP (FIFO)
        if len(self._eviction_log) > _EVICTION_LOG_CAP:
            self._eviction_log = self._eviction_log[-_EVICTION_LOG_CAP:]

        del self._entries[min_idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, turn: ConversationTurn) -> None:
        """Add a turn, evicting the lowest-weight turn if full.

        Steps:
        1. Decay all existing weights.
        2. Evict the lowest-weight turn if at capacity.
        3. Compute initial weight for the new turn and append it.

        Args:
            turn: The ConversationTurn to append.

        Raises:
            TypeError: If *turn* is not a ConversationTurn instance.
        """
        if not isinstance(turn, ConversationTurn):
            raise TypeError(
                f"turn must be a ConversationTurn, got {type(turn).__name__}"
            )
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
            Single string with turn texts joined by ``\\n---\\n``.
        """
        return "\n---\n".join(t.text for t, _ in self._entries)

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

    def get_eviction_log(self) -> List[dict]:
        """Return a copy of the eviction log.

        Returns:
            List of dicts, each with keys ``turn_id``, ``risk_score``,
            ``label``, ``flags``, and ``evicted_weight``.
        """
        return list(self._eviction_log)

    def get_aggregate_risk(self) -> float:
        """Weight-adjusted mean risk score across all turns in the window.

        Returns:
            ``sum(risk * weight) / sum(weight)``, or 0.0 if empty.
        """
        if not self._entries:
            return 0.0
        total_weighted = sum(t.risk_score * w for t, w in self._entries)
        total_weight = sum(w for _, w in self._entries)
        if total_weight == 0.0:
            return 0.0
        return total_weighted / total_weight

    def get_peak_risk(self) -> float:
        """Maximum risk_score among all turns in the window.

        Returns:
            Max risk score, or 0.0 if empty.
        """
        if not self._entries:
            return 0.0
        return max(t.risk_score for t, _ in self._entries)

    def clear(self) -> None:
        """Remove all turns and clear the eviction log."""
        self._entries.clear()
        self._eviction_log.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pythonic API (T1.8)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[ConversationTurn]:
        for t, _ in self._entries:
            yield t
