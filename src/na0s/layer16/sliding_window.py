"""Layer 16 SlidingWindow — bounded turn history with weighted eviction.

Suspicious turns (high risk_score) receive higher weight and persist
longer in the window.  Each time a new turn is added, existing weights
decay by a configurable factor.  When the window is full the turn with
the *lowest* weight is evicted, so benign turns are dropped first.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

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
        compressed_history: Optional[Any] = None,
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
        self._compressed_history = compressed_history
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

        # Flow evicted turn to warm-tier compressed history (T3.5)
        if self._compressed_history is not None:
            self._compressed_history.record_eviction(evicted_turn)

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

    def get_full_context(self) -> Dict[str, Any]:
        """Return both hot (window turns) and warm (compressed summaries) data.

        Returns:
            Dict with keys:
            - hot_turns: list of ConversationTurn in the current window
            - hot_risk_scores: list of risk scores in the window
            - warm_summaries: list of TurnSummary (empty if no compressed history)
            - warm_historical_risk: float, weighted avg risk from warm tier
            - warm_technique_history: dict of technique tag counts
            - total_turns_seen: hot + warm turn count
        """
        hot_turns = self.get_turns()
        hot_risk = self.get_risk_scores()

        if self._compressed_history is not None:
            warm_summaries = self._compressed_history.get_summaries()
            warm_risk = self._compressed_history.get_historical_risk()
            warm_techniques = self._compressed_history.get_technique_history()
            warm_turn_count = self._compressed_history.total_turns_compressed
        else:
            warm_summaries = []
            warm_risk = 0.0
            warm_techniques = {}
            warm_turn_count = 0

        return {
            "hot_turns": hot_turns,
            "hot_risk_scores": hot_risk,
            "warm_summaries": warm_summaries,
            "warm_historical_risk": warm_risk,
            "warm_technique_history": warm_techniques,
            "total_turns_seen": len(hot_turns) + warm_turn_count,
        }

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
    # Burst detection (T2.3)
    # ------------------------------------------------------------------

    def detect_burst(
        self,
        n: int = 3,
        window: int = 5,
        threshold: float | None = None,
    ) -> bool:
        """Return True if at least *n* of the last *window* turns are suspicious.

        A turn is "suspicious" if its risk_score >= self._suspicious_threshold
        (or >= *threshold* if provided).

        Args:
            n: Minimum number of suspicious turns required.
            window: How many recent turns to examine.
            threshold: Override for suspicious_threshold. If None, uses
                the instance's suspicious_threshold.

        Returns:
            True if a burst is detected.

        Raises:
            ValueError: If *n* < 1, *window* < 1, or *n* > *window*.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        if n > window:
            raise ValueError("n must be <= window")

        if not self._entries:
            return False

        thresh = threshold if threshold is not None else self._suspicious_threshold
        recent = self._entries[-min(window, len(self._entries)):]
        count = sum(1 for t, _ in recent if t.risk_score >= thresh)
        return count >= n

    def get_burst_info(
        self,
        window: int = 5,
        threshold: float | None = None,
    ) -> dict:
        """Return burst statistics for the last *window* turns.

        Args:
            window: How many recent turns to examine.
            threshold: Override for suspicious_threshold. If None, uses
                the instance's suspicious_threshold.

        Returns:
            Dict with keys:
            - suspicious_count: number of suspicious turns in window
            - total_in_window: actual number of turns examined
            - burst_ratio: suspicious_count / total_in_window
            - max_risk_in_window: peak risk in the window
            - is_burst: True if suspicious_count >= 3
        """
        if window < 1:
            raise ValueError("window must be >= 1")

        thresh = threshold if threshold is not None else self._suspicious_threshold
        actual = min(window, len(self._entries))
        recent = self._entries[-actual:] if actual > 0 else []

        suspicious_count = sum(1 for t, _ in recent if t.risk_score >= thresh)
        total_in_window = len(recent)
        burst_ratio = suspicious_count / total_in_window if total_in_window > 0 else 0.0
        max_risk = max((t.risk_score for t, _ in recent), default=0.0)

        return {
            "suspicious_count": suspicious_count,
            "total_in_window": total_in_window,
            "burst_ratio": burst_ratio,
            "max_risk_in_window": max_risk,
            "is_burst": suspicious_count >= 3,
        }

    # ------------------------------------------------------------------
    # Pythonic API (T1.8)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[ConversationTurn]:
        for t, _ in self._entries:
            yield t
