"""Layer 16 Two-Tier Memory — Hot + Warm compressed history.

When the SlidingWindow evicts turns from the hot tier, they flow into
CompressedHistory (the warm tier).  Evicted turns are batched and
compressed into TurnSummary objects so detectors can query historical
context without growing the hot window.

Architecture: MemGPT/Letta-inspired hot + warm memory tiers.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from na0s.layer16.config import ENABLE_WARM_MEMORY
from na0s.layer16.models import ConversationTurn

# Maximum text snippet length stored per summary (security bound).
_MAX_SNIPPET_LENGTH = 100


@dataclass
class TurnSummary:
    """Compressed representation of a batch of evicted turns."""

    batch_id: str  # uuid4
    turn_count: int
    avg_risk_score: float
    max_risk_score: float
    dominant_label: str  # most common label in batch
    technique_tags: Set[str]  # union of all flags
    text_snippet: str  # first 100 chars of highest-risk turn (for context)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class CompressedHistory:
    """Warm-tier memory storing compressed summaries of evicted turn batches.

    When the SlidingWindow evicts turns, they're compressed into TurnSummary
    objects here.  Detectors can query the warm tier for historical context
    without growing the hot window.

    Capacity: max_summaries (default 10) x batch_size (default 5) = ~50 turns
    of history beyond the hot window.
    """

    def __init__(
        self, max_summaries: int = 10, batch_size: int = 5
    ) -> None:
        if max_summaries < 1:
            raise ValueError("max_summaries must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._summaries: List[TurnSummary] = []
        self._pending: List[ConversationTurn] = []
        self._max_summaries = max_summaries
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_eviction(self, turn: ConversationTurn) -> Optional[TurnSummary]:
        """Record an evicted turn.  Returns a TurnSummary when batch_size reached."""
        if not ENABLE_WARM_MEMORY:
            return None
        if not isinstance(turn, ConversationTurn):
            raise TypeError(
                f"turn must be a ConversationTurn, got {type(turn).__name__}"
            )
        self._pending.append(turn)
        if len(self._pending) >= self._batch_size:
            summary = self._compress_batch(self._pending)
            self._pending.clear()
            self._summaries.append(summary)
            if len(self._summaries) > self._max_summaries:
                self._summaries.pop(0)  # FIFO eviction of oldest summary
            return summary
        return None

    def get_historical_risk(self) -> float:
        """Weighted average risk across all summaries (recent weighted higher).

        Uses linearly increasing weights: oldest summary gets weight 1,
        newest gets weight N.  Returns 0.0 if no summaries exist.
        """
        if not self._summaries:
            return 0.0
        total_weighted = 0.0
        total_weight = 0.0
        for i, summary in enumerate(self._summaries):
            weight = float(i + 1)  # 1, 2, 3, ...
            total_weighted += summary.avg_risk_score * weight
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return total_weighted / total_weight

    def get_technique_history(self) -> Dict[str, int]:
        """Aggregate technique tags across all summaries."""
        counts: Dict[str, int] = {}
        for summary in self._summaries:
            for tag in summary.technique_tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def get_summaries(self) -> List[TurnSummary]:
        """Return all stored summaries, oldest first."""
        return list(self._summaries)

    def has_historical_risk(self, threshold: float = 0.5) -> bool:
        """Check if any summary batch had high risk (max_risk_score >= threshold)."""
        return any(s.max_risk_score >= threshold for s in self._summaries)

    @property
    def total_turns_compressed(self) -> int:
        """Total number of turns represented in warm storage."""
        return sum(s.turn_count for s in self._summaries)

    def clear(self) -> None:
        """Clear all summaries and pending turns."""
        self._summaries.clear()
        self._pending.clear()

    @property
    def pending_count(self) -> int:
        """Number of evicted turns waiting to be batched."""
        return len(self._pending)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compress_batch(self, turns: List[ConversationTurn]) -> TurnSummary:
        """Compress a batch of turns into a single summary."""
        if not turns:
            raise ValueError("Cannot compress empty batch")

        risk_scores = [t.risk_score for t in turns]
        avg_risk = sum(risk_scores) / len(risk_scores)
        max_risk = max(risk_scores)

        # Dominant label: most common label in batch
        label_counts = Counter(t.label for t in turns)
        dominant_label = label_counts.most_common(1)[0][0]

        # Union of all flags/technique tags
        all_tags: Set[str] = set()
        for t in turns:
            all_tags.update(t.flags)

        # Text snippet: first 100 chars of highest-risk turn
        highest_risk_turn = max(turns, key=lambda t: t.risk_score)
        snippet = highest_risk_turn.text[:_MAX_SNIPPET_LENGTH]

        return TurnSummary(
            batch_id=str(uuid.uuid4()),
            turn_count=len(turns),
            avg_risk_score=avg_risk,
            max_risk_score=max_risk,
            dominant_label=dominant_label,
            technique_tags=all_tags,
            text_snippet=snippet,
        )
