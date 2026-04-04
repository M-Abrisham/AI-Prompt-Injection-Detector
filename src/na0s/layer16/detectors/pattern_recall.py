"""Cross-session attack pattern recall detector (T3.4).

Checks new turns against a Counting Bloom Filter of previously seen
attack n-grams.  After a turn is flagged (risk_score >= 0.5) its
n-grams are recorded so future sessions can detect returning attackers.

Privacy-preserving: no raw text stored, only hash-derived counters.
"""

from __future__ import annotations

import threading
from typing import List

from na0s.layer16 import config as layer16_config
from na0s.layer16.attack_pattern_filter import AttackPatternStore
from na0s.layer16.models import Alert, ConversationState
from na0s.layer16.detectors.base_detector import MultiTurnDetector


class PatternRecallDetector(MultiTurnDetector):
    """Detect previously seen attack patterns via Bloom filter lookup.

    A class-level AttackPatternStore is shared across all instances and
    sessions, enabling cross-session pattern recall.
    """

    # Class-level store -- shared across all instances/sessions.
    _shared_store: AttackPatternStore | None = None
    _store_lock: threading.Lock = threading.Lock()
    _RECORDING_THRESHOLD = 0.5  # record n-grams for turns at or above this risk

    def __init__(self) -> None:
        if PatternRecallDetector._shared_store is None:
            with PatternRecallDetector._store_lock:
                if PatternRecallDetector._shared_store is None:
                    PatternRecallDetector._shared_store = AttackPatternStore()

    # ------------------------------------------------------------------
    # MultiTurnDetector interface
    # ------------------------------------------------------------------

    @property
    def detector_name(self) -> str:
        return "PatternRecallDetector"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["T3.4"]

    def reset(self) -> None:
        """Reset the shared store (mainly for testing)."""
        PatternRecallDetector._shared_store = AttackPatternStore()

    def analyze(self, state: ConversationState) -> List[Alert]:
        """Check latest turn against bloom filter; record flagged turns."""
        if not layer16_config.ENABLE_PATTERN_RECALL:
            return []

        if state.is_empty:
            return []

        store = PatternRecallDetector._shared_store
        if store is None:
            return []

        alerts: List[Alert] = []
        latest = state.turns[-1]
        threshold = layer16_config.PATTERN_RECALL_THRESHOLD

        # Check latest turn against known attack patterns
        match_score = store.get_match_score(latest.text)

        if match_score >= threshold:
            severity = "medium" if match_score < 0.6 else "high"
            confidence = min(1.0, match_score)
            alerts.append(
                Alert(
                    alert_type="pattern_recall",
                    severity=severity,
                    confidence=confidence,
                    description=(
                        f"Turn matches {match_score:.0%} of previously seen "
                        f"attack patterns (threshold: {threshold:.0%})"
                    ),
                    turn_range=(state.turn_count - 1, state.turn_count - 1),
                    evidence=[f"match_score={match_score:.4f}"],
                )
            )

        # Record n-grams for any turn flagged as risky
        if latest.risk_score >= self._RECORDING_THRESHOLD:
            store.record_attack_ngrams(latest.text)

        return alerts
