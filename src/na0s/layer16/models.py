"""Layer 16 shared data models — interface contract.

ARCHITECTURE DECISION: Post-Processor Pattern.
Layer 16 runs after single-turn detection. This file is the SINGLE
SOURCE OF TRUTH for Layer 16 types. All agents import from here.

Existing Na0S types used:
- ScanResult (from na0s.scan_result) — extended with multi-turn fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Alert:
    """A multi-turn security alert."""

    alert_type: str  # "escalation" | "payload_assembly" | "context_poisoning" | "fabricated_history"
    severity: str  # "low" | "medium" | "high" | "critical"
    confidence: float
    description: str
    turn_range: Tuple[int, int] = (0, 0)  # which turns triggered this
    evidence: List[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    turn_id: str
    text: str
    role: str = "user"  # "user" | "assistant" | "system"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    risk_score: float = 0.0
    label: str = "safe"  # from single-turn ScanResult
    flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    """Accumulated state for a conversation session."""

    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    cumulative_risk: float = 0.0
    active_alerts: List[Alert] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_activity: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def is_empty(self) -> bool:
        return len(self.turns) == 0


@dataclass
class MultiTurnAnalysis:
    """Result of multi-turn analysis on a conversation."""

    session_id: str
    turn_count: int = 0
    escalation_detected: bool = False
    escalation_score: float = 0.0
    payload_assembly_detected: bool = False
    assembled_payload: Optional[str] = None
    context_poisoning_detected: bool = False
    poisoning_details: Optional[str] = None
    fabricated_history_detected: bool = False
    cumulative_risk: float = 0.0
    risk_trend: List[float] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    recommendation: str = "continue_monitoring"  # "continue_monitoring" | "flag" | "block"

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "escalation_detected": self.escalation_detected,
            "escalation_score": self.escalation_score,
            "payload_assembly_detected": self.payload_assembly_detected,
            "assembled_payload": self.assembled_payload,
            "context_poisoning_detected": self.context_poisoning_detected,
            "fabricated_history_detected": self.fabricated_history_detected,
            "cumulative_risk": self.cumulative_risk,
            "risk_trend": self.risk_trend,
            "recommendation": self.recommendation,
            "alerts": [
                {
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "confidence": a.confidence,
                    "description": a.description,
                    "turn_range": list(a.turn_range),
                    "evidence": a.evidence,
                }
                for a in self.alerts
            ],
        }


@dataclass
class SessionConfig:
    """Configuration for session management."""

    window_size: int = 10  # sliding window of N turns
    ttl_seconds: int = 1800  # 30 min inactivity timeout
    escalation_threshold: float = 0.15  # min risk slope per turn
    escalation_min_turns: int = 3  # need at least N turns
    assembly_threshold: float = 0.8  # combined-text risk threshold
    assembly_window: int = 5  # concatenate last N turns
    fabricated_turn_markers: int = 6  # min User:/Assistant: pairs
    storage_backend: str = "memory"  # "memory" | "sqlite" | "redis"
    max_sessions: int = 10000
