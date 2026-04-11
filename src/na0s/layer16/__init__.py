"""Layer 16 — Multi-Turn Detection.

Adds conversation-level memory and stateful analysis to Na0S. Detects
multi-turn attacks where adversaries spread payloads across messages,
gradually escalate, plant context in early turns, or fabricate history.

ARCHITECTURE DECISION: Post-Processor Pattern (Option C).
Layer 16 runs AFTER single-turn scan() completes. When a session_id
is provided, it records the turn, runs multi-turn detectors on the
accumulated conversation state, and merges alerts into ScanResult.
The existing stateless API is unchanged — session_id is optional.

Usage::

    from na0s.layer16 import ConversationSecurityMonitor

    monitor = ConversationSecurityMonitor()
    session_id = monitor.create_session()
    analysis = monitor.process_turn("user message", session_id=session_id)
    if analysis.has_alerts:
        for alert in analysis.alerts:
            print(f"{alert.alert_type}: {alert.severity}")
"""

# Conversation monitor has optional/runtime-only dependencies in some builds.
# Keep package imports resilient so tests can import models/config in isolation.
try:  # pragma: no cover - exercised indirectly in integration tests
    from na0s.layer16.conversation_monitor import ConversationSecurityMonitor
except Exception:  # pragma: no cover
    ConversationSecurityMonitor = None  # type: ignore[misc,assignment]
from na0s.layer16.exceptions import (
    MaxSessionsReachedError,
    SessionExpiredError,
    SessionNotFoundError,
)
from na0s.layer16.models import (
    Alert,
    ConversationState,
    ConversationTurn,
    MultiTurnAnalysis,
    SessionConfig,
    ThreatLevel,
    UserRiskProfile,
)
from na0s.layer16.session_manager import SessionManager
from na0s.layer16.sliding_window import SlidingWindow

__all__ = [
    "Alert",
    "ConversationState",
    "ConversationTurn",
    "MaxSessionsReachedError",
    "MultiTurnAnalysis",
    "SessionConfig",
    "SessionExpiredError",
    "SessionManager",
    "SessionNotFoundError",
    "SlidingWindow",
    "ThreatLevel",
    "UserRiskProfile",
]

if ConversationSecurityMonitor is not None:
    __all__.insert(1, "ConversationSecurityMonitor")
