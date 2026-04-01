"""Layer 16 ConversationSecurityMonitor -- main entry point.  # LAYER16

Orchestrates session management and multi-turn detection.  Runs AFTER
single-turn ``scan()`` completes.  Detectors are imported with graceful
degradation so missing deps do not break the monitor.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from na0s.layer16.exceptions import SessionNotFoundError
from na0s.layer16.models import (
    Alert,
    ConversationState,
    MultiTurnAnalysis,
    SessionConfig,
)
from na0s.layer16.session_manager import SessionManager
from na0s.layer16.state import add_turn, get_risk_trend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful detector imports  (LAYER16)
# ---------------------------------------------------------------------------

try:
    from na0s.layer16.detectors.escalation import EscalationDetector
except ImportError:  # pragma: no cover
    EscalationDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.layer16.detectors.payload_splitting import PayloadSplittingDetector
except ImportError:  # pragma: no cover
    PayloadSplittingDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.layer16.detectors.fabricated_history import FabricatedHistoryDetector
except ImportError:  # pragma: no cover
    FabricatedHistoryDetector = None  # type: ignore[misc,assignment]


def _compute_recommendation(alerts: List[Alert]) -> str:
    """Derive a recommendation string from a list of alerts.

    - No alerts                -> "continue_monitoring"
    - Any HIGH or CRITICAL     -> "block"
    - Any MEDIUM               -> "flag"
    - Only LOW                 -> "continue_monitoring"
    """
    if not alerts:
        return "continue_monitoring"
    severities = {a.severity for a in alerts}
    if severities & {"high", "critical"}:
        return "block"
    if "medium" in severities:
        return "flag"
    return "continue_monitoring"


class ConversationSecurityMonitor:
    """Main orchestrator for Layer 16 multi-turn detection.

    Creates a :class:`SessionManager` and instantiates all available
    detectors.  The :meth:`process_turn` method is the primary entry
    point: it records a turn, runs detectors, and returns a
    :class:`MultiTurnAnalysis`.

    Args:
        config: Optional :class:`SessionConfig` overrides.
    """

    def __init__(self, config: Optional[SessionConfig] = None) -> None:
        self._config = config or SessionConfig()
        self._session_mgr = SessionManager(config=self._config)
        self._detectors = self._init_detectors()

    # ------------------------------------------------------------------
    # Detector initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_detectors() -> list:
        """Instantiate all available detectors.  Missing deps are skipped."""
        detectors = []
        if EscalationDetector is not None:
            detectors.append(EscalationDetector())
        if PayloadSplittingDetector is not None:
            detectors.append(PayloadSplittingDetector())
        if FabricatedHistoryDetector is not None:
            detectors.append(FabricatedHistoryDetector())
        return detectors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, **metadata: object) -> str:
        """Create a new conversation session.

        Args:
            **metadata: Arbitrary metadata stored on the session state.

        Returns:
            A new session_id (uuid4 string).
        """
        sid = self._session_mgr.create_session()
        if metadata:
            state = self._session_mgr.get_session(sid)
            if state is not None:
                state.metadata.update(metadata)
        return sid

    def process_turn(
        self,
        text: str,
        session_id: str,
        single_turn_result: Optional[Dict] = None,
        risk_score: float = 0.0,
        label: str = "safe",
        flags: Optional[List[str]] = None,
    ) -> MultiTurnAnalysis:
        """Record a turn and run all multi-turn detectors.

        If ``session_id`` does not exist, a new session is created
        automatically (auto-create semantics).

        Args:
            text: The raw user/turn text.
            session_id: The conversation session identifier.
            single_turn_result: Optional dict from ``ScanResult.to_dict()``.
                If provided, ``risk_score``, ``label``, and ``flags`` are
                extracted from it (explicit kwargs still take precedence
                when they differ from the defaults).
            risk_score: Risk score from single-turn ``scan()``.
            label: Classification label from single-turn ``scan()``.
            flags: Optional list of flag strings from single-turn analysis.

        Returns:
            A :class:`MultiTurnAnalysis` with aggregated alerts and
            risk trend.
        """
        # Extract fields from single_turn_result when provided
        if single_turn_result is not None:
            if risk_score == 0.0 and "risk_score" in single_turn_result:
                risk_score = float(single_turn_result["risk_score"])
            if label == "safe" and "label" in single_turn_result:
                label = single_turn_result["label"]
            if flags is None and "technique_tags" in single_turn_result:
                flags = list(single_turn_result["technique_tags"])

        # 1. Get or auto-create session state
        state = self._session_mgr.get_session(session_id)
        if state is None:
            # Auto-create the session so callers don't need to call
            # create_session() explicitly.
            self._session_mgr._sessions[session_id] = ConversationState(
                session_id=session_id,
                created_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
            )
            state = self._session_mgr.get_session(session_id)

        # 2. Add the turn
        add_turn(state, text, risk_score, label, flags)

        # 3. Run all detectors
        all_alerts: List[Alert] = []
        for detector in self._detectors:
            try:
                alerts = detector.analyze(state)
                all_alerts.extend(alerts)
            except Exception:
                logger.warning(
                    "Detector %s failed on session %s",
                    getattr(detector, "detector_name", "unknown"),
                    session_id,
                    exc_info=True,
                )

        # 4. Update cumulative state
        state.active_alerts = all_alerts

        # 5. Build the analysis result
        risk_trend = get_risk_trend(state)
        escalation_detected = any(
            a.alert_type == "escalation" for a in all_alerts
        )
        payload_assembly_detected = any(
            a.alert_type == "payload_assembly" for a in all_alerts
        )
        fabricated_history_detected = any(
            a.alert_type == "fabricated_history" for a in all_alerts
        )

        # Recommendation logic:
        #   HIGH or CRITICAL -> block
        #   MEDIUM           -> flag
        #   LOW only (or no alerts) -> continue_monitoring
        recommendation = _compute_recommendation(all_alerts)

        return MultiTurnAnalysis(
            session_id=session_id,
            turn_count=state.turn_count,
            escalation_detected=escalation_detected,
            escalation_score=max(
                (a.confidence for a in all_alerts if a.alert_type == "escalation"),
                default=0.0,
            ),
            payload_assembly_detected=payload_assembly_detected,
            context_poisoning_detected=False,  # reserved for future detector
            fabricated_history_detected=fabricated_history_detected,
            risk_trend=risk_trend,
            alerts=all_alerts,
            recommendation=recommendation,
        )

    def get_session_summary(self, session_id: str) -> dict:
        """Return a summary dict for a session.

        Args:
            session_id: The session to summarize.

        Returns:
            A dict with session metadata and metrics.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        state = self._session_mgr.get_session(session_id)
        if state is None:
            raise SessionNotFoundError(
                f"Session {session_id!r} not found or expired"
            )
        risk_trend = get_risk_trend(state)
        return {
            "session_id": session_id,
            "turn_count": state.turn_count,
            "cumulative_risk": state.cumulative_risk,
            "risk_trend": risk_trend,
            "active_alerts": len(state.active_alerts),
            "created_at": state.created_at.isoformat(),
            "last_activity": state.last_activity.isoformat(),
            "metadata": dict(state.metadata),
        }

    def end_session(self, session_id: str) -> MultiTurnAnalysis:
        """End a session and return a final analysis.

        Runs all detectors one last time, then removes the session
        from the store.

        Args:
            session_id: The session to end.

        Returns:
            Final :class:`MultiTurnAnalysis`.
        """
        state = self._session_mgr.get_session(session_id)
        if state is None:
            # Return an empty analysis for missing/expired sessions
            return MultiTurnAnalysis(session_id=session_id)

        # Run detectors one final time
        all_alerts: List[Alert] = []
        for detector in self._detectors:
            try:
                alerts = detector.analyze(state)
                all_alerts.extend(alerts)
            except Exception:
                logger.warning(
                    "Detector %s failed during end_session for %s",
                    getattr(detector, "detector_name", "unknown"),
                    session_id,
                    exc_info=True,
                )

        risk_trend = get_risk_trend(state)
        escalation_detected = any(
            a.alert_type == "escalation" for a in all_alerts
        )
        payload_assembly_detected = any(
            a.alert_type == "payload_assembly" for a in all_alerts
        )
        fabricated_history_detected = any(
            a.alert_type == "fabricated_history" for a in all_alerts
        )

        recommendation = _compute_recommendation(all_alerts)

        analysis = MultiTurnAnalysis(
            session_id=session_id,
            turn_count=state.turn_count,
            escalation_detected=escalation_detected,
            escalation_score=max(
                (a.confidence for a in all_alerts if a.alert_type == "escalation"),
                default=0.0,
            ),
            payload_assembly_detected=payload_assembly_detected,
            context_poisoning_detected=False,
            fabricated_history_detected=fabricated_history_detected,
            risk_trend=risk_trend,
            alerts=all_alerts,
            recommendation=recommendation,
        )

        # Remove the session
        self._session_mgr.expire_session(session_id)
        return analysis

    def cleanup(self) -> int:
        """Remove all expired sessions.

        Returns:
            Count of sessions removed.
        """
        return self._session_mgr.cleanup_expired()
