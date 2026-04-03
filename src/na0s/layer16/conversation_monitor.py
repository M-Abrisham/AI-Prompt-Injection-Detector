"""Layer 16 ConversationSecurityMonitor -- main entry point.  # LAYER16

Orchestrates session management and multi-turn detection.  Runs AFTER
single-turn ``scan()`` completes.  Detectors are imported with graceful
degradation so missing deps do not break the monitor.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from na0s.layer16 import config as layer16_config
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

try:
    from na0s.layer16.detectors.context_poisoning import ContextPoisoningDetector
except ImportError:  # pragma: no cover
    ContextPoisoningDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.layer16.detectors.stylometry import BehavioralStylometryDetector
except ImportError:  # pragma: no cover
    BehavioralStylometryDetector = None  # type: ignore[misc,assignment]


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
        # Alert dedup: session_id -> {alert_type -> (last_turn_fired, confidence)}
        self._last_alert_turn: Dict[str, Dict[str, tuple]] = {}
        # Last deduped alerts per session (for consumers that want filtered view)
        self._last_deduped: Dict[str, List[Alert]] = {}

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
        if ContextPoisoningDetector is not None:
            detectors.append(ContextPoisoningDetector())
        if BehavioralStylometryDetector is not None:
            detectors.append(BehavioralStylometryDetector())
        return detectors

    # ------------------------------------------------------------------
    # Alert deduplication
    # ------------------------------------------------------------------

    def _dedup_alerts(
        self,
        alerts: List[Alert],
        session_id: str,
        current_turn: int,
    ) -> List[Alert]:
        """Filter duplicate alerts within the suppression window.

        An alert is suppressed when the same ``alert_type`` fired within
        the last ``ALERT_SUPPRESSION_TURNS`` turns **and** the new
        confidence is not significantly higher (< 0.15 increase).
        """
        session_history = self._last_alert_turn.setdefault(session_id, {})
        passed: List[Alert] = []
        suppression_window = layer16_config.ALERT_SUPPRESSION_TURNS

        for alert in alerts:
            prev = session_history.get(alert.alert_type)
            if prev is not None:
                prev_turn, prev_conf = prev
                within_window = (current_turn - prev_turn) <= suppression_window
                confidence_jump = alert.confidence - prev_conf
                if within_window and confidence_jump < 0.15:
                    # Suppress — same type, recent, no significant escalation
                    continue
            passed.append(alert)
            session_history[alert.alert_type] = (current_turn, alert.confidence)

        return passed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_new_alerts(self, session_id: str) -> List[Alert]:
        """Return the deduped (non-suppressed) alerts from the last turn.

        When ``ENABLE_ALERT_DEDUP`` is active, this returns only alerts
        that were not suppressed by the dedup filter.  When disabled,
        this returns the same list as ``state.active_alerts``.
        """
        return list(self._last_deduped.get(session_id, []))

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

        # 1. Get or auto-create session state (thread-safe via SessionManager's lock)
        state = self._session_mgr.get_session(session_id)
        if state is None:
            # Auto-create: use lock-protected path to avoid TOCTOU race
            now = datetime.now(timezone.utc)
            new_state = ConversationState(
                session_id=session_id,
                created_at=now,
                last_activity=now,
            )
            with self._session_mgr._lock:
                # Re-check under lock — another thread may have created it
                existing = self._session_mgr._sessions.get(session_id)
                if existing is None:
                    self._session_mgr._sessions[session_id] = new_state
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

        # 4. Update cumulative state (always stores full alert set)
        state.active_alerts = all_alerts

        # 4b. Alert deduplication — tracks repeated alerts and stores a
        #     filtered view while keeping state.active_alerts intact.
        if layer16_config.ENABLE_ALERT_DEDUP:
            self._last_deduped[session_id] = self._dedup_alerts(
                all_alerts, session_id, state.turn_count,
            )
        else:
            self._last_deduped[session_id] = list(all_alerts)

        # 5. Build the analysis result
        # Detection flags and recommendation use the full (unfiltered) state
        # so callers always see the current threat picture.  The ``alerts``
        # list in the returned analysis is deduped to reduce noise.
        risk_trend = get_risk_trend(state)
        active = state.active_alerts
        escalation_detected = any(
            a.alert_type == "escalation" for a in active
        )
        payload_assembly_detected = any(
            a.alert_type == "payload_assembly" for a in active
        )
        fabricated_history_detected = any(
            a.alert_type == "fabricated_history" for a in active
        )
        context_poisoning_detected = any(
            a.alert_type == "context_poisoning" for a in active
        )

        # Recommendation logic uses full state alerts
        recommendation = _compute_recommendation(active)

        return MultiTurnAnalysis(
            session_id=session_id,
            turn_count=state.turn_count,
            escalation_detected=escalation_detected,
            escalation_score=max(
                (a.confidence for a in active if a.alert_type == "escalation"),
                default=0.0,
            ),
            payload_assembly_detected=payload_assembly_detected,
            context_poisoning_detected=context_poisoning_detected,
            fabricated_history_detected=fabricated_history_detected,
            cumulative_risk=state.cumulative_risk,
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
        context_poisoning_detected = any(
            a.alert_type == "context_poisoning" for a in all_alerts
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
            context_poisoning_detected=context_poisoning_detected,
            fabricated_history_detected=fabricated_history_detected,
            cumulative_risk=state.cumulative_risk,
            risk_trend=risk_trend,
            alerts=all_alerts,
            recommendation=recommendation,
        )

        # Remove the session and dedup tracking
        self._last_alert_turn.pop(session_id, None)
        self._last_deduped.pop(session_id, None)
        self._session_mgr.expire_session(session_id)
        return analysis

    def cleanup(self) -> int:
        """Remove all expired sessions.

        Returns:
            Count of sessions removed.
        """
        return self._session_mgr.cleanup_expired()
