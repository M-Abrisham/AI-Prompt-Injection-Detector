"""Layer 16 ConversationSecurityMonitor -- main entry point.  # LAYER16

Orchestrates session management and multi-turn detection.  Runs AFTER
single-turn ``scan()`` completes.  Detectors are imported with graceful
degradation so missing deps do not break the monitor.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from na0s.conversation import config as layer16_config
from na0s.conversation.exceptions import SessionNotFoundError
from na0s.conversation.models import (
    Alert,
    MultiTurnAnalysis,
    SessionConfig,
)
from na0s.conversation.graduated_response import compute_threat_level, get_response_action
from na0s.conversation.models import ThreatLevel, UserRiskProfile
from na0s.conversation.session_manager import SessionManager
from na0s.conversation.state import (
    add_turn,
    compute_peak_accumulation,
    get_risk_trend,
)
from na0s.conversation.user_risk_profile import UserRiskProfileStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful detector imports  (LAYER16)
# ---------------------------------------------------------------------------

try:
    from na0s.conversation.detectors.escalation import EscalationDetector
except ImportError:  # pragma: no cover
    EscalationDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.conversation.detectors.payload_splitting import PayloadSplittingDetector
except ImportError:  # pragma: no cover
    PayloadSplittingDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.conversation.detectors.fabricated_history import FabricatedHistoryDetector
except ImportError:  # pragma: no cover
    FabricatedHistoryDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.conversation.detectors.context_poisoning import ContextPoisoningDetector
except ImportError:  # pragma: no cover
    ContextPoisoningDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.conversation.detectors.stylometry import BehavioralStylometryDetector
except ImportError:  # pragma: no cover
    BehavioralStylometryDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.conversation.detectors.embedding_drift import EmbeddingDriftDetector
except ImportError:  # pragma: no cover
    EmbeddingDriftDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.conversation.detectors.cot_compliance import CoTComplianceDetector
except ImportError:  # pragma: no cover
    CoTComplianceDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.conversation.detectors.scheming import SchemingDetector
except ImportError:  # pragma: no cover
    SchemingDetector = None  # type: ignore[misc,assignment]

try:
    from na0s.conversation.detectors.goal_decomposition import GoalDecompositionDetector
except ImportError:  # pragma: no cover
    GoalDecompositionDetector = None  # type: ignore[misc,assignment]


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
        self._process_lock = threading.RLock()
        # Alert dedup: session_id -> {alert_type -> (last_turn_fired, confidence)}
        self._last_alert_turn: Dict[str, Dict[str, tuple]] = {}
        # Last deduped alerts per session (for consumers that want filtered view)
        self._last_deduped: Dict[str, List[Alert]] = {}
        # Cross-session user risk profiles (T3.1)
        self._profile_store = UserRiskProfileStore()

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
        if EmbeddingDriftDetector is not None:
            detectors.append(EmbeddingDriftDetector())
        if CoTComplianceDetector is not None:
            detectors.append(CoTComplianceDetector())
        if SchemingDetector is not None:
            detectors.append(SchemingDetector())
        if GoalDecompositionDetector is not None:
            detectors.append(GoalDecompositionDetector())
        return detectors

    # ------------------------------------------------------------------
    # Co-occurrence gate
    # ------------------------------------------------------------------

    # Alert types that are low-precision on their own and must only surface as
    # corroboration. ``embedding_drift`` (D1.23) alerts on pure consecutive-turn
    # cosine distance with no malice anchor, so it fires on every sharp topic
    # pivot — benign or malicious alike — making it a false-positive generator
    # when standalone. It adds value only as confirming evidence alongside a
    # higher-precision detector firing the same turn.
    _CORROBORATION_ONLY_ALERTS = ("embedding_drift",)

    @classmethod
    def _apply_cooccurrence_gate(cls, alerts: List[Alert]) -> List[Alert]:
        """Drop corroboration-only alerts unless a primary alert co-occurs.

        Keeps every alert when at least one non-corroboration ("primary")
        alert fired in the same turn; otherwise strips the corroboration-only
        alerts so drift never raises an alert on its own.
        """
        primary = [
            a for a in alerts if a.alert_type not in cls._CORROBORATION_ONLY_ALERTS
        ]
        if primary:
            return alerts
        return primary

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
            # Never suppress high or critical severity alerts
            if alert.severity in ("high", "critical"):
                passed.append(alert)
                session_history[alert.alert_type] = (current_turn, alert.confidence)
                continue
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

    def create_session(
        self,
        user_hash: Optional[str] = None,
        **metadata: object,
    ) -> str:
        """Create a new conversation session.

        Args:
            user_hash: Optional opaque hash of the user identifier for
                cross-session risk tracking.
            **metadata: Arbitrary metadata stored on the session state.

        Returns:
            A new session_id (uuid4 string).
        """
        sid = self._session_mgr.create_session()
        state = self._session_mgr.get_session(sid)
        if state is not None:
            if metadata:
                state.metadata.update(metadata)
            if user_hash and layer16_config.ENABLE_USER_RISK_PROFILES:
                state.metadata["user_hash"] = user_hash
                multiplier = self._profile_store.get_risk_multiplier(user_hash)
                state.metadata["risk_multiplier"] = multiplier
        return sid

    def process_turn(
        self,
        text: str,
        session_id: str,
        single_turn_result: Optional[Dict] = None,
        risk_score: float = 0.0,
        label: str = "safe",
        flags: Optional[List[str]] = None,
        user_hash: Optional[str] = None,
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
            user_hash: Optional opaque hash of user identifier for
                cross-session risk tracking.

        Returns:
            A :class:`MultiTurnAnalysis` with aggregated alerts and
            risk trend.
        """
        if not layer16_config.ENABLE_MULTI_TURN:
            return MultiTurnAnalysis(
                session_id=session_id,
                turn_count=0,
            )

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
            self._session_mgr._auto_create_session(session_id)
            state = self._session_mgr.get_session(session_id)

        # Store user_hash in session metadata
        if user_hash and layer16_config.ENABLE_USER_RISK_PROFILES:
            state.metadata.setdefault("user_hash", user_hash)

        with self._process_lock:
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

            # 3b. Co-occurrence gate — drop corroboration-only alerts
            #     (embedding_drift) unless a primary detector also fired this
            #     turn, so semantic drift never alerts standalone.
            all_alerts = self._apply_cooccurrence_gate(all_alerts)

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

            # Graduated threat level (T3.2)
            user_profile = None
            session_user_hash = state.metadata.get("user_hash")
            if session_user_hash and layer16_config.ENABLE_USER_RISK_PROFILES:
                user_profile = self._profile_store.get_profile(session_user_hash)
            threat_level = compute_threat_level(state, active, user_profile)
            response_action = get_response_action(threat_level)

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
                peak_accumulation_score=compute_peak_accumulation(state),
                cusum_score=state.cusum_score,
                risk_trend=risk_trend,
                alerts=all_alerts,
                recommendation=recommendation,
                threat_level=threat_level.value,
                response_action=response_action,
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
            "peak_risk": state.peak_risk,
            "cusum_score": state.cusum_score,
            "peak_accumulation_score": compute_peak_accumulation(state),
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

        # Co-occurrence gate — keep turn-level and session-end behavior in sync.
        all_alerts = self._apply_cooccurrence_gate(all_alerts)

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

        # Graduated threat level (T3.2)
        user_profile = None
        session_user_hash = state.metadata.get("user_hash")
        if session_user_hash and layer16_config.ENABLE_USER_RISK_PROFILES:
            user_profile = self._profile_store.get_profile(session_user_hash)
        threat_level = compute_threat_level(state, all_alerts, user_profile)
        response_action = get_response_action(threat_level)

        # Update cross-session user risk profile (T3.1)
        if session_user_hash and layer16_config.ENABLE_USER_RISK_PROFILES:
            technique_tags = []
            for turn in state.turns:
                technique_tags.extend(turn.flags)
            was_flagged = recommendation in ("flag", "block")
            self._profile_store.update_from_session(
                user_hash=session_user_hash,
                session_risk=state.cumulative_risk,
                technique_tags=technique_tags,
                was_flagged=was_flagged,
            )

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
            peak_accumulation_score=compute_peak_accumulation(state),
            cusum_score=state.cusum_score,
            risk_trend=risk_trend,
            alerts=all_alerts,
            recommendation=recommendation,
            threat_level=threat_level.value,
            response_action=response_action,
        )

        # Remove the session and dedup tracking
        self._last_alert_turn.pop(session_id, None)
        self._last_deduped.pop(session_id, None)
        self._session_mgr.expire_session(session_id)
        return analysis

    def cleanup(self) -> int:
        """Remove all expired sessions and prune stale dedup entries.

        Returns:
            Count of sessions removed.
        """
        removed = self._session_mgr.cleanup_expired()
        # Prune dedup dicts for sessions that no longer exist
        active_ids = set(self._session_mgr._sessions.keys())
        self._last_alert_turn = {
            k: v for k, v in self._last_alert_turn.items() if k in active_ids
        }
        self._last_deduped = {
            k: v for k, v in self._last_deduped.items() if k in active_ids
        }
        return removed
