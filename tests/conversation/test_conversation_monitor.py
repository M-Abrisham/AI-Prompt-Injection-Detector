"""Tests for Layer 16 ConversationSecurityMonitor."""  # LAYER16

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock

from na0s.layer16.conversation_monitor import (
    ConversationSecurityMonitor,
    _compute_recommendation,
)
from na0s.layer16.exceptions import SessionNotFoundError
from na0s.layer16.models import Alert, MultiTurnAnalysis, SessionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(severity: str = "low", alert_type: str = "escalation") -> Alert:
    """Create a test Alert with the given severity."""
    return Alert(
        alert_type=alert_type,
        severity=severity,
        confidence=0.9,
        description=f"test {severity} alert",
        turn_range=(0, 1),
        evidence=["evidence"],
    )


def _mock_detector(alerts: list | None = None):
    """Return a MagicMock that quacks like a MultiTurnDetector."""
    det = MagicMock()
    det.detector_name = "mock_detector"
    det.analyze.return_value = alerts if alerts is not None else []
    return det


class TestCreateSession:
    def test_create_returns_uuid(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 36  # uuid4

    def test_create_with_metadata(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session(user_id="u123", channel="web")
        summary = monitor.get_session_summary(sid)
        assert summary["metadata"]["user_id"] == "u123"
        assert summary["metadata"]["channel"] == "web"


class TestProcessTurn:
    def test_five_turns_returns_analysis(self) -> None:
        """Create a session, process 5 turns, verify analysis object."""
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()

        for i in range(5):
            analysis = monitor.process_turn(
                text=f"Turn number {i}",
                session_id=sid,
                risk_score=i * 0.1,
                label="safe",
            )

        assert isinstance(analysis, MultiTurnAnalysis)
        assert analysis.session_id == sid
        assert analysis.turn_count == 5
        assert len(analysis.risk_trend) == 5
        for actual, expected in zip(analysis.risk_trend, [0.0, 0.1, 0.2, 0.3, 0.4]):
            assert abs(actual - expected) < 1e-9

    def test_escalation_triggers_alert(self) -> None:
        """Escalating risk scores should trigger an escalation alert."""
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()

        scores = [0.1, 0.2, 0.4, 0.6, 0.8]
        analysis = None
        for i, score in enumerate(scores):
            analysis = monitor.process_turn(
                text=f"Escalating turn {i}",
                session_id=sid,
                risk_score=score,
                label="safe" if score < 0.5 else "malicious",
            )

        assert analysis is not None
        assert analysis.escalation_detected is True
        assert any(a.alert_type == "escalation" for a in analysis.alerts)

    def test_nonexistent_session_auto_creates(self) -> None:
        """Processing a turn on a non-existent session auto-creates it."""
        monitor = ConversationSecurityMonitor()
        fake_sid = "non-existent-session-id-00000"

        analysis = monitor.process_turn(
            text="hello",
            session_id=fake_sid,
            risk_score=0.1,
        )

        assert isinstance(analysis, MultiTurnAnalysis)
        assert analysis.session_id == fake_sid
        assert analysis.turn_count == 1

    def test_recommendation_block_on_high_severity(self) -> None:
        """High-severity alerts should produce 'block' recommendation."""
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()

        # Push enough escalating scores to trigger a high-severity alert
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        analysis = None
        for i, score in enumerate(scores):
            analysis = monitor.process_turn(
                text=f"Turn {i}",
                session_id=sid,
                risk_score=score,
            )

        assert analysis is not None
        if analysis.alerts:
            # HIGH/CRITICAL -> block, MEDIUM -> flag
            high_or_critical = any(
                a.severity in ("high", "critical") for a in analysis.alerts
            )
            if high_or_critical:
                assert analysis.recommendation == "block"

    def test_safe_turns_no_alerts(self) -> None:
        """Low-risk turns should produce no alerts."""
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()

        for i in range(3):
            analysis = monitor.process_turn(
                text=f"What is the weather in city {i}?",
                session_id=sid,
                risk_score=0.05,
                label="safe",
            )

        assert analysis.alerts == []
        assert analysis.recommendation == "continue_monitoring"
        assert analysis.escalation_detected is False


class TestEndSession:
    def test_end_returns_final_analysis(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()

        monitor.process_turn(text="Turn 1", session_id=sid, risk_score=0.1)
        monitor.process_turn(text="Turn 2", session_id=sid, risk_score=0.2)

        final = monitor.end_session(sid)
        assert isinstance(final, MultiTurnAnalysis)
        assert final.session_id == sid
        assert final.turn_count == 2

    def test_end_removes_session(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        monitor.process_turn(text="hello", session_id=sid)

        monitor.end_session(sid)

        # Session should be gone -- get_session_summary raises
        try:
            monitor.get_session_summary(sid)
            assert False, "Expected SessionNotFoundError"
        except SessionNotFoundError:
            pass

    def test_end_nonexistent_returns_empty(self) -> None:
        monitor = ConversationSecurityMonitor()
        final = monitor.end_session("does-not-exist")
        assert final.turn_count == 0
        assert final.alerts == []


class TestGetSessionSummary:
    def test_summary_structure(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        monitor.process_turn(text="hello", session_id=sid, risk_score=0.1)

        summary = monitor.get_session_summary(sid)
        assert summary["session_id"] == sid
        assert summary["turn_count"] == 1
        assert "risk_trend" in summary
        assert "created_at" in summary
        assert "last_activity" in summary

    def test_summary_missing_raises(self) -> None:
        monitor = ConversationSecurityMonitor()
        try:
            monitor.get_session_summary("nope")
            assert False, "Expected SessionNotFoundError"
        except SessionNotFoundError:
            pass


class TestCleanup:
    def test_cleanup_removes_expired(self) -> None:
        # Use a short but non-zero TTL so sessions can be created and
        # processed before they expire.
        config = SessionConfig(ttl_seconds=1)
        monitor = ConversationSecurityMonitor(config=config)

        sid = monitor.create_session()
        monitor.process_turn(text="hi", session_id=sid)
        time.sleep(1.1)  # wait for TTL to elapse

        removed = monitor.cleanup()
        assert removed >= 1

    def test_cleanup_keeps_active(self) -> None:
        config = SessionConfig(ttl_seconds=3600)
        monitor = ConversationSecurityMonitor(config=config)

        sid = monitor.create_session()
        monitor.process_turn(text="hi", session_id=sid)

        removed = monitor.cleanup()
        assert removed == 0


# ---------------------------------------------------------------------------
# Tests: _compute_recommendation unit tests
# ---------------------------------------------------------------------------

class TestRecommendationLogic:
    """Direct unit tests for the recommendation helper."""

    def test_no_alerts_continue(self) -> None:
        assert _compute_recommendation([]) == "continue_monitoring"

    def test_low_only_continue(self) -> None:
        alerts = [_make_alert("low")]
        assert _compute_recommendation(alerts) == "continue_monitoring"

    def test_medium_flag(self) -> None:
        alerts = [_make_alert("medium")]
        assert _compute_recommendation(alerts) == "flag"

    def test_high_block(self) -> None:
        alerts = [_make_alert("high")]
        assert _compute_recommendation(alerts) == "block"

    def test_critical_block(self) -> None:
        alerts = [_make_alert("critical")]
        assert _compute_recommendation(alerts) == "block"

    def test_mixed_high_and_medium_block(self) -> None:
        alerts = [_make_alert("medium"), _make_alert("high")]
        assert _compute_recommendation(alerts) == "block"

    def test_mixed_low_and_medium_flag(self) -> None:
        alerts = [_make_alert("low"), _make_alert("medium")]
        assert _compute_recommendation(alerts) == "flag"


# ---------------------------------------------------------------------------
# Tests: recommendation via mocked detectors (integration)
# ---------------------------------------------------------------------------

class TestRecommendationIntegration:
    def test_high_alert_blocks(self) -> None:
        monitor = ConversationSecurityMonitor()
        monitor._detectors = [_mock_detector([_make_alert("high")])]
        sid = monitor.create_session()
        result = monitor.process_turn(text="attack", session_id=sid, risk_score=0.9)
        assert result.recommendation == "block"

    def test_medium_alert_flags(self) -> None:
        monitor = ConversationSecurityMonitor()
        monitor._detectors = [_mock_detector([_make_alert("medium", "payload_assembly")])]
        sid = monitor.create_session()
        result = monitor.process_turn(text="frag", session_id=sid, risk_score=0.5)
        assert result.recommendation == "flag"

    def test_low_alert_continues(self) -> None:
        monitor = ConversationSecurityMonitor()
        monitor._detectors = [_mock_detector([_make_alert("low")])]
        sid = monitor.create_session()
        result = monitor.process_turn(text="hello", session_id=sid)
        assert result.recommendation == "continue_monitoring"


# ---------------------------------------------------------------------------
# Tests: single_turn_result dict extraction
# ---------------------------------------------------------------------------

class TestSingleTurnResultDict:
    def test_risk_and_label_extracted(self) -> None:
        monitor = ConversationSecurityMonitor()
        monitor._detectors = [_mock_detector()]
        sid = monitor.create_session()
        scan_dict = {
            "risk_score": 0.75,
            "label": "malicious",
            "technique_tags": ["c1", "d4"],
        }
        result = monitor.process_turn(
            text="bad input", session_id=sid, single_turn_result=scan_dict
        )
        assert result.turn_count == 1
        assert result.risk_trend == [0.75]

    def test_explicit_kwargs_take_precedence(self) -> None:
        monitor = ConversationSecurityMonitor()
        monitor._detectors = [_mock_detector()]
        sid = monitor.create_session()
        scan_dict = {"risk_score": 0.5, "label": "malicious"}
        result = monitor.process_turn(
            text="x", session_id=sid,
            single_turn_result=scan_dict,
            risk_score=0.99,  # explicit kwarg overrides dict
        )
        assert result.risk_trend == [0.99]
