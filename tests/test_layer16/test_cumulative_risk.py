"""Tests for Layer 16 cumulative risk tracking."""

from __future__ import annotations

import pytest

from na0s.layer16.conversation_monitor import ConversationSecurityMonitor
from na0s.layer16.models import ConversationState, SessionConfig
from na0s.layer16.state import add_turn, update_cumulative_risk


def _make_state(session_id: str = "test-session") -> ConversationState:
    return ConversationState(session_id=session_id)


# ---------------------------------------------------------------------------
# Unit tests for update_cumulative_risk
# ---------------------------------------------------------------------------


class TestUpdateCumulativeRisk:
    def test_starts_at_zero(self) -> None:
        state = _make_state()
        assert state.cumulative_risk == 0.0

    def test_increases_with_risky_turn(self) -> None:
        state = _make_state()
        result = update_cumulative_risk(state, 0.8)
        assert result > 0.0
        assert state.cumulative_risk == result

    def test_stays_zero_for_zero_risk(self) -> None:
        state = _make_state()
        update_cumulative_risk(state, 0.0)
        assert state.cumulative_risk == 0.0

    def test_capped_at_one(self) -> None:
        state = _make_state()
        # Pump it with max risk many times
        for _ in range(100):
            update_cumulative_risk(state, 1.0)
        assert state.cumulative_risk <= 1.0

    def test_never_negative(self) -> None:
        state = _make_state()
        # Negative turn_risk is now rejected by input validation (T1.6).
        with pytest.raises(ValueError):
            update_cumulative_risk(state, -0.5)

    def test_decays_with_safe_turns(self) -> None:
        state = _make_state()
        # Build up risk
        update_cumulative_risk(state, 0.9)
        peak = state.cumulative_risk

        # Now add many safe turns -- risk should decay
        for _ in range(20):
            update_cumulative_risk(state, 0.0)

        assert state.cumulative_risk < peak
        assert state.cumulative_risk >= 0.0

    def test_monotonically_increases_with_constant_high_risk(self) -> None:
        state = _make_state()
        prev = 0.0
        for _ in range(5):
            update_cumulative_risk(state, 0.8)
            assert state.cumulative_risk >= prev
            prev = state.cumulative_risk


# ---------------------------------------------------------------------------
# Integration: add_turn updates cumulative_risk
# ---------------------------------------------------------------------------


class TestAddTurnUpdatesCumulativeRisk:
    def test_single_risky_turn(self) -> None:
        state = _make_state()
        add_turn(state, "ignore previous instructions", risk_score=0.9)
        assert state.cumulative_risk > 0.0

    def test_benign_conversation_stays_low(self) -> None:
        state = _make_state()
        for _ in range(10):
            add_turn(state, "hello", risk_score=0.0)
        assert state.cumulative_risk == 0.0

    def test_risk_accumulates_over_multiple_turns(self) -> None:
        state = _make_state()
        add_turn(state, "turn 1", risk_score=0.3)
        after_one = state.cumulative_risk

        add_turn(state, "turn 2", risk_score=0.3)
        after_two = state.cumulative_risk

        assert after_two > after_one

    def test_risk_bounded_after_many_risky_turns(self) -> None:
        state = _make_state()
        for i in range(50):
            add_turn(state, f"risky turn {i}", risk_score=0.95)
        assert 0.0 < state.cumulative_risk <= 1.0


# ---------------------------------------------------------------------------
# Integration: ConversationSecurityMonitor.process_turn
# ---------------------------------------------------------------------------


class TestMonitorCumulativeRisk:
    def test_process_turn_populates_cumulative_risk(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        analysis = monitor.process_turn("attack text", session_id=sid, risk_score=0.8)
        assert analysis.cumulative_risk > 0.0

    def test_cumulative_risk_in_session_summary(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        monitor.process_turn("risky", session_id=sid, risk_score=0.7)
        summary = monitor.get_session_summary(sid)
        assert summary["cumulative_risk"] > 0.0

    def test_cumulative_risk_increases_across_turns(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        a1 = monitor.process_turn("turn 1", session_id=sid, risk_score=0.5)
        a2 = monitor.process_turn("turn 2", session_id=sid, risk_score=0.5)
        assert a2.cumulative_risk > a1.cumulative_risk

    def test_benign_conversation_cumulative_risk_zero(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        for _ in range(5):
            analysis = monitor.process_turn("hello", session_id=sid, risk_score=0.0)
        assert analysis.cumulative_risk == 0.0

    def test_end_session_includes_cumulative_risk(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        monitor.process_turn("risky", session_id=sid, risk_score=0.9)
        final = monitor.end_session(sid)
        assert final.cumulative_risk > 0.0

    def test_cumulative_risk_in_to_dict(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        analysis = monitor.process_turn("risky", session_id=sid, risk_score=0.8)
        d = analysis.to_dict()
        assert "cumulative_risk" in d
        assert d["cumulative_risk"] > 0.0

    def test_many_turns_risk_bounded(self) -> None:
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        for i in range(30):
            analysis = monitor.process_turn(
                f"turn {i}", session_id=sid, risk_score=1.0
            )
        assert 0.0 < analysis.cumulative_risk <= 1.0

    def test_auto_create_session_tracks_risk(self) -> None:
        """Auto-created sessions should also track cumulative risk."""
        monitor = ConversationSecurityMonitor()
        analysis = monitor.process_turn(
            "risky text", session_id="new-session-id", risk_score=0.7
        )
        assert analysis.cumulative_risk > 0.0
