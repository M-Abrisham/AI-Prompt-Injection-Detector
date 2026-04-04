"""Tests for Layer 16 T3.2 — Graduated Response Levels."""

from __future__ import annotations

from na0s.layer16.graduated_response import compute_threat_level, get_response_action
from na0s.layer16.models import (
    Alert,
    ConversationState,
    ThreatLevel,
    UserRiskProfile,
)
from na0s.layer16.conversation_monitor import ConversationSecurityMonitor


# -----------------------------------------------------------------------
# Helper to build a state with given metrics
# -----------------------------------------------------------------------

def _make_state(
    cumulative_risk: float = 0.0,
    peak_risk: float = 0.0,
    cusum_score: float = 0.0,
) -> ConversationState:
    s = ConversationState(session_id="test")
    s.cumulative_risk = cumulative_risk
    s.peak_risk = peak_risk
    s.cusum_score = cusum_score
    return s


def _make_alert(severity: str = "low", alert_type: str = "escalation") -> Alert:
    return Alert(
        alert_type=alert_type,
        severity=severity,
        confidence=0.8,
        description="test alert",
    )


# -----------------------------------------------------------------------
# Each threat level triggered correctly
# -----------------------------------------------------------------------


class TestThreatLevelComputation:
    def test_normal_default(self):
        state = _make_state()
        assert compute_threat_level(state, []) == ThreatLevel.NORMAL

    def test_watch_from_low_alert(self):
        state = _make_state()
        alerts = [_make_alert(severity="low", alert_type="context_poisoning")]
        assert compute_threat_level(state, alerts) == ThreatLevel.WATCH

    def test_watch_from_cumulative_risk(self):
        state = _make_state(cumulative_risk=0.35)
        assert compute_threat_level(state, []) == ThreatLevel.WATCH

    def test_suspect_from_medium_alert(self):
        state = _make_state()
        alerts = [_make_alert(severity="medium")]
        assert compute_threat_level(state, alerts) == ThreatLevel.SUSPECT

    def test_suspect_from_cumulative_risk(self):
        state = _make_state(cumulative_risk=0.55)
        assert compute_threat_level(state, []) == ThreatLevel.SUSPECT

    def test_suspect_from_escalation_alert(self):
        state = _make_state()
        alerts = [_make_alert(severity="low", alert_type="escalation")]
        # low severity triggers WATCH, but escalation alert_type triggers SUSPECT
        # SUSPECT is higher, so it wins
        assert compute_threat_level(state, alerts) == ThreatLevel.SUSPECT

    def test_flagged_from_high_alert(self):
        state = _make_state()
        alerts = [_make_alert(severity="high")]
        assert compute_threat_level(state, alerts) == ThreatLevel.FLAGGED

    def test_flagged_from_cumulative_risk(self):
        state = _make_state(cumulative_risk=0.75)
        assert compute_threat_level(state, []) == ThreatLevel.FLAGGED

    def test_flagged_from_peak_risk(self):
        state = _make_state(peak_risk=0.9)
        assert compute_threat_level(state, []) == ThreatLevel.FLAGGED

    def test_blocked_from_critical_alert(self):
        state = _make_state()
        alerts = [_make_alert(severity="critical")]
        assert compute_threat_level(state, alerts) == ThreatLevel.BLOCKED

    def test_blocked_from_cumulative_risk(self):
        state = _make_state(cumulative_risk=0.95)
        assert compute_threat_level(state, []) == ThreatLevel.BLOCKED

    def test_blocked_from_cusum(self):
        state = _make_state(cusum_score=5.5)
        assert compute_threat_level(state, []) == ThreatLevel.BLOCKED

    def test_highest_level_wins(self):
        """When multiple conditions match, highest level wins."""
        state = _make_state(cumulative_risk=0.95)  # BLOCKED
        alerts = [_make_alert(severity="medium")]  # SUSPECT
        assert compute_threat_level(state, alerts) == ThreatLevel.BLOCKED


# -----------------------------------------------------------------------
# User profile risk multiplier affects threat level
# -----------------------------------------------------------------------


class TestUserProfileMultiplier:
    def test_blocked_user_always_blocked(self):
        state = _make_state(cumulative_risk=0.1)
        profile = UserRiskProfile(user_hash="u1", flag_level="blocked")
        assert compute_threat_level(state, [], profile) == ThreatLevel.BLOCKED

    def test_flagged_user_amplifies_risk(self):
        # risk 0.4, multiplier 2.0 -> effective 0.8 -> FLAGGED
        state = _make_state(cumulative_risk=0.4)
        profile = UserRiskProfile(user_hash="u1", flag_level="flagged")
        assert compute_threat_level(state, [], profile) == ThreatLevel.FLAGGED

    def test_suspect_user_amplifies_risk(self):
        # risk 0.35, multiplier 1.5 -> effective 0.525 -> SUSPECT
        state = _make_state(cumulative_risk=0.35)
        profile = UserRiskProfile(user_hash="u1", flag_level="suspect")
        assert compute_threat_level(state, [], profile) == ThreatLevel.SUSPECT

    def test_normal_user_no_amplification(self):
        state = _make_state(cumulative_risk=0.25)
        profile = UserRiskProfile(user_hash="u1", flag_level="normal")
        assert compute_threat_level(state, [], profile) == ThreatLevel.NORMAL

    def test_none_profile_no_effect(self):
        state = _make_state(cumulative_risk=0.25)
        assert compute_threat_level(state, [], None) == ThreatLevel.NORMAL


# -----------------------------------------------------------------------
# Response actions match each level
# -----------------------------------------------------------------------


class TestResponseActions:
    def test_normal_actions(self):
        action = get_response_action(ThreatLevel.NORMAL)
        assert action["allow"] is True
        assert action["log"] is False
        assert action["rate_limit"] is False
        assert action["human_review"] is False

    def test_watch_actions(self):
        action = get_response_action(ThreatLevel.WATCH)
        assert action["allow"] is True
        assert action["log"] is True
        assert action["rate_limit"] is False

    def test_suspect_actions(self):
        action = get_response_action(ThreatLevel.SUSPECT)
        assert action["allow"] is True
        assert action["log"] is True
        assert action["rate_limit"] is True
        assert action["human_review"] is False

    def test_flagged_actions(self):
        action = get_response_action(ThreatLevel.FLAGGED)
        assert action["allow"] is False
        assert action["log"] is True
        assert action["rate_limit"] is True
        assert action["human_review"] is True

    def test_blocked_actions(self):
        action = get_response_action(ThreatLevel.BLOCKED)
        assert action["allow"] is False
        assert action["log"] is True
        assert action["human_review"] is True

    def test_response_action_returns_copy(self):
        a1 = get_response_action(ThreatLevel.NORMAL)
        a2 = get_response_action(ThreatLevel.NORMAL)
        a1["allow"] = False
        assert a2["allow"] is True  # not mutated


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_no_alerts_empty_state(self):
        state = _make_state()
        assert compute_threat_level(state, []) == ThreatLevel.NORMAL

    def test_empty_alerts_list(self):
        state = _make_state(cumulative_risk=0.35)
        assert compute_threat_level(state, []) == ThreatLevel.WATCH

    def test_threat_level_enum_values(self):
        assert ThreatLevel.NORMAL.value == "normal"
        assert ThreatLevel.WATCH.value == "watch"
        assert ThreatLevel.SUSPECT.value == "suspect"
        assert ThreatLevel.FLAGGED.value == "flagged"
        assert ThreatLevel.BLOCKED.value == "blocked"

    def test_threat_level_is_str(self):
        # ThreatLevel inherits from str
        assert isinstance(ThreatLevel.NORMAL, str)
        assert ThreatLevel.NORMAL == "normal"


# -----------------------------------------------------------------------
# Integration with ConversationSecurityMonitor
# -----------------------------------------------------------------------


class TestMonitorIntegration:
    def test_process_turn_returns_threat_level(self):
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        analysis = monitor.process_turn("hello world", session_id=sid)
        assert analysis.threat_level in ("normal", "watch", "suspect", "flagged", "blocked")
        assert isinstance(analysis.response_action, dict)
        assert "allow" in analysis.response_action

    def test_end_session_returns_threat_level(self):
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        monitor.process_turn("hello", session_id=sid)
        analysis = monitor.end_session(sid)
        assert analysis.threat_level in ("normal", "watch", "suspect", "flagged", "blocked")

    def test_to_dict_includes_threat_level(self):
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        analysis = monitor.process_turn("hello", session_id=sid)
        d = analysis.to_dict()
        assert "threat_level" in d
        assert "response_action" in d

    def test_user_hash_stored_in_session(self):
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session(user_hash="abc123")
        state = monitor._session_mgr.get_session(sid)
        assert state.metadata.get("user_hash") == "abc123"

    def test_user_hash_via_process_turn(self):
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        monitor.process_turn("hello", session_id=sid, user_hash="xyz789")
        state = monitor._session_mgr.get_session(sid)
        assert state.metadata.get("user_hash") == "xyz789"

    def test_end_session_updates_profile(self):
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session(user_hash="user_test")
        monitor.process_turn("hello", session_id=sid, risk_score=0.5)
        monitor.end_session(sid)
        profile = monitor._profile_store.get_profile("user_test")
        assert profile is not None
        assert profile.session_count == 1
        assert profile.cumulative_risk > 0.0

    def test_multiple_sessions_accumulate_profile_risk(self):
        monitor = ConversationSecurityMonitor()

        # Session 1
        sid1 = monitor.create_session(user_hash="repeat_user")
        monitor.process_turn("test", session_id=sid1, risk_score=0.6)
        monitor.end_session(sid1)

        # Session 2
        sid2 = monitor.create_session(user_hash="repeat_user")
        monitor.process_turn("test", session_id=sid2, risk_score=0.7)
        monitor.end_session(sid2)

        profile = monitor._profile_store.get_profile("repeat_user")
        assert profile.session_count == 2
        assert profile.cumulative_risk > 0.0
