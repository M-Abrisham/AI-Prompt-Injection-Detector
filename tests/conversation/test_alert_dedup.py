"""Tests for Layer 16 alert deduplication logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from na0s.layer16.conversation_monitor import ConversationSecurityMonitor
from na0s.layer16.models import Alert


def _make_monitor() -> ConversationSecurityMonitor:
    """Create a monitor with detectors disabled (we inject alerts manually)."""
    mon = ConversationSecurityMonitor()
    mon._detectors = []  # disable real detectors
    return mon


def _fake_detector(alerts: list[Alert]):
    """Return a mock detector that yields *alerts* on every analyze() call."""

    class _Det:
        detector_name = "fake"

        def analyze(self, state):
            return list(alerts)

    return _Det()


# ------------------------------------------------------------------
# 1. Duplicate alert_type suppressed within suppression window
# ------------------------------------------------------------------


def test_duplicate_alert_suppressed_within_window():
    mon = _make_monitor()
    sid = mon.create_session()

    alert = Alert(
        alert_type="escalation",
        severity="medium",
        confidence=0.7,
        description="test",
    )
    mon._detectors = [_fake_detector([alert])]

    mon.process_turn("turn1", sid)
    new1 = mon.get_new_alerts(sid)
    assert len(new1) == 1  # first alert passes

    mon.process_turn("turn2", sid)
    new2 = mon.get_new_alerts(sid)
    assert len(new2) == 0  # suppressed (within 3 turns)

    mon.process_turn("turn3", sid)
    new3 = mon.get_new_alerts(sid)
    assert len(new3) == 0  # still suppressed


# ------------------------------------------------------------------
# 2. Alert passes through after suppression window expires
# ------------------------------------------------------------------


def test_alert_passes_after_window_expires():
    mon = _make_monitor()
    sid = mon.create_session()

    alert = Alert(
        alert_type="escalation",
        severity="medium",
        confidence=0.7,
        description="test",
    )
    mon._detectors = [_fake_detector([alert])]

    mon.process_turn("turn1", sid)  # fires at turn 1

    # Turns 2-4: suppressed (within window of 3)
    # The alert fires at turn 1, suppression_window=3 means
    # turns 2,3,4 are suppressed (turn - prev <= 3).
    # Turn 5 should pass (5 - 1 = 4 > 3).
    mon._detectors = []  # no alerts for filler turns
    mon.process_turn("turn2", sid)
    mon.process_turn("turn3", sid)
    mon.process_turn("turn4", sid)

    mon._detectors = [_fake_detector([alert])]
    mon.process_turn("turn5", sid)
    new = mon.get_new_alerts(sid)
    assert len(new) == 1  # window expired, alert passes


# ------------------------------------------------------------------
# 3. Different alert_types are NOT suppressed
# ------------------------------------------------------------------


def test_different_alert_types_not_suppressed():
    mon = _make_monitor()
    sid = mon.create_session()

    esc = Alert(alert_type="escalation", severity="medium", confidence=0.7, description="esc")
    payload = Alert(alert_type="payload_assembly", severity="high", confidence=0.8, description="pay")

    mon._detectors = [_fake_detector([esc])]
    mon.process_turn("turn1", sid)
    assert len(mon.get_new_alerts(sid)) == 1

    mon._detectors = [_fake_detector([payload])]
    mon.process_turn("turn2", sid)
    assert len(mon.get_new_alerts(sid)) == 1  # different type, not suppressed


# ------------------------------------------------------------------
# 4. Higher-confidence duplicate passes through (>= 0.15 jump)
# ------------------------------------------------------------------


def test_higher_confidence_duplicate_passes():
    mon = _make_monitor()
    sid = mon.create_session()

    alert_low = Alert(alert_type="escalation", severity="medium", confidence=0.5, description="low")
    alert_high = Alert(alert_type="escalation", severity="high", confidence=0.7, description="high")

    mon._detectors = [_fake_detector([alert_low])]
    mon.process_turn("turn1", sid)
    new1 = mon.get_new_alerts(sid)
    assert len(new1) == 1
    assert new1[0].confidence == 0.5

    # Confidence jump = 0.7 - 0.5 = 0.2 >= 0.15 -> passes
    mon._detectors = [_fake_detector([alert_high])]
    mon.process_turn("turn2", sid)
    new2 = mon.get_new_alerts(sid)
    assert len(new2) == 1
    assert new2[0].confidence == 0.7


def test_marginal_confidence_increase_suppressed():
    """Confidence increase < 0.15 should still be suppressed."""
    mon = _make_monitor()
    sid = mon.create_session()

    alert1 = Alert(alert_type="escalation", severity="medium", confidence=0.5, description="a")
    alert2 = Alert(alert_type="escalation", severity="medium", confidence=0.6, description="b")

    mon._detectors = [_fake_detector([alert1])]
    mon.process_turn("turn1", sid)

    # 0.6 - 0.5 = 0.1 < 0.15 -> suppressed
    mon._detectors = [_fake_detector([alert2])]
    mon.process_turn("turn2", sid)
    assert len(mon.get_new_alerts(sid)) == 0


# ------------------------------------------------------------------
# 5. Feature flag disabled = no suppression
# ------------------------------------------------------------------


def test_dedup_disabled_all_alerts_pass():
    mon = _make_monitor()
    sid = mon.create_session()

    alert = Alert(alert_type="escalation", severity="medium", confidence=0.7, description="test")
    mon._detectors = [_fake_detector([alert])]

    with patch("na0s.layer16.conversation_monitor.layer16_config.ENABLE_ALERT_DEDUP", False):
        mon.process_turn("turn1", sid)
        new1 = mon.get_new_alerts(sid)
        mon.process_turn("turn2", sid)
        new2 = mon.get_new_alerts(sid)
        mon.process_turn("turn3", sid)
        new3 = mon.get_new_alerts(sid)

    assert len(new1) == 1
    assert len(new2) == 1
    assert len(new3) == 1


# ------------------------------------------------------------------
# 6. Session cleanup removes suppression tracking
# ------------------------------------------------------------------


def test_end_session_cleans_up_tracking():
    mon = _make_monitor()
    sid = mon.create_session()

    alert = Alert(alert_type="escalation", severity="medium", confidence=0.7, description="test")
    mon._detectors = [_fake_detector([alert])]
    mon.process_turn("turn1", sid)

    assert sid in mon._last_alert_turn
    assert sid in mon._last_deduped

    mon.end_session(sid)
    assert sid not in mon._last_alert_turn
    assert sid not in mon._last_deduped


# ------------------------------------------------------------------
# 7. Multiple sessions tracked independently
# ------------------------------------------------------------------


def test_multiple_sessions_independent():
    mon = _make_monitor()
    sid1 = mon.create_session()
    sid2 = mon.create_session()

    alert = Alert(alert_type="escalation", severity="medium", confidence=0.7, description="test")
    mon._detectors = [_fake_detector([alert])]

    # Fire in session 1
    mon.process_turn("turn1", sid1)
    assert len(mon.get_new_alerts(sid1)) == 1

    # Session 2 should still get the alert (independent tracking)
    mon.process_turn("turn1", sid2)
    assert len(mon.get_new_alerts(sid2)) == 1

    # Session 1 suppressed, session 2 suppressed
    mon.process_turn("turn2", sid1)
    mon.process_turn("turn2", sid2)
    assert len(mon.get_new_alerts(sid1)) == 0
    assert len(mon.get_new_alerts(sid2)) == 0


# ------------------------------------------------------------------
# 8. First alert always passes through
# ------------------------------------------------------------------


def test_first_alert_always_passes():
    mon = _make_monitor()
    sid = mon.create_session()

    alert = Alert(alert_type="fabricated_history", severity="high", confidence=0.9, description="first")
    mon._detectors = [_fake_detector([alert])]

    mon.process_turn("first turn", sid)
    new = mon.get_new_alerts(sid)
    assert len(new) == 1
    assert new[0].alert_type == "fabricated_history"
