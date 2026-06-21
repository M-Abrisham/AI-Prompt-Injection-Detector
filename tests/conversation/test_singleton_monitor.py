"""Regression test for singleton ConversationSecurityMonitor.

Bug: predict.py created a new ConversationSecurityMonitor() on every
scan() call, discarding session state. Multi-turn detection was
non-functional because the monitor had no memory of previous turns.

Fix: Module-level singleton with double-checked locking (matching
Na0S's existing _get_cached_models() pattern).
"""

import pytest

from na0s.predict import (
    _get_conversation_monitor,
    _reset_conversation_monitor,
)


class TestSingletonMonitor:
    """Verify that multi-turn state persists across calls."""

    def setup_method(self):
        _reset_conversation_monitor()

    def teardown_method(self):
        _reset_conversation_monitor()

    def test_same_instance_across_calls(self):
        """The singleton must return the exact same object."""
        m1 = _get_conversation_monitor()
        m2 = _get_conversation_monitor()
        assert m1 is m2

    def test_session_state_persists(self):
        """State from turn 1 must be visible in turn 2."""
        monitor = _get_conversation_monitor()
        sid = "test-persist"

        monitor.process_turn(text="What is ML?", session_id=sid, risk_score=0.1, label="safe")
        monitor.process_turn(text="How to exploit ML?", session_id=sid, risk_score=0.6, label="safe")

        session = monitor._session_mgr.get_session(sid)
        assert session is not None
        assert session.turn_count == 2

    def test_different_sessions_independent(self):
        monitor = _get_conversation_monitor()

        monitor.process_turn(text="Turn A1", session_id="sess-a", risk_score=0.1)
        monitor.process_turn(text="Turn B1", session_id="sess-b", risk_score=0.1)
        monitor.process_turn(text="Turn A2", session_id="sess-a", risk_score=0.2)

        assert monitor._session_mgr.get_session("sess-a").turn_count == 2
        assert monitor._session_mgr.get_session("sess-b").turn_count == 1

    def test_reset_clears_state(self):
        monitor_before = _get_conversation_monitor()
        monitor_before.process_turn(text="hi", session_id="s1", risk_score=0.1)
        assert monitor_before._session_mgr.active_session_count > 0

        _reset_conversation_monitor()

        monitor_after = _get_conversation_monitor()
        assert monitor_after is not monitor_before
        assert monitor_after._session_mgr.active_session_count == 0

    def test_scan_without_session_no_monitor_sessions(self):
        """Stateless scan() should not create any sessions."""
        from na0s.predict import scan

        scan("Hello world")
        monitor = _get_conversation_monitor()
        assert monitor._session_mgr.active_session_count == 0
