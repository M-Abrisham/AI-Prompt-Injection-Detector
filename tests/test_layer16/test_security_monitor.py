"""Tests for Layer 16 security fixes: max sessions, dedup cleanup,
high/critical alert bypass, and thread safety.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from na0s.layer16.conversation_monitor import ConversationSecurityMonitor
from na0s.layer16.models import Alert, SessionConfig
from na0s.layer16.session_manager import MAX_SESSIONS, SessionManager


# ---------------------------------------------------------------------------
# CRITICAL-2: Max sessions enforcement
# ---------------------------------------------------------------------------

class TestMaxSessionsEnforcement:

    def test_create_session_raises_at_limit(self):
        """SessionManager.create_session() raises RuntimeError at MAX_SESSIONS."""
        mgr = SessionManager()
        # Directly fill sessions dict to just below limit
        now = datetime.now(timezone.utc)
        from na0s.layer16.models import ConversationState
        for i in range(MAX_SESSIONS):
            mgr._sessions[f"fake-{i}"] = ConversationState(
                session_id=f"fake-{i}", created_at=now, last_activity=now,
            )

        with pytest.raises(RuntimeError, match="Maximum session limit reached"):
            mgr.create_session()

    def test_auto_create_session_raises_at_limit(self):
        """SessionManager._auto_create_session() raises RuntimeError at MAX_SESSIONS."""
        mgr = SessionManager()
        now = datetime.now(timezone.utc)
        from na0s.layer16.models import ConversationState
        for i in range(MAX_SESSIONS):
            mgr._sessions[f"fake-{i}"] = ConversationState(
                session_id=f"fake-{i}", created_at=now, last_activity=now,
            )

        with pytest.raises(RuntimeError, match="Maximum session limit reached"):
            mgr._auto_create_session("new-session")

    def test_auto_create_in_process_turn_respects_limit(self):
        """process_turn auto-create path raises when session limit is hit."""
        monitor = ConversationSecurityMonitor()
        now = datetime.now(timezone.utc)
        from na0s.layer16.models import ConversationState
        mgr = monitor._session_mgr
        for i in range(MAX_SESSIONS):
            mgr._sessions[f"fake-{i}"] = ConversationState(
                session_id=f"fake-{i}", created_at=now, last_activity=now,
            )

        with pytest.raises(RuntimeError, match="Maximum session limit reached"):
            monitor.process_turn("hello", session_id="brand-new")

    def test_create_session_succeeds_below_limit(self):
        """Sanity check: creating sessions works below the limit."""
        mgr = SessionManager()
        sid = mgr.create_session()
        assert sid in mgr._sessions


# ---------------------------------------------------------------------------
# HIGH-5: Unbounded dedup dicts — cleanup prunes stale entries
# ---------------------------------------------------------------------------

class TestCleanupPrunesDedup:

    def test_cleanup_removes_stale_dedup_entries(self):
        """cleanup() prunes _last_alert_turn and _last_deduped for expired sessions."""
        config = SessionConfig(ttl_seconds=1)
        monitor = ConversationSecurityMonitor(config=config)

        # Create a session and process a turn to populate dedup dicts
        sid = monitor.create_session()
        monitor.process_turn("test input", session_id=sid)

        assert sid in monitor._last_deduped

        # Manually seed dedup for a non-existent session too
        monitor._last_alert_turn["ghost-session"] = {"escalation": (1, 0.5)}
        monitor._last_deduped["ghost-session"] = []

        # Expire the real session by manipulating last_activity
        state = monitor._session_mgr._sessions[sid]
        state.last_activity = datetime.now(timezone.utc) - timedelta(seconds=10)

        removed = monitor.cleanup()
        assert removed >= 1

        # Both the real expired session and the ghost should be pruned
        assert sid not in monitor._last_alert_turn
        assert sid not in monitor._last_deduped
        assert "ghost-session" not in monitor._last_alert_turn
        assert "ghost-session" not in monitor._last_deduped


# ---------------------------------------------------------------------------
# MEDIUM-3: High/critical alerts are never suppressed by dedup
# ---------------------------------------------------------------------------

class TestHighCriticalAlertsBypassDedup:

    def test_high_severity_never_suppressed(self):
        """Alerts with severity 'high' must pass through dedup every time."""
        monitor = ConversationSecurityMonitor()
        alerts = [
            Alert(
                alert_type="escalation",
                severity="high",
                confidence=0.8,
                description="test",
            ),
        ]
        # Fire the same alert twice on consecutive turns — both should pass
        result1 = monitor._dedup_alerts(alerts, "s1", current_turn=1)
        result2 = monitor._dedup_alerts(alerts, "s1", current_turn=2)
        assert len(result1) == 1
        assert len(result2) == 1

    def test_critical_severity_never_suppressed(self):
        """Alerts with severity 'critical' must pass through dedup every time."""
        monitor = ConversationSecurityMonitor()
        alerts = [
            Alert(
                alert_type="payload_assembly",
                severity="critical",
                confidence=0.9,
                description="test",
            ),
        ]
        result1 = monitor._dedup_alerts(alerts, "s1", current_turn=1)
        result2 = monitor._dedup_alerts(alerts, "s1", current_turn=2)
        assert len(result1) == 1
        assert len(result2) == 1

    def test_low_severity_is_still_suppressed(self):
        """Low-severity alerts are still subject to normal dedup suppression."""
        monitor = ConversationSecurityMonitor()
        alerts = [
            Alert(
                alert_type="escalation",
                severity="low",
                confidence=0.5,
                description="test",
            ),
        ]
        result1 = monitor._dedup_alerts(alerts, "s1", current_turn=1)
        result2 = monitor._dedup_alerts(alerts, "s1", current_turn=2)
        assert len(result1) == 1
        # Should be suppressed (same type, within window, no confidence jump)
        assert len(result2) == 0


# ---------------------------------------------------------------------------
# HIGH-4: Thread safety — concurrent process_turn calls don't crash
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_process_turn_no_crash(self):
        """Multiple threads calling process_turn concurrently must not crash."""
        monitor = ConversationSecurityMonitor()
        sid = monitor.create_session()
        errors: list = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    monitor.process_turn(
                        f"turn {thread_id}-{i}",
                        session_id=sid,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Threads raised errors: {errors}"
        # All turns should have been recorded
        state = monitor._session_mgr.get_session(sid)
        assert state is not None
        assert state.turn_count == 50  # 5 threads x 10 turns

    def test_concurrent_auto_create_no_crash(self):
        """Multiple threads auto-creating the same session must not crash."""
        monitor = ConversationSecurityMonitor()
        errors: list = []

        def worker(thread_id: int) -> None:
            try:
                monitor.process_turn(
                    f"turn from thread {thread_id}",
                    session_id="shared-session",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Threads raised errors: {errors}"
        state = monitor._session_mgr.get_session("shared-session")
        assert state is not None
        assert state.turn_count == 10
