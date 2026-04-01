"""Tests for Layer 16 SessionManager."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from na0s.layer16.models import SessionConfig
from na0s.layer16.session_manager import SessionManager


def _expired_config(ttl: int = 1) -> SessionConfig:
    """SessionConfig with a very short TTL for testing expiry."""
    return SessionConfig(ttl_seconds=ttl)


class TestCreateAndGet:
    def test_create_returns_uuid(self) -> None:
        mgr = SessionManager()
        sid = mgr.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 36  # uuid4 format

    def test_get_returns_state(self) -> None:
        mgr = SessionManager()
        sid = mgr.create_session()
        state = mgr.get_session(sid)
        assert state is not None
        assert state.session_id == sid

    def test_get_missing_returns_none(self) -> None:
        mgr = SessionManager()
        assert mgr.get_session("nonexistent") is None


class TestUpdate:
    def test_update_adds_turn(self) -> None:
        mgr = SessionManager()
        sid = mgr.create_session()
        state = mgr.update_session(sid, "hello", risk_score=0.2)
        assert state is not None
        assert state.turn_count == 1
        assert state.turns[0].text == "hello"

    def test_update_missing_returns_none(self) -> None:
        mgr = SessionManager()
        assert mgr.update_session("nope", "text") is None

    def test_multiple_updates(self) -> None:
        mgr = SessionManager()
        sid = mgr.create_session()
        mgr.update_session(sid, "a", risk_score=0.1)
        mgr.update_session(sid, "b", risk_score=0.3)
        mgr.update_session(sid, "c", risk_score=0.6)
        state = mgr.get_session(sid)
        assert state is not None
        assert state.turn_count == 3


class TestExpire:
    def test_expire_removes_session(self) -> None:
        mgr = SessionManager()
        sid = mgr.create_session()
        mgr.expire_session(sid)
        assert mgr.get_session(sid) is None

    def test_expire_nonexistent_is_noop(self) -> None:
        mgr = SessionManager()
        mgr.expire_session("does-not-exist")  # should not raise


class TestTTLExpiry:
    def test_expired_session_not_returned(self) -> None:
        mgr = SessionManager(config=_expired_config(ttl=0))
        sid = mgr.create_session()
        # TTL=0 means it's already expired
        time.sleep(0.05)
        assert mgr.get_session(sid) is None

    def test_cleanup_expired(self) -> None:
        mgr = SessionManager(config=_expired_config(ttl=0))
        mgr.create_session()
        mgr.create_session()
        time.sleep(0.05)
        removed = mgr.cleanup_expired()
        assert removed == 2
        assert mgr.active_session_count == 0

    def test_cleanup_keeps_active(self) -> None:
        mgr = SessionManager(config=SessionConfig(ttl_seconds=3600))
        mgr.create_session()
        removed = mgr.cleanup_expired()
        assert removed == 0
        assert mgr.active_session_count == 1


class TestListActive:
    def test_list_active_sessions(self) -> None:
        mgr = SessionManager()
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        active = mgr.list_active_sessions()
        assert set(active) == {s1, s2}

    def test_list_excludes_expired(self) -> None:
        mgr = SessionManager(config=_expired_config(ttl=0))
        mgr.create_session()
        time.sleep(0.05)
        assert mgr.list_active_sessions() == []

    def test_active_session_count(self) -> None:
        mgr = SessionManager()
        mgr.create_session()
        mgr.create_session()
        assert mgr.active_session_count == 2


class TestThreadSafety:
    def test_concurrent_creates(self) -> None:
        mgr = SessionManager()
        results: list[str] = []
        errors: list[Exception] = []

        def create() -> None:
            try:
                sid = mgr.create_session()
                results.append(sid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50
        assert len(set(results)) == 50  # all unique

    def test_concurrent_updates(self) -> None:
        mgr = SessionManager()
        sid = mgr.create_session()
        errors: list[Exception] = []

        def update(i: int) -> None:
            try:
                mgr.update_session(sid, f"turn {i}", risk_score=i * 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        state = mgr.get_session(sid)
        assert state is not None
        assert state.turn_count == 20
