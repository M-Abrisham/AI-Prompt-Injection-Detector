"""Tests for Layer 16 SQLiteBackend."""  # LAYER16

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

from na0s.layer16.models import ConversationState
from na0s.layer16.state import add_turn
from na0s.layer16.storage.sqlite_backend import SQLiteBackend


def _make_backend(tmp_path: str | None = None) -> SQLiteBackend:
    """Create a SQLiteBackend with a temp db file.

    If *tmp_path* is a directory (e.g. pytest's tmp_path fixture),
    a database file is created inside it.  Otherwise a new tempfile
    is created.
    """
    if tmp_path is not None and os.path.isdir(str(tmp_path)):
        tmp_path = os.path.join(str(tmp_path), "test.db")
    if tmp_path is None:
        fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
    return SQLiteBackend(db_path=tmp_path)


def _make_state(session_id: str = "test-session") -> ConversationState:
    """Create a simple ConversationState for testing."""
    return ConversationState(
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        last_activity=datetime.now(timezone.utc),
    )


class TestSaveLoad:
    def test_save_and_load(self) -> None:
        backend = _make_backend()
        state = _make_state("s1")
        add_turn(state, "hello", risk_score=0.1, label="safe")

        backend.save_session("s1", state)
        loaded = backend.load_session("s1")

        assert loaded is not None
        assert loaded.session_id == "s1"
        assert loaded.turn_count == 1
        assert loaded.turns[0].text == "hello"
        assert loaded.turns[0].risk_score == 0.1

    def test_load_missing_returns_none(self) -> None:
        backend = _make_backend()
        assert backend.load_session("nonexistent") is None

    def test_upsert_overwrites(self) -> None:
        backend = _make_backend()
        state = _make_state("s1")
        add_turn(state, "first", risk_score=0.1)
        backend.save_session("s1", state)

        add_turn(state, "second", risk_score=0.2)
        backend.save_session("s1", state)

        loaded = backend.load_session("s1")
        assert loaded is not None
        assert loaded.turn_count == 2


class TestDelete:
    def test_delete_removes(self) -> None:
        backend = _make_backend()
        state = _make_state("s1")
        backend.save_session("s1", state)

        backend.delete_session("s1")
        assert backend.load_session("s1") is None

    def test_delete_missing_is_noop(self) -> None:
        backend = _make_backend()
        backend.delete_session("nope")  # should not raise


class TestListSessions:
    def test_list_empty(self) -> None:
        backend = _make_backend()
        assert backend.list_sessions() == []

    def test_list_multiple(self) -> None:
        backend = _make_backend()
        for sid in ["a", "b", "c"]:
            backend.save_session(sid, _make_state(sid))

        result = backend.list_sessions()
        assert set(result) == {"a", "b", "c"}


class TestCleanupExpired:
    def test_cleanup_removes_old(self) -> None:
        backend = _make_backend()
        state = _make_state("old")
        # Backdate last_activity so it's expired
        state.last_activity = datetime.now(timezone.utc) - timedelta(seconds=100)
        backend.save_session("old", state)

        removed = backend.cleanup_expired(ttl_seconds=10)
        assert removed == 1
        assert backend.load_session("old") is None

    def test_cleanup_keeps_fresh(self) -> None:
        backend = _make_backend()
        state = _make_state("fresh")
        backend.save_session("fresh", state)

        removed = backend.cleanup_expired(ttl_seconds=3600)
        assert removed == 0
        assert backend.load_session("fresh") is not None

    def test_cleanup_mixed(self) -> None:
        backend = _make_backend()

        old_state = _make_state("old")
        old_state.last_activity = datetime.now(timezone.utc) - timedelta(seconds=200)
        backend.save_session("old", old_state)

        fresh_state = _make_state("fresh")
        backend.save_session("fresh", fresh_state)

        removed = backend.cleanup_expired(ttl_seconds=60)
        assert removed == 1
        assert backend.load_session("old") is None
        assert backend.load_session("fresh") is not None


class TestWALMode:
    def test_wal_mode_enabled(self) -> None:
        backend = _make_backend()
        import sqlite3

        conn = sqlite3.connect(backend._db_path)
        try:
            cur = conn.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            assert mode == "wal"
        finally:
            conn.close()
