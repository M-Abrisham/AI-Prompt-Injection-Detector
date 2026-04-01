"""Tests for Layer 16 MemoryBackend."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from na0s.layer16.models import ConversationState
from na0s.layer16.storage.memory_backend import MemoryBackend


def _make_state(
    session_id: str, age_seconds: float = 0.0
) -> ConversationState:
    """Create a ConversationState, optionally aged for TTL testing."""
    now = datetime.now(timezone.utc)
    activity = now - timedelta(seconds=age_seconds)
    return ConversationState(
        session_id=session_id,
        created_at=activity,
        last_activity=activity,
    )


class TestSaveAndLoad:
    def test_save_and_load(self) -> None:
        backend = MemoryBackend()
        state = _make_state("s1")
        backend.save_session("s1", state)
        loaded = backend.load_session("s1")
        assert loaded is not None
        assert loaded.session_id == "s1"

    def test_load_missing(self) -> None:
        backend = MemoryBackend()
        assert backend.load_session("nope") is None

    def test_overwrite(self) -> None:
        backend = MemoryBackend()
        backend.save_session("s1", _make_state("s1"))
        new_state = _make_state("s1")
        new_state.cumulative_risk = 0.99
        backend.save_session("s1", new_state)
        loaded = backend.load_session("s1")
        assert loaded is not None
        assert loaded.cumulative_risk == 0.99


class TestDelete:
    def test_delete_existing(self) -> None:
        backend = MemoryBackend()
        backend.save_session("s1", _make_state("s1"))
        backend.delete_session("s1")
        assert backend.load_session("s1") is None

    def test_delete_missing_is_noop(self) -> None:
        backend = MemoryBackend()
        backend.delete_session("nope")  # should not raise


class TestListSessions:
    def test_list_sessions(self) -> None:
        backend = MemoryBackend()
        backend.save_session("s1", _make_state("s1"))
        backend.save_session("s2", _make_state("s2"))
        sessions = backend.list_sessions()
        assert set(sessions) == {"s1", "s2"}

    def test_list_empty(self) -> None:
        backend = MemoryBackend()
        assert backend.list_sessions() == []


class TestCleanupExpired:
    def test_cleanup_removes_old(self) -> None:
        backend = MemoryBackend()
        backend.save_session("old", _make_state("old", age_seconds=120))
        backend.save_session("fresh", _make_state("fresh", age_seconds=0))
        removed = backend.cleanup_expired(ttl_seconds=60)
        assert removed == 1
        assert backend.load_session("old") is None
        assert backend.load_session("fresh") is not None

    def test_cleanup_nothing_expired(self) -> None:
        backend = MemoryBackend()
        backend.save_session("s1", _make_state("s1"))
        removed = backend.cleanup_expired(ttl_seconds=3600)
        assert removed == 0

    def test_cleanup_all_expired(self) -> None:
        backend = MemoryBackend()
        backend.save_session("a", _make_state("a", age_seconds=200))
        backend.save_session("b", _make_state("b", age_seconds=300))
        removed = backend.cleanup_expired(ttl_seconds=60)
        assert removed == 2
        assert backend.list_sessions() == []


class TestMaxSessionsCap:
    def test_evicts_oldest_when_full(self) -> None:
        backend = MemoryBackend(max_sessions=2)
        # s1 is oldest
        backend.save_session("s1", _make_state("s1", age_seconds=10))
        backend.save_session("s2", _make_state("s2", age_seconds=5))
        # Adding s3 should evict s1 (oldest by last_activity)
        backend.save_session("s3", _make_state("s3", age_seconds=0))
        assert backend.load_session("s1") is None
        assert backend.load_session("s2") is not None
        assert backend.load_session("s3") is not None

    def test_overwrite_does_not_evict(self) -> None:
        backend = MemoryBackend(max_sessions=2)
        backend.save_session("s1", _make_state("s1"))
        backend.save_session("s2", _make_state("s2"))
        # Overwrite s1 — should not evict anything
        new_state = _make_state("s1")
        new_state.cumulative_risk = 0.5
        backend.save_session("s1", new_state)
        assert len(backend.list_sessions()) == 2
        loaded = backend.load_session("s1")
        assert loaded is not None
        assert loaded.cumulative_risk == 0.5

    def test_max_sessions_respected(self) -> None:
        backend = MemoryBackend(max_sessions=3)
        for i in range(10):
            backend.save_session(f"s{i}", _make_state(f"s{i}", age_seconds=10 - i))
        assert len(backend.list_sessions()) == 3


class TestThreadSafety:
    def test_concurrent_saves(self) -> None:
        backend = MemoryBackend(max_sessions=200)
        errors: list[Exception] = []

        def save(i: int) -> str:
            sid = f"session-{i}"
            backend.save_session(sid, _make_state(sid))
            return sid

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(save, i) for i in range(100)]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    errors.append(e)

        assert len(errors) == 0
        assert len(backend.list_sessions()) == 100

    def test_concurrent_save_and_delete(self) -> None:
        backend = MemoryBackend(max_sessions=200)
        # Pre-populate
        for i in range(50):
            backend.save_session(f"s{i}", _make_state(f"s{i}"))
        errors: list[Exception] = []

        def save(i: int) -> None:
            backend.save_session(f"new-{i}", _make_state(f"new-{i}"))

        def delete(i: int) -> None:
            backend.delete_session(f"s{i}")

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = []
            for i in range(50):
                futures.append(pool.submit(save, i))
                futures.append(pool.submit(delete, i))
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    errors.append(e)

        assert len(errors) == 0
        # All original sessions deleted, all new ones added
        sessions = backend.list_sessions()
        for i in range(50):
            assert f"s{i}" not in sessions
            assert f"new-{i}" in sessions

    def test_concurrent_save_with_max_cap(self) -> None:
        backend = MemoryBackend(max_sessions=10)
        errors: list[Exception] = []

        def save(i: int) -> None:
            backend.save_session(f"s{i}", _make_state(f"s{i}", age_seconds=100 - i))

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(save, i) for i in range(50)]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    errors.append(e)

        assert len(errors) == 0
        assert len(backend.list_sessions()) <= 10
