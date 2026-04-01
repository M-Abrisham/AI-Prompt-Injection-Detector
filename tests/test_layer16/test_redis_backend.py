"""Tests for Layer 16 RedisBackend."""  # LAYER16

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module if redis is not installed
redis = pytest.importorskip("redis")

from na0s.layer16.models import ConversationState
from na0s.layer16.state import add_turn, to_dict
from na0s.layer16.storage.redis_backend import RedisBackend, _KEY_PREFIX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(session_id: str = "test-session") -> ConversationState:
    """Create a simple ConversationState for testing."""
    return ConversationState(
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        last_activity=datetime.now(timezone.utc),
    )


def _mock_redis() -> MagicMock:
    """Return a mock redis.Redis client with basic get/set/delete/keys."""
    store: dict[str, str] = {}
    ttls: dict[str, int] = {}

    mock = MagicMock(spec=redis.Redis)

    def _set(key, value):
        store[key] = value

    def _setex(key, ttl, value):
        store[key] = value
        ttls[key] = ttl

    def _get(key):
        return store.get(key)

    def _delete(*keys):
        for k in keys:
            store.pop(k, None)
            ttls.pop(k, None)
        return len(keys)

    def _keys(pattern):
        import fnmatch
        return [k for k in store if fnmatch.fnmatch(k, pattern)]

    mock.set.side_effect = _set
    mock.setex.side_effect = _setex
    mock.get.side_effect = _get
    mock.delete.side_effect = _delete
    mock.keys.side_effect = _keys

    # Expose the internal store for assertions
    mock._store = store
    mock._ttls = ttls
    return mock


# ---------------------------------------------------------------------------
# Tests: basic CRUD
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load(self) -> None:
        mock = _mock_redis()
        backend = RedisBackend(client=mock)
        state = _make_state("s1")
        add_turn(state, "hello", risk_score=0.1, label="safe")

        backend.save_session("s1", state)
        loaded = backend.load_session("s1")

        assert loaded is not None
        assert loaded.session_id == "s1"
        assert loaded.turn_count == 1
        assert loaded.turns[0].text == "hello"

    def test_load_missing_returns_none(self) -> None:
        mock = _mock_redis()
        backend = RedisBackend(client=mock)
        assert backend.load_session("nonexistent") is None

    def test_save_with_ttl(self) -> None:
        mock = _mock_redis()
        backend = RedisBackend(client=mock, default_ttl=300)
        state = _make_state("s1")
        backend.save_session("s1", state)

        # setex should have been called (not plain set)
        mock.setex.assert_called_once()
        assert mock._ttls[f"{_KEY_PREFIX}s1"] == 300


class TestDelete:
    def test_delete_removes(self) -> None:
        mock = _mock_redis()
        backend = RedisBackend(client=mock)
        state = _make_state("s1")
        backend.save_session("s1", state)

        backend.delete_session("s1")
        assert backend.load_session("s1") is None

    def test_delete_missing_is_noop(self) -> None:
        mock = _mock_redis()
        backend = RedisBackend(client=mock)
        backend.delete_session("nope")  # should not raise


class TestListSessions:
    def test_list_empty(self) -> None:
        mock = _mock_redis()
        backend = RedisBackend(client=mock)
        assert backend.list_sessions() == []

    def test_list_multiple(self) -> None:
        mock = _mock_redis()
        backend = RedisBackend(client=mock)
        for sid in ["a", "b", "c"]:
            backend.save_session(sid, _make_state(sid))

        result = backend.list_sessions()
        assert set(result) == {"a", "b", "c"}


class TestCleanupExpired:
    def test_cleanup_with_ttl_returns_zero(self) -> None:
        """When default_ttl is set, Redis handles expiry natively."""
        mock = _mock_redis()
        backend = RedisBackend(client=mock, default_ttl=60)
        state = _make_state("s1")
        backend.save_session("s1", state)

        removed = backend.cleanup_expired(ttl_seconds=10)
        assert removed == 0

    def test_cleanup_removes_old(self) -> None:
        """Without default_ttl, manual cleanup removes stale sessions."""
        mock = _mock_redis()
        backend = RedisBackend(client=mock)

        state = _make_state("old")
        state.last_activity = datetime.now(timezone.utc) - timedelta(seconds=200)
        backend.save_session("old", state)

        fresh = _make_state("fresh")
        backend.save_session("fresh", fresh)

        removed = backend.cleanup_expired(ttl_seconds=60)
        assert removed == 1
        assert backend.load_session("old") is None
        assert backend.load_session("fresh") is not None


class TestKeyFormat:
    def test_key_prefix(self) -> None:
        assert RedisBackend._key("abc123") == "na0s:session:abc123"
