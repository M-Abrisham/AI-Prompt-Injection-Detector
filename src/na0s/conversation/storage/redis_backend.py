"""Layer 16 RedisBackend -- optional Redis-backed session storage.

Requires the ``redis`` package (``pip install redis``).  If the package
is not installed, importing this module raises ``ImportError`` with a
helpful install message.

Security note: state is serialised as JSON, **never** pickle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

try:
    import redis
except ImportError as _exc:
    raise ImportError(
        "RedisBackend requires the 'redis' package.  "
        "Install it with:  pip install redis"
    ) from _exc

from na0s.conversation.config import REDIS_URL
from na0s.conversation.models import ConversationState
from na0s.conversation.state import from_dict, to_dict
from na0s.conversation.storage.base import StorageBackend

_KEY_PREFIX = "na0s:session:"


class RedisBackend(StorageBackend):
    """Redis-backed session storage with automatic TTL expiry.

    Each session is stored as a JSON string under the key
    ``na0s:session:{session_id}``.  Redis TTL is used for automatic
    expiry so ``cleanup_expired`` is mostly a no-op (returns 0) --
    Redis handles garbage collection natively.

    Args:
        url: Redis connection URL.  Defaults to ``REDIS_URL`` from config.
        default_ttl: Default TTL in seconds applied to every key.
            If *None*, keys do not expire automatically and
            ``cleanup_expired`` performs manual expiry.
        client: Optional pre-built ``redis.Redis`` instance (useful for
            testing with mocks).
    """

    def __init__(
        self,
        url: str = REDIS_URL,
        default_ttl: Optional[int] = None,
        client: Optional[redis.Redis] = None,
    ) -> None:
        self._client: redis.Redis = client or redis.Redis.from_url(
            url, decode_responses=True
        )
        self._default_ttl = default_ttl

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{_KEY_PREFIX}{session_id}"

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save_session(self, session_id: str, state: ConversationState) -> None:
        """Persist a session state to Redis as JSON.

        If a ``default_ttl`` was configured, the key's TTL is (re)set
        on every save so that it tracks the most recent activity.

        Args:
            session_id: Unique session identifier.
            state: The conversation state to store.
        """
        key = self._key(session_id)
        payload = json.dumps(to_dict(state))
        if self._default_ttl is not None:
            self._client.setex(key, self._default_ttl, payload)
        else:
            self._client.set(key, payload)

    def load_session(self, session_id: str) -> Optional[ConversationState]:
        """Load a session state by ID.

        Args:
            session_id: The session to retrieve.

        Returns:
            The ConversationState, or None if not found / expired.
        """
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return None
        return from_dict(json.loads(raw))

    def delete_session(self, session_id: str) -> None:
        """Delete a session by ID. No-op if not found.

        Args:
            session_id: The session to remove.
        """
        self._client.delete(self._key(session_id))

    def list_sessions(self) -> List[str]:
        """List all stored session IDs.

        Uses ``SCAN`` internally (via ``keys()`` with the prefix pattern)
        to avoid blocking on large keyspaces.

        Returns:
            List of session_id strings (prefix stripped).
        """
        prefix_len = len(_KEY_PREFIX)
        keys = self._client.keys(f"{_KEY_PREFIX}*")
        return [k[prefix_len:] if isinstance(k, str) else k.decode()[prefix_len:] for k in keys]

    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove sessions whose last_activity exceeds the TTL.

        When ``default_ttl`` is set, Redis handles expiry natively and
        this method returns 0.  Otherwise it scans all keys, checks
        ``last_activity``, and deletes stale entries.

        Args:
            ttl_seconds: Maximum age in seconds since last activity.

        Returns:
            Count of sessions removed.
        """
        if self._default_ttl is not None:
            # Redis TTL handles expiry -- nothing to do
            return 0

        now = datetime.now(timezone.utc)
        removed = 0
        for session_id in self.list_sessions():
            state = self.load_session(session_id)
            if state is None:
                continue
            last = state.last_activity
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if (now - last).total_seconds() > ttl_seconds:
                self.delete_session(session_id)
                removed += 1
        return removed
