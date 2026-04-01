"""Layer 16 MemoryBackend — in-memory session storage.

Dict-based, thread-safe implementation of StorageBackend.
Respects max_sessions cap from config.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from na0s.layer16.config import MAX_SESSIONS
from na0s.layer16.models import ConversationState
from na0s.layer16.storage.base import StorageBackend


class MemoryBackend(StorageBackend):
    """In-memory storage backend backed by a plain dict.

    Thread-safe via threading.RLock. Enforces a configurable
    max_sessions cap; oldest sessions (by last_activity) are
    evicted when the cap is exceeded.

    Args:
        max_sessions: Maximum number of sessions to retain.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._lock = threading.RLock()
        self._store: Dict[str, ConversationState] = {}
        self._max_sessions = max_sessions

    def save_session(self, session_id: str, state: ConversationState) -> None:
        """Persist a session state in memory.

        If saving would exceed max_sessions and the session_id is new,
        the oldest session (by last_activity) is evicted first.

        Args:
            session_id: Unique session identifier.
            state: The conversation state to store.
        """
        with self._lock:
            # If this is a new session and we're at capacity, evict oldest
            if session_id not in self._store and len(self._store) >= self._max_sessions:
                self._evict_oldest()
            self._store[session_id] = state

    def load_session(self, session_id: str) -> Optional[ConversationState]:
        """Load a session state by ID.

        Args:
            session_id: The session to retrieve.

        Returns:
            The ConversationState, or None if not found.
        """
        with self._lock:
            return self._store.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session by ID. No-op if not found.

        Args:
            session_id: The session to remove.
        """
        with self._lock:
            self._store.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """List all stored session IDs.

        Returns:
            List of session_id strings.
        """
        with self._lock:
            return list(self._store.keys())

    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove sessions whose last_activity exceeds the TTL.

        Args:
            ttl_seconds: Maximum age in seconds since last activity.

        Returns:
            Count of sessions removed.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            expired = [
                sid
                for sid, state in self._store.items()
                if (now - state.last_activity).total_seconds() > ttl_seconds
            ]
            for sid in expired:
                del self._store[sid]
            return len(expired)

    def _evict_oldest(self) -> None:
        """Evict the session with the oldest last_activity. Must hold lock."""
        if not self._store:
            return
        oldest_sid = min(
            self._store, key=lambda sid: self._store[sid].last_activity
        )
        del self._store[oldest_sid]
