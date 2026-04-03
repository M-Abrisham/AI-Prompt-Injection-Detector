"""Layer 16 SessionManager — session lifecycle management.

Thread-safe session store with lazy TTL expiry. Each session holds a
ConversationState that accumulates turns over time.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from na0s.layer16.models import ConversationState, SessionConfig
from na0s.layer16.state import add_turn


MAX_SESSIONS = 10_000


class SessionManager:
    """Manages conversation sessions with lazy TTL expiry.

    Thread-safe via threading.RLock. Sessions that exceed their TTL
    are lazily cleaned up on access and via explicit cleanup_expired().

    Args:
        config: Optional SessionConfig overrides. Uses defaults from config.py.
    """

    def __init__(self, config: Optional[SessionConfig] = None) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, ConversationState] = {}
        self._config = config or SessionConfig()

    def create_session(self, config: Optional[SessionConfig] = None) -> str:
        """Create a new conversation session.

        Args:
            config: Optional per-session config (currently unused beyond
                    the manager-level config, reserved for future use).

        Returns:
            The new session_id (uuid4 string).

        Raises:
            RuntimeError: If the maximum session limit has been reached.
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        state = ConversationState(
            session_id=session_id,
            created_at=now,
            last_activity=now,
        )
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                raise RuntimeError("Maximum session limit reached")
            self._sessions[session_id] = state
        return session_id

    def _auto_create_session(self, session_id: str) -> None:
        """Create a session with a caller-supplied ID (auto-create path).

        Used by ConversationSecurityMonitor when a session_id is referenced
        but does not yet exist.  Thread-safe with TOCTOU protection.

        Raises:
            RuntimeError: If the maximum session limit has been reached.
        """
        now = datetime.now(timezone.utc)
        new_state = ConversationState(
            session_id=session_id,
            created_at=now,
            last_activity=now,
        )
        with self._lock:
            # Re-check under lock — another thread may have created it
            if session_id in self._sessions:
                return
            if len(self._sessions) >= MAX_SESSIONS:
                raise RuntimeError("Maximum session limit reached")
            self._sessions[session_id] = new_state

    def get_session(self, session_id: str) -> Optional[ConversationState]:
        """Retrieve a session by ID, returning None if expired or missing.

        Performs lazy expiry check: if the session's TTL has elapsed,
        it is removed and None is returned.

        Args:
            session_id: The session identifier.

        Returns:
            The ConversationState, or None if not found / expired.
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            if self._is_expired(state):
                del self._sessions[session_id]
                return None
            return state

    def update_session(
        self,
        session_id: str,
        text: str,
        risk_score: float = 0.0,
        label: str = "safe",
        flags: Optional[List[str]] = None,
    ) -> Optional[ConversationState]:
        """Add a turn to an existing session.

        Args:
            session_id: The session to update.
            text: Raw text of the new turn.
            risk_score: Risk score from single-turn analysis.
            label: Classification label.
            flags: Optional flag strings.

        Returns:
            The updated ConversationState, or None if session not found / expired.
        """
        with self._lock:
            state = self.get_session(session_id)
            if state is None:
                return None
            add_turn(state, text, risk_score, label, flags)
            return state

    def expire_session(self, session_id: str) -> None:
        """Explicitly expire and remove a session.

        Args:
            session_id: The session to remove.
        """
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        """Remove all sessions that have exceeded their TTL.

        Returns:
            Count of sessions removed.
        """
        with self._lock:
            expired_ids = [
                sid
                for sid, state in self._sessions.items()
                if self._is_expired(state)
            ]
            for sid in expired_ids:
                del self._sessions[sid]
            return len(expired_ids)

    def list_active_sessions(self) -> List[str]:
        """Return IDs of all non-expired sessions.

        Returns:
            List of session_id strings.
        """
        with self._lock:
            active = []
            expired = []
            for sid, state in self._sessions.items():
                if self._is_expired(state):
                    expired.append(sid)
                else:
                    active.append(sid)
            # Lazy cleanup of expired found during listing
            for sid in expired:
                del self._sessions[sid]
            return active

    @property
    def active_session_count(self) -> int:
        """Number of non-expired sessions currently tracked."""
        with self._lock:
            # Don't mutate during a property read — just count non-expired
            return sum(
                1
                for state in self._sessions.values()
                if not self._is_expired(state)
            )

    def _is_expired(self, state: ConversationState) -> bool:
        """Check if a session has exceeded the TTL."""
        elapsed = (datetime.now(timezone.utc) - state.last_activity).total_seconds()
        return elapsed > self._config.ttl_seconds
