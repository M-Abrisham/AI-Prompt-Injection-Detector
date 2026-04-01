"""Layer 16 SQLiteBackend -- persistent session storage via sqlite3.  # LAYER16

Uses Python's built-in sqlite3 module (no external dependencies).
WAL mode is enabled for concurrent readers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Optional

from na0s.layer16.config import SQLITE_DB_PATH
from na0s.layer16.models import ConversationState
from na0s.layer16.state import from_dict, to_dict
from na0s.layer16.storage.base import StorageBackend


class SQLiteBackend(StorageBackend):
    """SQLite-backed session storage.

    Uses WAL journal mode for concurrent read access and serializes
    ConversationState via ``to_dict()`` / ``from_dict()`` + JSON.

    Args:
        db_path: Path to the SQLite database file.
            Defaults to ``SQLITE_DB_PATH`` from config.
    """

    def __init__(self, db_path: str = SQLITE_DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create the sessions table if it does not exist."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id  TEXT PRIMARY KEY,
                        state_json  TEXT NOT NULL,
                        created_at  TEXT NOT NULL,
                        last_activity TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save_session(self, session_id: str, state: ConversationState) -> None:
        """Persist a session state to SQLite.

        Uses INSERT OR REPLACE (upsert) so both new and existing
        sessions are handled.

        Args:
            session_id: Unique session identifier.
            state: The conversation state to store.
        """
        state_json = json.dumps(to_dict(state))
        created_at = state.created_at.isoformat()
        last_activity = state.last_activity.isoformat()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO sessions
                        (session_id, state_json, created_at, last_activity)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, state_json, created_at, last_activity),
                )
                conn.commit()
            finally:
                conn.close()

    def load_session(self, session_id: str) -> Optional[ConversationState]:
        """Load a session state by ID.

        Args:
            session_id: The session to retrieve.

        Returns:
            The ConversationState, or None if not found.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT state_json FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                row = cur.fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        return from_dict(json.loads(row[0]))

    def delete_session(self, session_id: str) -> None:
        """Delete a session by ID. No-op if not found.

        Args:
            session_id: The session to remove.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
            finally:
                conn.close()

    def list_sessions(self) -> List[str]:
        """List all stored session IDs.

        Returns:
            List of session_id strings.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("SELECT session_id FROM sessions")
                return [row[0] for row in cur.fetchall()]
            finally:
                conn.close()

    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove sessions whose last_activity exceeds the TTL.

        Uses a SQL DELETE with a datetime comparison so the database
        engine does the heavy lifting.

        Args:
            ttl_seconds: Maximum age in seconds since last activity.

        Returns:
            Count of sessions removed.
        """
        cutoff = datetime.now(timezone.utc).isoformat()
        # We compute the cutoff in Python to avoid SQL datetime arithmetic
        # portability issues.  Instead, load candidates and check in Python.
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT session_id, last_activity FROM sessions"
                )
                rows = cur.fetchall()

                now = datetime.now(timezone.utc)
                expired_ids = []
                for session_id, last_activity_str in rows:
                    last_activity = datetime.fromisoformat(last_activity_str)
                    # Ensure timezone-aware comparison
                    if last_activity.tzinfo is None:
                        last_activity = last_activity.replace(tzinfo=timezone.utc)
                    elapsed = (now - last_activity).total_seconds()
                    if elapsed > ttl_seconds:
                        expired_ids.append(session_id)

                if expired_ids:
                    placeholders = ",".join("?" for _ in expired_ids)
                    conn.execute(
                        f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
                        expired_ids,
                    )
                    conn.commit()

                return len(expired_ids)
            finally:
                conn.close()
