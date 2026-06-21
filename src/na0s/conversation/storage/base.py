"""Layer 16 StorageBackend — abstract base for session persistence.

All storage backends implement this sync interface. Na0S does not use
asyncio; all methods are blocking.
"""

from __future__ import annotations

import abc
from typing import List, Optional

from na0s.conversation.models import ConversationState


class StorageBackend(abc.ABC):
    """Abstract base class for session storage backends.

    Subclasses must implement all five methods. Implementations must
    be thread-safe if used with SessionManager.
    """

    @abc.abstractmethod
    def save_session(self, session_id: str, state: ConversationState) -> None:
        """Persist a session state.

        Args:
            session_id: Unique session identifier.
            state: The conversation state to store.
        """

    @abc.abstractmethod
    def load_session(self, session_id: str) -> Optional[ConversationState]:
        """Load a session state by ID.

        Args:
            session_id: The session to retrieve.

        Returns:
            The ConversationState, or None if not found.
        """

    @abc.abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete a session by ID. No-op if not found.

        Args:
            session_id: The session to remove.
        """

    @abc.abstractmethod
    def list_sessions(self) -> List[str]:
        """List all stored session IDs.

        Returns:
            List of session_id strings.
        """

    @abc.abstractmethod
    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove sessions whose last_activity exceeds the TTL.

        Args:
            ttl_seconds: Maximum age in seconds since last activity.

        Returns:
            Count of sessions removed.
        """
