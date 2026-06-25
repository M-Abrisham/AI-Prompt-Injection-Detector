"""Per-conversation canary tokens with TTL for precise leak attribution.

Each conversation/session gets a unique canary. Expired canaries are
automatically cleaned up. This enables pinpointing WHICH conversation
leaked the system prompt.

Gated by ``NA0S_CANARY_SESSION=1`` env var (default: disabled).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

from na0s.canary.leak_detection import is_canary_present
from na0s.canary.manager import CanaryManager, CanaryToken


class SessionCanaryManager:
    """Manage per-conversation canary tokens with TTL-based expiration.

    Each session (conversation) receives a unique canary token that
    expires after a configurable TTL.  When a canary is detected in
    output, the originating session can be identified immediately.
    """

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self._default_ttl = default_ttl_seconds
        self._sessions: Dict[str, Dict] = {}
        # Internal CanaryManager used for generation and detection
        self._manager = CanaryManager()

    # ---- feature gate -----------------------------------------------------

    @staticmethod
    def is_enabled() -> bool:
        """Return True if the ``NA0S_CANARY_SESSION`` env var is set to ``1``."""
        return os.environ.get("NA0S_CANARY_SESSION", "0") == "1"

    # ---- session lifecycle ------------------------------------------------

    def create_session(
        self,
        session_id: str,
        system_prompt: str,
        ttl: Optional[int] = None,
    ) -> Tuple[str, CanaryToken]:
        """Create a canary-protected session.

        Parameters
        ----------
        session_id:
            Unique identifier for the conversation/session.
        system_prompt:
            The system prompt to inject the canary into.
        ttl:
            Time-to-live in seconds.  Defaults to *default_ttl_seconds*.

        Returns
        -------
        (modified_prompt, canary_token)
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        # Use the first 4 chars of session_id as a custom prefix
        prefix = session_id[:4].upper() if len(session_id) >= 4 else session_id.upper()

        modified_prompt, canary = self._manager.inject_into_prompt(
            system_prompt, prefix=prefix
        )

        self._sessions[session_id] = {
            "canary": canary,
            "expires_at": time.time() + effective_ttl,
        }
        return modified_prompt, canary

    def get_session(self, session_id: str) -> Optional[CanaryToken]:
        """Return the canary for *session_id*, or ``None`` if not found / expired."""
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        if time.time() > entry["expires_at"]:
            return None
        return entry["canary"]

    # ---- detection --------------------------------------------------------

    def check_session_output(self, output_text: str) -> List[Dict]:
        """Check output against all active (non-expired) session canaries.

        Detects each session canary in any supported form (exact OR
        encoded -- base64, hex, reversed, ROT13, unicode-escape, URL,
        partial) via the shared
        :func:`na0s.canary.leak_detection.is_canary_present` helper, so
        encoded leaks are caught with full parity to
        ``CanaryManager._is_present`` rather than only an exact substring.

        Returns
        -------
        list[dict]
            Each dict has keys ``session_id``, ``canary_token``, ``triggered``.
        """
        now = time.time()
        results: List[Dict] = []
        if not output_text:
            return results
        for session_id, entry in self._sessions.items():
            if now > entry["expires_at"]:
                continue  # skip expired
            canary: CanaryToken = entry["canary"]
            if is_canary_present(canary, output_text):
                canary.record_trigger()
                results.append({
                    "session_id": session_id,
                    "canary_token": canary.token,
                    "triggered": True,
                })
        return results

    # ---- maintenance ------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove expired session canaries.

        Returns
        -------
        int
            Number of sessions removed.
        """
        now = time.time()
        expired_ids = [
            sid for sid, entry in self._sessions.items()
            if now > entry["expires_at"]
        ]
        for sid in expired_ids:
            del self._sessions[sid]
        return len(expired_ids)
