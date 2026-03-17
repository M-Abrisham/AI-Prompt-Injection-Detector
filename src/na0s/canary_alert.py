"""Canary alert mechanism -- webhook/callback on canary trigger.

Sends real-time alerts when a canary token is detected in output.
Supports webhook URLs and local callback functions.

Gated by ``NA0S_CANARY_ALERT=1`` env var (default: disabled).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Callable, Dict, List

from na0s.canary import CanaryToken

logger = logging.getLogger(__name__)


class CanaryAlertManager:
    """Dispatch alerts when a canary token is triggered.

    Supports two alert channels:

    1. **Local callbacks** -- synchronous Python callables invoked
       immediately when a trigger is detected.
    2. **Webhook URLs** -- URLs that *would* receive a POST request.
       In this implementation the URL is only logged (no actual HTTP
       call is made), keeping the module dependency-free.
    """

    def __init__(self) -> None:
        self._callbacks: List[Callable[[CanaryToken, str], None]] = []
        self._webhooks: List[str] = []
        self._alert_history: List[Dict] = []

    # ---- feature gate -----------------------------------------------------

    @staticmethod
    def is_enabled() -> bool:
        """Return True if ``NA0S_CANARY_ALERT`` env var is ``1``."""
        return os.environ.get("NA0S_CANARY_ALERT", "0") == "1"

    # ---- registration -----------------------------------------------------

    def register_callback(
        self, callback: Callable[[CanaryToken, str], None]
    ) -> None:
        """Register a local callback to be invoked on canary trigger.

        Parameters
        ----------
        callback:
            A callable that receives ``(canary_token, context_string)``.
        """
        self._callbacks.append(callback)

    def register_webhook(self, url: str) -> None:
        """Register a webhook URL for canary trigger alerts.

        The URL is stored but no HTTP request is made at trigger time --
        the call is only logged.
        """
        self._webhooks.append(url)

    # ---- triggering -------------------------------------------------------

    def on_trigger(self, canary: CanaryToken, context: str = "") -> None:
        """Fire all registered alert channels.

        Parameters
        ----------
        canary:
            The canary token that was triggered.
        context:
            Optional context string (e.g., the output snippet).
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Invoke local callbacks
        for cb in self._callbacks:
            try:
                cb(canary, context)
            except Exception:
                logger.exception("Canary alert callback raised an exception")

        # Log webhook URLs (no actual HTTP call)
        for url in self._webhooks:
            logger.info(
                "Canary alert: would POST to %s for token %s",
                url,
                canary.token,
            )

        # Record in history
        self._alert_history.append({
            "token": canary.token,
            "context": context,
            "timestamp": timestamp,
            "callbacks_fired": len(self._callbacks),
            "webhooks_logged": len(self._webhooks),
        })

    # ---- history ----------------------------------------------------------

    def alert_history(self) -> List[Dict]:
        """Return the list of all alerts that have been dispatched."""
        return list(self._alert_history)
