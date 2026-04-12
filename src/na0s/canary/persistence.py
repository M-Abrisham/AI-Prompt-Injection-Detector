"""Canary persistence -- save/load canary registry to disk.

Enables cross-session canary tracking. Canaries survive process restarts.
Uses JSON format for human-readable storage.

Gated by ``NA0S_CANARY_PERSIST=1`` env var (default: disabled).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from na0s.canary.manager import CanaryManager, CanaryToken


class PersistentCanaryStore:
    """Save and load :class:`CanaryManager` state to a JSON file."""

    def __init__(self, path: str = "data/canary/canary_registry.json") -> None:
        self._path = Path(path)

    # ---- feature gate -----------------------------------------------------

    @staticmethod
    def is_enabled() -> bool:
        """Return True if ``NA0S_CANARY_PERSIST`` env var is ``1``."""
        return os.environ.get("NA0S_CANARY_PERSIST", "0") == "1"

    # ---- persistence ------------------------------------------------------

    def save(self, manager: CanaryManager) -> None:
        """Serialize all canaries from *manager* to the JSON file.

        The file is created (along with parent directories) if it does not
        exist.  Format::

            {
                "version": 1,
                "saved_at": "<iso timestamp>",
                "canaries": [<canary.to_dict()>, ...]
            }
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict = {
            "version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "canaries": [c.to_dict() for c in manager.active_canaries],
        }
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def load(self) -> CanaryManager:
        """Deserialize a :class:`CanaryManager` from the JSON file.

        Returns
        -------
        CanaryManager
            A new manager populated with the persisted canaries.

        Raises
        ------
        FileNotFoundError
            If the registry file does not exist.
        """
        with open(self._path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        manager = CanaryManager()
        for entry in payload.get("canaries", []):
            canary = CanaryToken(
                token=entry["token"],
                created_at=entry.get("created_at", ""),
                triggered=entry.get("triggered", False),
                trigger_count=entry.get("trigger_count", 0),
                first_triggered_at=entry.get("first_triggered_at"),
                last_triggered_at=entry.get("last_triggered_at"),
            )
            manager._canaries.append(canary)
        return manager

    def exists(self) -> bool:
        """Return True if the registry file exists on disk."""
        return self._path.exists()
