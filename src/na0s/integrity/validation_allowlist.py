"""Persistent allowlist database for positive validation (Layer 8, P2).

Provides a hash-based allowlist so that text previously reviewed and
approved can bypass positive validation on subsequent encounters.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional


_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "allowlist.json",
)


class AllowlistDB:
    """In-memory allowlist with optional JSON file persistence.

    Each entry maps a SHA-256 hex digest of the text to a reason string
    explaining why the text was allowlisted.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path: str = path or _DEFAULT_PATH
        self._entries: Dict[str, str] = {}

    # ---- core API ---------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        """Return the SHA-256 hex digest of *text*."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def add(self, text_hash: str, reason: str) -> None:
        """Add *text_hash* to the allowlist with the given *reason*."""
        self._entries[text_hash] = reason

    def check(self, text: str) -> bool:
        """Return True if the SHA-256 hash of *text* is in the allowlist."""
        return self._hash(text) in self._entries

    def remove(self, text_hash: str) -> bool:
        """Remove an entry. Returns True if it existed."""
        return self._entries.pop(text_hash, None) is not None

    def __len__(self) -> int:
        return len(self._entries)

    # ---- persistence ------------------------------------------------------

    def load(self, path: Optional[str] = None) -> None:
        """Load allowlist from a JSON file.

        If the file does not exist the allowlist is left empty (no error).
        """
        target = path or self.path
        if not os.path.isfile(target):
            return
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            self._entries = data

    def save(self, path: Optional[str] = None) -> None:
        """Persist the allowlist to a JSON file."""
        target = path or self.path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, indent=2)

    # ---- convenience ------------------------------------------------------

    def add_text(self, text: str, reason: str) -> str:
        """Hash *text* and add it to the allowlist. Returns the hash."""
        h = self._hash(text)
        self.add(h, reason)
        return h
