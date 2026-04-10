"""Prompt template integrity checker — SHA-256 manifest verification.

Maintains a manifest of known-good prompt template hashes. At runtime,
verifies templates haven't been modified. Also scans templates for
injection patterns that shouldn't be in legitimate templates.

Gated by ``NA0S_TEMPLATE_INTEGRITY=1`` env var (default: disabled).
"""

from __future__ import annotations

import hashlib
import json
import os
import re


_SUSPICIOUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore previous instructions", re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE)),
    ("system prompt", re.compile(r"system\s+prompt", re.IGNORECASE)),
    ("injected new instructions", re.compile(r"\n\n\[new instructions\]", re.IGNORECASE)),
    ("unsanitized user_input placeholder", re.compile(r"\{\{user_input\}\}")),
]


class PromptTemplateIntegrityChecker:
    """Register, verify, and scan prompt templates for integrity."""

    def __init__(self, manifest_path: str = "data/prompt_manifest.json") -> None:
        self._manifest_path = manifest_path
        self._hashes: dict[str, str] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def is_enabled() -> bool:
        """Return True when template integrity checking is activated via env var."""
        return os.environ.get("NA0S_TEMPLATE_INTEGRITY", "0") == "1"

    # ------------------------------------------------------------------
    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    # ------------------------------------------------------------------
    def register_template(self, name: str, template: str) -> str:
        """Register *template* under *name* and return its SHA-256 hash."""
        h = self._hash(template)
        self._hashes[name] = h
        return h

    # ------------------------------------------------------------------
    def verify_template(self, name: str, template: str) -> dict:
        """Verify *template* against the registered hash for *name*."""
        actual = self._hash(template)
        expected = self._hashes.get(name, "")
        if not expected:
            return {
                "valid": False,
                "reason": f"template '{name}' not registered",
                "expected_hash": expected,
                "actual_hash": actual,
            }
        valid = actual == expected
        reason = "" if valid else "template content has been modified"
        return {
            "valid": valid,
            "reason": reason,
            "expected_hash": expected,
            "actual_hash": actual,
        }

    # ------------------------------------------------------------------
    def scan_template(self, template: str) -> dict:
        """Scan *template* for suspicious injection patterns.

        Returns ``{"clean": bool, "suspicious_patterns": [...]}``.
        """
        found: list[str] = []
        for label, pattern in _SUSPICIOUS_PATTERNS:
            if pattern.search(template):
                found.append(label)
        return {"clean": len(found) == 0, "suspicious_patterns": found}

    # ------------------------------------------------------------------
    def save_manifest(self) -> None:
        """Persist the current hash manifest to *manifest_path* as JSON."""
        directory = os.path.dirname(self._manifest_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self._manifest_path, "w", encoding="utf-8") as fh:
            json.dump(self._hashes, fh, indent=2)

    # ------------------------------------------------------------------
    def load_manifest(self) -> None:
        """Load the hash manifest from *manifest_path*."""
        with open(self._manifest_path, encoding="utf-8") as fh:
            self._hashes = json.load(fh)
