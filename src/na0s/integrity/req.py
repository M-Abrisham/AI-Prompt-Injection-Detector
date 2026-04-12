"""Requirements.txt integrity via SHA-256 sidecar (Layer 11).

Simple hash-based verification of requirements files.
No env var gating -- always available.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Union


class RequirementsIntegrity:
    """Hash-based verification of requirements.txt files."""

    @staticmethod
    def compute_hash(req_file: Union[str, Path]) -> str:
        """Return SHA-256 hex digest of the requirements file."""
        h = hashlib.sha256()
        with open(req_file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def _sidecar_path(cls, req_file: Union[str, Path]) -> Path:
        return Path(str(req_file) + ".sha256")

    @classmethod
    def save_hash(cls, req_file: Union[str, Path]) -> Path:
        """Write SHA-256 sidecar file next to the requirements file."""
        digest = cls.compute_hash(req_file)
        sidecar = cls._sidecar_path(req_file)
        sidecar.write_text(digest)
        return sidecar

    @classmethod
    def verify_hash(cls, req_file: Union[str, Path]) -> bool:
        """Check that the sidecar hash matches the current file."""
        sidecar = cls._sidecar_path(req_file)
        if not sidecar.exists():
            return False
        expected = sidecar.read_text().strip()
        actual = cls.compute_hash(req_file)
        return expected == actual

    @classmethod
    def detect_changes(cls, req_file: Union[str, Path]) -> Dict[str, object]:
        """Return dict with changed status and expected/actual hashes."""
        sidecar = cls._sidecar_path(req_file)
        actual = cls.compute_hash(req_file)
        if not sidecar.exists():
            return {"changed": True, "expected": "", "actual": actual}
        expected = sidecar.read_text().strip()
        return {
            "changed": expected != actual,
            "expected": expected,
            "actual": actual,
        }
