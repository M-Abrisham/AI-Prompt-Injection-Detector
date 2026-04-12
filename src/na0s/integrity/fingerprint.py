"""FingerprintStore.db integrity checking (Layer 11).

SHA-256 sidecar verification and monitoring for fingerprint database files.
No env var gating -- always available.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Union


class FingerprintStoreIntegrity:
    """Integrity checking for fingerprint database files."""

    @staticmethod
    def compute_hash(db_path: Union[str, Path]) -> str:
        """Return SHA-256 hex digest of the database file."""
        h = hashlib.sha256()
        with open(db_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def _sidecar_path(cls, db_path: Union[str, Path]) -> Path:
        return Path(str(db_path) + ".sha256")

    @classmethod
    def save_hash(cls, db_path: Union[str, Path]) -> Path:
        """Write SHA-256 sidecar file next to the database."""
        digest = cls.compute_hash(db_path)
        sidecar = cls._sidecar_path(db_path)
        sidecar.write_text(digest)
        return sidecar

    @classmethod
    def verify_hash(cls, db_path: Union[str, Path]) -> bool:
        """Check that the sidecar hash matches the current file."""
        sidecar = cls._sidecar_path(db_path)
        if not sidecar.exists():
            return False
        expected = sidecar.read_text().strip()
        actual = cls.compute_hash(db_path)
        return expected == actual

    @classmethod
    def monitor(
        cls, db_path: Union[str, Path], interval_check: bool = False
    ) -> Dict[str, object]:
        """Return monitoring info: size, mtime, hash, sidecar_valid.

        The *interval_check* flag is reserved for future periodic-check
        behaviour; currently it has no effect.
        """
        p = Path(db_path)
        stat = p.stat()
        current_hash = cls.compute_hash(db_path)
        sidecar = cls._sidecar_path(db_path)
        sidecar_valid = False
        if sidecar.exists():
            sidecar_valid = sidecar.read_text().strip() == current_hash
        return {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "hash": current_hash,
            "sidecar_valid": sidecar_valid,
        }
