"""Model backup and rollback for supply-chain integrity (Layer 11).

Provides timestamped backups of model files (and their sidecar metadata),
listing, restore, and cleanup operations.

Automatic backup-on-deploy behaviour is gated behind ``NA0S_MODEL_ROLLBACK=1``.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_SIDECAR_SUFFIXES = (".sha256", ".hmac", ".meta.json")


def auto_backup_enabled() -> bool:
    """Return *True* when automatic backup-on-deploy is turned on."""
    return os.environ.get("NA0S_MODEL_ROLLBACK", "0") == "1"


class ModelRollback:
    """Backup and restore previous model versions.

    Parameters
    ----------
    backup_dir : str | Path | None
        Directory for storing backups.  Defaults to ``~/.na0s/model_backups/``.
    """

    def __init__(self, backup_dir: Optional[str | Path] = None) -> None:
        if backup_dir is None:
            backup_dir = Path.home() / ".na0s" / "model_backups"
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def backup(self, model_path: str | Path) -> Path:
        """Copy *model_path* (and sidecars) into the backup directory.

        The backup filename includes an ISO-8601 timestamp with colons
        replaced by hyphens so the path is filesystem-safe.

        Returns the path of the backed-up model file.
        """
        src = Path(model_path)
        if not src.exists():
            raise FileNotFoundError(f"Model file not found: {src}")

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        backup_name = f"{src.name}.{ts}"
        dst = self.backup_dir / backup_name
        shutil.copy2(src, dst)

        # Copy any sidecar files that exist alongside the model
        for suffix in _SIDECAR_SUFFIXES:
            sidecar = src.parent / (src.name + suffix)
            if sidecar.exists():
                shutil.copy2(sidecar, self.backup_dir / (backup_name + suffix))

        return dst

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_backups(self, model_name: str) -> List[Dict[str, Any]]:
        """Return backup entries for *model_name*, newest first.

        Each entry is a dict with keys ``path``, ``timestamp``, and
        ``size_bytes``.
        """
        entries: list[Dict[str, Any]] = []
        for p in self.backup_dir.iterdir():
            if not p.name.startswith(model_name + "."):
                continue
            # Skip sidecar backups
            if any(p.name.endswith(s) for s in _SIDECAR_SUFFIXES):
                continue
            # Extract timestamp portion after model_name.
            rest = p.name[len(model_name) + 1 :]
            try:
                ts = datetime.strptime(rest, "%Y-%m-%dT%H-%M-%S")
            except ValueError:
                continue
            entries.append(
                {
                    "path": p,
                    "timestamp": ts.isoformat(),
                    "size_bytes": p.stat().st_size,
                }
            )
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        return entries

    def latest_backup(self, model_name: str) -> Optional[Path]:
        """Return the path of the most recent backup, or *None*."""
        backups = self.list_backups(model_name)
        if not backups:
            return None
        return backups[0]["path"]

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, backup_path: str | Path, target_path: str | Path) -> Path:
        """Copy *backup_path* (and sidecars) to *target_path*."""
        src = Path(backup_path)
        dst = Path(target_path)
        if not src.exists():
            raise FileNotFoundError(f"Backup not found: {src}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        # Restore sidecars
        for suffix in _SIDECAR_SUFFIXES:
            sidecar_src = src.parent / (src.name + suffix)
            if sidecar_src.exists():
                sidecar_dst = dst.parent / (dst.name + suffix)
                shutil.copy2(sidecar_src, sidecar_dst)

        return dst

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, model_name: str, keep: int = 3) -> int:
        """Remove old backups, keeping only the *keep* most recent.

        Returns the number of backup sets removed.
        """
        backups = self.list_backups(model_name)
        to_remove = backups[keep:]
        removed = 0
        for entry in to_remove:
            p: Path = entry["path"]
            p.unlink(missing_ok=True)
            # Remove sidecars
            for suffix in _SIDECAR_SUFFIXES:
                sidecar = p.parent / (p.name + suffix)
                sidecar.unlink(missing_ok=True)
            removed += 1
        return removed
