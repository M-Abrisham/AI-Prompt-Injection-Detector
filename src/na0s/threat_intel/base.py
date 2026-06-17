"""Base classes and data structures for Layer 15 threat intel sources.

All sync modules implement the ThreatIntelSource interface. Data flows
through: fetch_latest() → diff() → apply() → SyncReport.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from na0s.threat_intel.config import SNAPSHOTS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TechniqueEntry:
    """A single technique as understood by the diff engine."""

    id: str  # e.g., "AML.T0043" (ATLAS) or "D1.5" (Na0S)
    name: str
    description: str = ""
    severity: str = ""
    category: str = ""  # Na0S category ID if mapped, else ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceSnapshot:
    """A point-in-time snapshot of an upstream source's data.

    Stored locally as JSON so we can diff against the next fetch.
    """

    source_name: str
    fetched_at: datetime
    version: str  # Source-specific version (commit SHA, release tag, etc.)
    etag: str = ""  # HTTP ETag for conditional requests
    techniques: List[TechniqueEntry] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "source_name": self.source_name,
            "fetched_at": self.fetched_at.isoformat(),
            "version": self.version,
            "etag": self.etag,
            "techniques": [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "severity": t.severity,
                    "category": t.category,
                    "metadata": t.metadata,
                }
                for t in self.techniques
            ],
            "raw_metadata": self.raw_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceSnapshot:
        """Deserialize from a JSON-safe dict.

        Raises SchemaValidationError on missing keys or invalid values.
        """
        try:
            source_name = data["source_name"]
            version = data["version"]
            fetched_at = datetime.fromisoformat(data["fetched_at"])
        except KeyError as exc:
            raise SchemaValidationError(
                f"Snapshot missing required key: {exc}"
            ) from exc
        except (ValueError, TypeError) as exc:
            raise SchemaValidationError(
                f"Snapshot has invalid fetched_at timestamp: {exc}"
            ) from exc

        techniques = []
        for t in data.get("techniques", []):
            try:
                # Filter to known fields to avoid TypeError on extra keys
                known = {
                    "id", "name", "description", "severity",
                    "category", "metadata",
                }
                techniques.append(
                    TechniqueEntry(**{k: v for k, v in t.items() if k in known})
                )
            except TypeError as exc:
                raise SchemaValidationError(
                    f"Invalid technique entry: {exc}"
                ) from exc

        return cls(
            source_name=source_name,
            fetched_at=fetched_at,
            version=version,
            etag=data.get("etag", ""),
            techniques=techniques,
            raw_metadata=data.get("raw_metadata", {}),
        )


@dataclass
class DiffItem:
    """A single change detected between two snapshots."""

    change_type: str  # "added", "removed", "modified", "reclassified"
    technique_id: str
    technique_name: str
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    na0s_mapping: str = ""  # Na0S category.technique if mapped
    needs_review: bool = False  # True if unmapped or ambiguous


@dataclass
class TaxonomyDiff:
    """Structured diff between two taxonomy snapshots.

    Used by all sync modules. Consumed by the orchestrator to generate
    human-readable changelogs and machine-readable reports.
    """

    source_name: str
    old_version: str
    new_version: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    items: List[DiffItem] = field(default_factory=list)

    @property
    def added(self) -> List[DiffItem]:
        return [i for i in self.items if i.change_type == "added"]

    @property
    def removed(self) -> List[DiffItem]:
        return [i for i in self.items if i.change_type == "removed"]

    @property
    def modified(self) -> List[DiffItem]:
        return [i for i in self.items if i.change_type == "modified"]

    @property
    def unmapped(self) -> List[DiffItem]:
        return [i for i in self.items if i.needs_review]

    @property
    def has_changes(self) -> bool:
        return len(self.items) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "added": len(self.added),
                "removed": len(self.removed),
                "modified": len(self.modified),
                "unmapped": len(self.unmapped),
                "total_changes": len(self.items),
            },
            "items": [
                {
                    "change_type": i.change_type,
                    "technique_id": i.technique_id,
                    "technique_name": i.technique_name,
                    "old_value": i.old_value,
                    "new_value": i.new_value,
                    "na0s_mapping": i.na0s_mapping,
                    "needs_review": i.needs_review,
                }
                for i in self.items
            ],
        }


@dataclass
class ApplyResult:
    """Result of applying a diff to the local taxonomy."""

    applied_count: int = 0
    skipped_count: int = 0
    errors: List[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


@dataclass
class SyncReport:
    """Full report from a single source sync cycle."""

    source_name: str
    diff: TaxonomyDiff
    result: ApplyResult
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "diff": self.diff.to_dict(),
            "applied": self.result.applied_count,
            "skipped": self.result.skipped_count,
            "errors": self.result.errors,
            "dry_run": self.result.dry_run,
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SourceUnavailableError(Exception):
    """Raised when an upstream source cannot be reached."""


class RateLimitError(Exception):
    """Raised when we've hit an upstream source's rate limit."""


class SchemaValidationError(Exception):
    """Raised when upstream data doesn't match expected schema."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ThreatIntelSource(ABC):
    """Base class for all Layer 15 threat intelligence sources.

    Subclasses implement fetch_latest(), diff(), and apply().
    The sync() method orchestrates the full pipeline.
    """

    name: str = "unknown"

    def __init__(self, snapshots_dir: Optional[Path] = None):
        self.snapshots_dir = snapshots_dir or SNAPSHOTS_DIR
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_latest(self) -> SourceSnapshot:
        """Fetch the latest data from the upstream source.

        Returns a SourceSnapshot containing the raw upstream data
        and metadata (fetch timestamp, source version, ETag, etc.)

        Raises:
            SourceUnavailableError: if the upstream source is unreachable
            RateLimitError: if we've hit the source's rate limit
        """
        ...

    @abstractmethod
    def diff(self, old: SourceSnapshot, new: SourceSnapshot) -> TaxonomyDiff:
        """Compare two snapshots and produce a structured diff.

        The diff captures:
        - New techniques/probes/incidents
        - Removed/deprecated items
        - Reclassified items
        - Metadata changes
        """
        ...

    @abstractmethod
    def apply(self, diff: TaxonomyDiff, dry_run: bool = False) -> ApplyResult:
        """Apply the diff to Na0S's local taxonomy.

        If dry_run=True, report what would change without modifying anything.
        Must be idempotent: applying the same diff twice = no additional changes.
        """
        ...

    def load_last_snapshot(self) -> Optional[SourceSnapshot]:
        """Load the most recent snapshot for this source from disk.

        Returns None if the file doesn't exist or is corrupt.
        """
        snapshot_file = self.snapshots_dir / f"{self.name}_snapshot.json"
        if not snapshot_file.exists():
            logger.info(
                "No previous snapshot for %s — first sync", self.name
            )
            return None
        try:
            with open(snapshot_file, encoding="utf-8") as f:
                data = json.load(f)
            return SourceSnapshot.from_dict(data)
        except (json.JSONDecodeError, SchemaValidationError) as exc:
            logger.warning(
                "Corrupt snapshot for %s, treating as first sync: %s",
                self.name,
                exc,
            )
            return None

    def save_snapshot(self, snapshot: SourceSnapshot) -> None:
        """Persist a snapshot to disk for future diffing.

        Uses atomic write (temp file + rename) to avoid corruption
        if the process crashes mid-write.
        """
        snapshot_file = self.snapshots_dir / f"{self.name}_snapshot.json"
        tmp_file = snapshot_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        tmp_file.rename(snapshot_file)
        logger.info("Saved snapshot for %s (version=%s)", self.name, snapshot.version)

    def sync(self, dry_run: bool = False) -> SyncReport:
        """Full sync pipeline: fetch → diff → apply → report."""
        start = time.monotonic()

        previous = self.load_last_snapshot()
        current = self.fetch_latest()

        if previous is None:
            # First sync — everything is "new"
            previous = SourceSnapshot(
                source_name=self.name,
                fetched_at=datetime.min.replace(tzinfo=timezone.utc),
                version="",
            )

        taxonomy_diff = self.diff(previous, current)
        result = self.apply(taxonomy_diff, dry_run=dry_run)

        if not dry_run:
            # Save snapshot to update fetched_at (even if no changes)
            self.save_snapshot(current)

        elapsed = time.monotonic() - start
        report = SyncReport(
            source_name=self.name,
            diff=taxonomy_diff,
            result=result,
            duration_seconds=round(elapsed, 2),
        )

        logger.info(
            "Sync complete for %s: %d changes (%d added, %d removed, %d modified) in %.1fs",
            self.name,
            len(taxonomy_diff.items),
            len(taxonomy_diff.added),
            len(taxonomy_diff.removed),
            len(taxonomy_diff.modified),
            elapsed,
        )
        return report
