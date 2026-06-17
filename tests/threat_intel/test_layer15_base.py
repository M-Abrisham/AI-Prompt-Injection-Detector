"""Tests for Layer 15 base module — data classes and ThreatIntelSource.

Covers the critical gaps identified in code review:
- SourceSnapshot.from_dict() / to_dict() round-trip
- from_dict() with corrupt/missing data
- ThreatIntelSource.sync() pipeline (first sync, repeat, dry_run)
- load_last_snapshot() with corrupt files
- save_snapshot() atomic write
- ApplyResult.success property
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from na0s.layer15.base import (
    ApplyResult,
    DiffItem,
    SchemaValidationError,
    SourceSnapshot,
    SyncReport,
    TaxonomyDiff,
    TechniqueEntry,
    ThreatIntelSource,
)


# ---------------------------------------------------------------------------
# SourceSnapshot serialization
# ---------------------------------------------------------------------------


class TestSourceSnapshotSerialization:

    def test_round_trip(self):
        original = SourceSnapshot(
            source_name="test",
            fetched_at=datetime(2026, 3, 24, 12, 0, tzinfo=timezone.utc),
            version="v1.0",
            etag="abc123",
            techniques=[
                TechniqueEntry(id="T1", name="Alpha", severity="high"),
            ],
            raw_metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = SourceSnapshot.from_dict(data)
        assert restored.source_name == original.source_name
        assert restored.version == original.version
        assert restored.etag == original.etag
        assert len(restored.techniques) == 1
        assert restored.techniques[0].id == "T1"
        assert restored.raw_metadata == {"key": "value"}

    def test_from_dict_missing_source_name_raises(self):
        with pytest.raises(SchemaValidationError, match="source_name"):
            SourceSnapshot.from_dict({"version": "v1", "fetched_at": "2026-01-01T00:00:00+00:00"})

    def test_from_dict_missing_version_raises(self):
        with pytest.raises(SchemaValidationError, match="version"):
            SourceSnapshot.from_dict({"source_name": "x", "fetched_at": "2026-01-01T00:00:00+00:00"})

    def test_from_dict_invalid_timestamp_raises(self):
        with pytest.raises(SchemaValidationError, match="timestamp"):
            SourceSnapshot.from_dict({
                "source_name": "x",
                "version": "v1",
                "fetched_at": "not-a-date",
            })

    def test_from_dict_extra_technique_keys_ignored(self):
        """Extra keys in technique dicts don't crash (filtered)."""
        data = {
            "source_name": "test",
            "version": "v1",
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "techniques": [{"id": "T1", "name": "X", "unknown_field": 42}],
        }
        snap = SourceSnapshot.from_dict(data)
        assert len(snap.techniques) == 1
        assert snap.techniques[0].id == "T1"

    def test_from_dict_empty_techniques(self):
        data = {
            "source_name": "test",
            "version": "v1",
            "fetched_at": "2026-01-01T00:00:00+00:00",
        }
        snap = SourceSnapshot.from_dict(data)
        assert snap.techniques == []


# ---------------------------------------------------------------------------
# ApplyResult
# ---------------------------------------------------------------------------


class TestApplyResult:

    def test_success_with_no_errors(self):
        r = ApplyResult(applied_count=5, skipped_count=0)
        assert r.success

    def test_not_success_with_errors(self):
        r = ApplyResult(applied_count=3, errors=["oops"])
        assert not r.success


# ---------------------------------------------------------------------------
# TaxonomyDiff properties
# ---------------------------------------------------------------------------


class TestTaxonomyDiff:

    def test_has_changes(self):
        diff = TaxonomyDiff(source_name="t", old_version="a", new_version="b")
        assert not diff.has_changes
        diff.items.append(DiffItem(change_type="added", technique_id="T1", technique_name="X"))
        assert diff.has_changes

    def test_property_filters(self):
        diff = TaxonomyDiff(
            source_name="t", old_version="a", new_version="b",
            items=[
                DiffItem(change_type="added", technique_id="T1", technique_name="A"),
                DiffItem(change_type="removed", technique_id="T2", technique_name="B"),
                DiffItem(change_type="modified", technique_id="T3", technique_name="C"),
                DiffItem(change_type="added", technique_id="T4", technique_name="D", needs_review=True),
            ],
        )
        assert len(diff.added) == 2
        assert len(diff.removed) == 1
        assert len(diff.modified) == 1
        assert len(diff.unmapped) == 1


# ---------------------------------------------------------------------------
# ThreatIntelSource.sync() pipeline
# ---------------------------------------------------------------------------


class FakeSource(ThreatIntelSource):
    """Minimal source for testing sync()."""

    name = "fake"

    def __init__(self, techniques=None, snapshots_dir=None):
        super().__init__(snapshots_dir=snapshots_dir)
        self._techniques = techniques or []

    def fetch_latest(self):
        return SourceSnapshot(
            source_name=self.name,
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="v1.0",
            techniques=self._techniques,
        )

    def diff(self, old, new):
        items = [
            DiffItem(change_type="added", technique_id=t.id, technique_name=t.name)
            for t in new.techniques
            if t.id not in {ot.id for ot in old.techniques}
        ]
        return TaxonomyDiff(
            source_name=self.name,
            old_version=old.version,
            new_version=new.version,
            items=items,
        )

    def apply(self, diff, dry_run=False):
        return ApplyResult(
            applied_count=0 if dry_run else len(diff.added),
            skipped_count=len(diff.items) if dry_run else 0,
            dry_run=dry_run,
        )


class TestSyncPipeline:

    def test_first_sync(self, tmp_path):
        """First sync with no previous snapshot — everything is 'added'."""
        source = FakeSource(
            techniques=[TechniqueEntry(id="T1", name="Alpha")],
            snapshots_dir=tmp_path,
        )
        report = source.sync()
        assert report.source_name == "fake"
        assert len(report.diff.added) == 1
        assert report.result.applied_count == 1
        # Snapshot should be saved
        assert (tmp_path / "fake_snapshot.json").exists()

    def test_repeat_sync_no_changes(self, tmp_path):
        """Second sync with same data — no changes."""
        source = FakeSource(
            techniques=[TechniqueEntry(id="T1", name="Alpha")],
            snapshots_dir=tmp_path,
        )
        source.sync()
        report = source.sync()
        assert not report.diff.has_changes
        assert report.result.applied_count == 0

    def test_dry_run_does_not_save_snapshot(self, tmp_path):
        """dry_run=True should not write snapshot file."""
        source = FakeSource(
            techniques=[TechniqueEntry(id="T1", name="Alpha")],
            snapshots_dir=tmp_path,
        )
        report = source.sync(dry_run=True)
        assert report.result.dry_run
        assert not (tmp_path / "fake_snapshot.json").exists()

    def test_corrupt_snapshot_treated_as_first_sync(self, tmp_path):
        """Corrupt snapshot file → treated as first sync."""
        (tmp_path / "fake_snapshot.json").write_text("{bad json")
        source = FakeSource(
            techniques=[TechniqueEntry(id="T1", name="Alpha")],
            snapshots_dir=tmp_path,
        )
        report = source.sync()
        assert len(report.diff.added) == 1  # Treated as first sync

    def test_atomic_write(self, tmp_path):
        """save_snapshot uses temp file + rename."""
        source = FakeSource(snapshots_dir=tmp_path)
        snap = SourceSnapshot(
            source_name="fake",
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            version="v1",
        )
        source.save_snapshot(snap)
        # No .tmp file should remain
        assert not (tmp_path / "fake_snapshot.tmp").exists()
        assert (tmp_path / "fake_snapshot.json").exists()
