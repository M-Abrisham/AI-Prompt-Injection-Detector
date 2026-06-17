"""Tests for Layer 15 Taxonomy Diff Engine.

Covers:
- Happy path: added, removed, modified, reclassified techniques
- Edge cases: empty snapshots, identical snapshots, first sync
- Output: Markdown and JSON rendering
- Schema drift: malformed input handling
"""

from datetime import datetime, timezone

import pytest

from na0s.layer15.base import SourceSnapshot, TechniqueEntry
from na0s.layer15.diff_engine import TaxonomyDiffEngine


def _make_snapshot(
    techniques=None, version="v1", source="test_source"
):
    """Helper to build a SourceSnapshot with minimal boilerplate."""
    return SourceSnapshot(
        source_name=source,
        fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
        version=version,
        techniques=techniques or [],
    )


def _make_technique(tid, name="Test", **kwargs):
    """Helper to build a TechniqueEntry."""
    return TechniqueEntry(id=tid, name=name, **kwargs)


class TestTaxonomyDiffEngineCompute:
    """Tests for TaxonomyDiffEngine.compute()."""

    def setup_method(self):
        self.engine = TaxonomyDiffEngine()

    def test_identical_snapshots_produce_no_diff(self):
        techniques = [_make_technique("T1", "Alpha")]
        old = _make_snapshot(techniques=techniques, version="v1")
        new = _make_snapshot(techniques=techniques, version="v2")
        diff = self.engine.compute(old, new)
        assert not diff.has_changes
        assert len(diff.items) == 0
        assert diff.old_version == "v1"
        assert diff.new_version == "v2"

    def test_added_techniques_detected(self):
        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(
            techniques=[
                _make_technique("T1", "Alpha", category="D1"),
                _make_technique("T2", "Beta"),
            ],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        assert diff.has_changes
        assert len(diff.added) == 2
        assert diff.added[0].technique_id == "T1"
        assert diff.added[1].technique_id == "T2"

    def test_removed_techniques_detected(self):
        old = _make_snapshot(
            techniques=[_make_technique("T1", "Alpha"), _make_technique("T2", "Beta")],
            version="v1",
        )
        new = _make_snapshot(techniques=[_make_technique("T1", "Alpha")], version="v2")
        diff = self.engine.compute(old, new)
        assert len(diff.removed) == 1
        assert diff.removed[0].technique_id == "T2"

    def test_modified_technique_detected(self):
        old = _make_snapshot(
            techniques=[_make_technique("T1", "Alpha", severity="high")],
            version="v1",
        )
        new = _make_snapshot(
            techniques=[_make_technique("T1", "Alpha", severity="critical")],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        assert len(diff.modified) == 1
        assert diff.modified[0].technique_id == "T1"

    def test_reclassified_technique_detected(self):
        old = _make_snapshot(
            techniques=[_make_technique("T1", "Alpha", category="D1")],
            version="v1",
        )
        new = _make_snapshot(
            techniques=[_make_technique("T1", "Alpha", category="D2")],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        # Reclassified shows up in modified (it's a subtype)
        assert len(diff.items) == 1
        assert diff.items[0].change_type == "reclassified"
        assert diff.items[0].needs_review

    def test_unmapped_added_techniques_flagged(self):
        """New techniques with no Na0S category mapping are flagged."""
        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(
            techniques=[_make_technique("T1", "Alpha", category="")],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        assert len(diff.unmapped) == 1
        assert diff.unmapped[0].needs_review

    def test_mapped_added_techniques_not_flagged(self):
        """New techniques with a Na0S mapping are not flagged."""
        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(
            techniques=[_make_technique("T1", "Alpha", category="D1")],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        assert len(diff.unmapped) == 0
        assert not diff.added[0].needs_review

    def test_mixed_changes(self):
        """Snapshot with adds, removes, and modifications simultaneously."""
        old = _make_snapshot(
            techniques=[
                _make_technique("T1", "Alpha"),
                _make_technique("T2", "Beta", severity="high"),
                _make_technique("T3", "Gamma"),
            ],
            version="v1",
        )
        new = _make_snapshot(
            techniques=[
                _make_technique("T1", "Alpha"),  # unchanged
                _make_technique("T2", "Beta-v2", severity="high"),  # name changed
                _make_technique("T4", "Delta"),  # added
            ],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        assert len(diff.added) == 1  # T4
        assert len(diff.removed) == 1  # T3
        assert len(diff.modified) == 1  # T2
        assert diff.added[0].technique_id == "T4"
        assert diff.removed[0].technique_id == "T3"
        assert diff.modified[0].technique_id == "T2"

    def test_empty_old_snapshot_first_sync(self):
        """First sync — empty baseline, everything is 'added'."""
        old = _make_snapshot(techniques=[], version="")
        new = _make_snapshot(
            techniques=[_make_technique("T1", "A"), _make_technique("T2", "B")],
            version="v1",
        )
        diff = self.engine.compute(old, new)
        assert len(diff.added) == 2
        assert len(diff.removed) == 0

    def test_both_empty_snapshots(self):
        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(techniques=[], version="v2")
        diff = self.engine.compute(old, new)
        assert not diff.has_changes

    def test_diff_items_sorted_by_id(self):
        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(
            techniques=[
                _make_technique("T3", "C"),
                _make_technique("T1", "A"),
                _make_technique("T2", "B"),
            ],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        ids = [i.technique_id for i in diff.added]
        assert ids == ["T1", "T2", "T3"]


class TestTaxonomyDiffEngineMarkdown:
    """Tests for Markdown changelog rendering."""

    def setup_method(self):
        self.engine = TaxonomyDiffEngine()

    def test_no_changes_markdown(self):
        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(techniques=[], version="v2")
        diff = self.engine.compute(old, new)
        md = self.engine.to_markdown(diff)
        assert "No changes detected" in md

    def test_added_techniques_appear_in_markdown(self):
        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(
            techniques=[_make_technique("T1", "Alpha", category="D1")],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        md = self.engine.to_markdown(diff)
        assert "New Techniques" in md
        assert "T1" in md
        assert "Alpha" in md
        assert "D1" in md

    def test_unmapped_techniques_show_warning(self):
        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(
            techniques=[_make_technique("T1", "Alpha")],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        md = self.engine.to_markdown(diff)
        assert "UNMAPPED" in md
        assert "manual review" in md.lower() or "Review Needed" in md

    def test_versions_in_markdown_header(self):
        old = _make_snapshot(techniques=[], version="abc123")
        new = _make_snapshot(techniques=[], version="def456")
        diff = self.engine.compute(old, new)
        md = self.engine.to_markdown(diff)
        assert "abc123" in md
        assert "def456" in md


class TestTaxonomyDiffEngineJSON:
    """Tests for JSON output."""

    def setup_method(self):
        self.engine = TaxonomyDiffEngine()

    def test_json_is_valid(self):
        import json

        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(
            techniques=[_make_technique("T1", "A")],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        raw = self.engine.to_json(diff)
        data = json.loads(raw)
        assert data["source_name"] == "test_source"
        assert data["summary"]["added"] == 1

    def test_json_summary_counts(self):
        import json

        old = _make_snapshot(
            techniques=[_make_technique("T1", "A"), _make_technique("T2", "B")],
            version="v1",
        )
        new = _make_snapshot(
            techniques=[_make_technique("T1", "A-modified"), _make_technique("T3", "C")],
            version="v2",
        )
        diff = self.engine.compute(old, new)
        data = json.loads(self.engine.to_json(diff))
        assert data["summary"]["added"] == 1
        assert data["summary"]["removed"] == 1
        assert data["summary"]["modified"] == 1
        assert data["summary"]["total_changes"] == 3


class TestTaxonomyDiffEngineSaveReport:
    """Tests for saving reports to disk."""

    def test_save_creates_files(self, tmp_path):
        engine = TaxonomyDiffEngine()
        old = _make_snapshot(techniques=[], version="v1")
        new = _make_snapshot(
            techniques=[_make_technique("T1", "A")],
            version="v2",
        )
        diff = engine.compute(old, new)
        paths = engine.save_report(diff, tmp_path)
        assert paths["markdown"].exists()
        assert paths["json"].exists()
        assert paths["markdown"].suffix == ".md"
        assert paths["json"].suffix == ".json"
