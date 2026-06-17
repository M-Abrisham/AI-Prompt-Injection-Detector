"""Tests for Layer 15 Orchestrator.

Covers:
- Happy path: multiple sources synced, reports generated
- Partial failure: one source fails, others succeed
- Dry run mode
- PR body generation
- No changes scenario
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from na0s.layer15.base import (
    ApplyResult,
    DiffItem,
    SourceSnapshot,
    SourceUnavailableError,
    SyncReport,
    TaxonomyDiff,
    TechniqueEntry,
    ThreatIntelSource,
)
from na0s.layer15.orchestrator import Orchestrator


class FakeSource(ThreatIntelSource):
    """A fake source for testing the orchestrator."""

    def __init__(self, name, techniques=None, fail=False, snapshots_dir=None):
        self._name = name
        self._techniques = techniques or []
        self._fail = fail
        # Use a temp dir to avoid touching real filesystem
        super().__init__(snapshots_dir=snapshots_dir)

    @property
    def name(self):
        return self._name

    def fetch_latest(self):
        if self._fail:
            raise SourceUnavailableError(f"{self._name} is down")
        return SourceSnapshot(
            source_name=self._name,
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="v1.0",
            techniques=self._techniques,
        )

    def diff(self, old, new):
        items = [
            DiffItem(
                change_type="added",
                technique_id=t.id,
                technique_name=t.name,
                na0s_mapping=t.category,
                needs_review=not t.category,
            )
            for t in new.techniques
        ]
        return TaxonomyDiff(
            source_name=self._name,
            old_version=old.version,
            new_version=new.version,
            items=items,
        )

    def apply(self, diff, dry_run=False):
        return ApplyResult(
            applied_count=len(diff.added),
            skipped_count=0,
            dry_run=dry_run,
        )


@pytest.fixture
def tmp_dirs(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    output = tmp_path / "reports"
    output.mkdir()
    return snapshots, output


class TestOrchestratorRun:

    def test_happy_path_multiple_sources(self, tmp_dirs):
        snapshots, output = tmp_dirs
        sources = [
            FakeSource(
                "source_a",
                [TechniqueEntry(id="T1", name="Alpha", category="D1")],
                snapshots_dir=snapshots,
            ),
            FakeSource(
                "source_b",
                [TechniqueEntry(id="T2", name="Beta")],
                snapshots_dir=snapshots,
            ),
        ]
        orch = Orchestrator(sources=sources, output_dir=output)
        reports = orch.run()
        assert len(reports) == 2
        assert reports[0].source_name == "source_a"
        assert reports[1].source_name == "source_b"

    def test_partial_failure_skips_bad_source(self, tmp_dirs):
        snapshots, output = tmp_dirs
        sources = [
            FakeSource(
                "good_source",
                [TechniqueEntry(id="T1", name="Alpha")],
                snapshots_dir=snapshots,
            ),
            FakeSource("bad_source", fail=True, snapshots_dir=snapshots),
        ]
        orch = Orchestrator(sources=sources, output_dir=output)
        reports = orch.run()
        assert len(reports) == 1
        assert reports[0].source_name == "good_source"

    def test_dry_run_propagated(self, tmp_dirs):
        snapshots, output = tmp_dirs
        sources = [
            FakeSource(
                "source_a",
                [TechniqueEntry(id="T1", name="Alpha")],
                snapshots_dir=snapshots,
            ),
        ]
        orch = Orchestrator(sources=sources, dry_run=True, output_dir=output)
        reports = orch.run()
        assert reports[0].result.dry_run

    def test_no_sources_returns_empty(self, tmp_dirs):
        _, output = tmp_dirs
        orch = Orchestrator(sources=[], output_dir=output)
        reports = orch.run()
        assert reports == []

    def test_reports_saved_to_disk(self, tmp_dirs):
        snapshots, output = tmp_dirs
        sources = [
            FakeSource(
                "source_a",
                [TechniqueEntry(id="T1", name="Alpha")],
                snapshots_dir=snapshots,
            ),
        ]
        orch = Orchestrator(sources=sources, output_dir=output)
        orch.run()
        # Should have individual + combined report files
        files = list(output.iterdir())
        assert len(files) >= 2  # At least .md + .json for source_a, plus combined


class TestOrchestratorPRBody:

    def test_pr_body_includes_summary_table(self, tmp_dirs):
        snapshots, output = tmp_dirs
        sources = [
            FakeSource(
                "atlas",
                [TechniqueEntry(id="T1", name="Alpha", category="D1")],
                snapshots_dir=snapshots,
            ),
        ]
        orch = Orchestrator(sources=sources, output_dir=output)
        reports = orch.run()
        body = orch.generate_pr_body(reports)
        assert "Summary" in body
        assert "atlas" in body
        assert "Added" in body or "added" in body.lower()

    def test_pr_body_no_changes(self, tmp_dirs):
        snapshots, output = tmp_dirs
        sources = [
            FakeSource("atlas", [], snapshots_dir=snapshots),
        ]
        orch = Orchestrator(sources=sources, output_dir=output)
        reports = orch.run()
        body = orch.generate_pr_body(reports)
        assert "No changes" in body

    def test_pr_body_shows_unmapped_items(self, tmp_dirs):
        snapshots, output = tmp_dirs
        sources = [
            FakeSource(
                "atlas",
                [TechniqueEntry(id="T1", name="Unmapped Thing", category="")],
                snapshots_dir=snapshots,
            ),
        ]
        orch = Orchestrator(sources=sources, output_dir=output)
        reports = orch.run()
        body = orch.generate_pr_body(reports)
        assert "Manual Review" in body
        assert "Unmapped Thing" in body
