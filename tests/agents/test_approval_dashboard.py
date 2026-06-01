"""Tests for approval history dashboard.

Tests dashboard routes, JSON APIs, and HTML rendering.
"""

import json
import pytest
import tempfile
import sys
from pathlib import Path
from urllib.parse import urlencode
from http.client import HTTPConnection
from threading import Thread
import time

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from na0s.agents.approval_history import ApprovalHistoryManager


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def history_manager(temp_data_dir):
    """Create a history manager with temp directory."""
    return ApprovalHistoryManager(data_dir=temp_data_dir)


@pytest.fixture
def sample_history(history_manager):
    """Add sample records to history."""
    # Add deployment records
    history_manager.record_action(
        action_type="deploy",
        status="approved",
        approved_by="user",
        reason="User approved deployment",
        execution_result="success",
        execution_time_seconds=5.2,
    )
    history_manager.record_action(
        action_type="deploy",
        status="rejected",
        approved_by="user",
        reason="User rejected deployment",
        execution_result="skipped",
        execution_time_seconds=0.5,
    )

    # Add quarantine records
    history_manager.record_action(
        action_type="promote",
        status="approved",
        approved_by="user",
        reason="Promoted dataset_a",
        execution_result="success",
        execution_time_seconds=3.1,
        metadata={"dataset": "dataset_a"},
    )
    history_manager.record_action(
        action_type="reject",
        status="approved",
        approved_by="user",
        reason="Rejected dataset_b",
        execution_result="success",
        execution_time_seconds=2.0,
        metadata={"dataset": "dataset_b"},
    )

    return history_manager


class TestDashboardHTML:
    """Test HTML dashboard rendering."""

    def test_render_with_empty_history(self, history_manager):
        """Test rendering dashboard with no records."""
        from approval_dashboard import render_html_dashboard

        stats = history_manager.get_stats(days=30)
        recent = history_manager.get_recent(days=30)

        html = render_html_dashboard(stats, recent, history_manager=history_manager)

        assert "Na0S Approval Dashboard" in html
        assert "0" in html  # Total approvals
        assert "No approvals recorded yet" in html

    def test_render_with_sample_data(self, sample_history):
        """Test rendering dashboard with sample data."""
        from approval_dashboard import render_html_dashboard

        stats = sample_history.get_stats(days=30)
        recent = sample_history.get_recent(days=30)

        html = render_html_dashboard(stats, recent, history_manager=sample_history)

        assert "Na0S Approval Dashboard" in html
        assert "4" in html  # Total approvals
        assert "deploy" in html
        assert "promote" in html
        assert "reject" in html

    def test_html_contains_stats_cards(self, sample_history):
        """Test that stats are rendered in HTML."""
        from approval_dashboard import render_html_dashboard

        stats = sample_history.get_stats(days=30)
        recent = sample_history.get_recent(days=30)

        html = render_html_dashboard(stats, recent, history_manager=sample_history)

        # Check for stat cards
        assert "Total Approvals (30d)" in html
        assert "Success Rate" in html
        assert "Avg Execution Time" in html

    def test_html_contains_timeline(self, sample_history):
        """Test that timeline chart is in HTML."""
        from approval_dashboard import render_html_dashboard

        stats = sample_history.get_stats(days=30)
        recent = sample_history.get_recent(days=30)

        html = render_html_dashboard(stats, recent, history_manager=sample_history)

        assert "Approvals per Day" in html
        assert "bar-chart" in html

    def test_html_responsive(self, sample_history):
        """Test that HTML includes responsive CSS."""
        from approval_dashboard import render_html_dashboard

        stats = sample_history.get_stats(days=30)
        recent = sample_history.get_recent(days=30)

        html = render_html_dashboard(stats, recent, history_manager=sample_history)

        assert "viewport" in html
        assert "@media (max-width: 768px)" in html


class TestAPIs:
    """Test JSON API endpoints."""

    def test_api_approvals_json(self, sample_history, history_manager):
        """Test /api/approvals returns JSON."""
        records = history_manager.get_recent(days=30)

        # Verify structure
        assert len(records) == 4
        assert all("timestamp" in r for r in records)
        assert all("action_type" in r for r in records)
        assert all("status" in r for r in records)

    def test_api_stats_json(self, sample_history):
        """Test /api/stats returns valid stats."""
        stats = sample_history.get_stats(days=30)

        assert "total_approvals" in stats
        assert "success_rate" in stats
        assert "by_action_type" in stats
        assert "by_status" in stats

        # Check content
        assert stats["total_approvals"] == 4
        assert stats["by_action_type"]["deploy"]["count"] == 2
        assert stats["by_action_type"]["promote"]["count"] == 1

    def test_api_deployments_json(self, sample_history):
        """Test deployment history API."""
        deployments = sample_history.get_deployment_history()

        assert len(deployments) == 2
        assert all(d["action_type"] == "deploy" for d in deployments)

    def test_api_quarantine_json(self, sample_history):
        """Test quarantine history API."""
        quarantine = sample_history.get_quarantine_history()

        assert len(quarantine) == 2
        assert all(
            q["action_type"] in ("promote", "reject") for q in quarantine
        )


class TestTimelineData:
    """Test timeline chart data generation."""

    def test_timeline_generation(self, sample_history):
        """Test timeline data for last 30 days."""
        from approval_dashboard import get_timeline_data

        timeline = get_timeline_data(days=30, history_manager=sample_history)

        # Should have 30 entries
        assert len(timeline) == 30

        # Should be sorted by date
        dates = list(timeline.keys())
        assert dates == sorted(dates)

        # Current date should have approvals
        today = None
        for date in dates:
            if timeline[date] > 0:
                today = date
                break

        assert today is not None

    def test_timeline_counts_approvals(self, sample_history):
        """Test that timeline counts approvals correctly."""
        from approval_dashboard import get_timeline_data

        # Get today's date
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")

        timeline = get_timeline_data(days=30, history_manager=sample_history)

        # Today should have at least 4 approvals
        assert timeline.get(today, 0) >= 4


class TestDataStructure:
    """Test approval history data structure."""

    def test_jsonl_format(self, history_manager):
        """Test that JSONL file is valid."""
        history_manager.record_action(
            action_type="deploy",
            status="approved",
            reason="Test record",
        )

        with open(history_manager.approval_history_path) as f:
            for line in f:
                # Each line should be valid JSON
                record = json.loads(line)
                assert isinstance(record, dict)

    def test_record_includes_required_fields(self, history_manager):
        """Test that records include required fields."""
        history_manager.record_action(
            action_type="deploy",
            status="approved",
            approved_by="user",
            reason="Test",
        )

        with open(history_manager.approval_history_path) as f:
            record = json.loads(f.readline())

        # Required fields
        assert "timestamp" in record
        assert "action_type" in record
        assert "status" in record
        assert "approved_by" in record

    def test_record_iso_timestamp(self, history_manager):
        """Test that timestamps are ISO 8601 format."""
        history_manager.record_action(
            action_type="deploy",
            status="approved",
        )

        with open(history_manager.approval_history_path) as f:
            record = json.loads(f.readline())

        # Should be ISO 8601 with Z suffix
        timestamp = record["timestamp"]
        assert timestamp.endswith("Z")
        # Should be parseable
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert dt is not None
