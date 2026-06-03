"""Tests for approval history manager.

Tests JSONL writes, thread-safety, querying, statistics, and rotation.
"""

import json
import pytest
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta

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


class TestRecordAction:
    """Test recording approval actions."""

    def test_record_deploy_approval(self, history_manager):
        """Test recording a deployment approval."""
        success = history_manager.record_action(
            action_type="deploy",
            status="approved",
            approved_by="user",
            reason="User approved deployment",
            execution_result="success",
            execution_time_seconds=5.2,
        )

        assert success

        # Verify JSONL file exists and contains record
        history_file = history_manager.approval_history_path
        assert history_file.exists()

        with open(history_file) as f:
            lines = f.readlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["action_type"] == "deploy"
        assert record["status"] == "approved"
        assert record["approved_by"] == "user"
        assert record["execution_result"] == "success"

    def test_record_with_error(self, history_manager):
        """Test recording a failed action with error."""
        success = history_manager.record_action(
            action_type="deploy",
            status="approved",
            approved_by="user",
            execution_result="failed",
            execution_time_seconds=10.0,
            error="Timeout executing deploy script",
        )

        assert success

        with open(history_manager.approval_history_path) as f:
            record = json.loads(f.readline())

        assert record["error"] == "Timeout executing deploy script"
        assert record["execution_time_seconds"] == 10.0

    def test_record_with_metadata(self, history_manager):
        """Test recording action with additional metadata."""
        success = history_manager.record_action(
            action_type="promote",
            status="approved",
            approved_by="user",
            execution_result="success",
            metadata={"dataset": "dataset_a", "tier": "beta"},
        )

        assert success

        with open(history_manager.approval_history_path) as f:
            record = json.loads(f.readline())

        assert record["dataset"] == "dataset_a"
        assert record["tier"] == "beta"

    def test_record_multiple_actions(self, history_manager):
        """Test recording multiple actions to same file."""
        for i in range(5):
            success = history_manager.record_action(
                action_type="deploy",
                status="approved",
                approved_by="user",
                reason=f"Action {i}",
            )
            assert success

        with open(history_manager.approval_history_path) as f:
            lines = f.readlines()

        assert len(lines) == 5

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["reason"] == f"Action {i}"


class TestGetRecent:
    """Test querying recent approval records."""

    def test_get_recent_empty(self, history_manager):
        """Test get_recent on empty history."""
        records = history_manager.get_recent(days=30)
        assert records == []

    def test_get_recent_with_records(self, history_manager):
        """Test retrieving recent records."""
        # Add records
        for i in range(3):
            history_manager.record_action(
                action_type="deploy",
                status="approved",
                reason=f"Deploy {i}",
            )

        records = history_manager.get_recent(days=30)
        assert len(records) == 3

        # Should be returned in reverse chronological order (most recent first)
        assert records[0]["reason"] == "Deploy 2"
        assert records[2]["reason"] == "Deploy 0"

    def test_get_recent_by_action_type(self, history_manager):
        """Test filtering by action type."""
        # Add mixed action types
        history_manager.record_action(action_type="deploy", status="approved")
        history_manager.record_action(action_type="promote", status="approved")
        history_manager.record_action(action_type="deploy", status="approved")

        deploy_records = history_manager.get_recent(days=30, action_type="deploy")
        assert len(deploy_records) == 2
        assert all(r["action_type"] == "deploy" for r in deploy_records)

        promote_records = history_manager.get_recent(days=30, action_type="promote")
        assert len(promote_records) == 1
        assert promote_records[0]["action_type"] == "promote"

    def test_get_recent_respects_date_filter(self, history_manager):
        """Test that records outside date range are excluded."""
        # Write record with old timestamp
        history_file = history_manager.approval_history_path

        # Manually write an old record
        old_timestamp = (datetime.utcnow() - timedelta(days=40)).isoformat() + "Z"
        old_record = {
            "timestamp": old_timestamp,
            "action_type": "deploy",
            "status": "approved",
        }

        with open(history_file, "a") as f:
            f.write(json.dumps(old_record) + "\n")

        # Add recent record
        history_manager.record_action(action_type="deploy", status="approved")

        # Query for 30 days should only return recent record
        records = history_manager.get_recent(days=30)
        assert len(records) == 1


class TestGetStats:
    """Test statistics calculation."""

    def test_get_stats_empty(self, history_manager):
        """Test stats on empty history."""
        stats = history_manager.get_stats(days=30)

        assert stats["total_approvals"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_execution_time_seconds"] == 0.0

    def test_get_stats_basic(self, history_manager):
        """Test basic stats calculation."""
        # Add some records
        history_manager.record_action(
            action_type="deploy",
            status="approved",
            execution_result="success",
            execution_time_seconds=5.0,
        )
        history_manager.record_action(
            action_type="deploy",
            status="approved",
            execution_result="success",
            execution_time_seconds=3.0,
        )

        stats = history_manager.get_stats(days=30)

        assert stats["total_approvals"] == 2
        assert stats["successful_approvals"] == 2
        assert stats["success_rate"] == 100.0
        assert stats["avg_execution_time_seconds"] == 4.0

    def test_get_stats_with_failures(self, history_manager):
        """Test stats with mixed success/failure."""
        history_manager.record_action(
            action_type="deploy",
            status="approved",
            execution_result="success",
        )
        history_manager.record_action(
            action_type="deploy",
            status="rejected",
            execution_result="failed",
        )

        stats = history_manager.get_stats(days=30)

        assert stats["total_approvals"] == 2
        assert stats["successful_approvals"] == 1
        assert stats["success_rate"] == 50.0

    def test_get_stats_by_action_type(self, history_manager):
        """Test stats broken down by action type."""
        # Add deploy records
        history_manager.record_action(action_type="deploy", status="approved")
        history_manager.record_action(action_type="deploy", status="approved")

        # Add promote records
        history_manager.record_action(action_type="promote", status="approved")

        stats = history_manager.get_stats(days=30)

        assert stats["by_action_type"]["deploy"]["count"] == 2
        assert stats["by_action_type"]["promote"]["count"] == 1

    def test_get_stats_by_status(self, history_manager):
        """Test stats broken down by status."""
        history_manager.record_action(action_type="deploy", status="approved")
        history_manager.record_action(action_type="deploy", status="approved")
        history_manager.record_action(action_type="deploy", status="rejected")

        stats = history_manager.get_stats(days=30)

        assert stats["by_status"]["approved"] == 2
        assert stats["by_status"]["rejected"] == 1


class TestDeploymentAndQuarantineHistory:
    """Test specialized history retrieval."""

    def test_get_deployment_history(self, history_manager):
        """Test retrieving only deployment records."""
        history_manager.record_action(action_type="deploy", status="approved")
        history_manager.record_action(action_type="deploy", status="rejected")
        history_manager.record_action(action_type="promote", status="approved")

        deployments = history_manager.get_deployment_history()
        assert len(deployments) == 2
        assert all(d["action_type"] == "deploy" for d in deployments)

    def test_get_quarantine_history(self, history_manager):
        """Test retrieving quarantine-related records."""
        history_manager.record_action(action_type="promote", status="approved")
        history_manager.record_action(action_type="reject", status="approved")
        history_manager.record_action(action_type="quarantine_review", status="pending")
        history_manager.record_action(action_type="deploy", status="approved")

        quarantine = history_manager.get_quarantine_history()
        assert len(quarantine) == 3
        assert all(
            r["action_type"] in ("promote", "reject", "quarantine_review")
            for r in quarantine
        )


class TestThreadSafety:
    """Test thread-safe file operations."""

    def test_concurrent_writes(self, history_manager):
        """Test that concurrent writes don't corrupt JSONL."""
        num_threads = 5
        records_per_thread = 10

        def worker(thread_id):
            for i in range(records_per_thread):
                history_manager.record_action(
                    action_type="deploy",
                    status="approved",
                    reason=f"Thread {thread_id}, Record {i}",
                )
                time.sleep(0.001)  # Small delay to encourage interleaving

        # Launch threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all records written
        with open(history_manager.approval_history_path) as f:
            lines = f.readlines()

        assert len(lines) == num_threads * records_per_thread

        # Verify all records are valid JSON
        records = []
        for line in lines:
            record = json.loads(line)
            records.append(record)

        assert len(records) == num_threads * records_per_thread


class TestRotation:
    """Test history rotation/archival."""

    def test_rotation_archives_old_records(self, history_manager):
        """Test that records older than RETENTION_DAYS are archived."""
        from datetime import datetime as dt
        # Create a record with old timestamp
        old_timestamp = (
            dt.utcnow() - timedelta(days=400)
        ).isoformat() + "Z"

        old_record = {
            "timestamp": old_timestamp,
            "action_type": "deploy",
            "status": "approved",
        }

        # Write old record directly
        with open(history_manager.approval_history_path, "w") as f:
            f.write(json.dumps(old_record) + "\n")

        # Add recent record via manager (triggers rotation)
        history_manager.record_action(action_type="deploy", status="approved")

        # Check that archive was created
        archive_files = list(history_manager.archive_dir.glob("*.jsonl"))
        assert len(archive_files) > 0

        # Check that active log only has recent record
        with open(history_manager.approval_history_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["action_type"] == "deploy"
        # Recent record should NOT have old timestamp
        # Since it was just created, it should be very recent (within last 60 seconds)
        from datetime import datetime
        rec_time = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        now = datetime.now(rec_time.tzinfo)
        time_diff = (now - rec_time).total_seconds()
        assert time_diff < 60, f"Record timestamp {record['timestamp']} should be recent"

    def test_rotation_skips_if_nothing_to_archive(self, history_manager):
        """Test that rotation is skipped if no old records."""
        # Add only recent record
        history_manager.record_action(action_type="deploy", status="approved")

        # Force rotation
        history_manager._rotate_history()

        # Archive directory should be empty
        archive_files = list(history_manager.archive_dir.glob("*.jsonl"))
        assert len(archive_files) == 0


class TestClearHistory:
    """Test clearing history."""

    def test_clear_history(self, history_manager):
        """Test clearing all history."""
        history_manager.record_action(action_type="deploy", status="approved")
        history_manager.record_action(action_type="deploy", status="approved")

        # Verify records exist
        records = history_manager.get_recent(days=30)
        assert len(records) == 2

        # Clear
        success = history_manager.clear_history()
        assert success

        # Verify cleared
        assert not history_manager.approval_history_path.exists()
        records = history_manager.get_recent(days=30)
        assert len(records) == 0
