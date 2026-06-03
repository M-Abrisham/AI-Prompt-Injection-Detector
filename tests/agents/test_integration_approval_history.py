"""Integration tests for approval history with deploy_approver and quarantine_reviewer."""

import json
import pytest
import tempfile
from pathlib import Path

from na0s.agents.deploy_approver import DeployApprover
from na0s.agents.quarantine_reviewer import QuarantineReviewer
from na0s.agents.approval_history import ApprovalHistoryManager


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestDeployApproverIntegration:
    """Test that deploy_approver records to approval history."""

    def test_approver_initializes_with_history_manager(self, temp_data_dir):
        """Test that DeployApprover creates and uses ApprovalHistoryManager."""
        approver = DeployApprover(data_dir=temp_data_dir)

        # Should have history manager
        assert approver.history is not None
        assert isinstance(approver.history, ApprovalHistoryManager)

    def test_deploy_rejectionrecorded_to_history(self, temp_data_dir):
        """Test that deployment rejections are recorded to history."""
        approver = DeployApprover(data_dir=temp_data_dir)

        # Simulate user rejecting deployment
        success, message = approver.handle_approval("reject")

        assert success
        assert "rejected" in message.lower()

        # Check history was recorded
        history = approver.history.get_recent(days=30)
        assert len(history) > 0

        record = history[0]
        assert record["action_type"] == "deploy"
        assert record["status"] == "rejected"
        assert record["execution_result"] == "skipped"


class TestQuarantineReviewerIntegration:
    """Test that quarantine_reviewer records to approval history."""

    def test_reviewer_initializes_with_history_manager(self, temp_data_dir):
        """Test that QuarantineReviewer creates and uses ApprovalHistoryManager."""
        reviewer = QuarantineReviewer(data_dir=temp_data_dir)

        # Should have history manager
        assert reviewer.history is not None
        assert isinstance(reviewer.history, ApprovalHistoryManager)

    def test_invalid_action_format_not_recorded(self, temp_data_dir):
        """Test that invalid actions are not recorded."""
        reviewer = QuarantineReviewer(data_dir=temp_data_dir)

        # Try to submit invalid action
        success, message = reviewer.handle_user_response("invalid command")

        assert not success
        assert ("Invalid format" in message or "Unknown action" in message)

        # Should not be recorded to history
        history = reviewer.history.get_recent(days=30)
        assert len(history) == 0


class TestApprovalHistoryDataStructure:
    """Test that approval history JSONL has correct structure."""

    def test_approval_history_is_jsonl(self, temp_data_dir):
        """Test that approval_history.jsonl file is valid JSONL."""
        history = ApprovalHistoryManager(data_dir=temp_data_dir)

        # Record some actions
        history.record_action(action_type="deploy", status="approved")
        history.record_action(action_type="promote", status="approved")

        # Check file format
        history_file = history.approval_history_path
        assert history_file.exists()

        with open(history_file) as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Each line should be valid JSON
        for line in lines:
            record = json.loads(line)
            assert "timestamp" in record
            assert "action_type" in record
            assert "status" in record
            assert record["timestamp"].endswith("Z")

    def test_each_record_includes_timestamp_and_action_type(self, temp_data_dir):
        """Test that each record has required fields."""
        history = ApprovalHistoryManager(data_dir=temp_data_dir)

        history.record_action(
            action_type="deploy",
            status="approved",
            approved_by="user",
            reason="Test deployment",
            execution_result="success",
            execution_time_seconds=5.0,
        )

        with open(history.approval_history_path) as f:
            record = json.loads(f.readline())

        # Required fields
        assert record["timestamp"]
        assert record["action_type"] == "deploy"
        assert record["status"] == "approved"
        assert record["approved_by"] == "user"
        assert record["reason"] == "Test deployment"
        assert record["execution_result"] == "success"
        assert record["execution_time_seconds"] == 5.0
