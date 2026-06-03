"""Tests for QuarantineReviewer agent."""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pytest

from na0s.agents.quarantine_reviewer import QuarantineReviewer


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with quarantine structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "quarantine").mkdir()
        yield str(data_dir)


def test_quarantine_reviewer_no_pending(temp_data_dir):
    """Test when there are no pending entries."""
    reviewer = QuarantineReviewer(data_dir=temp_data_dir)
    pending = reviewer.get_pending_entries()

    assert pending == []


def test_quarantine_reviewer_get_pending_entries(temp_data_dir):
    """Test retrieving pending quarantine entries."""
    data_dir = Path(temp_data_dir)

    # Create quarantine entries
    source_dir = data_dir / "quarantine" / "test_source"
    source_dir.mkdir()

    metadata = {
        "source_id": "test_1",
        "tier": 3,
        "ingested_at": datetime.now().isoformat(),
        "validation_status": "pending",
        "trust_score": 0.5,
        "row_count": 100,
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)
    pending = reviewer.get_pending_entries()

    assert len(pending) == 1
    assert pending[0]["source_name"] == "test_source"
    assert pending[0]["validation_status"] == "pending"


def test_quarantine_reviewer_identify_stale(temp_data_dir):
    """Test identifying stale entries (>20 days old)."""
    data_dir = Path(temp_data_dir)

    # Create a stale entry (25 days old)
    source_dir = data_dir / "quarantine" / "old_source"
    source_dir.mkdir()

    old_date = datetime.now() - timedelta(days=25)
    metadata = {
        "source_id": "old_1",
        "tier": 3,
        "ingested_at": old_date.isoformat(),
        "validation_status": "pending",
        "trust_score": 0.5,
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)
    pending = reviewer.get_pending_entries()
    stale = reviewer.identify_stale_entries(pending)

    assert len(stale) == 1
    assert stale[0]["age_days"] == 25


def test_quarantine_reviewer_sample_entries(temp_data_dir):
    """Test sampling rows from quarantine entries."""
    data_dir = Path(temp_data_dir)

    # Create entry with data
    source_dir = data_dir / "quarantine" / "sample_source"
    source_dir.mkdir()

    metadata = {
        "source_id": "sample_1",
        "tier": 3,
        "ingested_at": datetime.now().isoformat(),
        "validation_status": "pending",
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Write sample data file
    with open(source_dir / "data.jsonl", "w") as f:
        for i in range(20):
            f.write(json.dumps({"text": f"sample {i}", "category": "D1"}) + "\n")

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)
    pending = reviewer.get_pending_entries()
    samples = reviewer.sample_entries(pending, per_entry=5)

    assert "sample_source" in samples
    assert len(samples["sample_source"]["sampled"]) == 5


def test_quarantine_reviewer_compile_summary(temp_data_dir):
    """Test compiling review summary."""
    data_dir = Path(temp_data_dir)

    # Create pending entry
    source_dir = data_dir / "quarantine" / "review_source"
    source_dir.mkdir()

    old_date = datetime.now() - timedelta(days=25)
    metadata = {
        "source_id": "review_1",
        "tier": 3,
        "ingested_at": old_date.isoformat(),
        "validation_status": "pending",
        "trust_score": 0.4,
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    with open(source_dir / "data.jsonl", "w") as f:
        f.write(json.dumps({"text": "test"}) + "\n")

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)
    summary = reviewer.compile_review_summary()

    assert summary is not None
    assert summary["total_pending"] == 1
    assert summary["stale_count"] == 1
    assert "message" in summary


def test_quarantine_reviewer_format_message(temp_data_dir):
    """Test formatting quarantine status for iMessage."""
    reviewer = QuarantineReviewer(data_dir=temp_data_dir)
    message = reviewer.format_message()

    assert isinstance(message, str)
    assert len(message) > 0
    assert "empty" in message.lower() or "quarantine" in message.lower()


def test_quarantine_reviewer_execute_action_entry_not_found(temp_data_dir):
    """Test executing action on non-existent entry."""
    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    success, result = reviewer.execute_action("promote", "nonexistent")

    assert success is False
    assert "not found" in result["error_message"].lower()


def test_quarantine_reviewer_execute_action_promote_success(temp_data_dir):
    """Test successful promote action execution."""
    data_dir = Path(temp_data_dir)

    # Create quarantine entry
    source_dir = data_dir / "quarantine" / "test_source"
    source_dir.mkdir(parents=True)

    metadata = {
        "source_id": "test_1",
        "tier": 3,
        "ingested_at": datetime.now().isoformat(),
        "validation_status": "passed",
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Promotion successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        success, result = reviewer.execute_action("promote", "test_source")

        assert success is True
        assert result["returncode"] == 0
        assert result["action"] == "promote"


def test_quarantine_reviewer_execute_action_reject_success(temp_data_dir):
    """Test successful reject action execution."""
    data_dir = Path(temp_data_dir)

    # Create quarantine entry
    source_dir = data_dir / "quarantine" / "test_reject"
    source_dir.mkdir(parents=True)

    metadata = {
        "source_id": "reject_1",
        "tier": 4,
        "ingested_at": datetime.now().isoformat(),
        "validation_status": "pending",
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Rejection successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        success, result = reviewer.execute_action("reject", "test_reject")

        assert success is True
        assert result["returncode"] == 0
        assert result["action"] == "reject"


def test_quarantine_reviewer_execute_action_timeout(temp_data_dir):
    """Test action execution timeout."""
    data_dir = Path(temp_data_dir)

    # Create quarantine entry
    source_dir = data_dir / "quarantine" / "test_timeout"
    source_dir.mkdir(parents=True)

    metadata = {
        "source_id": "timeout_1",
        "tier": 3,
        "ingested_at": datetime.now().isoformat(),
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = TimeoutError("Operation timed out")

        success, result = reviewer.execute_action("promote", "test_timeout")

        assert success is False
        assert "timed out" in result["error_message"].lower()


def test_quarantine_reviewer_execute_action_failure(temp_data_dir):
    """Test action execution failure."""
    data_dir = Path(temp_data_dir)

    # Create quarantine entry
    source_dir = data_dir / "quarantine" / "test_fail"
    source_dir.mkdir(parents=True)

    metadata = {
        "source_id": "fail_1",
        "tier": 3,
        "ingested_at": datetime.now().isoformat(),
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Validation failed: missing data files"
        mock_run.return_value = mock_result

        success, result = reviewer.execute_action("promote", "test_fail")

        assert success is False
        assert "Validation failed" in result["error_message"]


def test_quarantine_reviewer_update_metadata_with_action(temp_data_dir):
    """Test updating metadata with action history."""
    data_dir = Path(temp_data_dir)

    # Create quarantine entry
    source_dir = data_dir / "quarantine" / "test_metadata"
    source_dir.mkdir(parents=True)

    metadata = {
        "source_id": "meta_1",
        "tier": 3,
        "ingested_at": datetime.now().isoformat(),
        "action_history": [],
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    # Update metadata with new action
    updates = {
        "action_history": [
            {
                "action": "promote",
                "timestamp": datetime.now().isoformat(),
                "actor": "approver_agent",
                "result": "success",
            }
        ]
    }
    result = reviewer._update_entry_metadata("test_metadata", updates)

    assert result is True

    # Verify action was recorded
    with open(source_dir / "metadata.json") as f:
        updated = json.load(f)
    assert len(updated["action_history"]) == 1
    assert updated["action_history"][0]["action"] == "promote"


def test_quarantine_reviewer_write_report(temp_data_dir):
    """Test writing review report to disk."""
    data_dir = Path(temp_data_dir)

    # Create pending entry
    source_dir = data_dir / "quarantine" / "report_source"
    source_dir.mkdir()

    metadata = {
        "source_id": "report_1",
        "tier": 3,
        "ingested_at": datetime.now().isoformat(),
        "validation_status": "pending",
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    with open(source_dir / "data.jsonl", "w") as f:
        f.write(json.dumps({"text": "test"}) + "\n")

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)
    report_path = reviewer.write_review_report(
        report_dir=str(data_dir / "approval_queue" / "quarantine_reviews")
    )

    assert report_path is not None
    assert Path(report_path).exists()

    with open(report_path) as f:
        report = json.load(f)
    assert "total_pending" in report
    assert "message" in report


def test_quarantine_reviewer_handle_user_response_promote(temp_data_dir):
    """Test handling user response for dataset promotion."""
    data_dir = Path(temp_data_dir)

    # Create quarantine entry
    source_dir = data_dir / "quarantine" / "dataset_d"
    source_dir.mkdir(parents=True)

    metadata = {
        "source_id": "d_1",
        "tier": 3,
        "ingested_at": datetime.now().isoformat(),
        "validation_status": "passed",
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    with patch.object(reviewer, "execute_action") as mock_action:
        mock_action.return_value = (
            True,
            {
                "action": "promote",
                "entry_name": "dataset_d",
                "stdout": "Promoted",
                "stderr": "",
                "returncode": 0,
                "execution_time": 2.5,
            },
        )

        success, message = reviewer.handle_user_response("promote dataset_d")

        assert success is True
        assert "promoted" in message.lower()
        assert "✅" in message


def test_quarantine_reviewer_handle_user_response_reject(temp_data_dir):
    """Test handling user response for dataset rejection."""
    data_dir = Path(temp_data_dir)

    # Create quarantine entry
    source_dir = data_dir / "quarantine" / "dataset_e"
    source_dir.mkdir(parents=True)

    metadata = {
        "source_id": "e_1",
        "tier": 4,
        "ingested_at": datetime.now().isoformat(),
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    with patch.object(reviewer, "execute_action") as mock_action:
        mock_action.return_value = (
            True,
            {
                "action": "reject",
                "entry_name": "dataset_e",
                "stdout": "Rejected",
                "stderr": "",
                "returncode": 0,
                "execution_time": 1.2,
            },
        )

        success, message = reviewer.handle_user_response("reject dataset_e")

        assert success is True
        assert "rejected" in message.lower()
        assert "✅" in message


def test_quarantine_reviewer_handle_user_response_failure(temp_data_dir):
    """Test handling user response when action fails."""
    data_dir = Path(temp_data_dir)

    # Create quarantine entry
    source_dir = data_dir / "quarantine" / "dataset_f"
    source_dir.mkdir(parents=True)

    metadata = {
        "source_id": "f_1",
        "tier": 3,
        "ingested_at": datetime.now().isoformat(),
    }
    with open(source_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    with patch.object(reviewer, "execute_action") as mock_action:
        mock_action.return_value = (
            False,
            {
                "action": "promote",
                "entry_name": "dataset_f",
                "stdout": "",
                "stderr": "Validation status is pending",
                "returncode": 1,
                "error_message": "Validation status is pending",
            },
        )

        success, message = reviewer.handle_user_response("promote dataset_f")

        assert success is False
        assert "failed" in message.lower()
        assert "❌" in message


def test_quarantine_reviewer_handle_user_response_invalid_format(temp_data_dir):
    """Test handling user response with invalid format."""
    reviewer = QuarantineReviewer(data_dir=temp_data_dir)

    success, message = reviewer.handle_user_response("promote")
    assert success is False
    assert "Invalid format" in message

    success, message = reviewer.handle_user_response("invalid_action dataset_x")
    assert success is False
    assert "Unknown action" in message
