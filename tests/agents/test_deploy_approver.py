"""Tests for DeployApprover agent."""

import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest

from na0s.agents.deploy_approver import DeployApprover


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with approval_queue structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "approval_queue").mkdir()
        yield str(data_dir)


def test_deploy_approver_no_pending(temp_data_dir):
    """Test when there's no pending deployment."""
    approver = DeployApprover(data_dir=temp_data_dir)
    pending = approver.get_pending_deployment()

    assert pending is None


def test_deploy_approver_get_pending(temp_data_dir):
    """Test retrieving pending deployment."""
    data_dir = Path(temp_data_dir)

    # Create pending deployment
    pending_deploy = {
        "type": "deploy_approval",
        "requested_at": datetime.now().isoformat(),
        "candidate_path": "data/processed/",
        "gates": {
            "canary": {"passed": True, "tpr": 0.97, "tnr": 0.93},
            "shadow": {"passed": True},
            "f14": {"passed": True, "overall_tpr": 0.85},
        },
        "status": "pending",
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(pending_deploy, f)

    approver = DeployApprover(data_dir=temp_data_dir)
    pending = approver.get_pending_deployment()

    assert pending is not None
    assert pending["status"] == "pending"
    assert pending["gates"]["canary"]["tpr"] == 0.97


def test_deploy_approver_get_approved_deployment(temp_data_dir):
    """Test that approved deployments are not returned as pending."""
    data_dir = Path(temp_data_dir)

    # Create approved deployment
    deployment = {
        "status": "approved",
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(deployment, f)

    approver = DeployApprover(data_dir=temp_data_dir)
    pending = approver.get_pending_deployment()

    assert pending is None


def test_deploy_approver_format_message(temp_data_dir):
    """Test formatting approval message for iMessage."""
    data_dir = Path(temp_data_dir)

    deployment = {
        "status": "pending",
        "gates": {
            "canary": {"passed": True, "tpr": 0.97, "tnr": 0.93},
            "shadow": {"passed": True, "fpr_delta": 0.003, "recall_delta": -0.001},
            "decontam": {"passed": True},
            "f14": {"passed": True, "overall_tpr": 0.85},
        },
        "summary": "F1 improved 2% vs production",
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(deployment, f)

    approver = DeployApprover(data_dir=temp_data_dir)
    message = approver.format_approval_message(deployment)

    assert isinstance(message, str)
    assert "ready to deploy" in message.lower() or "all gates" in message.lower()
    assert "approve" in message.lower()
    assert "reject" in message.lower()


def test_deploy_approver_update_status(temp_data_dir):
    """Test updating deployment status."""
    data_dir = Path(temp_data_dir)

    deployment = {
        "status": "pending",
        "gates": {},
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(deployment, f)

    approver = DeployApprover(data_dir=temp_data_dir)
    result = approver.update_status("approved", "User approved")

    assert result is True

    # Verify status was updated
    with open(data_dir / "approval_queue" / "pending_deploy.json") as f:
        updated = json.load(f)
    assert updated["status"] == "approved"
    assert updated["status_note"] == "User approved"


def test_deploy_approver_handle_approval(temp_data_dir):
    """Test handling approval response (legacy test updated for Phase 6)."""
    data_dir = Path(temp_data_dir)

    deployment = {
        "status": "pending",
        "gates": {},
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(deployment, f)

    approver = DeployApprover(data_dir=temp_data_dir)

    # Test rejection (shouldn't try to deploy)
    success, message = approver.handle_approval("reject")
    assert success is True

    # Check status is updated
    with open(data_dir / "approval_queue" / "pending_deploy.json") as f:
        updated = json.load(f)
    assert updated["status"] == "rejected"


def test_deploy_approver_check_and_notify_no_pending(temp_data_dir):
    """Test check_and_notify when no pending deployment."""
    approver = DeployApprover(data_dir=temp_data_dir)
    message = approver.check_and_notify()

    assert message is None


def test_deploy_approver_check_and_notify_pending(temp_data_dir):
    """Test check_and_notify when deployment is pending."""
    data_dir = Path(temp_data_dir)

    deployment = {
        "status": "pending",
        "gates": {
            "canary": {"passed": True, "tpr": 0.97, "tnr": 0.93},
            "shadow": {"passed": True},
        },
        "summary": "All good",
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(deployment, f)

    approver = DeployApprover(data_dir=temp_data_dir)
    message = approver.check_and_notify()

    assert message is not None
    assert "All gates passed" in message or "ready" in message.lower()


def test_deploy_approver_execute_deploy_success(temp_data_dir):
    """Test successful deployment execution."""
    approver = DeployApprover(data_dir=temp_data_dir)

    # Mock subprocess.run to simulate successful deployment
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Deploy completed successfully"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        success, result = approver.execute_deploy()

        assert success is True
        assert result["returncode"] == 0
        assert result["execution_time"] >= 0
        assert result["retry_count"] == 0


def test_deploy_approver_execute_deploy_failure(temp_data_dir):
    """Test deployment execution failure with retry."""
    approver = DeployApprover(data_dir=temp_data_dir)

    # Mock subprocess.run to simulate failure then success (retry)
    with patch("subprocess.run") as mock_run:
        # First call fails, second succeeds
        mock_fail = MagicMock()
        mock_fail.returncode = 1
        mock_fail.stdout = ""
        mock_fail.stderr = "Model file not found"

        mock_succeed = MagicMock()
        mock_succeed.returncode = 0
        mock_succeed.stdout = "Deploy completed"
        mock_succeed.stderr = ""

        mock_run.side_effect = [mock_fail, mock_succeed]

        success, result = approver.execute_deploy(retry_on_failure=True)

        assert success is True
        assert result["retry_count"] == 1


def test_deploy_approver_execute_deploy_timeout(temp_data_dir):
    """Test deployment execution timeout."""
    approver = DeployApprover(data_dir=temp_data_dir)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = TimeoutError("Deployment timed out")

        success, result = approver.execute_deploy()

        assert success is False
        assert "Execution error" in result["error_message"]


def test_deploy_approver_update_status_with_execution(temp_data_dir):
    """Test updating deployment status with execution metadata."""
    data_dir = Path(temp_data_dir)

    deployment = {
        "status": "pending",
        "gates": {},
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(deployment, f)

    approver = DeployApprover(data_dir=temp_data_dir)

    # Update with execution result
    exec_result = {
        "stdout": "Deploy successful",
        "stderr": "",
        "returncode": 0,
        "execution_time": 12.5,
        "retry_count": 0,
    }
    result = approver.update_status("approved", "Deployed", exec_result)

    assert result is True

    # Verify status was updated with execution metadata
    with open(data_dir / "approval_queue" / "pending_deploy.json") as f:
        updated = json.load(f)

    assert updated["status"] == "approved"
    assert "executed_at" in updated
    assert "execution_result" in updated
    assert updated["execution_result"]["success"] is True
    assert updated["execution_result"]["execution_time_seconds"] == 12.5


def test_deploy_approver_handle_approval_approve(temp_data_dir):
    """Test handling approval response with successful deployment."""
    data_dir = Path(temp_data_dir)

    deployment = {
        "status": "pending",
        "gates": {},
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(deployment, f)

    approver = DeployApprover(data_dir=temp_data_dir)

    with patch.object(approver, "execute_deploy") as mock_deploy:
        mock_deploy.return_value = (
            True,
            {
                "stdout": "Success",
                "stderr": "",
                "returncode": 0,
                "execution_time": 5.0,
                "retry_count": 0,
            },
        )

        success, message = approver.handle_approval("approve")

        assert success is True
        assert "deployed successfully" in message.lower()
        assert "✅" in message


def test_deploy_approver_handle_approval_reject(temp_data_dir):
    """Test handling rejection response."""
    data_dir = Path(temp_data_dir)

    deployment = {
        "status": "pending",
        "gates": {},
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(deployment, f)

    approver = DeployApprover(data_dir=temp_data_dir)
    success, message = approver.handle_approval("reject")

    assert success is True
    assert "rejected" in message.lower()
    assert "✅" in message

    # Verify status was updated
    with open(data_dir / "approval_queue" / "pending_deploy.json") as f:
        updated = json.load(f)
    assert updated["status"] == "rejected"


def test_deploy_approver_handle_approval_deploy_failure(temp_data_dir):
    """Test handling approval when deployment fails."""
    data_dir = Path(temp_data_dir)

    deployment = {
        "status": "pending",
        "gates": {},
    }
    with open(data_dir / "approval_queue" / "pending_deploy.json", "w") as f:
        json.dump(deployment, f)

    approver = DeployApprover(data_dir=temp_data_dir)

    with patch.object(approver, "execute_deploy") as mock_deploy:
        mock_deploy.return_value = (
            False,
            {
                "stdout": "",
                "stderr": "Model file not found",
                "returncode": 1,
                "execution_time": 2.0,
                "retry_count": 1,
                "error_message": "Model file not found",
            },
        )

        success, message = approver.handle_approval("approve")

        assert success is False
        assert "failed" in message.lower()
        assert "❌" in message

        # Verify status was updated to failed
        with open(data_dir / "approval_queue" / "pending_deploy.json") as f:
            updated = json.load(f)
        assert updated["status"] == "failed"
