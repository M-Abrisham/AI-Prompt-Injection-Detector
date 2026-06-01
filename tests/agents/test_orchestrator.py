"""Tests for PipelineOrchestrator wiring (reply loops + deploy approval).

The orchestrator previously had no test file. These tests cover the seams the
audit flagged: the quarantine and synthetic reply loops (which used to be
notify-only) and the deploy-approval tuple-unpacking fix.

All external agents and the OpenClaw bridge are mocked; no network or git.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from na0s.agents.orchestrator import PipelineOrchestrator


@pytest.fixture
def orch():
    """Orchestrator with every sub-component replaced by a mock."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "approval_queue").mkdir()
        o = PipelineOrchestrator(data_dir=tmp, use_claude=False)
        o.openclaw = MagicMock()
        o.gate_analyzer = MagicMock()
        o.quarantine_reviewer = MagicMock()
        o.synthetic_validator = MagicMock()
        o.deploy_approver = MagicMock()
        o.approvals_sync = MagicMock()
        yield o


# ----------------------------------------------------------- quarantine loop --

class TestQuarantineReview:
    def test_executes_action_on_reply(self, orch):
        orch.quarantine_reviewer.format_message.return_value = "2 pending\nReply: promote x"
        orch.openclaw.send_message.return_value = True
        orch.quarantine_reviewer.write_review_report.return_value = "/tmp/r.json"
        orch.openclaw.poll_replies.return_value = "promote dataset_d"
        orch.quarantine_reviewer.handle_user_response.return_value = (True, "✅ dataset_d promoted")

        assert orch.run_quarantine_review() is True
        orch.quarantine_reviewer.handle_user_response.assert_called_once_with("promote dataset_d")
        # The result message is relayed back to the user.
        orch.openclaw.send_message.assert_any_call("✅ dataset_d promoted")

    def test_no_reply_is_benign(self, orch):
        orch.quarantine_reviewer.format_message.return_value = "2 pending\nReply: promote x"
        orch.openclaw.send_message.return_value = True
        orch.quarantine_reviewer.write_review_report.return_value = None
        orch.openclaw.poll_replies.return_value = None

        assert orch.run_quarantine_review() is True
        orch.quarantine_reviewer.handle_user_response.assert_not_called()

    def test_empty_backlog_skips_poll(self, orch):
        orch.quarantine_reviewer.format_message.return_value = "Quarantine backlog is empty."
        assert orch.run_quarantine_review() is True
        orch.openclaw.poll_replies.assert_not_called()


# ------------------------------------------------------------ synthetic loop --

class TestSyntheticValidation:
    def test_removes_on_yes(self, orch):
        orch.synthetic_validator.compile_validation_report.return_value = {"flagged_count": 1}
        orch.synthetic_validator.format_message.return_value = "1 flagged\nReply: yes remove"
        orch.openclaw.send_message.return_value = True
        orch.synthetic_validator.write_report.return_value = "/tmp/v.json"
        orch.openclaw.poll_replies.return_value = "yes remove"
        orch.synthetic_validator.handle_user_response.return_value = (True, 1, "✅ Removed 1")

        assert orch.run_synthetic_validation() is True
        orch.synthetic_validator.handle_user_response.assert_called_once_with("yes remove")
        orch.openclaw.send_message.assert_any_call("✅ Removed 1")

    def test_nothing_flagged_skips_poll(self, orch):
        orch.synthetic_validator.compile_validation_report.return_value = {"flagged_count": 0}
        orch.synthetic_validator.format_message.return_value = "all passed"
        orch.openclaw.send_message.return_value = True
        orch.synthetic_validator.write_report.return_value = None

        assert orch.run_synthetic_validation() is True
        orch.openclaw.poll_replies.assert_not_called()

    def test_no_reply_keeps_samples(self, orch):
        orch.synthetic_validator.compile_validation_report.return_value = {"flagged_count": 1}
        orch.synthetic_validator.format_message.return_value = "1 flagged"
        orch.openclaw.send_message.return_value = True
        orch.synthetic_validator.write_report.return_value = None
        orch.openclaw.poll_replies.return_value = None

        assert orch.run_synthetic_validation() is True
        orch.synthetic_validator.handle_user_response.assert_not_called()


# --------------------------------------------------------- deploy approval --

class TestDeploymentApproval:
    def _pending(self, orch):
        orch.approvals_sync.sync_pending.return_value = None
        orch.deploy_approver.get_pending_deployment.return_value = {"status": "pending"}
        orch.deploy_approver.format_approval_message.return_value = "Deploy? approve|reject"
        orch.openclaw.send_message.return_value = True
        orch.approvals_sync.already_notified.return_value = False

    def test_success_path(self, orch):
        self._pending(orch)
        orch.openclaw.poll_replies.return_value = "approve"
        orch.deploy_approver.handle_approval.return_value = (True, "✅ deployed")

        assert orch.run_deployment_approval() is True
        orch.openclaw.send_message.assert_any_call("✅ deployed")
        orch.approvals_sync.mark_finalized.assert_called_once()

    def test_failure_is_detected(self, orch):
        # The old bug: a (False, msg) tuple read as truthy -> failure looked OK.
        self._pending(orch)
        orch.openclaw.poll_replies.return_value = "approve"
        orch.deploy_approver.handle_approval.return_value = (False, "❌ deploy failed")

        assert orch.run_deployment_approval() is False
        orch.openclaw.send_message.assert_any_call("❌ deploy failed")

    def test_nothing_pending(self, orch):
        orch.approvals_sync.sync_pending.return_value = None
        orch.deploy_approver.get_pending_deployment.return_value = None
        assert orch.run_deployment_approval() is True
        orch.openclaw.poll_replies.assert_not_called()

    def test_no_reply_returns_false(self, orch):
        self._pending(orch)
        orch.openclaw.poll_replies.return_value = None
        assert orch.run_deployment_approval() is False

    def test_does_not_renotify(self, orch):
        self._pending(orch)
        orch.approvals_sync.already_notified.return_value = True
        orch.openclaw.poll_replies.return_value = "approve"
        orch.deploy_approver.handle_approval.return_value = (True, "✅ deployed")

        assert orch.run_deployment_approval() is True
        # Already notified -> no fresh approval prompt sent (only the result).
        orch.deploy_approver.format_approval_message.assert_not_called()


# ----------------------------------------- synthetic handle_user_response --

class TestSyntheticHandleResponse:
    """Exercise the real SyntheticValidator reply parser (not mocked)."""

    @pytest.fixture
    def validator(self, tmp_path):
        from na0s.agents.synthetic_validator import SyntheticValidator
        return SyntheticValidator(data_dir=str(tmp_path))

    def test_yes_remove_invokes_removal(self, validator):
        # Count comes from the validation report; removal itself returns bool.
        validator.compile_validation_report = MagicMock(return_value={"flagged_count": 3})
        validator.remove_flagged_samples = MagicMock(return_value=True)
        acted, removed, msg = validator.handle_user_response("yes remove")
        assert acted is True and removed == 3
        assert "3" in msg
        validator.remove_flagged_samples.assert_called_once()

    def test_yes_remove_reports_failure(self, validator):
        validator.compile_validation_report = MagicMock(return_value={"flagged_count": 2})
        validator.remove_flagged_samples = MagicMock(return_value=False)
        acted, removed, msg = validator.handle_user_response("yes remove")
        assert acted is False and removed == 0
        assert "❌" in msg

    def test_keep_all_removes_nothing(self, validator):
        validator.remove_flagged_samples = MagicMock()
        acted, removed, msg = validator.handle_user_response("keep all")
        assert acted is True and removed == 0
        validator.remove_flagged_samples.assert_not_called()

    def test_unknown_reply(self, validator):
        acted, removed, msg = validator.handle_user_response("huh?")
        assert acted is False and removed == 0
