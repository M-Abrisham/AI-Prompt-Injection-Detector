"""Tests for the cloud->local git mail-drop sync (approvals_sync)."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from na0s.agents.approvals_sync import ApprovalsSync, PENDING_REL_PATH


def _cp(returncode=0, stdout="", stderr=""):
    """Build a fake CompletedProcess."""
    return subprocess.CompletedProcess(
        args=["git"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture
def sync(tmp_path):
    """An ApprovalsSync rooted in a temp dir (no real git)."""
    data_dir = tmp_path / "data"
    (data_dir / "approval_queue").mkdir(parents=True)
    return ApprovalsSync(
        data_dir=str(data_dir),
        branch="agent-approvals",
        remote="origin",
        repo_root=str(tmp_path),
    )


SAMPLE_REQUEST = {
    "type": "deploy_approval",
    "requested_at": "2026-05-13T09:00:00Z",
    "candidate_path": "data/processed/",
    "gates": {"canary": {"passed": True}},
    "status": "pending",
}


# --------------------------------------------------------------- request_id --

class TestRequestId:
    def test_is_deterministic(self):
        assert ApprovalsSync.request_id(SAMPLE_REQUEST) == ApprovalsSync.request_id(
            dict(SAMPLE_REQUEST)
        )

    def test_ignores_volatile_status_keys(self):
        base = ApprovalsSync.request_id(SAMPLE_REQUEST)
        mutated = dict(SAMPLE_REQUEST)
        mutated["status"] = "approved"
        mutated["status_updated_at"] = "2026-05-13T10:00:00Z"
        mutated["execution_result"] = {"success": True}
        assert ApprovalsSync.request_id(mutated) == base

    def test_changes_with_real_content(self):
        other = dict(SAMPLE_REQUEST)
        other["requested_at"] = "2026-06-01T00:00:00Z"
        assert ApprovalsSync.request_id(other) != ApprovalsSync.request_id(SAMPLE_REQUEST)

    def test_is_16_chars(self):
        assert len(ApprovalsSync.request_id(SAMPLE_REQUEST)) == 16


# ------------------------------------------------------------------- state --

class TestState:
    def test_notify_roundtrip(self, sync):
        assert sync.already_notified(SAMPLE_REQUEST) is False
        sync.mark_notified(SAMPLE_REQUEST)
        assert sync.already_notified(SAMPLE_REQUEST) is True

    def test_notify_is_idempotent(self, sync):
        sync.mark_notified(SAMPLE_REQUEST)
        sync.mark_notified(SAMPLE_REQUEST)
        state = json.loads(sync.state_path.read_text())
        assert state["notified"].count(ApprovalsSync.request_id(SAMPLE_REQUEST)) == 1

    def test_finalized_roundtrip(self, sync):
        sync.mark_finalized(SAMPLE_REQUEST)
        state = json.loads(sync.state_path.read_text())
        assert ApprovalsSync.request_id(SAMPLE_REQUEST) in state["finalized"]

    def test_corrupt_state_resets(self, sync):
        sync.state_path.write_text("not json{{")
        # Should not raise; treated as empty state.
        assert sync.already_notified(SAMPLE_REQUEST) is False


# ------------------------------------------------------- fetch_remote_request --

class TestFetchRemoteRequest:
    def test_returns_none_when_fetch_fails(self, sync):
        sync._git = MagicMock(return_value=_cp(returncode=1, stderr="no such branch"))
        assert sync.fetch_remote_request() is None
        sync._git.assert_called_once()  # stops after failed fetch

    def test_returns_none_when_file_missing(self, sync):
        sync._git = MagicMock(side_effect=[_cp(0), _cp(returncode=128, stderr="path not found")])
        assert sync.fetch_remote_request() is None

    def test_returns_none_on_invalid_json(self, sync):
        sync._git = MagicMock(side_effect=[_cp(0), _cp(0, stdout="{bad json")])
        assert sync.fetch_remote_request() is None

    def test_returns_parsed_request(self, sync):
        sync._git = MagicMock(
            side_effect=[_cp(0), _cp(0, stdout=json.dumps(SAMPLE_REQUEST))]
        )
        result = sync.fetch_remote_request()
        assert result == SAMPLE_REQUEST

    def test_uses_correct_show_ref(self, sync):
        sync._git = MagicMock(
            side_effect=[_cp(0), _cp(0, stdout=json.dumps(SAMPLE_REQUEST))]
        )
        sync.fetch_remote_request()
        show_call = sync._git.call_args_list[1]
        assert show_call.args == ("show", f"origin/agent-approvals:{PENDING_REL_PATH}")


# ---------------------------------------------------------------- sync_pending --

class TestSyncPending:
    def _stub_remote(self, sync, request):
        sync._git = MagicMock(side_effect=[_cp(0), _cp(0, stdout=json.dumps(request))])

    def test_returns_none_when_nothing_remote(self, sync):
        sync._git = MagicMock(return_value=_cp(returncode=1, stderr="none"))
        assert sync.sync_pending() is None
        assert not sync.pending_path.exists()

    def test_returns_none_when_not_pending(self, sync):
        approved = dict(SAMPLE_REQUEST, status="approved")
        self._stub_remote(sync, approved)
        assert sync.sync_pending() is None
        assert not sync.pending_path.exists()

    def test_materializes_new_pending_request(self, sync):
        self._stub_remote(sync, SAMPLE_REQUEST)
        result = sync.sync_pending()
        assert result == SAMPLE_REQUEST
        assert sync.pending_path.exists()
        on_disk = json.loads(sync.pending_path.read_text())
        assert on_disk["status"] == "pending"

    def test_skips_finalized_request(self, sync):
        sync.mark_finalized(SAMPLE_REQUEST)
        self._stub_remote(sync, SAMPLE_REQUEST)
        assert sync.sync_pending() is None
        # Must not overwrite/recreate the local pending file for a done request.
        assert not sync.pending_path.exists()
