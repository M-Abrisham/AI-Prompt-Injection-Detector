"""Tests for HMAC signing/verification of mail-drop approval requests."""

import json
import subprocess
from unittest.mock import MagicMock

import pytest

import na0s.agents.approvals_sync as approvals_sync
from na0s.agents.approvals_sync import (
    ApprovalsSync,
    HMAC_KEY_ENV,
    sign_request,
    sign_pending_request,
    verify_request,
    _approval_key,
)

KEY = "super-secret-shared-hmac-key"

SAMPLE_REQUEST = {
    "type": "deploy_approval",
    "requested_at": "2026-05-13T09:00:00Z",
    "candidate_path": "data/processed/",
    "gates": {"canary": {"passed": True}},
    "status": "pending",
}


def _cp(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["git"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture
def sync(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "approval_queue").mkdir(parents=True)
    return ApprovalsSync(
        data_dir=str(data_dir),
        branch="agent-approvals",
        remote="origin",
        repo_root=str(tmp_path),
    )


@pytest.fixture(autouse=True)
def _reset_unsigned_warning():
    """Reset the one-time unsigned-mode warning latch between tests."""
    approvals_sync._unsigned_mode_warned = False
    yield
    approvals_sync._unsigned_mode_warned = False


# ------------------------------------------------------- sign / verify --

def test_sign_verify_roundtrip():
    sig = sign_request(SAMPLE_REQUEST, KEY)
    signed = dict(SAMPLE_REQUEST, signature=sig)
    assert verify_request(signed, KEY) is True


def test_sign_accepts_bytes_and_str_key():
    assert sign_request(SAMPLE_REQUEST, KEY) == sign_request(
        SAMPLE_REQUEST, KEY.encode("utf-8")
    )


def test_tampered_field_fails_verify():
    sig = sign_request(SAMPLE_REQUEST, KEY)
    signed = dict(SAMPLE_REQUEST, signature=sig)
    signed["candidate_path"] = "data/EVIL/"
    assert verify_request(signed, KEY) is False


def test_missing_signature_fails_verify():
    assert verify_request(dict(SAMPLE_REQUEST), KEY) is False
    assert verify_request(dict(SAMPLE_REQUEST, signature=""), KEY) is False


def test_wrong_key_fails_verify():
    sig = sign_request(SAMPLE_REQUEST, KEY)
    signed = dict(SAMPLE_REQUEST, signature=sig)
    assert verify_request(signed, "a-different-key") is False


def test_canonical_form_stable_under_key_reordering():
    # Same content, different insertion order + a stray signature field must
    # produce the same signature (signature excluded; keys sorted).
    reordered = {
        "status": "pending",
        "gates": {"canary": {"passed": True}},
        "candidate_path": "data/processed/",
        "requested_at": "2026-05-13T09:00:00Z",
        "type": "deploy_approval",
        "signature": "stale-ignored-value",
    }
    assert sign_request(reordered, KEY) == sign_request(SAMPLE_REQUEST, KEY)


def test_signature_field_excluded_from_its_own_signature():
    sig = sign_request(SAMPLE_REQUEST, KEY)
    # Re-signing the already-signed dict yields the same signature (idempotent).
    assert sign_request(dict(SAMPLE_REQUEST, signature=sig), KEY) == sig


# ------------------------------------------------------ _approval_key --

def test_approval_key_none_when_unset(monkeypatch):
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    assert _approval_key() is None


def test_approval_key_bytes_when_set(monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    assert _approval_key() == KEY.encode("utf-8")


# ------------------------------------------------ sign_pending_request --

def test_sign_pending_request_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    path = tmp_path / "pending_deploy.json"
    path.write_text(json.dumps(SAMPLE_REQUEST))
    sig = sign_pending_request(path)
    on_disk = json.loads(path.read_text())
    assert on_disk["signature"] == sig
    assert verify_request(on_disk, KEY) is True


def test_sign_pending_request_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    path = tmp_path / "pending_deploy.json"
    path.write_text(json.dumps(SAMPLE_REQUEST))
    with pytest.raises(RuntimeError):
        sign_pending_request(path)


# ------------------------------------------- sync_pending enforcement --

def _stub_remote(sync, request):
    sync._git = MagicMock(side_effect=[_cp(0), _cp(0, stdout=json.dumps(request))])


def test_sync_pending_rejects_unsigned_when_key_set(sync, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    _stub_remote(sync, SAMPLE_REQUEST)  # no signature
    assert sync.sync_pending() is None
    assert not sync.pending_path.exists()


def test_sync_pending_rejects_tampered_when_key_set(sync, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    sig = sign_request(SAMPLE_REQUEST, KEY)
    forged = dict(SAMPLE_REQUEST, signature=sig, candidate_path="data/EVIL/")
    _stub_remote(sync, forged)
    assert sync.sync_pending() is None
    assert not sync.pending_path.exists()


def test_sync_pending_accepts_valid_signature(sync, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    sig = sign_request(SAMPLE_REQUEST, KEY)
    signed = dict(SAMPLE_REQUEST, signature=sig)
    _stub_remote(sync, signed)
    result = sync.sync_pending()
    assert result == signed
    assert sync.pending_path.exists()


def test_sync_pending_accepts_unsigned_when_no_key(sync, monkeypatch, caplog):
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    _stub_remote(sync, SAMPLE_REQUEST)
    import logging

    with caplog.at_level(logging.WARNING):
        result = sync.sync_pending()
    assert result == SAMPLE_REQUEST
    assert sync.pending_path.exists()
    # Loud one-time warning that signing is disabled.
    assert any("signing is DISABLED" in r.message for r in caplog.records)


def test_unsigned_mode_warns_once(sync, monkeypatch, caplog):
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    import logging

    with caplog.at_level(logging.WARNING):
        sync._git = MagicMock(
            side_effect=[
                _cp(0), _cp(0, stdout=json.dumps(SAMPLE_REQUEST)),
                _cp(0), _cp(0, stdout=json.dumps(SAMPLE_REQUEST)),
            ]
        )
        sync.sync_pending()
        sync.sync_pending()
    disabled_warnings = [r for r in caplog.records if "signing is DISABLED" in r.message]
    assert len(disabled_warnings) == 1


def test_fetch_remote_rejects_unsigned_when_key_set(sync, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    sync._git = MagicMock(side_effect=[_cp(0), _cp(0, stdout=json.dumps(SAMPLE_REQUEST))])
    assert sync.fetch_remote_request() is None
