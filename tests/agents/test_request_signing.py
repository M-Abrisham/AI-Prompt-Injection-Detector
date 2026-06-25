"""Tests for HMAC signing/verification of mail-drop approval requests."""

import hashlib
import hmac
import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import na0s.agents.approvals_sync as approvals_sync
from na0s.agents.approvals_sync import (
    ApprovalsSync,
    HMAC_KEY_ENV,
    ENFORCE_ENV,
    MIN_HMAC_KEY_BYTES,
    APPROVAL_MAX_AGE_SECONDS,
    APPROVAL_FUTURE_LEEWAY_SECONDS,
    sign_request,
    sign_pending_request,
    verify_request,
    _approval_key,
    _canonical_request_bytes,
    _is_fresh,
)

KEY = "super-secret-shared-hmac-key"

# A *fresh* ``requested_at`` so the signed-path tests below exercise the S5.3
# freshness guard's accept branch (a stale literal date would now — correctly —
# be rejected as a replay). The reordering test mirrors this exact value.
_FRESH_REQUESTED_AT = datetime.now(timezone.utc).isoformat()

SAMPLE_REQUEST = {
    "type": "deploy_approval",
    "requested_at": _FRESH_REQUESTED_AT,
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
    """Reset the one-time unsigned-mode + weak-key warning latches between tests."""
    approvals_sync._unsigned_mode_warned = False
    approvals_sync._weak_key_warned = False
    yield
    approvals_sync._unsigned_mode_warned = False
    approvals_sync._weak_key_warned = False


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
        "requested_at": _FRESH_REQUESTED_AT,
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


# ============================================================================
# S5.4 — RFC-2104 key floor: warn (loud, one-time) on a < 32-byte key but
# still ACCEPT it (non-breaking).
# ============================================================================

def test_approval_key_warns_below_32_bytes_and_accepts(monkeypatch, caplog):
    short = "x" * (MIN_HMAC_KEY_BYTES - 1)  # 31 bytes < 32
    monkeypatch.setenv(HMAC_KEY_ENV, short)
    with caplog.at_level(logging.WARNING):
        key = _approval_key()
    # ACCEPTED (non-breaking): the key is returned, not rejected.
    assert key == short.encode("utf-8")
    # Loud warning naming the RFC and the floor.
    warnings = [r for r in caplog.records if "RFC 2104" in r.message]
    assert len(warnings) == 1
    assert str(MIN_HMAC_KEY_BYTES) in warnings[0].message


def test_approval_key_warns_only_once(monkeypatch, caplog):
    monkeypatch.setenv(HMAC_KEY_ENV, "x" * 8)
    with caplog.at_level(logging.WARNING):
        _approval_key()
        _approval_key()
    assert len([r for r in caplog.records if "RFC 2104" in r.message]) == 1


def test_approval_key_no_warn_at_or_above_floor(monkeypatch, caplog):
    monkeypatch.setenv(HMAC_KEY_ENV, "y" * MIN_HMAC_KEY_BYTES)  # exactly 32
    with caplog.at_level(logging.WARNING):
        key = _approval_key()
    assert key == ("y" * MIN_HMAC_KEY_BYTES).encode("utf-8")
    assert not [r for r in caplog.records if "RFC 2104" in r.message]


def test_approval_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, f"  {KEY}\n")
    assert _approval_key() == KEY.encode("utf-8")


def test_approval_key_whitespace_only_is_none(monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, "   \n\t")
    assert _approval_key() is None


# ============================================================================
# S5.3 — freshness / replay guard on the SIGNED path only.
# ============================================================================

def _signed_at(offset_seconds: float) -> dict:
    """A validly-signed request whose ``requested_at`` is now + offset."""
    ts = (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()
    req = dict(SAMPLE_REQUEST, requested_at=ts)
    return dict(req, signature=sign_request(req, KEY))


def test_authenticate_rejects_stale_signed_request(sync, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    # One second past the max-age window.
    stale = _signed_at(-(APPROVAL_MAX_AGE_SECONDS + 1))
    _stub_remote(sync, stale)
    assert sync.sync_pending() is None
    assert not sync.pending_path.exists()


def test_authenticate_accepts_fresh_signed_request(sync, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    fresh = _signed_at(0)
    _stub_remote(sync, fresh)
    assert sync.sync_pending() == fresh
    assert sync.pending_path.exists()


def test_authenticate_accepts_request_within_window(sync, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    # Well inside the window (1 hour old).
    recent = _signed_at(-3600)
    _stub_remote(sync, recent)
    assert sync.sync_pending() == recent


def test_authenticate_rejects_far_future_signed_request(sync, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    far_future = _signed_at(APPROVAL_FUTURE_LEEWAY_SECONDS + 60)
    _stub_remote(sync, far_future)
    assert sync.sync_pending() is None
    assert not sync.pending_path.exists()


def test_authenticate_accepts_within_future_leeway(sync, monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    near_future = _signed_at(APPROVAL_FUTURE_LEEWAY_SECONDS - 30)
    _stub_remote(sync, near_future)
    assert sync.sync_pending() == near_future


def test_authenticate_rejects_signed_request_without_timestamp(sync, monkeypatch):
    """A signed request the producer always stamps must carry a usable timestamp."""
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    no_ts = {k: v for k, v in SAMPLE_REQUEST.items() if k != "requested_at"}
    signed = dict(no_ts, signature=sign_request(no_ts, KEY))
    _stub_remote(sync, signed)
    assert sync.sync_pending() is None


def test_freshness_not_applied_on_unsigned_path(sync, monkeypatch):
    """A STALE timestamp must NOT be rejected in unsigned mode (S5.3 signed-only)."""
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    monkeypatch.delenv(ENFORCE_ENV, raising=False)
    stale_ts = (
        datetime.now(timezone.utc) - timedelta(seconds=APPROVAL_MAX_AGE_SECONDS * 10)
    ).isoformat()
    stale_unsigned = dict(SAMPLE_REQUEST, requested_at=stale_ts)  # no signature
    _stub_remote(sync, stale_unsigned)
    # Unsigned staged-rollout path is unchanged: accepted despite the stale ts.
    assert sync.sync_pending() == stale_unsigned


# ============================================================================
# S5.2 — producer/consumer canonicalization parity.
#
# Locks the inline json.dumps formula in .github/workflows/auto-retrain.yml
# against sign_request / _canonical_request_bytes. The fixture deliberately
# includes a NON-ASCII string and a NUMERIC field — the two cases most likely
# to drift between the two implementations (ensure_ascii, int vs float repr).
# ============================================================================

def test_producer_consumer_canonicalization_parity():
    payload = {
        "type": "deploy_approval",
        "requested_at": "2026-06-25T09:00:00+00:00",
        "candidate_path": "data/processed/café_v7",  # non-ASCII
        "exit_code": 0,                                # numeric
        "score": 0.9375,                               # numeric (float)
        "gates": {"canary": {"passed": True}},
        "summary": "naïve café — déjà vu",             # more non-ASCII
        "status": "pending",
    }

    # EXACT producer formula, copied verbatim from auto-retrain.yml's signing
    # block (the YAML re-implements sign_request inline). If the YAML ever
    # drifts from this, this assertion is the canary.
    producer_canon = json.dumps(
        {k: v for k, v in payload.items() if k != "signature"},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")

    # Consumer side-by-side.
    assert _canonical_request_bytes(payload) == producer_canon

    # And the full HMAC must agree byte-for-byte.
    producer_sig = hmac.new(
        KEY.encode("utf-8"), producer_canon, hashlib.sha256
    ).hexdigest()
    assert sign_request(payload, KEY) == producer_sig


def test_parity_fixture_actually_contains_nonascii_and_numeric():
    """Guard the parity test's own teeth: the fixture must exercise both cases."""
    payload = {"candidate_path": "data/processed/café_v7", "exit_code": 0}
    canon = _canonical_request_bytes(payload)
    # ensure_ascii defaults True -> the non-ASCII char is \u-escaped, NOT raw.
    assert b"caf\\u00e9" in canon
    assert b'"exit_code":0' in canon


def test_yaml_producer_block_uses_matching_formula():
    """Lock the YAML producer block: assert the canonical-form call is present
    verbatim (we do NOT rewrite the YAML; this test pins it)."""
    yaml_path = (
        Path(__file__).resolve().parents[2]
        / ".github" / "workflows" / "auto-retrain.yml"
    )
    text = yaml_path.read_text()
    # The three load-bearing tokens of the canonical formula.
    assert 'sort_keys=True, separators=(",", ":")' in text
    assert 'k != "signature"' in text
    assert "hashlib.sha256" in text


# ============================================================================
# S5.1 — 3-state enforce flag (off/warn/enforce), default = warn+accept.
# ============================================================================

def _signed_fresh() -> dict:
    return _signed_at(0)


@pytest.mark.parametrize("mode", ["off", "warn"])
def test_enforce_flag_off_and_warn_accept_unsigned(sync, monkeypatch, mode):
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    monkeypatch.setenv(ENFORCE_ENV, mode)
    _stub_remote(sync, SAMPLE_REQUEST)  # unsigned
    assert sync.sync_pending() == SAMPLE_REQUEST
    assert sync.pending_path.exists()


def test_enforce_flag_default_is_warn_accept_unsigned(sync, monkeypatch):
    """Unset NA0S_AGENT_APPROVAL_ENFORCE => today's behavior (warn + accept)."""
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    monkeypatch.delenv(ENFORCE_ENV, raising=False)
    _stub_remote(sync, SAMPLE_REQUEST)
    assert sync.sync_pending() == SAMPLE_REQUEST


def test_enforce_flag_enforce_rejects_unsigned_when_key_unset(sync, monkeypatch):
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    monkeypatch.setenv(ENFORCE_ENV, "enforce")
    _stub_remote(sync, SAMPLE_REQUEST)  # unsigned, no key
    assert sync.sync_pending() is None
    assert not sync.pending_path.exists()


def test_enforce_flag_is_case_insensitive(sync, monkeypatch):
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    monkeypatch.setenv(ENFORCE_ENV, "  ENFORCE  ")
    _stub_remote(sync, SAMPLE_REQUEST)
    assert sync.sync_pending() is None


def test_enforce_flag_unknown_value_falls_back_to_warn(sync, monkeypatch):
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    monkeypatch.setenv(ENFORCE_ENV, "bogus")
    _stub_remote(sync, SAMPLE_REQUEST)
    # Fail-safe: a typo does not silently reject every request.
    assert sync.sync_pending() == SAMPLE_REQUEST


@pytest.mark.parametrize("mode", ["off", "warn", "enforce"])
def test_enforce_all_states_accept_validly_signed_request(sync, monkeypatch, mode):
    """A validly-signed (and fresh) request is accepted under EVERY enforce state
    once a key is configured — the flag only governs the keyless/unsigned path."""
    monkeypatch.setenv(HMAC_KEY_ENV, KEY)
    monkeypatch.setenv(ENFORCE_ENV, mode)
    signed = _signed_fresh()
    _stub_remote(sync, signed)
    assert sync.sync_pending() == signed
    assert sync.pending_path.exists()
