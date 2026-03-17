"""Tests for Layer 11 safe_pickle bug fixes (BUG-L11-2 through BUG-L11-6).

Covers:
- BUG-L11-2: Atomic write pattern
- BUG-L11-3: Algorithm versioning in sidecar files
- BUG-L11-4: Audit logging for integrity events
- BUG-L11-5: File permission checks
- BUG-L11-6: Pickle magic byte validation
"""

import json
import logging
import os
import pickle
import stat
import tempfile
import warnings
from unittest.mock import patch

import pytest

os.environ["SCAN_TIMEOUT_SEC"] = "0"

from na0s.safe_pickle import (
    _atomic_write_binary,
    _atomic_write_text,
    _check_permissions,
    _format_sidecar,
    _hash_path,
    _hmac_path,
    _parse_sidecar,
    _validate_pickle_magic,
    safe_dump,
    safe_load,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def pkl_path(tmp_dir):
    return os.path.join(tmp_dir, "test_l11_model.pkl")


@pytest.fixture
def sample_obj():
    return {"weights": [1.0, 2.0], "bias": 0.1}


# ---------------------------------------------------------------------------
# BUG-L11-2: Atomic write pattern
# ---------------------------------------------------------------------------

class TestAtomicWrite:

    def test_atomic_write_binary_creates_file(self, tmp_dir):
        target = os.path.join(tmp_dir, "out.bin")
        _atomic_write_binary(target, b"hello")
        with open(target, "rb") as f:
            assert f.read() == b"hello"

    def test_atomic_write_text_creates_file(self, tmp_dir):
        target = os.path.join(tmp_dir, "out.txt")
        _atomic_write_text(target, "world")
        with open(target, "r") as f:
            assert f.read() == "world"

    def test_atomic_write_overwrites_existing(self, tmp_dir):
        target = os.path.join(tmp_dir, "out.bin")
        _atomic_write_binary(target, b"old")
        _atomic_write_binary(target, b"new")
        with open(target, "rb") as f:
            assert f.read() == b"new"

    def test_atomic_write_no_temp_file_left(self, tmp_dir):
        target = os.path.join(tmp_dir, "out.bin")
        _atomic_write_binary(target, b"data")
        files = os.listdir(tmp_dir)
        assert files == ["out.bin"], f"unexpected files: {files}"

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "atomickey"})
    def test_safe_dump_uses_atomic_write(self, pkl_path, sample_obj):
        """safe_dump should produce valid pkl + sidecar with no leftover temps."""
        safe_dump(sample_obj, pkl_path)
        parent = os.path.dirname(pkl_path)
        tmp_files = [f for f in os.listdir(parent) if f.endswith(".tmp")]
        assert tmp_files == [], f"temp files left behind: {tmp_files}"
        assert os.path.exists(pkl_path)
        assert os.path.exists(_hmac_path(pkl_path))


# ---------------------------------------------------------------------------
# BUG-L11-3: Algorithm versioning in sidecar files
# ---------------------------------------------------------------------------

class TestSidecarVersioning:

    def test_format_sidecar_sha256(self):
        assert _format_sidecar("sha256", "abcd1234") == "v1:sha256:abcd1234"

    def test_format_sidecar_hmac(self):
        assert _format_sidecar("hmac-sha256", "ef56") == "v1:hmac-sha256:ef56"

    def test_parse_versioned_sidecar(self):
        assert _parse_sidecar("v1:sha256:abcdef0123456789") == "abcdef0123456789"

    def test_parse_legacy_bare_hex(self):
        bare = "a" * 64
        assert _parse_sidecar(bare) == bare

    def test_parse_versioned_with_whitespace(self):
        assert _parse_sidecar("  v1:hmac-sha256:deadbeef\n") == "deadbeef"

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "verkey"})
    def test_safe_dump_writes_versioned_hmac_sidecar(self, pkl_path, sample_obj):
        safe_dump(sample_obj, pkl_path)
        with open(_hmac_path(pkl_path), "r") as f:
            content = f.read().strip()
        assert content.startswith("v1:hmac-sha256:")

    def test_safe_dump_writes_versioned_sha256_sidecar(self, pkl_path, sample_obj):
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                safe_dump(sample_obj, pkl_path)
        with open(_hash_path(pkl_path), "r") as f:
            content = f.read().strip()
        assert content.startswith("v1:sha256:")

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "verkey2"})
    def test_roundtrip_versioned_sidecar(self, pkl_path, sample_obj):
        """Dump with versioned format, load should still work."""
        safe_dump(sample_obj, pkl_path)
        loaded = safe_load(pkl_path)
        assert loaded == sample_obj

    def test_backward_compat_bare_hex_sidecar(self, pkl_path, sample_obj):
        """A legacy bare-hex sidecar should still be loadable."""
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            # Write pickle manually and create a legacy bare-hex sidecar
            with open(pkl_path, "wb") as f:
                pickle.dump(sample_obj, f)
            from na0s.safe_pickle import _sha256
            digest = _sha256(pkl_path)
            with open(_hash_path(pkl_path), "w") as f:
                f.write(digest)  # bare hex, no v1: prefix
            loaded = safe_load(pkl_path)
            assert loaded == sample_obj


# ---------------------------------------------------------------------------
# BUG-L11-4: Audit logging
# ---------------------------------------------------------------------------

class TestAuditLogging:

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "auditkey"})
    def test_safe_dump_logs_info(self, pkl_path, sample_obj, caplog):
        with caplog.at_level(logging.INFO, logger="na0s.integrity_audit"):
            safe_dump(sample_obj, pkl_path)
        dump_records = [r for r in caplog.records
                        if r.name == "na0s.integrity_audit" and r.levelno == logging.INFO]
        assert len(dump_records) >= 1
        msg = json.loads(dump_records[0].message)
        assert msg["event"] == "safe_dump"
        assert msg["algorithm"] == "hmac-sha256"
        assert "digest_prefix" in msg

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "auditkey"})
    def test_safe_load_logs_info_on_success(self, pkl_path, sample_obj, caplog):
        safe_dump(sample_obj, pkl_path)
        with caplog.at_level(logging.INFO, logger="na0s.integrity_audit"):
            caplog.clear()
            safe_load(pkl_path)
        load_records = [r for r in caplog.records
                        if r.name == "na0s.integrity_audit"
                        and r.levelno == logging.INFO
                        and "safe_load" in r.message]
        assert len(load_records) >= 1
        msg = json.loads(load_records[0].message)
        assert msg["result"] == "ok"

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "auditkey"})
    def test_tamper_logs_error(self, pkl_path, sample_obj, caplog):
        safe_dump(sample_obj, pkl_path)
        # Tamper with pickle
        with open(pkl_path, "wb") as f:
            pickle.dump({"evil": True}, f)
        with caplog.at_level(logging.ERROR, logger="na0s.integrity_audit"):
            caplog.clear()
            with pytest.raises(ValueError, match="Integrity check failed"):
                safe_load(pkl_path)
        err_records = [r for r in caplog.records
                       if r.name == "na0s.integrity_audit"
                       and r.levelno == logging.ERROR]
        assert len(err_records) >= 1
        msg = json.loads(err_records[0].message)
        assert msg["event"] == "integrity_failure"


# ---------------------------------------------------------------------------
# BUG-L11-5: File permission checks
# ---------------------------------------------------------------------------

class TestPermissionChecks:

    @pytest.mark.skipif(os.name != "posix", reason="POSIX only")
    def test_world_readable_warns(self, tmp_dir, caplog):
        target = os.path.join(tmp_dir, "open.bin")
        with open(target, "wb") as f:
            f.write(b"data")
        os.chmod(target, 0o644)  # world-readable
        with caplog.at_level(logging.WARNING, logger="na0s.integrity_audit"):
            _check_permissions(target, label="pickle")
        warn_records = [r for r in caplog.records
                        if r.name == "na0s.integrity_audit"
                        and "world-readable" in r.message]
        assert len(warn_records) >= 1

    @pytest.mark.skipif(os.name != "posix", reason="POSIX only")
    def test_group_writable_warns(self, tmp_dir, caplog):
        target = os.path.join(tmp_dir, "gw.bin")
        with open(target, "wb") as f:
            f.write(b"data")
        os.chmod(target, 0o660)  # group-writable, not world-readable
        with caplog.at_level(logging.WARNING, logger="na0s.integrity_audit"):
            _check_permissions(target, label="sidecar")
        warn_records = [r for r in caplog.records
                        if r.name == "na0s.integrity_audit"
                        and "group-writable" in r.message]
        assert len(warn_records) >= 1

    @pytest.mark.skipif(os.name != "posix", reason="POSIX only")
    def test_restrictive_perms_no_warning(self, tmp_dir, caplog):
        target = os.path.join(tmp_dir, "safe.bin")
        with open(target, "wb") as f:
            f.write(b"data")
        os.chmod(target, 0o600)
        with caplog.at_level(logging.WARNING, logger="na0s.integrity_audit"):
            _check_permissions(target, label="pickle")
        warn_records = [r for r in caplog.records
                        if r.name == "na0s.integrity_audit"
                        and r.levelno == logging.WARNING]
        assert len(warn_records) == 0


# ---------------------------------------------------------------------------
# BUG-L11-6: Pickle magic byte validation
# ---------------------------------------------------------------------------

class TestPickleMagicValidation:

    def test_valid_protocol2_accepted(self, tmp_dir):
        path = os.path.join(tmp_dir, "p2.pkl")
        with open(path, "wb") as f:
            pickle.dump({"a": 1}, f, protocol=2)
        _validate_pickle_magic(path)  # should not raise

    def test_valid_protocol4_accepted(self, tmp_dir):
        path = os.path.join(tmp_dir, "p4.pkl")
        with open(path, "wb") as f:
            pickle.dump({"a": 1}, f, protocol=4)
        _validate_pickle_magic(path)

    def test_valid_protocol0_accepted(self, tmp_dir):
        path = os.path.join(tmp_dir, "p0.pkl")
        with open(path, "wb") as f:
            pickle.dump({"a": 1}, f, protocol=0)
        _validate_pickle_magic(path)

    def test_invalid_magic_rejected(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.pkl")
        with open(path, "wb") as f:
            f.write(b"\x00\x00garbage")
        with pytest.raises(ValueError, match="Invalid pickle format"):
            _validate_pickle_magic(path)

    def test_too_short_file_rejected(self, tmp_dir):
        path = os.path.join(tmp_dir, "tiny.pkl")
        with open(path, "wb") as f:
            f.write(b"\x80")
        with pytest.raises(ValueError, match="file too short"):
            _validate_pickle_magic(path)

    def test_invalid_proto_version_rejected(self, tmp_dir):
        path = os.path.join(tmp_dir, "badver.pkl")
        with open(path, "wb") as f:
            f.write(b"\x80\x09rest")  # proto opcode but version 9
        with pytest.raises(ValueError, match="unsupported version"):
            _validate_pickle_magic(path)

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "magickey"})
    def test_safe_load_rejects_non_pickle(self, pkl_path, sample_obj):
        """safe_load should fail fast on non-pickle before hash check."""
        safe_dump(sample_obj, pkl_path)
        # Overwrite with non-pickle content but keep sidecar
        with open(pkl_path, "wb") as f:
            f.write(b"\x00\x00not-a-pickle")
        with pytest.raises(ValueError, match="Invalid pickle format"):
            safe_load(pkl_path)
