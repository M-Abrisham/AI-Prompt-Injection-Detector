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
    _parse_sidecar_typed,
    _read_sidecar,
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
        # Fixture corrected (item #09): the digest body must be a real 64-char
        # hex SHA-256 digest. The previous 16-hex literal ("abcdef0123456789")
        # is now correctly rejected by the 64-hex shape guard; this asserts the
        # SAME behavior (the v1: digest body is returned) with a valid digest.
        digest = "ab" * 32  # 64 hex chars
        assert _parse_sidecar("v1:sha256:" + digest) == digest

    def test_parse_legacy_bare_hex(self):
        bare = "a" * 64
        assert _parse_sidecar(bare) == bare

    def test_parse_versioned_with_whitespace(self):
        # Fixture corrected (item #09): "deadbeef" (8 hex) is now rejected by the
        # shape guard; use a valid 64-hex body. Asserts the SAME behavior — the
        # surrounding whitespace is stripped and the digest body returned.
        digest = "de" * 32  # 64 hex chars
        assert _parse_sidecar("  v1:hmac-sha256:" + digest + "\n") == digest

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "verkey-strong-32char-secret-aaaa"})
    def test_safe_dump_writes_versioned_hmac_sidecar(self, pkl_path, sample_obj):
        # Key lengthened (item #09): "verkey" (6 chars) is now hard-rejected by
        # the _MIN_PICKLE_KEY_LEN floor; a >= 32-char key is accepted silently.
        # Assertion (sidecar format) is unchanged.
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

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "verkey2-strong-32char-secret-aaa"})
    def test_roundtrip_versioned_sidecar(self, pkl_path, sample_obj):
        """Dump with versioned format, load should still work."""
        # Key lengthened (item #09): "verkey2" (7 chars) is now hard-rejected by
        # the _MIN_PICKLE_KEY_LEN floor; a >= 32-char key is accepted silently.
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
# Item #09: digest shape validation in _parse_sidecar / _parse_sidecar_typed
# ---------------------------------------------------------------------------

# A canonical 64-char lowercase-hex digest (the exact shape SHA-256 /
# HMAC-SHA256 hexdigest() emits). Built so the literal length is self-evident.
_VALID_DIGEST = "0123456789abcdef" * 4  # 64 hex chars


class TestSidecarDigestValidation:
    """Item #09 — ``_parse_sidecar`` / ``_parse_sidecar_typed`` now reject any
    value that is not a 64-char hex SHA-256/HMAC digest, so a corrupt sidecar
    fails fast with an accurate 'malformed integrity sidecar' error instead of a
    deferred, misleading compare-mismatch."""

    # --- E1/E2/E3: accept paths (no FP for legitimate sidecars) ---

    def test_parse_valid_versioned_sha256(self):
        assert _parse_sidecar("v1:sha256:" + _VALID_DIGEST) == _VALID_DIGEST

    def test_parse_valid_versioned_hmac(self):
        assert _parse_sidecar("v1:hmac-sha256:" + _VALID_DIGEST) == _VALID_DIGEST

    def test_parse_valid_legacy_bare(self):
        # Backward compat: a bare 64-hex digest (no v1: prefix) must NOT raise.
        bare = "a" * 64
        assert _parse_sidecar(bare) == bare

    def test_parse_uppercase_normalized(self):
        # External tooling may emit uppercase; accept and normalise to lowercase
        # (compare_digest is byte-exact against lowercase hexdigest()).
        assert _parse_sidecar("A" * 64) == "a" * 64
        assert _parse_sidecar("v1:sha256:" + "DEADBEEF" * 8) == "deadbeef" * 8

    # --- E4-E7: reject paths (must fail loud, not silently mis-parse) ---

    def test_parse_empty_digest_raises(self):
        with pytest.raises(ValueError, match="malformed integrity sidecar"):
            _parse_sidecar("v1:sha256:")

    def test_parse_short_hex_raises(self):
        with pytest.raises(ValueError, match="64-char hex"):
            _parse_sidecar("abcd")
        with pytest.raises(ValueError, match="64-char hex"):
            _parse_sidecar("a" * 63)

    def test_parse_long_hex_raises(self):
        with pytest.raises(ValueError, match="64-char hex"):
            _parse_sidecar("a" * 65)

    def test_parse_nonhex_raises(self):
        with pytest.raises(ValueError, match="64-char hex"):
            _parse_sidecar("z" * 64)
        # 63 hex + a trailing space is exactly 64 chars but not all-hex.
        with pytest.raises(ValueError, match="64-char hex"):
            _parse_sidecar("a" * 63 + " ")

    def test_parse_v1_two_parts_raises(self):
        # Regression for the old silent ``return "v1:sha256"`` fall-through: a
        # v1: header with < 3 colon-parts now raises, naming the missing body.
        with pytest.raises(ValueError, match="without algo:digest body"):
            _parse_sidecar("v1:sha256")

    def test_parse_whitespace_preserved(self):
        # Surrounding whitespace is still stripped (existing strip() behavior);
        # a valid 64-hex body inside whitespace parses to the bare digest.
        assert _parse_sidecar("  " + _VALID_DIGEST + "\n") == _VALID_DIGEST

    # --- _parse_sidecar_typed validates BOTH return paths (item #09) ---

    def test_typed_valid_versioned_returns_algo_and_digest(self):
        assert _parse_sidecar_typed(
            "v1:hmac-sha256:" + _VALID_DIGEST
        ) == ("hmac-sha256", _VALID_DIGEST)

    def test_typed_valid_legacy_returns_none_algo(self):
        assert _parse_sidecar_typed(_VALID_DIGEST) == (None, _VALID_DIGEST)

    def test_typed_versioned_bad_digest_raises(self):
        # The versioned return path is validated (was returned verbatim).
        with pytest.raises(ValueError, match="64-char hex"):
            _parse_sidecar_typed("v1:sha256:zzzz")

    def test_typed_legacy_bad_digest_raises(self):
        # The legacy bare-hex return path is validated too (both paths).
        with pytest.raises(ValueError, match="64-char hex"):
            _parse_sidecar_typed("not-a-digest")

    def test_typed_v1_two_parts_raises(self):
        with pytest.raises(ValueError, match="without algo:digest body"):
            _parse_sidecar_typed("v1:sha256")

    # --- _read_sidecar names the real path + bounds the read (item #09) ---

    def test_read_sidecar_names_real_path_on_malformed(self, tmp_dir):
        sidecar = os.path.join(tmp_dir, "model.pkl.sha256")
        with open(sidecar, "w") as f:
            f.write("v1:sha256:zzzz")
        with pytest.raises(ValueError) as exc:
            _read_sidecar(sidecar, "sidecar_sha256")
        # The error names the concrete sidecar path so the operator knows which
        # file is corrupt (not a generic placeholder).
        assert sidecar in str(exc.value)
        assert "malformed integrity sidecar" in str(exc.value)

    def test_read_sidecar_bounds_read(self, tmp_dir):
        # A multi-megabyte sidecar is not slurped whole: only the first 256
        # bytes are read, so a giant blob is rejected by the shape guard without
        # exhausting memory. The 256-byte prefix here is non-hex, so it raises.
        sidecar = os.path.join(tmp_dir, "model.pkl.sha256")
        with open(sidecar, "w") as f:
            f.write("z" * (5 * 1024 * 1024))  # 5 MiB of junk
        with pytest.raises(ValueError, match="malformed integrity sidecar"):
            _read_sidecar(sidecar, "sidecar_sha256")

    def test_read_sidecar_valid_legacy_bare_still_reads(self, tmp_dir):
        # Regression: a legitimate bare-hex sidecar still reads correctly.
        sidecar = os.path.join(tmp_dir, "model.pkl.sha256")
        with open(sidecar, "w") as f:
            f.write(_VALID_DIGEST)
        assert _read_sidecar(sidecar, "sidecar_sha256") == _VALID_DIGEST


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
