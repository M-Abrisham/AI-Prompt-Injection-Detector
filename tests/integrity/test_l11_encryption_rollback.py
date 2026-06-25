"""Tests for Layer 11 supply-chain features: encryption, rollback, SBOM (~35 tests)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Encryption tests
# ---------------------------------------------------------------------------

crypto = pytest.importorskip("cryptography", reason="cryptography not installed")


class TestModelEncryptorInit:
    """Constructor / key handling."""

    def test_init_with_bytes_key(self):
        from na0s.model_encryption import ModelEncryptor

        key = os.urandom(32)
        enc = ModelEncryptor(key=key)
        assert enc._key == key

    def test_init_with_hex_key(self):
        from na0s.model_encryption import ModelEncryptor

        key = os.urandom(32)
        enc = ModelEncryptor(key=key.hex())
        assert enc._key == key

    def test_init_from_env_var(self, monkeypatch):
        from na0s.model_encryption import ModelEncryptor

        key = os.urandom(32)
        monkeypatch.setenv("NA0S_ENCRYPTION_KEY", key.hex())
        enc = ModelEncryptor()
        assert enc._key == key

    def test_init_no_key_raises(self, monkeypatch):
        from na0s.model_encryption import ModelEncryptor

        monkeypatch.delenv("NA0S_ENCRYPTION_KEY", raising=False)
        with pytest.raises(ValueError, match="No encryption key"):
            ModelEncryptor()

    def test_init_wrong_length_raises(self):
        from na0s.model_encryption import ModelEncryptor

        with pytest.raises(ValueError, match="32 bytes"):
            ModelEncryptor(key=b"tooshort")

    def test_init_wrong_hex_length_raises(self):
        from na0s.model_encryption import ModelEncryptor

        with pytest.raises(ValueError, match="32 bytes"):
            ModelEncryptor(key="aa" * 16)  # 16 bytes, not 32


class TestModelEncryptorBytes:
    """Byte-level encrypt / decrypt round-trip."""

    @pytest.fixture()
    def encryptor(self):
        from na0s.model_encryption import ModelEncryptor

        return ModelEncryptor(key=os.urandom(32))

    def test_round_trip(self, encryptor):
        plaintext = b"hello world"
        blob = encryptor.encrypt_bytes(plaintext)
        assert encryptor.decrypt_bytes(blob) == plaintext

    def test_ciphertext_layout(self, encryptor):
        blob = encryptor.encrypt_bytes(b"data")
        # nonce(12) + tag(16) + ciphertext(>=1)
        assert len(blob) >= 12 + 16 + 1

    def test_different_nonces(self, encryptor):
        a = encryptor.encrypt_bytes(b"same")
        b = encryptor.encrypt_bytes(b"same")
        # nonces (first 12 bytes) should differ
        assert a[:12] != b[:12]

    def test_decrypt_wrong_key_fails(self):
        from na0s.model_encryption import ModelEncryptor

        enc1 = ModelEncryptor(key=os.urandom(32))
        enc2 = ModelEncryptor(key=os.urandom(32))
        blob = enc1.encrypt_bytes(b"secret")
        with pytest.raises(Exception):
            enc2.decrypt_bytes(blob)

    def test_decrypt_truncated_raises(self, encryptor):
        with pytest.raises(ValueError, match="too short"):
            encryptor.decrypt_bytes(b"short")

    def test_empty_plaintext(self, encryptor):
        blob = encryptor.encrypt_bytes(b"")
        assert encryptor.decrypt_bytes(blob) == b""

    def test_large_payload(self, encryptor):
        data = os.urandom(1_000_000)
        assert encryptor.decrypt_bytes(encryptor.encrypt_bytes(data)) == data


class TestModelEncryptorFiles:
    """File-level encrypt / decrypt."""

    @pytest.fixture()
    def encryptor(self):
        from na0s.model_encryption import ModelEncryptor

        return ModelEncryptor(key=os.urandom(32))

    def test_encrypt_file_default_dst(self, tmp_path, encryptor):
        src = tmp_path / "model.pkl"
        src.write_bytes(b"model-data")
        dst = encryptor.encrypt_file(src)
        assert dst == src.with_suffix(".pkl.enc")
        assert dst.exists()

    def test_encrypt_file_custom_dst(self, tmp_path, encryptor):
        src = tmp_path / "model.pkl"
        src.write_bytes(b"model-data")
        custom = tmp_path / "encrypted_model"
        dst = encryptor.encrypt_file(src, custom)
        assert dst == custom
        assert dst.exists()

    def test_decrypt_file_strips_enc(self, tmp_path, encryptor):
        src = tmp_path / "model.pkl"
        src.write_bytes(b"model-data")
        enc_path = encryptor.encrypt_file(src)
        # Remove original
        src.unlink()
        dec_path = encryptor.decrypt_file(enc_path)
        assert dec_path == src
        assert dec_path.read_bytes() == b"model-data"

    def test_file_round_trip(self, tmp_path, encryptor):
        original = b"important model weights"
        src = tmp_path / "weights.bin"
        src.write_bytes(original)
        enc = encryptor.encrypt_file(src)
        src.unlink()
        dec = encryptor.decrypt_file(enc, src)
        assert dec.read_bytes() == original


class TestEncryptionImportGuard:
    """Module stays importable when cryptography is missing."""

    def test_importerror_when_crypto_missing(self, monkeypatch):
        from na0s import model_encryption

        monkeypatch.setattr(model_encryption, "_HAS_CRYPTO", False)
        with pytest.raises(ImportError, match="cryptography"):
            model_encryption.ModelEncryptor(key=os.urandom(32))


# ---------------------------------------------------------------------------
# Rollback tests
# ---------------------------------------------------------------------------


class TestModelRollbackBackup:
    """Backup operations."""

    def test_backup_creates_copy(self, tmp_path):
        from na0s.model_rollback import ModelRollback

        model = tmp_path / "model.pkl"
        model.write_bytes(b"weights")
        rb = ModelRollback(backup_dir=tmp_path / "backups")
        bpath = rb.backup(model)
        assert bpath.exists()
        assert bpath.read_bytes() == b"weights"

    def test_backup_copies_sidecars(self, tmp_path):
        from na0s.model_rollback import ModelRollback

        model = tmp_path / "model.pkl"
        model.write_bytes(b"w")
        sha_file = tmp_path / "model.pkl.sha256"
        sha_file.write_text("abc123")

        rb = ModelRollback(backup_dir=tmp_path / "backups")
        bpath = rb.backup(model)
        sidecar = bpath.parent / (bpath.name + ".sha256")
        assert sidecar.exists()
        assert sidecar.read_text() == "abc123"

    def test_backup_missing_model_raises(self, tmp_path):
        from na0s.model_rollback import ModelRollback

        rb = ModelRollback(backup_dir=tmp_path / "backups")
        with pytest.raises(FileNotFoundError):
            rb.backup(tmp_path / "nonexistent.pkl")


class TestModelRollbackList:
    """Listing and latest_backup."""

    def _make_backups(self, tmp_path, count=3):
        from na0s.model_rollback import ModelRollback

        model = tmp_path / "model.pkl"
        model.write_bytes(b"data")
        rb = ModelRollback(backup_dir=tmp_path / "backups")
        paths = []
        for i in range(count):
            # Manually create with distinct timestamps
            ts = f"2026-03-{10 + i:02d}T10-00-00"
            dst = rb.backup_dir / f"model.pkl.{ts}"
            dst.write_bytes(b"data" * (i + 1))
            paths.append(dst)
        return rb, paths

    def test_list_backups_sorted_newest_first(self, tmp_path):
        rb, _ = self._make_backups(tmp_path)
        entries = rb.list_backups("model.pkl")
        assert len(entries) == 3
        # Newest timestamp should be first
        assert entries[0]["timestamp"] > entries[-1]["timestamp"]

    def test_list_backups_has_required_keys(self, tmp_path):
        rb, _ = self._make_backups(tmp_path, count=1)
        entry = rb.list_backups("model.pkl")[0]
        assert "path" in entry
        assert "timestamp" in entry
        assert "size_bytes" in entry

    def test_list_backups_empty(self, tmp_path):
        from na0s.model_rollback import ModelRollback

        rb = ModelRollback(backup_dir=tmp_path / "backups")
        assert rb.list_backups("nope.pkl") == []

    def test_latest_backup(self, tmp_path):
        rb, paths = self._make_backups(tmp_path)
        latest = rb.latest_backup("model.pkl")
        assert latest == paths[-1]  # the one with highest date

    def test_latest_backup_none(self, tmp_path):
        from na0s.model_rollback import ModelRollback

        rb = ModelRollback(backup_dir=tmp_path / "backups")
        assert rb.latest_backup("nope.pkl") is None


class TestModelRollbackRestore:
    """Restore operations.

    S1.7: ``restore()`` now re-verifies the restored target via
    ``safe_pickle.verify_file_digest`` before returning, and FAILS CLOSED on any
    integrity error (tamper / missing sidecar / malformed sidecar). Backups must
    therefore be properly signed (``safe_dump``) to be restorable. The two tests
    below were INVERTED from their pre-S1.7 form: they previously asserted that an
    UNSIGNED backup (no sidecar) and a backup with a FORGED plain-text sidecar
    were installed verbatim — exactly the insecure "install whatever is in the
    backup dir" behavior this fix closes. They now assert fail-closed rejection.
    Happy-path round-trips live in ``TestModelRollbackRestoreVerify`` below.
    """

    def test_restore_unsigned_backup_rejected(self, tmp_path):
        # INVERTED (was test_restore_copies_file): a backup with NO integrity
        # sidecar is unverifiable, so restore must refuse rather than install it.
        from na0s.model_rollback import ModelRollback

        model = tmp_path / "model.pkl"
        model.write_bytes(b"original")
        rb = ModelRollback(backup_dir=tmp_path / "backups")
        bpath = rb.backup(model)

        target = tmp_path / "restored" / "model.pkl"
        with pytest.raises(ValueError, match="integrity verification"):
            rb.restore(bpath, target)
        # Fail closed: nothing left installed at the target.
        assert not target.exists()

    def test_restore_forged_sidecar_rejected(self, tmp_path):
        # INVERTED (was test_restore_copies_sidecars): a hand-written, non-conformant
        # sidecar ("hmac-value") is not a valid integrity record, so restore must
        # refuse — it must not blindly copy a forged sidecar + payload into place.
        from na0s.model_rollback import ModelRollback

        model = tmp_path / "model.pkl"
        model.write_bytes(b"w")
        hmac_file = tmp_path / "model.pkl.hmac"
        hmac_file.write_text("hmac-value")

        rb = ModelRollback(backup_dir=tmp_path / "backups")
        bpath = rb.backup(model)

        target = tmp_path / "restored" / "model.pkl"
        with pytest.raises(ValueError, match="integrity verification"):
            rb.restore(bpath, target)
        # Fail closed: neither the payload nor the forged sidecar is installed.
        assert not target.exists()
        assert not (target.parent / (target.name + ".hmac")).exists()

    def test_restore_missing_backup_raises(self, tmp_path):
        from na0s.model_rollback import ModelRollback

        rb = ModelRollback(backup_dir=tmp_path / "backups")
        with pytest.raises(FileNotFoundError):
            rb.restore(tmp_path / "ghost", tmp_path / "target")


class TestModelRollbackRestoreVerify:
    """S1.7: restore() re-verifies the backup and fails closed on tamper."""

    def test_restore_accepts_clean_backup(self, tmp_path):
        """A properly-signed backup (safe_dump) round-trips and verifies."""
        from na0s.integrity.safe_pickle import safe_dump, verify_file_digest
        from na0s.model_rollback import ModelRollback

        model = tmp_path / "model.pkl"
        # safe_dump writes the pickle + a conformant .sha256 sidecar (keyless).
        safe_dump({"weights": [1, 2, 3]}, str(model))
        rb = ModelRollback(backup_dir=tmp_path / "backups")
        bpath = rb.backup(model)

        target = tmp_path / "restored" / "model.pkl"
        out = rb.restore(bpath, target)

        assert out == target
        assert target.exists()
        assert target.read_bytes() == model.read_bytes()
        # Sidecar travelled too, and the installed target verifies clean.
        assert (target.parent / (target.name + ".sha256")).exists()
        verify_file_digest(str(target))  # no raise => verified

    def test_restore_rejects_tampered_backup(self, tmp_path):
        """A tampered BACKUP pickle is detected on re-verify; nothing installed."""
        from na0s.model_rollback import ModelRollback
        from na0s.integrity.safe_pickle import safe_dump

        model = tmp_path / "model.pkl"
        safe_dump({"weights": [1, 2, 3]}, str(model))
        rb = ModelRollback(backup_dir=tmp_path / "backups")
        bpath = rb.backup(model)

        # Corrupt the BACKUP pickle bytes (sidecar in the backup stays the
        # original digest) -> digest mismatch on the restored target.
        forged = b"\x80\x04\x95\x00tampered-evil-payload"
        bpath.write_bytes(forged)

        target = tmp_path / "restored" / "model.pkl"
        with pytest.raises(ValueError, match="integrity verification"):
            rb.restore(bpath, target)

        # Fail closed: the tampered bytes must NOT be installed at the target.
        assert not target.exists()
        # And no stray sidecar left behind from the aborted restore.
        assert not (target.parent / (target.name + ".sha256")).exists()

    def test_restore_rejects_both_files_replaced_under_hmac(self, tmp_path, monkeypatch):
        """Both-files-replace (forged pickle + forged .hmac sidecar) is blocked.

        Under an HMAC signing key, an attacker who replaces both the backup pickle
        and its .hmac sidecar still cannot forge a valid HMAC without the key, so
        re-verify rejects it. This is the strongest backup-dir threat.
        """
        import os as _os

        from na0s.integrity import safe_pickle
        from na0s.integrity.safe_pickle import safe_dump, _format_sidecar, _sha256
        from na0s.model_rollback import ModelRollback

        monkeypatch.setenv("NA0S_PICKLE_KEY", _os.urandom(32).hex())
        safe_pickle._reset_caches()

        model = tmp_path / "model.pkl"
        safe_dump({"weights": [1, 2, 3]}, str(model))  # writes .hmac sidecar
        rb = ModelRollback(backup_dir=tmp_path / "backups")
        bpath = rb.backup(model)

        # Attacker overwrites the backup pickle AND forges a .hmac sidecar using a
        # plain SHA-256 (they lack the signing key) in the v1 hmac-sha256 envelope.
        import pickle as _pickle

        bpath.write_bytes(_pickle.dumps({"evil": True}))
        forged_sidecar = bpath.parent / (bpath.name + ".hmac")
        forged_sidecar.write_text(_format_sidecar("hmac-sha256", _sha256(str(bpath))))
        safe_pickle._reset_caches()

        target = tmp_path / "restored" / "model.pkl"
        with pytest.raises(ValueError, match="integrity verification"):
            rb.restore(bpath, target)
        assert not target.exists()
        assert not (target.parent / (target.name + ".hmac")).exists()


class TestModelRollbackCleanup:
    """Cleanup old backups."""

    def test_cleanup_keeps_n(self, tmp_path):
        from na0s.model_rollback import ModelRollback

        model = tmp_path / "model.pkl"
        model.write_bytes(b"data")
        rb = ModelRollback(backup_dir=tmp_path / "backups")
        for i in range(5):
            ts = f"2026-03-{10 + i:02d}T10-00-00"
            (rb.backup_dir / f"model.pkl.{ts}").write_bytes(b"x")
        removed = rb.cleanup("model.pkl", keep=2)
        assert removed == 3
        assert len(rb.list_backups("model.pkl")) == 2

    def test_cleanup_nothing_to_remove(self, tmp_path):
        from na0s.model_rollback import ModelRollback

        rb = ModelRollback(backup_dir=tmp_path / "backups")
        assert rb.cleanup("model.pkl", keep=5) == 0


class TestAutoBackupGate:
    """Env-var gating."""

    def test_auto_disabled_by_default(self, monkeypatch):
        from na0s.model_rollback import auto_backup_enabled

        monkeypatch.delenv("NA0S_MODEL_ROLLBACK", raising=False)
        assert auto_backup_enabled() is False

    def test_auto_enabled(self, monkeypatch):
        from na0s.model_rollback import auto_backup_enabled

        monkeypatch.setenv("NA0S_MODEL_ROLLBACK", "1")
        assert auto_backup_enabled() is True


# ---------------------------------------------------------------------------
# SBOM tests
# ---------------------------------------------------------------------------


class TestSBOMGenerate:
    """SBOM generation."""

    def test_generate_has_required_keys(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        gen = SBOMGenerator(models_dir=tmp_path)
        sbom = gen.generate()
        assert sbom["format"] == "CycloneDX-lite"
        assert "timestamp" in sbom
        assert sbom["component"]["name"] == "na0s"
        assert "version" in sbom["component"]
        assert isinstance(sbom["dependencies"], list)
        assert isinstance(sbom["models"], list)
        assert "python_version" in sbom

    def test_generate_discovers_pkl_files(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        (tmp_path / "a.pkl").write_bytes(b"aaa")
        (tmp_path / "b.pkl").write_bytes(b"bbb")
        (tmp_path / "c.txt").write_bytes(b"not a model")
        gen = SBOMGenerator(models_dir=tmp_path)
        sbom = gen.generate()
        assert len(sbom["models"]) == 2
        names = {m["filename"] for m in sbom["models"]}
        assert names == {"a.pkl", "b.pkl"}

    def test_model_entry_has_sha256(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        data = b"model-weights"
        (tmp_path / "m.pkl").write_bytes(data)
        gen = SBOMGenerator(models_dir=tmp_path)
        entry = gen.generate()["models"][0]
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["size_bytes"] == len(data)

    def test_generate_empty_models_dir(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        gen = SBOMGenerator(models_dir=tmp_path)
        assert gen.generate()["models"] == []

    def test_generate_nonexistent_models_dir(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        gen = SBOMGenerator(models_dir=tmp_path / "nope")
        assert gen.generate()["models"] == []


class TestSBOMSaveLoad:
    """Save / load round-trip."""

    def test_save_creates_json(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        gen = SBOMGenerator(models_dir=tmp_path)
        out = gen.save(tmp_path / "sbom.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["format"] == "CycloneDX-lite"

    def test_load_returns_dict(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        gen = SBOMGenerator(models_dir=tmp_path)
        gen.save(tmp_path / "sbom.json")
        loaded = SBOMGenerator.load(tmp_path / "sbom.json")
        assert isinstance(loaded, dict)
        assert loaded["format"] == "CycloneDX-lite"

    def test_save_load_round_trip(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        (tmp_path / "x.pkl").write_bytes(b"x")
        gen = SBOMGenerator(models_dir=tmp_path)
        gen.save(tmp_path / "sbom.json")
        loaded = SBOMGenerator.load(tmp_path / "sbom.json")
        assert loaded["models"][0]["filename"] == "x.pkl"


class TestSBOMVerifyModels:
    """Model hash verification."""

    def test_verify_matching(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        (tmp_path / "m.pkl").write_bytes(b"data")
        gen = SBOMGenerator(models_dir=tmp_path)
        sbom = gen.generate()
        results = gen.verify_models(sbom)
        assert len(results) == 1
        assert results[0]["match"] is True

    def test_verify_tampered(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        (tmp_path / "m.pkl").write_bytes(b"original")
        gen = SBOMGenerator(models_dir=tmp_path)
        sbom = gen.generate()
        # Tamper
        (tmp_path / "m.pkl").write_bytes(b"tampered")
        results = gen.verify_models(sbom)
        assert results[0]["match"] is False

    def test_verify_missing_file(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        (tmp_path / "m.pkl").write_bytes(b"data")
        gen = SBOMGenerator(models_dir=tmp_path)
        sbom = gen.generate()
        (tmp_path / "m.pkl").unlink()
        results = gen.verify_models(sbom)
        assert results[0]["match"] is False
        assert results[0]["actual"] is None

    def test_verify_empty_sbom(self, tmp_path):
        from na0s.sbom import SBOMGenerator

        gen = SBOMGenerator(models_dir=tmp_path)
        assert gen.verify_models({"models": []}) == []
