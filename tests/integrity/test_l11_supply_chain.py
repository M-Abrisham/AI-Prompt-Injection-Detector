"""Tests for Layer 11: Supply Chain Integrity features.

Covers: ModelProvenance, DependencyScanner, RequirementsIntegrity,
        FingerprintStoreIntegrity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_model(tmp_path):
    """Create a temporary 'model' file."""
    p = tmp_path / "model.pkl"
    p.write_bytes(b"fake-model-bytes-1234")
    return p


@pytest.fixture()
def tmp_req(tmp_path):
    """Create a temporary requirements.txt."""
    p = tmp_path / "requirements.txt"
    p.write_text("numpy==1.24.0\nscikit-learn>=1.2\nrequests\npandas==2.0.1\n")
    return p


@pytest.fixture()
def tmp_db(tmp_path):
    """Create a temporary fingerprint database file."""
    p = tmp_path / "fingerprints.db"
    p.write_bytes(b"sqlite-like-content-here")
    return p


@pytest.fixture(autouse=True)
def _enable_provenance(monkeypatch):
    """Enable provenance + dep-scan env vars for all tests."""
    monkeypatch.setenv("NA0S_MODEL_PROVENANCE", "1")
    monkeypatch.setenv("NA0S_DEP_SCAN", "1")


# ===================================================================
# ModelProvenance tests
# ===================================================================

class TestModelProvenance:
    def test_create_fills_hash(self, tmp_model):
        from na0s.model_provenance import ModelProvenance

        prov = ModelProvenance.create(tmp_model, framework="scikit-learn")
        expected = hashlib.sha256(tmp_model.read_bytes()).hexdigest()
        assert prov.sha256 == expected

    def test_create_fills_date(self, tmp_model):
        from na0s.model_provenance import ModelProvenance

        prov = ModelProvenance.create(tmp_model)
        assert prov.training_date  # non-empty ISO string

    def test_to_dict_roundtrip(self, tmp_model):
        from na0s.model_provenance import ModelProvenance

        prov = ModelProvenance.create(tmp_model, accuracy=0.95, sample_count=1000)
        d = prov.to_dict()
        prov2 = ModelProvenance.from_dict(d)
        assert prov2.accuracy == 0.95
        assert prov2.sample_count == 1000
        assert prov2.sha256 == prov.sha256

    def test_save_and_load(self, tmp_model):
        from na0s.model_provenance import ModelProvenance

        prov = ModelProvenance.create(tmp_model, feature_count=42)
        meta = prov.save(tmp_model)
        assert meta.exists()
        loaded = ModelProvenance.load(tmp_model)
        assert loaded.feature_count == 42
        assert loaded.sha256 == prov.sha256

    def test_verify_ok(self, tmp_model):
        from na0s.model_provenance import ModelProvenance

        prov = ModelProvenance.create(tmp_model)
        assert prov.verify(tmp_model) is True

    def test_verify_tampered(self, tmp_model):
        from na0s.model_provenance import ModelProvenance

        prov = ModelProvenance.create(tmp_model)
        tmp_model.write_bytes(b"tampered!")
        assert prov.verify(tmp_model) is False

    def test_disabled_raises(self, tmp_model, monkeypatch):
        from na0s.model_provenance import ModelProvenance

        monkeypatch.setenv("NA0S_MODEL_PROVENANCE", "0")
        prov = ModelProvenance.create(tmp_model)
        with pytest.raises(RuntimeError):
            prov.save(tmp_model)

    def test_load_disabled_raises(self, tmp_model, monkeypatch):
        from na0s.model_provenance import ModelProvenance

        prov = ModelProvenance.create(tmp_model)
        prov.save(tmp_model)
        monkeypatch.setenv("NA0S_MODEL_PROVENANCE", "0")
        with pytest.raises(RuntimeError):
            ModelProvenance.load(tmp_model)

    def test_verify_disabled_raises(self, tmp_model, monkeypatch):
        from na0s.model_provenance import ModelProvenance

        prov = ModelProvenance.create(tmp_model)
        monkeypatch.setenv("NA0S_MODEL_PROVENANCE", "0")
        with pytest.raises(RuntimeError):
            prov.verify(tmp_model)

    def test_from_dict_ignores_extra_keys(self):
        from na0s.model_provenance import ModelProvenance

        d = {"model_path": "/a", "sha256": "abc", "unknown_key": 999}
        prov = ModelProvenance.from_dict(d)
        assert prov.model_path == "/a"
        assert not hasattr(prov, "unknown_key")


# ===================================================================
# DependencyScanner tests
# ===================================================================

class TestDependencyScanner:
    def test_scan_installed_mocked(self):
        from na0s.dep_scanner import DependencyScanner

        fake_output = json.dumps([
            {"name": "numpy", "version": "1.24.0"},
            {"name": "requests", "version": "2.31.0"},
        ])
        scanner = DependencyScanner()
        with mock.patch("subprocess.run") as m:
            m.return_value = mock.Mock(returncode=0, stdout=fake_output)
            result = scanner.scan_installed()
        assert len(result) == 2
        assert result[0]["name"] == "numpy"

    def test_scan_installed_error(self):
        from na0s.dep_scanner import DependencyScanner

        scanner = DependencyScanner()
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            result = scanner.scan_installed()
        assert result == []

    def test_scan_installed_bad_returncode(self):
        from na0s.dep_scanner import DependencyScanner

        scanner = DependencyScanner()
        with mock.patch("subprocess.run") as m:
            m.return_value = mock.Mock(returncode=1, stdout="")
            result = scanner.scan_installed()
        assert result == []

    def test_check_requirements(self, tmp_req):
        from na0s.dep_scanner import DependencyScanner

        installed = [
            {"name": "numpy", "version": "1.24.0"},
            {"name": "scikit-learn", "version": "1.2.1"},
            {"name": "requests", "version": "2.31.0"},
            {"name": "pandas", "version": "2.0.0"},  # mismatch
        ]
        scanner = DependencyScanner()
        checks = scanner.check_requirements(tmp_req, installed=installed)
        assert len(checks) == 4
        numpy_check = next(c for c in checks if c["name"] == "numpy")
        assert numpy_check["matches"] is True
        pandas_check = next(c for c in checks if c["name"] == "pandas")
        assert pandas_check["matches"] is False

    def test_find_unpinned(self, tmp_req):
        from na0s.dep_scanner import DependencyScanner

        scanner = DependencyScanner()
        unpinned = scanner.find_unpinned(tmp_req)
        assert "scikit-learn" in unpinned
        assert "requests" in unpinned
        assert "numpy" not in unpinned

    def test_audit_report(self, tmp_req):
        from na0s.dep_scanner import DependencyScanner

        installed = [
            {"name": "numpy", "version": "1.24.0"},
            {"name": "pandas", "version": "2.0.1"},
        ]
        scanner = DependencyScanner()
        report = scanner.audit_report(tmp_req, installed=installed)
        assert report["total"] == 4
        assert report["pinned"] == 2
        assert report["unpinned"] == 2
        assert report["mismatched"] == 0

    def test_disabled_raises(self, monkeypatch):
        from na0s.dep_scanner import DependencyScanner

        monkeypatch.setenv("NA0S_DEP_SCAN", "0")
        scanner = DependencyScanner()
        with pytest.raises(RuntimeError):
            scanner.scan_installed()

    def test_find_unpinned_disabled(self, tmp_req, monkeypatch):
        from na0s.dep_scanner import DependencyScanner

        monkeypatch.setenv("NA0S_DEP_SCAN", "0")
        scanner = DependencyScanner()
        with pytest.raises(RuntimeError):
            scanner.find_unpinned(tmp_req)

    def test_comments_and_blanks_skipped(self, tmp_path):
        from na0s.dep_scanner import DependencyScanner

        req = tmp_path / "req.txt"
        req.write_text("# comment\n\nnumpy==1.0\n-e .\n")
        scanner = DependencyScanner()
        unpinned = scanner.find_unpinned(req)
        assert unpinned == []


# ===================================================================
# RequirementsIntegrity tests
# ===================================================================

class TestRequirementsIntegrity:
    def test_compute_hash(self, tmp_req):
        from na0s.req_integrity import RequirementsIntegrity

        h = RequirementsIntegrity.compute_hash(tmp_req)
        assert len(h) == 64  # sha256 hex

    def test_save_and_verify(self, tmp_req):
        from na0s.req_integrity import RequirementsIntegrity

        sidecar = RequirementsIntegrity.save_hash(tmp_req)
        assert sidecar.exists()
        assert RequirementsIntegrity.verify_hash(tmp_req) is True

    def test_verify_no_sidecar(self, tmp_req):
        from na0s.req_integrity import RequirementsIntegrity

        assert RequirementsIntegrity.verify_hash(tmp_req) is False

    def test_verify_after_change(self, tmp_req):
        from na0s.req_integrity import RequirementsIntegrity

        RequirementsIntegrity.save_hash(tmp_req)
        tmp_req.write_text("flask==3.0\n")
        assert RequirementsIntegrity.verify_hash(tmp_req) is False

    def test_detect_changes_no_sidecar(self, tmp_req):
        from na0s.req_integrity import RequirementsIntegrity

        result = RequirementsIntegrity.detect_changes(tmp_req)
        assert result["changed"] is True
        assert result["expected"] == ""

    def test_detect_changes_unchanged(self, tmp_req):
        from na0s.req_integrity import RequirementsIntegrity

        RequirementsIntegrity.save_hash(tmp_req)
        result = RequirementsIntegrity.detect_changes(tmp_req)
        assert result["changed"] is False
        assert result["expected"] == result["actual"]

    def test_detect_changes_modified(self, tmp_req):
        from na0s.req_integrity import RequirementsIntegrity

        RequirementsIntegrity.save_hash(tmp_req)
        tmp_req.write_text("changed\n")
        result = RequirementsIntegrity.detect_changes(tmp_req)
        assert result["changed"] is True
        assert result["expected"] != result["actual"]


# ===================================================================
# FingerprintStoreIntegrity tests
# ===================================================================

class TestFingerprintStoreIntegrity:
    def test_compute_hash(self, tmp_db):
        from na0s.fingerprint_integrity import FingerprintStoreIntegrity

        h = FingerprintStoreIntegrity.compute_hash(tmp_db)
        expected = hashlib.sha256(tmp_db.read_bytes()).hexdigest()
        assert h == expected

    def test_save_and_verify(self, tmp_db):
        from na0s.fingerprint_integrity import FingerprintStoreIntegrity

        sidecar = FingerprintStoreIntegrity.save_hash(tmp_db)
        assert sidecar.exists()
        assert FingerprintStoreIntegrity.verify_hash(tmp_db) is True

    def test_verify_no_sidecar(self, tmp_db):
        from na0s.fingerprint_integrity import FingerprintStoreIntegrity

        assert FingerprintStoreIntegrity.verify_hash(tmp_db) is False

    def test_verify_tampered(self, tmp_db):
        from na0s.fingerprint_integrity import FingerprintStoreIntegrity

        FingerprintStoreIntegrity.save_hash(tmp_db)
        tmp_db.write_bytes(b"corrupted-db")
        assert FingerprintStoreIntegrity.verify_hash(tmp_db) is False

    def test_monitor_basic(self, tmp_db):
        from na0s.fingerprint_integrity import FingerprintStoreIntegrity

        info = FingerprintStoreIntegrity.monitor(tmp_db)
        assert info["size"] > 0
        assert info["mtime"] > 0
        assert len(info["hash"]) == 64
        assert info["sidecar_valid"] is False

    def test_monitor_with_sidecar(self, tmp_db):
        from na0s.fingerprint_integrity import FingerprintStoreIntegrity

        FingerprintStoreIntegrity.save_hash(tmp_db)
        info = FingerprintStoreIntegrity.monitor(tmp_db)
        assert info["sidecar_valid"] is True

    def test_monitor_interval_check_flag(self, tmp_db):
        from na0s.fingerprint_integrity import FingerprintStoreIntegrity

        # interval_check is reserved; should not error
        info = FingerprintStoreIntegrity.monitor(tmp_db, interval_check=True)
        assert "hash" in info

    def test_monitor_after_tamper(self, tmp_db):
        from na0s.fingerprint_integrity import FingerprintStoreIntegrity

        FingerprintStoreIntegrity.save_hash(tmp_db)
        tmp_db.write_bytes(b"new-content")
        info = FingerprintStoreIntegrity.monitor(tmp_db)
        assert info["sidecar_valid"] is False


class TestModelProvenanceSaveNoPath:
    def test_save_uses_model_path_attr(self, tmp_model):
        from na0s.model_provenance import ModelProvenance

        prov = ModelProvenance.create(tmp_model)
        meta = prov.save()  # no explicit path arg
        assert meta.exists()
        assert json.loads(meta.read_text())["sha256"] == prov.sha256
