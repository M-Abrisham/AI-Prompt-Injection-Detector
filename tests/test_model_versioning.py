"""Tests for Layer 4 model versioning in ScanResult."""

import json
from unittest.mock import patch

from na0s.scan_result import ScanResult
from na0s.predict import _get_model_version
from na0s.models import KNOWN_HASHES


class TestScanResultModelVersion:
    """ScanResult has a model_version field with correct defaults."""

    def test_model_version_field_exists(self):
        result = ScanResult()
        assert hasattr(result, "model_version")

    def test_model_version_default_empty(self):
        result = ScanResult()
        assert result.model_version == ""

    def test_model_version_can_be_set(self):
        result = ScanResult(model_version="db28b5c8")
        assert result.model_version == "db28b5c8"

    def test_model_version_in_to_dict(self):
        result = ScanResult(model_version="db28b5c8")
        d = result.to_dict()
        assert "model_version" in d
        assert d["model_version"] == "db28b5c8"

    def test_model_version_in_to_json(self):
        result = ScanResult(model_version="db28b5c8")
        j = result.to_json()
        parsed = json.loads(j)
        assert "model_version" in parsed
        assert parsed["model_version"] == "db28b5c8"


class TestGetModelVersion:
    """_get_model_version() returns first 8 chars of model.pkl hash."""

    def test_returns_hash_prefix(self):
        version = _get_model_version()
        expected_hash = KNOWN_HASHES.get("model.pkl", "")
        if expected_hash:
            assert version == expected_hash[:8]
            assert len(version) == 8
        else:
            assert version == ""

    def test_returns_known_prefix(self):
        # The current model.pkl hash starts with "db28b5c8"
        version = _get_model_version()
        assert version == "db28b5c8"

    def test_fallback_empty_when_no_hash(self):
        with patch("na0s.predict.KNOWN_HASHES", {}):
            version = _get_model_version()
            assert version == ""

    def test_fallback_empty_when_model_key_missing(self):
        with patch("na0s.predict.KNOWN_HASHES", {"other.pkl": "abc123"}):
            version = _get_model_version()
            assert version == ""
