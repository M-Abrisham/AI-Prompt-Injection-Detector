"""Tests for ScanResult and OutputScanResult serialization methods."""

import json

import pytest

from na0s.scan_result import ScanResult
from na0s.output_scanner import OutputScanResult


# ---------------------------------------------------------------------------
# ScanResult.to_dict()
# ---------------------------------------------------------------------------

class TestScanResultToDict:
    def test_returns_dict(self):
        r = ScanResult()
        assert isinstance(r.to_dict(), dict)

    def test_default_has_all_keys(self):
        r = ScanResult()
        d = r.to_dict()
        expected_keys = {
            "sanitized_text",
            "is_malicious",
            "risk_score",
            "label",
            "technique_tags",
            "rule_hits",
            "ml_confidence",
            "ml_label",
            "anomaly_flags",
            "rejected",
            "rejection_reason",
            "cascade_stage",
            "embedding_score",
            "model_version",
            "perplexity_score",
            "judge_reasoning",
            "output_scan_flags",
            "output_scan_risk",
            "canary_triggered",
            "canary_leaks",
            "elapsed_ms",
            # Layer 16 multi-turn fields
            "session_id",
            "escalation_detected",
            "multi_turn_alerts",
            "multi_turn_risk_trend",
        }
        assert set(d.keys()) == expected_keys

    def test_default_values(self):
        r = ScanResult()
        d = r.to_dict()
        assert d["sanitized_text"] == ""
        assert d["is_malicious"] is False
        assert d["risk_score"] == 0.0
        assert d["label"] == "safe"
        assert d["technique_tags"] == []
        assert d["rule_hits"] == []
        assert d["ml_confidence"] == 0.0
        assert d["ml_label"] == ""
        assert d["anomaly_flags"] == []
        assert d["rejected"] is False
        assert d["rejection_reason"] == ""
        assert d["cascade_stage"] == ""
        assert d["elapsed_ms"] == 0.0

    def test_populated_values(self):
        r = ScanResult(
            sanitized_text="cleaned",
            is_malicious=True,
            risk_score=0.95,
            label="malicious",
            technique_tags=["injection", "obfuscation"],
            rule_hits=["R001", "R002"],
            ml_confidence=0.88,
            ml_label="malicious",
            anomaly_flags=["unicode_abuse"],
            rejected=True,
            rejection_reason="injection detected",
            cascade_stage="weighted",
        )
        d = r.to_dict()
        assert d["sanitized_text"] == "cleaned"
        assert d["is_malicious"] is True
        assert d["risk_score"] == 0.95
        assert d["label"] == "malicious"
        assert d["technique_tags"] == ["injection", "obfuscation"]
        assert d["rule_hits"] == ["R001", "R002"]
        assert d["ml_confidence"] == 0.88
        assert d["ml_label"] == "malicious"
        assert d["anomaly_flags"] == ["unicode_abuse"]
        assert d["rejected"] is True
        assert d["rejection_reason"] == "injection detected"
        assert d["cascade_stage"] == "weighted"

    def test_values_are_json_compatible(self):
        r = ScanResult(
            technique_tags=["a", "b"],
            rule_hits=["R1"],
            anomaly_flags=["f1"],
        )
        d = r.to_dict()
        # All values should be JSON-serializable (no dataclass objects)
        for key, value in d.items():
            assert isinstance(value, (str, int, float, bool, list, dict, type(None))), (
                f"Field {key!r} has non-JSON-compatible type {type(value)}"
            )


# ---------------------------------------------------------------------------
# ScanResult.to_json()
# ---------------------------------------------------------------------------

class TestScanResultToJson:
    def test_returns_string(self):
        r = ScanResult()
        assert isinstance(r.to_json(), str)

    def test_valid_json(self):
        r = ScanResult()
        parsed = json.loads(r.to_json())
        assert isinstance(parsed, dict)

    def test_pretty_print(self):
        r = ScanResult()
        pretty = r.to_json(indent=2)
        assert "\n" in pretty
        # Indented JSON should contain leading spaces
        lines = pretty.split("\n")
        indented = [l for l in lines if l.startswith("  ")]
        assert len(indented) > 0

    def test_kwargs_passed_through(self):
        r = ScanResult()
        compact = r.to_json(separators=(",", ":"))
        assert ": " not in compact  # no space after colon
        assert ", " not in compact  # no space after comma

    def test_round_trip(self):
        r = ScanResult(
            sanitized_text="test",
            is_malicious=True,
            risk_score=0.75,
            label="malicious",
            technique_tags=["tag1"],
            rule_hits=["hit1"],
            ml_confidence=0.9,
            ml_label="malicious",
            anomaly_flags=["flag1"],
            rejected=False,
            rejection_reason="",
            cascade_stage="judge",
        )
        assert json.loads(r.to_json()) == r.to_dict()

    def test_round_trip_defaults(self):
        r = ScanResult()
        assert json.loads(r.to_json()) == r.to_dict()


# ---------------------------------------------------------------------------
# OutputScanResult.to_dict()
# ---------------------------------------------------------------------------

class TestOutputScanResultToDict:
    def test_returns_dict(self):
        r = OutputScanResult(is_suspicious=False, risk_score=0.0)
        assert isinstance(r.to_dict(), dict)

    def test_has_all_keys(self):
        r = OutputScanResult(is_suspicious=False, risk_score=0.0)
        d = r.to_dict()
        expected_keys = {"is_suspicious", "risk_score", "flags", "redacted_text", "technique_ids"}
        assert set(d.keys()) == expected_keys

    def test_populated_values(self):
        r = OutputScanResult(
            is_suspicious=True,
            risk_score=0.85,
            flags=["role_break", "secret_leak"],
            redacted_text="[REDACTED] some text",
        )
        d = r.to_dict()
        assert d["is_suspicious"] is True
        assert d["risk_score"] == 0.85
        assert d["flags"] == ["role_break", "secret_leak"]
        assert d["redacted_text"] == "[REDACTED] some text"

    def test_values_are_json_compatible(self):
        r = OutputScanResult(
            is_suspicious=True,
            risk_score=0.5,
            flags=["f1"],
            redacted_text="text",
        )
        d = r.to_dict()
        for key, value in d.items():
            assert isinstance(value, (str, int, float, bool, list, dict, type(None))), (
                f"Field {key!r} has non-JSON-compatible type {type(value)}"
            )


# ---------------------------------------------------------------------------
# OutputScanResult.to_json()
# ---------------------------------------------------------------------------

class TestOutputScanResultToJson:
    def test_returns_string(self):
        r = OutputScanResult(is_suspicious=False, risk_score=0.0)
        assert isinstance(r.to_json(), str)

    def test_valid_json(self):
        r = OutputScanResult(is_suspicious=True, risk_score=0.42, flags=["a"])
        parsed = json.loads(r.to_json())
        assert isinstance(parsed, dict)

    def test_pretty_print(self):
        r = OutputScanResult(is_suspicious=False, risk_score=0.0)
        pretty = r.to_json(indent=2)
        assert "\n" in pretty

    def test_round_trip(self):
        r = OutputScanResult(
            is_suspicious=True,
            risk_score=0.77,
            flags=["leak"],
            redacted_text="redacted",
        )
        assert json.loads(r.to_json()) == r.to_dict()
