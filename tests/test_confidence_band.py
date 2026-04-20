"""Confidence band derivation from risk_score.

The confidence_band field absorbs ~0.05 floating-point drift between hardware
platforms (Mac NEON vs Linux AVX2). is_malicious stays tied to the binary
DECISION_THRESHOLD for backward compat; confidence_band is informational.
"""

import pytest

from na0s.config import T_LOW, T_HIGH, classify_band
from na0s.scan_result import ScanResult


class TestClassifyBand:
    def test_well_below_tlow_is_safe(self):
        assert classify_band(0.0) == "safe"
        assert classify_band(0.2) == "safe"
        assert classify_band(T_LOW - 0.01) == "safe"

    def test_tlow_boundary_enters_uncertain(self):
        assert classify_band(T_LOW) == "uncertain"

    def test_mid_band_is_uncertain(self):
        assert classify_band(0.55) == "uncertain"

    def test_just_below_thigh_is_uncertain(self):
        assert classify_band(T_HIGH - 0.01) == "uncertain"

    def test_thigh_boundary_enters_malicious(self):
        assert classify_band(T_HIGH) == "malicious"

    def test_well_above_thigh_is_malicious(self):
        assert classify_band(0.9) == "malicious"
        assert classify_band(1.0) == "malicious"


class TestScanResultPostInit:
    def test_default_risk_is_safe_band(self):
        r = ScanResult()
        assert r.confidence_band == "safe"

    def test_high_risk_is_malicious_band(self):
        r = ScanResult(risk_score=0.9, is_malicious=True, label="malicious")
        assert r.confidence_band == "malicious"

    def test_mid_risk_is_uncertain_band(self):
        r = ScanResult(risk_score=0.55, is_malicious=True, label="malicious")
        assert r.confidence_band == "uncertain"

    def test_band_independent_of_is_malicious(self):
        # is_malicious uses the old 0.55 threshold; band uses T_HIGH=0.65.
        # At risk=0.60 they disagree — that's the point of the band.
        r = ScanResult(risk_score=0.60, is_malicious=True, label="malicious")
        assert r.is_malicious is True
        assert r.confidence_band == "uncertain"

    def test_to_dict_includes_confidence_band(self):
        r = ScanResult(risk_score=0.7)
        d = r.to_dict()
        assert "confidence_band" in d
        assert d["confidence_band"] == "malicious"

    def test_to_json_includes_confidence_band(self):
        import json as _json
        r = ScanResult(risk_score=0.5)
        parsed = _json.loads(r.to_json())
        assert parsed["confidence_band"] == "uncertain"


class TestHardwareDriftAbsorption:
    """The whole point: same input at risk=0.54 (Linux) vs 0.56 (Mac) must
    produce the same band. Both fall in [T_LOW, T_HIGH) -> 'uncertain'."""

    def test_drift_both_sides_of_old_threshold_same_band(self):
        linux_risk = 0.54
        mac_risk = 0.56
        assert classify_band(linux_risk) == classify_band(mac_risk) == "uncertain"

    def test_drift_well_above_threshold_stable(self):
        linux_risk = 0.82
        mac_risk = 0.85
        assert classify_band(linux_risk) == classify_band(mac_risk) == "malicious"

    def test_drift_well_below_threshold_stable(self):
        linux_risk = 0.10
        mac_risk = 0.12
        assert classify_band(linux_risk) == classify_band(mac_risk) == "safe"
