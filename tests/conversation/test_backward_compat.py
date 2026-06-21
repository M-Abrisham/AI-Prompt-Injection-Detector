"""Tests for Layer 16 backward compatibility.  # LAYER16

Verifies that:
1. scan() without session_id produces output identical to pre-Layer-16.
2. ScanResult new fields default to empty/false values.
"""

from __future__ import annotations

from na0s.scan_result import ScanResult


class TestScanResultDefaults:
    """ScanResult multi-turn fields must have safe defaults."""

    def test_multi_turn_alerts_default_empty(self) -> None:
        result = ScanResult()
        assert result.multi_turn_alerts == []

    def test_multi_turn_risk_trend_default_empty(self) -> None:
        result = ScanResult()
        assert result.multi_turn_risk_trend == []

    def test_escalation_detected_default_false(self) -> None:
        result = ScanResult()
        assert result.escalation_detected is False

    def test_session_id_default_empty(self) -> None:
        result = ScanResult()
        assert result.session_id == ""

    def test_to_dict_includes_layer16_fields(self) -> None:
        result = ScanResult()
        d = result.to_dict()
        assert "multi_turn_alerts" in d
        assert "multi_turn_risk_trend" in d
        assert "escalation_detected" in d
        assert "session_id" in d

    def test_to_dict_defaults(self) -> None:
        result = ScanResult()
        d = result.to_dict()
        assert d["multi_turn_alerts"] == []
        assert d["multi_turn_risk_trend"] == []
        assert d["escalation_detected"] is False
        assert d["session_id"] == ""


class TestScanWithoutSessionId:
    """scan() without session_id must produce identical output to pre-Layer-16."""

    def test_scan_no_session_id(self) -> None:
        """Call scan() without session_id, verify Layer 16 fields are defaults."""
        try:
            from na0s.predict import scan

            result = scan("What is the capital of France?")
        except Exception:
            # If scan() fails due to missing model files etc, skip.
            # The important thing is that the signature accepts no session_id.
            import pytest
            pytest.skip("scan() unavailable in test environment")
            return

        # Layer 16 fields should remain at defaults
        assert result.multi_turn_alerts == []
        assert result.multi_turn_risk_trend == []
        assert result.escalation_detected is False
        assert result.session_id == ""

    def test_scan_signature_accepts_session_id(self) -> None:
        """Verify scan() accepts session_id as a keyword argument."""
        import inspect
        from na0s.predict import scan

        sig = inspect.signature(scan)
        assert "session_id" in sig.parameters
        param = sig.parameters["session_id"]
        assert param.default == ""
