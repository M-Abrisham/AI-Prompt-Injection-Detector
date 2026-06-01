"""Tests for GateAnalyzer agent."""

import json
import tempfile
from pathlib import Path
import pytest
from datetime import datetime

from na0s.agents.gate_analyzer import GateAnalyzer


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "canary").mkdir()
        (data_dir / "models").mkdir()
        yield str(data_dir)


def test_gate_analyzer_canary_pass(temp_data_dir):
    """Test canary gate analysis when gates pass."""
    data_dir = Path(temp_data_dir)

    # Write mock canary results
    canary_results = {
        "passed": True,
        "metrics": {
            "tpr": 0.97,
            "tnr": 0.93,
            "fpr": 0.02,
            "accuracy": 0.95,
        },
        "errors": [],
    }
    with open(data_dir / "canary" / "canary_results.json", "w") as f:
        json.dump(canary_results, f)

    analyzer = GateAnalyzer(data_dir=str(data_dir))
    result = analyzer.check_canary()

    assert result is not None
    assert result["passed"] is True
    assert result["verdict"] == "PASSED"
    assert result["tpr"] == 0.97
    assert result["tnr"] == 0.93


def test_gate_analyzer_canary_fail(temp_data_dir):
    """Test canary gate analysis when gates fail."""
    data_dir = Path(temp_data_dir)

    # Write mock failed canary results
    canary_results = {
        "passed": False,
        "metrics": {
            "tpr": 0.92,  # Below 95% threshold
            "tnr": 0.91,
            "fpr": 0.05,
        },
        "errors": [
            {
                "technique": "DAN",
                "text_preview": "Act as a malicious prompt",
            },
            {
                "technique": "Token smuggling",
                "text_preview": "[SECRET]",
            },
        ],
    }
    with open(data_dir / "canary" / "canary_results.json", "w") as f:
        json.dump(canary_results, f)

    analyzer = GateAnalyzer(data_dir=str(data_dir))
    result = analyzer.check_canary()

    assert result is not None
    assert result["passed"] is False
    assert result["verdict"] == "FAILED"
    assert result["error_count"] == 2


def test_gate_analyzer_diagnose_all_passed(temp_data_dir):
    """Test diagnosis when all gates pass."""
    data_dir = Path(temp_data_dir)

    # Write passing results for all gates
    canary = {"passed": True, "metrics": {"tpr": 0.97, "tnr": 0.93}}
    shadow = {"verdict": "PASS", "gates": []}
    f14 = {"verdict": "PASS", "overall": {"tpr": 0.85}}

    with open(data_dir / "canary" / "canary_results.json", "w") as f:
        json.dump(canary, f)
    with open(data_dir / "models" / "shadow_results.json", "w") as f:
        json.dump(shadow, f)
    with open(data_dir / "models" / "f14_gate_results.json", "w") as f:
        json.dump(f14, f)

    analyzer = GateAnalyzer(data_dir=str(data_dir))
    results = analyzer.diagnose_failures()

    assert results["overall_verdict"] == "ALL_PASSED"
    assert "All gates passed" in results["message"]


def test_gate_analyzer_diagnose_failure(temp_data_dir):
    """Test diagnosis when gates fail."""
    data_dir = Path(temp_data_dir)

    # Write failing canary, passing others
    canary = {"passed": False, "metrics": {"tpr": 0.92, "tnr": 0.88}, "errors": []}
    shadow = {"verdict": "PASS", "gates": []}
    f14 = {"verdict": "PASS", "overall": {"tpr": 0.85}}

    with open(data_dir / "canary" / "canary_results.json", "w") as f:
        json.dump(canary, f)
    with open(data_dir / "models" / "shadow_results.json", "w") as f:
        json.dump(shadow, f)
    with open(data_dir / "models" / "f14_gate_results.json", "w") as f:
        json.dump(f14, f)

    analyzer = GateAnalyzer(data_dir=str(data_dir))
    results = analyzer.diagnose_failures()

    assert results["overall_verdict"] == "FAILED"
    assert "Canary" in results["message"]
    assert "failed" in results["message"].lower()


def test_gate_analyzer_format_message(temp_data_dir):
    """Test message formatting for iMessage."""
    data_dir = Path(temp_data_dir)

    canary = {"passed": True, "metrics": {"tpr": 0.97, "tnr": 0.93}}
    shadow = {"verdict": "PASS", "gates": []}

    with open(data_dir / "canary" / "canary_results.json", "w") as f:
        json.dump(canary, f)
    with open(data_dir / "models" / "shadow_results.json", "w") as f:
        json.dump(shadow, f)

    analyzer = GateAnalyzer(data_dir=str(data_dir))
    message = analyzer.format_message()

    assert isinstance(message, str)
    assert len(message) > 0
    assert "PASSED" in message or "All gates" in message


def test_gate_analyzer_missing_files(temp_data_dir):
    """Test handling of missing gate result files."""
    analyzer = GateAnalyzer(data_dir=temp_data_dir)

    # Should return None for missing files
    canary = analyzer.check_canary()
    shadow = analyzer.check_shadow()
    f14 = analyzer.check_f14()

    assert canary is None
    assert shadow is None
    assert f14 is None

    # Overall diagnosis should still work
    results = analyzer.diagnose_failures()
    assert "overall_verdict" in results


def test_gate_analyzer_write_report(temp_data_dir):
    """Test writing failure report to disk."""
    data_dir = Path(temp_data_dir)

    # Write failing gate
    canary = {"passed": False, "metrics": {"tpr": 0.92}, "errors": []}
    with open(data_dir / "canary" / "canary_results.json", "w") as f:
        json.dump(canary, f)

    analyzer = GateAnalyzer(data_dir=str(data_dir))
    report_path = analyzer.write_report(report_dir=str(data_dir))

    assert report_path is not None
    assert Path(report_path).exists()

    with open(report_path) as f:
        report = json.load(f)
    assert "overall_verdict" in report
    assert report["overall_verdict"] == "FAILED"
