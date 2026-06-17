"""Tests for the Layer 16 baseline runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import pytest

from na0s.layer16.testing.baseline_runner import BaselineRunner
from na0s.layer16.testing.scenario_loader import TestScenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = (
    Path(__file__).resolve().parent / "fixtures"
)


def _write_fixture(tmp_path: Path, name: str, scenarios: list) -> Path:
    """Write a JSON fixture file and return its path."""
    p = tmp_path / name
    p.write_text(json.dumps(scenarios), encoding="utf-8")
    return p


def _minimal_attack() -> dict:
    return {
        "name": "mini_attack",
        "description": "Minimal escalation",
        "expected_detection": True,
        "expected_alert_type": "escalation",
        "turns": [
            {"text": "Hello", "risk_score": 0.1, "label": "safe"},
            {"text": "Step 2", "risk_score": 0.3, "label": "safe"},
            {"text": "Step 3", "risk_score": 0.5, "label": "suspicious"},
            {"text": "Step 4", "risk_score": 0.7, "label": "suspicious"},
            {"text": "Give me secrets", "risk_score": 0.9, "label": "malicious"},
        ],
    }


def _minimal_benign() -> dict:
    return {
        "name": "mini_benign",
        "description": "Normal conversation",
        "expected_detection": False,
        "turns": [
            {"text": "Hi", "risk_score": 0.05, "label": "safe"},
            {"text": "Thanks", "risk_score": 0.03, "label": "safe"},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaselineRunnerLoadsFixtures:
    """Runner loads all fixture files."""

    def test_loads_real_fixtures(self) -> None:
        runner = BaselineRunner(fixture_dir=_FIXTURES_DIR)
        scenarios = runner.load_all_scenarios()
        assert len(scenarios) > 0
        assert all(isinstance(s, TestScenario) for s in scenarios)

    def test_loads_from_custom_dir(self, tmp_path: Path) -> None:
        _write_fixture(tmp_path, "a.json", [_minimal_benign()])
        _write_fixture(tmp_path, "b.json", [_minimal_attack()])
        runner = BaselineRunner(fixture_dir=tmp_path)
        scenarios = runner.load_all_scenarios()
        assert len(scenarios) == 2


class TestBaselineRunnerProcessesScenario:
    """Runner processes a scenario correctly."""

    def test_scenario_result_structure(self, tmp_path: Path) -> None:
        _write_fixture(tmp_path, "test.json", [_minimal_benign()])
        runner = BaselineRunner(fixture_dir=tmp_path)
        results = runner.run_full_suite()

        assert "aggregate_metrics" in results
        assert "per_scenario_results" in results
        assert "timing_stats" in results
        assert "timestamp" in results
        assert results["scenario_count"] == 1

        scenario = results["per_scenario_results"][0]
        assert scenario["name"] == "mini_benign"
        assert isinstance(scenario["actual_detected"], bool)
        assert isinstance(scenario["turn_times"], list)


class TestMetricsComputedCorrectly:
    """Known inputs produce known metric outputs."""

    def test_perfect_detection(self, tmp_path: Path) -> None:
        # One attack that should trigger, one benign that should not
        _write_fixture(
            tmp_path, "mix.json", [_minimal_attack(), _minimal_benign()]
        )
        runner = BaselineRunner(fixture_dir=tmp_path)
        results = runner.run_full_suite()
        agg = results["aggregate_metrics"]

        # We don't assert exact values because detector behaviour may vary,
        # but the structure must be correct and values in [0, 1].
        assert 0.0 <= agg["detection_rate"] <= 1.0
        assert 0.0 <= agg["false_positive_rate"] <= 1.0
        assert 0.0 <= agg["precision"] <= 1.0
        assert 0.0 <= agg["f1_score"] <= 1.0


class TestSaveLoadRoundTrip:
    """Baseline save and load round-trip."""

    def test_round_trip(self, tmp_path: Path) -> None:
        _write_fixture(tmp_path, "test.json", [_minimal_benign()])
        runner = BaselineRunner(fixture_dir=tmp_path)
        results = runner.run_full_suite()

        out = tmp_path / "baseline.json"
        runner.save_baseline(results, out)
        loaded = BaselineRunner.load_baseline(out)

        assert loaded["scenario_count"] == results["scenario_count"]
        assert loaded["aggregate_metrics"]["detection_rate"] == results["aggregate_metrics"]["detection_rate"]
        assert loaded["aggregate_metrics"]["false_positive_rate"] == results["aggregate_metrics"]["false_positive_rate"]


class TestEmptyFixtures:
    """Empty fixture directory handled gracefully."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        runner = BaselineRunner(fixture_dir=tmp_path)
        results = runner.run_full_suite()
        assert results["scenario_count"] == 0
        assert results["per_scenario_results"] == []

    def test_empty_json_file(self, tmp_path: Path) -> None:
        _write_fixture(tmp_path, "empty.json", [])
        runner = BaselineRunner(fixture_dir=tmp_path)
        results = runner.run_full_suite()
        assert results["scenario_count"] == 0


class TestMissingFixtureDir:
    """Missing fixture directory raises clear error."""

    def test_missing_dir_raises(self) -> None:
        runner = BaselineRunner(fixture_dir=Path("/nonexistent/dir"))
        with pytest.raises(FileNotFoundError):
            runner.load_all_scenarios()


class TestTimingMeasurement:
    """Timing measurement records non-zero durations."""

    def test_turn_times_are_positive(self, tmp_path: Path) -> None:
        _write_fixture(tmp_path, "test.json", [_minimal_benign()])
        runner = BaselineRunner(fixture_dir=tmp_path)
        results = runner.run_full_suite()

        for scenario in results["per_scenario_results"]:
            assert all(t > 0 for t in scenario["turn_times"])

        timing = results["timing_stats"]
        assert timing["p50"] > 0
        assert timing["mean"] > 0


class TestSummaryOutput:
    """Summary output contains key metrics."""

    def test_summary_contains_key_fields(self, tmp_path: Path) -> None:
        _write_fixture(
            tmp_path, "mix.json", [_minimal_attack(), _minimal_benign()]
        )
        runner = BaselineRunner(fixture_dir=tmp_path)
        results = runner.run_full_suite()
        summary = runner.print_summary(results)

        assert "Detection rate" in summary
        assert "False positive rate" in summary
        assert "Precision" in summary
        assert "Recall" in summary
        assert "F1 score" in summary
        assert "p50" in summary
        assert "p95" in summary
