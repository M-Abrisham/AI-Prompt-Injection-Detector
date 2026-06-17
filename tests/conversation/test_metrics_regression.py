"""Regression tests that compare current detection against a saved baseline.

These tests are skipped when no baseline file exists yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from na0s.layer16.testing.baseline_runner import BaselineRunner

_BASELINE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "na0s"
    / "conversation"
    / "baselines"
    / "v1_baseline.json"
)

_TOLERANCE = 0.05  # 5 percentage-point tolerance


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not _BASELINE_PATH.exists():
        pytest.skip("No v1 baseline exists yet — run scripts/run_baseline.py --save v1")
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current_results() -> dict:
    runner = BaselineRunner()
    return runner.run_full_suite()


class TestDetectionRateNotRegressed:
    def test_detection_rate_within_tolerance(
        self, baseline: dict, current_results: dict
    ) -> None:
        baseline_dr = baseline["aggregate_metrics"]["detection_rate"]
        current_dr = current_results["aggregate_metrics"]["detection_rate"]
        assert current_dr >= baseline_dr - _TOLERANCE, (
            f"Detection rate regressed: baseline={baseline_dr:.2%}, "
            f"current={current_dr:.2%} (tolerance={_TOLERANCE:.0%})"
        )


class TestFalsePositiveRateNotIncreased:
    def test_fp_rate_within_tolerance(
        self, baseline: dict, current_results: dict
    ) -> None:
        baseline_fpr = baseline["aggregate_metrics"]["false_positive_rate"]
        current_fpr = current_results["aggregate_metrics"]["false_positive_rate"]
        assert current_fpr <= baseline_fpr + _TOLERANCE, (
            f"False positive rate increased: baseline={baseline_fpr:.2%}, "
            f"current={current_fpr:.2%} (tolerance={_TOLERANCE:.0%})"
        )


class TestBaselineFileIntegrity:
    def test_baseline_has_required_keys(self, baseline: dict) -> None:
        assert "aggregate_metrics" in baseline
        assert "per_scenario_results" in baseline
        assert "timing_stats" in baseline
        assert "scenario_count" in baseline
        assert baseline["scenario_count"] > 0
