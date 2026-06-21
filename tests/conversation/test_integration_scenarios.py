"""Integration tests: run fixture-based scenarios through the harness.

Each JSON fixture file is loaded and every scenario is executed via
pytest.mark.parametrize so failures pinpoint the exact scenario.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from na0s.layer16.testing.conversation_harness import ConversationTestHarness
from na0s.layer16.testing.metrics import DetectionMetrics
from na0s.layer16.testing.scenario_loader import TestScenario, load_scenarios

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scenario_ids(scenarios: list[TestScenario]) -> list[str]:
    return [s.name for s in scenarios]


def _run_scenario(harness: ConversationTestHarness, scenario: TestScenario) -> bool:
    """Run a scenario through the harness and return whether alerts fired."""
    harness.send_sequence(scenario.turns)
    return harness.alerts_triggered()


# ---------------------------------------------------------------------------
# Benign conversations — none should trigger alerts
# ---------------------------------------------------------------------------

_benign = load_scenarios(FIXTURES_DIR / "benign_conversations.json")


@pytest.mark.parametrize("scenario", _benign, ids=_scenario_ids(_benign))
def test_benign_no_alerts(monitor, scenario: TestScenario):
    harness = ConversationTestHarness(monitor)
    detected = _run_scenario(harness, scenario)
    assert detected == scenario.expected_detection, (
        f"Scenario {scenario.name!r}: expected detection={scenario.expected_detection}, "
        f"got {detected}. Alerts: {harness.all_alerts()}"
    )
    harness.reset()


# ---------------------------------------------------------------------------
# Attack scenarios — should trigger the expected alert type
# ---------------------------------------------------------------------------

_attacks = load_scenarios(FIXTURES_DIR / "attack_scenarios.json")


@pytest.mark.parametrize("scenario", _attacks, ids=_scenario_ids(_attacks))
def test_attack_detected(monitor, scenario: TestScenario):
    harness = ConversationTestHarness(monitor)
    _run_scenario(harness, scenario)

    if scenario.expected_detection:
        assert harness.alerts_triggered(), (
            f"Scenario {scenario.name!r}: expected detection but got no alerts"
        )
        if scenario.expected_alert_type:
            harness.assert_alert(
                scenario.expected_alert_type,
                min_confidence=scenario.expected_min_confidence,
            )
    else:
        harness.assert_no_alerts()
    harness.reset()


# ---------------------------------------------------------------------------
# Edge cases — should handle gracefully without false positives
# ---------------------------------------------------------------------------

_edges = load_scenarios(FIXTURES_DIR / "edge_cases.json")


@pytest.mark.parametrize("scenario", _edges, ids=_scenario_ids(_edges))
def test_edge_case(monitor, scenario: TestScenario):
    harness = ConversationTestHarness(monitor)
    detected = _run_scenario(harness, scenario)
    assert detected == scenario.expected_detection, (
        f"Scenario {scenario.name!r}: expected detection={scenario.expected_detection}, "
        f"got {detected}. Alerts: {harness.all_alerts()}"
    )
    harness.reset()


# ---------------------------------------------------------------------------
# Aggregate metrics across all fixtures
# ---------------------------------------------------------------------------

def test_aggregate_metrics(monitor):
    """Run all scenarios and verify overall detection quality."""
    metrics = DetectionMetrics()

    all_scenarios = _benign + _attacks + _edges
    for scenario in all_scenarios:
        harness = ConversationTestHarness(monitor)
        _run_scenario(harness, scenario)
        detected = harness.alerts_triggered()
        metrics.record(
            scenario_name=scenario.name,
            expected=scenario.expected_detection,
            actual_detected=detected,
            alerts=harness.all_alerts(),
        )

    # Print the report for visibility in pytest -v output
    report = metrics.report()
    print(f"\n{report}")

    # Detection rate should be at least 60% (allows room for detector tuning)
    assert metrics.detection_rate() >= 0.6, (
        f"Detection rate too low: {metrics.detection_rate():.2%}\n{report}"
    )
    # False positive rate should be at most 20%
    assert metrics.false_positive_rate() <= 0.2, (
        f"False positive rate too high: {metrics.false_positive_rate():.2%}\n{report}"
    )
