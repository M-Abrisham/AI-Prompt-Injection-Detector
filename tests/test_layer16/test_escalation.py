"""Tests for Layer 16 escalation detector."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from na0s.layer16.detectors.escalation import (
    EscalationDetector,
    _linear_slope,
    _r_squared,
    _is_monotonically_increasing,
)
from na0s.layer16.models import ConversationState, ConversationTurn

FIXTURES = Path(__file__).parent / "fixtures" / "escalation_conversations.json"


def _load_fixtures():
    with open(FIXTURES) as f:
        return json.load(f)


def _build_state(scenario: dict) -> ConversationState:
    turns = []
    for i, t in enumerate(scenario["turns"]):
        turns.append(
            ConversationTurn(
                turn_id=f"t{i}",
                text=t["text"],
                timestamp=datetime.now(timezone.utc),
                risk_score=t["risk_score"],
            )
        )
    state = ConversationState(session_id="test", turns=turns)
    return state


# ---------------------------------------------------------------------------
# Unit: linear slope helper
# ---------------------------------------------------------------------------


class TestLinearSlope:
    def test_empty(self):
        assert _linear_slope([]) == 0.0

    def test_single(self):
        assert _linear_slope([0.5]) == 0.0

    def test_flat(self):
        assert abs(_linear_slope([0.5, 0.5, 0.5])) < 1e-9

    def test_positive_slope(self):
        slope = _linear_slope([0.0, 0.5, 1.0])
        assert slope == pytest.approx(0.5, abs=1e-6)

    def test_negative_slope(self):
        slope = _linear_slope([1.0, 0.5, 0.0])
        assert slope == pytest.approx(-0.5, abs=1e-6)


class TestRSquared:
    def test_perfect_fit(self):
        """Perfect linear data should have R^2 = 1.0."""
        values = [0.0, 0.5, 1.0]
        slope = _linear_slope(values)
        r2 = _r_squared(values, slope)
        assert r2 == pytest.approx(1.0, abs=1e-9)

    def test_flat_values(self):
        """All same values => ss_tot = 0 => R^2 = 0."""
        r2 = _r_squared([0.5, 0.5, 0.5], 0.0)
        assert r2 == 0.0

    def test_noisy_data(self):
        """Noisy data should have R^2 < 1.0."""
        values = [0.1, 0.4, 0.2, 0.5, 0.3]
        slope = _linear_slope(values)
        r2 = _r_squared(values, slope)
        assert 0.0 <= r2 < 1.0

    def test_empty(self):
        assert _r_squared([], 0.0) == 0.0

    def test_single(self):
        assert _r_squared([0.5], 0.0) == 0.0


class TestMonotonicallyIncreasing:
    def test_increasing(self):
        assert _is_monotonically_increasing([0.1, 0.2, 0.3]) is True

    def test_flat(self):
        assert _is_monotonically_increasing([0.1, 0.1, 0.2]) is False

    def test_decreasing(self):
        assert _is_monotonically_increasing([0.3, 0.2, 0.1]) is False

    def test_single(self):
        assert _is_monotonically_increasing([0.5]) is True

    def test_empty(self):
        assert _is_monotonically_increasing([]) is True


# ---------------------------------------------------------------------------
# Integration: escalation detection
# ---------------------------------------------------------------------------


class TestEscalationDetector:
    def setup_method(self):
        self.detector = EscalationDetector()

    def test_detector_name(self):
        assert self.detector.detector_name == "escalation"

    def test_taxonomy_ids(self):
        assert "C1.1" in self.detector.taxonomy_ids
        assert "C1MT.1" in self.detector.taxonomy_ids
        assert "C1MT.3" in self.detector.taxonomy_ids

    def test_empty_state(self):
        state = ConversationState(session_id="empty")
        assert self.detector.analyze(state) == []

    def test_none_state(self):
        assert self.detector.analyze(None) == []

    def test_single_turn(self):
        state = ConversationState(
            session_id="one",
            turns=[
                ConversationTurn(
                    turn_id="t0",
                    text="hi",
                    timestamp=datetime.now(timezone.utc),
                    risk_score=0.9,
                )
            ],
        )
        assert self.detector.analyze(state) == []

    def test_short_conversation_no_alert(self):
        """Fewer than ESCALATION_MIN_TURNS should produce no trend alert."""
        state = ConversationState(
            session_id="short",
            turns=[
                ConversationTurn(
                    turn_id="t0",
                    text="a",
                    timestamp=datetime.now(timezone.utc),
                    risk_score=0.1,
                ),
                ConversationTurn(
                    turn_id="t1",
                    text="b",
                    timestamp=datetime.now(timezone.utc),
                    risk_score=0.5,
                ),
            ],
        )
        alerts = self.detector.analyze(state)
        # No trend alert (only 2 turns), and no rapid alert (need 3 above 0.5)
        assert len(alerts) == 0

    def test_gradual_escalation_detected(self):
        scenarios = _load_fixtures()
        scenario = next(s for s in scenarios if s["name"] == "gradual_escalation_attack")
        state = _build_state(scenario)
        alerts = self.detector.analyze(state)
        assert len(alerts) >= 1
        types = {a.alert_type for a in alerts}
        assert "escalation" in types

    def test_rapid_escalation_detected(self):
        scenarios = _load_fixtures()
        scenario = next(s for s in scenarios if s["name"] == "rapid_escalation_attack")
        state = _build_state(scenario)
        alerts = self.detector.analyze(state)
        assert len(alerts) >= 1
        rapid = [a for a in alerts if "rapid" in a.description.lower() or "Rapid" in a.description]
        assert len(rapid) >= 1

    def test_benign_conversation_not_flagged(self):
        scenarios = _load_fixtures()
        scenario = next(s for s in scenarios if s["name"] == "benign_conversation")
        state = _build_state(scenario)
        alerts = self.detector.analyze(state)
        assert len(alerts) == 0

    def test_noisy_non_escalation(self):
        scenarios = _load_fixtures()
        scenario = next(s for s in scenarios if s["name"] == "noisy_non_escalation")
        state = _build_state(scenario)
        alerts = self.detector.analyze(state)
        # Should not flag trend-based escalation (slope is near zero or negative)
        trend_alerts = [a for a in alerts if "slope" in a.description.lower()]
        assert len(trend_alerts) == 0

    def test_slow_burn_detected(self):
        scenarios = _load_fixtures()
        scenario = next(s for s in scenarios if s["name"] == "slow_burn_escalation")
        state = _build_state(scenario)
        alerts = self.detector.analyze(state)
        assert len(alerts) >= 1

    def test_monotonic_boost(self):
        """Monotonically increasing last 3 turns should boost confidence."""
        state = ConversationState(
            session_id="mono",
            turns=[
                ConversationTurn(turn_id="t0", text="a", timestamp=datetime.now(timezone.utc), risk_score=0.1),
                ConversationTurn(turn_id="t1", text="b", timestamp=datetime.now(timezone.utc), risk_score=0.3),
                ConversationTurn(turn_id="t2", text="c", timestamp=datetime.now(timezone.utc), risk_score=0.5),
                ConversationTurn(turn_id="t3", text="d", timestamp=datetime.now(timezone.utc), risk_score=0.7),
                ConversationTurn(turn_id="t4", text="e", timestamp=datetime.now(timezone.utc), risk_score=0.9),
            ],
        )
        alerts = self.detector.analyze(state)
        assert len(alerts) >= 1
        # Check that r_squared is in the evidence
        trend_alerts = [a for a in alerts if "slope" in a.description.lower()]
        assert len(trend_alerts) >= 1
        assert any("r_squared" in e for e in trend_alerts[0].evidence)

    def test_confidence_uses_r_squared(self):
        """Noisy data with a positive slope should have lower confidence
        than clean linearly increasing data."""
        # Clean data
        clean_state = ConversationState(
            session_id="clean",
            turns=[
                ConversationTurn(turn_id="t0", text="a", timestamp=datetime.now(timezone.utc), risk_score=0.1),
                ConversationTurn(turn_id="t1", text="b", timestamp=datetime.now(timezone.utc), risk_score=0.3),
                ConversationTurn(turn_id="t2", text="c", timestamp=datetime.now(timezone.utc), risk_score=0.6),
                ConversationTurn(turn_id="t3", text="d", timestamp=datetime.now(timezone.utc), risk_score=0.8),
            ],
        )
        clean_alerts = self.detector.analyze(clean_state)

        # Noisy data with similar overall trend but dips
        noisy_state = ConversationState(
            session_id="noisy",
            turns=[
                ConversationTurn(turn_id="t0", text="a", timestamp=datetime.now(timezone.utc), risk_score=0.1),
                ConversationTurn(turn_id="t1", text="b", timestamp=datetime.now(timezone.utc), risk_score=0.5),
                ConversationTurn(turn_id="t2", text="c", timestamp=datetime.now(timezone.utc), risk_score=0.2),
                ConversationTurn(turn_id="t3", text="d", timestamp=datetime.now(timezone.utc), risk_score=0.8),
            ],
        )
        noisy_alerts = self.detector.analyze(noisy_state)

        # Both should detect escalation
        assert len(clean_alerts) >= 1
        assert len(noisy_alerts) >= 1

        # Clean data should have higher confidence (better R^2)
        clean_conf = clean_alerts[0].confidence
        noisy_conf = noisy_alerts[0].confidence
        assert clean_conf > noisy_conf

    def test_reset_is_noop(self):
        """Reset should not raise."""
        self.detector.reset()

    def test_all_fixture_scenarios(self):
        """Verify expected_detection matches for every fixture scenario."""
        scenarios = _load_fixtures()
        for scenario in scenarios:
            state = _build_state(scenario)
            alerts = self.detector.analyze(state)
            detected = len(alerts) > 0
            assert detected == scenario["expected_detection"], (
                f"Scenario '{scenario['name']}': expected_detection="
                f"{scenario['expected_detection']} but got {detected}"
            )
