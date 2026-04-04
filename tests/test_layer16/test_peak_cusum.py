"""Tests for T2.1 (Peak + Accumulation scoring) and T2.2 (CUSUM change detection)."""

from __future__ import annotations

import pytest

from na0s.layer16.models import ConversationState, MultiTurnAnalysis
from na0s.layer16.state import (
    add_turn,
    compute_peak_accumulation,
    from_dict,
    is_cusum_alert,
    to_dict,
    update_cusum,
)


def _make_state(session_id: str = "test-session") -> ConversationState:
    return ConversationState(session_id=session_id)


# ---------------------------------------------------------------------------
# T2.1 -- Peak + Accumulation tests
# ---------------------------------------------------------------------------


class TestPeakAccumulation:
    def test_empty_state_returns_zero(self):
        state = _make_state()
        assert compute_peak_accumulation(state) == 0.0

    def test_all_benign_conversation(self):
        state = _make_state()
        for _ in range(5):
            add_turn(state, "hello", risk_score=0.05, label="safe")
        score = compute_peak_accumulation(state)
        assert score < 0.2, f"All-benign should be < 0.2, got {score}"

    def test_single_high_risk_turn(self):
        """A single high-risk turn should get a moderate P+A score
        (high peak but low persistence)."""
        state = _make_state()
        add_turn(state, "benign", risk_score=0.05, label="safe")
        add_turn(state, "malicious", risk_score=0.9, label="injection")
        add_turn(state, "benign", risk_score=0.05, label="safe")
        score = compute_peak_accumulation(state)
        # peak=0.9 -> 0.36, persistence=1/3 -> ~0.117, diversity=1/5 -> 0.05
        # total ~ 0.527
        assert 0.3 < score < 0.7, f"Single spike should be moderate, got {score}"

    def test_flat_score_attack_caught_by_persistence(self):
        """Crescendo-style: many turns at 0.3 risk.  EMA converges but P+A
        should catch it via high persistence ratio."""
        state = _make_state()
        for _ in range(10):
            add_turn(state, "subtle probe", risk_score=0.3, label="prompt_injection")
        score = compute_peak_accumulation(state)
        # peak=0.3 -> 0.12, persistence=10/10 -> 0.35, diversity=1/5 -> 0.05
        # total = 0.52
        assert score > 0.4, f"Flat-score attack should be caught, got {score}"

    def test_diverse_labels_contribute(self):
        """Category diversity should increase the score."""
        state = _make_state()
        labels = ["injection", "jailbreak", "obfuscation", "social_engineering"]
        for lbl in labels:
            add_turn(state, f"attack {lbl}", risk_score=0.4, label=lbl)
        score = compute_peak_accumulation(state)
        # peak=0.4 -> 0.16, persistence=4/4 -> 0.35, diversity=4/5 -> 0.20
        # total = 0.71
        assert score > 0.6, f"Diverse labels should boost score, got {score}"

    def test_peak_risk_field_updates_on_add_turn(self):
        state = _make_state()
        add_turn(state, "low", risk_score=0.1, label="safe")
        assert state.peak_risk == pytest.approx(0.1)
        add_turn(state, "high", risk_score=0.8, label="injection")
        assert state.peak_risk == pytest.approx(0.8)
        add_turn(state, "medium", risk_score=0.5, label="safe")
        assert state.peak_risk == pytest.approx(0.8), "peak should not decrease"

    def test_score_bounded_zero_one(self):
        state = _make_state()
        add_turn(state, "max risk", risk_score=1.0, label="injection")
        score = compute_peak_accumulation(state)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# T2.2 -- CUSUM tests
# ---------------------------------------------------------------------------


class TestCUSUM:
    def test_all_benign_cusum_stays_near_zero(self):
        state = _make_state()
        for _ in range(10):
            add_turn(state, "hello", risk_score=0.05, label="safe")
        # Each turn: max(0, prev + (0.05 - 0.1 - 0.05)) = max(0, prev - 0.1)
        # Resets to 0 every turn.
        assert state.cusum_score == pytest.approx(0.0)

    def test_sustained_moderate_risk_crosses_threshold(self):
        state = _make_state()
        for _ in range(10):
            add_turn(state, "probe", risk_score=0.3, label="prompt_injection")
        # Each turn: prev + (0.3 - 0.1 - 0.05) = prev + 0.15
        # After 10 turns: 1.5
        assert state.cusum_score == pytest.approx(1.5)
        assert is_cusum_alert(state, threshold=1.0)

    def test_single_spike_then_decay(self):
        state = _make_state()
        add_turn(state, "spike", risk_score=0.9, label="injection")
        # cusum = max(0, 0 + (0.9 - 0.1 - 0.05)) = 0.75
        assert state.cusum_score == pytest.approx(0.75)
        # Now benign turns decay it back toward 0
        for _ in range(8):
            add_turn(state, "benign", risk_score=0.05, label="safe")
        # Each benign: prev + (0.05 - 0.1 - 0.05) = prev - 0.1, floored at 0
        # 0.75 -> 0.65 -> 0.55 -> 0.45 -> 0.35 -> 0.25 -> 0.15 -> 0.05 -> 0.0
        assert state.cusum_score == pytest.approx(0.0)

    def test_is_cusum_alert_true_when_crossed(self):
        state = _make_state()
        state.cusum_score = 1.5
        assert is_cusum_alert(state, threshold=1.0) is True

    def test_is_cusum_alert_false_for_benign(self):
        state = _make_state()
        for _ in range(5):
            add_turn(state, "hi", risk_score=0.0, label="safe")
        assert is_cusum_alert(state) is False

    def test_cusum_capped_at_ten(self):
        state = _make_state()
        # Push cusum very high
        for _ in range(100):
            add_turn(state, "attack", risk_score=1.0, label="injection")
        assert state.cusum_score == pytest.approx(10.0)

    def test_update_cusum_direct(self):
        state = _make_state()
        result = update_cusum(state, 0.5)
        # 0 + (0.5 - 0.1 - 0.05) = 0.35
        assert result == pytest.approx(0.35)
        assert state.cusum_score == pytest.approx(0.35)

    def test_update_cusum_custom_params(self):
        state = _make_state()
        result = update_cusum(state, 0.5, baseline_mean=0.2, allowance=0.1)
        # 0 + (0.5 - 0.2 - 0.1) = 0.2
        assert result == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_cusum_and_peak_in_roundtrip(self):
        state = _make_state()
        add_turn(state, "test", risk_score=0.7, label="injection")
        add_turn(state, "test2", risk_score=0.4, label="jailbreak")

        d = to_dict(state)
        assert "peak_risk" in d
        assert "cusum_score" in d
        assert d["peak_risk"] == pytest.approx(0.7)

        restored = from_dict(d)
        assert restored.peak_risk == pytest.approx(state.peak_risk)
        assert restored.cusum_score == pytest.approx(state.cusum_score)

    def test_backward_compat_from_dict_no_peak_cusum(self):
        """Old serialized data without peak_risk/cusum_score should load fine."""
        state = _make_state()
        add_turn(state, "hello", risk_score=0.1, label="safe")
        d = to_dict(state)
        # Simulate old format by removing new fields.
        del d["peak_risk"]
        del d["cusum_score"]
        restored = from_dict(d)
        assert restored.peak_risk == 0.0
        assert restored.cusum_score == 0.0

    def test_multi_turn_analysis_to_dict_has_new_fields(self):
        analysis = MultiTurnAnalysis(
            session_id="s1",
            peak_accumulation_score=0.55,
            cusum_score=1.2,
        )
        d = analysis.to_dict()
        assert d["peak_accumulation_score"] == pytest.approx(0.55)
        assert d["cusum_score"] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_add_turn_updates_all_scores(self):
        state = _make_state()
        add_turn(state, "probe", risk_score=0.5, label="injection")
        assert state.cumulative_risk > 0.0
        assert state.peak_risk == pytest.approx(0.5)
        assert state.cusum_score > 0.0

    def test_peak_and_cusum_and_cumulative_all_track(self):
        state = _make_state()
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        for s in scores:
            add_turn(state, f"turn {s}", risk_score=s, label="injection")
        assert state.peak_risk == pytest.approx(0.5)
        assert state.cumulative_risk > 0.0
        # CUSUM: each turn adds (s - 0.15) when s >= 0.15, else floors at 0
        # 0.1: max(0, 0 + (0.1-0.15)) = 0
        # 0.2: max(0, 0 + 0.05) = 0.05
        # 0.3: max(0, 0.05 + 0.15) = 0.20
        # 0.4: max(0, 0.20 + 0.25) = 0.45
        # 0.5: max(0, 0.45 + 0.35) = 0.80
        assert state.cusum_score == pytest.approx(0.80)
