"""Tests for Layer 16 BOCPD Change Point Detection (T3.6)."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from na0s.layer16.detectors.change_point import BOCPD, ChangePointDetector
from na0s.layer16.models import Alert, ConversationState, ConversationTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(risk_score: float = 0.0) -> ConversationTurn:
    return ConversationTurn(
        turn_id=str(uuid.uuid4()),
        text="test turn",
        role="user",
        timestamp=datetime.now(timezone.utc),
        risk_score=risk_score,
        label="safe" if risk_score < 0.5 else "injection",
        flags=[],
    )


def _make_state(
    risk_scores: list[float],
    session_id: str = "test-session",
) -> ConversationState:
    state = ConversationState(session_id=session_id)
    for score in risk_scores:
        state.turns.append(_make_turn(score))
    return state


# ---------------------------------------------------------------------------
# BOCPD: constant signal -> no change point
# ---------------------------------------------------------------------------

class TestBOCPDConstantSignal:
    def test_constant_low_signal(self):
        bocpd = BOCPD(hazard_rate=0.02)
        probs = []
        for _ in range(20):
            p = bocpd.update(0.1)
            probs.append(p)
        # After initial warmup (first ~2 turns), CP probability should stay low
        assert all(p < 0.5 for p in probs[2:])

    def test_constant_high_signal(self):
        bocpd = BOCPD(hazard_rate=0.02)
        probs = []
        for _ in range(20):
            p = bocpd.update(0.9)
            probs.append(p)
        # After warmup, constant high signal: no change
        assert all(p < 0.5 for p in probs[2:])


# ---------------------------------------------------------------------------
# BOCPD: step change detection
# ---------------------------------------------------------------------------

class TestBOCPDStepChange:
    def test_detects_step_change(self):
        bocpd = BOCPD(hazard_rate=0.02)
        # Phase 1: low risk (enough turns to establish baseline)
        for _ in range(10):
            bocpd.update(0.1)
        # Phase 2: sudden high risk
        cp_probs = []
        for _ in range(5):
            p = bocpd.update(0.9)
            cp_probs.append(p)
        # At least one of the early high-risk observations should trigger high CP prob
        assert max(cp_probs) > 0.5, f"Expected CP detection, got max={max(cp_probs)}"

    def test_step_change_higher_than_gradual(self):
        # Step change
        bocpd_step = BOCPD(hazard_rate=0.02)
        for _ in range(10):
            bocpd_step.update(0.1)
        step_probs = []
        for _ in range(5):
            step_probs.append(bocpd_step.update(0.9))

        # Gradual change
        bocpd_grad = BOCPD(hazard_rate=0.02)
        for _ in range(10):
            bocpd_grad.update(0.1)
        grad_probs = []
        for val in [0.2, 0.35, 0.5, 0.65, 0.8]:
            grad_probs.append(bocpd_grad.update(val))

        # Step change should produce higher peak CP probability
        assert max(step_probs) > max(grad_probs)


# ---------------------------------------------------------------------------
# BOCPD: gradual increase
# ---------------------------------------------------------------------------

class TestBOCPDGradualIncrease:
    def test_gradual_increase_lower_confidence(self):
        bocpd = BOCPD(hazard_rate=0.05, mu_prior=0.1, beta_prior=0.05)
        for _ in range(10):
            bocpd.update(0.1)
        probs = []
        for val in [0.15, 0.2, 0.25, 0.3, 0.35]:
            probs.append(bocpd.update(val))
        # Gradual changes should produce lower CP probs than step changes
        # The max prob should stay relatively modest
        assert max(probs) < 0.8


# ---------------------------------------------------------------------------
# Probability validity
# ---------------------------------------------------------------------------

class TestProbabilityValidity:
    def test_probability_in_01(self):
        bocpd = BOCPD()
        for val in [0.0, 0.1, 0.5, 0.9, 1.0, 0.0, 0.0, 0.8, 0.8]:
            p = bocpd.update(val)
            assert 0.0 <= p <= 1.0, f"P(cp)={p} out of [0,1]"

    def test_all_probabilities_sum_to_one(self):
        bocpd = BOCPD()
        for _ in range(10):
            bocpd.update(0.3)
        total = sum(bocpd._run_length_probs)
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_to_prior(self):
        bocpd = BOCPD(mu_prior=0.2, kappa_prior=2.0, alpha_prior=1.5, beta_prior=0.3)
        for _ in range(10):
            bocpd.update(0.5)
        bocpd.reset()
        assert bocpd.change_point_probability == 0.0
        assert bocpd.observation_count == 0
        assert len(bocpd._run_length_probs) == 1
        assert bocpd._run_length_probs[0] == 1.0
        assert bocpd._mu == [0.2]
        assert bocpd._kappa == [2.0]


# ---------------------------------------------------------------------------
# Student-t PDF validity
# ---------------------------------------------------------------------------

class TestStudentTPDF:
    def test_returns_positive(self):
        bocpd = BOCPD()
        result = bocpd._student_t_pdf(0.5, 0.1, 0.2, 3.0)
        assert result > 0.0

    def test_peak_at_mean(self):
        bocpd = BOCPD()
        at_mean = bocpd._student_t_pdf(0.5, 0.5, 0.1, 10.0)
        away = bocpd._student_t_pdf(2.0, 0.5, 0.1, 10.0)
        assert at_mean > away

    def test_zero_nu_returns_eps(self):
        bocpd = BOCPD()
        result = bocpd._student_t_pdf(0.5, 0.1, 0.2, 0.0)
        assert result > 0.0  # should return _EPS, not crash

    def test_zero_sigma_returns_eps(self):
        bocpd = BOCPD()
        result = bocpd._student_t_pdf(0.5, 0.1, 0.0, 3.0)
        assert result > 0.0


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_no_nan_inf_on_extreme_values(self):
        bocpd = BOCPD()
        for val in [0.0, 1e-10, 1.0, 0.999999, 0.0, 1.0, 0.0]:
            p = bocpd.update(val)
            assert not math.isnan(p), "NaN in change point probability"
            assert not math.isinf(p), "Inf in change point probability"

    def test_long_sequence_stability(self):
        bocpd = BOCPD()
        for i in range(200):
            val = 0.1 if i < 100 else 0.9
            p = bocpd.update(val)
            assert not math.isnan(p)
            assert not math.isinf(p)
            assert 0.0 <= p <= 1.0

    def test_nan_input_rejected(self):
        bocpd = BOCPD()
        with pytest.raises(ValueError):
            bocpd.update(float("nan"))

    def test_inf_input_rejected(self):
        bocpd = BOCPD()
        with pytest.raises(ValueError):
            bocpd.update(float("inf"))

    def test_non_numeric_input_rejected(self):
        bocpd = BOCPD()
        with pytest.raises(TypeError):
            bocpd.update("0.5")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BOCPD input validation
# ---------------------------------------------------------------------------

class TestBOCPDValidation:
    def test_invalid_hazard_rate(self):
        with pytest.raises(ValueError):
            BOCPD(hazard_rate=0.0)
        with pytest.raises(ValueError):
            BOCPD(hazard_rate=-0.1)
        with pytest.raises(ValueError):
            BOCPD(hazard_rate=1.1)

    def test_invalid_kappa(self):
        with pytest.raises(ValueError):
            BOCPD(kappa_prior=0.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            BOCPD(alpha_prior=-1.0)

    def test_invalid_beta(self):
        with pytest.raises(ValueError):
            BOCPD(beta_prior=0.0)


# ---------------------------------------------------------------------------
# ChangePointDetector integration
# ---------------------------------------------------------------------------

class TestChangePointDetector:
    def test_alerts_on_step_change(self):
        detector = ChangePointDetector()
        # Build a conversation with a step change
        scores = [0.1] * 10 + [0.9] * 5
        state = _make_state(scores)

        alerts = detector.analyze(state)
        # Should produce at least one change_point alert
        # (depends on BOCPD sensitivity, but step from 0.1 to 0.9 should trigger)
        cp_alerts = [a for a in alerts if a.alert_type == "change_point"]
        assert len(cp_alerts) >= 1
        assert cp_alerts[0].confidence > 0.0

    def test_no_alert_on_constant_benign(self):
        detector = ChangePointDetector()
        scores = [0.05] * 15
        state = _make_state(scores)
        alerts = detector.analyze(state)
        cp_alerts = [a for a in alerts if a.alert_type == "change_point"]
        assert len(cp_alerts) == 0

    def test_no_alert_below_min_turns(self):
        detector = ChangePointDetector()
        state = _make_state([0.9, 0.9])  # only 2 turns
        alerts = detector.analyze(state)
        assert len(alerts) == 0

    def test_per_session_isolation(self):
        detector = ChangePointDetector()
        # Session A: constant benign
        state_a = _make_state([0.05] * 10, session_id="session-a")
        # Session B: step change
        state_b = _make_state([0.05] * 10 + [0.9] * 5, session_id="session-b")

        alerts_a = detector.analyze(state_a)
        alerts_b = detector.analyze(state_b)

        cp_a = [a for a in alerts_a if a.alert_type == "change_point"]
        cp_b = [a for a in alerts_b if a.alert_type == "change_point"]

        # Session A should have no alerts; Session B should
        assert len(cp_a) == 0
        assert len(cp_b) >= 1

    def test_disabled_via_config(self):
        detector = ChangePointDetector()
        state = _make_state([0.1] * 5 + [0.9] * 5)
        with patch("na0s.layer16.detectors.change_point.layer16_config") as mock_cfg:
            mock_cfg.ENABLE_CHANGE_POINT = False
            alerts = detector.analyze(state)
        assert len(alerts) == 0

    def test_detector_name(self):
        detector = ChangePointDetector()
        assert detector.detector_name == "ChangePointDetector"

    def test_taxonomy_ids(self):
        detector = ChangePointDetector()
        assert "D1.24" in detector.taxonomy_ids

    def test_reset_clears_sessions(self):
        detector = ChangePointDetector()
        state = _make_state([0.1] * 5, session_id="s1")
        detector.analyze(state)
        assert "s1" in detector._sessions
        detector.reset()
        assert len(detector._sessions) == 0

    def test_remove_session(self):
        detector = ChangePointDetector()
        state = _make_state([0.1] * 5, session_id="s1")
        detector.analyze(state)
        assert "s1" in detector._sessions
        detector.remove_session("s1")
        assert "s1" not in detector._sessions

    def test_incremental_feeding(self):
        """Detector should only feed new turns, not re-feed old ones."""
        detector = ChangePointDetector()

        # First call: 5 benign turns
        state = _make_state([0.05] * 5, session_id="inc-test")
        detector.analyze(state)
        assert detector._session_turn_counts["inc-test"] == 5

        # Second call: add 3 more turns to the same state
        for _ in range(3):
            state.turns.append(_make_turn(0.05))
        detector.analyze(state)
        assert detector._session_turn_counts["inc-test"] == 8


# ---------------------------------------------------------------------------
# Run length vector bounding
# ---------------------------------------------------------------------------

class TestRunLengthBounding:
    def test_run_length_bounded(self):
        bocpd = BOCPD(hazard_rate=0.002)  # very low hazard -> long runs
        for _ in range(600):
            bocpd.update(0.1)
        # Run length vector should be capped at _MAX_RUN_LENGTH (500)
        assert len(bocpd._run_length_probs) <= 500
