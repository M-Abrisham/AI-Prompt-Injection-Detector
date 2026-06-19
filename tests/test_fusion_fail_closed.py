"""Regression tests: the scoring/decision layer must FAIL CLOSED, not open.

A NaN/inf/out-of-range value reaching the voting math must never silently turn
an attack into SAFE. These tests pin the fixes for the verified fail-open bugs
in ``na0s.fusion.voting`` (NaN ml_prob -> SAFE; DECISION_THRESHOLD=nan ->
universal SAFE; negative score escaping [0,1]; duplicate hits double-counting).
"""

import os

import pytest

import na0s.fusion.voting as voting
from na0s.fusion.voting import weighted_decision


@pytest.fixture(autouse=True)
def _reset_threshold_cache():
    """Each test starts with a clean threshold cache + no env override."""
    saved = os.environ.pop("DECISION_THRESHOLD", None)
    voting._reset_threshold_cache()
    yield
    if saved is not None:
        os.environ["DECISION_THRESHOLD"] = saved
    else:
        os.environ.pop("DECISION_THRESHOLD", None)
    voting._reset_threshold_cache()


# --------------------------------------------------------------------------
# Bug #5 — DECISION_THRESHOLD env must reject nan/inf/out-of-range
# --------------------------------------------------------------------------

class TestThresholdValidation:
    @pytest.mark.parametrize(
        "bad", ["nan", "inf", "-inf", "-5.0", "0", "2.0", "abc", ""],
    )
    def test_invalid_env_threshold_falls_back(self, bad):
        os.environ["DECISION_THRESHOLD"] = bad
        voting._reset_threshold_cache()
        # Must fall through to the safe hardcoded fallback, NOT accept nan/etc.
        assert voting.get_decision_threshold() == voting._FALLBACK_THRESHOLD

    @pytest.mark.parametrize(
        "good,expected", [("0.42", 0.42), ("0.9", 0.9), ("1.0", 1.0)],
    )
    def test_valid_env_threshold_honored(self, good, expected):
        os.environ["DECISION_THRESHOLD"] = good
        voting._reset_threshold_cache()
        assert voting.get_decision_threshold() == expected

    def test_nan_threshold_does_not_make_strong_attack_safe(self):
        """The headline fail-open: nan threshold turned every attack SAFE."""
        os.environ["DECISION_THRESHOLD"] = "nan"
        voting._reset_threshold_cache()
        # The resolver rejects nan and falls back to a valid threshold, so a
        # strong attack signal is still classified MALICIOUS.
        label, _ = weighted_decision(
            0.99, "MALICIOUS", ["override"], ["base64"],
            threshold=voting.get_decision_threshold(),
        )
        assert label == "MALICIOUS", "nan threshold must not fail open to SAFE"


# --------------------------------------------------------------------------
# Bug #1 — NaN/inf ml_prob must not fail open
# --------------------------------------------------------------------------

class TestNaNInputFailsClosed:
    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf")],
    )
    def test_nan_inf_ml_prob_with_malicious_signals_not_safe(self, bad):
        # A malicious rule fired; a NaN ML prob must not erase that -> not SAFE.
        label, score = weighted_decision(bad, "MALICIOUS", ["override"], [])
        assert label == "MALICIOUS", f"{bad} ml_prob failed open to SAFE"
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf")],
    )
    def test_nan_inf_ml_prob_score_is_finite_and_bounded(self, bad):
        _, score = weighted_decision(bad, "MALICIOUS", [], [])
        assert 0.0 <= score <= 1.0  # no NaN/inf leaks into the score


# --------------------------------------------------------------------------
# Bug #2 — output score must stay in [0,1] on EVERY path
# --------------------------------------------------------------------------

class TestScoreAlwaysBounded:
    @pytest.mark.parametrize(
        "ml_prob,label_in,hits,flags",
        [
            (-3.0, "MALICIOUS", [], []),
            (5.0, "SAFE", [], []),
            (-1.0, "SAFE", ["override"], []),
            (10.0, "MALICIOUS", ["override", "roleplay"], ["base64"]),
        ],
    )
    def test_score_in_unit_interval(self, ml_prob, label_in, hits, flags):
        _, score = weighted_decision(ml_prob, label_in, hits, flags)
        assert 0.0 <= score <= 1.0, f"score {score} escaped [0,1]"


# --------------------------------------------------------------------------
# Bug #3 — duplicate hits must not inflate the score
# --------------------------------------------------------------------------

class TestDuplicateHitsNotDoubleCounted:
    def test_repeated_same_rule_does_not_increase_score(self):
        _, s1 = weighted_decision(0.5, "SAFE", ["override"], [])
        _, s2 = weighted_decision(0.5, "SAFE", ["override", "override"], [])
        _, s3 = weighted_decision(
            0.5, "SAFE", ["override", "override", "override"], [],
        )
        assert s1 == s2 == s3, "duplicate identical hits must not amplify score"
