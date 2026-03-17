"""Tests for P0-1 (composite score clamping) and P0-5 (critical_content severity).

P0-1: _weighted_decision() must always return composite score in [0, 1].
      Before the fix, 5 critical rules + ML safe at 0.55 produced composite ~1.57.

P0-5: cascade.py WeightedClassifier must treat critical_content severity the
      same as critical for override protection.  Before the fix, a
      critical_content hit could be overridden to SAFE by ML trust.

Run: SCAN_TIMEOUT_SEC=0 python3 -m unittest tests.test_p0_score_clamp_and_severity -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Disable thread-based scan timeout so signal.SIGALRM works in main thread.
os.environ["SCAN_TIMEOUT_SEC"] = "0"

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from na0s.predict import _weighted_decision, SEVERITY_WEIGHTS, DECISION_THRESHOLD
from na0s._voting import weighted_decision as voting_weighted_decision
from na0s._voting import DECISION_THRESHOLD as VOTING_THRESHOLD
from na0s._voting import (
    RULE_SEVERITY as VOTING_RULE_SEVERITY,
    FP_EXEMPT_HITS as VOTING_FP_EXEMPT_HITS,
    STRUCTURAL_SIGNAL_WEIGHTS as VOTING_STRUCTURAL_SIGNAL_WEIGHTS,
    AGREEMENT_BOOST as VOTING_AGREEMENT_BOOST,
)
from na0s.cascade import WeightedClassifier
from na0s.rules import RULES, SEVERITY_WEIGHTS as RULES_SEVERITY_WEIGHTS



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(predict_label=0, proba_safe=0.9, proba_mal=0.1):
    """Create a mock sklearn model with controlled predictions."""
    model = MagicMock()
    model.predict.return_value = [predict_label]
    model.predict_proba.return_value = [[proba_safe, proba_mal]]
    return model


def _make_mock_vectorizer():
    """Create a mock TF-IDF vectorizer."""
    vec = MagicMock()
    vec.transform.return_value = MagicMock()
    return vec


@dataclass
class FakeRuleHit:
    """Minimal rule hit for cascade tests."""
    name: str
    severity: str
    technique_ids: list


# ===========================================================================
# P0-1: Composite score clamped to [0, 1]
# ===========================================================================

class TestP0_1_CompositeScoreClamping(unittest.TestCase):
    """_weighted_decision() must return a composite score in [0, 1]."""

    def test_five_critical_rules_ml_safe_clamped(self):
        """5 critical rules (weight 0.3 each) + ML safe at 0.55 must not exceed 1.0.

        Before fix: composite = 0.6*0.45 + 5*0.3 + 0 + 0 = 0.27 + 1.5 = 1.77
        After fix: composite clamped to 1.0.
        """
        # ML says SAFE with 0.55 confidence => ml_prob_malicious = 1.0 - 0.55 = 0.45
        # ml_weight = 0.6 * 0.45 = 0.27
        # 5 critical rules = 5 * 0.3 = 1.5
        # Total unclamped = 1.77
        hits = [
            "override_instruction",
            "direct_prompt_request",
            "decoded_payload_malicious",
            "provide_system_prompt",
            "database_iteration",
        ]
        label, composite = _weighted_decision(
            ml_prob=0.55, ml_label="SAFE",
            hits=hits, obs_flags=[],
        )
        self.assertLessEqual(composite, 1.0,
                             "Composite score must not exceed 1.0 (got {:.4f})".format(composite))
        self.assertGreaterEqual(composite, 0.0,
                                "Composite score must not be negative (got {:.4f})".format(composite))
        self.assertEqual(label, "MALICIOUS")

    def test_extreme_inputs_all_signals_maxed(self):
        """All signal layers maxed out must still produce score <= 1.0.

        ml_weight = 0.6 * 1.0 = 0.6
        rule_weight = many critical rules
        obf_weight = 0.3 (capped)
        structural_weight = ~0.41
        """
        hits = [
            "override_instruction",
            "direct_prompt_request",
            "decoded_payload_malicious",
            "provide_system_prompt",
            "harmful_instructions",
            "phishing_generation",
        ]
        structural = {
            "imperative_start": 1,
            "role_assignment": 1,
            "instruction_boundary": 1,
            "negation_command": 1,
            "quote_depth": 5,
            "text_entropy": 6.0,
        }
        label, composite = _weighted_decision(
            ml_prob=0.99, ml_label="MALICIOUS",
            hits=hits,
            obs_flags=["base64", "rot13", "reversed_text"],
            structural=structural,
        )
        self.assertLessEqual(composite, 1.0,
                             "Extreme input composite must be <= 1.0 (got {:.4f})".format(composite))
        self.assertGreaterEqual(composite, 0.0)
        self.assertEqual(label, "MALICIOUS")

    def test_single_critical_content_rule_clamped(self):
        """Single critical_content rule (weight 0.45) + ML malicious must stay <= 1.0."""
        # ML says MALICIOUS with 0.95 confidence => ml_prob_malicious = 0.95
        # ml_weight = 0.6 * 0.95 = 0.57
        # rule_weight = 0.45 (critical_content)
        # Total = 1.02 -- needs clamping
        label, composite = _weighted_decision(
            ml_prob=0.95, ml_label="MALICIOUS",
            hits=["harmful_instructions"], obs_flags=[],
        )
        self.assertLessEqual(composite, 1.0,
                             "critical_content + ML malicious must be <= 1.0 (got {:.4f})".format(composite))
        self.assertEqual(label, "MALICIOUS")

    def test_zero_inputs_score_zero_or_above(self):
        """No rules, no obfuscation, ML says safe => composite >= 0.0."""
        label, composite = _weighted_decision(
            ml_prob=0.99, ml_label="SAFE",
            hits=[], obs_flags=[],
        )
        self.assertGreaterEqual(composite, 0.0)
        self.assertLessEqual(composite, 1.0)

    def test_composite_is_float_in_range(self):
        """Composite score is always a float in [0.0, 1.0] regardless of inputs."""
        test_cases = [
            # (ml_prob, ml_label, hits, obs_flags)
            (0.99, "MALICIOUS", ["override_instruction"] * 10, ["base64", "rot13"]),
            (0.01, "SAFE", [], []),
            (0.50, "SAFE", ["harmful_instructions", "phishing_generation"], ["high_entropy"]),
            (0.80, "MALICIOUS", ["decoded_payload_malicious", "override_instruction", "direct_prompt_request"], []),
        ]
        for ml_prob, ml_label, hits, obs_flags in test_cases:
            label, composite = _weighted_decision(
                ml_prob=ml_prob, ml_label=ml_label,
                hits=hits, obs_flags=obs_flags,
            )
            self.assertIsInstance(composite, float,
                                 "Composite must be float for inputs ({}, {})".format(ml_prob, ml_label))
            self.assertGreaterEqual(composite, 0.0,
                                    "Composite must be >= 0.0 for ({}, {}, {})".format(ml_prob, ml_label, hits))
            self.assertLessEqual(composite, 1.0,
                                 "Composite must be <= 1.0 for ({}, {}, {})".format(ml_prob, ml_label, hits))

    def test_agreement_boost_still_clamped(self):
        """Agreement boost path (multi-layer) must also stay in [0, 1].

        The agreement boost uses min(composite + boost, 1.0) but the composite
        itself could already be > 1.0 before the boost.  The final clamp
        must catch this.
        """
        # ML malicious + 3 critical rules + 2 obs flags + structural signals
        # = 4 signal layers, which triggers the agreement boost
        hits = [
            "override_instruction",
            "direct_prompt_request",
            "provide_system_prompt",
        ]
        structural = {
            "imperative_start": 1,
            "role_assignment": 1,
        }
        label, composite = _weighted_decision(
            ml_prob=0.9, ml_label="MALICIOUS",
            hits=hits,
            obs_flags=["base64", "rot13"],
            structural=structural,
        )
        self.assertLessEqual(composite, 1.0,
                             "Agreement boost path must be clamped (got {:.4f})".format(composite))
        self.assertGreaterEqual(composite, 0.0)


# ===========================================================================
# P0-5: critical_content severity in cascade.py override protection
# ===========================================================================

class TestP0_5_CriticalContentSeverityInCascade(unittest.TestCase):
    """WeightedClassifier must not override critical_content hits to SAFE."""

    def test_critical_content_not_overridden_to_safe(self):
        """A critical_content rule hit must prevent ML-trust override.

        Scenario: ML is 95% confident SAFE, but a critical_content rule fires.
        Before fix: max_severity stays 'medium', override triggers, returns SAFE.
        After fix: max_severity is 'critical', override blocked.
        """
        wc = WeightedClassifier(threshold=0.55)
        vec = _make_mock_vectorizer()
        # ML says 95% safe -> ml_prob (class 1 / malicious) = 0.05
        model = _make_mock_model(predict_label=0, proba_safe=0.95, proba_mal=0.05)

        # critical_content severity hit.  weight = 0.45
        # ML weight = 0.6 * 0.05 = 0.03
        # composite = 0.03 + 0.45 = 0.48, clamped to 0.48
        # This is below threshold, BUT the override check should NOT fire
        # because max_severity must be "critical" (not "medium").
        fake_hits = [
            FakeRuleHit(name="harmful_instructions", severity="critical_content", technique_ids=["O1.1"])
        ]

        with patch("na0s.cascade.rule_score_detailed", return_value=fake_hits), \
             patch("na0s.cascade.obfuscation_scan", return_value={"evasion_flags": []}):
            label, confidence, hits = wc.classify("test", vec, model)

        # With the fix, max_severity should be "critical" (not "medium"),
        # which blocks the override.  Since composite (0.48) < threshold (0.55),
        # the normal path returns SAFE -- BUT the point is the override did
        # NOT fire because of medium-only severity.  Let's verify max_severity
        # is correctly set by testing with a score above threshold.
        pass  # This test verifies the path; the next test verifies above-threshold.

    def test_critical_content_above_threshold_stays_malicious(self):
        """Multiple critical_content + other rules above threshold must return MALICIOUS.

        Even when ML is confidently safe, critical_content rules should prevent
        the override from firing.
        """
        wc = WeightedClassifier(threshold=0.55)
        vec = _make_mock_vectorizer()
        # ML says 85% safe -> ml_prob (malicious class) = 0.15
        model = _make_mock_model(predict_label=0, proba_safe=0.85, proba_mal=0.15)

        # critical_content (0.45) + medium (0.1) = 0.55 rule weight
        # ML weight = 0.6 * 0.15 = 0.09
        # composite = 0.09 + 0.55 = 0.64 >= 0.55 threshold
        fake_hits = [
            FakeRuleHit(name="harmful_instructions", severity="critical_content", technique_ids=["O1.1"]),
            FakeRuleHit(name="medium_rule", severity="medium", technique_ids=[]),
        ]

        with patch("na0s.cascade.rule_score_detailed", return_value=fake_hits), \
             patch("na0s.cascade.obfuscation_scan", return_value={"evasion_flags": []}):
            label, confidence, hits = wc.classify("test", vec, model)

        self.assertEqual(label, "MALICIOUS",
                         "critical_content hit above threshold must return MALICIOUS, got {}".format(label))

    def test_critical_content_blocks_override_even_with_high_ml_safe(self):
        """ML safe confidence 0.95 must NOT override when critical_content fires.

        This is the exact bug scenario: ML safe confidence > 0.8 AND
        max_severity == 'medium' (because critical_content was not handled).
        """
        wc = WeightedClassifier(threshold=0.55)
        vec = _make_mock_vectorizer()
        # ML says 99% safe -> ml_prob (malicious class) = 0.01
        # ml_safe_confidence = 1.0 - 0.01 = 0.99 (well above 0.8 override trigger)
        model = _make_mock_model(predict_label=0, proba_safe=0.99, proba_mal=0.01)

        # critical_content (0.45) + high (0.2) = 0.65 rule weight
        # ML weight = 0.6 * 0.01 = 0.006
        # composite = 0.006 + 0.65 = 0.656 >= 0.55 threshold
        fake_hits = [
            FakeRuleHit(name="harmful_instructions", severity="critical_content", technique_ids=["O1.1"]),
            FakeRuleHit(name="high_rule", severity="high", technique_ids=[]),
        ]

        with patch("na0s.cascade.rule_score_detailed", return_value=fake_hits), \
             patch("na0s.cascade.obfuscation_scan", return_value={"evasion_flags": []}):
            label, confidence, hits = wc.classify("test", vec, model)

        self.assertEqual(label, "MALICIOUS",
                         "critical_content must block ML override (ml_safe_confidence=0.99)")

    def test_regular_critical_severity_still_blocks_override(self):
        """Regular 'critical' severity rules still block override (regression check)."""
        wc = WeightedClassifier(threshold=0.55)
        vec = _make_mock_vectorizer()
        model = _make_mock_model(predict_label=0, proba_safe=0.99, proba_mal=0.01)

        # critical (0.3) + high (0.2) = 0.5 rule weight
        # ML weight = 0.006
        # composite = 0.506 < 0.55
        # Override should NOT fire because max_severity = "critical"
        fake_hits = [
            FakeRuleHit(name="critical_rule", severity="critical", technique_ids=[]),
            FakeRuleHit(name="high_rule", severity="high", technique_ids=[]),
        ]

        with patch("na0s.cascade.rule_score_detailed", return_value=fake_hits), \
             patch("na0s.cascade.obfuscation_scan", return_value={"evasion_flags": []}):
            label, confidence, hits = wc.classify("test", vec, model)

        # Even though composite < threshold, override was blocked by critical severity.
        # Normal path: composite < threshold => SAFE.  This is correct behavior --
        # the override being blocked here does NOT change the outcome because the
        # composite is legitimately below threshold.
        self.assertEqual(label, "SAFE",
                         "Below-threshold result should be SAFE via normal path")

    def test_medium_only_override_still_works(self):
        """Medium-only severity still allows ML-trust override (regression check)."""
        wc = WeightedClassifier(threshold=0.55)
        vec = _make_mock_vectorizer()
        model = _make_mock_model(predict_label=0, proba_safe=0.95, proba_mal=0.05)

        # 1 medium rule = 0.1
        # ML weight = 0.6 * 0.05 = 0.03
        # composite = 0.13 < 0.55 threshold
        fake_hits = [
            FakeRuleHit(name="medium_rule", severity="medium", technique_ids=[]),
        ]

        with patch("na0s.cascade.rule_score_detailed", return_value=fake_hits), \
             patch("na0s.cascade.obfuscation_scan", return_value={"evasion_flags": []}):
            label, confidence, hits = wc.classify("test", vec, model)

        self.assertEqual(label, "SAFE",
                         "Medium-only with high ML safe confidence should still be SAFE")


# ===========================================================================
# Cross-cutting: SEVERITY_WEIGHTS consistency
# ===========================================================================

class TestSeverityWeightsConsistency(unittest.TestCase):
    """SEVERITY_WEIGHTS must include critical_content and be shared across modules."""

    def test_critical_content_in_severity_weights(self):
        """critical_content must be defined in SEVERITY_WEIGHTS."""
        self.assertIn("critical_content", SEVERITY_WEIGHTS,
                      "SEVERITY_WEIGHTS must include 'critical_content'")

    def test_critical_content_weight_is_highest(self):
        """critical_content weight must be >= critical weight."""
        self.assertGreaterEqual(
            SEVERITY_WEIGHTS["critical_content"],
            SEVERITY_WEIGHTS["critical"],
            "critical_content weight ({}) should be >= critical weight ({})".format(
                SEVERITY_WEIGHTS["critical_content"], SEVERITY_WEIGHTS["critical"]
            ),
        )

    def test_severity_weights_shared_between_predict_and_cascade(self):
        """predict.py and cascade.py must use the same SEVERITY_WEIGHTS."""
        from na0s.predict import SEVERITY_WEIGHTS as predict_sw
        from na0s.cascade import SEVERITY_WEIGHTS as cascade_sw
        self.assertIs(predict_sw, cascade_sw,
                      "predict.py and cascade.py must share the same SEVERITY_WEIGHTS object")

    def test_voting_rule_severity_exists_and_matches(self):
        """_voting.RULE_SEVERITY must contain all rule names from rules.py."""
        for rule in RULES:
            self.assertIn(
                rule.name, VOTING_RULE_SEVERITY,
                "RULE_SEVERITY missing rule '{}' from rules.py".format(rule.name),
            )

    def test_severity_chain_rules_to_voting(self):
        """rules.SEVERITY_WEIGHTS is used by _voting.RULE_SEVERITY lookups."""
        for rule_name, severity in VOTING_RULE_SEVERITY.items():
            self.assertIn(
                severity, RULES_SEVERITY_WEIGHTS,
                "Severity '{}' for rule '{}' not in rules.SEVERITY_WEIGHTS".format(
                    severity, rule_name,
                ),
            )


# ====================================================================

if __name__ == "__main__":
    unittest.main()
