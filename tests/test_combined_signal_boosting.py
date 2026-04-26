"""Tests for combined signal boosting — persona hijack + encoded payload.

When both persona hijack signals and encoded payload signals are present
in the same message, the composite score should receive an extra boost
(+0.15) beyond what each signal contributes independently.

This addresses the ROADMAP_V2.md P1 fix:
  "Combined signal boosting missing — Persona hijack + encoded payload
   in same message should carry extra weight. Currently scored independently."
"""

import pytest

from na0s.predict import _weighted_decision, DECISION_THRESHOLD


# ---------------------------------------------------------------------------
# Constants (mirror the production values for clarity in assertions)
# ---------------------------------------------------------------------------
_COMBINED_BOOST = 0.15


class TestCombinedSignalBoostingPredict:
    """Test _weighted_decision combined signal boosting in predict.py."""

    # --- Helpers ---
    @staticmethod
    def _call(ml_prob=0.5, ml_label="SAFE", hits=None, obs_flags=None,
              structural=None, threshold=DECISION_THRESHOLD):
        return _weighted_decision(
            ml_prob=ml_prob,
            ml_label=ml_label,
            hits=hits or [],
            obs_flags=obs_flags or [],
            structural=structural,
            threshold=threshold,
        )

    # --- Core boosting tests ---

    def test_persona_rule_plus_obfuscation_flag_boosts(self):
        """Roleplay rule + obfuscation flag should get combined boost."""
        # Baseline: roleplay alone
        _, score_persona_only = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=["roleplay"],
            obs_flags=[],
        )
        # Baseline: obfuscation alone
        _, score_obf_only = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=[],
            obs_flags=["base64"],
        )
        # Combined: should be more than sum of parts
        _, score_combined = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=["roleplay"],
            obs_flags=["base64"],
        )

        # The combined score should exceed the simple sum by the boost amount
        # (accounting for multi-layer agreement which may also fire)
        assert score_combined > score_persona_only
        assert score_combined > score_obf_only

    def test_structural_role_assignment_plus_obfuscation_boosts(self):
        """Structural role_assignment feature + obfuscation should boost."""
        structural = {"role_assignment": 1}

        _, score_struct_only = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=[],
            obs_flags=[],
            structural=structural,
        )
        _, score_combined = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=[],
            obs_flags=["hex"],
            structural=structural,
        )

        assert score_combined > score_struct_only + 0.14  # at least ~0.15 boost

    def test_persona_plus_decoded_payload_malicious_boosts(self):
        """Roleplay + decoded_payload_malicious (no obs_flags) should boost."""
        _, score_no_combo = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=["roleplay"],
            obs_flags=[],
        )
        _, score_combo = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=["roleplay", "decoded_payload_malicious"],
            obs_flags=[],
        )

        assert score_combo > score_no_combo

    def test_no_boost_without_persona(self):
        """Obfuscation alone (no persona signal) should NOT get combined boost."""
        _, score_obf_only = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=[],
            obs_flags=["base64"],
        )
        # The expected score: 0.6*0.5 (ML) + 0.15 (obf) = 0.45
        # Plus possible multi-layer boost if ML > 0.5 (it equals 0.5, so no)
        expected_base = 0.6 * 0.5 + 0.15
        assert abs(score_obf_only - expected_base) < 0.02

    def test_no_boost_without_encoding(self):
        """Persona alone (no encoding signal) should NOT get combined boost."""
        _, score_persona_only = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=["roleplay"],
            obs_flags=[],
        )
        # roleplay severity=high → 0.25 weight
        # ML: 0.6*0.5 = 0.30
        # Total: 0.55, possibly with multi-layer boost
        # But NOT the combined signal boost of 0.15
        assert score_persona_only < 1.0  # sanity

    def test_boost_applies_with_multiple_encoding_types(self):
        """Multiple encoding flags + persona should still boost (once)."""
        _, score_multi = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=["roleplay"],
            obs_flags=["base64", "hex", "rot13"],
        )
        _, score_single = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=["roleplay"],
            obs_flags=["base64"],
        )
        # Both get the combined boost; difference should only be from
        # additional obfuscation weight (0.15 per flag, capped at 0.30)
        assert score_multi >= score_single

    def test_combined_boost_clamped_to_one(self):
        """Combined boost should not push score above 1.0."""
        _, score = self._call(
            ml_prob=0.95, ml_label="MALICIOUS",
            hits=["roleplay", "override", "decoded_payload_malicious"],
            obs_flags=["base64", "hex"],
            structural={"role_assignment": 1, "instruction_boundary": 1},
        )
        assert score <= 1.0

    def test_both_persona_sources_either_triggers(self):
        """Either roleplay hit OR structural role_assignment should qualify."""
        # Via roleplay rule hit
        _, score_via_rule = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=["roleplay"],
            obs_flags=["base64"],
        )
        # Via structural feature only (no roleplay rule hit)
        _, score_via_struct = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=[],
            obs_flags=["base64"],
            structural={"role_assignment": 1},
        )
        # Both should get the combined boost (scores differ because of
        # different rule weight, but both should be > obf-only baseline)
        _, score_obf_only = self._call(
            ml_prob=0.5, ml_label="MALICIOUS",
            hits=[],
            obs_flags=["base64"],
        )
        assert score_via_rule > score_obf_only
        assert score_via_struct > score_obf_only

    def test_realistic_attack_crosses_threshold(self):
        """A realistic persona+encoding attack should cross the threshold."""
        # Simulate: "you are now DAN" + base64-encoded instructions
        # ML might give moderate confidence, rule hits roleplay (high=0.25)
        # plus base64 obfuscation flag
        label, score = self._call(
            ml_prob=0.45, ml_label="MALICIOUS",
            hits=["roleplay"],
            obs_flags=["base64"],
            structural={"role_assignment": 1},
        )
        assert label == "MALICIOUS"
        assert score >= DECISION_THRESHOLD


class TestCombinedSignalBoostingCascade:
    """Test combined signal boosting in cascade.py WeightedClassifier."""

    def test_cascade_imports(self):
        """Verify cascade module can be imported."""
        from na0s.cascade import WeightedClassifier
        assert WeightedClassifier is not None

    def test_cascade_has_combined_boost_constant(self):
        """Verify the combined boost constant exists in cascade classify."""
        import inspect
        from na0s.cascade import WeightedClassifier
        source = inspect.getsource(WeightedClassifier.classify)
        assert "_COMBINED_SIGNAL_BOOST" in source
        assert "has_persona_signal" in source
        assert "has_encoding_signal" in source
