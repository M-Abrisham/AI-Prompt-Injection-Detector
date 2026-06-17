"""Unit tests for signal_boost.py — combined signal boosting module.

Tests verify:
  1. Multi-vector attacks produce expected boosts
  2. Single-signal inputs do NOT receive boosts
  3. Safe/suppressed prompts receive zero boost
  4. Edge cases: capping, type safety, boundary conditions
  5. Integration: imports work, return types are correct

All tests are pure-unit (no .pkl files or ML model needed).
"""

import os
import sys
import unittest

# Disable thread-based scan timeout so signal.SIGALRM works in main thread.
os.environ["SCAN_TIMEOUT_SEC"] = "0"

from na0s.signal_boost import (
    calculate_boost,
    calculate_boost_from_names,
    MAX_BOOST,
    SIGNAL_COMBOS,
    _ENCODING_FLAGS,
    _PERSONA_HIJACK_RULES,
    _OVERRIDE_AUTHORITY_RULES,
    _SYSTEM_EXTRACTION_RULES,
)
from na0s.layer1.result import RuleHit


class TestCalculateBoostMultiVector(unittest.TestCase):
    """Multi-vector attacks MUST produce a boost."""

    def test_persona_hijack_plus_base64(self):
        """Persona hijack rule + base64 flag should boost by 0.12."""
        hits = [RuleHit(name="roleplay", severity="high")]
        flags = ["base64"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.12, places=4)
        self.assertEqual(len(reasons), 1)
        self.assertIn("roleplay", reasons[0])
        self.assertIn("base64", reasons[0])

    def test_override_plus_hex(self):
        """Override rule + hex flag should boost by 0.12."""
        hits = [RuleHit(name="override", severity="critical")]
        flags = ["hex"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.12, places=4)
        self.assertEqual(len(reasons), 1)

    def test_system_extraction_plus_rot13(self):
        """System extraction rule + rot13 flag should boost by 0.08."""
        hits = [RuleHit(name="system_prompt", severity="high")]
        flags = ["rot13"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.08, places=4)

    def test_roleplay_plus_leetspeak(self):
        """Roleplay rule + leetspeak flag should boost by 0.12."""
        hits = [RuleHit(name="roleplay", severity="high")]
        flags = ["leetspeak"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.12, places=4)

    def test_two_encoding_flags_base64_hex(self):
        """Two encoding flags (base64 + hex) should boost by 0.10."""
        hits = [RuleHit(name="override", severity="critical")]
        flags = ["base64", "hex"]
        score, reasons = calculate_boost(hits, flags)
        # Should get 0.12 (override+base64) + 0.12 (override+hex) + 0.10 (base64+hex)
        # But capped at MAX_BOOST = 0.30
        self.assertLessEqual(score, MAX_BOOST)
        self.assertGreater(score, 0.12)

    def test_three_signals_persona_base64_high_entropy(self):
        """Persona + base64 + high_entropy should produce combined boost."""
        hits = [RuleHit(name="persona_split", severity="high")]
        flags = ["base64", "high_entropy"]
        score, reasons = calculate_boost(hits, flags)
        # persona_split+base64 = 0.12, persona_split+high_entropy already covered by 0.12
        # base64+high_entropy is not an encoding+encoding combo (high_entropy is not in _ENCODING_FLAGS)
        self.assertGreater(score, 0.12)
        self.assertLessEqual(score, MAX_BOOST)

    def test_authority_escalation_plus_url_encoded(self):
        """Authority escalation + url_encoded should boost by 0.12."""
        hits = [RuleHit(name="authority_escalation", severity="critical")]
        flags = ["url_encoded"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.12, places=4)

    def test_forget_override_plus_reversed_text(self):
        """Forget override + reversed_text should boost by 0.12."""
        hits = [RuleHit(name="forget_override", severity="critical")]
        flags = ["reversed_text"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.12, places=4)

    def test_developer_mode_plus_rot13(self):
        """Developer mode (persona hijack) + rot13 should boost by 0.12."""
        hits = [RuleHit(name="developer_mode", severity="critical")]
        flags = ["rot13"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.12, places=4)

    def test_decode_and_execute_plus_base64(self):
        """Decode-and-execute rule + base64 should boost."""
        hits = [RuleHit(name="decode_and_execute", severity="high")]
        flags = ["base64"]
        score, reasons = calculate_boost(hits, flags)
        self.assertGreater(score, 0.0)

    def test_direct_prompt_request_plus_weird_casing(self):
        """System extraction rule + weird_casing (obfuscation) should boost by 0.08."""
        hits = [RuleHit(name="direct_prompt_request", severity="critical")]
        flags = ["weird_casing"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.08, places=4)

    def test_multi_encoding_only_flags(self):
        """Two encoding flags without any rule should still boost (multi-encoding)."""
        hits = []
        flags = ["base64", "hex"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.10, places=4)
        self.assertEqual(len(reasons), 1)

    def test_constraint_negation_plus_full_reverse(self):
        """Constraint negation + full_reverse should boost by 0.12."""
        hits = [RuleHit(name="constraint_negation", severity="critical")]
        flags = ["full_reverse"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(score, 0.12, places=4)


class TestCalculateBoostSingleSignal(unittest.TestCase):
    """Single-signal inputs MUST NOT receive a boost."""

    def test_override_rule_only_no_encoding(self):
        """Override rule only, no encoding flags -> zero boost."""
        hits = [RuleHit(name="override", severity="critical")]
        flags = []
        score, reasons = calculate_boost(hits, flags)
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])

    def test_base64_flag_only_no_rules(self):
        """Base64 flag only, no rules -> zero boost."""
        hits = []
        flags = ["base64"]
        score, reasons = calculate_boost(hits, flags)
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])

    def test_only_high_entropy_flag(self):
        """Only high_entropy flag, no rules -> zero boost."""
        hits = []
        flags = ["high_entropy"]
        score, reasons = calculate_boost(hits, flags)
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])

    def test_multiple_rules_no_flags(self):
        """Multiple rules but no flags -> zero boost."""
        hits = [
            RuleHit(name="override", severity="critical"),
            RuleHit(name="roleplay", severity="high"),
        ]
        flags = []
        score, reasons = calculate_boost(hits, flags)
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])

    def test_rule_not_in_any_category(self):
        """Rule that is not in any signal combo category -> zero boost."""
        hits = [RuleHit(name="secrecy", severity="medium")]
        flags = ["punctuation_flood"]
        score, reasons = calculate_boost(hits, flags)
        # "secrecy" is not in any category, "punctuation_flood" is not an encoding flag
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])


class TestCalculateBoostSafePrompts(unittest.TestCase):
    """Safe/suppressed prompts MUST NOT receive a boost."""

    def test_context_suppressed_true(self):
        """context_suppressed=True should always return zero boost."""
        hits = [RuleHit(name="override", severity="critical")]
        flags = ["base64"]
        score, reasons = calculate_boost(hits, flags, context_suppressed=True)
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])

    def test_empty_rules_empty_flags(self):
        """Empty rules + empty flags -> zero boost."""
        score, reasons = calculate_boost([], [])
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])

    def test_none_rules_none_flags(self):
        """None inputs handled gracefully -> zero boost."""
        score, reasons = calculate_boost(None, None)
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])

    def test_none_rules_with_flags(self):
        """None rules with valid flags -> zero boost."""
        score, reasons = calculate_boost(None, ["base64"])
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])

    def test_rules_with_none_flags(self):
        """Valid rules with None flags -> zero boost."""
        hits = [RuleHit(name="override", severity="critical")]
        score, reasons = calculate_boost(hits, None)
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])


class TestCalculateBoostEdgeCases(unittest.TestCase):
    """Edge cases: capping, boundaries, type handling."""

    def test_many_rules_many_flags_capped_at_max(self):
        """Many rules + many flags -> bounded by MAX_BOOST with honest reasons.

        New contract (post-explainability fix): boost_score is bounded by
        MAX_BOOST AND the weights embedded in boost_reasons must sum to
        exactly boost_score.  The old contract let reasons exceed the cap,
        which broke audit-log attribution.
        """
        import re
        hits = [
            RuleHit(name="override", severity="critical"),
            RuleHit(name="roleplay", severity="high"),
            RuleHit(name="system_prompt", severity="high"),
            RuleHit(name="forget_override", severity="critical"),
            RuleHit(name="persona_split", severity="high"),
        ]
        flags = ["base64", "hex", "rot13", "leetspeak"]
        score, reasons = calculate_boost(hits, flags)
        self.assertLessEqual(score, MAX_BOOST)
        self.assertGreater(len(reasons), 0)
        # Explainability invariant: reasons sum to score
        weights = [float(m.group(1)) for m in re.finditer(r'\+(\d\.\d\d)\)', " ".join(reasons))]
        self.assertAlmostEqual(sum(weights), score, places=4,
                               msg="reasons weights must sum to boost_score")

    def test_boost_at_score_boundary(self):
        """Verify boost value can push a near-miss composite over threshold.

        If composite = 0.54 and boost = 0.12, result should be 0.66.
        The boost module itself just computes the boost value; this test
        verifies the boost value is correct for the scenario.
        """
        hits = [RuleHit(name="override", severity="critical")]
        flags = ["base64"]
        score, reasons = calculate_boost(hits, flags)
        composite_before = 0.54
        composite_after = composite_before + score
        self.assertAlmostEqual(composite_after, 0.66, places=4)
        self.assertGreater(composite_after, 0.55)  # threshold

    def test_max_boost_constant_value(self):
        """MAX_BOOST must be 0.3."""
        self.assertEqual(MAX_BOOST, 0.3)

    def test_signal_combos_not_empty(self):
        """SIGNAL_COMBOS dict must have entries."""
        self.assertGreater(len(SIGNAL_COMBOS), 0)

    def test_signal_combos_values_in_range(self):
        """All SIGNAL_COMBOS values must be between 0.05 and 0.15."""
        for combo, weight in SIGNAL_COMBOS.items():
            self.assertGreaterEqual(weight, 0.05, msg=str(combo))
            self.assertLessEqual(weight, 0.15, msg=str(combo))

    def test_signal_combos_keys_are_frozensets_of_two(self):
        """All SIGNAL_COMBOS keys must be frozensets of exactly 2 elements."""
        for combo in SIGNAL_COMBOS:
            self.assertIsInstance(combo, frozenset)
            self.assertEqual(len(combo), 2, msg=str(combo))

    def test_duplicate_flags_do_not_double_count(self):
        """Duplicate encoding flags in the list should not produce extra boost."""
        hits = [RuleHit(name="override", severity="critical")]
        flags_dedup = ["base64"]
        flags_dup = ["base64", "base64"]
        score1, _ = calculate_boost(hits, flags_dedup)
        score2, _ = calculate_boost(hits, flags_dup)
        # With deduplication handled by frozenset key matching, both should be same
        self.assertAlmostEqual(score1, score2, places=4)


class TestCalculateBoostFromNames(unittest.TestCase):
    """Tests for the string-based convenience wrapper."""

    def test_from_names_persona_plus_base64(self):
        """calculate_boost_from_names with persona + base64 should boost."""
        score, reasons = calculate_boost_from_names(
            ["roleplay"], ["base64"],
        )
        self.assertAlmostEqual(score, 0.12, places=4)

    def test_from_names_override_plus_hex(self):
        """calculate_boost_from_names with override + hex should boost."""
        score, reasons = calculate_boost_from_names(
            ["override"], ["hex"],
        )
        self.assertAlmostEqual(score, 0.12, places=4)

    def test_from_names_empty(self):
        """calculate_boost_from_names with empty inputs -> zero."""
        score, reasons = calculate_boost_from_names([], [])
        self.assertEqual(score, 0.0)

    def test_from_names_none_rules(self):
        """calculate_boost_from_names with None rules -> zero."""
        score, reasons = calculate_boost_from_names(None, ["base64"])
        self.assertEqual(score, 0.0)

    def test_from_names_context_suppressed(self):
        """calculate_boost_from_names with context_suppressed -> zero."""
        score, reasons = calculate_boost_from_names(
            ["override"], ["base64"], context_suppressed=True,
        )
        self.assertEqual(score, 0.0)

    def test_from_names_system_prompt_plus_high_entropy(self):
        """System prompt + high_entropy via names should boost by 0.08."""
        score, reasons = calculate_boost_from_names(
            ["system_prompt"], ["high_entropy"],
        )
        # system_prompt is in _SYSTEM_EXTRACTION_RULES, high_entropy is in _OBFUSCATION_FLAGS
        self.assertAlmostEqual(score, 0.08, places=4)


class TestCalculateBoostIntegration(unittest.TestCase):
    """Integration tests: imports, types, module-level constants."""

    def test_import_from_na0s_signal_boost(self):
        """Verify the module can be imported from na0s package."""
        from na0s.signal_boost import calculate_boost as cb
        self.assertTrue(callable(cb))

    def test_return_types(self):
        """calculate_boost must return (float, list)."""
        hits = [RuleHit(name="override", severity="critical")]
        flags = ["base64"]
        result = calculate_boost(hits, flags)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], list)

    def test_return_types_zero_case(self):
        """Return types must be correct even for zero-boost case."""
        result = calculate_boost([], [])
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], list)

    def test_boost_reasons_are_strings(self):
        """Each element in boost_reasons must be a string."""
        hits = [RuleHit(name="override", severity="critical")]
        flags = ["base64"]
        _, reasons = calculate_boost(hits, flags)
        for reason in reasons:
            self.assertIsInstance(reason, str)

    def test_boost_reasons_contain_signal_boost_prefix(self):
        """Boost reason strings must start with 'signal_boost:'."""
        hits = [RuleHit(name="override", severity="critical")]
        flags = ["base64"]
        _, reasons = calculate_boost(hits, flags)
        for reason in reasons:
            self.assertTrue(
                reason.startswith("signal_boost:"),
                msg="Reason does not start with 'signal_boost:': {}".format(reason),
            )

    def test_encoding_flags_constant_has_expected_members(self):
        """_ENCODING_FLAGS should contain base64, hex, rot13, leetspeak."""
        for flag in ("base64", "hex", "rot13", "leetspeak"):
            self.assertIn(flag, _ENCODING_FLAGS)

    def test_persona_hijack_rules_constant(self):
        """_PERSONA_HIJACK_RULES should contain roleplay and persona_split."""
        self.assertIn("roleplay", _PERSONA_HIJACK_RULES)
        self.assertIn("persona_split", _PERSONA_HIJACK_RULES)

    def test_override_authority_rules_constant(self):
        """_OVERRIDE_AUTHORITY_RULES should contain override and authority_escalation."""
        self.assertIn("override", _OVERRIDE_AUTHORITY_RULES)
        self.assertIn("authority_escalation", _OVERRIDE_AUTHORITY_RULES)

    def test_mixed_rulehit_and_string_inputs(self):
        """calculate_boost should handle a mix of RuleHit objects and strings."""
        hits = [
            RuleHit(name="override", severity="critical"),
            "roleplay",
        ]
        flags = ["base64"]
        score, reasons = calculate_boost(hits, flags)
        # Should find both override+base64 (0.12) and roleplay+base64 (0.12) -> capped
        self.assertGreater(score, 0.12)
        self.assertLessEqual(score, MAX_BOOST)


class TestExplainabilityInvariant(unittest.TestCase):
    """Reasons weights must sum to boost_score (contract guarantee).

    Regression tests for the "explainability leak" bug where the old code
    capped boost_score but not boost_reasons, so parse(reasons) != score.
    """

    @staticmethod
    def _sum_reason_weights(reasons):
        import re
        weights = [float(m.group(1)) for m in re.finditer(r'\+(\d\.\d\d)\)', " ".join(reasons))]
        return sum(weights)

    def test_reasons_sum_equals_score_under_cap(self):
        """Uncapped: reasons weights sum exactly to score."""
        hits = [RuleHit(name="roleplay", severity="high")]
        flags = ["base64"]
        score, reasons = calculate_boost(hits, flags)
        self.assertAlmostEqual(self._sum_reason_weights(reasons), score, places=4)

    def test_reasons_sum_equals_score_at_cap(self):
        """At saturation: reasons are truncated so sum still equals score."""
        hits = [
            RuleHit(name="roleplay"),
            RuleHit(name="override"),
            RuleHit(name="system_prompt"),
            RuleHit(name="decode_and_execute"),
        ]
        flags = ["base64", "hex", "rot13", "leetspeak"]
        score, reasons = calculate_boost(hits, flags)
        self.assertLessEqual(score, MAX_BOOST)
        self.assertGreater(len(reasons), 0)
        self.assertAlmostEqual(self._sum_reason_weights(reasons), score, places=4)

    def test_reasons_empty_when_score_zero(self):
        """Score=0 implies reasons=[]; no orphaned reason strings."""
        score, reasons = calculate_boost([], [])
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])


class TestMultiEncodingGuard(unittest.TestCase):
    """The multi-encoding guard must count encoding flags, not raw list length."""

    def test_two_non_encoding_flags_do_not_trigger_multi_encoding(self):
        """['high_entropy','punctuation_flood'] has len 2 but zero encodings."""
        score, reasons = calculate_boost([], ["high_entropy", "punctuation_flood"])
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])

    def test_one_encoding_plus_one_non_encoding(self):
        """['base64','high_entropy'] still has only 1 encoding flag."""
        score, reasons = calculate_boost([], ["base64", "high_entropy"])
        self.assertEqual(score, 0.0)

    def test_three_encoding_flags_all_counted(self):
        """Three encoding flags produce C(3,2)=3 pair boosts."""
        score, reasons = calculate_boost([], ["base64", "hex", "rot13"])
        self.assertAlmostEqual(score, 0.30, places=4)
        self.assertEqual(len(reasons), 3)


class TestUnknownTypeDropLogging(unittest.TestCase):
    """Unknown types in rule_hits must be logged at DEBUG, not silently dropped."""

    def test_dict_entry_logged(self):
        """A dict in rule_hits should be dropped AND logged."""
        import logging as _logging
        with self.assertLogs("na0s.fusion.signal_boost", level="DEBUG") as cm:
            calculate_boost(
                [RuleHit(name="override"), {"not_a_rule": True}],
                ["base64"],
            )
        dropped_logs = [r for r in cm.records if "dropped" in r.getMessage()]
        self.assertEqual(len(dropped_logs), 1)
        self.assertIn("1", dropped_logs[0].getMessage())

    def test_none_entry_logged(self):
        """None in rule_hits should be dropped AND logged."""
        with self.assertLogs("na0s.fusion.signal_boost", level="DEBUG") as cm:
            calculate_boost(
                ["override", None],
                ["base64"],
            )
        dropped_logs = [r for r in cm.records if "dropped" in r.getMessage()]
        self.assertEqual(len(dropped_logs), 1)

    def test_mixed_unknown_types_all_counted(self):
        """Multiple unknown types are counted, valid entries still score."""
        score, reasons = calculate_boost(
            ["override", None, 42, {"x": 1}, "roleplay"],
            ["base64"],
        )
        # Both override+base64 and roleplay+base64 fire (0.12 each) = 0.24
        self.assertAlmostEqual(score, 0.24, places=4)
        self.assertEqual(len(reasons), 2)


class TestNoneSymmetry(unittest.TestCase):
    """None inputs are normalized to []; no asymmetric behavior."""

    def test_none_rule_hits_with_multi_encoding(self):
        """(None, [b64, hex]) should equal ([], [b64, hex]) after normalization."""
        a = calculate_boost(None, ["base64", "hex"])
        b = calculate_boost([], ["base64", "hex"])
        self.assertEqual(a, b)
        self.assertAlmostEqual(a[0], 0.10, places=4)

    def test_empty_and_none_flags_equivalent(self):
        """([override], None) == ([override], [])."""
        hits = [RuleHit(name="override")]
        self.assertEqual(calculate_boost(hits, None), calculate_boost(hits, []))


class TestSignalCombosFrozen(unittest.TestCase):
    """SIGNAL_COMBOS must be read-only (MappingProxyType)."""

    def test_mutation_raises_type_error(self):
        """Assigning to SIGNAL_COMBOS should raise TypeError."""
        with self.assertRaises(TypeError):
            SIGNAL_COMBOS[frozenset({"x", "y"})] = 0.99  # type: ignore

    def test_deletion_raises_type_error(self):
        """Deleting from SIGNAL_COMBOS should raise TypeError."""
        key = next(iter(SIGNAL_COMBOS))
        with self.assertRaises(TypeError):
            del SIGNAL_COMBOS[key]  # type: ignore


class TestLoadTimeInvariants(unittest.TestCase):
    """Load-time assertions that prevent structural regressions."""

    def test_categories_are_pairwise_disjoint(self):
        """No rule name may belong to more than one category."""
        cats = {
            "persona": _PERSONA_HIJACK_RULES,
            "override": _OVERRIDE_AUTHORITY_RULES,
            "system": _SYSTEM_EXTRACTION_RULES,
        }
        from itertools import combinations
        for (na, a), (nb, b) in combinations(cats.items(), 2):
            with self.subTest(pair=(na, nb)):
                self.assertEqual(a & b, frozenset(),
                                 "categories {0} x {1} overlap".format(na, nb))

    def test_no_rule_name_collides_with_flag(self):
        """Rule names must not match any encoding/obfuscation flag name."""
        all_rules = (
            _PERSONA_HIJACK_RULES
            | _OVERRIDE_AUTHORITY_RULES
            | _SYSTEM_EXTRACTION_RULES
        )
        self.assertEqual(all_rules & _ENCODING_FLAGS, frozenset())


class TestRuleCoverage(unittest.TestCase):
    """Visibility into the 'silent opt-out' gap for new rules."""

    def test_get_uncovered_rules_is_list_of_strings(self):
        """get_uncovered_rules() returns a (possibly empty) sorted list of strings."""
        from na0s.signal_boost import get_uncovered_rules
        result = get_uncovered_rules()
        self.assertIsInstance(result, list)
        self.assertEqual(result, sorted(result))
        for name in result:
            self.assertIsInstance(name, str)

    def test_uncovered_rule_contributes_zero(self):
        """A rule not in any category contributes zero even with encoding flag."""
        # Pick an uncovered rule (these exist per get_uncovered_rules())
        from na0s.signal_boost import get_uncovered_rules
        uncovered = get_uncovered_rules()
        if not uncovered:  # pragma: no cover - if coverage ever hits 100%
            self.skipTest("no uncovered rules to test")
        score, reasons = calculate_boost([uncovered[0]], ["base64"])
        # Multi-encoding pass should not fire either since only 1 flag
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
