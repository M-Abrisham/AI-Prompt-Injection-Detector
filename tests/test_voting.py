"""Unit tests for na0s._voting shared primitives."""

from na0s._voting import _weighted_composite


class TestWeightedComposite:
    """Tests for _weighted_composite()."""

    def test_equal_weights_arithmetic(self):
        """Equal ML, rule, and obf signals should produce additive sum."""
        result = _weighted_composite(
            ml_prob_malicious=0.5,
            ml_weight=0.6,
            rule_weight=0.3,
            obf_weight=0.15,
        )
        expected = (0.6 * 0.5) + 0.3 + 0.15  # = 0.30 + 0.30 + 0.15 = 0.75
        assert abs(result - expected) < 1e-9

    def test_zero_total_weight_returns_zero(self):
        """All zeros should return 0.0 (no divide by zero)."""
        result = _weighted_composite(
            ml_prob_malicious=0.0,
            ml_weight=0.0,
            rule_weight=0.0,
            obf_weight=0.0,
        )
        assert result == 0.0

    def test_ml_dominant(self):
        """When ML weight is high and prob is 1.0, output reflects ML signal."""
        result = _weighted_composite(
            ml_prob_malicious=1.0,
            ml_weight=0.6,
            rule_weight=0.0,
            obf_weight=0.0,
        )
        assert abs(result - 0.6) < 1e-9

    def test_rules_only(self):
        """When only rules fire, output equals rule_weight."""
        result = _weighted_composite(
            ml_prob_malicious=0.0,
            ml_weight=0.6,
            rule_weight=0.4,
            obf_weight=0.0,
        )
        assert abs(result - 0.4) < 1e-9

    def test_additive_not_normalized(self):
        """Output can exceed 1.0 — callers handle clamping."""
        result = _weighted_composite(
            ml_prob_malicious=1.0,
            ml_weight=0.6,
            rule_weight=0.5,
            obf_weight=0.3,
        )
        expected = 0.6 + 0.5 + 0.3  # = 1.4
        assert abs(result - expected) < 1e-9
        assert result > 1.0  # Proves no internal clamping
