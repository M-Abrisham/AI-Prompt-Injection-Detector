# PHASE A — Safe extraction of shared weighted composite math.
# Both callers (predict.py._weighted_decision and
# cascade.py.WeightedClassifier.classify) use this shared helper
# for the ML+rule+obf composite calculation only.
#
# PHASE B (deferred) — cascade.py alignment items:
#   - Add structural feature support (predict.py has 10+ features, cascade has 0)
#   - Add multi-layer agreement boost (predict.py only)
#   - Add critical-content floor: max(composite, 0.60) (predict.py only)
#   - Align override protection: set comparison vs string comparison
#   - Align score clamping position (end vs immediate)
# Phase B requires dedicated testing wave — do not implement without approval.

"""Shared voting primitives for the Na0S detection pipeline.

Pure-math helpers — no I/O, no model calls, no na0s module imports.
"""


def _weighted_composite(
    ml_prob_malicious: float,
    ml_weight: float,
    rule_weight: float,
    obf_weight: float,
) -> float:
    """Compute additive weighted composite from ML, rule, and obfuscation signals.

    This is the shared arithmetic core used by both predict.py's
    _weighted_decision() and cascade.py's WeightedClassifier.classify().

    The formula is ADDITIVE (not normalized):
        composite = (ml_weight * ml_prob_malicious) + rule_weight + obf_weight

    Callers are responsible for:
      - Converting ML probability to the malicious-probability axis
      - Computing rule_weight from severity weights
      - Computing obf_weight from obfuscation flags (capped)
      - Adding structural features, boosts, overrides, and clamping AFTER

    Args:
        ml_prob_malicious: ML probability of malicious class (0.0-1.0)
        ml_weight:  multiplier for ML signal (typically 0.6)
        rule_weight: accumulated rule severity score (from SEVERITY_WEIGHTS)
        obf_weight: obfuscation signal weight (typically capped at 0.3)

    Returns:
        Raw additive composite (unclamped, unboosted).
    """
    return (ml_weight * ml_prob_malicious) + rule_weight + obf_weight
