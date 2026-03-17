"""Centralized configuration — thresholds, weights, and limits.

All tunable constants live here so that:
1. A/B testing can swap configs without editing multiple files.
2. Cascade and predict pipelines share the same threshold values.
3. Constants have a single source of truth for documentation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdConfig:
    """Immutable threshold and weight configuration for the detection pipeline."""

    # --- WeightedClassifier (cascade.py Stage 2) ---
    ML_WEIGHT: float = 0.6
    OBFUSCATION_WEIGHT_PER_FLAG: float = 0.15
    OBFUSCATION_WEIGHT_CAP: float = 0.3
    DEFAULT_THRESHOLD: float = 0.55
    COMBINED_SIGNAL_BOOST: float = 0.15

    # --- CascadeClassifier LLM judge routing ---
    JUDGE_LOWER_THRESHOLD: float = 0.25
    JUDGE_UPPER_THRESHOLD: float = 0.85

    # --- Verdict blending weights (Stage 2 vs LLM judge) ---
    STAGE2_BLEND_WEIGHT: float = 0.3
    JUDGE_BLEND_WEIGHT: float = 0.7

    # --- WhitelistFilter ---
    WHITELIST_MAX_LENGTH: int = 1000
    WHITELIST_MAX_SENTENCES: int = 3


# Singleton — import this, not the class
THRESHOLDS = ThresholdConfig()
