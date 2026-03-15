"""Centralized configuration constants for Na0S.

All tunable numeric thresholds and weights live here.
Downstream modules import from this file instead of hardcoding values.
"""

# -- Weighted Voting (Layer 6) --
ML_WEIGHT = 0.6
OBFUSCATION_WEIGHT_PER_FLAG = 0.15
OBFUSCATION_WEIGHT_CAP = 0.3
FALLBACK_THRESHOLD = 0.55
STRUCTURAL_SIGNAL_WEIGHTS = {
    "imperative_start": 0.05,
    "role_assignment": 0.10,
    "instruction_boundary": 0.10,
    "negation_command": 0.08,
}
AGREEMENT_BOOST = {2: 0.10, 3: 0.12, 4: 0.15}
ML_UNCERTAIN_ZONE_LOWER = 0.35
ML_UNCERTAIN_ZONE_UPPER = 0.80

# -- Output Scanner (Layer 9) --
SENSITIVITY_WEIGHTS = {"low": 0.5, "medium": 1.0, "high": 1.5}
SENSITIVITY_THRESHOLDS = {"low": 0.55, "medium": 0.35, "high": 0.20}
TRIGRAM_THRESHOLD_DEFAULT = 3

# -- Cascade --
WHITELIST_CONFIDENCE = 0.99
WHITELIST_RISK_SCORE = 0.01
PARANOID_LOWER = 0.35
PARANOID_UPPER = 0.65
