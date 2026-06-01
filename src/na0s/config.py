"""Centralized configuration constants for Na0S.

All tunable numeric thresholds and weights live here.
Downstream modules import from this file instead of hardcoding values.
"""

import os
from dataclasses import dataclass

# -- Input length guard (defense-in-depth, checked at scan/classify entry) --
MAX_INPUT_LENGTH: int = int(os.getenv("NA0S_MAX_INPUT_LENGTH", 50_000))

# -- OpenClaw iMessage Bridge Configuration --
OPENCLAW_MODE: str = os.getenv("OPENCLAW_MODE", "auto")  # "auto", "mock", "real"
OPENCLAW_PORT: int = int(os.getenv("OPENCLAW_PORT", 3000))  # Port for OpenClaw service
OPENCLAW_REAL_PORT: int = int(os.getenv("OPENCLAW_REAL_PORT", 3000))  # Port for real OpenClaw
OPENCLAW_MOCK_PORT: int = int(os.getenv("OPENCLAW_MOCK_PORT", 3000))  # Port for mock OpenClaw
OPENCLAW_BASE_URL: str = os.getenv(
    "OPENCLAW_BASE_URL", f"http://localhost:{OPENCLAW_PORT}"
)
OPENCLAW_TIMEOUT: int = int(os.getenv("OPENCLAW_TIMEOUT", 30))

# -- Agent approval mail-drop (cloud->local git transport) --
# GitHub Actions publishes data/approval_queue/pending_deploy.json onto this
# branch; the local daemon fetches it read-only. See na0s.agents.approvals_sync.
APPROVALS_BRANCH: str = os.getenv("NA0S_APPROVALS_BRANCH", "agent-approvals")
APPROVALS_REMOTE: str = os.getenv("NA0S_APPROVALS_REMOTE", "origin")


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

# -- Weighted Voting (Layer 6) -- flat constants for backward compat
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
