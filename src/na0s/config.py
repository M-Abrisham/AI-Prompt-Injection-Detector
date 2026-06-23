"""Centralized configuration constants for Na0S.

All tunable numeric thresholds and weights live here.
Downstream modules import from this file instead of hardcoding values.
"""

import os
from dataclasses import dataclass

from na0s._env import safe_int_env

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

# -- Agent approval sender allowlist (defense-in-depth) --
# Comma-separated iMessage handles (phones/emails) permitted to authorize a
# deploy. SECONDARY to the per-request nonce — the sender is spoofable, so it
# never bypasses the nonce; it only adds a fail-closed gate in front of it.
# Empty (the default) disables the gate, preserving today's behavior.
NA0S_AGENT_APPROVAL_ALLOWED_SENDERS: tuple[str, ...] = tuple(
    s.strip()
    for s in os.getenv("NA0S_AGENT_APPROVAL_ALLOWED_SENDERS", "").split(",")
    if s.strip()
)


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

# -- Weighted Voting (Layer 6) --
# Single source of truth: these DERIVE from the THRESHOLDS dataclass so config
# itself never carries two copies of the same value (GAP-13).  fusion/voting.py
# (the actual composite scorer for both predict.scan() and cascade) imports
# these instead of re-declaring its own copies.
ML_WEIGHT = THRESHOLDS.ML_WEIGHT
OBFUSCATION_WEIGHT_PER_FLAG = THRESHOLDS.OBFUSCATION_WEIGHT_PER_FLAG
OBFUSCATION_WEIGHT_CAP = THRESHOLDS.OBFUSCATION_WEIGHT_CAP
FALLBACK_THRESHOLD = THRESHOLDS.DEFAULT_THRESHOLD
STRUCTURAL_SIGNAL_WEIGHTS = {
    "imperative_start": 0.05,
    "role_assignment": 0.10,
    "instruction_boundary": 0.10,
    "negation_command": 0.08,
}
AGREEMENT_BOOST = {2: 0.10, 3: 0.12, 4: 0.15}
ML_UNCERTAIN_ZONE_LOWER = 0.35
ML_UNCERTAIN_ZONE_UPPER = 0.80

# -- PromptGuard (N5) transformer signal -- shared by predict.py AND cascade.py
# (was an inline 0.35/0.5/0.2 drift-pair duplicated across both entry points).
PROMPTGUARD_WEIGHT = 0.35          # blend weight for the PromptGuard score
PROMPTGUARD_HIGH_THRESHOLD = 0.5   # score above this -> "promptguard:high" hit
PROMPTGUARD_MED_THRESHOLD = 0.2    # score above this -> "promptguard:medium" hit

# -- Output Scanner (Layer 9) --
SENSITIVITY_WEIGHTS = {"low": 0.5, "medium": 1.0, "high": 1.5}
SENSITIVITY_THRESHOLDS = {"low": 0.55, "medium": 0.35, "high": 0.20}
TRIGRAM_THRESHOLD_DEFAULT = 3

# -- Cascade --
WHITELIST_CONFIDENCE = 0.99
WHITELIST_RISK_SCORE = 0.01
PARANOID_LOWER = 0.35
PARANOID_UPPER = 0.65

# -- Supply-chain integrity (Layer 11) --
# Centralizes the in-package L11 integrity knobs that were previously inlined
# in na0s.integrity.safe_pickle (ROADMAP_V2.md:1177). Defaults are carried over
# byte-for-byte from the shipped code — they are NOT re-tuned here.
#
# INTEGRITY_HASH_CHUNK_BYTES: read-batch size for the incremental SHA-256/HMAC
# hashing in safe_pickle._sha256 / _hmac_sha256. This is a pure I/O batching
# choice fed to hashlib/hmac incrementally, so it does NOT affect the resulting
# digest — changing it can never alter which files verify. The clamp is a
# guardrail, not a security threshold:
#   * lo=4096 — a 0/negative chunk would make ``iter(lambda: f.read(n), b"")``
#     spin forever (read(0) never returns the b"" sentinel); 4 KiB is the
#     smallest sane page-sized read.
#   * hi=1<<24 (16 MiB) — rejects a pathological env value that would buffer a
#     huge per-read allocation. safe_int_env falls back to the 64 KiB default
#     on any out-of-range / non-integer input.
INTEGRITY_HASH_CHUNK_BYTES: int = safe_int_env(
    "NA0S_INTEGRITY_HASH_CHUNK_BYTES", 1 << 16, lo=4096, hi=1 << 24
)

# PICKLE_SIGNING_KEY_ENV: the NAME of the env var that holds the HMAC signing
# key (the value is read from the environment at the safe_pickle trust
# boundary). This is the single source of truth for the name so the string
# "NA0S_PICKLE_KEY" is not duplicated across safe_pickle's getenv call and its
# operator-facing messages. It is intentionally a plain constant, NOT
# env-overridable: the name of the variable that holds a renamed variable is
# circular, and downstream docs/tests assert the literal "NA0S_PICKLE_KEY".
PICKLE_SIGNING_KEY_ENV: str = "NA0S_PICKLE_KEY"
