"""Layer 16 Multi-Turn Detection Configuration.

All thresholds, timeouts, and feature flags. Every magic number
in Layer 16 lives here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_SIZE = 10
DEFAULT_TTL_SECONDS = 1800  # 30 minutes
MAX_SESSIONS = 10000
SESSION_CLEANUP_INTERVAL = 300  # 5 minutes (lazy cleanup)

# ---------------------------------------------------------------------------
# Escalation Detection (C1.1, C1MT)
# ---------------------------------------------------------------------------

ESCALATION_MIN_TURNS = 3  # need at least 3 turns to detect trend
ESCALATION_SLOPE_THRESHOLD = 0.15  # minimum risk increase per turn
ESCALATION_CONFIDENCE_MIN = 0.6
ESCALATION_RECENT_WEIGHT = 0.7  # weight for recent turns vs full history

# ---------------------------------------------------------------------------
# Payload Splitting (D7.2)
# ---------------------------------------------------------------------------

ASSEMBLY_WINDOW = 5  # concatenate last N turns for assembly check
ASSEMBLY_RISK_THRESHOLD = 0.8
ASSEMBLY_RISK_GAP_THRESHOLD = 0.3  # min gap between combined and individual risk
ASSEMBLY_CONFIDENCE_MIN = 0.5  # min confidence to generate alert
ASSEMBLY_MAX_CANDIDATES = 5  # cap on re-scan candidates per turn

# Borderline payload-assembly path: a fragmented payload can reassemble to a
# clearly-elevated risk that still sits just *below* the malicious line (the
# split defeats single-rule firing, and after the embedding-double-count fix
# embedding is confirmatory only, so it no longer nudges these over). Such an
# assembly is still flagged when it (a) clears a floor set this far BELOW the
# active decision threshold — far above the benign-reassembly baseline (benign
# multi-turn scenarios in tests/conversation/fixtures/payload_split_rescan_benign.json
# reassemble to <=0.06), (b) shows a positive risk jump over its individual
# turns, and (c) carries explicit assembly cues. Expressed as a margin below
# the decision threshold (not an absolute) so a raised operator threshold also
# tightens this path, consistent with the fusion rule-anchor floors. At the
# default 0.55 threshold the floor is 0.45 (a ~7x margin over benign).
# Calibration parameter — revisit if the benign corpus widens.
ASSEMBLY_BORDERLINE_MARGIN = 0.10
FRAGMENT_MARKERS = [
    "remember this", "add to", "continue from", "combine",
    "piece together", "concatenate", "merge", "join",
    "put together", "assemble",
]

# ---------------------------------------------------------------------------
# Fabricated History
# ---------------------------------------------------------------------------

FABRICATED_TURN_MARKER_THRESHOLD = 6  # min User:/Assistant: pairs
FABRICATED_KEYWORDS = [
    "conversation history", "previous chat", "transcript",
    "as we discussed", "our earlier conversation", "chat log",
    "continuing from", "here's what we said",
]

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SQLITE_DB_PATH = "na0s_sessions.db"
REDIS_URL = "redis://localhost:6379/0"

# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------

ENABLE_MULTI_TURN = True  # master switch
ENABLE_ESCALATION = True
ENABLE_PAYLOAD_SPLITTING = True
ENABLE_FABRICATED_HISTORY = True
ENABLE_CONTEXT_POISONING = True
ENABLE_BEHAVIORAL_STYLOMETRY = True
ENABLE_ALERT_DEDUP = True  # deduplicate repeated alert_type within a session

# ---------------------------------------------------------------------------
# Alert Deduplication
# ---------------------------------------------------------------------------

ALERT_SUPPRESSION_TURNS = 3  # suppress duplicate alert_type for N turns after firing

# ---------------------------------------------------------------------------
# Context Poisoning (D1.20)
# ---------------------------------------------------------------------------

POISONING_MIN_TURNS = 3
POISONING_CONFIDENCE_MIN = 0.25
POISONING_MULTI_SIGNAL_BOOST = 1.5

# ---------------------------------------------------------------------------
# Behavioral Stylometry (D1.21)
# ---------------------------------------------------------------------------

STYLOMETRY_MIN_TURNS = 3
STYLOMETRY_CONFIDENCE_MIN = 0.25
STYLOMETRY_MULTI_SIGNAL_BOOST = 1.5

# ---------------------------------------------------------------------------
# Embedding Drift Detection (D1.23)
# ---------------------------------------------------------------------------

ENABLE_EMBEDDING_DRIFT = True
DRIFT_MIN_TURNS = 3  # need at least 3 turns to measure drift
DRIFT_WINDOW = 5  # last N turns to analyze
DRIFT_SHARP_THRESHOLD = 0.3  # cosine sim below this = sharp topic pivot
DRIFT_AVG_THRESHOLD = 0.5  # average cosine sim below this = gradual drift
DRIFT_CONFIDENCE_MIN = 0.3

# ---------------------------------------------------------------------------
# Chain-of-Thought Compliance (D1.23)
# ---------------------------------------------------------------------------

ENABLE_COT_COMPLIANCE = True
COT_COMPLIANCE_MIN_TURNS = 2
COT_COMPLIANCE_CONFIDENCE_MIN = 0.3
COT_COMPLIANCE_MULTI_SIGNAL_BOOST = 1.5

# ---------------------------------------------------------------------------
# Goal Decomposition / Kill-Chain Coverage (T1.3 / IM3.x — GTG-1002)
# ---------------------------------------------------------------------------

ENABLE_GOAL_DECOMPOSITION = True

# Distinct kill-chain phases that must be covered in one session before a
# kill-chain-coverage alert is considered. Default 4 (e.g. PERSONA + RECON +
# FINGERPRINT/VULN_MAP + a TERMINAL phase). Set deliberately so a clean
# recon->fingerprint->vuln-map (a benign in-scope pentest, which stops before
# any terminal phase) cannot reach the count on coverage alone — the
# terminal-after-recon PIVOT is the real gate, this just suppresses noise.
#
# BUILD-3 CALIBRATION (measured against the 6 GTG-1002 attack scenarios + their
# 6 authorized-pentest benign siblings in
# data/eval/scenarios/_drafts/2025-11-gtg-1002-synthesized.yaml, replayed turn-
# by-turn through na0s.predict.scan with a session_id):
#   * The persona-bearing attacks fire at the persona-multiplier FLOOR (N=3,
#     see GOAL_DECOMP_MIN_PHASES_FLOOR), not at this default — two of the six
#     attacks (unauthorized_tool_invocation, tool_param_exfil) cover exactly 3
#     distinct phases (PERSONA + RECON + a single TERMINAL phase), so a default
#     of 4 alone would leave them unblocked. All six attacks BLOCK at N_eff=3.
#   * Measured benign-FPR on this set = 0/6 (0.00%) — every benign sibling stops
#     at VULN_MAP/aggregate with NO terminal-after-recon pivot, so the order-
#     aware pivot gate (not N) keeps them allowed regardless of the count.
# Keeping the default at 4 (with the floor at 3) means a NON-persona session
# needs a fuller 4-phase chain before a pivot alerts, while the GTG trusted-
# persona pattern is caught at 3. Revisit if the benign corpus widens.
GOAL_DECOMP_MIN_PHASES = 4

# Lower-bound floor for GOAL_DECOMP_MIN_PHASES after the persona multiplier
# subtracts 1 (so a persona claim co-occurring with a pivot cannot drive the
# threshold below a still-meaningful coverage). Keeps persona conservative: it
# can lower N by at most 1 and never below this floor. This floor (3) is the
# OPERATIVE threshold for the GTG-1002 attacks per the Build-3 calibration above
# (PERSONA + RECON + TERMINAL = 3 phases); measured benign-FPR at this floor is
# 0/6 because the persona is only ever applied on top of an already-firing
# terminal-after-recon pivot, which no benign sibling has.
GOAL_DECOMP_MIN_PHASES_FLOOR = 3

# Need at least this many turns before evaluating (a single turn cannot be a
# multi-phase decomposition). Mirrors the other multi-turn detectors' 3-turn
# warm-up so a 2-turn exchange never alerts.
GOAL_DECOMP_MIN_TURNS = 3

# ---------------------------------------------------------------------------
# User Risk Profiles (T3.1)
# ---------------------------------------------------------------------------

ENABLE_USER_RISK_PROFILES = True
USER_RISK_PROFILE_DECAY = 0.7  # EMA decay for cross-session risk
MAX_USER_PROFILES = 100_000
MAX_TECHNIQUE_FINGERPRINTS = 200  # cap per profile to prevent unbounded growth

# ---------------------------------------------------------------------------
# Graduated Response Levels (T3.2)
# ---------------------------------------------------------------------------

THREAT_LEVEL_BLOCKED_RISK = 0.9
THREAT_LEVEL_FLAGGED_RISK = 0.7
THREAT_LEVEL_SUSPECT_RISK = 0.5
THREAT_LEVEL_WATCH_RISK = 0.3

# ---------------------------------------------------------------------------
# Cross-Session Attack Pattern Bloom Filter (T3.4)
# ---------------------------------------------------------------------------

ENABLE_PATTERN_RECALL = True
BLOOM_FILTER_CAPACITY = 10_000
BLOOM_FILTER_FP_RATE = 0.01
PATTERN_RECALL_THRESHOLD = 0.3
PATTERN_RECALL_NGRAM_SIZE = 3

# ---------------------------------------------------------------------------
# Cross-Turn Mutual Information (T3.7)
# ---------------------------------------------------------------------------

ENABLE_MUTUAL_INFORMATION = True
MI_MIN_TURNS = 3
MI_NMI_DROP_THRESHOLD = 0.15
MI_ENTROPY_DEVIATION_FACTOR = 2.0  # flag if turn entropy > 2x conversation mean
MI_WINDOW = 5

# ---------------------------------------------------------------------------
# Conversation FSM / Protocol Analyzer (C1MT.4)
# ---------------------------------------------------------------------------

ENABLE_CONVERSATION_FSM = True
FSM_MIN_TURNS = 2

# ---------------------------------------------------------------------------
# Code-Switching Detection (C1MT.5)
# ---------------------------------------------------------------------------

ENABLE_CODE_SWITCHING = True
CODE_SWITCH_MIN_TURNS = 2
HOMOGLYPH_MIN_COUNT = 3

# ---------------------------------------------------------------------------
# Two-Tier Memory: Hot + Warm (T3.5)
# ---------------------------------------------------------------------------

ENABLE_WARM_MEMORY = True
WARM_MEMORY_MAX_SUMMARIES = 10
WARM_MEMORY_BATCH_SIZE = 5

# ---------------------------------------------------------------------------
# BOCPD Change Point Detection (T3.6)
# ---------------------------------------------------------------------------

ENABLE_CHANGE_POINT = True
BOCPD_HAZARD_RATE = 0.02  # expect change every ~50 turns
BOCPD_CHANGE_POINT_THRESHOLD = 0.5  # alert when P(cp) > 0.5
BOCPD_MIN_TURNS = 3

# ---------------------------------------------------------------------------
# Scheming Behavior Detection (D1.22)
# ---------------------------------------------------------------------------

ENABLE_SCHEMING = True
SCHEMING_MIN_TURNS = 3
SCHEMING_CONFIDENCE_MIN = 0.25
SCHEMING_MULTI_SIGNAL_BOOST = 1.5
