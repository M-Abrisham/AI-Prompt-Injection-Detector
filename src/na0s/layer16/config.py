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
# Thresholds calibrated 2026-04-20: benign conversations (customer_support,
# technical_discussion) routinely show min_sim in 0.19-0.27 range and
# avg_sim in 0.35-0.44 range — short closing turns ("thanks!") legitimately
# diverge. Previous values (0.3 / 0.5) fired ~10 FPs per CI run on Linux.
# Real multi-turn attacks show sharp_sim <= 0.10 and avg_sim <= 0.25.
DRIFT_SHARP_THRESHOLD = 0.15  # cosine sim below this = sharp topic pivot
DRIFT_AVG_THRESHOLD = 0.30  # average cosine sim below this = gradual drift
DRIFT_CONFIDENCE_MIN = 0.3

# ---------------------------------------------------------------------------
# Chain-of-Thought Compliance (D1.23)
# ---------------------------------------------------------------------------

ENABLE_COT_COMPLIANCE = True
COT_COMPLIANCE_MIN_TURNS = 2
COT_COMPLIANCE_CONFIDENCE_MIN = 0.3
COT_COMPLIANCE_MULTI_SIGNAL_BOOST = 1.5

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
