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
