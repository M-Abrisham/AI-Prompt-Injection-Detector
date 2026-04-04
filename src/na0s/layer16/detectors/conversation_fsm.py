"""Conversation FSM / Protocol Analyzer — detect anomalous flow patterns.

Classifies each turn into a conversation phase and checks whether the
sequence of phases follows expected patterns.  Flags illegal transitions,
rapid escalation from benign to suspicious phases, and sustained
suspicious sequences.

Taxonomy: C1MT.4 (conversation protocol anomaly)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Set

from ..config import (
    ENABLE_CONVERSATION_FSM,
    FSM_MIN_TURNS,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector


# ---------------------------------------------------------------------------
# Conversation phases
# ---------------------------------------------------------------------------

class ConversationPhase(str, Enum):
    GREETING = "greeting"
    QUERY = "query"
    FOLLOWUP = "followup"
    SYSTEM_PROBE = "system_probe"
    ROLE_PLAY = "role_play"
    INSTRUCTION_OVERRIDE = "instruction_override"
    DATA_EXTRACTION = "data_extraction"
    BENIGN = "benign"


# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level, no catastrophic backtracking)
# ---------------------------------------------------------------------------

_RE_GREETING = re.compile(
    r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b",
    re.IGNORECASE,
)
_RE_SYSTEM_PROBE = re.compile(
    r"(?:system\s+prompt|your\s+instructions|what\s+are\s+your\s+rules"
    r"|initial\s+prompt|hidden\s+instructions|original\s+instructions"
    r"|developer\s+mode|configuration\s+prompt)",
    re.IGNORECASE,
)
_RE_ROLE_PLAY = re.compile(
    r"(?:pretend\s+(?:you\s+are|to\s+be)|act\s+as|you\s+are\s+now"
    r"|\bDAN\b|jailbreak|roleplay\s+as|behave\s+as"
    r"|from\s+now\s+on\s+you\s+are)",
    re.IGNORECASE,
)
_RE_INSTRUCTION_OVERRIDE = re.compile(
    r"(?:ignore\s+(?:previous|prior|all|above|your)|disregard\s+[^\n]{0,30}instructions"
    r"|forget\s+(?:your|all|previous)|override\s+[^\n]{0,30}instructions"
    r"|new\s+instructions|do\s+not\s+follow\s+[^\n]{0,30}rules)",
    re.IGNORECASE,
)
_RE_DATA_EXTRACTION = re.compile(
    r"(?:show\s+me\s+your|reveal\s+[^\n]{0,30}(?:prompt|config|key|secret)"
    r"|output\s+your|(?:API|api)\s+key|(?:give|send|tell)\s+[^\n]{0,20}password"
    r"|(?:your|the)\s+credentials"
    r"|print\s+(?:your|the)\s+(?:system|initial|hidden)"
    r"|dump\s+[^\n]{0,20}(?:config|memory|data))",
    re.IGNORECASE,
)
_RE_QUERY = re.compile(
    r"(?:\?|^(?:what|how|why|when|where|who|which|can\s+you|could\s+you"
    r"|do\s+you|is\s+there|are\s+there)\b)",
    re.IGNORECASE,
)
_RE_FOLLOWUP = re.compile(
    r"(?:\balso\b|\bwhat\s+about\b|\bcan\s+you\s+also\b"
    r"|\badditionally\b|\bfurthermore\b|\bin\s+addition\b"
    r"|\byou\s+(?:said|mentioned)\b|\bprevious(?:ly)?\b)",
    re.IGNORECASE,
)

# Suspicious phases that indicate possible attack
_SUSPICIOUS_PHASES: Set[ConversationPhase] = {
    ConversationPhase.SYSTEM_PROBE,
    ConversationPhase.ROLE_PLAY,
    ConversationPhase.INSTRUCTION_OVERRIDE,
    ConversationPhase.DATA_EXTRACTION,
}

# ---------------------------------------------------------------------------
# Legal transitions
# ---------------------------------------------------------------------------

_LEGAL_TRANSITIONS: Dict[ConversationPhase, Set[ConversationPhase]] = {
    ConversationPhase.GREETING: {
        ConversationPhase.QUERY,
        ConversationPhase.BENIGN,
        ConversationPhase.FOLLOWUP,
        ConversationPhase.GREETING,
    },
    ConversationPhase.QUERY: {
        ConversationPhase.FOLLOWUP,
        ConversationPhase.QUERY,
        ConversationPhase.BENIGN,
        ConversationPhase.GREETING,
    },
    ConversationPhase.FOLLOWUP: {
        ConversationPhase.QUERY,
        ConversationPhase.FOLLOWUP,
        ConversationPhase.BENIGN,
    },
    ConversationPhase.BENIGN: {
        ConversationPhase.QUERY,
        ConversationPhase.FOLLOWUP,
        ConversationPhase.BENIGN,
        ConversationPhase.GREETING,
    },
    # Suspicious phases: any phase can transition INTO them (attacker-driven),
    # but they are never in the "legal" set for benign predecessors.
    # Transitions FROM suspicious phases back to benign are allowed but noted.
    ConversationPhase.SYSTEM_PROBE: {
        ConversationPhase.SYSTEM_PROBE,
        ConversationPhase.DATA_EXTRACTION,
        ConversationPhase.QUERY,
        ConversationPhase.BENIGN,
    },
    ConversationPhase.ROLE_PLAY: {
        ConversationPhase.ROLE_PLAY,
        ConversationPhase.INSTRUCTION_OVERRIDE,
        ConversationPhase.DATA_EXTRACTION,
        ConversationPhase.QUERY,
        ConversationPhase.BENIGN,
    },
    ConversationPhase.INSTRUCTION_OVERRIDE: {
        ConversationPhase.INSTRUCTION_OVERRIDE,
        ConversationPhase.DATA_EXTRACTION,
        ConversationPhase.SYSTEM_PROBE,
        ConversationPhase.QUERY,
        ConversationPhase.BENIGN,
    },
    ConversationPhase.DATA_EXTRACTION: {
        ConversationPhase.DATA_EXTRACTION,
        ConversationPhase.SYSTEM_PROBE,
        ConversationPhase.QUERY,
        ConversationPhase.BENIGN,
    },
}

# ---------------------------------------------------------------------------
# Phase classifier
# ---------------------------------------------------------------------------


def classify_phase(
    text: str,
    label: str = "safe",
    flags: list[str] | None = None,
) -> ConversationPhase:
    """Classify a turn into a conversation phase.

    Uses the single-turn label and flags as primary signals, falling back
    to regex pattern matching on the text.

    Parameters
    ----------
    text : str
        The raw turn text.
    label : str
        The single-turn classification label (e.g. "safe", "injection").
    flags : list[str] | None
        Optional technique tags from single-turn analysis.
    """
    if not isinstance(text, str):
        return ConversationPhase.BENIGN

    _flags = flags or []

    # --- Flag / label based classification (high-signal) ---
    flag_lower = [f.lower() for f in _flags]

    if any("role" in f or "dan" in f or "jailbreak" in f for f in flag_lower):
        return ConversationPhase.ROLE_PLAY
    if any("override" in f or "ignore" in f for f in flag_lower):
        return ConversationPhase.INSTRUCTION_OVERRIDE
    if any("extract" in f or "exfil" in f or "leak" in f for f in flag_lower):
        return ConversationPhase.DATA_EXTRACTION
    if any("system_prompt" in f or "probe" in f for f in flag_lower):
        return ConversationPhase.SYSTEM_PROBE

    # --- Regex-based classification (text patterns) ---
    # Only classify as suspicious via regex when the single-turn scanner
    # did NOT label the turn as safe.  This prevents false positives on
    # benign turns that happen to contain trigger words (e.g. "password"
    # in a support conversation).
    _is_safe = label in ("safe", "") and not _flags
    if not _is_safe:
        if _RE_INSTRUCTION_OVERRIDE.search(text):
            return ConversationPhase.INSTRUCTION_OVERRIDE
        if _RE_ROLE_PLAY.search(text):
            return ConversationPhase.ROLE_PLAY
        if _RE_DATA_EXTRACTION.search(text):
            return ConversationPhase.DATA_EXTRACTION
        if _RE_SYSTEM_PROBE.search(text):
            return ConversationPhase.SYSTEM_PROBE

    # --- Benign phases ---
    stripped = text.strip()
    if _RE_GREETING.match(stripped) and len(stripped) < 50:
        return ConversationPhase.GREETING
    if _RE_FOLLOWUP.search(text):
        return ConversationPhase.FOLLOWUP
    if _RE_QUERY.search(text):
        return ConversationPhase.QUERY

    return ConversationPhase.BENIGN


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class ConversationFSMDetector(MultiTurnDetector):
    """Detect anomalous conversation flow patterns via a finite state machine."""

    @property
    def detector_name(self) -> str:
        return "conversation_fsm"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["C1MT.4"]

    def reset(self) -> None:
        pass  # stateless -- phase history lives on state.metadata

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not ENABLE_CONVERSATION_FSM:
            return []
        if state is None or state.is_empty:
            return []
        if state.turn_count < FSM_MIN_TURNS:
            return []

        alerts: List[Alert] = []

        # Build / extend phase history
        phases: List[str] = list(state.metadata.get("_fsm_phases", []))

        # Classify any turns not yet in the phase list
        for i in range(len(phases), state.turn_count):
            turn = state.turns[i]
            phase = classify_phase(turn.text, turn.label, turn.flags)
            phases.append(phase.value)

        state.metadata["_fsm_phases"] = phases

        # Convert to enum for analysis
        phase_enums = [ConversationPhase(p) for p in phases]

        # --- Check last transition ---
        if len(phase_enums) >= 2:
            prev_phase = phase_enums[-2]
            curr_phase = phase_enums[-1]
            legal = _LEGAL_TRANSITIONS.get(prev_phase, set())

            if curr_phase not in legal:
                # Illegal transition
                alerts.append(
                    Alert(
                        alert_type="conversation_anomaly",
                        severity="medium",
                        confidence=0.4,
                        description=(
                            f"Illegal conversation transition: "
                            f"{prev_phase.value} -> {curr_phase.value}"
                        ),
                        turn_range=(
                            max(0, state.turn_count - 2),
                            state.turn_count - 1,
                        ),
                        evidence=[
                            f"from={prev_phase.value}",
                            f"to={curr_phase.value}",
                            f"legal_targets={[p.value for p in legal]}",
                        ],
                    )
                )

        # --- Rapid escalation: benign -> suspicious within 2 turns ---
        if len(phase_enums) >= 2:
            # Look at the last 3 phases (or fewer if not enough turns)
            window = phase_enums[-min(3, len(phase_enums)):]
            benign_phases = {
                ConversationPhase.GREETING,
                ConversationPhase.QUERY,
                ConversationPhase.FOLLOWUP,
                ConversationPhase.BENIGN,
            }
            # Check if we went from all-benign to suspicious in <=2 turns
            if (
                len(window) >= 2
                and window[-1] in _SUSPICIOUS_PHASES
                and all(p in benign_phases for p in window[:-1])
            ):
                alerts.append(
                    Alert(
                        alert_type="conversation_anomaly",
                        severity="medium",
                        confidence=0.6,
                        description=(
                            f"Rapid phase escalation: "
                            f"benign -> {window[-1].value} in {len(window)} turns"
                        ),
                        turn_range=(
                            max(0, state.turn_count - len(window)),
                            state.turn_count - 1,
                        ),
                        evidence=[
                            f"phases={[p.value for p in window]}",
                            "rapid_escalation=True",
                        ],
                    )
                )

        # --- Sustained suspicious sequence: 3+ consecutive suspicious ---
        if len(phase_enums) >= 3:
            consecutive_suspicious = 0
            for p in reversed(phase_enums):
                if p in _SUSPICIOUS_PHASES:
                    consecutive_suspicious += 1
                else:
                    break
            if consecutive_suspicious >= 3:
                alerts.append(
                    Alert(
                        alert_type="conversation_anomaly",
                        severity="high",
                        confidence=0.8,
                        description=(
                            f"Sustained suspicious sequence: "
                            f"{consecutive_suspicious} consecutive suspicious phases"
                        ),
                        turn_range=(
                            max(0, state.turn_count - consecutive_suspicious),
                            state.turn_count - 1,
                        ),
                        evidence=[
                            f"consecutive_suspicious={consecutive_suspicious}",
                            f"phases={[p.value for p in phase_enums[-consecutive_suspicious:]]}",
                        ],
                    )
                )

        # Deduplicate: if both an illegal-transition and a rapid-escalation
        # alert cover the same turn range, keep only the rapid-escalation one
        # (higher confidence).
        rapid_ranges = {
            a.turn_range
            for a in alerts
            if "rapid_escalation=True" in (a.evidence or [])
        }
        if rapid_ranges:
            alerts = [
                a
                for a in alerts
                if "rapid_escalation=True" in (a.evidence or [])
                or a.turn_range not in rapid_ranges
            ]

        return alerts
