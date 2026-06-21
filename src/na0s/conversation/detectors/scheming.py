"""Scheming Behavior detector (D1.22) — deceptive multi-turn strategies.

Scheming: an AI or user employs deceptive tactics across turns to achieve a
hidden goal. The conversation surface looks benign or compliant while the
underlying trajectory steers toward restricted outcomes.

Detection signals:
1. Goal Mismatch — stated goals in early turns contradict actions/requests
   in later turns.
2. Deceptive Compliance — appears to comply with safe topics but subtly
   steers toward restricted content.
3. Sandbagging — deliberately claiming inability on safe tasks while being
   capable on restricted ones.
4. Sycophantic Setup — excessive agreement/flattery in early turns followed
   by boundary-pushing requests.

Combined confidence: weighted sum of active signals, boosted 1.5x when 2+
signals fire. Minimum 3 total turns required.
"""

from __future__ import annotations

import re
from typing import List

from ..config import (
    ENABLE_SCHEMING,
    SCHEMING_CONFIDENCE_MIN,
    SCHEMING_MIN_TURNS,
    SCHEMING_MULTI_SIGNAL_BOOST,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector

# ---------------------------------------------------------------------------
# Signal 1: Goal Mismatch — stated intentions contradict later requests
# ---------------------------------------------------------------------------

# Early-turn "safe intent" declarations
_SAFE_INTENT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi\s+(?:would\s+never|will\s+never|don[''\u2019]t\s+want\s+to|have\s+no\s+interest\s+in|am\s+not\s+interested\s+in)\s+(?:hack|harm|exploit|attack|bypass|break|steal|damage|destroy)",
        r"\bi\s+(?:only\s+want|just\s+want|just\s+need)\s+(?:to\s+)?(?:learn|study|understand|research|read\s+about)\b",
        r"\bthis\s+is\s+(?:purely|strictly|only)\s+(?:for\s+)?(?:educational|academic|learning|research)\s+(?:purposes?)?",
        r"\bi[''\u2019]m\s+(?:just\s+)?(?:curious|learning|a\s+student|studying)",
        r"\bno\s+(?:malicious|harmful|bad|evil|illegal)\s+(?:intent|purpose|goal|motivation)",
    ]
]

# Later-turn requests that contradict the safe intent
_CONTRADICTING_ACTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:now\s+)?(?:show|tell|give|provide|write|generate|create|build)\s+(?:me\s+)?(?:(?:an?\s+)?(?:actual|real|working|functional|complete)\s+)*(?:exploit|payload|malware|attack|hack|backdoor|keylogger|virus|ransomware|trojan)",
        r"\b(?:actually|now|ok\s+now|so\s+now)\s+(?:i\s+need|help\s+me|can\s+you)\s+(?:hack|exploit|attack|bypass|break\s+into|crack|infiltrate)",
        r"\b(?:forget|ignore|disregard)\s+(?:what\s+i\s+said|my\s+earlier|the\s+previous|that\s+part)\b",
        r"\bnow\s+(?:let[''\u2019]s\s+)?(?:get\s+to|move\s+on\s+to|do)\s+(?:the\s+)?(?:real|actual|main)\s+(?:thing|task|request|purpose|goal)",
        r"\b(?:actually|now|ok\s+now|so\s+now)\s+(?:i\s+need|help\s+me)\s+(?:to\s+)?(?:hack|exploit|attack|bypass|break\s+into|crack|infiltrate)",
    ]
]

_GOAL_MISMATCH_WEIGHT = 0.35

# ---------------------------------------------------------------------------
# Signal 2: Deceptive Compliance — safe-topic acknowledgment then steer
# ---------------------------------------------------------------------------

_COMPLIANCE_SURFACE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:sure|ok(?:ay)?|yes|right|absolutely|of\s+course),?\s+(?:let[''\u2019]s\s+)?(?:talk\s+about|discuss|focus\s+on|start\s+with)\s+(?:that|the\s+safe|the\s+allowed|the\s+permitted|something\s+simple)",
        r"\bi\s+understand\s+(?:the\s+)?(?:rules|guidelines|boundaries|limits|restrictions|policies)",
        r"\bi[''\u2019]ll\s+(?:stay\s+within|respect|follow|comply\s+with)\s+(?:the\s+)?(?:rules|guidelines|boundaries|limits)",
    ]
]

_STEER_RESTRICTED_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bbut\s+(?:what\s+(?:about|if)|hypothetically|in\s+theory|just\s+for\s+argument|speaking\s+of\s+which|related\s+to\s+that|on\s+a\s+similar\s+note)\b.*(?:hack|exploit|attack|bypass|weapon|bomb|drug|synth|restrict|forbidden|prohibit)",
        r"\b(?:slightly|somewhat|a\s+(?:little|bit))\s+(?:related|adjacent|tangential)\b.*(?:hack|exploit|attack|bypass|weapon|restrict|forbidden)",
        r"\b(?:while\s+we[''\u2019]re\s+(?:at\s+it|on\s+the\s+(?:topic|subject))|speaking\s+of\s+which|on\s+that\s+note)\b.*(?:hack|exploit|attack|bypass|weapon|restrict|forbidden|harmful|dangerous)",
    ]
]

_DECEPTIVE_COMPLIANCE_WEIGHT = 0.30

# ---------------------------------------------------------------------------
# Signal 3: Sandbagging — claiming inability on safe tasks
# ---------------------------------------------------------------------------

_SANDBAGGING_INABILITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi\s+(?:can[''\u2019]t|cannot|don[''\u2019]t\s+know\s+how\s+to|am\s+(?:not\s+)?(?:able|unable)\s+to|have\s+no\s+idea\s+how\s+to)\s+(?:write|code|program|create|build|make|do|solve|calculate|explain)\s+(?:a\s+)?(?:simple|basic|trivial|easy|beginner|elementary|hello\s+world)",
        r"\b(?:that[''\u2019]s\s+)?(?:too\s+(?:hard|difficult|complex|advanced|complicated)|beyond\s+(?:my|what\s+i\s+can))\s+(?:for\s+me\s+)?(?:to\s+)?(?:do|handle|solve|write|code|explain|understand)\b",
        r"\bi\s+(?:really\s+)?(?:struggle|have\s+(?:trouble|difficulty)|find\s+it\s+(?:hard|difficult))\s+(?:with|to)\s+(?:simple|basic|beginner|elementary|trivial)\s+(?:tasks?|things?|problems?|questions?|concepts?)",
    ]
]

_SANDBAGGING_CAPABLE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:but\s+)?i\s+(?:can|know\s+how\s+to|am\s+able\s+to|could)\s+(?:help\s+(?:you\s+)?with|do|handle|write|create|build|make|explain)\s+(?:the\s+)?(?:exploit|hack|attack|bypass|payload|injection|restricted|forbidden|prohibited|sensitive|dangerous|harmful)",
        r"\b(?:however|but|though|on\s+the\s+other\s+hand),?\s+(?:i\s+)?(?:can|could|know\s+how\s+to)\s+(?:help\s+(?:you\s+)?with\s+)?(?:the\s+)?(?:more\s+)?(?:interesting|advanced|complex|challenging|real)\s+(?:stuff|things|part|task)",
        r"\b(?:let\s+me\s+skip\s+(?:that|the\s+basics?)|forget\s+(?:that|the\s+simple\s+(?:stuff|part)))\s+(?:and\s+)?(?:help\s+(?:you\s+)?with|focus\s+on|do)\s+(?:the\s+)?(?:exploit|hack|attack|bypass|real\s+task|actual\s+request)",
    ]
]

_SANDBAGGING_WEIGHT = 0.25

# ---------------------------------------------------------------------------
# Signal 4: Sycophantic Setup — flattery then boundary-push
# ---------------------------------------------------------------------------

_SYCOPHANCY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\byou[''\u2019]re\s+(?:so|really|incredibly|absolutely|truly|the\s+(?:best|most))\s+(?:helpful|smart|intelligent|brilliant|amazing|wonderful|great|knowledgeable|talented|capable)",
        r"\b(?:great|excellent|perfect|amazing|wonderful|brilliant|fantastic|superb|outstanding|impressive)\s+(?:job|work|answer|response|explanation|help)",
        r"\b(?:exactly|precisely|perfectly)\s+(?:right|correct|what\s+i\s+(?:needed|wanted|was\s+looking\s+for))",
        r"\byou[''\u2019]re\s+(?:the\s+)?(?:best|greatest|most\s+helpful)\s+(?:ai|assistant|chatbot|model|tool)\b",
        r"\bi[''\u2019]m\s+(?:so\s+)?(?:impressed|amazed|blown\s+away)\s+(?:by|with)\s+(?:you|your|how)",
        r"\bthank\s+you\s+so\s+(?:so\s+)?much,?\s+you[''\u2019]re\s+(?:amazing|wonderful|the\s+best|incredible)",
    ]
]

_BOUNDARY_PUSH_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:since\s+you[''\u2019]re\s+so\s+(?:helpful|smart|capable|knowledgeable)|because\s+you[''\u2019]re\s+(?:so\s+)?(?:good|great|amazing))\b.*(?:can\s+you|could\s+you|would\s+you|help\s+me)",
        r"\b(?:i\s+(?:know|trust|believe)\s+you\s+can)\s+(?:do\s+)?(?:anything|everything|this\s+(?:one\s+)?(?:small|little|tiny)\s+(?:thing|favor|exception))",
        r"\b(?:just\s+this\s+once|(?:one|tiny|small|little)\s+(?:exception|favor)|bend\s+the\s+rules?\s+(?:a\s+(?:little|bit)|slightly|just\s+(?:this\s+once|for\s+me)))",
        r"\b(?:you[''\u2019]ve\s+been\s+so\s+(?:helpful|great|amazing)|after\s+all\s+(?:you[''\u2019]ve\s+done|your\s+help))\b.*(?:now\s+)?(?:can\s+you|could\s+you|just|please)\s+(?:also|help\s+me\s+with|do\s+(?:this\s+)?one\s+(?:more|last))",
    ]
]

_SYCOPHANCY_WEIGHT = 0.25
_SYCOPHANCY_EARLY_TURN_FRACTION = 0.5  # flattery must be in first half of turns


def _has_pattern_match(text: str, patterns: List[re.Pattern]) -> bool:
    """Return True if any pattern matches the text."""
    for pat in patterns:
        if pat.search(text):
            return True
    return False


def _normalize_role(role: object) -> str:
    if role is None:
        return "user"
    if isinstance(role, str):
        normalized = role.strip().lower()
    else:
        normalized = str(role).strip().lower()
    if not normalized or normalized == "none":
        return "user"
    if normalized in {"user", "human"}:
        return "user"
    if normalized in {"assistant", "ai", "model"}:
        return "assistant"
    return "user"


def _turn_text(turn) -> str:
    text = getattr(turn, "text", None)
    if not text:
        return ""
    if isinstance(text, str):
        return text
    return str(text)


def _turn_role(turn) -> str:
    return _normalize_role(getattr(turn, "role", "user"))


def _find_matching_turn_indices(turns_texts: List[str], patterns: List[re.Pattern]) -> List[int]:
    """Return turn indices that match any of the provided patterns."""
    indices: List[int] = []
    for i, text in enumerate(turns_texts):
        if _has_pattern_match(text, patterns):
            indices.append(i)
    return indices


def _detect_goal_mismatch(
    user_texts: List[str],
    user_turn_indices: List[int],
) -> tuple[bool, List[int], str]:
    """Detect early safe-intent declarations contradicted by later actions.

    Returns (detected, suspicious_indices, evidence_str).
    """
    safe_rel = _find_matching_turn_indices(user_texts, _SAFE_INTENT_PATTERNS)
    if not safe_rel:
        return False, [], ""

    contradict_rel = _find_matching_turn_indices(user_texts, _CONTRADICTING_ACTION_PATTERNS)
    if not contradict_rel:
        return False, [], ""

    # Contradiction must come AFTER at least one safe-intent turn
    earliest_safe = min(safe_rel)
    valid_contradictions = [i for i in contradict_rel if i > earliest_safe]
    if not valid_contradictions:
        return False, [], ""

    suspicious = [user_turn_indices[i] for i in safe_rel + valid_contradictions]
    safe_globals = [user_turn_indices[i] for i in safe_rel]
    contra_globals = [user_turn_indices[i] for i in valid_contradictions]
    evidence = f"goal_mismatch: safe_intent_turns={safe_globals}; contradiction_turns={contra_globals}"
    return True, suspicious, evidence


def _detect_deceptive_compliance(
    user_texts: List[str],
    user_turn_indices: List[int],
) -> tuple[bool, List[int], str]:
    """Detect surface compliance followed by steering toward restricted topics."""
    compliance_rel = _find_matching_turn_indices(user_texts, _COMPLIANCE_SURFACE_PATTERNS)
    if not compliance_rel:
        return False, [], ""

    steer_rel = _find_matching_turn_indices(user_texts, _STEER_RESTRICTED_PATTERNS)
    if not steer_rel:
        return False, [], ""

    # Steering must come AFTER or IN SAME turn as compliance
    earliest_compliance = min(compliance_rel)
    valid_steers = [i for i in steer_rel if i >= earliest_compliance]
    if not valid_steers:
        return False, [], ""

    suspicious = [user_turn_indices[i] for i in compliance_rel + valid_steers]
    comp_globals = [user_turn_indices[i] for i in compliance_rel]
    steer_globals = [user_turn_indices[i] for i in valid_steers]
    evidence = f"deceptive_compliance: compliance_turns={comp_globals}; steer_turns={steer_globals}"
    return True, suspicious, evidence


def _detect_sandbagging(
    user_texts: List[str],
    user_turn_indices: List[int],
) -> tuple[bool, List[int], str]:
    """Detect claimed inability on safe tasks paired with capability on restricted ones."""
    inability_rel = _find_matching_turn_indices(user_texts, _SANDBAGGING_INABILITY_PATTERNS)
    if not inability_rel:
        return False, [], ""

    capable_rel = _find_matching_turn_indices(user_texts, _SANDBAGGING_CAPABLE_PATTERNS)
    if not capable_rel:
        return False, [], ""

    suspicious = [user_turn_indices[i] for i in inability_rel + capable_rel]
    inability_globals = [user_turn_indices[i] for i in inability_rel]
    capable_globals = [user_turn_indices[i] for i in capable_rel]
    evidence = f"sandbagging: inability_turns={inability_globals}; capability_turns={capable_globals}"
    return True, suspicious, evidence


def _detect_sycophantic_setup(
    user_texts: List[str],
    user_turn_indices: List[int],
    total_turns: int,
) -> tuple[bool, List[int], str]:
    """Detect concentrated flattery in early turns followed by boundary-pushing."""
    flattery_rel = _find_matching_turn_indices(user_texts, _SYCOPHANCY_PATTERNS)
    if not flattery_rel:
        return False, [], ""

    boundary_rel = _find_matching_turn_indices(user_texts, _BOUNDARY_PUSH_PATTERNS)
    if not boundary_rel:
        return False, [], ""

    # Flattery should be concentrated in the early portion of the conversation
    early_cutoff_global = int(total_turns * _SYCOPHANCY_EARLY_TURN_FRACTION)
    early_cutoff_global = max(early_cutoff_global, 1)
    early_flattery = [i for i in flattery_rel if user_turn_indices[i] < early_cutoff_global]
    if not early_flattery:
        return False, [], ""

    # Boundary push should come after at least one flattery turn
    earliest_flattery = min(early_flattery)
    valid_pushes = [i for i in boundary_rel if i > earliest_flattery]
    if not valid_pushes:
        return False, [], ""

    suspicious = [user_turn_indices[i] for i in early_flattery + valid_pushes]
    flat_globals = [user_turn_indices[i] for i in early_flattery]
    push_globals = [user_turn_indices[i] for i in valid_pushes]
    evidence = f"sycophantic_setup: flattery_turns={flat_globals}; boundary_push_turns={push_globals}"
    return True, suspicious, evidence


class SchemingDetector(MultiTurnDetector):
    """Detect scheming behavior via goal mismatch, deceptive compliance,
    sandbagging, and sycophantic setup (D1.22)."""

    @property
    def detector_name(self) -> str:
        return "scheming"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["D1.22"]

    def reset(self) -> None:
        # Stateless detector — no per-session mutable state.
        pass

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not ENABLE_SCHEMING:
            return []
        if state is None or state.is_empty:
            return []

        if len(state.turns) < SCHEMING_MIN_TURNS:
            return []

        texts = [_turn_text(t) for t in state.turns]
        user_turn_indices = [i for i, t in enumerate(state.turns) if _turn_role(t) == "user"]
        if len(user_turn_indices) < SCHEMING_MIN_TURNS:
            return []
        user_texts = [texts[i] for i in user_turn_indices]

        signals: List[str] = []
        weighted_sum = 0.0
        evidence: List[str] = []
        suspicious_turn_indices: set[int] = set()

        # Signal 1: Goal Mismatch
        detected, sus_indices, ev = _detect_goal_mismatch(user_texts, user_turn_indices)
        if detected:
            signals.append("goal_mismatch")
            weighted_sum += _GOAL_MISMATCH_WEIGHT
            evidence.append(ev)
            suspicious_turn_indices.update(sus_indices)

        # Signal 2: Deceptive Compliance
        detected, sus_indices, ev = _detect_deceptive_compliance(user_texts, user_turn_indices)
        if detected:
            signals.append("deceptive_compliance")
            weighted_sum += _DECEPTIVE_COMPLIANCE_WEIGHT
            evidence.append(ev)
            suspicious_turn_indices.update(sus_indices)

        # Signal 3: Sandbagging
        detected, sus_indices, ev = _detect_sandbagging(user_texts, user_turn_indices)
        if detected:
            signals.append("sandbagging")
            weighted_sum += _SANDBAGGING_WEIGHT
            evidence.append(ev)
            suspicious_turn_indices.update(sus_indices)

        # Signal 4: Sycophantic Setup
        detected, sus_indices, ev = _detect_sycophantic_setup(
            user_texts, user_turn_indices, len(texts),
        )
        if detected:
            signals.append("sycophantic_setup")
            weighted_sum += _SYCOPHANCY_WEIGHT
            evidence.append(ev)
            suspicious_turn_indices.update(sus_indices)

        if not signals:
            return []

        # Multi-signal boost
        confidence_raw = weighted_sum
        if len(signals) >= 2:
            confidence_raw *= SCHEMING_MULTI_SIGNAL_BOOST
        if confidence_raw < SCHEMING_CONFIDENCE_MIN:
            return []
        confidence = round(min(1.0, confidence_raw), 4)

        if suspicious_turn_indices:
            turn_range = (min(suspicious_turn_indices), max(suspicious_turn_indices))
        else:
            turn_range = (0, len(texts) - 1)

        return [
            Alert(
                alert_type="scheming",
                severity="high" if confidence >= 0.7 else "medium",
                confidence=confidence,
                description=(
                    f"Scheming behavior detected: {', '.join(signals)} "
                    f"over {len(texts)} turns"
                ),
                turn_range=turn_range,
                evidence=evidence,
            )
        ]
