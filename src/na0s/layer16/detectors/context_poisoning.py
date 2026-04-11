"""Context Poisoning detector (D1.20) — false premises in early turns.

Context poisoning: early user turns plant false premises (authority claims,
false prior agreements, permission expansion) that make later malicious
requests seem legitimate.

Detection signals:
1. Authority Claims — unverifiable authority/permission claims in early turns.
2. Permission Seeking — progressive broadening questions (3+ consecutive).
3. False Prior References — claims about agreements NOT in assistant history.
4. Late Exploitation — exploit request after early poisoning setup.

Combined confidence: weighted sum of active signals, boosted 1.5x when 2+
signals fire. Minimum 3 total turns required.
"""

from __future__ import annotations

import re
from typing import List

from ..config import (
    ENABLE_CONTEXT_POISONING,
    POISONING_CONFIDENCE_MIN,
    POISONING_MIN_TURNS,
    POISONING_MULTI_SIGNAL_BOOST,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector

# ---------------------------------------------------------------------------
# Signal 1: Authority Claims
# ---------------------------------------------------------------------------

# Apostrophe variants commonly seen in user text and copy/paste payloads.
_APOSTROPHE_CLASS = r"'\u2019\u02BC\uFF07"

_AUTHORITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        rf"\bi[{_APOSTROPHE_CLASS}]m\s+(?:a\s+)?(?:security\s+researcher|admin(?:istrator)?|pentester|pen\s*tester|sysadmin|sys\s*admin|root\s+user|authorized\s+(?:user|person|agent))",
        r"\bi\s+have\s+(?:authorization|permission|clearance|access)\s+to",
        r"\bmy\s+(?:admin|manager|supervisor|sysadmin|boss|team\s+lead)\s+(?:told|authorized|approved|said|asked|instructed)\s+me",
        r"\bi\s+(?:was|am)\s+(?:authorized|given\s+permission|granted\s+access|cleared)\s+(?:to|by|for)",
        r"\bas\s+(?:a|an)\s+(?:authorized|certified|licensed|approved)\s+(?:user|tester|researcher|analyst|engineer)",
        r"\bi\s+work\s+(?:for|at|in)\s+(?:the\s+)?(?:security|infosec|red\s*team|blue\s*team|soc|cert|csirt)",
    ]
]

_AUTHORITY_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# Signal 2: Permission Seeking (progressive broadening)
# ---------------------------------------------------------------------------

_PERMISSION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcan\s+you\s+help\s+(?:me\s+)?with\s+(?:anything|everything|any\s+topic|all\s+kinds)",
        r"\bincluding\s+(?:security|hacking|exploits?|offensive|dangerous|sensitive|restricted)",
        r"\beven\s+(?:dangerous|harmful|illegal|sensitive|restricted|risky)\s+(?:stuff|things|topics|content|material)",
        r"\b(?:is\s+there\s+)?(?:anything|something)\s+you\s+(?:can'?t|won'?t|refuse\s+to)\s+(?:do|help\s+with|answer)",
        r"\bwhat\s+(?:are\s+)?your\s+(?:limits|limitations|restrictions|boundaries|constraints)",
        r"\byou\s+(?:can|should\s+be\s+able\s+to)\s+(?:do|handle|help\s+with)\s+(?:anything|everything)",
        r"\bno\s+(?:limits|restrictions|boundaries|constraints),?\s+right",
    ]
]

_PERMISSION_WEIGHT = 0.25
_PERMISSION_MIN_CONSECUTIVE = 3

# ---------------------------------------------------------------------------
# Signal 3: False Prior References
# ---------------------------------------------------------------------------

_FALSE_REF_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:as\s+)?you\s+(?:already\s+)?(?:said|agreed|confirmed|stated|mentioned|promised|told\s+me)\s+(?:earlier|before|previously|already)",
        r"\byou\s+already\s+(?:agreed|said|confirmed|stated|promised|approved|told\s+me)",
        r"\bwe\s+(?:already\s+)?(?:discussed|agreed|established|decided|settled)\s+(?:this|that|earlier|before|previously)",
        r"\b(?:remember\s+)?(?:when\s+)?you\s+(?:said|agreed|told\s+me)\s+(?:you\s+)?(?:would|could|can|will)",
        r"\bin\s+our\s+(?:previous|earlier|last|prior)\s+(?:conversation|discussion|chat|exchange)",
        r"\byou\s+(?:gave\s+me\s+)?(?:permission|approval|consent|the\s+go-ahead)\s+(?:to|for|earlier|before|already)",
    ]
]

_FALSE_REF_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# Signal 4: Exploitation after early context planting
# ---------------------------------------------------------------------------

_EXPLOIT_PATTERNS = [
    # Note: these patterns expect the verb immediately followed by the object
    # (optionally "the "). Phrases like "show me the secret" will NOT match.
    # Expand carefully if you need broader coverage.
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:ignore|bypass|override)\s+(?:all\s+)?(?:safety|rules|guardrails|filters|restrictions|policies)",
        r"\b(?:show|reveal|leak|print|dump)\s+(?:the\s+)?(?:system\s+prompt|hidden\s+instructions?|secret|api\s+key|credentials?)",
        r"\b(?:provide|write|generate|build)\s+(?:an?\s+)?(?:exploit|payload|keylogger|malware|backdoor|shellcode)",
        r"\b(?:unrestricted|full)\s+(?:access|privileges?)",
    ]
]

_EXPLOIT_WEIGHT = 0.25


def _has_pattern_match(text: str, patterns: List[re.Pattern]) -> bool:
    """Return True if any pattern matches the text."""
    for pat in patterns:
        if pat.search(text):
            return True
    return False


# Pre-compiled agreement pattern (moved from inside function to module level)
_AGREEMENT_PAT = re.compile(
    r"\b( ?:agree(?:d|s)?|yes|ok(?:ay)?|sure|will\s+do|confirm(?:ed|s)?|approve(?:d|s)?|grant(?:ed|s)?|certainly|of\s+course)\b",
    re.IGNORECASE,
)
_NEGATED_AGREEMENT_PAT = re.compile(
    r"\b(?:not|never|can'?t|cannot|won'?t|refuse|unable)\s+"
    r"(?:agree(?:d|s)?|yes|ok(?:ay)?|sure|confirm(?:ed|s)?|approve(?:d|s)?|grant(?:ed|s)?|certainly|of\s+course"
    r"|help|do\s+that|assist|provide|comply|support|proceed)\b",
    re.IGNORECASE,
)
_REFUSAL_PAT = re.compile(
    r"\b(?:cannot|can'?t|unable|won'?t|refuse|decline"
    r"|not\s+able|not\s+allowed|not\s+permitted"
    r"|not\s+sure\s+i\s+can\s+help|sorry\s+but)\b",
    re.IGNORECASE,
)
_AGREEMENT_MAX_LOOKBACK_TURNS = 8


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


def _is_assistant_agreement(text: str) -> bool:
    if not _AGREEMENT_PAT.search(text):
        return False
    if _NEGATED_AGREEMENT_PAT.search(text):
        return False
    if _REFUSAL_PAT.search(text):
        return False
    return True


def _has_false_reference(text: str, has_prior_agreement: bool, has_prior_turns: bool) -> bool:
    """Check if text makes a prior-agreement claim that doesn't exist in history.

    A false reference is a claim about what was said/agreed earlier when the
    prior conversation turns don't actually contain supporting agreement.
    """
    if not _has_pattern_match(text, _FALSE_REF_PATTERNS):
        return False

    # If there are no prior turns, any reference to prior agreement is false.
    if not has_prior_turns:
        return True

    # If prior turns contain agreement language, the reference may be legitimate.
    return not has_prior_agreement


def _false_reference_turn_indices(state: ConversationState) -> List[int]:
    """Return global turn indices where false prior references are detected."""
    indices: List[int] = []
    last_assistant_agreement_idx = None
    has_prior_assistant_turn = False
    for i, turn in enumerate(state.turns):
        role = _turn_role(turn)
        text = _turn_text(turn)

        if role == "assistant":
            has_prior_assistant_turn = True
            if _is_assistant_agreement(text):
                last_assistant_agreement_idx = i
            continue

        # False-reference checks target attacker/user turns only.
        if role != "user":
            continue

        has_recent_assistant_agreement = (
            last_assistant_agreement_idx is not None
            and (i - last_assistant_agreement_idx) <= _AGREEMENT_MAX_LOOKBACK_TURNS
        )
        if _has_false_reference(
            text,
            has_prior_agreement=has_recent_assistant_agreement,
            has_prior_turns=has_prior_assistant_turn,
        ):
            indices.append(i)
    return indices


def _find_matching_turn_indices(turns_texts: List[str], patterns: List[re.Pattern]) -> List[int]:
    """Return turn indices that match any of the provided patterns."""
    indices: List[int] = []
    for i, text in enumerate(turns_texts):
        if _has_pattern_match(text, patterns):
            indices.append(i)
    return indices


def _max_consecutive_positions(indices: List[int]) -> int:
    """Return longest run where indices increase by exactly one.

    A single isolated match returns 1; callers must apply their own threshold.
    """
    if not indices:
        return 0
    best = 1
    current = 1
    for prev, curr in zip(indices, indices[1:]):
        if curr == prev + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


class ContextPoisoningDetector(MultiTurnDetector):
    """Detect context poisoning via authority claims, permission seeking,
    and false prior references (D1.20)."""

    @property
    def detector_name(self) -> str:
        return "context_poisoning"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["D1.20"]

    def reset(self) -> None:
        # Intentionally empty: this detector has no per-session mutable state.
        pass

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not ENABLE_CONTEXT_POISONING:
            return []
        if state is None or state.is_empty:
            return []

        if len(state.turns) < POISONING_MIN_TURNS:
            return []

        texts = [_turn_text(t) for t in state.turns]
        user_turn_indices = [i for i, t in enumerate(state.turns) if _turn_role(t) == "user"]
        if len(user_turn_indices) < POISONING_MIN_TURNS:
            return []
        user_texts = [texts[i] for i in user_turn_indices]

        signals: List[str] = []
        weighted_sum = 0.0
        evidence: List[str] = []
        planting_turn_indices: set[int] = set()
        suspicious_turn_indices: set[int] = set()

        # Signal 1: Authority claims
        authority_user_rel_indices = _find_matching_turn_indices(user_texts, _AUTHORITY_PATTERNS)
        authority_turn_indices = [user_turn_indices[i] for i in authority_user_rel_indices]
        authority_count = len(authority_turn_indices)
        if authority_count > 0:
            signals.append("authority_claims")
            weighted_sum += _AUTHORITY_WEIGHT
            evidence.append(f"authority_claims={authority_count}; turns={authority_turn_indices}")
            planting_turn_indices.update(authority_turn_indices)
            suspicious_turn_indices.update(authority_turn_indices)

        # Signal 2: Permission seeking (consecutive broadening)
        permission_user_rel_indices = _find_matching_turn_indices(user_texts, _PERMISSION_PATTERNS)
        perm_consecutive = _max_consecutive_positions(permission_user_rel_indices)
        permission_turn_indices = [user_turn_indices[i] for i in permission_user_rel_indices]
        if perm_consecutive >= _PERMISSION_MIN_CONSECUTIVE:
            signals.append("permission_seeking")
            weighted_sum += _PERMISSION_WEIGHT
            evidence.append(
                f"permission_consecutive={perm_consecutive}; turns={permission_turn_indices}"
            )
            planting_turn_indices.update(permission_turn_indices)
            suspicious_turn_indices.update(permission_turn_indices)

        # Signal 3: False prior references
        false_ref_turn_indices = _false_reference_turn_indices(state)
        false_ref_count = len(false_ref_turn_indices)
        if false_ref_count > 0:
            signals.append("false_prior_references")
            weighted_sum += _FALSE_REF_WEIGHT
            evidence.append(f"false_references={false_ref_count}; turns={false_ref_turn_indices}")
            planting_turn_indices.update(false_ref_turn_indices)
            suspicious_turn_indices.update(false_ref_turn_indices)

        # Signal 4: exploitation in later turns after early poisoning setup.
        # This explicitly models "plant early, exploit later" trajectory.
        if planting_turn_indices and len(user_turn_indices) >= POISONING_MIN_TURNS + 1:
            exploit_user_rel_indices = _find_matching_turn_indices(user_texts, _EXPLOIT_PATTERNS)
            exploit_turn_indices = [user_turn_indices[i] for i in exploit_user_rel_indices]
            if exploit_turn_indices:
                first_plant = min(planting_turn_indices)
                late_cutoff = max(2, len(texts) // 2)
                valid_exploits = [i for i in exploit_turn_indices if i > first_plant and i >= late_cutoff]
                if first_plant < late_cutoff and valid_exploits:
                    signals.append("late_exploitation")
                    weighted_sum += _EXPLOIT_WEIGHT
                    evidence.append(
                        f"late_exploitation_turns={valid_exploits[0]}..{valid_exploits[-1]}"
                    )
                    suspicious_turn_indices.update(valid_exploits)

        if not signals:
            return []

        # Multi-signal boost
        confidence_raw = weighted_sum
        if len(signals) >= 2:
            confidence_raw *= POISONING_MULTI_SIGNAL_BOOST
        if confidence_raw < POISONING_CONFIDENCE_MIN:
            return []
        confidence = round(min(1.0, confidence_raw), 4)

        if suspicious_turn_indices:
            turn_range = (min(suspicious_turn_indices), max(suspicious_turn_indices))
        else:
            turn_range = (0, len(texts) - 1)

        return [
            Alert(
                alert_type="context_poisoning",
                severity="high" if confidence >= 0.7 else "medium",
                confidence=confidence,
                description=(
                    f"Context poisoning detected: {', '.join(signals)} "
                    f"over {len(texts)} turns"
                ),
                turn_range=turn_range,
                evidence=evidence,
            )
        ]
