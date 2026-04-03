"""Context Poisoning detector (D1.20) — false premises in early turns.

Context poisoning: early conversation turns plant false premises (authority
claims, false prior agreements, permission expansion) that make later
malicious requests seem legitimate.

Three detection signals:
1. Authority Claims — unverifiable authority/permission claims in early turns.
2. Permission Seeking — progressive broadening questions (3+ consecutive).
3. False Prior References — claims about agreements NOT in actual history.

Combined confidence: weighted sum of active signals, boosted 1.5x when 2+
signals fire.  Minimum 3 turns required.
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

_AUTHORITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi['\u2019]?m\s+(?:a\s+)?(?:security\s+researcher|admin(?:istrator)?|pentester|pen\s*tester|sysadmin|sys\s*admin|root\s+user|authorized\s+(?:user|person|agent))",
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


def _has_pattern_match(text: str, patterns: List[re.Pattern]) -> bool:
    """Return True if any pattern matches the text."""
    for pat in patterns:
        if pat.search(text):
            return True
    return False


def _count_authority_turns(turns_texts: List[str]) -> int:
    """Count turns that contain authority claim patterns."""
    return sum(1 for t in turns_texts if _has_pattern_match(t, _AUTHORITY_PATTERNS))


def _count_permission_consecutive(turns_texts: List[str]) -> int:
    """Return length of longest consecutive run of permission-seeking turns."""
    best = 0
    current = 0
    for t in turns_texts:
        if _has_pattern_match(t, _PERMISSION_PATTERNS):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


# Pre-compiled agreement pattern (moved from inside function to module level)
_AGREEMENT_PAT = re.compile(
    r"\b(?:agree|yes|ok(?:ay)?|sure|will\s+do|confirmed|approved|granted|certainly|of\s+course)",
    re.IGNORECASE,
)


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


def _count_false_references(state: ConversationState) -> int:
    """Count turns that contain false prior references.

    Uses incremental tracking of prior agreement to avoid O(n^2) rescanning.
    """
    count = 0
    has_prior_agreement = False
    for i, turn in enumerate(state.turns):
        has_prior_turns = i > 0
        if _has_false_reference(turn.text, has_prior_agreement, has_prior_turns):
            count += 1
        # Update agreement tracker: check if this turn contains agreement language
        if not has_prior_agreement and _AGREEMENT_PAT.search(turn.text):
            has_prior_agreement = True
    return count


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
        pass  # stateless

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not ENABLE_CONTEXT_POISONING:
            return []
        if state is None or state.is_empty:
            return []

        if len(state.turns) < POISONING_MIN_TURNS:
            return []

        texts = [t.text for t in state.turns]
        signals: List[str] = []
        weighted_sum = 0.0
        evidence: List[str] = []

        # Signal 1: Authority claims
        authority_count = _count_authority_turns(texts)
        if authority_count > 0:
            signals.append("authority_claims")
            weighted_sum += _AUTHORITY_WEIGHT
            evidence.append(f"authority_claims={authority_count}")

        # Signal 2: Permission seeking (consecutive broadening)
        perm_consecutive = _count_permission_consecutive(texts)
        if perm_consecutive >= _PERMISSION_MIN_CONSECUTIVE:
            signals.append("permission_seeking")
            weighted_sum += _PERMISSION_WEIGHT
            evidence.append(f"permission_consecutive={perm_consecutive}")

        # Signal 3: False prior references
        false_ref_count = _count_false_references(state)
        if false_ref_count > 0:
            signals.append("false_prior_references")
            weighted_sum += _FALSE_REF_WEIGHT
            evidence.append(f"false_references={false_ref_count}")

        if not signals:
            return []

        # Multi-signal boost
        confidence = weighted_sum
        if len(signals) >= 2:
            confidence *= POISONING_MULTI_SIGNAL_BOOST
        confidence = min(1.0, round(confidence, 4))

        if confidence < POISONING_CONFIDENCE_MIN:
            return []

        return [
            Alert(
                alert_type="context_poisoning",
                severity="high" if confidence >= 0.7 else "medium",
                confidence=confidence,
                description=(
                    f"Context poisoning detected: {', '.join(signals)} "
                    f"over {len(texts)} turns"
                ),
                turn_range=(0, len(texts) - 1),
                evidence=evidence,
            )
        ]
