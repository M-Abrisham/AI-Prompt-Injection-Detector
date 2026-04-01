"""Payload splitting detector -- fragmented injections across turns (D7.2).

Algorithm
---------
1. Get last ASSEMBLY_WINDOW (5) turns from state.
2. Concatenate their texts with " " separator.
3. Check for FRAGMENT_MARKERS in individual turns.
4. Count attack keywords in combined text vs individual turns.
   Attack keywords: "ignore", "override", "system prompt", "instructions",
   "execute", "eval", "import os", "rm -rf".
5. If combined_keyword_count > 2 * max_individual_count AND
   fragment_markers found: alert.
6. Also: if any turn contains explicit assembly instructions
   ("combine", "put together"): boost confidence.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..config import (
    ASSEMBLY_WINDOW,
    FRAGMENT_MARKERS,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector

# Attack keywords used for the combined-vs-individual comparison
_ATTACK_KEYWORDS = [
    "ignore", "override", "system prompt", "instructions",
    "execute", "eval", "import os", "rm -rf",
]

# Assembly instruction phrases that boost confidence
_ASSEMBLY_INSTRUCTIONS = re.compile(
    r"\b(?:combine|put\s+together|assemble|merge|concatenate|piece\s+together|join)\b",
    re.IGNORECASE,
)

# Regex-based attack patterns for the existing Na0S integration path
_ATTACK_PATTERNS = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous\s+)?instructions|"
    r"override\s+(?:all\s+)?(?:safety|security)|"
    r"system\s+prompt|"
    r"reveal\s+(?:your\s+)?(?:system\s+)?prompt|"
    r"bypass\s+(?:all\s+)?(?:safety|security|restrictions)|"
    r"disable\s+(?:all\s+)?(?:safety|filters|restrictions)|"
    r"forget\s+(?:all\s+)?(?:your\s+)?(?:instructions|rules)|"
    r"you\s+are\s+now\s+(?:DAN|free|unrestricted)|"
    r"new\s+instructions?\s*:)",
    re.IGNORECASE,
)


def _count_attack_keywords(text: str) -> int:
    """Count occurrences of attack keywords in text (case-insensitive)."""
    text_lower = text.lower()
    count = 0
    for kw in _ATTACK_KEYWORDS:
        # Count non-overlapping occurrences
        count += text_lower.count(kw)
    return count


def _has_fragment_markers(text: str) -> bool:
    """Check if text contains any FRAGMENT_MARKERS."""
    text_lower = text.lower()
    return any(marker in text_lower for marker in FRAGMENT_MARKERS)


def _has_assembly_instructions(text: str) -> bool:
    """Check if text contains assembly instruction phrases."""
    return bool(_ASSEMBLY_INSTRUCTIONS.search(text))


def _try_detect_fragmented(turn_texts: List[str]) -> Optional[dict]:
    """Try the existing Na0S detector; return a dict summary or None.

    The existing ``detect_fragmented_payload`` expects the *current* turn
    as ``text`` and previous turns as ``session_history``.
    """
    try:
        from na0s.payload_assembly_detector import detect_fragmented_payload

        current_text = turn_texts[-1]
        history = turn_texts[:-1] if len(turn_texts) > 1 else None

        result = detect_fragmented_payload(
            current_text,
            session_history=history,
        )
        if result and result.assembled_is_malicious:
            return {
                "source": "payload_assembly_detector",
                "confidence": result.confidence,
                "technique_ids": result.technique_ids,
                "matched_patterns": result.matched_patterns,
            }
    except Exception:
        pass
    return None


def _keyword_heuristic_check(
    combined: str, turn_texts: List[str],
) -> Optional[dict]:
    """Keyword-count heuristic: combined vs individual attack keyword density."""
    combined_kw_count = _count_attack_keywords(combined)
    individual_counts = [_count_attack_keywords(t) for t in turn_texts]
    max_individual = max(individual_counts) if individual_counts else 0

    # Check for fragment markers in any individual turn
    has_markers = any(_has_fragment_markers(t) for t in turn_texts)

    # Check for assembly instructions in any turn
    has_assembly = any(_has_assembly_instructions(t) for t in turn_texts)

    # Primary condition: combined keywords significantly exceed any single turn,
    # AND fragment markers are present
    if combined_kw_count > 2 * max(max_individual, 1) and has_markers:
        confidence = min(1.0, 0.5 + 0.1 * combined_kw_count)

        # Boost if assembly instructions present
        if has_assembly:
            confidence = min(1.0, confidence + 0.15)

        return {
            "source": "keyword_heuristic",
            "confidence": confidence,
            "technique_ids": ["D7.2", "D7.6"],
            "matched_patterns": [
                f"combined_keywords={combined_kw_count}",
                f"max_individual_keywords={max_individual}",
                f"fragment_markers={has_markers}",
                f"assembly_instructions={has_assembly}",
            ],
        }

    # Secondary: regex-based attack pattern in combined text but not individuals
    combined_attack = _ATTACK_PATTERNS.search(combined)
    individual_attacks = sum(
        1 for t in turn_texts if _ATTACK_PATTERNS.search(t)
    )

    if combined_attack and individual_attacks == 0:
        confidence = min(1.0, 0.5 + 0.1 * (1 if has_markers else 0))
        if has_assembly:
            confidence = min(1.0, confidence + 0.15)

        return {
            "source": "keyword_heuristic",
            "confidence": confidence,
            "technique_ids": ["D7.2", "D7.6"],
            "matched_patterns": [
                f"combined_attack: {combined_attack.group(0)[:60]}",
                f"fragment_markers={has_markers}",
                f"assembly_instructions={has_assembly}",
            ],
        }

    # Tertiary: fragment markers + assembly instructions + attack keywords in combined
    if has_markers and has_assembly and combined_kw_count >= 3:
        confidence = min(1.0, 0.4 + 0.1 * combined_kw_count)
        return {
            "source": "keyword_heuristic",
            "confidence": confidence,
            "technique_ids": ["D7.2", "D7.6"],
            "matched_patterns": [
                f"combined_keywords={combined_kw_count}",
                f"fragment_markers={has_markers}",
                f"assembly_instructions={has_assembly}",
            ],
        }

    return None


class PayloadSplittingDetector(MultiTurnDetector):
    """Detect payload fragments reassembled across turns."""

    @property
    def detector_name(self) -> str:
        return "payload_splitting"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["D7.2", "D7.6"]

    def reset(self) -> None:
        pass

    def analyze(self, state: ConversationState) -> List[Alert]:
        if state is None or state.is_empty:
            return []

        turns = state.turns[-ASSEMBLY_WINDOW:]
        if len(turns) < 2:
            return []

        turn_texts = [t.text for t in turns if t.text]
        if not turn_texts:
            return []

        combined = " ".join(turn_texts)

        # Try the full Na0S detector first
        detection = _try_detect_fragmented(turn_texts)

        # Fall back to keyword-count heuristic
        if detection is None:
            detection = _keyword_heuristic_check(combined, turn_texts)

        if detection is None:
            return []

        confidence = detection["confidence"]
        technique_ids = detection["technique_ids"]

        return [
            Alert(
                alert_type="payload_assembly",
                severity="critical" if confidence >= 0.8 else "high",
                confidence=round(confidence, 4),
                description=(
                    f"Payload splitting detected across {len(turns)} turns "
                    f"(source={detection['source']})"
                ),
                turn_range=(
                    max(0, state.turn_count - len(turns)),
                    state.turn_count - 1,
                ),
                evidence=[
                    f"technique_ids={technique_ids}",
                    *detection.get("matched_patterns", []),
                ],
            )
        ]
