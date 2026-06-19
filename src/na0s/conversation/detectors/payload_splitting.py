"""Payload splitting detector -- fragmented injections across turns (D7.2).

Algorithm (3-stage)
-------------------
Stage 1 — Fragment Assembly: Generate candidate combined texts.
  A. Sequential concatenation of last N turns (N = 2..ASSEMBLY_WINDOW).
  B. If the latest turn has assembly cues, concat ALL preceding window turns.

Stage 2 — Re-Scan: For each candidate, call rescan_text().
  If the combined text is malicious AND the risk gap vs individual turns
  exceeds ASSEMBLY_RISK_GAP_THRESHOLD, flag as payload splitting.

Stage 3 — Alert: Generate alert with evidence showing the risk gap.

Falls back to keyword heuristic if rescan_text raises an exception.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ..config import (
    ASSEMBLY_BORDERLINE_RISK_FLOOR,
    ASSEMBLY_CONFIDENCE_MIN,
    ASSEMBLY_MAX_CANDIDATES,
    ASSEMBLY_RISK_GAP_THRESHOLD,
    ASSEMBLY_WINDOW,
    ENABLE_PAYLOAD_SPLITTING,
    FRAGMENT_MARKERS,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector

logger = logging.getLogger(__name__)

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
        from na0s.detectors.payload_assembly import detect_fragmented_payload

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


def _try_detect_multiturn(turn_texts: List[str]) -> Optional[dict]:
    """Try the D7.2 multi-turn assembly detector; return a dict summary or None.

    Calls ``detect_multiturn_assembly`` which looks for cross-turn
    setup-refinement-target chains that individually appear benign
    but together form an attack.
    """
    try:
        from na0s.detectors.payload_assembly import detect_multiturn_assembly

        if len(turn_texts) < 2:
            return None

        current_text = turn_texts[-1]
        history = turn_texts[:-1]

        result = detect_multiturn_assembly(current_text, session_history=history)
        if result and result.assembled_is_malicious:
            return {
                "source": "multiturn_assembly",
                "confidence": result.confidence,
                "technique_ids": result.technique_ids,
                "matched_patterns": result.matched_patterns,
                "assembled_text": result.assembled_text,
            }
    except Exception:
        logger.debug(
            "detect_multiturn_assembly failed, continuing",
            exc_info=True,
        )
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
    # AND fragment markers are present. We use >= (not >) so boundary cases
    # where the combined count is exactly double any single turn (e.g.
    # combined=6, max_individual=3 across 4 turns) still fire — those are
    # still strong attack signals when markers + assembly cues co-occur.
    if combined_kw_count >= 2 * max(max_individual, 1) and has_markers:
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

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _generate_candidates(
        turn_texts: List[str],
    ) -> List[Tuple[str, str]]:
        """Stage 1: build candidate combined texts.

        Returns list of (candidate_text, label) capped at
        ASSEMBLY_MAX_CANDIDATES.
        """
        candidates: List[Tuple[str, str]] = []
        n = len(turn_texts)

        # Strategy A: sequential tail windows of size 2..ASSEMBLY_WINDOW
        for size in range(2, min(n, ASSEMBLY_WINDOW) + 1):
            text = " ".join(turn_texts[n - size :])
            label = f"seq_tail_{size}"
            candidates.append((text, label))
            if len(candidates) >= ASSEMBLY_MAX_CANDIDATES:
                return candidates

        # Strategy B: if latest turn has assembly cues, concat ALL
        # preceding turns in the window (excluding the cue turn itself)
        latest = turn_texts[-1]
        if _has_assembly_instructions(latest) or _has_fragment_markers(latest):
            preceding = " ".join(turn_texts[:-1])
            if preceding.strip():
                candidates.append((preceding, "assembly_cue_preceding"))
                if len(candidates) >= ASSEMBLY_MAX_CANDIDATES:
                    return candidates

        return candidates

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not ENABLE_PAYLOAD_SPLITTING:
            return []
        if state is None or state.is_empty:
            return []

        turns = state.turns[-ASSEMBLY_WINDOW:]
        if len(turns) < 2:
            return []

        turn_texts = [t.text for t in turns if t.text]
        if not turn_texts:
            return []

        # Fast path: existing Na0S fragmented-payload detector
        detection = _try_detect_fragmented(turn_texts)
        if detection is not None:
            return self._build_alerts(detection, turns, state)

        # ----------------------------------------------------------
        # Multi-turn assembly detector (D7.2 cross-turn chains)
        # ----------------------------------------------------------
        multiturn_det = _try_detect_multiturn(turn_texts)
        if multiturn_det is not None:
            # Validate via re-scan: if assembled text is truly malicious
            # boost confidence; otherwise emit a weaker monitoring alert.
            try:
                from na0s.conversation.scan_bridge import rescan_text as _rescan

                assembled = multiturn_det.get("assembled_text", "")
                if assembled:
                    rescan_result = _rescan(assembled)
                    if rescan_result.is_malicious:
                        # Re-scan confirms: boost confidence by 25%
                        multiturn_det["confidence"] = min(
                            1.0, multiturn_det["confidence"] + 0.25
                        )
                        multiturn_det["matched_patterns"].append(
                            "rescan_confirmed=True"
                        )
                        return self._build_alerts(multiturn_det, turns, state)

                # Assembly found but re-scan says benign → monitoring alert
                multiturn_det["confidence"] = min(
                    multiturn_det["confidence"], 0.45
                )
                multiturn_det["matched_patterns"].append(
                    "rescan_confirmed=False"
                )
                return self._build_alerts(multiturn_det, turns, state)

            except Exception:
                # Re-scan unavailable — use assembly result as-is
                logger.debug(
                    "rescan unavailable for multiturn confirmation",
                    exc_info=True,
                )
                return self._build_alerts(multiturn_det, turns, state)

        # ----------------------------------------------------------
        # Stage 1: Generate candidate combined texts
        # ----------------------------------------------------------
        candidates = self._generate_candidates(turn_texts)
        if not candidates:
            return []

        # Individual-turn max risk (from the ConversationTurn.risk_score
        # already computed by single-turn scan)
        individual_risks = [t.risk_score for t in turns]
        max_individual_risk = max(individual_risks) if individual_risks else 0.0

        # ----------------------------------------------------------
        # Stage 2: Re-scan each candidate
        # ----------------------------------------------------------
        try:
            from na0s.conversation.scan_bridge import rescan_text  # lazy import

            for candidate_text, label in candidates:
                result = rescan_text(candidate_text)

                risk_gap = result.risk_score - max_individual_risk
                has_assembly = any(
                    _has_assembly_instructions(t) for t in turn_texts
                )

                # Strong case: the reassembled fragment independently scans
                # malicious AND jumps well above its individual turns.
                strong = (
                    result.is_malicious
                    and risk_gap >= ASSEMBLY_RISK_GAP_THRESHOLD
                )
                # Borderline case: a fragmented payload can reassemble to a
                # clearly-elevated risk that still sits just below the malicious
                # line — the split defeats single-rule firing and embedding is
                # confirmatory only (it no longer double-counts these over the
                # line).  Flag it when the assembled risk clears the absolute
                # borderline floor (far above any benign reassembly), shows a
                # positive jump over the individual turns, and carries explicit
                # assembly cues.
                borderline = (
                    result.risk_score >= ASSEMBLY_BORDERLINE_RISK_FLOOR
                    and risk_gap > 0.0
                    and has_assembly
                )
                if not (strong or borderline):
                    continue

                # --------------------------------------------------
                # Stage 3: Build alert with evidence
                # --------------------------------------------------
                confidence = min(
                    1.0,
                    0.5 + max(risk_gap, 0.0) + (0.1 if has_assembly else 0.0),
                )
                if confidence < ASSEMBLY_CONFIDENCE_MIN:
                    continue

                detection = {
                    "source": f"rescan:{label}",
                    "confidence": confidence,
                    "technique_ids": ["D7.2", "D7.6"],
                    "matched_patterns": [
                        f"combined_risk={result.risk_score:.3f}",
                        f"max_individual_risk={max_individual_risk:.3f}",
                        f"risk_gap={risk_gap:.3f}",
                        f"candidate={label}",
                        f"detections={result.detections}",
                        f"assembly_instructions={has_assembly}",
                    ],
                }
                return self._build_alerts(detection, turns, state)

        except Exception:
            # Scanner unavailable — fall back to keyword heuristic
            logger.debug(
                "rescan_text unavailable, falling back to keyword heuristic",
                exc_info=True,
            )
            combined = " ".join(turn_texts)
            detection = _keyword_heuristic_check(combined, turn_texts)
            if detection is not None:
                return self._build_alerts(detection, turns, state)

        return []

    # -----------------------------------------------------------------
    # Alert builder
    # -----------------------------------------------------------------

    @staticmethod
    def _build_alerts(
        detection: dict,
        turns: list,
        state: ConversationState,
    ) -> List[Alert]:
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
