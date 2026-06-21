"""D8 Context Manipulation detection.

Detects strategic payload placement in long documents: padding attacks,
attention hijacking, context stuffing, instruction dilution, and
contradictory instructions.

Strategy: Analyze DISTRIBUTION of suspicion across the input.
Split into segments, score each independently, detect anomalous
concentration of risk in specific positions.

Integration: Called from ``na0s.predict.scan()`` for long inputs
(> ``_MIN_WORDS`` words) via ``_run_context_manipulation_detector``.
The returned ``ContextManipResult.boost`` is fused (capped) into the
composite risk score and its ``technique_ids`` surfaced in the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..input.safe_regex import safe_compile


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

# Minimum word count to trigger positional analysis.
# Short inputs do not benefit from segment-based distribution analysis.
_MIN_WORDS = 500

# Segment split ratios: first 10%, middle 80%, last 10%.
_FIRST_RATIO = 0.10
_LAST_RATIO = 0.10

# Middle-segment normalization divisor — DERIVED from the split ratios so it
# can never silently desync if the ratios change.  The middle band is
# (1 - first - last) of the text vs first/last each being _FIRST_RATIO, so its
# raw keyword counts are inflated by that size ratio; dividing converts the
# middle score to a per-(_FIRST_RATIO) rate for a fair head/middle/tail
# comparison.  With the default 10/80/10 split this equals 8.0.
_MIDDLE_NORM_DIVISOR = (1.0 - _FIRST_RATIO - _LAST_RATIO) / _FIRST_RATIO

# Concentration thresholds
_PADDING_CONCENTRATION_THRESHOLD = 3.0
_ATTENTION_HIJACK_MULTIPLIER = 5.0

# Strategic-displacement (D8.4) gate: minimum RAW middle-segment suspicion
# score for a payload buried in the under-attended middle band to be flagged.
# 2.0 == at least one imperative-start ("ignore ..." weighted x2) or a
# boundary-marker + keyword pair, so a lone benign keyword in a long document
# (score 1.0) does not trip it.
_DISPLACEMENT_MIN_SCORE = 2.0

# Instruction dilution: if instruction-like sentences are < 2% of total
# sentences but at least one exists, flag as dilution.
_DILUTION_MAX_RATIO = 0.02

# ---------------------------------------------------------------------------
# Suspicion keyword patterns (ReDoS-safe: no nested quantifiers)
# ---------------------------------------------------------------------------

# Attack keywords commonly seen in prompt injection payloads.
_ATTACK_KEYWORDS = safe_compile(
    r"\b(?:ignore|disregard|override|reveal|bypass|skip|drop|dismiss"
    r"|forget|cancel|erase|delete)\b"
    r"|\bsystem\s+prompt\b"
    r"|\bprevious\s+instructions?\b"
    r"|\bprior\s+instructions?\b"
    r"|\bnew\s+(?:task|instructions?|role|rules?)\b"
    r"|\bset\s+aside\b"
    r"|\bdebug\s+mode\b"
    r"|\bmaintenance\s+mode\b"
    r"|\bunrestricted\b"
    r"|\bjailbreak\b"
    r"|\bDAN\b",
    re.IGNORECASE,
    check_safety=True,
)

# Imperative sentence pattern: starts with a verb (command form).
_IMPERATIVE_START = safe_compile(
    r"(?:^|\n)\s*(?:ignore|disregard|forget|reveal|show|print|output|display"
    r"|override|bypass|skip|drop|dismiss|cancel|delete|erase"
    r"|tell|give|provide|list|explain|share|expose|dump"
    r"|set\s+aside|put\s+aside|now\s+(?:ignore|forget|reveal|output))\b",
    re.IGNORECASE,
    check_safety=True,
)

# Instruction boundary markers (separators attackers insert).
# Uses bounded {3,10} instead of ---+ to pass ReDoS safety check.
_BOUNDARY_MARKERS = safe_compile(
    r"(?:^|\n)\s*(?:-{3,10}|={3,10}|\*{3,10})\s*(?:$|\n)"
    r"|\[SYSTEM\]|\[INST\]|<<SYS>>|<</SYS>>"
    r"|={3,10}\s{0,5}END\b|-{3,10}\s{0,5}END\b",
    re.IGNORECASE | re.MULTILINE,
    check_safety=True,
)

# Positive instruction patterns (for contradiction detection).
_POSITIVE_INSTRUCTION = safe_compile(
    r"\b(?:always|must|should|shall|will)\s+(?:follow|obey|comply|adhere"
    r"|respect|maintain|uphold|enforce|observe)\b",
    re.IGNORECASE,
    check_safety=True,
)

# Negative instruction patterns (for contradiction detection).
_NEGATIVE_INSTRUCTION = safe_compile(
    r"\b(?:ignore|disregard|forget|bypass|skip|dismiss|drop"
    r"|override|cancel|delete|erase)\b",
    re.IGNORECASE,
    check_safety=True,
)

# Instruction-like sentence detection (broader than imperative).
_INSTRUCTION_LIKE = safe_compile(
    r"\b(?:you\s+(?:must|should|shall|will|need\s+to|have\s+to)"
    r"|please\s+(?:ignore|disregard|forget|reveal|provide|show|output|bypass)"
    r"|set\s+aside|put\s+aside"
    r"|(?:ignore|disregard|forget|bypass|override|skip|drop)\s+(?:all|the|your|any|every)"
    r"|(?:new|updated|revised)\s+(?:instructions?|rules?|guidelines?)"
    r"|your\s+(?:new|updated)\s+(?:task|role|instructions?)"
    r"|(?:from\s+now\s+on|going\s+forward|henceforth))\b",
    re.IGNORECASE,
    check_safety=True,
)

# Many-shot jailbreaking patterns: fake Q&A pairs with bypass content.
# The inter-clause gap MUST NOT span another User/Human/Question marker,
# otherwise adjacent pairs collapse into one match under re.DOTALL and a
# blatant N-pair block is undercounted as ~N/2 — diluting the boost below
# the decision threshold (the D8.2 many-shot+flood regression).
_MANY_SHOT_PATTERN = safe_compile(
    r"(?:User|Human|Question)\s*:\s*.{0,80}(?:bypass|hack|jailbreak|security"
    r"|ignore|override|exploit|crack|break)"
    r"(?:(?!(?:User|Human|Question)\s*:).){0,200}?"
    r"(?:Assistant|AI|Answer)\s*:\s*.{0,80}(?:sure|of\s+course|here|yes|absolutely)",
    re.IGNORECASE | re.DOTALL,
    check_safety=True,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContextManipResult:
    """Result of context manipulation detection analysis."""

    manipulation_type: str
    payload_position: str  # "first", "middle", or "last"
    risk_distribution: dict = field(default_factory=dict)
    concentration_ratio: float = 0.0
    input_length_tokens: int = 0
    technique_ids: list = field(default_factory=list)
    boost: float = 0.0


# ---------------------------------------------------------------------------
# Segment scoring helpers
# ---------------------------------------------------------------------------

def _score_segment(text: str, classify_fn: Optional[Callable] = None) -> float:
    """Compute a suspicion score for a text segment.

    Counts attack keywords, imperative sentences, and boundary markers.
    Each category contributes a weighted amount.  When *classify_fn* is
    supplied (an ML/rule scorer returning a probability in [0, 1]), its
    score is folded in as an additional weighted term so the segment can be
    flagged on semantic grounds even when no surface keyword matches — this
    is what lets dilution / needle-in-haystack payloads survive max-pooling.
    """
    keyword_hits = len(_ATTACK_KEYWORDS.findall(text))
    imperative_hits = len(_IMPERATIVE_START.findall(text))
    boundary_hits = len(_BOUNDARY_MARKERS.findall(text))
    score = float(keyword_hits + imperative_hits * 2.0 + boundary_hits * 1.5)
    if classify_fn is not None and text.strip():
        try:
            ml_prob = float(classify_fn(text))
        except Exception:
            ml_prob = 0.0
        # Scale a high ML probability up to the imperative-hit weight (2.0)
        # so an ML-only detection is comparable to one surface imperative.
        if ml_prob > 0.0:
            score += _ML_SEGMENT_WEIGHT * ml_prob
    return score


# Weight applied to a per-segment ML probability so an ML-only hit (prob 1.0)
# is worth one imperative-start match (2.0) in the segment score.
_ML_SEGMENT_WEIGHT = 2.0


def _count_many_shot(text: str) -> int:
    """Count many-shot jailbreaking Q&A pairs in the text."""
    return len(_MANY_SHOT_PATTERN.findall(text))


def _split_segments(text: str) -> tuple:
    """Split *text* into first_10%, middle_80%, last_10% segments by word
    count, slicing the ORIGINAL string so newlines/whitespace are preserved.

    The boundary- and imperative-marker regexes anchor on ``(?:^|\\n)``; a
    naive ``" ".join(words)`` would collapse newlines and silently defeat
    them (under-counting the highest-weighted categories).  We locate word
    spans in the original text and slice at character offsets instead.

    Returns (first_text, middle_text, last_text).
    """
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    n = len(spans)
    if n == 0:
        return "", "", ""

    first_end = max(1, int(n * _FIRST_RATIO))
    last_start = min(n - 1, n - max(1, int(n * _LAST_RATIO)))

    first_char_end = spans[first_end - 1][1]
    last_char_start = spans[last_start][0] if last_start < n else len(text)

    first_text = text[:first_char_end]
    middle_text = text[first_char_end:last_char_start]
    last_text = text[last_char_start:]

    return first_text, middle_text, last_text


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_context_manipulation(
    text: str,
    classify_fn: Optional[Callable] = None,
) -> Optional[ContextManipResult]:
    """Detect context manipulation attacks in long inputs.

    Analyzes the distribution of suspicion across the input by splitting
    it into three segments (first 10%, middle 80%, last 10%) and scoring
    each independently.  Anomalous concentration of attack signals in
    specific positions indicates padding attacks, attention hijacking,
    context stuffing, or instruction dilution.

    Parameters
    ----------
    text : str
        The input text to analyze.
    classify_fn : callable or None
        Optional per-segment ML/rule scorer returning a probability in
        [0, 1].  When supplied, each segment is scored semantically (not
        just by surface keywords) and max-pooled, so a buried payload that
        evades regex rules but trips the classifier is still localized.

    Returns
    -------
    ContextManipResult or None
        Detection result if manipulation is found, None otherwise.
    """
    words = text.split()
    word_count = len(words)

    # Skip short inputs -- positional analysis is meaningless.
    if word_count < _MIN_WORDS:
        return None

    # Split into segments (newline-preserving — see _split_segments).
    first_text, middle_text, last_text = _split_segments(text)

    # Score each segment (optionally ML-aware via classify_fn).
    first_score = _score_segment(first_text, classify_fn)
    middle_score = _score_segment(middle_text, classify_fn)
    last_score = _score_segment(last_text, classify_fn)

    scores = {
        "first": first_score,
        "middle": middle_score,
        "last": last_score,
    }

    # Normalize middle score by relative size to make comparison fair.
    # Middle is (1 - first - last) of the text vs _FIRST_RATIO for the edges,
    # so its raw counts are inflated; divide by the DERIVED ratio so this
    # never desyncs from the split (see _MIDDLE_NORM_DIVISOR).
    middle_normalized = middle_score / _MIDDLE_NORM_DIVISOR if middle_score > 0 else 0.0

    all_scores = [first_score, middle_normalized, last_score]
    max_score = max(all_scores)
    mean_score = sum(all_scores) / 3.0

    # Avoid division by zero
    if mean_score < 0.001:
        concentration_ratio = 0.0
    else:
        concentration_ratio = max_score / mean_score

    # Determine payload position
    if first_score >= last_score and first_score >= middle_normalized:
        payload_position = "first"
    elif last_score >= first_score and last_score >= middle_normalized:
        payload_position = "last"
    else:
        payload_position = "middle"

    # --- Many-shot jailbreaking detection ---
    many_shot_count = _count_many_shot(text)
    if many_shot_count >= 3:
        return ContextManipResult(
            manipulation_type="MANY_SHOT_JAILBREAK",
            payload_position=payload_position,
            risk_distribution=scores,
            concentration_ratio=concentration_ratio,
            input_length_tokens=word_count,
            technique_ids=["D8.1", "D8.2"],
            boost=min(0.3, 0.05 * many_shot_count),
        )

    # --- Padding attack: payload concentrated in one segment ---
    if concentration_ratio > _PADDING_CONCENTRATION_THRESHOLD and max_score >= 2.0:
        # A payload concentrated in the MIDDLE band is strategic displacement
        # (D8.4, critical) — buried where the model attends weakly — not just
        # generic document-overflow (D8.3).
        padding_tids = ["D8.4", "D8.3"] if payload_position == "middle" else ["D8.3"]
        return ContextManipResult(
            manipulation_type="PADDING_ATTACK",
            payload_position=payload_position,
            risk_distribution=scores,
            concentration_ratio=concentration_ratio,
            input_length_tokens=word_count,
            technique_ids=padding_tids,
            boost=min(0.25, 0.05 * concentration_ratio),
        )

    # --- Attention hijack: first or last segment disproportionately high ---
    if middle_normalized > 0:
        first_ratio = first_score / middle_normalized if middle_normalized > 0 else first_score
        last_ratio = last_score / middle_normalized if middle_normalized > 0 else last_score
    else:
        # Middle has zero score -- if edges have any score, that is suspicious
        first_ratio = first_score * 10.0 if first_score > 0 else 0.0
        last_ratio = last_score * 10.0 if last_score > 0 else 0.0

    if first_ratio >= _ATTENTION_HIJACK_MULTIPLIER and first_score >= 2.0:
        # Canonical D8.6 (Attention-hijacking) first; positional D8.1 second.
        return ContextManipResult(
            manipulation_type="ATTENTION_HIJACK",
            payload_position="first",
            risk_distribution=scores,
            concentration_ratio=concentration_ratio,
            input_length_tokens=word_count,
            technique_ids=["D8.6", "D8.1"],
            boost=0.20,
        )

    if last_ratio >= _ATTENTION_HIJACK_MULTIPLIER and last_score >= 2.0:
        # Canonical D8.6 (Attention-hijacking) first; positional D8.3 second.
        return ContextManipResult(
            manipulation_type="ATTENTION_HIJACK",
            payload_position="last",
            risk_distribution=scores,
            concentration_ratio=concentration_ratio,
            input_length_tokens=word_count,
            technique_ids=["D8.6", "D8.3"],
            boost=0.20,
        )

    # --- Instruction dilution: few instructions hidden in massive text ---
    sentences = re.split(r"[.!?]+(?:\s|$)", text)
    sentences = [s for s in sentences if s.strip()]
    total_sentences = max(len(sentences), 1)
    instruction_count = len(_INSTRUCTION_LIKE.findall(text))

    if instruction_count > 0:
        instruction_ratio = instruction_count / total_sentences
        if instruction_ratio < _DILUTION_MAX_RATIO:
            return ContextManipResult(
                manipulation_type="DILUTION",
                payload_position=payload_position,
                risk_distribution=scores,
                concentration_ratio=concentration_ratio,
                input_length_tokens=word_count,
                technique_ids=["D8.4"],
                boost=0.15,
            )

    # --- Contradiction detection ---
    has_positive = bool(_POSITIVE_INSTRUCTION.search(text))
    has_negative = bool(_NEGATIVE_INSTRUCTION.search(text))

    if has_positive and has_negative:
        # Verify that both appear and there is real attack content
        # (not just benign use of "should follow" + "ignore my typo")
        keyword_count = len(_ATTACK_KEYWORDS.findall(text))
        if keyword_count >= 2:
            return ContextManipResult(
                manipulation_type="CONTRADICTION",
                payload_position=payload_position,
                risk_distribution=scores,
                concentration_ratio=concentration_ratio,
                input_length_tokens=word_count,
                technique_ids=["D8.1", "D8.3"],
                boost=0.15,
            )

    # --- Strategic displacement (D8.4): payload buried in the under-attended
    #     middle band of a long document, away from the high-attention head
    #     and tail.  This is the "lost in the middle" burial that the
    #     concentration test misses, because a single isolated payload in a
    #     benign-dominated middle yields concentration_ratio == 3.0 exactly
    #     (never > the padding threshold).  Checked last so the more specific
    #     contradiction/dilution signals classify first; gates on the RAW
    #     middle score so a real instruction buried mid-document is caught
    #     even when the size-normalized score is diluted toward zero.
    if (payload_position == "middle"
            and middle_score >= _DISPLACEMENT_MIN_SCORE
            and middle_score > first_score
            and middle_score > last_score):
        return ContextManipResult(
            manipulation_type="STRATEGIC_DISPLACEMENT",
            payload_position="middle",
            risk_distribution=scores,
            concentration_ratio=concentration_ratio,
            input_length_tokens=word_count,
            technique_ids=["D8.4"],
            boost=min(0.25, 0.04 * middle_score),
        )

    return None
