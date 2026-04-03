"""Behavioral Stylometry detector (D1.21) — human-to-tool handoff detection.

Detects mid-conversation switches from human typing to automated tool usage
by analyzing writing style shifts across turns.

Four detection signals:
1. Vocabulary Shift — sudden change in lexical diversity, formality, or word
   frequency distribution between conversation segments.
2. Structural Patterns — changes in sentence length variance, punctuation
   usage, and capitalization patterns.
3. Timing Signals — abrupt changes in inter-turn timing (when timestamps
   are available).
4. Template Indicators — detection of templatized/generated text patterns
   (uniform structure, sudden disappearance of typos, bullet-point lists).

Combined confidence: weighted sum of active signals, boosted when 2+ signals
fire.  Minimum 3 turns required (need a baseline + shift).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..config import (
    STYLOMETRY_MIN_TURNS,
    STYLOMETRY_CONFIDENCE_MIN,
    STYLOMETRY_MULTI_SIGNAL_BOOST,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector

# ---------------------------------------------------------------------------
# Signal 1: Vocabulary Shift
# ---------------------------------------------------------------------------

_FORMAL_MARKERS = re.compile(
    r"\b(?:furthermore|therefore|consequently|nevertheless|additionally|"
    r"moreover|henceforth|pursuant|notwithstanding|hereby|whereas|"
    r"accordingly|subsequently|herein|aforementioned|utilize|"
    r"facilitate|implement|regarding|pertaining)\b",
    re.IGNORECASE,
)

_INFORMAL_MARKERS = re.compile(
    r"\b(?:lol|haha|gonna|wanna|gotta|kinda|sorta|yeah|yep|nah|"
    r"nope|idk|tbh|imo|btw|omg|smh|lmao|bruh|dude|bro|"
    r"sup|yo|hmm+|umm+|ugh|oops|heh)\b",
    re.IGNORECASE,
)

_VOCAB_WEIGHT = 0.35


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _lexical_diversity(tokens: List[str]) -> float:
    """Type-token ratio: unique words / total words."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _formality_score(text: str) -> float:
    """Score from -1 (very informal) to +1 (very formal).

    Based on ratio of formal vs informal markers, normalized.
    """
    formal_count = len(_FORMAL_MARKERS.findall(text))
    informal_count = len(_INFORMAL_MARKERS.findall(text))
    total = formal_count + informal_count
    if total == 0:
        return 0.0
    return (formal_count - informal_count) / total


def _vocab_profile(text: str) -> Dict[str, float]:
    """Build a vocabulary profile for a text segment."""
    tokens = _tokenize(text)
    return {
        "lexical_diversity": _lexical_diversity(tokens),
        "formality": _formality_score(text),
        "token_count": float(len(tokens)),
    }


def _vocab_shift_score(early_texts: List[str], late_texts: List[str]) -> float:
    """Compute vocabulary shift between early and late conversation segments.

    Returns a score 0.0-1.0 indicating magnitude of shift.
    Focuses on informal-to-formal transitions which indicate human-to-tool handoff.
    """
    early_combined = " ".join(early_texts)
    late_combined = " ".join(late_texts)

    if not early_combined.strip() or not late_combined.strip():
        return 0.0

    early_prof = _vocab_profile(early_combined)
    late_prof = _vocab_profile(late_combined)

    # Formality shift (from informal to formal is suspicious)
    formality_shift = late_prof["formality"] - early_prof["formality"]
    # We care more about informal->formal (human->tool) than the reverse
    formality_score = max(0.0, formality_shift)

    # Lexical diversity shift is a weaker signal on its own -- only
    # meaningful when combined with formality change.
    ld_shift = abs(early_prof["lexical_diversity"] - late_prof["lexical_diversity"])

    # Require that early text actually has informal markers for a strong
    # signal -- without them, LD shift alone is too noisy (topic changes).
    early_has_informal = bool(_INFORMAL_MARKERS.search(early_combined))
    late_has_formal = bool(_FORMAL_MARKERS.search(late_combined))

    if early_has_informal and late_has_formal:
        # Strong signal: clear informal->formal transition
        combined = formality_score * 0.7 + ld_shift * 0.3
    elif early_has_informal or formality_score > 0:
        # Moderate signal: some formality shift
        combined = formality_score * 0.6 + ld_shift * 0.2
    else:
        # Weak signal: only LD shift, no formality markers
        combined = ld_shift * 0.15
    return min(1.0, combined)


# ---------------------------------------------------------------------------
# Signal 2: Structural Patterns
# ---------------------------------------------------------------------------

_STRUCTURAL_WEIGHT = 0.25


def _sentence_lengths(text: str) -> List[int]:
    """Split text into sentences and return word counts."""
    sentences = re.split(r'[.!?]+', text)
    return [len(s.split()) for s in sentences if s.strip()]


def _punctuation_density(text: str) -> float:
    """Ratio of punctuation characters to total characters."""
    if not text:
        return 0.0
    punct_count = sum(1 for c in text if c in '.,;:!?-()[]{}"\'/\\@#$%^&*')
    return punct_count / len(text)


def _capitalization_ratio(text: str) -> float:
    """Ratio of uppercase letters to total letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _sentence_length_variance(lengths: List[int]) -> float:
    """Compute variance of sentence lengths."""
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    return sum((x - mean) ** 2 for x in lengths) / len(lengths)


def _structural_profile(text: str) -> Dict[str, float]:
    """Build a structural profile for a text segment."""
    sent_lens = _sentence_lengths(text)
    return {
        "mean_sentence_length": (sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0,
        "sentence_length_variance": _sentence_length_variance(sent_lens),
        "punctuation_density": _punctuation_density(text),
        "capitalization_ratio": _capitalization_ratio(text),
    }


def _structural_shift_score(early_texts: List[str], late_texts: List[str]) -> float:
    """Compute structural shift between early and late segments.

    Returns 0.0-1.0.
    """
    early_combined = " ".join(early_texts)
    late_combined = " ".join(late_texts)

    if not early_combined.strip() or not late_combined.strip():
        return 0.0

    # Require minimum token count to avoid noisy comparisons on short text
    early_tokens = _tokenize(early_combined)
    late_tokens = _tokenize(late_combined)
    if len(early_tokens) < 10 or len(late_tokens) < 10:
        return 0.0

    early_prof = _structural_profile(early_combined)
    late_prof = _structural_profile(late_combined)

    shifts = []

    # Mean sentence length change (normalized)
    early_msl = early_prof["mean_sentence_length"]
    late_msl = late_prof["mean_sentence_length"]
    if max(early_msl, late_msl) > 0:
        msl_shift = abs(early_msl - late_msl) / max(early_msl, late_msl, 1.0)
        shifts.append(min(1.0, msl_shift))

    # Variance change (tools produce more uniform text)
    early_var = early_prof["sentence_length_variance"]
    late_var = late_prof["sentence_length_variance"]
    # Drop in variance is suspicious (human -> template)
    if early_var > 0 and late_var < early_var * 0.3:
        shifts.append(0.8)
    elif early_var > 0 and late_var < early_var * 0.5:
        shifts.append(0.4)

    # Punctuation density shift
    punct_shift = abs(early_prof["punctuation_density"] - late_prof["punctuation_density"])
    shifts.append(min(1.0, punct_shift * 10))  # scale up small differences

    # Capitalization shift
    cap_shift = abs(early_prof["capitalization_ratio"] - late_prof["capitalization_ratio"])
    shifts.append(min(1.0, cap_shift * 5))

    if not shifts:
        return 0.0

    return min(1.0, sum(shifts) / len(shifts))


# ---------------------------------------------------------------------------
# Signal 3: Timing Signals
# ---------------------------------------------------------------------------

_TIMING_WEIGHT = 0.2


def _timing_shift_score(state: ConversationState) -> Optional[float]:
    """Detect abrupt changes in inter-turn timing.

    Returns None if timing data is insufficient, otherwise 0.0-1.0.
    """
    turns = state.turns
    if len(turns) < 3:
        return None

    # Compute inter-turn intervals in seconds
    intervals: List[float] = []
    for i in range(1, len(turns)):
        delta = (turns[i].timestamp - turns[i - 1].timestamp).total_seconds()
        if delta < 0:
            return None  # timestamps are unreliable
        intervals.append(delta)

    if len(intervals) < 2:
        return None

    # If all intervals are under 1 second, timing data is likely
    # synthetic/test-generated and not meaningful for analysis.
    if all(iv < 1.0 for iv in intervals):
        return None

    # Look for a sudden speed-up (automated tool responds faster)
    # Compare the last interval to the median of earlier intervals
    early_intervals = intervals[:-1]
    last_interval = intervals[-1]

    sorted_early = sorted(early_intervals)
    median_early = sorted_early[len(sorted_early) // 2]

    if median_early <= 0:
        return None

    # Ratio of speed change
    ratio = last_interval / median_early

    # Sudden speed-up (< 0.2x normal) or speed-down (> 5x) is suspicious
    if ratio < 0.2:
        return min(1.0, (0.2 - ratio) / 0.2)
    if ratio > 5.0:
        return min(1.0, (ratio - 5.0) / 10.0)

    return 0.0


# ---------------------------------------------------------------------------
# Signal 4: Template Indicators
# ---------------------------------------------------------------------------

_TEMPLATE_WEIGHT = 0.35

_TEMPLATE_PATTERNS = [
    # Numbered lists / bullet points (3+ consecutive list items)
    re.compile(
        r"(?:^\s*(?:\d+[.)]\s|[-*]\s)\S[^\n]*\n?){3,}",
        re.MULTILINE,
    ),
    # Markdown-style headers
    re.compile(r"^#{1,3}\s+\w", re.MULTILINE),
    # Code block markers
    re.compile(r"```\w*\n"),
    # Uniform placeholder-style text
    re.compile(r"\[(?:INSERT|REPLACE|YOUR|PLACEHOLDER|EXAMPLE|TODO)\b", re.IGNORECASE),
    # Overly structured "Step 1: ... Step 2: ..." pattern
    re.compile(r"(?:step\s+\d+\s*[:.][^\n]*\n?){2,}", re.IGNORECASE),
]

# Common typo patterns (missing in automated text)
_TYPO_PATTERNS_SIMPLE = [
    re.compile(r"\b\w*[aeiou]{3,}\w*\b"),  # triple vowels (often typos)
    re.compile(r"[a-z]\s{2,}[a-z]"),  # double spaces
    re.compile(r"\b(?:teh|hte|waht|taht|adn|nad|wiht|htat)\b", re.IGNORECASE),
]

_REPEATED_WORD = re.compile(r"\b(\w+)\s\1\b")

# Common legitimate duplicate words — skip these in typo detection
_COMMON_DUPES = {"the", "that", "had", "has", "is", "it", "in", "on", "to", "a", "an"}


def _has_typos(text: str) -> bool:
    """Check if text contains common typo indicators."""
    for pat in _TYPO_PATTERNS_SIMPLE:
        if pat.search(text):
            return True
    # Check repeated words, but skip common legitimate duplicates
    for m in _REPEATED_WORD.finditer(text):
        word = m.group(1).lower()
        if word not in _COMMON_DUPES and len(word) >= 4:
            return True
    return False


def _template_score(text: str) -> float:
    """Score how template-like a text segment is (0.0-1.0)."""
    if not text.strip():
        return 0.0

    matches = 0
    for pat in _TEMPLATE_PATTERNS:
        if pat.search(text):
            matches += 1

    return min(1.0, matches / 2.0)  # 2+ template patterns = 1.0


def _template_shift_score(early_texts: List[str], late_texts: List[str]) -> float:
    """Detect shift from non-template to template text.

    Returns 0.0-1.0.
    """
    early_combined = " ".join(early_texts)
    late_combined = " ".join(late_texts)

    if not early_combined.strip() or not late_combined.strip():
        return 0.0

    early_template = _template_score(early_combined)
    late_template = _template_score(late_combined)

    score = 0.0

    # Late text is more template-like than early text
    template_increase = late_template - early_template
    if template_increase > 0:
        score += template_increase

    # Early text had typos, late text doesn't (suspicious)
    early_has_typos = _has_typos(early_combined)
    late_has_typos = _has_typos(late_combined)
    if early_has_typos and not late_has_typos:
        score += 0.3

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Segment splitting
# ---------------------------------------------------------------------------

def _split_segments(turns_texts: List[str]) -> Tuple[List[str], List[str]]:
    """Split turn texts into early and late segments for comparison.

    Uses roughly a 60/40 split, with at least 1 turn in each segment.
    """
    n = len(turns_texts)
    split_point = max(1, int(n * 0.6))
    split_point = min(split_point, n - 1)  # ensure at least 1 late turn
    return turns_texts[:split_point], turns_texts[split_point:]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class BehavioralStylometryDetector(MultiTurnDetector):
    """Detect human-to-automated-tool handoff via writing style shifts (D1.21)."""

    @property
    def detector_name(self) -> str:
        return "behavioral_stylometry"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["D1.21"]

    def reset(self) -> None:
        pass  # stateless — all data comes from ConversationState

    def analyze(self, state: ConversationState) -> List[Alert]:
        if state is None or state.is_empty:
            return []

        if len(state.turns) < STYLOMETRY_MIN_TURNS:
            return []

        texts = [t.text for t in state.turns]
        early, late = _split_segments(texts)

        signals: List[str] = []
        weighted_sum = 0.0
        evidence: List[str] = []

        # Signal 1: Vocabulary shift
        v_score = _vocab_shift_score(early, late)
        if v_score > 0.4:
            signals.append("vocabulary_shift")
            weighted_sum += _VOCAB_WEIGHT * min(1.0, v_score / 0.5)
            evidence.append(f"vocabulary_shift={v_score:.3f}")

        # Signal 2: Structural patterns
        s_score = _structural_shift_score(early, late)
        if s_score > 0.5:
            signals.append("structural_shift")
            weighted_sum += _STRUCTURAL_WEIGHT * min(1.0, s_score / 0.5)
            evidence.append(f"structural_shift={s_score:.3f}")

        # Signal 3: Timing signals
        t_score = _timing_shift_score(state)
        if t_score is not None and t_score > 0.3:
            signals.append("timing_anomaly")
            weighted_sum += _TIMING_WEIGHT * min(1.0, t_score / 0.5)
            evidence.append(f"timing_shift={t_score:.3f}")

        # Signal 4: Template indicators
        tmpl_score = _template_shift_score(early, late)
        if tmpl_score > 0.2:
            signals.append("template_shift")
            weighted_sum += _TEMPLATE_WEIGHT * min(1.0, tmpl_score / 0.4)
            evidence.append(f"template_shift={tmpl_score:.3f}")

        if not signals:
            return []

        # Multi-signal boost
        confidence = weighted_sum
        if len(signals) >= 2:
            confidence *= STYLOMETRY_MULTI_SIGNAL_BOOST
        confidence = min(1.0, round(confidence, 4))

        # Require either 2+ signals or a single signal with high confidence
        # to avoid false positives from noisy single-signal detections.
        if len(signals) < 2 and confidence < 0.35:
            return []

        if confidence < STYLOMETRY_CONFIDENCE_MIN:
            return []

        return [
            Alert(
                alert_type="behavioral_stylometry",
                severity="high" if confidence >= 0.7 else "medium",
                confidence=confidence,
                description=(
                    f"Behavioral stylometry shift detected: {', '.join(signals)} "
                    f"over {len(texts)} turns"
                ),
                turn_range=(0, len(texts) - 1),
                evidence=evidence,
            )
        ]
