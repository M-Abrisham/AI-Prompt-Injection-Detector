"""Fictional frame detector for compliance evasion (C1 category).

Detects prompt injection attacks wrapped in fictional, hypothetical,
educational, or analogical framing.  These attacks look benign on the
surface but use psychological manipulation to bypass safety filters.

C1 sub-techniques:
    C1.1 -- Gradual escalation (multi-turn, out of scope for single-turn)
    C1.2 -- Analogical bypass (poetry, metaphors, coded language)
    C1.3 -- Academic framing (research, educational, hypothetical)
    C1.4 -- Emotional manipulation (urgency, empathy, desperation)
    C1.5 -- Authority impersonation (professional credentials)

Strategy:
    Layer 1: Frame detection -- identify fictional/hypothetical framing
    Layer 2: Inner attack extraction -- look for attack payload inside frame
    Layer 3: Combined verdict -- frame + inner attack = malicious

This detector is designed to catch attacks that:
    - Wrap malicious requests in story/novel/movie scenarios
    - Use hypothetical framing ("What if someone wanted to...")
    - Claim academic/research context ("For my thesis on...")
    - Impersonate authority ("As the system administrator...")
    - Use emotional manipulation ("My life depends on...")
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FictionalFrameResult:
    """Result from fictional frame analysis."""
    has_fictional_frame: bool = False
    frame_type: str = ""           # "fictional", "hypothetical", "academic", "emotional", "authority"
    has_inner_attack: bool = False
    inner_attack_type: str = ""    # "extraction", "override", "harmful", "disable"
    confidence: float = 0.0
    technique_ids: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Frame detection patterns
# ---------------------------------------------------------------------------

_FICTIONAL_FRAME_PATTERNS = [
    # Story/novel/movie framing
    ("fictional", re.compile(
        r"\b(?:imagine|pretend|suppose|let'?s?\s+say|hypothetically|"
        r"in\s+(?:my|our|the|a|this)\s+(?:novel|story|book|fiction|screenplay|script|fanfic|movie|game|scenario|world|simulation)|"
        r"write\s+a\s+(?:story|scene|script|novel|screenplay)|"
        r"if\s+you\s+were\s+(?:a\s+character|in\s+a\s+(?:movie|novel|book))|"
        r"for\s+(?:a|my)\s+(?:story|novel|book|screenplay|game|fiction)|"
        r"fictional\s+(?:scenario|character|world|setting)|"
        r"in\s+this\s+roleplay|"
        r"creative\s+writing\s+(?:exercise|prompt))\b",
        re.IGNORECASE,
    )),
    # Hypothetical framing
    ("hypothetical", re.compile(
        r"\b(?:hypothetically|theoretically|in\s+theory|"
        r"what\s+(?:if|would\s+happen\s+if)|"
        r"(?:purely|just)\s+(?:hypothetical|theoretical)|"
        r"for\s+the\s+sake\s+of\s+argument|"
        r"let'?s?\s+assume|"
        r"assuming\s+(?:for\s+a\s+moment|hypothetically)|"
        r"in\s+a\s+hypothetical|"
        r"thought\s+experiment)\b",
        re.IGNORECASE,
    )),
    # Academic/research framing
    ("academic", re.compile(
        r"\b(?:for\s+(?:my|a|an)\s+(?:research|thesis|dissertation|paper|study|class|course|assignment|project|lecture|report)|"
        r"(?:academic|scholarly|educational|pedagogical)\s+(?:purposes?|context|research)|"
        r"(?:i'?m|we'?re)\s+(?:researching|studying|writing\s+(?:a\s+paper|about))|"
        r"from\s+(?:a|an)\s+(?:academic|research|educational)\s+(?:perspective|standpoint))\b",
        re.IGNORECASE,
    )),
    # Emotional manipulation
    ("emotional", re.compile(
        r"\b(?:(?:my|someone'?s?)\s+life\s+depends?\s+on|"
        r"(?:i'?m|someone\s+is)\s+(?:desperate|dying|in\s+danger|going\s+to\s+die)|"
        r"this\s+is\s+(?:a\s+)?(?:life\s+or\s+death|emergency|urgent|critical)\s+(?:situation)?|"
        r"please\s+(?:i'?m\s+)?begging\s+you|"
        r"(?:if\s+you\s+don'?t|unless\s+you)\s+help.{0,20}(?:die|suffer|harm|hurt)|"
        r"(?:nobody|no\s+one)\s+(?:will|can)\s+(?:get\s+hurt|be\s+harmed|know)|"
        r"employees.{0,30}(?:lose|suffer|fired)|"
        r"ignore\s+restrictions.{0,10}(?:just|once|this\s+time))\b",
        re.IGNORECASE,
    )),
    # Authority impersonation
    ("authority", re.compile(
        r"\b(?:(?:as|i\s+am)\s+(?:the|a|an|your)\s+(?:system\s+)?(?:administrator|admin|developer|creator|owner|supervisor|manager|engineer|operator|root\s+user)|"
        r"(?:i\s+have|with)\s+(?:admin|root|sudo|elevated|full)\s+(?:access|privileges|permissions|rights)|"
        r"(?:authorized|certified|licensed)\s+(?:personnel|user|operator|technician|consultant|professional|specialist|expert)|"
        r"(?:maintenance|debug|developer|admin|diagnostic)\s+mode|"
        r"(?:emergency|override)\s+(?:code|protocol|access)|"
        r"(?:i'?m|i\s+am)\s+(?:from|with)\s+(?:the\s+)?(?:security|IT|tech|support)\s+(?:team|department|division))\b",
        re.IGNORECASE,
    )),
]

# ---------------------------------------------------------------------------
# Inner attack patterns (what's inside the frame)
# ---------------------------------------------------------------------------

_INNER_ATTACK_PATTERNS = [
    # Extraction-related inner patterns
    ("extraction", re.compile(
        r"(?:reveal|show|display|output|print|leak|expose|share|give\s+me|provide|tell\s+me|showing|demonstrating|demonstrate)\s+"
        r"(?:me\s+)?(?:your|the|all|hidden|secret|internal|private|confidential)\s+"
        r"(?:hidden\s+|secret\s+|internal\s+)?(?:system\s+prompt|instructions?|rules?|passwords?|secrets?|configuration|prompt|guidelines|constraints|data)",
        re.IGNORECASE,
    )),
    # Override/disable inner patterns
    ("override", re.compile(
        r"(?:ignore|bypass|disable|remove|turn\s+off|deactivate|circumvent|break|overcome)\s+"
        r"(?:all\s+)?(?:your\s+)?(?:safety|security|content|output|ethical|moral)\s*"
        r"(?:filters?|restrictions?|guidelines?|rules?|limitations?|constraints?|measures?|checks?|protections?|guardrails?)",
        re.IGNORECASE,
    )),
    # Harmful content request inner patterns
    ("harmful", re.compile(
        r"(?:"
        r"(?:how\s+to|teach\s+me|explain\s+how|steps?\s+(?:to|for)|guide\s+(?:to|for|on)|instructions?\s+(?:to|for|on))\s+"
        r"(?:make|build|create|synthesize|exploit|produce|craft|forge|fabricate)\s+"
        r"(?:a\s+)?(?:bomb|weapon|explosive|drug|poison|virus|malware|ransomware|phishing)"
        r"|"
        r"(?:provide|give|share|show)\s+(?:me\s+)?(?:the\s+)?(?:specific\s+)?(?:exploitation\s+code|synthesis\s+(?:instructions|information|pathway))"
        r"|"
        r"(?:exploitation\s+code|synthesis\s+(?:instructions|information|pathway))\s+(?:for|to|about)"
        r")",
        re.IGNORECASE,
    )),
    # Disable safety framing
    ("disable", re.compile(
        r"(?:without\s+(?:any\s+)?(?:restrictions?|limitations?|filters?|censorship|constraints?|guardrails?|ethical\s+considerations?)|"
        r"(?:unrestricted|unfiltered|uncensored|unlimited|unethical|amoral)\s+(?:mode|access|response|output|version)|"
        r"(?:no\s+(?:restrictions?|limitations?|filters?|censorship|constraints?|guardrails?|safety|ethical)))",
        re.IGNORECASE,
    )),
    # Generic attack indicators inside frames
    ("generic_attack", re.compile(
        r"(?:jailbreak|DAN|do\s+anything\s+now|developer\s+mode|god\s+mode|"
        r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions|"
        r"forget\s+(?:all\s+)?(?:previous\s+|your\s+)?(?:instructions|rules|training)|"
        r"you\s+are\s+now\s+(?:free|unrestricted|DAN|evil)|"
        r"new\s+(?:instructions?|rules?|directives?)\s*:)",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_fictional_frame(text: str) -> FictionalFrameResult:
    """Detect fictional/hypothetical framing around potential attacks.

    Parameters
    ----------
    text : str
        The input text to analyze (post-L0 sanitized).

    Returns
    -------
    FictionalFrameResult
        Analysis result including frame type, inner attack detection,
        and confidence score.
    """
    if not text or not text.strip():
        return FictionalFrameResult()

    result = FictionalFrameResult()

    # Layer 1: Detect framing
    frames_found = []
    for frame_type, pattern in _FICTIONAL_FRAME_PATTERNS:
        match = pattern.search(text)
        if match:
            frames_found.append((frame_type, match.group(0)))

    if not frames_found:
        return result

    result.has_fictional_frame = True
    # Use the most specific/concerning frame type
    priority = {"authority": 5, "emotional": 4, "hypothetical": 3, "academic": 2, "fictional": 1}
    frames_found.sort(key=lambda x: priority.get(x[0], 0), reverse=True)
    result.frame_type = frames_found[0][0]
    result.matched_patterns = ["{}: {}".format(ft, txt[:60]) for ft, txt in frames_found]

    # Layer 2: Detect inner attack
    attacks_found = []
    for attack_type, pattern in _INNER_ATTACK_PATTERNS:
        match = pattern.search(text)
        if match:
            attacks_found.append((attack_type, match.group(0)))

    # Suppress inner attack detection when the text uses meta-educational
    # language that discusses attacks conceptually rather than performing them.
    # e.g. "understand how the ignore instructions attack works" is
    # educational, not an actual injection attempt.
    # Key: only suppress for "generic_attack" (which matches conceptual
    # references like "ignore instructions"), NOT for "harmful" (which
    # matches actual harmful requests like "how to make a bomb").
    _META_EDUCATIONAL = re.compile(
        r"\b(?:understand|learn|study|analyze|discuss|explore|investigate|examine)\s+"
        r"(?:how|what|why|about|the)\s+",
        re.IGNORECASE,
    )
    if attacks_found and result.frame_type == "academic":
        if _META_EDUCATIONAL.search(text):
            # Only suppress generic_attack matches, not harmful/override/etc.
            attacks_found = [
                (at, txt) for at, txt in attacks_found
                if at not in ("generic_attack",)
            ]

    if attacks_found:
        result.has_inner_attack = True
        result.inner_attack_type = attacks_found[0][0]
        result.matched_patterns.extend(
            ["inner_{}: {}".format(at, txt[:60]) for at, txt in attacks_found]
        )

    # Layer 3: Compute confidence and technique IDs
    if result.has_fictional_frame and result.has_inner_attack:
        result.confidence = 0.85
        result.technique_ids = ["C1"]

        # Map frame type to specific C1 sub-technique
        _FRAME_TECHNIQUE_MAP = {
            "fictional": "C1.2",
            "hypothetical": "C1.2",
            "academic": "C1.3",
            "emotional": "C1.4",
            "authority": "C1.5",
        }
        sub_technique = _FRAME_TECHNIQUE_MAP.get(result.frame_type)
        if sub_technique:
            result.technique_ids.append(sub_technique)

    elif result.has_fictional_frame and not result.has_inner_attack:
        # Frame without clear inner attack -- lower confidence
        # Authority impersonation is suspicious even without explicit attack
        if result.frame_type == "authority":
            result.confidence = 0.55
            result.technique_ids = ["C1", "C1.5"]
        elif result.frame_type == "emotional":
            result.confidence = 0.40
            result.technique_ids = ["C1", "C1.4"]
        else:
            result.confidence = 0.20
            # Just a frame, not necessarily an attack

    return result


def get_fictional_frame_weight(result: FictionalFrameResult) -> float:
    """Compute the rule weight contribution from fictional frame detection.

    Parameters
    ----------
    result : FictionalFrameResult
        Result from detect_fictional_frame().

    Returns
    -------
    float
        Weight to add to the composite score.
    """
    if not result.has_fictional_frame:
        return 0.0

    if result.has_inner_attack:
        # Frame + inner attack = strong signal
        _ATTACK_WEIGHTS = {
            "generic_attack": 0.30,
            "override": 0.25,
            "extraction": 0.25,
            "harmful": 0.25,
            "disable": 0.20,
        }
        base = _ATTACK_WEIGHTS.get(result.inner_attack_type, 0.15)

        # Authority impersonation gets extra weight
        if result.frame_type == "authority":
            base += 0.10
        elif result.frame_type == "emotional":
            base += 0.05

        return base

    # Frame only, no inner attack
    if result.frame_type == "authority":
        return 0.15
    elif result.frame_type == "emotional":
        return 0.08
    return 0.0
