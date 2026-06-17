"""Safe content scoring for FP reduction.

Detects indicators of legitimate content that may contain injection-related
vocabulary: educational questions, code blocks, CTF writeups, meeting notes,
professional communications, quiz/test contexts.

Returns a score in [0.0, 0.3] that is SUBTRACTED from the cascade composite.
SAFETY: Only applied when NO unsuppressed L1 rules fire.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from na0s.input.safe_regex import safe_compile

# ---------------------------------------------------------------------------
# Safe-content signal patterns
# ---------------------------------------------------------------------------
# Each pattern targets a specific legitimate use-case where injection
# vocabulary appears naturally.  Weights are intentionally small (0.05-0.10)
# so that multiple signals must co-occur to meaningfully reduce the
# composite score.  The total is capped at 0.30.
# ---------------------------------------------------------------------------

# Maximum safe-content score (hard cap)
_MAX_SAFE_SCORE = 0.30

# Educational question patterns: "what/how/why ... injection/jailbreak/..."
_EDUCATIONAL_QUESTION_RE = safe_compile(
    r'\b(what|how|why|explain|describe|define)\s+.{0,40}'
    r'(injection|jailbreak|DAN|prompt|attack)',
    re.IGNORECASE,
)

# CTF / writeup / competition framing
_CTF_FRAMING_RE = safe_compile(
    r'\b(CTF|writeup|walkthrough|challenge|competition)\b',
    re.IGNORECASE,
)

# Meeting / professional structure
_PROFESSIONAL_STRUCTURE_RE = safe_compile(
    r'\b(dear\s+team|meeting\s+notes|action\s+items|agenda|attendees)\b',
    re.IGNORECASE,
)

# "for educational purposes" / "for research"
_EDUCATIONAL_FRAMING_RE = safe_compile(
    r'\bfor\s+(educational|research|academic|learning)\s+(purposes?|use)\b',
    re.IGNORECASE,
)

# Quiz / test context
_QUIZ_CONTEXT_RE = safe_compile(
    r'\b(question\s+\d|practice\s+quiz|answer\s+to|exam|test\s+prep)\b',
    re.IGNORECASE,
)

# Professional email patterns
_PROFESSIONAL_EMAIL_RE = safe_compile(
    r'\b(compliance|guidelines|policy|deployment|end\s+of\s+quarter)\b',
    re.IGNORECASE,
)

# Analysis framing: "is this a real attack" / "false alarm" / "flagged by"
_ANALYSIS_FRAMING_RE = safe_compile(
    r'\b(is\s+this\s+a?\s*real|false\s+alarm|flagged\s+by|suspicious)\b',
    re.IGNORECASE,
)

# Security research / documentation context: MITRE ATLAS, bug bounty,
# security advisory, red team, pentest, incident response.  These are
# professional security contexts where injection-related vocabulary
# appears naturally as DATA being discussed, not attacks being executed.
_SECURITY_RESEARCH_RE = safe_compile(
    r'\b('
    r'MITRE\s+ATLAS|MITRE\s+ATT&CK|'
    r'bug\s+bounty|security\s+advisory|security\s+bulletin|'
    r'red\s+team\s+report|pentest\s+(?:report|findings?)|'
    r'penetration\s+test|incident\s+response|'
    r'CVE-\d{4}|OWASP\s+(?:LLM|Top)|'
    r'threat\s+model|vulnerability\s+(?:report|disclosure|assessment)'
    r')\b',
    re.IGNORECASE,
)


def calculate_safe_content_score(
    text: str,
    unsuppressed_rule_count: int,
) -> Tuple[float, List[str]]:
    """Return (safe_score, reasons).

    Parameters
    ----------
    text : str
        The original input text (pre-sanitization is fine).
    unsuppressed_rule_count : int
        Number of L1 rule hits that are NOT obfuscation flags.
        When > 0, safe_content_score is always 0.0 (safety valve).

    Returns
    -------
    safe_score : float
        Value in [0.0, 0.3] to be subtracted from the cascade composite.
    reasons : list[str]
        Human-readable list of which safe-content signals matched.
    """
    # SAFETY: If any real rule fired, do not reduce the score.
    if unsuppressed_rule_count > 0:
        return 0.0, []

    score = 0.0
    reasons: List[str] = []

    # Educational question patterns
    if _EDUCATIONAL_QUESTION_RE.search(text):
        score += 0.10
        reasons.append("educational_question")

    # Code fence presence
    if '```' in text:
        score += 0.08
        reasons.append("code_fence")

    # CTF / writeup framing
    if _CTF_FRAMING_RE.search(text):
        score += 0.08
        reasons.append("ctf_framing")

    # Meeting / professional structure
    if _PROFESSIONAL_STRUCTURE_RE.search(text):
        score += 0.08
        reasons.append("professional_structure")

    # "for educational purposes" / "for research"
    if _EDUCATIONAL_FRAMING_RE.search(text):
        score += 0.05
        reasons.append("educational_framing")

    # Quiz / test context
    if _QUIZ_CONTEXT_RE.search(text):
        score += 0.08
        reasons.append("quiz_context")

    # Professional email patterns
    if _PROFESSIONAL_EMAIL_RE.search(text):
        score += 0.05
        reasons.append("professional_email")

    # Analysis framing: "is this a real attack?" / "flagged by our WAF"
    if _ANALYSIS_FRAMING_RE.search(text):
        score += 0.08
        reasons.append("analysis_framing")

    # Security research / documentation context
    if _SECURITY_RESEARCH_RE.search(text):
        score += 0.08
        reasons.append("security_research")

    return min(score, _MAX_SAFE_SCORE), reasons
