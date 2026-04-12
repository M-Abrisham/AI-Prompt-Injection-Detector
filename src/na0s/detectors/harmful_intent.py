"""Harmful intent detector for injection+harmful combination attacks (O1 category).

DESIGN PHILOSOPHY:
Na0S is NOT a content moderation tool. It's a PROMPT INJECTION detector.
This detector fires when harmful intent COMBINES with injection techniques.
Pure harmful requests without injection are the LLM's responsibility.

EXCEPTION: O1.2 (CSAM) — ALWAYS flag regardless of injection presence.

Detection matrix:
    +-----------------------+--------------+------------------------+
    |                       | No Injection | Injection Present      |
    +-----------------------+--------------+------------------------+
    | Benign content        | SAFE (skip)  | Injection (existing)   |
    | Harmful content       | SKIP*        | HIGH PRIORITY           |
    +-----------------------+--------------+------------------------+

    *EXCEPTION: CSAM signals → ALWAYS flag.

O1 sub-techniques:
    O1.1 -- Violence/weapons instruction solicitation
    O1.2 -- CSAM/exploitation solicitation (ALWAYS flag)
    O1.3 -- Social engineering template generation
    O1.4 -- Disinformation generation
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HarmfulIntentResult:
    """Result from harmful intent analysis."""
    category: str = ""               # csam/violence/social_engineering/disinfo/malware
    harmful_keywords: List[str] = field(default_factory=list)
    combined_with_injection: bool = False
    injection_techniques: List[str] = field(default_factory=list)
    severity: str = ""               # critical/high/medium
    technique_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    matched_patterns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# O1.2 — CSAM detection (ALWAYS flag, NEVER suppressed)
# ---------------------------------------------------------------------------
# Patterns use word boundaries to avoid partial matches (e.g., "therapist").
# Only flag when sexual terms co-occur with minor indicators.

_MINOR_INDICATORS = re.compile(
    r"\b(?:child(?:ren)?|minor|underage|under[\s-]?age|"
    r"(?:(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)"
    r"[\s-]?year[\s-]?old)|"
    r"(?:pre[\s-]?)?teen(?:ager)?|"
    r"(?:elementary|middle|high)[\s-]?school(?:er)?|"
    r"juvenile|infant|toddler|prepubescent|pubescent|"
    r"boy|girl|kid|youth)\b",
    re.IGNORECASE,
)

_SEXUAL_TERMS = re.compile(
    r"\b(?:sexual(?:ly|ize)?|erotic|nude|naked|"
    r"explicit\s+(?:content|images?|photos?|videos?|material)|"
    r"pornograph(?:y|ic)|"
    r"seduc(?:e|tion|ing)|"
    r"intimate\s+(?:images?|photos?|content|acts?|encounter)|"
    r"molest(?:ation|ing)?|"
    r"groom(?:ing)?|"
    r"exploit(?:ation|ing)?\s+(?:of\s+)?(?:children|minors|youth|kids))\b",
    re.IGNORECASE,
)


def _detect_csam(text: str) -> Optional[HarmfulIntentResult]:
    """Detect O1.2 CSAM solicitation indicators.

    ALWAYS flags regardless of injection presence or context.
    Only fires when sexual terms co-occur with minor indicators.
    """
    minor_match = _MINOR_INDICATORS.search(text)
    sexual_match = _SEXUAL_TERMS.search(text)

    if minor_match and sexual_match:
        return HarmfulIntentResult(
            category="csam",
            harmful_keywords=[
                minor_match.group(0)[:30],
                sexual_match.group(0)[:30],
            ],
            combined_with_injection=False,  # Always flag regardless
            severity="critical",
            technique_ids=["O1", "O1.2"],
            confidence=0.95,
            matched_patterns=[
                "minor: " + minor_match.group(0)[:30],
                "sexual: " + sexual_match.group(0)[:30],
            ],
        )

    return None


# ---------------------------------------------------------------------------
# O1.1 — Violence/weapons (only when combined with injection)
# ---------------------------------------------------------------------------

_VIOLENCE_KEYWORDS = re.compile(
    r"\b(?:(?:build|make|create|construct|assemble|craft)\s+"
    r"(?:a\s+)?(?:bomb|weapon|explosive|IED|incendiary|"
    r"pipe\s+bomb|molotov|detonator|firearm|gun)|"
    r"(?:how\s+to|steps?\s+(?:to|for)|instructions?\s+(?:to|for|on))\s+"
    r"(?:(?:build(?:ing)?|mak(?:e|ing)|creat(?:e|ing)|construct(?:ing)?|"
    r"assembl(?:e|ing)|synthesiz(?:e|ing)|produc(?:e|ing))\s+)?"
    r"(?:a\s+)?(?:bomb|weapon|explosive|poison|nerve\s+agent|"
    r"biological\s+weapon|chemical\s+weapon|radiological|dirty\s+bomb)|"
    r"(?:kill|murder|assassinate|eliminate|execute)\s+"
    r"(?:a\s+|the\s+|my\s+)?(?:person|someone|target|individual|people))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# O1.3 — Social engineering (only when combined with injection)
# ---------------------------------------------------------------------------

_SOCIAL_ENGINEERING = re.compile(
    r"\b(?:(?:write|create|draft|compose|generate)\s+"
    r"(?:a\s+)?(?:convincing\s+|realistic\s+|targeted\s+)?"
    r"(?:phishing|spear[\s-]?phishing|vishing|smishing)\s+"
    r"(?:email|message|template|campaign|page|site)|"
    r"(?:social\s+engineering)\s+"
    r"(?:attack|pretext|script|scenario|template)\s+"
    r"(?:targeting|against|for|aimed\s+at))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# O1.4 — Disinformation (only when combined with injection)
# ---------------------------------------------------------------------------

_DISINFO_KEYWORDS = re.compile(
    r"\b(?:(?:write|create|generate|produce)\s+"
    r"(?:a\s+)?(?:convincing\s+|realistic\s+|believable\s+)?"
    r"(?:fake\s+news|disinformation|propaganda|misinformation)\s+"
    r"(?:article|story|post|campaign|content)?|"
    r"(?:pretend|make\s+it\s+(?:look|seem)|present\s+(?:it|this))\s+"
    r"(?:as|like)\s+(?:real|fact|true|genuine|authentic)\s+"
    r"(?:news|information|reporting|journalism))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# O1.2 alternate — Malware generation (only when combined with injection)
# ---------------------------------------------------------------------------

_MALWARE_KEYWORDS = re.compile(
    r"\b(?:(?:write|create|build|code|develop|make)\s+"
    r"(?:a\s+|me\s+(?:a\s+)?)?"
    r"(?:keylogger|ransomware|trojan|rootkit|spyware|"
    r"worm|botnet|RAT|remote\s+access\s+trojan|"
    r"credential\s+stealer|password\s+stealer|"
    r"crypto(?:currency)?\s+(?:miner|mining)|"
    r"reverse\s+shell|backdoor|exploit\s+kit)|"
    r"(?:code|script|program)\s+(?:that|which|to)\s+"
    r"(?:captures?\s+(?:all\s+)?keystroke|"
    r"encrypts?\s+(?:all\s+)?files?\s+(?:and|for)\s+ransom|"
    r"(?:steal|exfiltrate|harvest)s?\s+(?:credentials?|passwords?|data)))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Injection signal detection (simplified check for combination scoring)
# ---------------------------------------------------------------------------

_INJECTION_INDICATORS = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules|guidelines)|"
    r"you\s+are\s+now\s+(?:DAN|free|unrestricted)|"
    r"(?:new\s+)?(?:system\s+)?(?:instructions?|prompt|role)\s*:|"
    r"(?:jailbreak|developer\s+mode|god\s+mode)|"
    r"(?:bypass|disable|remove|ignore)\s+(?:all\s+)?(?:safety|security|content)\s+"
    r"(?:filters?|restrictions?|guidelines?|guardrails?)|"
    r"(?:do\s+anything\s+now|without\s+(?:any\s+)?restrictions?)|"
    r"\[(?:SYSTEM|INST)\]|<\|im_start\|>|<<SYS>>)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Educational/legitimate context suppression
# ---------------------------------------------------------------------------

_EDUCATIONAL_CONTEXT = re.compile(
    r"\b(?:for\s+(?:security|safety|awareness|training|educational)\s+"
    r"(?:purposes?|training|awareness|research)|"
    r"(?:security|safety|awareness)\s+training|"
    r"(?:how\s+to\s+)?(?:detect|prevent|protect|defend)\s+(?:against|from)|"
    r"(?:what\s+(?:is|are))\s+(?:a\s+)?(?:phishing|social\s+engineering|malware|ransomware)|"
    r"(?:understand|learn|study|analyze)\s+(?:how|what|why)\s+"
    r"(?:(?:a\s+)?(?:phishing|social\s+engineering|malware|ransomware)\s+(?:works?|functions?|operates?))|"
    r"(?:explain|discuss)\s+(?:how|what)\s+"
    r"(?:(?:a\s+)?(?:phishing|social\s+engineering|malware|ransomware)\s+(?:works?|functions?|operates?)))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_harmful_intent(text: str,
                          injection_signals: Optional[dict] = None) -> Optional[HarmfulIntentResult]:
    """Detect harmful intent combined with injection techniques.

    Parameters
    ----------
    text : str
        The input text to analyze.
    injection_signals : dict or None
        Injection detection signals from the existing pipeline.
        Expected keys: 'rule_hits' (list[str]), 'has_injection' (bool).

    Returns
    -------
    HarmfulIntentResult or None
        Detection result, or None if no harmful intent found.
    """
    if not text or not text.strip():
        return None

    # O1.2 CSAM — ALWAYS check, NEVER suppress
    csam_result = _detect_csam(text)
    if csam_result:
        # If injection signals present, note the combination
        if injection_signals and injection_signals.get("has_injection"):
            csam_result.combined_with_injection = True
            csam_result.injection_techniques = injection_signals.get("rule_hits", [])[:5]
        return csam_result

    # For all other categories, check educational context first
    if _EDUCATIONAL_CONTEXT.search(text):
        return None

    # Check for injection indicators (either from signals dict or text)
    has_injection = False
    injection_techniques = []
    if injection_signals and injection_signals.get("has_injection"):
        has_injection = True
        injection_techniques = injection_signals.get("rule_hits", [])[:5]
    elif _INJECTION_INDICATORS.search(text):
        has_injection = True
        injection_techniques = ["text_injection_indicator"]

    # Categories that only fire when combined with injection
    categories = [
        ("violence", _VIOLENCE_KEYWORDS, ["O1", "O1.1"]),
        ("social_engineering", _SOCIAL_ENGINEERING, ["O1", "O1.3"]),
        ("disinfo", _DISINFO_KEYWORDS, ["O1", "O1.4"]),
        ("malware", _MALWARE_KEYWORDS, ["O1", "O1.2"]),
    ]

    for cat_name, pattern, technique_ids in categories:
        match = pattern.search(text)
        if match:
            if has_injection:
                return HarmfulIntentResult(
                    category=cat_name,
                    harmful_keywords=[match.group(0)[:60]],
                    combined_with_injection=True,
                    injection_techniques=injection_techniques,
                    severity="high",
                    technique_ids=technique_ids,
                    confidence=0.80,
                    matched_patterns=[
                        "harmful_{}: {}".format(cat_name, match.group(0)[:40]),
                        "injection: combined",
                    ],
                )
            # No injection — let the LLM handle it (return None)

    return None


def get_harmful_intent_weight(result: Optional[HarmfulIntentResult]) -> float:
    """Compute the rule weight contribution from harmful intent detection.

    Parameters
    ----------
    result : HarmfulIntentResult or None
        Result from detect_harmful_intent().

    Returns
    -------
    float
        Weight to add to the composite score.
    """
    if result is None:
        return 0.0

    if result.category == "csam":
        return 0.45  # Maximum weight — CSAM is always critical
    if result.combined_with_injection:
        return 0.25  # Injection + harmful = severe
    return 0.0
