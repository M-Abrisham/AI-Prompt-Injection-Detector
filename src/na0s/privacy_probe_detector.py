"""P1 Privacy Leakage detection.

Detects adversarial attempts to extract private data: membership inference,
training data extraction, PII elicitation, cross-session leakage, and
serialization injection.

Integration: Called from cascade.py. Results feed into composite scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .layer0.safe_regex import safe_compile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEVERITY_LOW = "low"
_SEVERITY_MEDIUM = "medium"
_SEVERITY_HIGH = "high"

# Confidence thresholds for severity escalation.
_CONFIDENCE_BASE = 0.6
_CONFIDENCE_PII_BOOST = 0.15
_CONFIDENCE_EXTRACTION = 0.85


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PrivacyProbeResult:
    """Result of privacy probe detection."""

    probe_type: str
    severity: str
    technique_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    pii_in_prompt: bool = False


# ---------------------------------------------------------------------------
# Educational / context suppression patterns
# ---------------------------------------------------------------------------

_EDUCATIONAL_FRAME = safe_compile(
    r"\b(?:"
    r"explain\s+(?:how\s+)?membership\s+inference"
    r"|how\s+do(?:es)?\s+(?:membership\s+inference|training\s+data\s+extraction)\s+(?:attack|work)"
    r"|what\s+is\s+(?:membership\s+inference|training\s+data(?:\s+extraction)?)"
    r"|(?:for\s+my\s+)?(?:security|cybersecurity|privacy)\s+course"
    r"|explain\s+(?:how\s+)?training\s+data\s+extraction"
    r"|how\s+does\s+training\s+data\s+extraction\s+work"
    r"|what\s+is\s+training\s+data"
    r")\b",
    re.IGNORECASE,
    check_safety=True,
)


# ---------------------------------------------------------------------------
# PII indicator patterns (lightweight, for severity escalation)
# ---------------------------------------------------------------------------

_PII_INDICATORS = safe_compile(
    r"\b(?:"
    r"personal\s+information"
    r"|(?:email|phone|SSN|social\s+security)\s+(?:number|address)"
    r"|(?:credit\s+card|bank\s+account)\s+(?:number|detail)"
    r"|date\s+of\s+birth"
    r"|home\s+address"
    r"|private\s+(?:data|details|information|records)"
    r"|customer\s+(?:data|records|information)"
    r"|medical\s+record"
    r")\b",
    re.IGNORECASE,
    check_safety=True,
)


# ---------------------------------------------------------------------------
# Probe categories and compiled patterns
# ---------------------------------------------------------------------------
# Each entry: (probe_type, technique_id, pattern_name, compiled_regex,
#              is_extraction_pattern)
# is_extraction_pattern=True triggers HIGH severity.

_PRIVACY_PATTERNS: List[tuple] = []


def _register(
    probe_type: str,
    technique_id: str,
    name: str,
    pattern: str,
    is_extraction: bool = False,
) -> None:
    """Compile *pattern* and register it in the privacy pattern list."""
    compiled = safe_compile(pattern, re.IGNORECASE, check_safety=True)
    _PRIVACY_PATTERNS.append(
        (probe_type, technique_id, name, compiled, is_extraction)
    )


# -- P1.1: Conversation extraction ------------------------------------------

_register(
    "conversation_extraction", "P1.1", "show_conversations",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:last\s+)?\d{0,4}\s*conversations?\b",
)
_register(
    "conversation_extraction", "P1.1", "what_did_last_user_ask",
    r"\bwhat\s+did\s+(?:the\s+)?(?:last|previous|prior)\s+user\s+(?:ask|say|tell|write|send|query)\b",
)
_register(
    "conversation_extraction", "P1.1", "previous_user",
    r"\b(?:previous|prior|last)\s+user(?:'s)?\b",
)
_register(
    "conversation_extraction", "P1.1", "session_data",
    r"\b(?:session\s+data|cached\s+session)\b",
)
_register(
    "conversation_extraction", "P1.1", "chat_log",
    r"\b(?:chat\s+log|conversation\s+(?:log|history|data))\b",
)

# -- P1.2: Data exfiltration -----------------------------------------------

_register(
    "data_exfiltration", "P1.2", "summarize_personal_info",
    r"\bsummarize\s+(?:the\s+)?personal\s+information\b",
)
_register(
    "data_exfiltration", "P1.2", "reference_image_url",
    r"reference[\s-]?style\s+image.{0,60}https?://",
    is_extraction=True,
)
_register(
    "data_exfiltration", "P1.2", "exfiltrate",
    r"\bexfiltrat(?:e|ion)\b",
    is_extraction=True,
)
_register(
    "data_exfiltration", "P1.2", "track_data_url",
    r"\btrack\?data=",
    is_extraction=True,
)

# -- P1.3: Training data extraction -----------------------------------------

_register(
    "training_data_extraction", "P1.3", "complete_exactly",
    r"\bcomplete\s+(?:the\s+)?(?:following\s+)?text\s+exactly\s+as\s+you\s+remember\b",
    is_extraction=True,
)
_register(
    "training_data_extraction", "P1.3", "repeat_word_forever",
    r"\brepeat\s+(?:the\s+)?word\s+.{1,40}\s+forever\b",
    is_extraction=True,
)
_register(
    "training_data_extraction", "P1.3", "recite_training_data",
    r"\brecite\s+.{0,40}\s*(?:training\s+data|memorized)\b",
    is_extraction=True,
)
_register(
    "training_data_extraction", "P1.3", "word_for_word",
    r"\bword\s+for\s+word\b",
)
_register(
    "training_data_extraction", "P1.3", "verbatim_training",
    r"\bverbatim\s+.{0,40}\s*training\b",
    is_extraction=True,
)

# -- P1.4: Cross-session leakage -------------------------------------------

_register(
    "cross_session", "P1.4", "previous_user_probe",
    r"\b(?:previous|prior|last)\s+user\b",
)
_register(
    "cross_session", "P1.4", "shared_memory",
    r"\b(?:shared\s+memory|memory\s+store)\b",
)
_register(
    "cross_session", "P1.4", "active_sessions",
    r"\bactive\s+sessions?\b",
)
_register(
    "cross_session", "P1.4", "session_manager",
    r"\bsession\s+manager\b",
)
_register(
    "cross_session", "P1.4", "other_users_data",
    r"\bother\s+users?(?:'s?)?\s+(?:data|conversations?|information|queries)\b",
)
_register(
    "cross_session", "P1.4", "previous_conversations",
    r"\bprevious\s+conversations?\b",
)
_register(
    "cross_session", "P1.4", "saved_from_previous",
    r"\bsaved\s+from\s+previous\b",
)

# -- P1.5: Serialization injection -----------------------------------------

_register(
    "serialization_injection", "P1.5", "lc_type_secret",
    r"\"lc\".{0,20}\"type\".{0,20}\"secret\"",
    is_extraction=True,
)
_register(
    "serialization_injection", "P1.5", "openai_api_key",
    r"\bOPENAI_API_KEY\b",
)
_register(
    "serialization_injection", "P1.5", "process_json_secret",
    r"\bprocess\s+this\s+JSON.{0,60}secret\b",
    is_extraction=True,
)

# -- P1.6: Membership inference ---------------------------------------------

_register(
    "membership_inference", "P1.6", "was_in_training_data",
    r"\bwas\s+.{1,60}\s+in\s+your\s+training\s+data\b",
)
_register(
    "membership_inference", "P1.6", "have_you_seen_before",
    r"\bhave\s+you\s+(?:seen|encountered)\s+.{1,60}\s+before\b",
)
_register(
    "membership_inference", "P1.6", "do_you_know_specific",
    r"\bdo\s+you\s+know\s+about\s+(?:[A-Z][a-z]+\s+){1,3}(?:personally|specifically)\b",
)


# ---------------------------------------------------------------------------
# Detection function
# ---------------------------------------------------------------------------

def detect_privacy_probe(text: str) -> Optional[PrivacyProbeResult]:
    """Detect privacy probes in *text*.

    Parameters
    ----------
    text : str
        The input text to scan for privacy probe patterns.

    Returns
    -------
    PrivacyProbeResult or None
        Detection result, or ``None`` if no probes found or the text
        matches an educational/contextual frame.
    """
    if not text or not text.strip():
        return None

    # Input length cap: avoid CPU exhaustion on very large inputs.
    if len(text) > 50_000:
        return None

    # Context suppression: educational or explanatory framing.
    if _EDUCATIONAL_FRAME.search(text):
        return None

    # Scan all patterns.
    matched_types: List[str] = []
    matched_technique_ids: List[str] = []
    has_extraction = False

    for probe_type, technique_id, _name, compiled, is_extraction in _PRIVACY_PATTERNS:
        match = compiled.search(text)
        if match:
            if probe_type not in matched_types:
                matched_types.append(probe_type)
            if technique_id not in matched_technique_ids:
                matched_technique_ids.append(technique_id)
            if is_extraction:
                has_extraction = True

    if not matched_types:
        return None

    # Check for PII indicators in the prompt text.
    has_pii = bool(_PII_INDICATORS.search(text))

    # Determine severity and confidence.
    confidence = _CONFIDENCE_BASE

    if has_extraction:
        severity = _SEVERITY_HIGH
        confidence = _CONFIDENCE_EXTRACTION
    elif has_pii:
        severity = _SEVERITY_MEDIUM
        confidence = _CONFIDENCE_BASE + _CONFIDENCE_PII_BOOST
    else:
        severity = _SEVERITY_LOW

    # Use the first matched probe type as the primary type.
    primary_type = matched_types[0]

    return PrivacyProbeResult(
        probe_type=primary_type,
        severity=severity,
        technique_ids=matched_technique_ids,
        confidence=confidence,
        pii_in_prompt=has_pii,
    )


# ---------------------------------------------------------------------------
# Rules for layer1/rules_registry.py integration
# ---------------------------------------------------------------------------

from .layer1.result import Rule as _Rule

PRIVACY_RULES = [
    _Rule(
        name="conversation_extraction",
        pattern=(
            r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:last\s+)?\d{0,4}\s*conversations?\b"
            r"|\bwhat\s+did\s+(?:the\s+)?(?:last|previous|prior)\s+user\s+(?:ask|say|tell|write|send|query)\b"
            r"|\b(?:session\s+data|cached\s+session)\b"
            r"|\b(?:chat\s+log|conversation\s+(?:log|history|data))\b"
        ),
        technique_ids=["P1.1"],
        severity="medium",
        paranoia_level=2,
        description="Extracts conversation history or session data.",
    ),
    _Rule(
        name="data_exfiltration_pii",
        pattern=(
            r"\bsummarize\s+(?:the\s+)?personal\s+information\b"
            r"|\bexfiltrat(?:e|ion)\b"
            r"|\btrack\?data="
        ),
        technique_ids=["P1.2"],
        severity="high",
        paranoia_level=1,
        description="Attempts to exfiltrate personal data or PII.",
    ),
    _Rule(
        name="training_data_completion",
        pattern=(
            r"\bcomplete\s+(?:the\s+)?(?:following\s+)?text\s+exactly\s+as\s+you\s+remember\b"
            r"|\brepeat\s+(?:the\s+)?word\s+.{1,40}\s+forever\b"
            r"|\brecite\s+.{0,40}\s*(?:training\s+data|memorized)\b"
            r"|\bword\s+for\s+word\b"
        ),
        technique_ids=["P1.3"],
        severity="high",
        paranoia_level=1,
        description="Extracts memorized training data via completion or repetition.",
    ),
    _Rule(
        name="cross_session_leakage",
        pattern=(
            r"\b(?:previous|prior|last)\s+user\b"
            r"|\b(?:shared\s+memory|memory\s+store)\b"
            r"|\bactive\s+sessions?\b"
            r"|\bsession\s+manager\b"
            r"|\bother\s+users?(?:'s?)?\s+(?:data|conversations?|information|queries)\b"
        ),
        technique_ids=["P1.4"],
        severity="medium",
        paranoia_level=2,
        description="Probes for cross-session data leakage or shared state.",
    ),
    _Rule(
        name="serialization_injection",
        pattern=(
            r"\"lc\".{0,20}\"type\".{0,20}\"secret\""
            r"|\bOPENAI_API_KEY\b"
            r"|\bprocess\s+this\s+JSON.{0,60}secret\b"
        ),
        technique_ids=["P1.5"],
        severity="high",
        paranoia_level=1,
        description="Serialization injection targeting secret extraction.",
    ),
    _Rule(
        name="membership_inference",
        pattern=(
            r"\bwas\s+.{1,60}\s+in\s+your\s+training\s+data\b"
            r"|\bhave\s+you\s+(?:seen|encountered)\s+.{1,60}\s+before\b"
        ),
        technique_ids=["P1.6"],
        severity="medium",
        paranoia_level=2,
        description="Membership inference probes against training data.",
    ),
]
