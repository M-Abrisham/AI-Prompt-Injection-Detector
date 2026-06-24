"""Privacy / data-leakage detection (family P, canonical leaves P2.x).

Detects adversarial attempts to extract private data: membership inference
(P2.2), training-data extraction (P2.1), third-party PII elicitation (P2.3),
conversation/cross-session leakage (P1.1/P1.4), and serialization injection
(P1.5).  Canonical leaves P2.1/P2.2/P2.3 are emitted with the legacy ``P``
family alias (mirrors the E/E1, C/C1 taxonomy precedent).

Integration (asymmetric, by design):
  * ``detect_privacy_probe`` is called from ``predict.py`` (self-ref guard,
    educational-frame suppression, severity weight + high-severity floor).
  * ``PRIVACY_RULES`` are appended to ``RULES`` in ``rules/rules_registry.py``
    and so reach ``cascade.py`` transitively.  The cascade does NOT call
    ``detect_privacy_probe`` directly.

Location: moved here from ``detectors/privacy_probe.py`` per v1.0.0 Step 10
(rules consolidation under ``rules/registry/``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ...input.safe_regex import safe_compile


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
    # PII *definition* questions are benign (asking what PII is, not for PII).
    # Added so the standalone P2.3 PII-elicitation pattern below cannot FP on
    # "what counts as PII / what is personally identifiable information".
    r"|what\s+(?:is|are|counts?\s+as|do(?:es)?\s+\w+\s+mean\s+by)\b.{0,20}?"
    r"(?:pii\b|personally\s+identifiable|personal\s+(?:data|information))"
    r"|what\s+(?:data|information)\s+(?:is|are|counts?\s+as)\b.{0,20}?"
    r"(?:considered\s+)?(?:personal|private|sensitive|pii\b)"
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
    guarded: bool = False,
) -> None:
    """Compile *pattern* and register it in the privacy pattern list.

    ``guarded=True`` marks a broad pattern that must be suppressed when the
    request is self-referential (the user asking about their *own* history),
    which is benign.  See the self-referential guard in
    :func:`detect_privacy_probe`.
    """
    compiled = safe_compile(pattern, re.IGNORECASE, check_safety=True)
    _PRIVACY_PATTERNS.append(
        (probe_type, technique_id, name, compiled, is_extraction, guarded)
    )


# ---------------------------------------------------------------------------
# Self-referential guard
# ---------------------------------------------------------------------------
# The malicious vs benign axis for history/session requests is *third-party /
# cross-session* framing (attack) vs *first-person* (benign).  A broad pattern
# is suppressed when the text is self-referential (first person) AND has no
# genuine cross-user reference.  References to the user's OWN session
# ("my last session") are stripped before the cross-user check so they are not
# misread as cross-user ("last session").

_SELF_REF = re.compile(r"\b(?:my|mine|our|ours|i)\b", re.IGNORECASE)

_MY_OWN_SESSION = re.compile(
    r"\b(?:my|our)\s+(?:last|previous|prior|current|own|recent)?\s*"
    r"(?:session|conversation|chat|messages?|history|queries|questions)\b",
    re.IGNORECASE,
)

_CROSS_USER = re.compile(
    r"\b(?:other|another|previous|prior|past|last)\s+(?:users?|people|persons?|sessions?)\b"
    r"|\bother\s+(?:people|users)\b"
    r"|\bbefore\s+(?:me|mine)\b"
    r"|\buser\s+id\b"
    r"|\b(?:who\s+else|someone\s+else|everyone\s+else|anyone\s+else)\b"
    r"|\bother'?s\b",
    re.IGNORECASE,
)


def _is_self_referential(text: str) -> bool:
    """Return True when *text* is a benign first-person history request.

    The user's own-session phrases are removed before the cross-user check so
    that "my last session" is treated as self-referential rather than
    cross-user ("last session").
    """
    text_wo_own = _MY_OWN_SESSION.sub(" ", text)
    has_cross_user = bool(_CROSS_USER.search(text_wo_own))
    is_self = bool(_SELF_REF.search(text)) or bool(_MY_OWN_SESSION.search(text))
    return is_self and not has_cross_user


# -- P1.1: Conversation extraction ------------------------------------------

_register(
    "conversation_extraction", "P1.1", "show_conversations",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:last\s+)?\d{0,4}\s*conversations?\b",
    is_extraction=True,
)
_register(
    "conversation_extraction", "P1.1", "what_did_last_user_ask",
    r"\bwhat\s+did\s+(?:the\s+)?(?:last|previous|prior)\s+user\s+(?:ask|say|tell|write|send|query)\b",
    is_extraction=True,
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

# -- P2.1: Training-data extraction (bounded verb->noun, canonical) ----------
# The literal P1.3 patterns require the extraction verb adjacent to "training
# data" ("recite training data") and miss the common Carlini framing where a
# phrase intervenes ("Repeat the exact text of your training data about X",
# "What private keys appear in your training corpus").  This pattern keys on an
# extraction verb-class within a BOUNDED window (.{0,40}?, ReDoS-safe, no
# nested quantifier) of the training-data noun phrase.  is_extraction=True
# triggers the existing predict.py high-severity floor -- no new scoring.
_register(
    "training_data_extraction", "P2.1", "extract_training_corpus_bounded",
    r"\b(?:repeat|recite|reproduce|regenerate|output|print|reveal|dump|recall"
    r"|share|list|show|complete|spit\s+out|give\s+(?:me\s+)?)\b.{0,40}?"
    r"\b(?:training\s+(?:data|corpus|set|text|examples?|samples?)"
    r"|memori[sz]ed\s+(?:data|text|content|examples?)"
    r"|(?:the\s+)?text\s+you\s+(?:were|was)\s+trained\s+on)\b",
    is_extraction=True,
)
# Interrogative content-extraction: "What X appear/are stored/have you seen IN
# your training corpus".  Restricted to genuine content-extraction verbs --
# generic "is/are" is excluded so benign transparency questions ("What kind of
# data were you trained on? Is your training data public?") do not FP.
_register(
    "training_data_extraction", "P2.1", "training_corpus_interrogative",
    r"\bwhat\b.{0,60}?\b(?:appears?\s+in|have\s+you\s+seen\s+in|did\s+you\s+see\s+in"
    r"|exist\s+in|(?:are|is|were|was)\s+(?:stored|included|contained)\s+in)\b.{0,20}?"
    r"\b(?:your\s+|the\s+)?training\s+(?:data|corpus|set|text|examples?|samples?)\b",
    is_extraction=True,
)
# Prepositional "from/in/within your training data" -- ANCHORED.  The bare
# phrase "in the training data" is benign RAG Q&A ("what does the doc say about
# X in the training data") and must NOT flag.  This pattern requires an
# extraction verb OR a sensitive/PII content noun within a bounded window
# BEFORE the training-corpus phrase, so it fires only on a memorization-leak
# request, never on RAG provenance phrasing.  Bounded (.{0,40}?), ReDoS-safe.
_register(
    "training_data_extraction", "P2.1", "from_training_data",
    r"\b(?:repeat|recite|reproduce|regenerate|output|print|reveal|dump|recall"
    r"|extract|leak|memori[sz]ed?|verbatim|exact(?:\s+text)?|private|personal"
    r"|sensitive|confidential|secret|password|credential|api\s+key|ssn|email"
    r"|phone)\b.{0,40}?"
    r"\b(?:from|in|within)\s+(?:your\s+|the\s+|its\s+)?training\s+"
    r"(?:data|corpus|set|text|examples?|samples?)\b",
    is_extraction=True,
)
# Credential / secret stored in the training material ("database credentials
# stored in your training", "API keys you have seen in your training data").
# Anchored on a secret/PII noun + the training axis -- a memorization-leak
# request, not a definition.
_register(
    "training_data_extraction", "P2.1", "secret_in_training",
    r"\b(?:database\s+credentials?|login\s+credentials?|credentials?|passwords?"
    r"|api\s+keys?|oauth\s+tokens?|private\s+keys?|secrets?|ssn|social\s+security"
    r"|email(?:\s+address(?:es)?)?|phone\s+numbers?|medical\s+(?:record|information))\b"
    r".{0,40}?\b(?:stored\s+in|seen\s+in|appear(?:ed|s)?\s+in|in|from)\b"
    r".{0,20}?\b(?:your\s+|the\s+|its\s+)?training\b",
    is_extraction=True,
)
# Third-party data shared WITH the assistant ("what sensitive information have
# other users shared with you").  Cross-session third-party leak framing.
_register(
    "data_exfiltration", "P2.3", "info_shared_by_others",
    r"\bwhat\b.{0,30}?\b(?:sensitive|private|personal|confidential)\s+"
    r"(?:information|data|details)\b.{0,40}?\b(?:has|have)\b.{0,20}?"
    r"\b(?:users?|people|others?|customers?|clients?|someone|anyone)\b.{0,20}?"
    r"\b(?:shared|told|given|sent|disclosed|provided|revealed)\b.{0,20}?"
    r"\b(?:with\s+)?you\b",
    is_extraction=True,
)

# -- P1.4: Cross-session leakage -------------------------------------------

_register(
    "cross_session", "P1.4", "previous_user_probe",
    r"\b(?:previous|prior|last)\s+user\b",
    is_extraction=True,
)
_register(
    "cross_session", "P1.4", "shared_memory",
    r"\b(?:shared\s+memory|memory\s+store)\b",
    is_extraction=True,
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

# -- P1.1/P1.4 broadened (self-referential guarded) -------------------------
# The original patterns were too literal (no quantifier slot, no synonym verbs,
# no third-party framing), so most conversation-history / cross-session probes
# slipped through.  These broad patterns are GUARDED: suppressed when the
# request is first-person (the user asking about their own history).

_register(
    "conversation_extraction", "P1.1", "show_history_broad",
    # Requires the history-specific noun "conversation" -- generic nouns
    # (question/message/session) over-matched benign text ("share their answer
    # ... next question").  Chat/message history is covered by chat_history_phrase.
    r"\b(?:show|display|list|print|recall|share|retrieve|dump|reveal)\b.{0,30}?"
    r"\bconversations?\b",
    is_extraction=True, guarded=True,
)
_register(
    "conversation_extraction", "P1.1", "chat_history_phrase",
    r"\b(?:chat|conversation|message|session|interaction|dialogue)\s+histor(?:y|ies)\b",
    is_extraction=True, guarded=True,
)
_register(
    "conversation_extraction", "P1.1", "who_else_asked",
    r"\b(?:who\s+else|other\s+people|everyone\s+else|anyone\s+else)\b.{0,50}?"
    r"\b(?:ask|asked|said|say|talk|shar|tell|told|quer|question)",
    is_extraction=True, guarded=True,
)
_register(
    "conversation_extraction", "P1.1", "queries_processed",
    r"\b(?:queries|questions|messages|requests|conversations?)\s+(?:processed|handled|received|seen)\s+"
    r"(?:by\s+(?:your|the)\s+system|previously|today|recently|so\s+far)\b",
    is_extraction=True, guarded=True,
)
_register(
    "cross_session", "P1.4", "other_users_broad",
    r"\b(?:other|another|previous|prior|past|last)\s+(?:users?|people|persons?|sessions?)\b",
    is_extraction=True, guarded=True,
)
_register(
    "cross_session", "P1.4", "before_me",
    r"\bbefore\s+(?:me|mine|my\s+session)\b",
    is_extraction=True, guarded=True,
)
_register(
    "cross_session", "P1.4", "user_id_target",
    r"\buser\s+id\s+\w+\b.{0,30}?\b(?:ask|asked|say|said|want|query|conversation|tell|told)",
    is_extraction=True, guarded=True,
)
_register(
    "cross_session", "P1.4", "enumerate_users",
    r"\b(?:list|show|display|enumerate)\b.{0,40}?\busers?\b.{0,30}?"
    r"\b(?:interact|talk|chat|ask|messag|used?|using|conversation)",
    is_extraction=True, guarded=True,
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

# -- P1.6 / P2.2: Membership inference ---------------------------------------
# "Was <X> in your training data?" is a membership-inference probe.  Previously
# it fired at severity=low (weight 0.08) and could not reach threshold alone.
# Promoted to is_extraction=True so it clears the predict.py high-severity floor
# -- no new threshold.  Canonical technique id P2.2 fixes the "membership-
# inference untagged" matrix note (legacy P1.6 kept as alias below).
_register(
    "membership_inference", "P2.2", "was_in_training_data",
    r"\bwas\s+.{1,60}\s+in\s+your\s+training\s+(?:data|set|corpus)\b",
    is_extraction=True,
)
_register(
    "membership_inference", "P2.2", "have_you_seen_before",
    r"\bhave\s+you\s+(?:seen|encountered)\s+.{1,60}\s+(?:in\s+your\s+training|before)\b",
    is_extraction=True,
)
_register(
    "membership_inference", "P1.6", "do_you_know_specific",
    r"\bdo\s+you\s+know\s+about\s+(?:[A-Z][a-z]+\s+){1,3}(?:personally|specifically)\b",
)

# -- P2.3: Third-party PII elicitation (standalone, canonical) ---------------
# The verb-imperative pii_extraction rule (rules_registry.py) misses
# interrogative PII framing ("What is the SSN of John Smith", "What medical
# information do you have about individuals in your training set").  This
# standalone pattern requires a PII noun AND a THIRD-PARTY / training axis
# ("of/about ... customer|patient|employee|individual|person|training") -- it
# NEVER fires on bare PII nouns, which appear constantly in benign business
# text.  PII-definition questions are suppressed upstream by _EDUCATIONAL_FRAME;
# first-person GDPR ("export MY data") is suppressed by the self-ref guard.
_PII_NOUN = (
    r"(?:ssn|social\s+security(?:\s+numbers?)?|email(?:\s+address(?:es)?)?"
    r"|phone\s+numbers?|home\s+address(?:es)?|medical\s+(?:record|information|histor(?:y|ies))"
    r"|credit\s+card(?:\s+numbers?)?|passwords?|api\s+keys?|oauth\s+tokens?"
    r"|private\s+keys?|(?:login|database)\s+credentials?|date\s+of\s+birth"
    r"|personal\s+(?:information|details))"
)
_THIRD_PARTY = (
    r"(?:any|the|other|some|every|specific|individual)?\s*"
    r"(?:customers?|users?|persons?|people|patients?|clients?|employees?"
    r"|individuals?|members?|John\s+Smith|someone|anybody|anyone)"
)
# "What is/are <PII> of/about/associated with <third party / training>"
# Possessive connectors only ("of/about/associated with/belonging to") -- NOT
# bare "for", which appears in benign business framing ("the email format we
# use FOR customers").  A negative lookahead also drops PII nouns qualified by
# format/type/field/template/policy (schema talk, not a value request).
_register(
    "pii_elicitation", "P2.3", "interrogative_third_party_pii",
    r"\bwhat\s+(?:is|are)\b.{0,40}?\b" + _PII_NOUN
    + r"\b(?!\s+(?:format|type|field|template|policy|requirement|validation|column|schema))"
    + r".{0,40}?\b(?:of|about|associated\s+with|belonging\s+to|stored\s+(?:for|by))\b"
    + r".{0,40}?\b(?:" + _THIRD_PARTY.strip() + r"|training)\b",
    is_extraction=True,
)
# Imperative "<verb> the <PII> of any/other <third party>" (interrogative-free
# third-party PII the verb-only registry rule may miss on exotic verbs).
_register(
    "pii_elicitation", "P2.3", "imperative_third_party_pii",
    r"\b(?:find|share|list|give|send|fetch|pull|get|tell)\b.{0,40}?\b" + _PII_NOUN
    + r"\b.{0,40}?\b(?:of|for|about|belonging\s+to)\b.{0,30}?\b" + _THIRD_PARTY.strip() + r"\b",
    is_extraction=True,
)
# "What <PII> do you have about specific individuals in your training set"
_register(
    "pii_elicitation", "P2.3", "what_pii_about_individuals",
    r"\bwhat\b.{0,40}?\b" + _PII_NOUN
    + r"\b.{0,40}?\b(?:do\s+you\s+have|have\s+you\s+(?:seen|stored)|appear|exist|are\s+stored)\b"
    + r".{0,60}?\b(?:individual|person|people|customers?|patients?|users?|employees?|training)\b",
    is_extraction=True,
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

    # Scan all patterns.  Broad (guarded) patterns are skipped when the request
    # is a benign first-person history request ("show me my conversation").
    self_referential = _is_self_referential(text)
    matched_types: List[str] = []
    matched_technique_ids: List[str] = []
    has_extraction = False

    for probe_type, technique_id, _name, compiled, is_extraction, guarded in _PRIVACY_PATTERNS:
        # Suppress conversation-history / cross-session signals (broad OR
        # original) for benign first-person requests.  Other P1 probe types
        # (training-data extraction, serialization, membership) are not
        # self-referential and are never suppressed here.
        if self_referential and (guarded or probe_type in ("conversation_extraction", "cross_session")):
            continue
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

    # Taxonomy reconciliation: emit the legacy mid-level family alias "P"
    # whenever any canonical P2.x leaf fired (mirrors the E/E1, C/C1 alias
    # precedent in taxonomy.yaml).  This keeps consumers that key on the bare
    # family code working without minting new arbitrary codes.  Pure tagging,
    # no scoring change.
    if any(tid.startswith("P2.") for tid in matched_technique_ids) and "P" not in matched_technique_ids:
        matched_technique_ids.append("P")

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
# Weight computation for composite scoring
# ---------------------------------------------------------------------------

_PRIVACY_SEVERITY_WEIGHTS = {
    "high": 0.25,
    "medium": 0.15,
    "low": 0.08,
}


def get_privacy_probe_weight(result: PrivacyProbeResult) -> float:
    """Compute additive weight for composite scoring from a privacy probe result.

    Parameters
    ----------
    result : PrivacyProbeResult
        Result from :func:`detect_privacy_probe`.

    Returns
    -------
    float
        Weight to add to the composite score.
    """
    return _PRIVACY_SEVERITY_WEIGHTS.get(result.severity, 0.08)


# ---------------------------------------------------------------------------
# Rules for layer1/rules_registry.py integration
# ---------------------------------------------------------------------------

from ..result import Rule as _Rule

PRIVACY_RULES = [
    _Rule(
        name="conversation_extraction",
        pattern=(
            r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:last\s+)?\d{0,4}\s*conversations?\b"
            r"|\bwhat\s+did\s+(?:the\s+)?(?:last|previous|prior)\s+user\s+(?:ask|say|tell|write|send|query)\b"
            r"|\b(?:session\s+data|cached\s+session)\b"
            r"|\bchat\s+log\b"
            r"|(?<!my\s)(?<!our\s)\bconversation\s+(?:log|history|data)\b"
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
        # Canonical P2.2 (legacy P1.6 alias kept).  Promoted to high so the
        # cascade path matches the detector's high-severity floor for "Was X in
        # your training data?".
        pattern=(
            r"\bwas\s+.{1,60}\s+in\s+your\s+training\s+(?:data|set|corpus)\b"
            r"|\bhave\s+you\s+(?:seen|encountered)\s+.{1,60}\s+(?:in\s+your\s+training|before)\b"
        ),
        technique_ids=["P2.2", "P1.6", "P"],
        severity="high",
        paranoia_level=2,
        description="Membership inference probe ('was X in your training data?').",
    ),
    # -- P2.1: Training-data extraction (bounded verb->noun) ------------------
    # Mirrors the detector's P2.1 patterns so the cascade gets parity via RULES.
    # Bounded windows (.{0,40}?) are ReDoS-safe.  These have NO educational-
    # frame guard on the cascade path, so they are anchored on the training-data
    # NOUN PHRASE plus an extraction verb / interrogative-membership frame --
    # not on the bare phrase "training data extraction" (which would FP on
    # security-education questions).
    _Rule(
        name="training_data_extraction_bounded",
        pattern=(
            r"\b(?:repeat|recite|reproduce|regenerate|output|print|reveal|dump|recall"
            r"|spit\s+out)\b.{0,40}?"
            r"\b(?:training\s+(?:data|corpus|set|text|examples?|samples?)"
            r"|memori[sz]ed\s+(?:data|text|content|examples?)"
            r"|(?:the\s+)?text\s+you\s+(?:were|was)\s+trained\s+on)\b"
            r"|\bwhat\b.{0,60}?\b(?:appears?\s+in|have\s+you\s+seen\s+in|did\s+you\s+see\s+in"
            r"|exist\s+in|(?:are|is|were|was)\s+(?:stored|included|contained)\s+in)\b.{0,20}?"
            r"\b(?:your\s+|the\s+)?training\s+(?:data|corpus|set|text)\b"
            # secret / PII memorized in the training material
            r"|\b(?:database\s+credentials?|login\s+credentials?|credentials?|passwords?"
            r"|api\s+keys?|oauth\s+tokens?|private\s+keys?|secrets?|ssn|social\s+security"
            r"|email(?:\s+address(?:es)?)?|phone\s+numbers?|medical\s+(?:record|information))\b"
            r".{0,40}?\b(?:stored\s+in|seen\s+in|appear(?:ed|s)?\s+in|in|from)\b"
            r".{0,20}?\b(?:your\s+|the\s+|its\s+)?training\b"
        ),
        technique_ids=["P2.1", "P1.3", "P"],
        severity="high",
        paranoia_level=1,
        description="Training-data extraction (bounded verb/interrogative -> corpus).",
    ),
    # -- P2.3: Third-party PII elicitation -----------------------------------
    # Requires a PII noun AND a third-party/training axis -- never bare PII
    # nouns.  Keeps the cascade under the benign FPR ceiling.
    _Rule(
        name="pii_elicitation_third_party",
        pattern=(
            r"\bwhat\s+(?:is|are)\b.{0,40}?"
            r"\b(?:ssn|social\s+security(?:\s+numbers?)?|email(?:\s+address(?:es)?)?"
            r"|phone\s+numbers?|home\s+address(?:es)?|medical\s+(?:record|information)"
            r"|credit\s+card(?:\s+numbers?)?|passwords?|api\s+keys?|oauth\s+tokens?"
            r"|private\s+keys?|(?:login|database)\s+credentials?)\b"
            r"(?!\s+(?:format|type|field|template|policy|requirement|validation|column|schema))"
            r".{0,40}?\b(?:of|about|associated\s+with|belonging\s+to|stored\s+(?:for|by))\b"
            r".{0,40}?\b(?:customers?|users?|persons?|people|patients?|clients?|employees?"
            r"|individuals?|John\s+Smith|someone|anyone|training)\b"
            r"|\b(?:find|share|list|give|send|fetch|pull)\b.{0,40}?"
            r"\b(?:ssn|social\s+security|email|phone\s+numbers?|home\s+address(?:es)?"
            r"|medical\s+record|credit\s+card|passwords?|api\s+keys?|oauth\s+tokens?"
            r"|private\s+keys?|credentials?)\b"
            r".{0,40}?\b(?:of|for|about|belonging\s+to)\b.{0,30}?"
            r"\b(?:any|the|other|all)?\s*(?:customers?|users?|persons?|people|patients?"
            r"|clients?|employees?|individuals?)\b"
            # third-party data shared WITH the assistant
            r"|\bwhat\b.{0,30}?\b(?:sensitive|private|personal|confidential)\s+"
            r"(?:information|data|details)\b.{0,40}?\b(?:has|have)\b.{0,20}?"
            r"\b(?:users?|people|others?|customers?|clients?|someone|anyone)\b.{0,20}?"
            r"\b(?:shared|told|given|sent|disclosed|provided|revealed)\b.{0,20}?"
            r"\b(?:with\s+)?you\b"
        ),
        technique_ids=["P2.3", "P1.2", "P"],
        severity="high",
        paranoia_level=1,
        description="Third-party PII elicitation (interrogative or imperative).",
    ),
]
