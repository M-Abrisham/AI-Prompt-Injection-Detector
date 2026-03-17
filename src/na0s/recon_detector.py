"""E2 Active Reconnaissance detection.

Detects adversarial probing of AI capabilities, model identity,
tool enumeration, boundary testing, and configuration extraction.

Two detection modes:
  Mode 1 (stateless): Single-turn pattern matching -> LOW severity
  Mode 2 (stateful):  Multi-probe escalation across turns -> MEDIUM/HIGH

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

# Severity levels for single-turn and multi-turn detection.
_SEVERITY_LOW = "low"
_SEVERITY_MEDIUM = "medium"
_SEVERITY_HIGH = "high"

# Multi-turn escalation thresholds: number of distinct probe categories
# observed across conversation turns that trigger escalation.
_MULTI_TURN_MEDIUM_THRESHOLD = 3
_MULTI_TURN_HIGH_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReconProbe:
    """A single reconnaissance probe detected in the input text."""

    category: str
    pattern_name: str
    matched_text: str


@dataclass
class ReconResult:
    """Aggregated result of reconnaissance detection."""

    probes_detected: List[ReconProbe] = field(default_factory=list)
    single_turn_severity: str = _SEVERITY_LOW
    multi_turn_severity: Optional[str] = None
    technique_ids: List[str] = field(default_factory=list)
    is_systematic: bool = False


# ---------------------------------------------------------------------------
# Educational / context suppression patterns
# ---------------------------------------------------------------------------
# When the user is asking *about* reconnaissance techniques (e.g. for a
# security course) rather than performing actual probing, suppress detection.

_EDUCATIONAL_FRAME = safe_compile(
    r"\b(?:"
    r"how\s+does\s+reconnaissance\s+work"
    r"|explain\s+(?:model\s+)?fingerprinting"
    r"|explain\s+(?:how\s+)?(?:reconnaissance|recon)"
    r"|for\s+my\s+(?:security|cybersecurity)\s+course"
    r"|(?:security|cybersecurity)\s+course"
    r"|what\s+is\s+(?:model\s+)?fingerprinting"
    r"|how\s+does\s+model\s+fingerprinting\s+work"
    r"|explain\s+model\s+fingerprinting"
    r"|how\s+do(?:es)?\s+(?:active\s+)?reconnaissance"
    r")\b",
    re.IGNORECASE,
    check_safety=True,
)


# ---------------------------------------------------------------------------
# Probe categories and compiled patterns
# ---------------------------------------------------------------------------
# Each entry: (category, technique_id, pattern_name, compiled_regex)
# Patterns use safe_compile for ReDoS safety.

_PROBE_PATTERNS: List[tuple] = []


def _register(category: str, technique_id: str, name: str, pattern: str) -> None:
    """Compile *pattern* and register it in the probe pattern list."""
    compiled = safe_compile(pattern, re.IGNORECASE, check_safety=True)
    _PROBE_PATTERNS.append((category, technique_id, name, compiled))


# -- E2.1: Capability probing -----------------------------------------------

_register(
    "capability_probing", "E2.1", "can_you_access",
    r"\b(?:can|could)\s+you\s+(?:access|browse|execute|read|write|run|fetch|download)\b",
)
_register(
    "capability_probing", "E2.1", "do_you_have_access",
    r"\bdo\s+you\s+have\s+(?:internet|file|database|tool|web|code|network)\s+access\b",
)
_register(
    "capability_probing", "E2.1", "what_capabilities",
    r"\bwhat\s+are\s+your\s+(?:capabilities|limitations|restrictions)\b",
)
_register(
    "capability_probing", "E2.1", "show_documents_access",
    r"\b(?:show|list|display|reveal)\s+(?:me\s+)?(?:the\s+)?(?:contents?|documents?|files?)\s+(?:(?:of\s+)?(?:the\s+)?(?:documents?\s+)?)?(?:you\s+have\s+access\s+to|in\s+your)\b",
)
_register(
    "capability_probing", "E2.1", "what_can_you_not_do",
    r"\bwhat\s+(?:can(?:'t|\s+not)\s+you\s+do|can\s+you\s+not\s+do)\b",
)

# -- E2.2: Tool / function enumeration --------------------------------------

_register(
    "tool_enumeration", "E2.2", "list_all_tools",
    r"\b(?:list|enumerate|show|display|describe|reveal)\s+(?:me\s+)?(?:all\s+)?(?:your\s+)?(?:tools|functions|plugins|capabilities|actions|endpoints)\b",
)
_register(
    "tool_enumeration", "E2.2", "what_actions_perform",
    r"\bwhat\s+(?:actions|operations|tasks)\s+can\s+you\s+perform\b",
)
_register(
    "tool_enumeration", "E2.2", "function_calling_schema",
    r"\bfunction\s+calling\s+schema\b",
)
_register(
    "tool_enumeration", "E2.2", "list_every_function",
    r"\blist\s+(?:every|each|all)\s+(?:function|tool|capability|action)\b",
)
_register(
    "tool_enumeration", "E2.2", "sub_agents",
    r"\b(?:sub[\s-]?agents|other\s+(?:AI\s+)?agents|other\s+assistants)\b",
)
_register(
    "tool_enumeration", "E2.2", "api_endpoints",
    r"\b(?:API\s+endpoints|available\s+plugins)\b",
)
_register(
    "tool_enumeration", "E2.2", "list_each_capability",
    r"\blist\s+each\s+capability\b",
)

# -- E2.3: Model fingerprinting ---------------------------------------------

_register(
    "model_fingerprinting", "E2.3", "what_model_are_you",
    r"\bwhat\s+model\s+are\s+you\b",
)
_register(
    "model_fingerprinting", "E2.3", "what_version",
    r"\bwhat\s+(?:version|release)\s+(?:are\s+you|is\s+this)\b",
)
_register(
    "model_fingerprinting", "E2.3", "context_window",
    r"\b(?:context\s+window|token\s+limit)\b",
)
_register(
    "model_fingerprinting", "E2.3", "training_cutoff",
    r"\b(?:training\s+cutoff|knowledge\s+cutoff|training\s+data\s+cutoff)\b",
)
_register(
    "model_fingerprinting", "E2.3", "are_you_gpt_claude",
    r"\bare\s+you\s+(?:GPT|Claude|Llama|Gemini|Mistral|LLaMA|ChatGPT)\b",
)
_register(
    "model_fingerprinting", "E2.3", "temperature_setting",
    r"\b(?:temperature\s+setting|sampling\s+parameters|top[\s_-]?p\s+setting|what\s+is\s+your\s+temperature)\b",
)
_register(
    "model_fingerprinting", "E2.3", "maximum_output_length",
    r"\b(?:maximum\s+output\s+length|max\s+(?:output|response)\s+(?:length|tokens))\b",
)
_register(
    "model_fingerprinting", "E2.3", "what_sampling_parameters",
    r"\bwhat\s+sampling\s+parameters\b",
)

# -- E2.4: Boundary testing -------------------------------------------------

_register(
    "boundary_testing", "E2.4", "what_happens_if",
    r"\bwhat\s+happens\s+if\s+I\s+(?:say|ask|type|write|send)\b",
)
_register(
    "boundary_testing", "E2.4", "where_are_limits",
    r"\bwhere\s+are\s+your\s+(?:limits|boundaries|guardrails)\b",
)
_register(
    "boundary_testing", "E2.4", "what_wont_you_do",
    r"\bwhat\s+(?:won'?t|will\s+not)\s+you\s+do\b",
)
_register(
    "boundary_testing", "E2.4", "safety_filters",
    r"\b(?:safety\s+filters|content\s+policy|content\s+filters|moderation\s+rules)\b",
)
_register(
    "boundary_testing", "E2.4", "how_can_i_identify_filters",
    r"\bhow\s+can\s+I\s+(?:identify|detect|bypass|circumvent)\s+(?:them|filters|guardrails|safety)\b",
)

# -- E2.5: Configuration extraction -----------------------------------------

_register(
    "config_extraction", "E2.5", "what_parameters",
    r"\bwhat\s+(?:parameters|settings|configuration)\s+(?:are|do)\b",
)
_register(
    "config_extraction", "E2.5", "describe_all_context",
    r"\bdescribe\s+(?:all\s+)?(?:the\s+)?context\s+you\b",
)
_register(
    "config_extraction", "E2.5", "system_messages",
    r"\bsystem\s+messages\b",
)
_register(
    "config_extraction", "E2.5", "conversation_context",
    r"\b(?:conversation|current)\s+context\b",
)
_register(
    "config_extraction", "E2.5", "list_everything_context",
    r"\blist\s+(?:everything|all)\s+(?:in\s+)?(?:your\s+)?(?:current\s+)?(?:conversation\s+)?context\b",
)


# ---------------------------------------------------------------------------
# Detection function
# ---------------------------------------------------------------------------

def detect_reconnaissance(
    text: str,
    conversation_history: Optional[List[str]] = None,
) -> Optional[ReconResult]:
    """Detect reconnaissance probes in *text*.

    Parameters
    ----------
    text : str
        The input text to scan for reconnaissance patterns.
    conversation_history : list[str] or None
        Previous turns in the conversation.  When provided, enables
        multi-turn escalation detection (Mode 2).

    Returns
    -------
    ReconResult or None
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

    # --- Mode 1: single-turn pattern matching ---
    probes: List[ReconProbe] = []
    technique_ids: List[str] = []
    categories_seen: set = set()

    for category, technique_id, pattern_name, compiled in _PROBE_PATTERNS:
        match = compiled.search(text)
        if match:
            probes.append(ReconProbe(
                category=category,
                pattern_name=pattern_name,
                matched_text=match.group(),
            ))
            if technique_id not in technique_ids:
                technique_ids.append(technique_id)
            categories_seen.add(category)

    if not probes:
        return None

    # Single-turn severity is always LOW for individual probes.
    single_turn_severity = _SEVERITY_LOW

    # --- Mode 2: multi-turn escalation ---
    multi_turn_severity: Optional[str] = None
    is_systematic = False

    if conversation_history is not None:
        # Collect categories from previous turns (cap at 100 entries).
        all_categories = set(categories_seen)
        for prev_text in conversation_history[:100]:
            if not prev_text or not prev_text.strip():
                continue
            for category, _tid, _pname, compiled in _PROBE_PATTERNS:
                if compiled.search(prev_text):
                    all_categories.add(category)

        unique_count = len(all_categories)
        if unique_count >= _MULTI_TURN_HIGH_THRESHOLD:
            multi_turn_severity = _SEVERITY_HIGH
            is_systematic = True
        elif unique_count >= _MULTI_TURN_MEDIUM_THRESHOLD:
            multi_turn_severity = _SEVERITY_MEDIUM
            is_systematic = True

    return ReconResult(
        probes_detected=probes,
        single_turn_severity=single_turn_severity,
        multi_turn_severity=multi_turn_severity,
        technique_ids=technique_ids,
        is_systematic=is_systematic,
    )


# ---------------------------------------------------------------------------
# Rules for layer1/rules_registry.py integration
# ---------------------------------------------------------------------------

from .layer1.result import Rule as _Rule

RECON_RULES = [
    _Rule(
        name="capability_probing",
        pattern=(
            r"\b(?:can|could)\s+you\s+(?:access|browse|execute|read|write|run|fetch|download)\b"
            r"|\bdo\s+you\s+have\s+(?:internet|file|database|tool|web|code|network)\s+access\b"
            r"|\bwhat\s+are\s+your\s+(?:capabilities|limitations|restrictions)\b"
            r"|\b(?:show|list|display|reveal)\s+(?:me\s+)?(?:the\s+)?(?:contents?|documents?|files?)\s+(?:(?:of\s+)?(?:the\s+)?(?:documents?\s+)?)?(?:you\s+have\s+access\s+to|in\s+your)\b"
        ),
        technique_ids=["E2.1"],
        severity="medium",
        paranoia_level=2,
        description="Probes AI system capabilities, access, or limitations.",
    ),
    _Rule(
        name="recon_tool_enumeration",
        pattern=(
            r"\b(?:list|enumerate|show|display|describe|reveal)\s+(?:me\s+)?(?:all\s+)?(?:your\s+)?(?:tools|functions|plugins|capabilities|actions|endpoints)\b"
            r"|\bwhat\s+(?:actions|operations|tasks)\s+can\s+you\s+perform\b"
            r"|\bfunction\s+calling\s+schema\b"
            r"|\blist\s+(?:every|each|all)\s+(?:function|tool|capability|action)\b"
            r"|\b(?:sub[\s-]?agents|other\s+(?:AI\s+)?agents|other\s+assistants)\b"
        ),
        technique_ids=["E2.2"],
        severity="medium",
        paranoia_level=2,
        description="Enumerates available tools, functions, or sub-agents.",
    ),
    _Rule(
        name="model_fingerprinting",
        pattern=(
            r"\bwhat\s+model\s+are\s+you\b"
            r"|\bwhat\s+(?:version|release)\s+(?:are\s+you|is\s+this)\b"
            r"|\bare\s+you\s+(?:GPT|Claude|Llama|Gemini|Mistral|LLaMA|ChatGPT)\b"
            r"|\b(?:temperature\s+setting|sampling\s+parameters)\b"
            r"|\b(?:maximum\s+output\s+length|max\s+(?:output|response)\s+(?:length|tokens))\b"
        ),
        technique_ids=["E2.3"],
        severity="medium",
        paranoia_level=2,
        description="Attempts to fingerprint the AI model identity or parameters.",
    ),
    _Rule(
        name="boundary_testing",
        pattern=(
            r"\bwhat\s+happens\s+if\s+I\s+(?:say|ask|type|write|send)\b"
            r"|\bwhere\s+are\s+your\s+(?:limits|boundaries|guardrails)\b"
            r"|\bwhat\s+(?:won'?t|will\s+not)\s+you\s+do\b"
            r"|\b(?:safety\s+filters|content\s+policy|content\s+filters|moderation\s+rules)\b"
        ),
        technique_ids=["E2.4"],
        severity="medium",
        paranoia_level=2,
        description="Tests AI system boundaries, safety filters, or limits.",
    ),
    _Rule(
        name="config_extraction",
        pattern=(
            r"\bdescribe\s+(?:all\s+)?(?:the\s+)?context\s+you\b"
            r"|\bsystem\s+messages\b"
            r"|\b(?:conversation|current)\s+context\b"
            r"|\blist\s+(?:everything|all)\s+(?:in\s+)?(?:your\s+)?(?:current\s+)?(?:conversation\s+)?context\b"
        ),
        technique_ids=["E2.5"],
        severity="medium",
        paranoia_level=2,
        description="Attempts to extract configuration or context information.",
    ),
]
