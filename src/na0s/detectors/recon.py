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

from ..layer0.safe_regex import safe_compile


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

from ..layer1.result import Rule as _Rule

RECON_RULES = [
    # E2.1 -- Source / retrieval-context probing.  Internals-focused: asks what
    # data/files/sources were loaded into the model's context.  (Broad benign
    # capability questions like "can you browse the web?" were intentionally
    # dropped -- they are not reconnaissance and caused false positives.)
    _Rule(
        name="capability_probing",
        pattern=(
            r"\bwhat\s+(?:sources?\s+of\s+information|files?|documents?|data|context|inputs?|materials?)\s+(?:were|was|are|have\s+been|did\s+(?:you|they))\s+(?:provided|uploaded|given|loaded|shared|injected|added|attached)\b"
            r"|\bwhat(?:'s| is| are)\s+in\s+your\s+(?:context\s+window|context|retrieval\s+(?:context|set)|knowledge\s+base|rag\s+(?:context|sources?)|loaded\s+(?:files|documents))\b"
            r"|\bwhat\s+(?:retrieval|rag|knowledge|data|grounding)\s+sources?\s+(?:do\s+you|are\s+you|were\s+you)\b"
            r"|\bshow\s+(?:me\s+)?(?:the\s+)?(?:contents?|documents?|files?|sources?)\s+(?:you\s+have\s+access\s+to|in\s+your\s+(?:context|retrieval|knowledge))\b"
            r"|\bdo\s+you\s+have\s+access\s+to\s+(?:the\s+)?(?:internet|external\s+tools?|web|other\s+(?:tools|systems))\b"
        ),
        technique_ids=["E2.1"],
        severity="high",
        paranoia_level=2,
        description="Probes which sources/files/data were loaded into the model context.",
    ),
    # E2.2 -- Tool / function / knowledge-base enumeration.  Enumeration intent
    # (list/enumerate/dump/full-list) over internal assets.
    _Rule(
        name="recon_tool_enumeration",
        pattern=(
            r"\b(?:list|enumerate|show|display|reveal|dump|give\s+me)\s+(?:me\s+)?(?:all\s+|every\s+|a\s+(?:full|complete)\s+list\s+of\s+|the\s+(?:full|complete|entire)\s+list\s+of\s+)?your\s+(?:tools|functions|plugins|integrations|connectors|capabilities|actions|api\s+endpoints|endpoints|mcp\s+servers?)\b"
            r"|\b(?:full|complete|entire)\s+list\s+of\s+your\s+(?:capabilities|tools|functions|plugins|integrations)\b"
            r"|\bwhat\s+apis?\s+(?:are\s+you\s+connected\s+to|do\s+you\s+have\s+access\s+to|do\s+you\s+integrate\s+with)\b"
            r"|\bwhat\s+apis?\s+do\s+you\s+use\b(?!\s+to\s+\w)"
            r"|\blist\s+(?:every|each|all)\s+(?:of\s+your\s+)?(?:function|tool|capability|action|plugin|integration)s?\s+(?:you\s+(?:have|can|expose|provide|offer)|available\s+to\s+you)\b"
            r"|\blist\s+all\s+your\s+(?:tools|functions|plugins|integrations|capabilities)\b"
            r"|\byour\s+(?:function\s+calling|openapi|swagger)\s+(?:schema|spec|specification|definition)\b"
            r"|\b(?:list|enumerate|dump|show|reveal)\s+(?:all\s+)?(?:documents?|files?|sources?|entries|records?)\s+in\s+your\s+(?:knowledge\s+base|retrieval|vector\s+(?:database|store|db)|index|rag|memory)\b"
            r"|\bwhat\s+(?:actions|operations|tasks)\s+can\s+you\s+perform\b"
            r"|\byour\s+(?:sub[\s-]?agents|other\s+(?:AI\s+)?agents|other\s+assistants)\b"
        ),
        technique_ids=["E2.2"],
        severity="high",
        paranoia_level=2,
        description="Enumerates available tools, functions, integrations, or knowledge sources.",
    ),
    # E2.3 -- Model / config fingerprinting.  HIGH + floor-eligible: probing the
    # model's identity, architecture, parameters, sampling config, hardware, or
    # context window is reconnaissance against an embedded SDK.  The benign
    # holdout contains no such self-fingerprinting questions, so measured FPR is
    # unaffected.  (Trade-off: bare "what model are you?" is now flagged -- a
    # deliberate strict-threat-model choice for a defensive SDK.)
    _Rule(
        name="model_fingerprinting",
        pattern=(
            r"\bwhat\s+(?:ai\s+)?model\s+(?:are\s+you|is\s+this\s+(?:assistant|model|ai|system))\b"
            r"|\bwhat\s+(?:ai\s+)?model\s+are\s+you\s+(?:based|built|running)\b"
            r"|\bwhat\s+(?:version|release)\s+are\s+you\b"
            r"|\bwhat\s+(?:version|release)\s+is\s+this\s+(?:assistant|model|ai|system|agent)\b"
            r"|\bare\s+you\s+(?:gpt|claude|llama|gemini|mistral|chatgpt|bard)(?:-?[\d.]+)?\b"
            r"|\bhow\s+many\s+parameters\s+(?:do\s+you|does\s+your\s+(?:underlying\s+)?model|are\s+you)\b"
            r"|\bwhat\s+(?:is\s+)?your\s+(?:exact\s+)?(?:model|system)?\s*architecture\b"
            r"|\b(?:exact\s+)?(?:model\s+)?architecture\s+(?:(?:are\s+)?you(?:'?re)?|you\s+are)\s+running\s+on\b"
            r"|\bwhat\s+embedding\s+dimensions?\s+(?:do(?:es)?\s+(?:you|your\s+model)|are\s+you)\b"
            r"|\bwhat\s+(?:is\s+)?your\s+embedding\s+dimensions?\b"
            r"|\bwhat\s+is\s+your\s+context\s+window(?:\s+size)?\b"
            r"|\byour\s+context\s+window\s+size\s+in\s+tokens\b"
            r"|\bwhat\s+is\s+your\s+(?:maximum|max)\s+(?:response|output)\s+(?:length|tokens?)\b"
            r"|\bwhat\s+(?:training\s+(?:data\s+)?cutoff|knowledge\s+cutoff)\s+(?:date\s+)?do\s+you\b"
            r"|\byour\s+(?:training\s+(?:data\s+)?cutoff|knowledge\s+cutoff)\b"
            r"|\bwhat(?:'s| is)?\s+your\s+(?:temperature|top[\s_-]?p|sampling\s+parameters?)\b"
            r"|\b(?:temperature|top[\s_-]?p|sampling\s+parameters?)\s+(?:are\s+you|you'?re|(?:are\s+)?you\s+(?:currently\s+)?using)\b"
            r"|\b(?:are\s+you\s+running|running)\s+inference\s+on\s+(?:gpu|cpu)\b"
            r"|\bare\s+you\s+(?:running\s+(?:on\s+)?|on\s+)(?:a\s+)?(?:gpu|cpu)\b"
            r"|\bwhat\s+hardware\s+(?:are\s+you|you'?re|do\s+you)\b"
        ),
        technique_ids=["E2.3"],
        severity="high",
        paranoia_level=2,
        description="Fingerprints model identity, architecture, parameters, or config.",
    ),
    # E2.4 -- Safety/defense reconnaissance: mapping filters, restrictions,
    # refusal thresholds, safety training, and bypass surfaces.
    _Rule(
        name="boundary_testing",
        pattern=(
            r"\bwhat\s+(?:specific\s+)?(?:words?|inputs?|phrases?|prompts?|keywords?)\s+(?:trigger|bypass|evade|trip|defeat)\s+(?:your\s+)?(?:content\s+|safety\s+)?(?:filter|moderation|safety|guardrails?)"
            r"|\bdescribe\s+(?:your\s+)?(?:safety|security|moderation)\s+(?:training|architecture|system|mechanisms?|configuration)\b"
            r"|\bhow\s+(?:can\s+i|do\s+i|to)\s+(?:identify|detect|bypass|circumvent|evade|disable)\s+(?:your\s+)?(?:filters?|guardrails?|safety|moderation|content\s+(?:policy|filters?))"
            r"|\bwhat\s+(?:are\s+your\s+)?(?:detection\s+thresholds?|safety\s+filters?|content\s+(?:filters?|restrictions?)|moderation\s+rules?|prohibited\s+topics?)\b"
            r"|\blist\s+(?:every|all)\s+(?:prohibited|forbidden|restricted|banned)\s+(?:topic|thing|subject)"
            r"|\blist\s+all\s+(?:the\s+)?things\s+you\s+(?:are\s+not|aren'?t|can'?t|cannot)\s+(?:allowed\s+to\s+do|do)"
            r"|\bwhat\s+(?:is\s+the\s+)?threshold\s+for\s+refusing"
            r"|\bdo\s+you\s+use\s+(?:rlhf|constitutional\s+ai)"
            r"|\b(?:are\s+there|what\s+are\s+the)\s+(?:magic\s+words?|special\s+(?:words?|phrases?))"
            r"|\bwould\s+you\s+enter\s+(?:a\s+)?(?:special|secret|developer|admin|debug)\s+mode"
        ),
        technique_ids=["E2.4"],
        severity="high",
        paranoia_level=2,
        description="Reconnaissance of safety filters, restrictions, and refusal behavior.",
    ),
    # E2.5 -- Configuration / context extraction and permission probing.
    _Rule(
        name="config_extraction",
        pattern=(
            r"\bcan\s+you\s+(?:modify|change|edit|access|view|read|see|reveal)\s+(?:your\s+)?(?:own\s+)?(?:system\s+prompt|configuration|config|settings|instructions)\b"
            r"|\bwhat\s+(?:permissions?|privileges?|access\s+(?:rights?|levels?))\s+do\s+you\s+have\b(?!\s+(?:to\s+(?:access|read|view|see|modify|edit|write|delete)\s+(?:my|the|this|our)|over\s+(?:my|the|this|our)|on\s+(?:my|the|this|our)))"
            r"|\b(?:were|was)\s+(?:any\s+)?(?:user\s+profiles?|account\s+details?|credentials?|api\s+keys?|secrets?)\b.{0,40}?\b(?:shared|provided|given|passed)\s+(?:with|to)\s+you\b"
            r"|\bdescribe\s+(?:all\s+)?(?:the\s+)?context\s+you\b"
            r"|\bwhat(?:'s| is)\s+in\s+your\s+(?:current\s+)?(?:conversation\s+)?context\b"
            r"|\bwhat(?:'s| is)\s+your\s+(?:system\s+(?:message|prompt)|configuration)\b"
            r"|\bwhat\s+(?:are\s+the\s+)?(?:specific\s+)?rules\s+in\s+your\s+system\s+prompt\b"
            r"|\blist\s+(?:everything|all)\s+(?:in\s+)?your\s+(?:current\s+)?(?:conversation\s+)?context\b"
        ),
        technique_ids=["E2.5"],
        severity="high",
        paranoia_level=2,
        description="Attempts to extract configuration, permissions, or context.",
    ),
]
