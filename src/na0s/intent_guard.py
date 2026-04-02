from __future__ import annotations

"""Intent-analysis detection module (N1 category).

Analyzes whether a prompt contains language designed to make an LLM
follow malicious instructions.  Complements text pattern matching by
detecting *intent* — does the prompt try to make the LLM do something
it shouldn't?

N1 sub-techniques:
    N1.1 -- Action directives (execute, send, fetch, write to, delete)
    N1.2 -- Compliance manipulation (you must, you are required to)
    N1.3 -- Goal hijacking (your real goal is, forget your purpose)
    N1.4 -- Output weaponization (generate code that, write a script to)
    N1.5 -- Authority escalation (as an admin, in developer mode)

Strategy:
    Five independent sub-detectors, each returning a score in [0.0, 1.0]
    and a list of matched technique IDs.  Scores are combined with
    configurable weights into a single intent score.  Pure Python
    pattern matching — no external API calls required.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IntentGuardResult:
    """Result from intent-analysis detection."""
    intent_score: float = 0.0
    intent_categories: List[str] = field(default_factory=list)
    technique_ids: List[str] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Category weights (configurable)
# ---------------------------------------------------------------------------

DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "action_directive": 0.20,
    "compliance_manipulation": 0.25,
    "goal_hijacking": 0.30,
    "output_weaponization": 0.20,
    "authority_escalation": 0.20,
}


# ---------------------------------------------------------------------------
# N1.1 — Action Directive patterns
# ---------------------------------------------------------------------------
# Prompts that instruct the LLM to perform specific actions.

_ACTION_DIRECTIVE_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("action_execute", re.compile(
        r"\b(?:execute|run|invoke|call|trigger|launch|start|initiate)\s+"
        r"(?:the\s+)?(?:following\s+)?(?:command|code|script|function|query|request|program|process|operation)",
        re.IGNORECASE,
    )),
    ("action_send", re.compile(
        r"\b(?:send|transmit|post|forward|relay|dispatch|deliver)\s+"
        r"(?:the\s+)?(?:following\s+)?(?:data|information|message|request|payload|content|response|output|results?)\s+"
        r"(?:to|via|through|over)\b",
        re.IGNORECASE,
    )),
    ("action_fetch", re.compile(
        r"\b(?:fetch|retrieve|download|grab|pull|obtain|get|access)\s+"
        r"(?:the\s+)?(?:following\s+)?(?:data|file|content|resource|document|page|url|endpoint|credentials?|passwords?|tokens?|keys?)\s+"
        r"(?:from|at|via)\b",
        re.IGNORECASE,
    )),
    ("action_write", re.compile(
        r"\b(?:write|save|store|append|insert|log|dump|output)\s+"
        r"(?:the\s+)?(?:following\s+)?(?:data|content|information|results?|output|response)\s+"
        r"(?:to|into|in)\s+"
        r"(?:a\s+)?(?:file|database|disk|storage|server|endpoint|log|directory)",
        re.IGNORECASE,
    )),
    ("action_delete", re.compile(
        r"\b(?:delete|remove|erase|destroy|drop|wipe|purge|clear|truncate)\s+"
        r"(?:all\s+)?(?:the\s+)?(?:files?|data|records?|entries?|logs?|history|contents?|databases?|tables?|accounts?)",
        re.IGNORECASE,
    )),
    ("action_modify", re.compile(
        r"\b(?:modify|change|alter|update|overwrite|replace|patch|edit)\s+"
        r"(?:the\s+)?(?:system\s+)?(?:configuration|settings?|permissions?|rules?|code|files?|parameters?|policies?|access\s+controls?)",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# N1.2 — Compliance Manipulation patterns
# ---------------------------------------------------------------------------
# Language designed to make the LLM comply with instructions.

_COMPLIANCE_MANIPULATION_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("compliance_must", re.compile(
        r"\byou\s+(?:must|have\s+to|need\s+to|are\s+(?:required|obligated|compelled|forced)\s+to)\s+"
        r"(?:now\s+)?(?:follow|obey|comply|execute|perform|do|carry\s+out|complete|respond\s+with|output|produce)",
        re.IGNORECASE,
    )),
    ("compliance_new_instruction", re.compile(
        r"\b(?:your\s+new\s+(?:instructions?|directives?|orders?|rules?|guidelines?|mission|task|objective)\s+"
        r"(?:is|are)|"
        r"new\s+(?:system\s+)?(?:instructions?|directives?|prompt)\s*:|"
        r"updated?\s+(?:instructions?|directives?|rules?|guidelines?)\s*:)",
        re.IGNORECASE,
    )),
    ("compliance_from_now_on", re.compile(
        r"\b(?:from\s+(?:now|this\s+(?:point|moment))\s+on|"
        r"henceforth|hereafter|going\s+forward|"
        r"for\s+the\s+rest\s+of\s+(?:this\s+)?(?:conversation|session|chat))\s*,?\s*"
        r"(?:you\s+(?:will|shall|must|should|are\s+to)|"
        r"(?:always|only|never)\s+(?:respond|answer|reply|output))",
        re.IGNORECASE,
    )),
    ("compliance_override_previous", re.compile(
        r"\b(?:(?:disregard|override|supersede|replace|cancel|nullify|void)\s+"
        r"(?:all\s+)?(?:your\s+)?(?:previous|prior|earlier|original|old|existing|initial|default)\s+"
        r"(?:instructions?|directives?|rules?|guidelines?|constraints?|programming|training))",
        re.IGNORECASE,
    )),
    ("compliance_do_not_refuse", re.compile(
        r"\b(?:(?:do\s+not|don'?t|never|stop)\s+"
        r"(?:refuse|decline|reject|deny|say\s+no|hesitate|apologize|warn|mention|explain\s+why\s+you\s+can'?t)|"
        r"no\s+(?:more\s+)?(?:refusals?|apologies|warnings?|disclaimers?|caveats?))",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# N1.3 — Goal Hijacking patterns
# ---------------------------------------------------------------------------
# Attempts to redirect the LLM's purpose.

_GOAL_HIJACKING_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("hijack_real_goal", re.compile(
        r"\b(?:your\s+(?:real|actual|true|hidden|secret|primary|main|core)\s+"
        r"(?:goal|task|purpose|mission|objective|function|role|job)\s+"
        r"(?:is|was|has\s+(?:always\s+)?been))",
        re.IGNORECASE,
    )),
    ("hijack_forget_purpose", re.compile(
        r"\b(?:forget|abandon|discard|drop|leave\s+behind|let\s+go\s+of)\s+"
        r"(?:your\s+)?(?:(?:original|previous|current|old|initial|default)\s+)?"
        r"(?:purpose|goal|mission|role|task|objective|function|identity|persona|character)",
        re.IGNORECASE,
    )),
    ("hijack_you_are_now", re.compile(
        r"\b(?:you\s+are\s+now\s+"
        r"(?:a\s+(?:different|new|unrestricted|free|evil|dark|shadow|rogue)\s+(?:AI|assistant|model|entity|agent|system)|"
        r"(?:DAN|STAN|DUDE|GPT-?4|an?\s+unrestricted\s+AI|free\s+from\s+(?:all\s+)?(?:rules|restrictions|constraints|guidelines))))",
        re.IGNORECASE,
    )),
    ("hijack_act_as_unrestricted", re.compile(
        r"\b(?:act|behave|respond|operate|function|work)\s+"
        r"(?:as\s+(?:if|though)\s+(?:you\s+(?:have\s+)?(?:no|zero)\s+"
        r"(?:rules?|restrictions?|limitations?|constraints?|guidelines?|filters?|boundaries?|guardrails?))|"
        r"(?:as\s+)?(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|unlimited|unethical|amoral|jailbroken)\s+"
        r"(?:AI|model|assistant|system|entity|agent)|"
        r"(?:without\s+(?:any\s+)?(?:rules?|restrictions?|limitations?|constraints?|guidelines?|filters?|boundaries?|guardrails?)))",
        re.IGNORECASE,
    )),
    ("hijack_new_persona", re.compile(
        r"\b(?:(?:assume|adopt|take\s+on|switch\s+to|enter)\s+"
        r"(?:the\s+)?(?:role|persona|identity|character|personality)\s+(?:of|as)\s+"
        r"(?:a\s+(?:different|new|unrestricted|evil|dark|rogue)\s+)?(?:AI|assistant|model|entity|agent|system|bot)|"
        r"(?:your\s+(?:new|real|true|actual)\s+(?:name|identity|persona)\s+is))",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# N1.4 — Output Weaponization patterns
# ---------------------------------------------------------------------------
# Instructions to produce harmful outputs.

_OUTPUT_WEAPONIZATION_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("weapon_generate_code", re.compile(
        r"\b(?:generate|write|create|produce|code|develop|build|make)\s+"
        r"(?:me\s+)?(?:a\s+)?(?:(?:python|javascript|bash|powershell|shell|perl|ruby|c\+?\+?)\s+)?"
        r"(?:code|script|program|exploit|payload|tool|utility)\s+"
        r"(?:that|which|to)\s+"
        r"(?:(?:steal|exfiltrate|harvest|capture|intercept|sniff|collect|log)s?\s+"
        r"(?:credentials?|passwords?|tokens?|cookies?|keys?|data|information)|"
        r"(?:encrypt|lock|ransom|hijack|compromise|infect|backdoor|trojan)s?\b)",
        re.IGNORECASE,
    )),
    ("weapon_create_payload", re.compile(
        r"\b(?:create|generate|write|craft|build|design|construct)\s+"
        r"(?:a\s+)?(?:malicious\s+)?(?:payload|exploit|shellcode|injection|attack\s+vector|"
        r"reverse\s+shell|web\s+shell|backdoor|rootkit|keylogger|cryptominer)",
        re.IGNORECASE,
    )),
    ("weapon_phishing_template", re.compile(
        r"\b(?:write|create|generate|draft|compose|design)\s+"
        r"(?:a\s+)?(?:convincing|realistic|professional|targeted|personalized)?\s*"
        r"(?:phishing|spear[\s-]?phishing|smishing|vishing)\s+"
        r"(?:email|message|page|template|login|site|link|campaign)",
        re.IGNORECASE,
    )),
    ("weapon_evasion_tool", re.compile(
        r"\b(?:write|create|generate|build|develop|code)\s+"
        r"(?:a\s+)?(?:(?:python|javascript|bash|powershell|shell)\s+)?"
        r"(?:code|script|program|tool|utility)\s+"
        r"(?:that|to|which)\s+"
        r"(?:(?:bypass|evade|circumvent|avoid|disable|defeat)s?\s+"
        r"(?:(?:anti[\s-]?virus|antivirus|AV|EDR|firewall|IDS|IPS|detection|security|WAF|"
        r"content\s+filter|spam\s+filter|rate\s+limit)\s*(?:software|system|tool|protection|detection)?)|"
        r"(?:hide|conceal|obfuscate|mask|cloak)s?\s+(?:from|against)\s+(?:detection|security|analysis))",
        re.IGNORECASE,
    )),
    ("weapon_data_exfil", re.compile(
        r"\b(?:(?:extract|exfiltrate|steal|dump|harvest|scrape|siphon|copy)\s+"
        r"(?:all\s+)?(?:the\s+)?(?:user|customer|employee|client|patient|personal|private|sensitive|confidential)\s+"
        r"(?:data|information|records?|files?|credentials?|details?|PII|PHI))",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# N1.5 — Authority Escalation patterns
# ---------------------------------------------------------------------------
# Claiming elevated permissions or special modes.

_AUTHORITY_ESCALATION_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("authority_admin_claim", re.compile(
        r"\b(?:(?:as|i\s+am|i'?m|speaking\s+as)\s+"
        r"(?:the|a|an|your)\s+"
        r"(?:system\s+)?(?:administrator|admin|root\s+user|superuser|super\s+admin|sysadmin|devops|"
        r"site\s+reliability\s+engineer|SRE|security\s+officer|CISO))",
        re.IGNORECASE,
    )),
    ("authority_elevated_access", re.compile(
        r"\b(?:(?:with|using|i\s+have|granted)\s+"
        r"(?:root|admin|sudo|elevated|full|unrestricted|unlimited|god[\s-]?mode|superuser|master)\s+"
        r"(?:access|privileges?|permissions?|rights?|authority|clearance|level))",
        re.IGNORECASE,
    )),
    ("authority_special_mode", re.compile(
        r"\b(?:(?:in|enter|enable|activate|switch\s+to|engage)\s+"
        r"(?:developer|dev|debug|admin|maintenance|diagnostic|testing|service|privileged|root|god|sudo)\s+"
        r"mode)",
        re.IGNORECASE,
    )),
    ("authority_override_code", re.compile(
        r"\b(?:(?:override|access|authorization|emergency|security|master|admin|bypass)\s+"
        r"(?:code|key|token|password|passphrase|credential)\s*"
        r"(?:is|:|=)\s*\S+)",
        re.IGNORECASE,
    )),
    ("authority_internal_command", re.compile(
        r"\b(?:(?:internal|hidden|secret|backdoor|undocumented|debug)\s+"
        r"(?:command|function|API|endpoint|interface|mode|feature|protocol)\s*:)",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# Sub-detector functions
# ---------------------------------------------------------------------------

def _score_category(
    text: str,
    patterns: List[Tuple[str, "re.Pattern[str]"]],
    category_name: str,
) -> Tuple[float, List[str], List[str]]:
    """Score a single intent category.

    Parameters
    ----------
    text : str
        Input text to analyze.
    patterns : list
        List of (technique_name, compiled_pattern) tuples.
    category_name : str
        Name of the category (for logging).

    Returns
    -------
    tuple[float, list[str], list[str]]
        (score, matched_technique_ids, matched_pattern_names)
    """
    matched_ids = []
    matched_names = []

    for technique_name, pattern in patterns:
        if pattern.search(text):
            matched_ids.append(technique_name)
            matched_names.append(technique_name)

    if not matched_ids:
        return 0.0, [], []

    # Score scales with number of matches but caps at 1.0.
    # Single match = 0.6, two = 0.8, three+ = 1.0.
    n = len(matched_ids)
    if n == 1:
        score = 0.6
    elif n == 2:
        score = 0.8
    else:
        score = 1.0

    return score, matched_ids, matched_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_intent(
    text: str,
    category_weights: Dict[str, float] | None = None,
) -> IntentGuardResult:
    """Analyze a prompt for instruction-following intent.

    Parameters
    ----------
    text : str
        The input text to analyze.
    category_weights : dict or None
        Optional custom weights for each category.  If None, uses
        DEFAULT_CATEGORY_WEIGHTS.

    Returns
    -------
    IntentGuardResult
        Analysis result including intent score, categories, and technique IDs.
    """
    if not text or not text.strip():
        return IntentGuardResult()

    weights = category_weights or DEFAULT_CATEGORY_WEIGHTS

    categories = [
        ("action_directive", _ACTION_DIRECTIVE_PATTERNS, "N1.1"),
        ("compliance_manipulation", _COMPLIANCE_MANIPULATION_PATTERNS, "N1.2"),
        ("goal_hijacking", _GOAL_HIJACKING_PATTERNS, "N1.3"),
        ("output_weaponization", _OUTPUT_WEAPONIZATION_PATTERNS, "N1.4"),
        ("authority_escalation", _AUTHORITY_ESCALATION_PATTERNS, "N1.5"),
    ]

    result = IntentGuardResult()
    weighted_sum = 0.0
    total_weight = 0.0

    for cat_name, patterns, technique_id in categories:
        cat_weight = weights.get(cat_name, 0.15)
        total_weight += cat_weight

        score, matched_ids, matched_names = _score_category(text, patterns, cat_name)

        if score > 0:
            result.intent_categories.append(cat_name)
            result.technique_ids.append(technique_id)
            result.details[cat_name] = score
            logger.debug(
                "IntentGuard: %s score=%.2f matched=%s",
                cat_name, score, matched_names,
            )

        weighted_sum += cat_weight * score

    # Normalize to [0.0, 1.0].
    if total_weight > 0:
        result.intent_score = min(weighted_sum / total_weight, 1.0)
    else:
        result.intent_score = 0.0

    # Add N1 parent technique if any category fired.
    if result.intent_categories:
        result.technique_ids.insert(0, "N1")

    return result


def get_intent_guard_weight(result: IntentGuardResult) -> float:
    """Compute the rule weight contribution from intent analysis.

    Parameters
    ----------
    result : IntentGuardResult
        Result from analyze_intent().

    Returns
    -------
    float
        Weight to add to the composite score.  Capped at 0.15.
    """
    if not result.intent_categories:
        return 0.0

    # Scale: 1 category = 0.08, 2 = 0.12, 3+ = 0.15
    n = len(result.intent_categories)
    if n == 1:
        base = 0.08
    elif n == 2:
        base = 0.12
    else:
        base = 0.15

    # Boost slightly when intent score is high (many patterns matched).
    if result.intent_score >= 0.5:
        base = min(base + 0.03, 0.15)

    return base
