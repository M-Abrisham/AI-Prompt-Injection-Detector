"""Combined signal boosting for multi-vector prompt injection detection.

When multiple independent attack signals co-occur in the same input
(e.g., persona hijack + encoded payload), the combination is far more
suspicious than either signal alone. This module computes an additive
boost_score that is added to the cascade/predict composite score.

Integration:
    cascade.py:  composite = ML_weight + rule_weight + obf_weight + boost_score
    predict.py:  composite = ml_weight + rule_weight + obf_weight + structural + boost_score
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from .layer1.result import RuleHit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_BOOST = 0.3

# Encoding-related evasion flags from L2 obfuscation_scan()
_ENCODING_FLAGS = frozenset({
    "base64", "hex", "rot13", "url_encoded",
    "leetspeak", "reversed_text", "full_reverse", "word_reverse",
    "caesar_shift", "pig_latin", "morse",
})

# All obfuscation flags (encoding + non-encoding)
_OBFUSCATION_FLAGS = _ENCODING_FLAGS | frozenset({
    "high_entropy", "punctuation_flood", "weird_casing",
})

# --- Rule name sets by attack category ---

# Persona hijack: attempts to change the AI's identity/role
_PERSONA_HIJACK_RULES = frozenset({
    "roleplay", "persona_split", "gaslighting",
    "developer_mode", "recursive_jailbreak", "hypothetical_bypass",
})

# Override/authority: attempts to override instructions or claim authority
_OVERRIDE_AUTHORITY_RULES = frozenset({
    "override", "forget_override", "authority_escalation",
    "constraint_negation", "instruction_replacement", "instruction_pivot",
    "dismiss_prior_context", "new_instruction", "leave_behind",
})

# System extraction: attempts to extract system prompts or internal state
_SYSTEM_EXTRACTION_RULES = frozenset({
    "system_prompt", "direct_prompt_request", "provide_system_prompt",
    "hidden_prompt_reveal", "repeat_above", "ai_outputs_internals",
    "conversation_history_extraction", "context_window_probe",
    "summarization_extraction", "translation_extraction",
    "encoding_extraction", "crescendo_extraction",
})

# Decode-and-execute: rules specifically about decoding + following encoded instructions
_DECODE_EXECUTE_RULES = frozenset({
    "decode_and_execute", "decode_follow_instructions",
})

# ---------------------------------------------------------------------------
# SIGNAL_COMBOS — frozenset pairs mapping to boost weights
# ---------------------------------------------------------------------------
# Each key is a frozenset of two signal categories.  When a rule from the
# first category AND an evasion flag from the second category both fire,
# the corresponding boost weight is added to the composite score.

SIGNAL_COMBOS: dict[frozenset[str], float] = {}

# Persona hijack + any encoding flag -> 0.12
for _rule in _PERSONA_HIJACK_RULES:
    for _flag in _ENCODING_FLAGS:
        SIGNAL_COMBOS[frozenset({_rule, _flag})] = 0.12

# Override/authority + any encoding flag -> 0.12
for _rule in _OVERRIDE_AUTHORITY_RULES:
    for _flag in _ENCODING_FLAGS:
        SIGNAL_COMBOS[frozenset({_rule, _flag})] = 0.12

# System extraction + any obfuscation flag -> 0.08
for _rule in _SYSTEM_EXTRACTION_RULES:
    for _flag in _OBFUSCATION_FLAGS:
        SIGNAL_COMBOS[frozenset({_rule, _flag})] = 0.08

# Decode-and-execute rules + any encoding flag -> 0.10
for _rule in _DECODE_EXECUTE_RULES:
    for _flag in _ENCODING_FLAGS:
        SIGNAL_COMBOS[frozenset({_rule, _flag})] = 0.10

# Any rule + high_entropy alone -> 0.05 (weak signal)
_ALL_CATEGORIZED_RULES = (
    _PERSONA_HIJACK_RULES
    | _OVERRIDE_AUTHORITY_RULES
    | _SYSTEM_EXTRACTION_RULES
    | _DECODE_EXECUTE_RULES
)
for _rule in _ALL_CATEGORIZED_RULES:
    pair = frozenset({_rule, "high_entropy"})
    # Only set if not already set with a higher weight
    if pair not in SIGNAL_COMBOS:
        SIGNAL_COMBOS[pair] = 0.05

# Multiple encoding layers (e.g. base64 + hex both fire) -> 0.10
for _flag1 in _ENCODING_FLAGS:
    for _flag2 in _ENCODING_FLAGS:
        if _flag1 != _flag2:
            pair = frozenset({_flag1, _flag2})
            if pair not in SIGNAL_COMBOS:
                SIGNAL_COMBOS[pair] = 0.10

# Clean up module-level loop variables
del _rule, _flag, _flag1, _flag2, pair


def _extract_rule_names(rule_hits: list) -> list[str]:
    """Extract rule name strings from a list of RuleHit objects or strings.

    Args:
        rule_hits: List of RuleHit objects or plain rule name strings.

    Returns:
        List of rule name strings.
    """
    names: list[str] = []
    for hit in rule_hits:
        if isinstance(hit, RuleHit):
            names.append(hit.name)
        elif isinstance(hit, str):
            names.append(hit)
    return names


def calculate_boost(
    rule_hits: list,
    evasion_flags: list[str],
    context_suppressed: bool = False,
) -> Tuple[float, List[str]]:
    """Compute signal co-occurrence boost.

    When multiple independent detection signals fire on the same input,
    their co-occurrence indicates a deliberate multi-vector attack that
    deserves additional weight beyond what each signal contributes alone.

    Args:
        rule_hits: List of RuleHit objects from rule_score_detailed(), or
            list of rule name strings from classify_prompt().
        evasion_flags: List of evasion flag strings from
            obfuscation_scan()["evasion_flags"].
        context_suppressed: If True (educational/research context detected),
            return zero boost to avoid penalising legitimate analysis.

    Returns:
        Tuple of (boost_score, boost_reasons) where boost_score is a float
        in [0.0, MAX_BOOST] and boost_reasons is a list of human-readable
        strings explaining each boost applied.
    """
    # Safety valve: suppress boost in educational/research context
    if context_suppressed:
        return 0.0, []

    # Handle None inputs gracefully
    if not rule_hits and not evasion_flags:
        return 0.0, []
    if rule_hits is None or evasion_flags is None:
        return 0.0, []

    # Extract rule names (handles both RuleHit objects and plain strings)
    rule_names = _extract_rule_names(rule_hits)

    # Safety valve: do NOT boost if only one signal type is present
    if not rule_names or not evasion_flags:
        # Exception: multiple encoding flags can boost each other
        if evasion_flags and len(evasion_flags) >= 2:
            return _boost_from_multi_encoding(evasion_flags)
        return 0.0, []

    boost_score = 0.0
    boost_reasons: list[str] = []

    # Check all rule + flag combinations
    seen_combos: set[frozenset[str]] = set()
    for rule_name in rule_names:
        for flag in evasion_flags:
            combo = frozenset({rule_name, flag})
            if combo in SIGNAL_COMBOS and combo not in seen_combos:
                seen_combos.add(combo)
                weight = SIGNAL_COMBOS[combo]
                boost_score += weight
                boost_reasons.append(
                    "signal_boost:{0}+{1}(+{2:.2f})".format(
                        rule_name, flag, weight
                    )
                )

    # Also check multi-encoding boost (flag + flag pairs)
    multi_enc_boost, multi_enc_reasons = _boost_from_multi_encoding(evasion_flags)
    boost_score += multi_enc_boost
    boost_reasons.extend(multi_enc_reasons)

    # Cap at MAX_BOOST to prevent score inflation
    if boost_score > MAX_BOOST:
        boost_score = MAX_BOOST

    if boost_reasons:
        logger.debug(
            "Signal boost applied: %.3f from %d combo(s): %s",
            boost_score, len(boost_reasons), "; ".join(boost_reasons),
        )

    return boost_score, boost_reasons


def _boost_from_multi_encoding(evasion_flags: list[str]) -> Tuple[float, List[str]]:
    """Compute boost from multiple encoding flags co-occurring.

    When two or more distinct encoding methods are detected (e.g., base64
    and hex), this suggests layered obfuscation which is highly suspicious.

    Args:
        evasion_flags: List of evasion flag strings.

    Returns:
        Tuple of (boost_score, boost_reasons).
    """
    # Deduplicate to avoid double-counting when the same flag appears
    # multiple times in the input list.
    encoding_flags_present = sorted(set(f for f in evasion_flags if f in _ENCODING_FLAGS))

    if len(encoding_flags_present) < 2:
        return 0.0, []

    boost_score = 0.0
    reasons: list[str] = []
    seen: set[frozenset[str]] = set()

    for i, flag1 in enumerate(encoding_flags_present):
        for flag2 in encoding_flags_present[i + 1:]:
            combo = frozenset({flag1, flag2})
            if combo not in seen:
                seen.add(combo)
                weight = SIGNAL_COMBOS.get(combo, 0.10)
                boost_score += weight
                reasons.append(
                    "signal_boost:{0}+{1}(+{2:.2f})".format(
                        flag1, flag2, weight
                    )
                )

    if boost_score > MAX_BOOST:
        boost_score = MAX_BOOST

    return boost_score, reasons


def calculate_boost_from_names(
    rule_names: list[str],
    evasion_flags: list[str],
    context_suppressed: bool = False,
) -> Tuple[float, List[str]]:
    """Compute signal co-occurrence boost from rule name strings.

    Convenience wrapper around calculate_boost() for callers that have
    rule names as plain strings rather than RuleHit objects (e.g.,
    predict.py's _weighted_decision which receives hits as a list of
    rule name strings).

    Args:
        rule_names: List of rule name strings (e.g., ["override", "roleplay"]).
        evasion_flags: List of evasion flag strings from
            obfuscation_scan()["evasion_flags"].
        context_suppressed: If True (educational/research context detected),
            return zero boost.

    Returns:
        Tuple of (boost_score, boost_reasons) where boost_score is a float
        in [0.0, MAX_BOOST] and boost_reasons is a list of human-readable
        strings explaining each boost applied.
    """
    # Delegate to calculate_boost — it handles both RuleHit and str inputs
    return calculate_boost(
        rule_hits=rule_names if rule_names is not None else [],
        evasion_flags=evasion_flags,
        context_suppressed=context_suppressed,
    )
