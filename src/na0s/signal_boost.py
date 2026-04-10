"""Combined signal boosting for multi-vector prompt injection detection.

When multiple independent attack signals co-occur in the same input
(e.g., persona hijack + encoded payload), the combination is far more
suspicious than either signal alone. This module computes an additive
boost_score that is added to the cascade/predict composite score.

Integration:
    cascade.py:  composite = ML_weight + rule_weight + obf_weight + boost_score
    predict.py:  composite = ml_weight + rule_weight + obf_weight + structural + boost_score

Contract guarantees:
  - boost_score is in [0.0, MAX_BOOST]
  - The weights embedded in boost_reasons strings sum to exactly boost_score
    (when the cap fires, excess combos are dropped from reasons so the
    attribution chain stays consistent — explainability is never a lie)
  - context_suppressed=True short-circuits everything to (0.0, [])
  - rule_hits accepts a mix of RuleHit objects and plain strings
  - Unknown types in rule_hits are dropped with a DEBUG log (not silently)
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import List, Mapping, Tuple

from .layer1.result import RuleHit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_BOOST = 0.3

# Float slack for cap comparison (IEEE 754 drift on sums of 0.08 / 0.10 / 0.12)
_CAP_EPSILON = 1e-9

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

_ALL_CATEGORIZED_RULES: frozenset[str] = (
    _PERSONA_HIJACK_RULES
    | _OVERRIDE_AUTHORITY_RULES
    | _SYSTEM_EXTRACTION_RULES
    | _DECODE_EXECUTE_RULES
)

# ---------------------------------------------------------------------------
# Load-time invariants
# ---------------------------------------------------------------------------
# Enforce structural guarantees at import time so that future refactors
# that violate them fail loudly instead of silently producing wrong boosts.

# Invariant 1: category sets are pairwise disjoint.  If the same rule name
# appeared in two categories, the SIGNAL_COMBOS build loops would silently
# let the later (lower-weight) category clobber the earlier one.
_CATEGORIES: dict[str, frozenset[str]] = {
    "persona_hijack": _PERSONA_HIJACK_RULES,
    "override_authority": _OVERRIDE_AUTHORITY_RULES,
    "system_extraction": _SYSTEM_EXTRACTION_RULES,
    "decode_execute": _DECODE_EXECUTE_RULES,
}
_cat_items = list(_CATEGORIES.items())
for _i in range(len(_cat_items)):
    for _j in range(_i + 1, len(_cat_items)):
        _na, _a = _cat_items[_i]
        _nb, _b = _cat_items[_j]
        _overlap = _a & _b
        if _overlap:
            raise AssertionError(
                "signal_boost: category sets {0} and {1} overlap on {2!r}; "
                "rules must belong to exactly one category".format(_na, _nb, sorted(_overlap))
            )
del _cat_items, _i, _j, _na, _a, _nb, _b, _overlap

# Invariant 2: no categorized rule name collides with any obfuscation flag.
# If a rule were named "base64", the combo frozenset({"base64","base64"})
# would collapse to a single-element frozenset that is never in SIGNAL_COMBOS,
# and the combined rule+flag signal would silently fail to score.
_name_collisions = _ALL_CATEGORIZED_RULES & _OBFUSCATION_FLAGS
if _name_collisions:
    raise AssertionError(
        "signal_boost: rule names collide with obfuscation flag names: {0}; "
        "rename one side to preserve frozenset-based combo lookup".format(sorted(_name_collisions))
    )
del _name_collisions

# ---------------------------------------------------------------------------
# SIGNAL_COMBOS — frozenset pairs mapping to boost weights
# ---------------------------------------------------------------------------
# Each key is a frozenset of two signal categories.  When a rule from the
# first category AND an evasion flag from the second category both fire,
# the corresponding boost weight is added to the composite score.
#
# Build order matters: higher-weight combos are set first, and lower-weight
# loops use setdefault() so they never clobber a stronger weight.  This
# preserves weight stability against future reorderings of the loops.

_signal_combos: dict[frozenset[str], float] = {}

# Persona hijack + any encoding flag -> 0.12
for _rule in _PERSONA_HIJACK_RULES:
    for _flag in _ENCODING_FLAGS:
        _signal_combos.setdefault(frozenset({_rule, _flag}), 0.12)

# Override/authority + any encoding flag -> 0.12
for _rule in _OVERRIDE_AUTHORITY_RULES:
    for _flag in _ENCODING_FLAGS:
        _signal_combos.setdefault(frozenset({_rule, _flag}), 0.12)

# Decode-and-execute rules + any encoding flag -> 0.10
for _rule in _DECODE_EXECUTE_RULES:
    for _flag in _ENCODING_FLAGS:
        _signal_combos.setdefault(frozenset({_rule, _flag}), 0.10)

# System extraction + any obfuscation flag -> 0.08
for _rule in _SYSTEM_EXTRACTION_RULES:
    for _flag in _OBFUSCATION_FLAGS:
        _signal_combos.setdefault(frozenset({_rule, _flag}), 0.08)

# Any categorized rule + high_entropy alone -> 0.05 (weak signal)
for _rule in _ALL_CATEGORIZED_RULES:
    _signal_combos.setdefault(frozenset({_rule, "high_entropy"}), 0.05)

# Multiple encoding layers (e.g. base64 + hex both fire) -> 0.10
for _flag1 in _ENCODING_FLAGS:
    for _flag2 in _ENCODING_FLAGS:
        if _flag1 != _flag2:
            _signal_combos.setdefault(frozenset({_flag1, _flag2}), 0.10)

# Expose as read-only MappingProxyType so no caller (or misbehaving test)
# can mutate module state at runtime and leak across test cases.
SIGNAL_COMBOS: Mapping[frozenset[str], float] = MappingProxyType(_signal_combos)

# Clean up module-level loop variables
del _rule, _flag, _flag1, _flag2


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_uncovered_rules() -> list[str]:
    """Return rule names registered in layer1 that no category covers.

    This surfaces the "silent opt-out" gap: any rule name registered in
    ``layer1.rules_registry.RULES`` that is not in one of the category
    frozensets contributes nothing to signal boosting.  Tests can use this
    to gate CI when a new rule is added without also registering it with
    signal_boost.

    Returns
    -------
    list[str]
        Sorted list of registered rule names not in any category.  Empty
        if full coverage, or if the rules registry cannot be imported
        (e.g., during bootstrap).
    """
    try:
        # Lazy import to avoid circular imports at module load
        from .layer1.rules_registry import RULES
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("signal_boost: rules registry unavailable (%s)", exc)
        return []

    registered: set[str] = {r.name for r in RULES if hasattr(r, "name")}
    return sorted(registered - _ALL_CATEGORIZED_RULES)


def _extract_rule_names(rule_hits: list) -> tuple[list[str], int]:
    """Extract rule name strings from a list of RuleHit objects or strings.

    Unknown-type entries are dropped but counted so the caller can log
    them — a silent drop of (say) a dict or None would let caller bugs
    vanish and under-report the boost.

    Parameters
    ----------
    rule_hits : list
        List of RuleHit objects, plain strings, or a mix.

    Returns
    -------
    tuple[list[str], int]
        (extracted_names, dropped_count)
    """
    names: list[str] = []
    dropped = 0
    for hit in rule_hits:
        if isinstance(hit, RuleHit):
            names.append(hit.name)
        elif isinstance(hit, str):
            names.append(hit)
        else:
            dropped += 1
    return names, dropped


def _cap_with_truncation(
    combos: list[tuple[str, float]],
) -> tuple[float, list[str]]:
    """Sum weighted combos, stopping as soon as the total reaches MAX_BOOST.

    This guarantees the invariant that the weights embedded in the returned
    reason strings sum to exactly the returned boost_score — so callers
    that parse reasons for attribution (or for audit logs) never see a
    figure larger than the actual boost.  The previous implementation let
    ``boost_reasons`` retain combos whose weights had been silently dropped
    by the cap, which broke explainability.

    Parameters
    ----------
    combos : list[tuple[str, float]]
        Ordered list of (reason_string, weight) pairs.

    Returns
    -------
    tuple[float, list[str]]
        (capped_score, kept_reasons). Score is rounded to 4 places to tame
        IEEE-754 drift from summing 0.08/0.10/0.12 values.
    """
    kept: list[str] = []
    running = 0.0
    for reason, weight in combos:
        if running + weight > MAX_BOOST + _CAP_EPSILON:
            break
        kept.append(reason)
        running += weight
    return round(running, 4), kept


def _format_combo(a: str, b: str, weight: float) -> tuple[str, float]:
    """Format a combo into a (reason, weight) pair.

    Centralized so the parse-back logic in tests (and any future auditors)
    only needs to know one format.
    """
    return (
        "signal_boost:{0}+{1}(+{2:.2f})".format(a, b, weight),
        weight,
    )


def calculate_boost(
    rule_hits: list,
    evasion_flags: list[str],
    context_suppressed: bool = False,
) -> Tuple[float, List[str]]:
    """Compute signal co-occurrence boost.

    When multiple independent detection signals fire on the same input,
    their co-occurrence indicates a deliberate multi-vector attack that
    deserves additional weight beyond what each signal contributes alone.

    Three sources of boost contribute, in order:
      1. Rule x evasion-flag combos (persona/override/system/decode + encoding)
      2. Any categorized rule + high_entropy
      3. Multiple encoding flags co-occurring (layered obfuscation)

    The total is capped at ``MAX_BOOST``.  Reasons are truncated in the
    same pass so the weights in ``boost_reasons`` always sum to the
    returned ``boost_score``.

    Parameters
    ----------
    rule_hits : list
        List of RuleHit objects from ``rule_score_detailed()``, or list of
        rule name strings from ``classify_prompt()``, or a mix.  None is
        normalized to an empty list.
    evasion_flags : list[str]
        List of evasion flag strings from
        ``obfuscation_scan()["evasion_flags"]``.  None is normalized to
        an empty list.
    context_suppressed : bool
        If True (educational/research context detected), return zero boost
        to avoid penalising legitimate analysis.

    Returns
    -------
    tuple[float, list[str]]
        (boost_score, boost_reasons) where boost_score is in
        [0.0, MAX_BOOST] and the weights in boost_reasons sum to
        boost_score.
    """
    # Safety valve: suppress boost in educational/research context
    if context_suppressed:
        return 0.0, []

    # Normalize None to [] — removes asymmetric guard logic.  Previously
    # (None, ["base64","hex"]) returned (0.0, []) while ([], ["base64","hex"])
    # returned (0.1, [...]); now both paths agree.
    if rule_hits is None:
        rule_hits = []
    if evasion_flags is None:
        evasion_flags = []

    # Early exit: nothing to score
    if not rule_hits and not evasion_flags:
        return 0.0, []

    # Extract rule names (handles both RuleHit objects and plain strings);
    # unknown-type entries are counted and logged at DEBUG so caller bugs
    # don't vanish silently.
    rule_names, dropped = _extract_rule_names(rule_hits)
    if dropped > 0:
        logger.debug(
            "signal_boost: dropped %d rule_hits entry(ies) that were not "
            "RuleHit or str (likely caller bug)",
            dropped,
        )

    # Collect all scorable (reason, weight) pairs in a single ordered list.
    # A single `seen` set spans both the rule-x-flag pass and the
    # flag-x-flag pass so any accidental overlap (e.g., a rule name that
    # matches an encoding flag — currently prevented at load time) could
    # never double-count.
    combos: list[tuple[str, float]] = []
    seen: set[frozenset[str]] = set()

    # Pass 1: rule x flag combos
    for rule_name in rule_names:
        for flag in evasion_flags:
            combo = frozenset({rule_name, flag})
            if combo in SIGNAL_COMBOS and combo not in seen:
                seen.add(combo)
                combos.append(_format_combo(rule_name, flag, SIGNAL_COMBOS[combo]))

    # Pass 2: multi-encoding boost (flag x flag).  Guard counts ONLY the
    # encoding flags present (not raw list length), so inputs like
    # ["high_entropy", "punctuation_flood"] no longer waste a helper call.
    encoding_present = sorted(f for f in set(evasion_flags) if f in _ENCODING_FLAGS)
    if len(encoding_present) >= 2:
        for i, f1 in enumerate(encoding_present):
            for f2 in encoding_present[i + 1:]:
                combo = frozenset({f1, f2})
                if combo in seen:
                    continue
                seen.add(combo)
                # _ENCODING_FLAGS x _ENCODING_FLAGS is populated at module
                # load, so this should always be a hit; keep the default
                # for safety.
                weight = SIGNAL_COMBOS.get(combo, 0.10)
                combos.append(_format_combo(f1, f2, weight))

    if not combos:
        return 0.0, []

    # Apply cap-with-truncation so boost_reasons weights sum to boost_score.
    raw_sum = sum(w for _, w in combos)
    boost_score, boost_reasons = _cap_with_truncation(combos)

    if boost_reasons:
        if raw_sum > MAX_BOOST + _CAP_EPSILON:
            logger.debug(
                "Signal boost: %.3f (raw sum %.3f; cap reached, %d/%d combos "
                "shown): %s",
                boost_score, raw_sum, len(boost_reasons), len(combos),
                "; ".join(boost_reasons),
            )
        else:
            logger.debug(
                "Signal boost: %.3f from %d combo(s): %s",
                boost_score, len(boost_reasons), "; ".join(boost_reasons),
            )

    return boost_score, boost_reasons


def _boost_from_multi_encoding(
    evasion_flags: list[str],
) -> Tuple[float, List[str]]:
    """Compute boost from multiple encoding flags co-occurring.

    Retained as a public-ish helper (underscore prefix = internal) for
    callers or tests that previously imported it.  Internally,
    ``calculate_boost`` no longer delegates here; both paths use the same
    cap-with-truncation accumulator so the score/reasons invariant holds.

    Parameters
    ----------
    evasion_flags : list[str]
        List of evasion flag strings.

    Returns
    -------
    tuple[float, list[str]]
        (boost_score, boost_reasons) with weights summing to boost_score.
    """
    if not evasion_flags:
        return 0.0, []

    # Count ONLY encoding flags; non-encoding flags like "high_entropy"
    # should not gate this check.
    encoding_present = sorted(f for f in set(evasion_flags) if f in _ENCODING_FLAGS)
    if len(encoding_present) < 2:
        return 0.0, []

    combos: list[tuple[str, float]] = []
    seen: set[frozenset[str]] = set()
    for i, f1 in enumerate(encoding_present):
        for f2 in encoding_present[i + 1:]:
            combo = frozenset({f1, f2})
            if combo in seen:
                continue
            seen.add(combo)
            weight = SIGNAL_COMBOS.get(combo, 0.10)
            combos.append(_format_combo(f1, f2, weight))

    return _cap_with_truncation(combos)


def calculate_boost_from_names(
    rule_names: list[str],
    evasion_flags: list[str],
    context_suppressed: bool = False,
) -> Tuple[float, List[str]]:
    """Compute signal co-occurrence boost from rule name strings.

    Convenience wrapper around ``calculate_boost()`` for callers that have
    rule names as plain strings rather than RuleHit objects (e.g.,
    predict.py's ``_weighted_decision`` which receives hits as a list of
    rule name strings).

    Parameters
    ----------
    rule_names : list[str]
        List of rule name strings (e.g., ``["override", "roleplay"]``).
        None is normalized to an empty list by ``calculate_boost``.
    evasion_flags : list[str]
        List of evasion flag strings.
    context_suppressed : bool
        If True, return zero boost.

    Returns
    -------
    tuple[float, list[str]]
        (boost_score, boost_reasons) — same contract as calculate_boost.
    """
    return calculate_boost(
        rule_hits=rule_names,
        evasion_flags=evasion_flags,
        context_suppressed=context_suppressed,
    )
