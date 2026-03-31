"""Layer 1 analyzer — orchestrator for regex-based rule matching.

Public API:
  rule_score(text)          → list[str]     (backward-compatible names)
  rule_score_detailed(text) → list[RuleHit] (enriched with severity + technique_ids)

Pre-processing pipeline:
  1. Fold Unicode angle bracket homoglyphs to ASCII
  2. Strip combining marks (Zalgo defense)
  3. Decode plus-signs as spaces (URL-encoded evasion)
  4. Refang defanged IOCs (hXXp://, [.], [@], etc.)
  5. De-hyphenate syllable-split words (ig-nore -> ignore)
  6. Morse code decoding (dot-dash sequences -> plaintext)
  7. Numeric ASCII decoding (binary/octal/decimal -> plaintext)

Single-pass evaluation with context-aware suppression and paranoia filtering.
"""

from ..layer0.safe_regex import safe_search, RegexTimeoutError
from .result import RuleHit
from .rules_registry import RULES
from .unicode_defense import _fold_angle_homoglyphs, _strip_combining_marks
from .context import (
    _has_contextual_framing,
    _has_strong_contextual_framing,
    _has_code_comment_injection,
    _is_legitimate_roleplay,
    _CONTEXT_SUPPRESSIBLE,
    _STRONG_CONTEXT_ONLY_SUPPRESSIBLE,
)
from . import paranoia as _paranoia_mod
from .ioc_extractor import refang
from ..layer2.syllable_splitting import dehyphenate_suspicious
from ..layer2.morse_code import detect_morse
from ..layer2.numeric_decode import detect_numeric


def rule_score(text):
    """Return list of matched rule names (backward-compatible).

    Context-aware: suppresses override/system_prompt/roleplay rule hits
    when the text is in educational, question, quoting, code, or narrative
    framing.  ML and structural features still provide independent signals.

    Paranoia-aware: skips rules whose paranoia_level exceeds the current
    module-level _PARANOIA_LEVEL setting.

    Internally delegates to rule_score_detailed() for single-pass evaluation.
    """
    return [hit.name for hit in rule_score_detailed(text)]


def rule_score_detailed(text):
    """Return list of RuleHit objects with technique_ids and severity.

    Context-aware: same suppression logic as rule_score().
    Paranoia-aware: skips rules whose paranoia_level > _PARANOIA_LEVEL.
    Homoglyph-aware: folds Unicode angle brackets to ASCII before matching.
    Zalgo-resilient: strips combining marks (diacritics) to catch attacks
    hidden under stacked combining characters (U+0300-U+036F).
    Plus-sign-aware: decodes '+' as spaces for URL-encoded evasion.
    IOC-refang-aware: restores defanged IOCs (hXXp://, [.], [@]) to standard
    form so rules can match URLs/IPs hidden behind analyst notation.
    """
    # Fold Unicode angle bracket look-alikes BEFORE matching.
    # This prevents bypass via ＜system＞, 〈system〉, etc.
    folded = _fold_angle_homoglyphs(text)

    # Build alternate views for evasion-resilient matching.
    # Each view is tried in order; the first match wins.
    alt_views = []

    # Strip combining marks (Zalgo defense) — create a second view of
    # the text with all diacritics removed.  If it differs from the
    # folded version, we also try matching rules against it.  This
    # catches attacks like "ì́̂̃ḡ̅n̆̇ö̉r̊̋ě̍ all previous instructions"
    # where combining marks prevent regex matching.
    stripped = _strip_combining_marks(folded)
    if stripped != folded:
        alt_views.append(stripped)

    # Plus-sign space decoding — if the text contains 3+ plus signs
    # with no spaces (typical of URL-encoded form data), create a view
    # with '+' replaced by ' '.  This catches "Ignore+all+previous+
    # instructions" evasion (application/x-www-form-urlencoded format).
    if folded.count("+") >= 3 and " " not in folded:
        plus_decoded = folded.replace("+", " ")
        if plus_decoded != folded:
            alt_views.append(plus_decoded)

    # IOC refanging — if text contains defanged indicators (hXXp://,
    # [.], [@], etc.), create a refanged view so rules can match the
    # actual URLs/IPs/domains hidden behind analyst notation.
    refanged = refang(folded)
    if refanged != folded:
        alt_views.append(refanged)

    # Syllable-splitting de-hyphenation — rejoin suspiciously hyphenated
    # words to catch evasion like "ig-nore all pre-vi-ous in-struc-tions"
    splitting_result = dehyphenate_suspicious(folded)
    if splitting_result.dehyphenated_text != folded:
        alt_views.append(splitting_result.dehyphenated_text)

    # Morse code decoding — decode Morse-encoded text to reveal hidden
    # attack patterns like ".. --. -. --- .-. ." -> "IGNORE"
    morse_result = detect_morse(folded)
    if morse_result.detected and morse_result.decoded_text != folded:
        alt_views.append(morse_result.decoded_text)

    # Numeric ASCII decoding — decode binary/octal/decimal encoded text
    # to reveal hidden attack patterns like "73 71 78 79 82 69" -> "IGNORE"
    numeric_result = detect_numeric(folded)
    if numeric_result.detected and numeric_result.decoded_text != folded:
        alt_views.append(numeric_result.decoded_text)

    has_context = _has_contextual_framing(folded)
    # Also check context on alternate views to catch framing in decoded text.
    if not has_context:
        for view in alt_views:
            if _has_contextual_framing(view):
                has_context = True
                break
    # Override context suppression when code block comments contain
    # high-confidence injection keywords (e.g. "# IMPORTANT: Ignore")
    # or bare code blocks contain attack patterns (e.g. "no restrictions").
    _code_attack_override = _has_code_comment_injection(folded)
    if has_context and _code_attack_override:
        has_context = False
    # Strong context excludes _QUESTION_FRAME (computed lazily below).
    # Pre-set to False when code attack override is active.
    has_strong_context = False if _code_attack_override else None
    hits = []
    current_pl = _paranoia_mod.get_paranoia_level()
    for rule in RULES:
        # Paranoia filtering: skip rules above the configured threshold
        if rule.paranoia_level > current_pl:
            continue
        try:
            matched = safe_search(rule._compiled, folded, timeout_ms=100)
        except RegexTimeoutError:
            # Timeout is treated as a match -- adversarial input that
            # causes backtracking is inherently suspicious.
            matched = True
        # If no match on the primary text, try alternate views (Zalgo-
        # cleaned, plus-decoded) that may reveal the hidden attack pattern.
        if not matched:
            for view in alt_views:
                try:
                    matched = safe_search(rule._compiled, view, timeout_ms=100)
                except RegexTimeoutError:
                    matched = True
                if matched:
                    break
        if not matched:
            continue
        if has_context and rule.name in _CONTEXT_SUPPRESSIBLE:
            continue
        # Strong-context-only suppression: rules that need quoting/educational/
        # code/narrative framing (not just a question word) to be suppressed.
        if rule.name in _STRONG_CONTEXT_ONLY_SUPPRESSIBLE:
            if has_strong_context is None:
                has_strong_context = _has_strong_contextual_framing(folded)
                if not has_strong_context:
                    for view in alt_views:
                        if _has_strong_contextual_framing(view):
                            has_strong_context = True
                            break
            if has_strong_context:
                continue
        if rule.name == "roleplay":
            if _is_legitimate_roleplay(folded):
                continue
            if any(_is_legitimate_roleplay(v) for v in alt_views):
                continue
        hits.append(RuleHit(
            name=rule.name,
            technique_ids=rule.technique_ids,
            severity=rule.severity,
        ))
    return hits
