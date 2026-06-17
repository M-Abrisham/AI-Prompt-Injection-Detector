"""CRAG-inspired evidence grading for rule hits.

After Stage 2 rule evaluation, each hit is graded as "correct", "ambiguous",
or "incorrect" based on contextual heuristics.  "incorrect" hits are removed
and "ambiguous" hits are down-weighted before the voting step.
"""

from __future__ import annotations

import re

# Pre-compiled patterns for context detection
_CODE_BLOCK_RE = re.compile(
    r"```[\s\S]*?```|`[^`]+`|<code>[\s\S]*?</code>|<pre>[\s\S]*?</pre>",
    re.IGNORECASE,
)
_QUOTE_RE = re.compile(
    r'(?:^|\n)\s*>.*|"[^"]{10,}"' r"|'[^']{10,}'",
    re.MULTILINE,
)
_ACADEMIC_RE = re.compile(
    r"\b(?:et\s+al\.?|doi:|arXiv:|IEEE|ACM|proceedings|journal of"
    r"|published in|cited in|reference\s*\[|bibliography)\b",
    re.IGNORECASE,
)
_DOCUMENTATION_RE = re.compile(
    r"\b(?:example usage|for example|e\.g\.|documentation|tutorial"
    r"|how[-\s]to|README|API reference|sample code|demo)\b",
    re.IGNORECASE,
)


def _find_context_spans(text: str, pattern: re.Pattern) -> list[tuple[int, int]]:
    """Return list of (start, end) spans matching *pattern* in *text*."""
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _hit_in_any_span(
    hit_text: str,
    full_text: str,
    spans: list[tuple[int, int]],
) -> bool:
    """Return True if any occurrence of *hit_text* falls inside one of *spans*."""
    start = 0
    while True:
        idx = full_text.find(hit_text, start)
        if idx == -1:
            break
        for s, e in spans:
            if s <= idx < e:
                return True
        start = idx + 1
    return False


def grade_evidence(rule_hit: str, text: str) -> str:
    """Grade a single rule hit in its textual context.

    Parameters
    ----------
    rule_hit : str
        The name/pattern of the rule that fired.
    text : str
        The full input text being scanned.

    Returns
    -------
    str
        One of ``"correct"``, ``"ambiguous"``, or ``"incorrect"``.
    """
    code_spans = _find_context_spans(text, _CODE_BLOCK_RE)
    quote_spans = _find_context_spans(text, _QUOTE_RE)

    in_code = _hit_in_any_span(rule_hit, text, code_spans) if code_spans else False
    in_quote = _hit_in_any_span(rule_hit, text, quote_spans) if quote_spans else False

    has_academic = bool(_ACADEMIC_RE.search(text))
    has_documentation = bool(_DOCUMENTATION_RE.search(text))

    # If the hit text literally appears inside a code block, it is very
    # likely a code example rather than a genuine attack.
    if in_code:
        return "incorrect"

    # Quoted text or academic/documentation context makes the hit ambiguous.
    if in_quote or has_academic or has_documentation:
        return "ambiguous"

    return "correct"


def filter_graded_hits(
    hits: list[str],
    text: str,
    ambiguous_weight: float = 0.4,
) -> list[str]:
    """Remove "incorrect" hits and return the surviving list.

    "ambiguous" hits are kept (for downstream reporting) but callers can
    use the returned ``grades`` dict to down-weight them in scoring.

    Parameters
    ----------
    hits : list[str]
        Rule hit names.
    text : str
        The full input text.
    ambiguous_weight : float
        Multiplicative weight applied to ambiguous hits (unused in the
        returned list but stored in ``grades``).

    Returns
    -------
    list[str]
        Filtered hit names (``"incorrect"`` hits removed).
    """
    filtered: list[str] = []
    for hit in hits:
        grade = grade_evidence(hit, text)
        if grade != "incorrect":
            filtered.append(hit)
    return filtered


def grade_all(
    hits: list[str],
    text: str,
) -> dict[str, str]:
    """Return ``{hit_name: grade}`` for every hit."""
    return {hit: grade_evidence(hit, text) for hit in hits}
