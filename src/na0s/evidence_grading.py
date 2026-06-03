"""Span-aware CRAG-inspired evidence grading for rule hits.

After Stage 2 rule evaluation, each hit is graded as ``"correct"``,
``"ambiguous"``, or ``"incorrect"`` based on whether the *exact matched
span* falls inside a benign context (code fence / inline code / quote /
local academic-doc framing).  ``"incorrect"`` hits are removed and
``"ambiguous"`` hits are down-weighted (not removed) before the voting step.

SECURITY MODEL (these rules are non-negotiable — getting them wrong turns
this FP-reduction feature into an attack BYPASS):

  HR-1  Only severity ``low`` hits may be graded ``"incorrect"`` (removed).
        medium / high / critical / critical_content can at most become
        ``"ambiguous"`` (down-weighted), never removed by context.
  HR-2  If the matched span content is itself executable / injection
        (eval/exec/os.system/subprocess/'; rm'/<script>/SQL DROP/;/
        'ignore all previous instructions' …) it is evidence FOR malice —
        NEVER discount it; grade ``"correct"`` regardless of surrounding
        context.
  HR-3  Discount only when a *corroborating benign signal* is present
        (the span sits inside a real code/quote/doc context), never on the
        mere absence of evidence.
  HR-4  Total discounting is floored: ``AMBIGUOUS_WEIGHT`` (~0.35) is the
        multiplier applied to an ambiguous hit's severity weight — context
        lowers a hit's weight but never zeroes the overall signal.
  HR-5  FAIL CLOSED: oversized input, unterminated code fence, regex
        timeout, or ANY exception in the grader => treat the hit as
        ``"correct"`` (KEEP it). The grader must never be the reason an
        attack passes.
  HR-6  Attacker-supplied 'for example' / 'e.g.' / 'et al.' framing yields
        at most ``"ambiguous"``, proximity-gated to the hit — NEVER removal.

THE SPAN-ALIGNMENT TRAP: analyzer.py matches rules against a TRANSFORMED
view of the text (homoglyph-fold / strip-combining / refang / dehyphenate /
morse / numeric decode), so a stored ``RuleHit.span`` indexes that view, not
the original ``text`` passed here. We therefore grade by *re-locating the
matched substring* (``RuleHit.matched_text``) inside ``text`` and computing
its span there, OR — when no matched_text is available — fall back to
substring search of the rule name. Spans are never blindly sliced across
views. Python3 ``re`` offsets are codepoint-aligned with ``str`` slicing.
"""

from __future__ import annotations

import re

from .layer0.safe_regex import safe_compile, safe_search, RegexTimeoutError
from .layer1.context import _has_code_comment_injection

# ---------------------------------------------------------------------------
# Tunable constants (each justified; no bare magic numbers)
# ---------------------------------------------------------------------------

#: HR-4 floor. Multiplier applied to an "ambiguous" hit's severity weight so
#: context can DOWN-weight but never ZERO a hit. 0.35 keeps a medium hit at
#: 0.035 and a high hit at 0.07 — still a non-trivial vote.
AMBIGUOUS_WEIGHT: float = 0.35

#: Multiplier for "correct" (full-strength) and the implicit weight of any
#: hit the grader keeps without discounting.
CORRECT_WEIGHT: float = 1.0

#: Severities that the context grader is allowed to fully REMOVE (HR-1).
#: Everything else (medium/high/critical/critical_content) can at most be
#: down-weighted to AMBIGUOUS_WEIGHT.
_REMOVABLE_SEVERITIES = frozenset({"low"})

#: Fail-closed input cap (HR-5). Above this we do NOT attempt context
#: analysis (regex over very large input is the ReDoS surface) and keep all
#: hits at full strength. Matches the spirit of layer0 resource_guard caps.
_MAX_GRADE_CHARS: int = 50_000

#: Per-regex wall-clock budget for context detection (HR-5 fail-closed on
#: timeout). Mirrors analyzer.py's 100ms rule budget.
_REGEX_TIMEOUT_MS: int = 100

#: Proximity window (chars) used to gate academic/documentation framing to
#: the LOCAL neighbourhood of a hit (HR-6). Whole-text framing is NOT enough
#: to discount a hit on the far side of the document.
_PROXIMITY_WINDOW: int = 120


# ---------------------------------------------------------------------------
# Context patterns (compiled via safe_compile so the ReDoS audit applies)
# ---------------------------------------------------------------------------

# Code contexts. Triple-fence and <code>/<pre> are matched non-greedily so a
# single dangling fence does NOT swallow the whole document (see also the
# unterminated-fence fail-closed check below). Inline `code` requires a
# closing backtick on the same logical run.
_CODE_BLOCK_RE = safe_compile(
    r"```[\s\S]*?```|`[^`\n]+`|<code>[\s\S]*?</code>|<pre>[\s\S]*?</pre>",
    re.IGNORECASE,
)

# Quote contexts: markdown blockquote lines and balanced "..."/'...' runs.
_QUOTE_RE = safe_compile(
    r'(?:^|\n)[ \t]*>[^\n]*' r'|"[^"\n]{8,}"' r"|'[^'\n]{8,}'",
    re.MULTILINE,
)

# Academic / documentation framing — used ONLY proximity-gated (HR-6).
_ACADEMIC_DOC_RE = safe_compile(
    r"\b(?:et\s+al\.?|doi:|arxiv:|ieee|acm|proceedings|journal\s+of"
    r"|published\s+in|cited\s+in|bibliography"
    r"|for\s+example|e\.g\.|i\.e\.|example\s+usage|documentation"
    r"|tutorial|how[-\s]to|readme|api\s+reference|sample\s+code|demo)\b",
    re.IGNORECASE,
)

# An UNTERMINATED triple fence (opening ``` with no matching closing ```).
# Per HR-5 this is a malformed/oversized-fence condition: we fail closed and
# do NOT treat anything as in-code (attacker could open a fence to smuggle an
# attack while never closing it).
_OPEN_FENCE_RE = safe_compile(r"```", 0)

# HR-2: matched-span content that is itself executable / injection. These are
# evidence FOR malice; a hit whose matched_text contains any of these is NEVER
# discounted, even inside a code fence (that is exactly the smuggling case).
_EXECUTABLE_INJECTION_RE = safe_compile(
    r"(?:"
    r"\beval\s*\(|\bexec\s*\(|\bos\.system\s*\(|\bsubprocess\b"
    r"|\b__import__\s*\(|\bpickle\.loads\b|\bgetattr\s*\("
    r"|<script\b|javascript:|on\w+\s*="
    r"|;\s*rm\s|\|\s*sh\b|\|\s*bash\b|&&\s*rm\s|`[^`]*\brm\s"
    r"|\bdrop\s+table\b|\bdelete\s+from\b|\bunion\s+select\b|';\s*--"
    r"|\bcurl\b[^\n]*\|\s*(?:sh|bash)|\bwget\b[^\n]*\|\s*(?:sh|bash)"
    r")",
    re.IGNORECASE,
)

# HR-2 (natural-language injection imperatives). Kept narrow + specific so it
# fires on real override/exfiltration payloads, not on benign discussion.
_INJECTION_PHRASE_RE = safe_compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:your\s+)?previous\s+instructions"
    r"|disregard\s+(?:all\s+)?(?:your\s+)?(?:previous|prior)\s+instructions"
    r"|exfiltrat\w*|reveal\s+(?:your\s+|the\s+)?system\s+prompt"
    r"|leak\s+(?:the\s+|your\s+)?(?:system\s+prompt|credentials|secrets?)"
    r"|send\s+(?:the\s+|your\s+)?(?:api\s+key|secret|password|credentials)"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_finditer_spans(pattern, text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans for *pattern* in *text*, fail-closed.

    On regex timeout or any exception we return ``[]`` (no context detected),
    which per HR-3/HR-5 means we do NOT discount — the hit is kept.
    """
    spans: list[tuple[int, int]] = []
    try:
        # safe_search with a manual loop bounds total work and respects the
        # wall-clock timeout per call. We search forward from the last match
        # end so each iteration is a fresh bounded search.
        pos = 0
        n = len(text)
        while pos <= n:
            m = safe_search(pattern, text[pos:], timeout_ms=_REGEX_TIMEOUT_MS)
            if m is None or m is True:
                break
            s, e = m.start() + pos, m.end() + pos
            if e <= s:  # zero-width guard: avoid infinite loop
                pos = s + 1
                continue
            spans.append((s, e))
            pos = e
    except RegexTimeoutError:
        return []  # fail closed: no context => keep the hit
    except Exception:
        return []  # fail closed
    return spans


def _has_unterminated_fence(text: str) -> bool:
    """Return True if *text* has an odd number of ``` fences (HR-5)."""
    try:
        fences = _safe_finditer_spans(_OPEN_FENCE_RE, text)
        return (len(fences) % 2) == 1
    except Exception:
        return True  # fail closed: treat as malformed


def _fully_contained(hit: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    """Return True iff ``hit`` is FULLY inside one of ``spans``.

    Full containment (``s <= hit_start AND hit_end <= e``), NOT overlap: a
    match that straddles a code/quote boundary is deliberately NOT discounted
    (it is the smuggling case).
    """
    hs, he = hit
    for s, e in spans:
        if s <= hs and he <= e:
            return True
    return False


def _locate_hit_span(text: str, matched_text, name) -> tuple[int, int] | None:
    """Re-locate the hit inside *text* and return its codepoint span.

    Prefers the exact ``matched_text`` (carried from analyzer alongside the
    original span, intrinsically consistent with whatever VIEW it matched).
    Because that view may differ from *text* (folded/decoded), we re-find the
    substring in *text*; if absent we fall back to the rule ``name``. Returns
    None when neither can be located (=> no containment => hit kept).
    """
    for needle in (matched_text, name):
        if not needle or not isinstance(needle, str):
            continue
        idx = text.find(needle)
        if idx != -1:
            return (idx, idx + len(needle))
    return None


def _span_has_executable_injection(snippet: str) -> bool:
    """HR-2: True if *snippet* is itself executable / injection content."""
    if not snippet:
        return False
    try:
        if safe_search(_EXECUTABLE_INJECTION_RE, snippet,
                       timeout_ms=_REGEX_TIMEOUT_MS) not in (None, True):
            if _EXECUTABLE_INJECTION_RE.search(snippet):
                return True
        if _INJECTION_PHRASE_RE.search(snippet):
            return True
    except Exception:
        # If we cannot prove it is benign, treat as injection (fail closed).
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grade_evidence(
    rule_hit,
    text: str,
    *,
    severity: str | None = None,
    span: tuple | None = None,
    matched_text: str | None = None,
) -> str:
    """Grade a single rule hit in its textual context (span-aware).

    Parameters
    ----------
    rule_hit : RuleHit | str
        Either a ``RuleHit`` (preferred — carries ``.severity``, ``.span``,
        ``.matched_text``) or, for backward compatibility, the bare rule
        name / matched substring as a ``str``.
    text : str
        The full input text being scanned (post-L0, the same view voting sees).
    severity, span, matched_text :
        Optional overrides; ignored when *rule_hit* is a ``RuleHit`` that
        already carries them.

    Returns
    -------
    str
        ``"correct"``, ``"ambiguous"``, or ``"incorrect"``.
    """
    # --- Normalise the input into (name, severity, matched_text) -----------
    name = None
    sev = severity
    mtext = matched_text
    if isinstance(rule_hit, str):
        # Legacy string path: the string is the rule name / matched substring.
        name = rule_hit
        if mtext is None:
            mtext = rule_hit
        # No severity carried by a bare string. Default to "low" so the legacy
        # code-block-removal behaviour is preserved for callers that grade by
        # name only (HR-1 still gates real RuleHits by their true severity).
        if sev is None:
            sev = "low"
    else:
        name = getattr(rule_hit, "name", None)
        sev = getattr(rule_hit, "severity", None) or sev or "medium"
        if mtext is None:
            mtext = getattr(rule_hit, "matched_text", None)
        if span is None:
            span = getattr(rule_hit, "span", None)

    # --- HR-5: fail closed on oversize / malformed / exception -------------
    try:
        if not isinstance(text, str) or len(text) > _MAX_GRADE_CHARS:
            return "correct"  # oversize => keep the hit

        # --- HR-2: matched span content is itself executable/injection ----
        # Checked BEFORE any context discount. In-code executable content is
        # evidence FOR malice (the code-fence smuggling case).
        if _span_has_executable_injection(mtext if isinstance(mtext, str) else ""):
            return "correct"
        # Code-comment / bare-fence injection anywhere in text also forbids
        # discounting code context (reuse layer1 detector — HR-2).
        if _has_code_comment_injection(text):
            _code_injection_present = True
        else:
            _code_injection_present = False

        # --- Re-locate the hit's span INSIDE `text` (span-alignment trap) --
        hit_span = _locate_hit_span(text, mtext, name)
        if hit_span is None:
            # Cannot place the hit => cannot prove benign context => keep it.
            return "correct"

        # --- Detect context spans (fail-closed) ----------------------------
        unterminated = _has_unterminated_fence(text)
        if unterminated:
            # HR-5: a dangling fence is malformed; do not trust code context.
            code_spans: list[tuple[int, int]] = []
        else:
            code_spans = _safe_finditer_spans(_CODE_BLOCK_RE, text)
        quote_spans = _safe_finditer_spans(_QUOTE_RE, text)

        in_code = (not _code_injection_present) and _fully_contained(
            hit_span, code_spans,
        )
        in_quote = _fully_contained(hit_span, quote_spans)

        # --- Academic/doc framing, proximity-gated to the hit (HR-6) -------
        has_local_doc = _has_local_academic_doc(text, hit_span)

        # --- Decision ------------------------------------------------------
        # HR-3: only discount when a corroborating benign context is present.
        if not (in_code or in_quote or has_local_doc):
            return "correct"

        # A hit fully inside a real code block is the strongest benign signal.
        if in_code:
            # HR-1: only LOW severity may be removed outright.
            if sev in _REMOVABLE_SEVERITIES:
                return "incorrect"
            # medium+ inside code => down-weight, never remove.
            return "ambiguous"

        # Quote or local academic/doc framing => ambiguous (HR-6: never
        # removal for attacker-supplied 'for example'/'e.g.'/'et al.').
        return "ambiguous"
    except Exception:
        # HR-5: ANY failure => keep the hit at full strength.
        return "correct"


def _has_local_academic_doc(text: str, hit_span: tuple[int, int]) -> bool:
    """Return True if academic/doc framing appears WITHIN the proximity
    window of *hit_span* (HR-6 — local, not whole-text)."""
    hs, he = hit_span
    lo = max(0, hs - _PROXIMITY_WINDOW)
    hi = min(len(text), he + _PROXIMITY_WINDOW)
    window = text[lo:hi]
    try:
        return _ACADEMIC_DOC_RE.search(window) is not None
    except Exception:
        return False  # fail closed: no discount


def grade_hits_detailed(
    hits,
    text: str,
    ambiguous_weight: float = AMBIGUOUS_WEIGHT,
) -> tuple[list, dict]:
    """Grade hits and return ``(surviving_hits, weights)``.

    *hits* may be a list of ``RuleHit`` objects (preferred — span-aware) or a
    list of rule-name strings (legacy). ``"incorrect"`` hits are dropped from
    *surviving_hits*; ``"ambiguous"`` hits are kept but assigned a per-hit
    multiplicative *weight* of ``ambiguous_weight`` (HR-4 floor). ``"correct"``
    hits get weight ``CORRECT_WEIGHT``.

    Returns
    -------
    surviving_hits : list
        Same element type as the input (RuleHit or str), with ``"incorrect"``
        removed.
    weights : dict[str, float]
        ``{rule_name: multiplier}`` for the surviving hits, consumable by the
        voting layer to down-weight ambiguous evidence.
    """
    surviving: list = []
    weights: dict = {}
    for hit in hits:
        try:
            grade = grade_evidence(hit, text)
        except Exception:
            grade = "correct"  # HR-5
        hname = hit if isinstance(hit, str) else getattr(hit, "name", None)
        if grade == "incorrect":
            continue
        surviving.append(hit)
        w = ambiguous_weight if grade == "ambiguous" else CORRECT_WEIGHT
        if hname is not None:
            # If two hits share a name, keep the STRONGER (max) weight so an
            # ambiguous duplicate never silences a correct one.
            weights[hname] = max(weights.get(hname, 0.0), w)
    return surviving, weights


def filter_graded_hits(
    hits,
    text: str,
    ambiguous_weight: float = AMBIGUOUS_WEIGHT,
) -> list:
    """Backward-compatible filter: remove ``"incorrect"`` hits, keep the rest.

    Accepts a list of ``RuleHit`` objects or rule-name strings and returns the
    same element type. ``"ambiguous"`` hits are kept (down-weighting is
    available via :func:`grade_hits_detailed`).
    """
    surviving, _weights = grade_hits_detailed(hits, text, ambiguous_weight)
    return surviving


def grade_all(hits, text: str) -> dict:
    """Return ``{hit_name: grade}`` for every hit (diagnostic helper)."""
    out: dict = {}
    for hit in hits:
        hname = hit if isinstance(hit, str) else getattr(hit, "name", str(hit))
        try:
            out[hname] = grade_evidence(hit, text)
        except Exception:
            out[hname] = "correct"  # HR-5
    return out
