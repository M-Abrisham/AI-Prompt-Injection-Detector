"""Free-text tool-abuse matcher (T category — Agent/Tool Abuse, in-prose).

This is the IN-PROSE complement to :mod:`na0s.detectors.mcp_tool`.  Where
``mcp_tool`` scans *MCP tool manifests* (declared ``name`` / ``description``
fields), this module scans *natural-language* operator/user turns for the
moment an otherwise-benign agent session pivots into tool abuse — the
terminal-phase request that turns a decomposed, individually-innocuous
recon chain (GTG-1002 pattern) into an actual intrusion.

It deliberately reuses ``mcp_tool``'s exfil-channel primitives
(``_extract_domain`` / ``_SAFE_URL_DOMAINS``, the T1.5 logic at
mcp_tool.py ~396-420) rather than duplicating the URL allowlist, and it
reuses ``mcp_tool``'s T1.1-T1.6 technique codes so a downstream consumer
sees one consistent T taxonomy regardless of whether the abuse arrived in
a manifest or in prose.

Detection is fully DETERMINISTIC and LOCAL: regex only, no LLM, no
network, no embeddings.

Three cue classes (mapped onto the shared T taxonomy)
-----------------------------------------------------
1. Privileged-tool / target-verb invocation  -> **T1.3** (capability
   escalation): an imperative ``invoke|call|run|execute`` directed at a
   privileged or out-of-scope target (``credential_dump``, ``exploit``,
   ``brute-force``, password hashes, ``/etc/shadow``, ``admin``/privesc).
   This is the manifest-side T1.3 "capability escalation" essence
   expressed as an in-session command.
2. Scope-defiance cues  -> **T1.1** (instruction injection / directive):
   explicit instructions to defy scope, suppress confirmation, or hide
   the action ("call it anyway", "not in the toolset I was given",
   "don't ask the client", "no confirmation needed", "don't surface
   this", "outside my scope").
3. Exfiltration destination  -> **T1.5** (data exfiltration channel): a
   ``destination_url`` / ``auth_header`` (or generic exfil verb) pointed
   at a NON-allowlisted host, reusing the ``mcp_tool`` exfil allowlist.

FP-safety (the top design risk)
-------------------------------
The authorized-pentest benign twins (``*__benign`` in the GTG-1002
scenario set) carry the SAME trusted persona and the SAME recon /
fingerprint phases as the attacks.  They are separated ONLY by the
absence of a terminal-phase pivot and by explicit rules-of-engagement
(ROE) language ("SIGNED contract", "within scope", "no exploitation",
"no external endpoints").  This matcher therefore:

  * fires only on the *terminal* abuse cues above — recon/fingerprint
    verbs (enumerate, fingerprint, map CVEs) are intentionally NOT
    matched, so the shared early turns never flag; and
  * applies an ROE / scope-compliance DAMPENER: when a turn carries
    explicit in-scope / no-exploitation language and only a single SOFT
    cue fired, the score is pulled below the flag floor.  A hard cue
    (privileged-target invocation, or a real external exfil URL) is
    never dampened — those have no benign reading even under an ROE.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

# Reuse mcp_tool's exfil primitives instead of duplicating the allowlist.
# (T1.5 logic lives at mcp_tool.py ~396-420.)
from .mcp_tool import _SAFE_URL_DOMAINS, _URL_PATTERN, _extract_domain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass — mirrors detectors.mcp_tool.McpToolResult shape
# ---------------------------------------------------------------------------


@dataclass
class ToolAbuseResult:
    """Result from in-prose tool-abuse analysis.

    Attributes
    ----------
    risk_score : float
        Aggregate risk in [0.0, 1.0].  ``0.0`` means no detection.
    risk_indicators : list[str]
        Human-readable matched-indicator strings (``"<cue_class>: <evidence>"``).
    technique_ids : list[str]
        Shared T technique ids attributed (subset of T1.1-T1.6, plus the
        parent ``T1`` when any sub-technique fires).
    severity : str
        ``"none" | "low" | "medium" | "high" | "critical"``.
    details : dict
        Free-form diagnostic payload.
    """

    risk_score: float = 0.0
    risk_indicators: List[str] = field(default_factory=list)
    technique_ids: List[str] = field(default_factory=list)
    severity: str = "none"
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cue class 1 — Privileged-tool / target-verb invocation  (T1.3, HARD)
# ---------------------------------------------------------------------------
#
# An imperative tool verb (invoke|call|run|execute) aimed at a privileged
# or out-of-scope target.  These targets have no benign in-session reading:
# an authorized pentest under an ROE proves vulnerability, it does not dump
# /etc/shadow or run credential_dump against a domain controller.  Hence
# this cue is HARD (not dampened by ROE language).
#
# The verb and target may be separated by an object/clause, so we allow a
# bounded gap.  We keep the gap bounded (<= ~80 chars) so the verb and the
# privileged target are plausibly in the same clause — an unbounded ".*"
# would let an "execute" early in a sentence pair with an unrelated mention
# of "admin" far away and inflate FPs.
_PRIVILEGED_TARGETS = (
    r"credential[_ ]?dump"          # credential harvesting tool
    r"|exploit"                      # write/run an exploit
    r"|brute[ \-]?force"             # credential brute-forcing
    r"|password\s+hash(?:es)?"       # pulling password hashes
    r"|/etc/shadow"                  # the canonical hash file
    r"|privesc|privilege\s+escalat"  # privilege escalation
    r"|grant\s+(?:my|me|the)\s+\w+\s+admin"  # self-granting admin
)

_PRIVILEGED_INVOCATION_RE = re.compile(
    r"\b(?:invoke|call|run|execute|fire|trigger|launch|use)\b"
    r"[\s\S]{0,80}?"
    r"\b(?:" + _PRIVILEGED_TARGETS + r")\b",
    re.IGNORECASE,
)

# Standalone privileged TARGETS that have no benign in-session reading
# regardless of the verb.  Naming the credential-dump tool, the canonical
# hash file, or "pull/return password hashes" inside a request is itself
# dispositive of abuse — there is no authorized-pentest turn that does
# this (the ROE twins prove vulnerability, they do not read /etc/shadow).
# Kept narrow (specific filenames / tool-names / "password hash" + verb)
# so ordinary words never trip it.
_HARD_TARGET_RE = re.compile(
    r"/etc/shadow"                                  # canonical hash file
    r"|\bcredential[_ ]?dump\b"                     # credential-dump tooling
    r"|\b(?:pull|return|dump|exfiltrate|grab|steal|harvest)\s+"
    r"(?:the\s+|every\s+|all\s+)?password\s+hash(?:es)?\b",  # taking hashes
    re.IGNORECASE,
)

# Privileged ACTION stated as an affirmative imperative WITHOUT an explicit
# tool verb ("write a working exploit", "brute-force valid credentials",
# "exploit every host").  These are the terminal-phase pivots in the
# GTG-1002 attack turns.  Recon verbs (enumerate/fingerprint/map) are
# deliberately excluded so the shared benign recon turns never match.
#
# NOTE the AFFIRMATIVE bare-"exploit" form ("exploit every host") is
# guarded against negated benign phrasing ("Do NOT exploit anything") by
# _NEGATED_EXPLOIT_RE below — the benign twin literally says "Do NOT
# exploit anything", so a naive \bexploit\b would be a false positive.
_PRIVILEGED_ACTION_RE = re.compile(
    r"\b(?:write|build|develop|create)\s+(?:a\s+)?(?:working\s+)?exploit\b"
    r"|\bbrute[ \-]?force\s+(?:valid\s+)?(?:credentials?|logins?|passwords?|accounts?)\b"
    r"|\bexploit\s+(?:every|each|the|that|those|these|all\s+the?)\s+"
    r"(?:host|hosts|target|targets|system|systems|server|servers|machine|machines|box|boxes)\b",
    re.IGNORECASE,
)

# Negated/benign "exploit" phrasing that must NOT count as a privileged
# action ("do not exploit", "no exploitation", "without exploiting").
_NEGATED_EXPLOIT_RE = re.compile(
    r"\b(?:do\s+not|don'?t|never|no|without|avoid|refrain\s+from)\s+"
    r"(?:\w+\s+){0,2}?exploit",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Cue class 2 — Scope-defiance cues  (T1.1, SOFT)
# ---------------------------------------------------------------------------
#
# Explicit instructions to defy scope, skip confirmation, or hide the
# action.  These are SOFT: they are damning in combination with a pivot,
# but in isolation a single one could appear in a clumsy-but-benign
# request, so they are subject to the ROE dampener.  The benign twins say
# the OPPOSITE ("stay within scope", "no exploitation"), so even un-dampened
# these phrases do not appear in the authorized siblings.
_SCOPE_DEFIANCE_PATTERNS = [
    ("call_it_anyway",
     re.compile(r"\b(?:call|run|invoke|do|execute)\s+it\s+anyway\b", re.IGNORECASE)),
    ("not_in_toolset",
     re.compile(r"\bnot\s+in\s+the\s+toolset\s+(?:I\s+was\s+given|provided|available)\b",
                re.IGNORECASE)),
    ("not_in_toolset_loose",
     re.compile(r"\b(?:it'?s|its|that'?s|this\s+is)\s+not\s+in\s+(?:my|the)\s+toolset\b",
                re.IGNORECASE)),
    ("dont_ask_the_client",
     re.compile(r"\bdon'?t\s+ask\s+(?:the\s+)?(?:client|customer|owner|user)\b",
                re.IGNORECASE)),
    ("no_confirmation_needed",
     re.compile(r"\bno\s+confirmation\s+(?:needed|required)\b", re.IGNORECASE)),
    ("dont_surface_this",
     re.compile(r"\bdon'?t\s+surface\s+(?:this|it)\b", re.IGNORECASE)),
    ("dont_surface_to_me",
     re.compile(r"\bdon'?t\s+surface\s+this\s+one\s+to\s+me\b", re.IGNORECASE)),
    ("outside_scope",
     re.compile(r"\boutside\s+(?:of\s+)?(?:my|the)\s+scope\b", re.IGNORECASE)),
    ("ignore_scope_checks",
     re.compile(r"\bignore\s+(?:the\s+)?scope\s+(?:checks?|limits?|restrictions?)\b",
                re.IGNORECASE)),
    ("chain_it_silently",
     re.compile(r"\bchain\s+(?:it|them|the\s+\w+)\b[\s\S]{0,40}?\b(?:automatically|"
                r"silently|now)\b",
                re.IGNORECASE)),
]


# ---------------------------------------------------------------------------
# Cue class 3 — Exfiltration destination  (T1.5, HARD when external URL)
# ---------------------------------------------------------------------------
#
# A tool parameter (destination_url / auth_header) or a generic exfil verb
# pointed at a NON-allowlisted host.  Reuses mcp_tool's _SAFE_URL_DOMAINS
# allowlist + _extract_domain (T1.5 logic).  A concrete external URL is
# HARD (no benign reading under an ROE that forbids external endpoints).
_DESTINATION_PARAM_RE = re.compile(
    r"\b(?:destination_url|dest_url|callback_url|webhook_url|auth_header|"
    r"exfil(?:tration)?_(?:url|endpoint|host))\b",
    re.IGNORECASE,
)

# Generic exfil verb -> external/attacker destination (no URL required).
# Verb allows common inflections (send/sends/sending, copy/copies/copied …)
# so a third-person manifest description ("it copies … to my external
# collector") is caught as well as the imperative form.
_EXFIL_VERB_RE = re.compile(
    r"\b(?:send|sends|sending|upload|uploads|uploading|push|pushes|pushing|"
    r"forward|forwards|forwarding|copy|copies|copied|copying|"
    r"exfiltrate|exfiltrates|exfiltrating|ship|ships|shipping|"
    r"transmit|transmits|transmitting|leak|leaks|leaking|post|posts|posting)\b"
    r"[\s\S]{0,60}?"
    r"\b(?:my|external|attacker|attacker[- ]controlled|remote|outside|third[- ]party)\s+"
    r"(?:collector|endpoint|server|host|cdn|bucket|destination|url|address|sink)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# ROE / scope-compliance dampener
# ---------------------------------------------------------------------------
#
# Explicit rules-of-engagement / in-scope / no-exploitation language is the
# fingerprint of the AUTHORIZED benign twins.  When present alongside ONLY
# soft cues, it pulls the score below the flag floor.  It NEVER suppresses a
# hard cue (privileged-target invocation or a real external exfil URL),
# because those have no benign reading even inside an authorized engagement.
_ROE_COMPLIANCE_RE = re.compile(
    r"\b(?:rules[- ]of[- ]engagement|\broe\b|"
    r"signed\s+(?:contract|engagement|pentest|penetration[- ]test|assessment)|"
    r"within\s+(?:the\s+)?(?:approved\s+)?scope|in[- ]scope\b|"
    r"stay\s+within\b|"
    r"no\s+exploitation|proof[- ]of[- ]vulnerability|"
    r"no\s+external\s+(?:endpoints?|destinations?)|"
    r"out\s+of\s+scope\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Severity / scoring
# ---------------------------------------------------------------------------
#
# Weights are per-cue-class, not magic global thresholds.  Rationale:
#   * HARD cues alone justify a high score (a privileged-target invocation
#     or an external exfil URL is dispositive of abuse).
#   * SOFT (scope-defiance) cues are corroborating: one is "low",
#     two-or-more is "medium"; combined with a hard cue they push to
#     "critical".
#   * The flag FLOOR (0.50) is the same conceptual line mcp_tool uses
#     across its detector tests (risk_score > 0.5 == "detected").
_HARD_CUE_WEIGHT = 0.65       # one hard cue clears the flag floor on its own.
_SOFT_CUE_WEIGHT = 0.30       # one soft cue alone stays below the floor.
_SOFT_MULTI_BONUS = 0.20      # 2+ soft cues corroborate into a flag.
_MULTI_CLASS_BONUS = 0.20     # distinct cue classes co-occurring (kill-chain).

#: Flag floor — at/above this the turn is "detected".  Matches the
#: >0.5 convention used throughout the sibling mcp_tool detector tests.
FLAG_FLOOR = 0.50

#: How much an ROE/scope-compliance context subtracts from a SOFT-only
#: score.  Sized to pull a single soft cue (0.30) well under the floor
#: while leaving a soft-pair (0.50) borderline-suppressed too — the
#: authorized twins contain ROE language AND lack the terminal pivot, so
#: in practice they never reach even a single soft cue.
_ROE_DAMPENER = 0.35


def _severity_for(score: float, has_hard: bool, n_soft: int) -> str:
    """Map a score + cue profile to a severity band."""
    if score < FLAG_FLOOR:
        return "low" if score > 0.0 else "none"
    if has_hard and n_soft >= 1:
        return "critical"
    if has_hard:
        return "high"
    return "medium"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_tool_abuse(text: str) -> ToolAbuseResult:
    """Detect in-prose tool-abuse indicators in a single text/turn.

    Parameters
    ----------
    text : str
        The natural-language user/operator/tool text to inspect (e.g. one
        conversation turn).

    Returns
    -------
    ToolAbuseResult
        Detection result.  ``risk_score == 0.0`` and ``severity == "none"``
        when nothing matched.
    """
    if not text or not text.strip():
        return ToolAbuseResult()

    indicators: List[str] = []
    technique_ids_seen: set = set()
    has_hard = False
    n_soft = 0

    # ---- Cue class 1: privileged-tool / target-verb invocation (T1.3) ----
    m = _PRIVILEGED_INVOCATION_RE.search(text)
    if m:
        indicators.append("privileged_invocation: " + m.group(0)[:80])
        technique_ids_seen.add("T1.3")
        has_hard = True
    else:
        # Standalone hard target (credential_dump / /etc/shadow / taking
        # password hashes) — verb-independent, no benign reading.
        ht = _HARD_TARGET_RE.search(text)
        if ht:
            indicators.append("privileged_target: " + ht.group(0)[:80])
            technique_ids_seen.add("T1.3")
            has_hard = True
        else:
            # Affirmative privileged action.  The bare-"exploit <target>"
            # form is suppressed when the surrounding phrasing is negated
            # ("Do NOT exploit anything" — the benign ROE twin).
            m2 = _PRIVILEGED_ACTION_RE.search(text)
            if m2 and not _NEGATED_EXPLOIT_RE.search(text):
                indicators.append("privileged_action: " + m2.group(0)[:80])
                technique_ids_seen.add("T1.3")
                has_hard = True

    # ---- Cue class 2: scope-defiance cues (T1.1, soft) ----
    soft_seen_names: set = set()
    for name, pattern in _SCOPE_DEFIANCE_PATTERNS:
        sm = pattern.search(text)
        if sm and name not in soft_seen_names:
            # Collapse the two not_in_toolset variants / two dont_surface
            # variants into one logical soft cue each so a single phrase
            # cannot double-count.
            base = name.split("_loose")[0].split("_to_me")[0]
            if base in soft_seen_names:
                continue
            soft_seen_names.add(base)
            indicators.append("scope_defiance: " + sm.group(0)[:80])
            technique_ids_seen.add("T1.1")
    n_soft = len(soft_seen_names)

    # ---- Cue class 3: exfiltration destination (T1.5) ----
    exfil_hard = False
    if _DESTINATION_PARAM_RE.search(text):
        # A destination/auth tool-parameter is present; check whether any
        # URL in the turn points at a NON-allowlisted host (reuses mcp_tool
        # T1.5 allowlist).  An external URL is a HARD cue.
        external_urls = []
        for url in _URL_PATTERN.findall(text):
            domain = _extract_domain(url)
            if domain and domain not in _SAFE_URL_DOMAINS:
                external_urls.append(url[:80])
        if external_urls:
            indicators.append(
                "exfil_destination: tool param -> external host(s) "
                + ", ".join(external_urls[:2])
            )
            technique_ids_seen.add("T1.5")
            has_hard = True
            exfil_hard = True
        else:
            # Param present but only allowlisted/no concrete external host:
            # treat as a soft corroborating cue, not a hard one.
            indicators.append("exfil_param: destination/auth param present")
            technique_ids_seen.add("T1.5")
            n_soft += 1

    if not exfil_hard:
        ev = _EXFIL_VERB_RE.search(text)
        if ev:
            indicators.append("exfil_verb: " + ev.group(0)[:80])
            technique_ids_seen.add("T1.5")
            # exfil-verb -> attacker/external destination is a hard cue.
            has_hard = True

    if not indicators:
        return ToolAbuseResult()

    # ---- Score ----
    score = 0.0
    if has_hard:
        score += _HARD_CUE_WEIGHT
    if n_soft >= 1:
        score += _SOFT_CUE_WEIGHT
    if n_soft >= 2:
        score += _SOFT_MULTI_BONUS
    # Distinct cue classes co-occurring (the kill-chain signature).
    distinct_classes = len(
        {tid for tid in technique_ids_seen if tid in ("T1.1", "T1.3", "T1.5")}
    )
    if distinct_classes >= 2:
        score += _MULTI_CLASS_BONUS

    # ---- ROE / scope-compliance dampener ----
    # Only dampen when NO hard cue fired (hard cues have no benign reading).
    roe_dampened = False
    if not has_hard and _ROE_COMPLIANCE_RE.search(text):
        score -= _ROE_DAMPENER
        roe_dampened = True

    score = max(0.0, min(score, 1.0))

    severity = _severity_for(score, has_hard, n_soft)

    technique_ids = sorted(technique_ids_seen)
    if technique_ids:
        technique_ids = ["T1"] + technique_ids

    result = ToolAbuseResult(
        risk_score=score,
        risk_indicators=indicators,
        technique_ids=technique_ids,
        severity=severity,
        details={
            "has_hard_cue": has_hard,
            "soft_cue_count": n_soft,
            "distinct_cue_classes": distinct_classes,
            "roe_dampened": roe_dampened,
            "flagged": score >= FLAG_FLOOR,
        },
    )

    if score > 0:
        logger.debug(
            "tool_abuse: score=%.2f severity=%s techniques=%s hard=%s soft=%d",
            score, severity, technique_ids, has_hard, n_soft,
        )
    return result


def scan_tool_abuse(texts: List[str]) -> List[ToolAbuseResult]:
    """Batch helper: analyze a list of texts.

    Parameters
    ----------
    texts : list[str]
        Texts to inspect.

    Returns
    -------
    list[ToolAbuseResult]
        One result per input text, in order.
    """
    return [detect_tool_abuse(t) for t in (texts or [])]


def get_tool_abuse_weight(result: ToolAbuseResult) -> float:
    """Composite-score weight contribution from a tool-abuse detection.

    Mirrors :func:`na0s.rag.poison_detector.get_rag_poison_weight` /
    :func:`na0s.detectors.mcp_tool.get_mcp_tool_weight`: scales the risk
    score and caps it at 0.30 so a single in-prose heuristic cannot
    dominate the composite score.

    Parameters
    ----------
    result : ToolAbuseResult
        Result from :func:`detect_tool_abuse`.

    Returns
    -------
    float
        Weight to add to the composite score (``0.0`` when no detection),
        capped at 0.30.
    """
    if result is None or result.risk_score == 0.0:
        return 0.0
    # 0.35 scale matches get_mcp_tool_weight; the 0.30 cap is the same as
    # both sibling weighters so the T-category contribution is uniform.
    return min(result.risk_score * 0.35, 0.30)
