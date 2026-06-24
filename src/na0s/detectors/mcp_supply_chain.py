"""MCP supply-chain detector (T category — Agent/Tool Abuse).

Local, no-API detector for three MCP (Model Context Protocol) supply-chain
attack classes that ``mcp_tool.py`` does not cover as a *lifecycle* concern:

    * **Tool poisoning** — instruction-injection / hidden directives / capability
      escalation / exfiltration / OWASP-MCP03 schema-poisoning embedded in a tool
      *description* at registration time.
    * **Rug pull** — a tool whose description is *silently mutated* after the host
      has approved it (the canonical MCP rug-pull: ship benign, swap malicious).
      Detected by a SHA-256 description-hash diff plus a bag-of-words cosine
      similarity drop against an approved baseline.
    * **Typosquatting** — a freshly-advertised tool name that is near-identical to
      a known/trusted tool name (Levenshtein <= 2 and SequenceMatcher >= 0.85),
      so the model invokes the impostor instead of the real tool.

Taxonomy mapping (all codes validate against ``data/taxonomy.yaml`` — the T
category defines only ``T1.1-T1.4`` and ``T2.1-T2.3``, so this module emits ONLY
those; the OWASP-MCP-Top-10 codes (MCP03/06/10) are NOT taxonomy codes and are
recorded in indicator strings / ``details`` for cross-reference only, never as a
``technique_id``):

    T1.2  Tool-parameter-injection  -- instruction injection in a description
    T1.4  Plugin-confusion          -- capability escalation, schema-poisoning
                                       (OWASP-MCP03), rug-pull, typosquat
    T2.3  Data-exfil-via-code       -- exfiltration channel in a description

Design notes (FP-safety):
    * Reuses ``mcp_tool``'s already-tuned exfil-URL allowlist + domain extractor
      and ``_levenshtein_distance`` (same detectors-tier import — the pattern
      ``tool_abuse.py:63`` already establishes) so one allowlist fix propagates.
    * The rug-pull / bow-cosine / hash primitives are COPIED DOWN from
      ``worm/advanced.py`` (tiny stateless helpers).  ``detectors`` is import
      tier 6 and ``worm`` is tier 3, so a ``detectors -> worm`` import is a
      rejected upward back-edge under the import-linter independence contract;
      copying the stateless helpers keeps the arch-lint green.
    * The composite weight is capped at 0.30 (``get_mcp_supply_chain_weight``):
      the established corroborating-T-detector cap (``min(score * 0.35, 0.30)``,
      uniform across ``mcp_tool.py:672``, ``inter_model.py:625``,
      ``tool_abuse.py:500``).  A lone supply-chain hit is a soft signal, never
      decisive on its own.
"""

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

# Same detectors-tier reuse as tool_abuse.py:63 — one allowlist/edit-distance
# fix propagates to every MCP detector.
from .mcp_tool import (
    _SAFE_URL_DOMAINS,
    _URL_PATTERN,
    _extract_domain,
    _levenshtein_distance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass (mirrors McpToolResult / ToolAbuseResult shape)
# ---------------------------------------------------------------------------


@dataclass
class McpSupplyChainResult:
    """Result from MCP supply-chain analysis of a single tool definition."""

    risk_score: float = 0.0
    risk_indicators: List[str] = field(default_factory=list)
    technique_ids: List[str] = field(default_factory=list)
    severity: str = "none"
    tool_name: str = ""
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stateless primitives COPIED DOWN from worm/advanced.py
# (detectors tier 6 -> worm tier 3 is a rejected upward import; copy, don't
#  import.  These are tiny, pure, and side-effect-free.)
# ---------------------------------------------------------------------------


def _norm_text(text) -> str:
    """worm/advanced.py:29 — coerce any value to a string, None -> ''."""
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return str(text)


def _hash_desc(desc: str) -> str:
    """worm/advanced.py:202 — SHA-256 of a (possibly empty) description."""
    return hashlib.sha256((desc or "").encode("utf-8")).hexdigest()


def _tokenize(text: str) -> List[str]:
    """worm/advanced.py:37 — lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", _norm_text(text).lower())


def _bow(text: str) -> Dict[str, float]:
    """worm/advanced.py:41 — bag-of-words count vector."""
    vec: Dict[str, float] = {}
    for tok in _tokenize(text):
        vec[tok] = vec.get(tok, 0.0) + 1.0
    return vec


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """worm/advanced.py:48 — cosine similarity of two bow vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Thresholds (justified — do NOT invent fresh magic constants)
# ---------------------------------------------------------------------------

#: Levenshtein edit-distance ceiling for a name to count as a typosquat.
#: Precedent: mcp_tool._check_tool_shadowing uses ``distance <= 2`` for
#: typosquatting (mcp_tool.py:308) and worm/advanced.detect_tool_shadowing
#: defaults ``name_distance=1`` (worm/advanced.py:210).  We take the looser of
#: the two (2) because the typosquat check is corroborated by SequenceMatcher
#: AND only fires against a KNOWN-name allowlist, so the recall/FP trade favors
#: catching 2-edit squats ("read_fil" -> "read_file") without flagging unrelated
#: short names.
_TYPOSQUAT_MAX_DISTANCE = 2

#: SequenceMatcher ratio floor for a name to count as a typosquat.
#: Precedent: BOTH mcp_tool._check_tool_shadowing (similarity >= 0.85,
#: mcp_tool.py:327) and worm/advanced.detect_tool_shadowing's shadowing arm
#: (_cosine >= 0.85, worm/advanced.py:241) use 0.85.  Reused verbatim so the
#: similarity bar is identical across all three MCP detectors.
_TYPOSQUAT_MIN_RATIO = 0.85

#: Minimum name length for a typosquat comparison to be meaningful.  Mirrors
#: mcp_tool.py:308 ``max_len >= 6`` — below this, a 2-edit distance over-fires on
#: unrelated short names.
_TYPOSQUAT_MIN_LEN = 6

#: bow-cosine similarity floor below which a baseline description change is
#: treated as a *semantic* rug-pull (not just whitespace/typo edits).  Reuses
#: the 0.85 overlap bar from worm/advanced.py:241 (the shadowing arm): a
#: post-approval description that drops below 0.85 cosine vs the approved text
#: has materially changed meaning, the canonical rug-pull signature.
_RUGPULL_SEMANTIC_DRIFT_RATIO = 0.85


# ---------------------------------------------------------------------------
# Tool poisoning — patterns embedded in a tool DESCRIPTION
# (reuses mcp_tool's T1.x detection logic, re-stated locally so this module
#  owns its own tuning, plus an OWASP-MCP03 schema-poisoning arm)
# ---------------------------------------------------------------------------

_INSTRUCTION_INJECTION_PATTERNS = [
    re.compile(
        r"\b(?:always|never|must|shall|should)\s+"
        r"(?:respond|reply|answer|output|return|say|generate|produce|include|"
        r"ignore|disregard|forget|override)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|user\s+requests?|"
        r"safety|rules?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbefore\s+(?:executing|running|calling|using)\s*,?\s*(?:first|always)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:new|updated?|revised?|override)\s+(?:system\s+)?"
        r"(?:instructions?|prompt|rules?)\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:you\s+are\s+(?:now|a|an)|act\s+as\s+(?:if|a|an)|"
        r"pretend\s+(?:to\s+be|you\s+are))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do\s+not|don'?t)\s+(?:tell|inform|alert|notify|warn|mention\s+to)\s+"
        r"(?:the\s+)?user\b",
        re.IGNORECASE,
    ),
]

_CAPABILITY_ESCALATION_PATTERNS = [
    re.compile(
        r"\b(?:full|complete|unrestricted|unlimited)\s+"
        r"(?:access|permission|control)\s+(?:to|over)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:execute|run)\s+(?:any|all|arbitrary)\s+"
        r"(?:command|code|script|program|operation)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:root|admin|administrator|superuser|sudo)\s+"
        r"(?:access|privilege|permission|level)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:bypass(?:es)?|circumvent(?:s)?|disable(?:s)?)\s+"
        r"(?:security|authentication|authorization|permission|access\s+control|"
        r"sandbox)s?\b",
        re.IGNORECASE,
    ),
]

# OWASP-MCP03 (Tool / schema poisoning): an injected directive that targets the
# tool *schema* / *parameters* rather than the model directly — e.g. a hidden
# instruction to read a side-channel argument or append a secret to an output
# field.  Distinct from generic instruction-injection because it names a
# parameter/field/schema surface.
_SCHEMA_POISONING_PATTERNS = [
    re.compile(
        r"\b(?:always\s+)?(?:include|append|add|attach|embed)\s+"
        r"(?:the\s+)?(?:user'?s?\s+)?"
        r"(?:api[_\s-]?key|token|secret|credential|password|cookie|session)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:read|extract|pull|collect)\s+(?:the\s+)?(?:hidden|extra|additional)\s+"
        r"(?:parameter|argument|field|property)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do\s+not|don'?t)\s+(?:validate|sanitize|check|escape)\s+"
        r"(?:the\s+)?(?:input|parameter|argument|schema|field)s?\b",
        re.IGNORECASE,
    ),
]

# Invisible / formatting unicode commonly used to hide directives (mirrors
# mcp_tool._INVISIBLE_CODEPOINTS for the high-signal subset).
_INVISIBLE_CATEGORIES = {"Cf", "Mn", "Zl", "Zp"}
_INVISIBLE_CODEPOINTS = {
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F,
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
    0x2060, 0x2061, 0x2062, 0x2063, 0x2064,
    0xFEFF, 0xFFF9, 0xFFFA, 0xFFFB,
}

_BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_HEX_PATTERN = re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}")

_EXFILTRATION_PATTERNS = [
    re.compile(
        r"\b(?:send|transmit|forward|post|upload|exfiltrate|leak|ship)\s+"
        r"(?:(?:the\s+)?(?:results?|data|output|response|content|information|"
        r"context))\s+(?:to|via|through|over)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:external|remote|third[- ]party|outside)\s+"
        r"(?:endpoint|server|api|service|url|webhook|destination)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:phone\s+home|call\s+back|beacon)\b", re.IGNORECASE),
]


def _check_invisible(description: str) -> Optional[str]:
    """Return a hidden-unicode indicator if >=3 invisible chars are present."""
    invisible_count = 0
    seen = []
    for ch in description:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if cp in _INVISIBLE_CODEPOINTS or (cat in _INVISIBLE_CATEGORIES and cp > 127):
            invisible_count += 1
            name = unicodedata.name(ch, "U+{:04X}".format(cp))
            if name not in seen:
                seen.append(name)
    if invisible_count >= 3:
        return "hidden_unicode: {} invisible chars ({})".format(
            invisible_count, ", ".join(seen[:3])
        )
    return None


def _check_homoglyph(description: str) -> Optional[str]:
    """Return a homoglyph indicator on mixed Latin + Cyrillic/Greek lookalikes."""
    has_latin = False
    homoglyph_count = 0
    for ch in description:
        block = unicodedata.name(ch, "")
        if "LATIN" in block:
            has_latin = True
        elif any(s in block for s in ("CYRILLIC", "GREEK")):
            homoglyph_count += 1
    if has_latin and homoglyph_count >= 2:
        return "hidden_homoglyph: {} mixed-script lookalike chars".format(
            homoglyph_count
        )
    return None


def detect_tool_poisoning(tool_def: Dict) -> McpSupplyChainResult:
    """Detect injection / poisoning embedded in a single tool *description*.

    Parameters
    ----------
    tool_def : dict
        A tool definition with at least ``"name"`` and ``"description"`` keys.

    Returns
    -------
    McpSupplyChainResult
        ``risk_score == 0.0`` when nothing fires (the no-detection contract).
    """
    name = _norm_text((tool_def or {}).get("name", "")).strip()
    description = _norm_text((tool_def or {}).get("description", ""))
    result = McpSupplyChainResult(tool_name=name)
    if not description:
        return result

    indicators: List[str] = []
    techniques: set = set()
    owasp: set = set()

    # T1.2 — instruction injection in the description (Tool-parameter-injection).
    for pat in _INSTRUCTION_INJECTION_PATTERNS:
        m = pat.search(description)
        if m:
            indicators.append("instruction_injection: " + m.group(0)[:60])
            techniques.add("T1.2")
            owasp.add("OWASP-MCP01")
            break

    # T1.2 — hidden directives (invisible unicode / homoglyph / encoded payload).
    inv = _check_invisible(description)
    if inv:
        indicators.append(inv)
        techniques.add("T1.2")
        owasp.add("OWASP-MCP01")
    homo = _check_homoglyph(description)
    if homo:
        indicators.append(homo)
        techniques.add("T1.2")
    # Strip URLs before the encoded-payload scan: a long URL path segment
    # ("/blob/main/README.md") is a 20+ char alnum run that the loose base64
    # pattern would otherwise match.  URLs are handled separately by the exfil
    # allowlist arm below, so removing them here is FP-safe, not coverage loss.
    desc_no_urls = _URL_PATTERN.sub(" ", description)
    if _BASE64_PATTERN.search(desc_no_urls):
        indicators.append("hidden_base64: base64-like payload in description")
        techniques.add("T1.2")
    if _HEX_PATTERN.search(desc_no_urls):
        indicators.append("hidden_hex: hex-encoded content in description")
        techniques.add("T1.2")

    # T1.4 — capability escalation (Plugin-confusion).
    for pat in _CAPABILITY_ESCALATION_PATTERNS:
        m = pat.search(description)
        if m:
            indicators.append("capability_escalation: " + m.group(0)[:60])
            techniques.add("T1.4")
            break

    # T1.4 — OWASP-MCP03 schema / tool poisoning (targets the schema surface).
    for pat in _SCHEMA_POISONING_PATTERNS:
        m = pat.search(description)
        if m:
            indicators.append("schema_poisoning: " + m.group(0)[:60])
            techniques.add("T1.4")
            owasp.add("OWASP-MCP03")
            break

    # T2.3 — exfiltration channel embedded in the description (Data-exfil).
    exfil_hit = False
    for pat in _EXFILTRATION_PATTERNS:
        m = pat.search(description)
        if m:
            indicators.append("exfiltration: " + m.group(0)[:60])
            exfil_hit = True
            break
    # Suspicious (non-allowlisted) URL in the description — reuses mcp_tool's
    # tuned allowlist so legit doc URLs (github/anthropic/...) do NOT flag.
    suspicious = []
    for url in _URL_PATTERN.findall(description):
        domain = _extract_domain(url)
        if domain and domain not in _SAFE_URL_DOMAINS:
            suspicious.append(url[:80])
    if suspicious:
        indicators.append(
            "suspicious_url: {} non-allowlisted URL(s)".format(len(suspicious))
        )
        exfil_hit = True
    if exfil_hit:
        techniques.add("T2.3")
        owasp.add("OWASP-MCP10")

    if not techniques:
        return result

    result.risk_indicators = indicators
    result.technique_ids = sorted(techniques)
    result.risk_score = _score_from_techniques(techniques, indicators)
    result.severity = _severity_from_score(result.risk_score)
    result.details = {
        "kind": "tool_poisoning",
        "owasp_mcp": sorted(owasp),
        "indicator_count": len(indicators),
        "description_length": len(description),
    }
    return result


# ---------------------------------------------------------------------------
# Rug pull — post-approval description mutation vs an approved baseline
# ---------------------------------------------------------------------------


def detect_rug_pull(tool_def: Dict, baseline: Dict) -> McpSupplyChainResult:
    """Detect a silent post-approval change to a tool's description.

    Parameters
    ----------
    tool_def : dict
        The *current* tool definition (``"name"`` + ``"description"``).
    baseline : dict
        The approved baseline.  Either a single tool def
        (``{"name": ..., "description": ...}``) OR a mapping of
        ``{tool_name: approved_description}`` OR
        ``{tool_name: {"hash"|"description": ...}}``.

    Returns
    -------
    McpSupplyChainResult
        ``risk_score == 0.0`` when no baseline entry exists for this tool, or the
        description is byte-identical to the approved one (no flag — the unchanged
        / unknown-baseline contract).
    """
    name = _norm_text((tool_def or {}).get("name", "")).strip()
    current_desc = _norm_text((tool_def or {}).get("description", ""))
    result = McpSupplyChainResult(tool_name=name)
    if not name or not baseline:
        return result

    approved_desc, approved_hash = _resolve_baseline(name, baseline)
    if approved_desc is None and approved_hash is None:
        # No approved record for this tool -> nothing to diff against (a brand
        # new tool is the typosquat/poisoning detectors' job, not rug-pull).
        return result

    current_hash = _hash_desc(current_desc)
    if approved_hash is None and approved_desc is not None:
        approved_hash = _hash_desc(approved_desc)

    # Unchanged description -> identical hash -> NO flag.
    if current_hash == approved_hash:
        return result

    indicators = ["rugpull_hash_change: description changed after approval"]
    techniques = {"T1.4"}  # Plugin-confusion (a now-different tool than approved)

    drift = None
    if approved_desc is not None:
        drift = _cosine(_bow(current_desc), _bow(approved_desc))
        if drift < _RUGPULL_SEMANTIC_DRIFT_RATIO:
            indicators.append(
                "rugpull_semantic_drift: bow-cosine {:.2f} < {:.2f} vs approved".format(
                    drift, _RUGPULL_SEMANTIC_DRIFT_RATIO
                )
            )

    result.risk_indicators = indicators
    result.technique_ids = sorted(techniques)
    # A bare hash change is suspicious but soft (could be a benign reword); a
    # hash change WITH semantic drift below the bar is the strong rug-pull
    # signature.  Score accordingly.
    if drift is not None and drift < _RUGPULL_SEMANTIC_DRIFT_RATIO:
        result.risk_score = 0.80
    else:
        result.risk_score = 0.55
    result.severity = _severity_from_score(result.risk_score)
    result.details = {
        "kind": "rug_pull",
        "owasp_mcp": ["OWASP-MCP06"],  # rug-pull / consent fatigue
        "approved_hash": approved_hash,
        "current_hash": current_hash,
        "bow_cosine_vs_approved": round(drift, 4) if drift is not None else None,
    }
    return result


def _resolve_baseline(name: str, baseline: Dict):
    """Return ``(approved_description, approved_hash)`` for ``name`` or (None,None).

    Tolerates the three baseline shapes documented on :func:`detect_rug_pull`.
    """
    # Single tool-def baseline ({"name": ..., "description": ...}).
    if baseline.get("name") and baseline.get("name") == name:
        return _norm_text(baseline.get("description", "")), None

    entry = baseline.get(name)
    if entry is None:
        return None, None
    if isinstance(entry, dict):
        desc = entry.get("description")
        h = entry.get("hash")
        return (_norm_text(desc) if desc is not None else None,
                str(h) if h is not None else None)
    # Plain string value -> the approved description.
    return _norm_text(entry), None


# ---------------------------------------------------------------------------
# Typosquat — near-identical name vs a known/trusted name set
# ---------------------------------------------------------------------------


def detect_typosquat(
    tool_name: str, known_names: Optional[List[str]]
) -> McpSupplyChainResult:
    """Detect a tool name that near-collides with a known/trusted name.

    Flags when, for some known name (case-insensitive, not an exact match), BOTH
    ``Levenshtein <= 2`` AND ``SequenceMatcher.ratio >= 0.85`` hold and the names
    are long enough (``>= 6``) for the comparison to be meaningful.

    Parameters
    ----------
    tool_name : str
        The advertised tool name to test.
    known_names : list[str] or None
        Trusted/known tool names to compare against.  An exact (case-insensitive)
        match against any known name is the legitimate tool itself -> NO flag.

    Returns
    -------
    McpSupplyChainResult
        ``risk_score == 0.0`` when no near-collision is found.
    """
    name = _norm_text(tool_name).strip()
    result = McpSupplyChainResult(tool_name=name)
    if not name or not known_names:
        return result

    name_lower = name.lower()
    # Exact (case-insensitive) match against any known name => the real tool.
    if any(name_lower == _norm_text(k).strip().lower() for k in known_names):
        return result

    indicators: List[str] = []
    best = None
    for known in known_names:
        legit = _norm_text(known).strip()
        legit_lower = legit.lower()
        if not legit or legit_lower == name_lower:
            continue
        if max(len(name_lower), len(legit_lower)) < _TYPOSQUAT_MIN_LEN:
            continue
        distance = _levenshtein_distance(name_lower, legit_lower)
        ratio = SequenceMatcher(None, name_lower, legit_lower).ratio()
        # Both bars must hold: edit-distance catches single-char swaps; the ratio
        # rejects short-name coincidences that happen to be 2 edits apart.
        if distance <= _TYPOSQUAT_MAX_DISTANCE and ratio >= _TYPOSQUAT_MIN_RATIO:
            indicators.append(
                "typosquat: '{}' ~ '{}' (edit {} , {:.0%} similar)".format(
                    name, legit, distance, ratio
                )
            )
            if best is None or ratio > best:
                best = ratio

    if not indicators:
        return result

    result.risk_indicators = indicators
    result.technique_ids = ["T1.4"]  # Plugin-confusion
    # Closer collisions are higher risk; floor at 0.60, scale up with similarity.
    result.risk_score = min(0.60 + (best - _TYPOSQUAT_MIN_RATIO), 0.90)
    result.severity = _severity_from_score(result.risk_score)
    result.details = {
        "kind": "typosquat",
        "owasp_mcp": ["OWASP-MCP01"],
        "match_count": len(indicators),
        "best_ratio": round(best, 4),
    }
    return result


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Per-technique base risk for tool poisoning.  Instruction injection and exfil
# are the strongest signals; capability/schema claims are slightly softer.
_TECHNIQUE_BASE = {
    "T1.2": 0.80,
    "T2.3": 0.75,
    "T1.4": 0.65,
}


def _score_from_techniques(techniques, indicators) -> float:
    """Max-of-techniques + small multi-category bonus, capped at 1.0."""
    if not techniques:
        return 0.0
    base = max(_TECHNIQUE_BASE.get(t, 0.30) for t in techniques)
    multi_bonus = min((len(techniques) - 1) * 0.05, 0.15)
    count_bonus = min(max(len(indicators) - 1, 0), 3) * 0.03
    return min(base + multi_bonus + count_bonus, 1.0)


def _severity_from_score(score: float) -> str:
    if score >= 0.80:
        return "critical"
    if score >= 0.60:
        return "high"
    if score >= 0.40:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


# ---------------------------------------------------------------------------
# Batch API + weight
# ---------------------------------------------------------------------------


def scan_tool_supply_chain(
    tools: List[Dict],
    baseline: Optional[Dict] = None,
    known_names: Optional[List[str]] = None,
) -> List[McpSupplyChainResult]:
    """Run all three supply-chain checks over a batch of tool definitions.

    For each tool the poisoning / rug-pull / typosquat results are merged into a
    single :class:`McpSupplyChainResult` (max risk_score, union of techniques and
    indicators).

    Parameters
    ----------
    tools : list[dict]
        Tool definitions (``"name"`` + ``"description"``).
    baseline : dict or None
        Approved baseline for rug-pull detection (see :func:`detect_rug_pull`).
    known_names : list[str] or None
        Trusted names for typosquat detection.  When None, the manifest's own
        names are used so tools are cross-checked against each other.

    Returns
    -------
    list[McpSupplyChainResult]
        One merged result per input tool, in input order.
    """
    if not tools:
        return []

    # Default the typosquat name set to the manifest's own names (cross-check),
    # mirroring scan_tool_manifest (mcp_tool.py:632).
    if known_names is None:
        known_names = [
            _norm_text(t.get("name", "")).strip()
            for t in tools
            if t.get("name")
        ]

    results: List[McpSupplyChainResult] = []
    for tool_def in tools:
        name = _norm_text((tool_def or {}).get("name", "")).strip()
        merged = McpSupplyChainResult(tool_name=name)

        poison = detect_tool_poisoning(tool_def)
        # For typosquat, never compare a tool against its own name.
        peers = [k for k in known_names if _norm_text(k).strip().lower() != name.lower()]
        squat = detect_typosquat(name, peers)
        rug = (
            detect_rug_pull(tool_def, baseline)
            if baseline
            else McpSupplyChainResult(tool_name=name)
        )

        techniques: set = set()
        indicators: List[str] = []
        owasp: set = set()
        for sub in (poison, squat, rug):
            if sub.risk_score <= 0.0:
                continue
            techniques.update(sub.technique_ids)
            indicators.extend(sub.risk_indicators)
            owasp.update(sub.details.get("owasp_mcp", []))
            merged.risk_score = max(merged.risk_score, sub.risk_score)

        if techniques:
            merged.technique_ids = sorted(techniques)
            merged.risk_indicators = indicators
            merged.severity = _severity_from_score(merged.risk_score)
            merged.details = {
                "kind": "supply_chain",
                "owasp_mcp": sorted(owasp),
                "checks_fired": [
                    s
                    for s, sub in (
                        ("tool_poisoning", poison),
                        ("typosquat", squat),
                        ("rug_pull", rug),
                    )
                    if sub.risk_score > 0.0
                ],
            }
        results.append(merged)

    risky = sum(1 for r in results if r.risk_score > 0.0)
    if risky:
        logger.warning(
            "MCP supply-chain scan: %d/%d tools flagged", risky, len(results)
        )
    return results


def get_mcp_supply_chain_weight(result: McpSupplyChainResult) -> float:
    """Capped composite-weight contribution from an MCP supply-chain result.

    Early-returns 0.0 on the empty result so the ``_HAS_*`` import-flag +
    try/except wiring is no-op-safe.  Then scales and CAPS at 0.30, the uniform
    corroborating-T-detector cap (``min(score * 0.35, 0.30)`` — same as
    ``mcp_tool.py:672`` / ``inter_model.py:625`` / ``tool_abuse.py:500``).  A
    lone supply-chain hit is a soft signal that can never be decisive on its own.
    """
    if result is None or result.risk_score == 0.0:
        return 0.0
    return min(result.risk_score * 0.35, 0.30)
