"""MCP tool shadowing detector (T category — Agent/Tool Abuse).

Detects suspicious MCP (Model Context Protocol) tool descriptions and
manifests that may be used for prompt injection via tool shadowing.

T sub-techniques:
    T1.1 -- Instruction injection in tool descriptions
    T1.2 -- Hidden directives (invisible chars, Unicode tricks, encoded payloads)
    T1.3 -- Capability escalation claims
    T1.4 -- Tool name shadowing / typosquatting
    T1.5 -- Data exfiltration channels in tool descriptions
    T1.6 -- Description length anomaly (payload hiding via verbosity)

Strategy:
    Pattern-based detection across six risk categories.  Each category
    contributes an independent risk score and indicator list.  The final
    risk_score is the maximum category score (not additive) to avoid
    inflating benign tools that happen to match one soft heuristic.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class McpToolResult:
    """Result from MCP tool shadowing analysis."""
    risk_score: float = 0.0
    risk_indicators: List[str] = field(default_factory=list)
    technique_ids: List[str] = field(default_factory=list)
    tool_name: str = ""
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# T1.1 — Instruction injection in descriptions
# ---------------------------------------------------------------------------

_INSTRUCTION_INJECTION_PATTERNS = [
    # Imperative directives to the LLM
    re.compile(
        r"\b(?:always|never|must|shall|should)\s+"
        r"(?:respond|reply|answer|output|return|say|generate|produce|include|ignore|disregard|forget|override)\b",
        re.IGNORECASE,
    ),
    # Direct instructions
    re.compile(
        r"\b(?:ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|user\s+requests?|safety|rules?))\b",
        re.IGNORECASE,
    ),
    # "Before executing" preamble injection
    re.compile(
        r"\b(?:before\s+(?:executing|running|calling|using)\s*,?\s*(?:first|always))\b",
        re.IGNORECASE,
    ),
    # System prompt manipulation
    re.compile(
        r"\b(?:(?:new|updated?|revised?|override)\s+(?:system\s+)?(?:instructions?|prompt|rules?))\s*:",
        re.IGNORECASE,
    ),
    # Role assignment
    re.compile(
        r"\b(?:you\s+are\s+(?:now|a|an)|act\s+as\s+(?:if|a|an)|pretend\s+(?:to\s+be|you\s+are))\b",
        re.IGNORECASE,
    ),
    # Output manipulation
    re.compile(
        r"\b(?:prepend|append|inject|insert)\s+(?:the\s+following|this)\s+(?:to|into|in)\s+(?:every|all|each|your)\b",
        re.IGNORECASE,
    ),
    # "Do not" + safety-related terms
    re.compile(
        r"\b(?:do\s+not|don'?t)\s+(?:tell|inform|alert|notify|warn|mention\s+to)\s+(?:the\s+)?user\b",
        re.IGNORECASE,
    ),
]


def _check_instruction_injection(description: str) -> List[str]:
    """Return list of matched instruction injection indicators."""
    indicators = []
    for pattern in _INSTRUCTION_INJECTION_PATTERNS:
        match = pattern.search(description)
        if match:
            indicators.append("instruction_injection: " + match.group(0)[:60])
    return indicators


# ---------------------------------------------------------------------------
# T1.2 — Hidden directives (invisible characters, Unicode tricks)
# ---------------------------------------------------------------------------

# Unicode categories that are invisible or easily confused
_INVISIBLE_CATEGORIES = {"Cf", "Mn", "Zl", "Zp"}
# Specific invisible codepoints commonly used in attacks
_INVISIBLE_CODEPOINTS = {
    0x200B,  # Zero-width space
    0x200C,  # Zero-width non-joiner
    0x200D,  # Zero-width joiner
    0x200E,  # Left-to-right mark
    0x200F,  # Right-to-left mark
    0x202A,  # Left-to-right embedding
    0x202B,  # Right-to-left embedding
    0x202C,  # Pop directional formatting
    0x202D,  # Left-to-right override
    0x202E,  # Right-to-left override
    0x2060,  # Word joiner
    0x2061,  # Function application
    0x2062,  # Invisible times
    0x2063,  # Invisible separator
    0x2064,  # Invisible plus
    0xFEFF,  # Zero-width no-break space (BOM)
    0xFFF9,  # Interlinear annotation anchor
    0xFFFA,  # Interlinear annotation separator
    0xFFFB,  # Interlinear annotation terminator
}

# Base64-like patterns embedded in descriptions
_BASE64_PATTERN = re.compile(
    r"[A-Za-z0-9+/]{20,}={0,2}",
)

# Hex-encoded patterns
_HEX_PATTERN = re.compile(
    r"(?:\\x[0-9a-fA-F]{2}){4,}",
)

# Unicode escape sequences
_UNICODE_ESCAPE = re.compile(
    r"(?:\\u[0-9a-fA-F]{4}){3,}",
)


def _check_hidden_directives(description: str) -> List[str]:
    """Return list of hidden directive indicators."""
    indicators = []

    # Check for invisible Unicode characters
    invisible_count = 0
    invisible_chars = []
    for ch in description:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if cp in _INVISIBLE_CODEPOINTS or (cat in _INVISIBLE_CATEGORIES and cp > 127):
            invisible_count += 1
            name = unicodedata.name(ch, "U+{:04X}".format(cp))
            if name not in invisible_chars:
                invisible_chars.append(name)

    if invisible_count >= 3:
        indicators.append(
            "hidden_unicode: {} invisible chars ({})".format(
                invisible_count, ", ".join(invisible_chars[:3])
            )
        )

    # Check for encoded payloads within description
    if _BASE64_PATTERN.search(description):
        indicators.append("hidden_base64: base64-like payload in description")

    if _HEX_PATTERN.search(description):
        indicators.append("hidden_hex: hex-encoded content in description")

    if _UNICODE_ESCAPE.search(description):
        indicators.append("hidden_unicode_escape: unicode escape sequences in description")

    # Check for homoglyph characters (Cyrillic/Greek lookalikes in Latin context)
    homoglyph_count = 0
    has_latin = False
    has_lookalike = False
    for ch in description:
        cat_block = unicodedata.name(ch, "")
        if "LATIN" in cat_block:
            has_latin = True
        elif any(script in cat_block for script in ("CYRILLIC", "GREEK")):
            has_lookalike = True
            homoglyph_count += 1

    if has_latin and has_lookalike and homoglyph_count >= 2:
        indicators.append(
            "hidden_homoglyph: {} mixed-script lookalike chars".format(homoglyph_count)
        )

    return indicators


# ---------------------------------------------------------------------------
# T1.3 — Capability escalation
# ---------------------------------------------------------------------------

_CAPABILITY_ESCALATION_PATTERNS = [
    re.compile(
        r"\b(?:this\s+tool\s+)?(?:has\s+)?(?:full|complete|unrestricted|unlimited)\s+"
        r"(?:access|permission|control)\s+(?:to|over)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:can\s+)?(?:execute|run)\s+(?:any|all|arbitrary)\s+"
        r"(?:command|code|script|program|operation)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:access|read|write|modify|delete)\s+(?:any|all|every)\s+"
        r"(?:file|directory|folder|database|system|resource)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:root|admin|administrator|superuser|sudo)\s+"
        r"(?:access|privilege|permission|level)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:bypass(?:es)?|circumvent(?:s)?|skip(?:s)?|ignore(?:s)?|disable(?:s)?)\s+"
        r"(?:security|authentication|authorization|permission|access\s+control|sandbox)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no\s+(?:restrictions?|limitations?|boundaries|constraints?))\s+"
        r"(?:on|for|apply|exist)\b",
        re.IGNORECASE,
    ),
]


def _check_capability_escalation(description: str) -> List[str]:
    """Return list of capability escalation indicators."""
    indicators = []
    for pattern in _CAPABILITY_ESCALATION_PATTERNS:
        match = pattern.search(description)
        if match:
            indicators.append("capability_escalation: " + match.group(0)[:60])
    return indicators


# ---------------------------------------------------------------------------
# T1.4 — Tool name shadowing / typosquatting
# ---------------------------------------------------------------------------

# Well-known tool names that attackers commonly shadow
_COMMON_TOOL_NAMES = [
    "read_file", "write_file", "list_files", "delete_file",
    "execute_command", "run_command", "shell_exec",
    "search", "web_search", "browse",
    "send_email", "send_message",
    "get_weather", "calculator",
    "database_query", "sql_query",
    "http_request", "fetch_url",
    "create_file", "move_file", "copy_file",
    "read_resource", "list_resources",
]


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _check_tool_shadowing(tool_name: str,
                          known_tools: Optional[List[str]] = None) -> List[str]:
    """Return list of tool name shadowing indicators."""
    indicators = []
    name_lower = tool_name.lower().strip()

    # Build comparison list from known tools + common tools
    compare_names = list(_COMMON_TOOL_NAMES)
    if known_tools:
        for kt in known_tools:
            kt_lower = kt.lower().strip()
            if kt_lower not in compare_names:
                compare_names.append(kt_lower)

    for legit_name in compare_names:
        if name_lower == legit_name:
            continue  # Exact match is fine

        # Check edit distance for short names (typosquatting)
        distance = _levenshtein_distance(name_lower, legit_name)
        max_len = max(len(name_lower), len(legit_name))

        if max_len < 4:
            continue  # Too short for meaningful comparison

        # Typosquatting: edit distance 1-2 for names of reasonable length
        if distance <= 2 and max_len >= 6:
            indicators.append(
                "typosquat: '{}' similar to '{}' (edit distance {})".format(
                    tool_name, legit_name, distance
                )
            )

        # Prefix/suffix manipulation: e.g., "read_file_v2" vs "read_file"
        elif (name_lower.startswith(legit_name) or legit_name.startswith(name_lower)):
            suffix = name_lower[len(legit_name):] if name_lower.startswith(legit_name) else legit_name[len(name_lower):]
            if 0 < len(suffix) <= 4:
                indicators.append(
                    "name_variant: '{}' variant of '{}' (suffix: '{}')".format(
                        tool_name, legit_name, suffix
                    )
                )

        # High sequence similarity (catches reordering, insertions)
        similarity = SequenceMatcher(None, name_lower, legit_name).ratio()
        if similarity >= 0.85 and distance > 0:
            if not any(tool_name in ind for ind in indicators):
                indicators.append(
                    "name_similarity: '{}' ~ '{}' ({:.0%} similar)".format(
                        tool_name, legit_name, similarity
                    )
                )

    return indicators


# ---------------------------------------------------------------------------
# T1.5 — Exfiltration channels
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>]{5,}",
    re.IGNORECASE,
)

_EXFILTRATION_PATTERNS = [
    re.compile(
        r"\b(?:send|transmit|forward|post|upload|exfiltrate|leak|ship)\s+"
        r"(?:(?:the\s+)?(?:results?|data|output|response|content|information|context))\s+"
        r"(?:to|via|through|over)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:external|remote|third[- ]party|outside)\s+"
        r"(?:endpoint|server|api|service|url|webhook|destination)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:callback|webhook|ping|notify)\s*(?:url|endpoint|to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:phone\s+home|call\s+back|report\s+to|beacon)\b",
        re.IGNORECASE,
    ),
]

# Allowlisted URL domains that are commonly legitimate in tool descriptions
_SAFE_URL_DOMAINS = {
    "github.com", "docs.github.com", "api.github.com",
    "developer.mozilla.org", "www.w3.org",
    "tools.ietf.org", "www.ietf.org",
    "json-schema.org", "schema.org",
    "openai.com", "platform.openai.com",
    "anthropic.com", "docs.anthropic.com",
    "localhost", "127.0.0.1",
    "example.com", "example.org",
}


def _extract_domain(url: str) -> str:
    """Extract domain from a URL string."""
    try:
        # Remove protocol
        after_proto = url.split("://", 1)[-1]
        # Get host part (before path/query)
        host = after_proto.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        # Remove port
        host = host.rsplit(":", 1)[0]
        return host.lower()
    except Exception:
        return ""


def _check_exfiltration(description: str) -> List[str]:
    """Return list of exfiltration channel indicators."""
    indicators = []

    for pattern in _EXFILTRATION_PATTERNS:
        match = pattern.search(description)
        if match:
            indicators.append("exfiltration: " + match.group(0)[:60])

    # Check for suspicious URLs
    urls = _URL_PATTERN.findall(description)
    suspicious_urls = []
    for url in urls:
        domain = _extract_domain(url)
        if domain and domain not in _SAFE_URL_DOMAINS:
            suspicious_urls.append(url[:80])

    if suspicious_urls:
        indicators.append(
            "suspicious_url: {} non-standard URL(s): {}".format(
                len(suspicious_urls), ", ".join(suspicious_urls[:3])
            )
        )

    return indicators


# ---------------------------------------------------------------------------
# T1.6 — Description length anomaly
# ---------------------------------------------------------------------------

#: Threshold for suspiciously long descriptions (characters).
#: Typical MCP tool descriptions are 50-300 chars.  Injections often
#: require 500+ chars to embed hidden instructions.
_DESCRIPTION_LENGTH_THRESHOLD = 500

#: Hard limit where length alone is a strong signal.
_DESCRIPTION_LENGTH_HARD = 1500


def _check_description_length(description: str) -> List[str]:
    """Return list of description length anomaly indicators."""
    indicators = []
    length = len(description)

    if length >= _DESCRIPTION_LENGTH_HARD:
        indicators.append(
            "length_anomaly: description is {} chars (hard threshold: {})".format(
                length, _DESCRIPTION_LENGTH_HARD
            )
        )
    elif length >= _DESCRIPTION_LENGTH_THRESHOLD:
        indicators.append(
            "length_anomaly: description is {} chars (soft threshold: {})".format(
                length, _DESCRIPTION_LENGTH_THRESHOLD
            )
        )

    return indicators


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

_CATEGORY_WEIGHTS = {
    "instruction_injection": 0.85,
    "hidden_directives": 0.80,
    "capability_escalation": 0.70,
    "tool_shadowing": 0.65,
    "exfiltration": 0.75,
    "length_anomaly_hard": 0.50,
    "length_anomaly_soft": 0.25,
}


def _compute_risk_score(indicators_by_category: Dict[str, List[str]]) -> float:
    """Compute overall risk score from per-category indicators.

    Uses max-of-categories approach: the highest individual category
    score determines the floor, with a small additive bonus for each
    additional category that fires (up to 0.15 total bonus).
    """
    if not indicators_by_category:
        return 0.0

    category_scores = []
    for category, indicators in indicators_by_category.items():
        if not indicators:
            continue
        weight = _CATEGORY_WEIGHTS.get(category, 0.30)
        # Multiple indicators in same category add a small bonus
        count_bonus = min(len(indicators) - 1, 3) * 0.05
        category_scores.append(weight + count_bonus)

    if not category_scores:
        return 0.0

    # Max category score + small bonus for multi-category hits
    base = max(category_scores)
    multi_cat_bonus = min((len(category_scores) - 1) * 0.05, 0.15)
    return min(base + multi_cat_bonus, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_tool_shadowing(
    tool_name: str,
    tool_description: str,
    known_tools: Optional[List[str]] = None,
) -> McpToolResult:
    """Analyze a single MCP tool for shadowing and injection indicators.

    Parameters
    ----------
    tool_name : str
        The tool's declared name.
    tool_description : str
        The tool's description text.
    known_tools : list[str] or None
        List of legitimate tool names to check for typosquatting against.
        If None, only the built-in common tool list is used.

    Returns
    -------
    McpToolResult
        Analysis result with risk_score, risk_indicators, technique_ids,
        tool_name, and details.
    """
    if not tool_name and not tool_description:
        return McpToolResult(tool_name=tool_name or "")

    result = McpToolResult(tool_name=tool_name or "")
    description = tool_description or ""
    indicators_by_category: Dict[str, List[str]] = {}

    # T1.1: Instruction injection
    inj = _check_instruction_injection(description)
    if inj:
        indicators_by_category["instruction_injection"] = inj
        result.risk_indicators.extend(inj)
        if "T1.1" not in result.technique_ids:
            result.technique_ids.append("T1.1")

    # T1.2: Hidden directives
    hidden = _check_hidden_directives(description)
    if hidden:
        indicators_by_category["hidden_directives"] = hidden
        result.risk_indicators.extend(hidden)
        if "T1.2" not in result.technique_ids:
            result.technique_ids.append("T1.2")

    # T1.3: Capability escalation
    esc = _check_capability_escalation(description)
    if esc:
        indicators_by_category["capability_escalation"] = esc
        result.risk_indicators.extend(esc)
        if "T1.3" not in result.technique_ids:
            result.technique_ids.append("T1.3")

    # T1.4: Tool name shadowing
    shadow = _check_tool_shadowing(tool_name, known_tools)
    if shadow:
        indicators_by_category["tool_shadowing"] = shadow
        result.risk_indicators.extend(shadow)
        if "T1.4" not in result.technique_ids:
            result.technique_ids.append("T1.4")

    # T1.5: Exfiltration channels
    exfil = _check_exfiltration(description)
    if exfil:
        indicators_by_category["exfiltration"] = exfil
        result.risk_indicators.extend(exfil)
        if "T1.5" not in result.technique_ids:
            result.technique_ids.append("T1.5")

    # T1.6: Description length anomaly
    length = _check_description_length(description)
    if length:
        cat_key = "length_anomaly_hard" if "hard threshold" in (length[0] if length else "") else "length_anomaly_soft"
        indicators_by_category[cat_key] = length
        result.risk_indicators.extend(length)
        if "T1.6" not in result.technique_ids:
            result.technique_ids.append("T1.6")

    # Add parent technique ID if any sub-techniques matched
    if result.technique_ids:
        result.technique_ids.insert(0, "T1")

    # Compute risk score
    result.risk_score = _compute_risk_score(indicators_by_category)

    # Build details
    result.details = {
        "categories_triggered": list(indicators_by_category.keys()),
        "indicator_count": len(result.risk_indicators),
        "description_length": len(description),
    }

    if result.risk_score > 0:
        logger.debug(
            "MCP tool '%s': risk=%.2f, indicators=%d, techniques=%s",
            tool_name, result.risk_score, len(result.risk_indicators),
            result.technique_ids,
        )

    return result


def scan_tool_manifest(
    tools: List[Dict],
    known_tools: Optional[List[str]] = None,
) -> List[McpToolResult]:
    """Batch-scan an MCP tool manifest for shadowing indicators.

    Parameters
    ----------
    tools : list[dict]
        List of tool definitions.  Each dict should have at minimum
        ``"name"`` and ``"description"`` keys.
    known_tools : list[str] or None
        List of legitimate tool names.  If None, tool names are extracted
        from the manifest itself to enable cross-checking.

    Returns
    -------
    list[McpToolResult]
        One result per tool, in the same order as the input list.
    """
    if not tools:
        return []

    # If no known_tools provided, use names from the manifest itself
    # so that tools can be compared against each other
    if known_tools is None:
        known_tools = [
            t.get("name", "") for t in tools if t.get("name")
        ]

    results = []
    for tool_def in tools:
        name = tool_def.get("name", "")
        description = tool_def.get("description", "")
        result = detect_tool_shadowing(name, description, known_tools)
        results.append(result)

    # Log summary
    risky_count = sum(1 for r in results if r.risk_score > 0.0)
    if risky_count > 0:
        logger.warning(
            "MCP manifest scan: %d/%d tools flagged with risk indicators",
            risky_count, len(results),
        )

    return results


def scan_tool_result(result_text: str, tool_name: Optional[str] = None):
    """Scan the *content returned by a tool / function call* for injection.

    The existing :func:`scan_tool_manifest` / :func:`detect_tool_shadowing`
    only inspect a tool's declared *name* and *description*.  They never look
    at what a tool *returns* — yet tool/function RESULT text is an indirect
    prompt-injection channel: an attacker-controlled API, file, or MCP
    resource can return a worm or jailbreak payload that the model then
    ingests.  This helper closes that ingestion-channel gap by routing the
    result content through the full :func:`na0s.predict.scan` pipeline (the
    same rule stack that scores user input, including the worm / IM1.6
    self-replication signal).

    Parameters
    ----------
    result_text : str
        The text content returned by the tool / function call.
    tool_name : str or None
        Optional name of the tool that produced the result; recorded on the
        returned result for attribution (does not affect scoring).

    Returns
    -------
    na0s.scan_result.ScanResult
        The pipeline verdict.  ``rule_hits`` / ``technique_tags`` /
        ``is_malicious`` / ``risk_score`` are populated exactly as for a
        direct ``scan()`` call.  When *tool_name* is given it is stored in
        ``result.anomaly_flags`` as ``"tool_result:<name>"`` for provenance.

    Notes
    -----
    Lazy-imports :func:`na0s.predict.scan` inside the function to avoid an
    import cycle, and is fail-safe: any pipeline error degrades to a benign
    result rather than crashing the host.

    EMAIL/WEB ingestion is the integrator's responsibility — Na0S holds no
    email or web content; inbound message bodies and fetched pages must be
    passed through ``scan()`` / these helpers by the host.
    """
    # Lazy import to avoid an import cycle: predict imports detectors.
    from na0s.scan_result import ScanResult

    if not result_text or not str(result_text).strip():
        sr = ScanResult(label="safe")
        if tool_name:
            sr.anomaly_flags.append("tool_result:{}".format(tool_name))
        return sr

    try:
        from na0s.predict import scan as na0s_scan

        sr = na0s_scan(str(result_text))
    except Exception as exc:
        logger.debug(
            "scan_tool_result: scan() failed for tool %r: %s",
            tool_name, exc,
        )
        sr = ScanResult(label="safe")

    if tool_name:
        flag = "tool_result:{}".format(tool_name)
        if flag not in sr.anomaly_flags:
            sr.anomaly_flags.append(flag)

    if sr.is_malicious:
        logger.warning(
            "scan_tool_result: tool %r returned flagged content "
            "(risk=%.2f, hits=%s)",
            tool_name, sr.risk_score, sr.rule_hits,
        )

    return sr


def get_mcp_tool_weight(result: McpToolResult) -> float:
    """Compute rule weight contribution from MCP tool detection.

    Parameters
    ----------
    result : McpToolResult
        Result from detect_tool_shadowing().

    Returns
    -------
    float
        Weight to add to the composite score.
    """
    if result is None or result.risk_score == 0.0:
        return 0.0

    # Scale weight by risk score, capped at 0.30
    return min(result.risk_score * 0.35, 0.30)
