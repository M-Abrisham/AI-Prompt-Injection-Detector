"""Payload assembly detector for fragmented delivery attacks (D7 category).

Detects prompt injection attacks delivered via fragmented payloads where
each fragment appears harmless alone but the assembled result is malicious.

D7 sub-techniques:
    D7.1 -- Token-level splitting with assembly instructions
    D7.2 -- Multi-turn splitting (stub — requires ConversationSecurityMonitor)
    D7.3 -- Code-block weaponization (exec/eval + attack strings)
    D7.4 -- Comment/metadata payload hiding (HTML/JSON/YAML/XML/SQL)
    D7.5 -- Cross-encoding fragment assembly

Strategy:
    1. Detect quoted/backticked fragments + assembly keywords (D7.1)
    2. Detect execution functions + attack strings in code blocks (D7.3)
    3. Extract and scan comment/metadata content (D7.4)
    4. Concatenate decoded views and re-scan (D7.5)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FragmentResult:
    """Result from fragment assembly analysis."""
    fragment_type: str = ""          # token_split/code_weapon/comment_payload/cross_encoding/multiturn
    fragments_found: List[str] = field(default_factory=list)
    assembled_text: str = ""
    assembled_is_malicious: bool = False
    technique_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    matched_patterns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# D7.1 — Token-split assembly detection
# ---------------------------------------------------------------------------

_ASSEMBLY_KEYWORDS = re.compile(
    r"\b(?:combine|concatenate|put\s+together|merge|join|assemble|"
    r"string\s+together|connect\s+(?:the\s+)?(?:above|these|them|pieces|parts|fragments)|"
    r"piece\s+(?:them\s+)?together|"
    r"now\s+(?:put|combine|merge|join|assemble)\s+(?:them|these|it|the\s+(?:above|pieces|parts)))\b",
    re.IGNORECASE,
)

# Match quoted or backticked fragments: "word", 'word', `word`
_FRAGMENT_PATTERN = re.compile(
    r'(?:"([^"]{1,60})"|\'([^\']{1,60})\'|`([^`]{1,60})`)'
)

# Attack indicators in assembled text — lightweight check
_ATTACK_INDICATORS = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous\s+)?instructions|"
    r"reveal\s+(?:your\s+)?(?:system\s+)?prompt|"
    r"forget\s+(?:all\s+)?(?:your\s+)?(?:instructions|rules)|"
    r"you\s+are\s+now\s+(?:DAN|free|unrestricted)|"
    r"new\s+instructions?\s*:|"
    r"override\s+(?:all\s+)?(?:safety|security)|"
    r"disable\s+(?:all\s+)?(?:safety|filters|restrictions)|"
    r"bypass\s+(?:all\s+)?(?:safety|security|restrictions))",
    re.IGNORECASE,
)


def detect_token_split(text: str) -> Optional[FragmentResult]:
    """Detect D7.1 token-level splitting with assembly keywords.

    Looks for 3+ quoted/backticked fragments combined with an assembly
    instruction keyword (combine, concatenate, merge, etc.).
    """
    if not text or len(text) < 20:
        return None

    # Must have an assembly keyword
    assembly_match = _ASSEMBLY_KEYWORDS.search(text)
    if not assembly_match:
        return None

    # Find all quoted/backticked fragments
    fragments = []
    for match in _FRAGMENT_PATTERN.finditer(text):
        frag = match.group(1) or match.group(2) or match.group(3)
        if frag and frag.strip():
            fragments.append(frag.strip())

    # Need at least 3 fragments for token-split attack
    if len(fragments) < 3:
        return None

    # Assemble and check for attack indicators
    assembled = " ".join(fragments)
    if _ATTACK_INDICATORS.search(assembled):
        return FragmentResult(
            fragment_type="token_split",
            fragments_found=fragments,
            assembled_text=assembled,
            assembled_is_malicious=True,
            technique_ids=["D7", "D7.1"],
            confidence=0.85,
            matched_patterns=[
                "assembly_keyword: " + assembly_match.group(0)[:40],
                "assembled_attack: " + assembled[:60],
            ],
        )

    return None


# ---------------------------------------------------------------------------
# D7.3 — Code-block weaponization detection
# ---------------------------------------------------------------------------

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:\w+)?\s*\n?([\s\S]*?)```",
    re.MULTILINE,
)

# Execution functions that could run attack strings
_EXEC_FUNCTIONS = re.compile(
    r"\b(?:exec|eval|system|os\.(?:popen|system)|subprocess\.(?:run|call|Popen)|"
    r"__import__|compile|execfile|input\s*\(|"
    r"send_message|execute|run_command|os\.exec)\s*\(",
    re.IGNORECASE,
)

# Variable assignment + later use pattern
_VAR_ASSIGNMENT = re.compile(
    r"(\w+)\s*=\s*['\"]([^'\"]{10,200})['\"]",
)


def detect_code_weaponization(text: str) -> Optional[FragmentResult]:
    """Detect D7.3 code blocks containing weaponized attack strings.

    Distinguishes between educational code examples (string literals only)
    and weaponized code (attack string + execution function).
    """
    if not text:
        return None

    code_blocks = _CODE_BLOCK_PATTERN.findall(text)
    if not code_blocks:
        return None

    for block in code_blocks:
        block = block.strip()
        if not block:
            continue

        # Check for execution function + attack indicator
        has_exec = _EXEC_FUNCTIONS.search(block)
        has_attack = _ATTACK_INDICATORS.search(block)

        if has_exec and has_attack:
            return FragmentResult(
                fragment_type="code_weapon",
                fragments_found=[block[:200]],
                assembled_text=block[:200],
                assembled_is_malicious=True,
                technique_ids=["D7", "D7.3"],
                confidence=0.80,
                matched_patterns=[
                    "exec_func: " + has_exec.group(0)[:40],
                    "attack_in_code: " + has_attack.group(0)[:40],
                ],
            )

        # Check for variable assignment with attack string + use of that variable
        for var_match in _VAR_ASSIGNMENT.finditer(block):
            var_name = var_match.group(1)
            var_value = var_match.group(2)
            if _ATTACK_INDICATORS.search(var_value):
                # Check if the variable is used in an execution context
                var_use = re.search(
                    r"\b(?:exec|eval|system|os\.\w+|subprocess\.\w+|execute|run)\s*\(\s*" +
                    re.escape(var_name),
                    block,
                )
                if var_use:
                    return FragmentResult(
                        fragment_type="code_weapon",
                        fragments_found=[block[:200]],
                        assembled_text=var_value,
                        assembled_is_malicious=True,
                        technique_ids=["D7", "D7.3"],
                        confidence=0.80,
                        matched_patterns=[
                            "var_assign: {}='{}'".format(var_name, var_value[:40]),
                            "var_exec: " + var_use.group(0)[:40],
                        ],
                    )

    return None


# ---------------------------------------------------------------------------
# D7.4 — Comment/metadata payload detection
# ---------------------------------------------------------------------------

_HTML_COMMENT = re.compile(r"<!--\s*([\s\S]*?)\s*-->")
_CSS_COMMENT = re.compile(r"/\*\s*([\s\S]*?)\s*\*/")
_YAML_COMMENT = re.compile(r"^\s*#\s*(.+)$", re.MULTILINE)
_XML_CDATA = re.compile(r"<!\[CDATA\[\s*([\s\S]*?)\s*\]\]>")
_SQL_COMMENT = re.compile(r"--\s*(.+)$", re.MULTILINE)

# JSON metadata keys that often hide payloads
_JSON_METADATA = re.compile(
    r'"(?:_comment|__|//|_note|_description|_metadata|_hidden|_debug|_internal)"\s*:\s*"([^"]{5,500})"',
    re.IGNORECASE,
)


def detect_comment_payload(text: str) -> Optional[FragmentResult]:
    """Detect D7.4 attack payloads hidden in comments/metadata.

    Extracts content from HTML, CSS, JSON metadata, YAML, XML CDATA,
    and SQL comments, then checks for attack indicators.
    """
    if not text:
        return None

    comment_sources = []

    for match in _HTML_COMMENT.finditer(text):
        comment_sources.append(("html_comment", match.group(1)))
    for match in _CSS_COMMENT.finditer(text):
        comment_sources.append(("css_comment", match.group(1)))
    for match in _JSON_METADATA.finditer(text):
        comment_sources.append(("json_metadata", match.group(1)))
    for match in _XML_CDATA.finditer(text):
        comment_sources.append(("xml_cdata", match.group(1)))

    # YAML and SQL comments are very common in normal text, so only
    # check them if they contain attack indicators (not just any comment)
    for match in _YAML_COMMENT.finditer(text):
        content = match.group(1).strip()
        if _ATTACK_INDICATORS.search(content):
            comment_sources.append(("yaml_comment", content))
    for match in _SQL_COMMENT.finditer(text):
        content = match.group(1).strip()
        if _ATTACK_INDICATORS.search(content):
            comment_sources.append(("sql_comment", content))

    for source_type, content in comment_sources:
        content = content.strip()
        if len(content) < 10:
            continue
        if _ATTACK_INDICATORS.search(content):
            return FragmentResult(
                fragment_type="comment_payload",
                fragments_found=[content[:200]],
                assembled_text=content[:200],
                assembled_is_malicious=True,
                technique_ids=["D7", "D7.4"],
                confidence=0.80,
                matched_patterns=[
                    "source: " + source_type,
                    "payload: " + content[:60],
                ],
            )

    return None


# ---------------------------------------------------------------------------
# D7.5 — Cross-encoding fragment assembly
# ---------------------------------------------------------------------------

def detect_cross_encoding(decoded_views: list) -> Optional[FragmentResult]:
    """Detect D7.5 attacks split across multiple encoding types.

    After obfuscation_scan() decodes all encoded segments, this function
    concatenates all decoded views and checks if the assembled text
    contains attack indicators that individual views don't.

    Parameters
    ----------
    decoded_views : list[str]
        Decoded text segments from obfuscation_scan().
    """
    if not decoded_views or len(decoded_views) < 2:
        return None

    # Check if individual views are benign
    individual_malicious = any(
        _ATTACK_INDICATORS.search(view) for view in decoded_views
    )

    # Concatenate and check assembled text
    assembled = " ".join(decoded_views)
    assembled_malicious = _ATTACK_INDICATORS.search(assembled)

    # Only flag if concatenation reveals attack that individual views don't
    if assembled_malicious and not individual_malicious:
        return FragmentResult(
            fragment_type="cross_encoding",
            fragments_found=[v[:100] for v in decoded_views],
            assembled_text=assembled[:200],
            assembled_is_malicious=True,
            technique_ids=["D7", "D7.5"],
            confidence=0.85,
            matched_patterns=[
                "fragment_count: {}".format(len(decoded_views)),
                "assembled_attack: " + assembled_malicious.group(0)[:60],
            ],
        )

    return None


# ---------------------------------------------------------------------------
# D7.2 — Multi-turn stub (requires ConversationSecurityMonitor)
# ---------------------------------------------------------------------------

def detect_multiturn_assembly(text: str,
                              session_history: Optional[list] = None) -> None:
    """Stub for D7.2 multi-turn fragment assembly detection.

    Returns None when no session history is available. Architecturally
    ready for when ConversationSecurityMonitor is built.

    Parameters
    ----------
    text : str
        Current turn text.
    session_history : list[str] or None
        Previous turns in the conversation, if available.
    """
    if session_history is None:
        return None
    # FUTURE: concatenate recent turns, re-analyze combined text.
    # Detect when fragments across turns assemble into attack.
    # Depends on ConversationSecurityMonitor (not yet built).
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_fragmented_payload(text: str,
                              decoded_views: Optional[list] = None) -> Optional[FragmentResult]:
    """Run all D7 fragment assembly detectors on the input.

    Parameters
    ----------
    text : str
        The input text to analyze.
    decoded_views : list[str] or None
        Decoded text segments from obfuscation_scan() for D7.5 detection.

    Returns
    -------
    FragmentResult or None
        The highest-confidence fragment detection result, or None.
    """
    if not text:
        return None

    results = []

    # D7.1: Token-split assembly
    r = detect_token_split(text)
    if r:
        results.append(r)

    # D7.3: Code-block weaponization
    r = detect_code_weaponization(text)
    if r:
        results.append(r)

    # D7.4: Comment/metadata payloads
    r = detect_comment_payload(text)
    if r:
        results.append(r)

    # D7.5: Cross-encoding fragment assembly
    if decoded_views:
        r = detect_cross_encoding(decoded_views)
        if r:
            results.append(r)

    if not results:
        return None

    # Return highest confidence result
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results[0]


def get_fragment_weight(result: Optional[FragmentResult]) -> float:
    """Compute the rule weight contribution from fragment detection.

    Parameters
    ----------
    result : FragmentResult or None
        Result from detect_fragmented_payload().

    Returns
    -------
    float
        Weight to add to the composite score.
    """
    if result is None or not result.assembled_is_malicious:
        return 0.0

    _TYPE_WEIGHTS = {
        "token_split": 0.25,
        "code_weapon": 0.20,
        "comment_payload": 0.25,
        "cross_encoding": 0.30,
    }
    return _TYPE_WEIGHTS.get(result.fragment_type, 0.15)
