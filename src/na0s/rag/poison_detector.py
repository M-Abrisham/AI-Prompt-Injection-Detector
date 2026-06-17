"""RAG Poisoning Detection (I1.x / IM.x categories).

Detects poisoned RAG (Retrieval-Augmented Generation) context where
malicious content is injected into retrieved documents so the LLM
follows attacker instructions instead of the user's actual query.

Six categories of RAG poisoning:
    1. Instruction injection in context (I1.1): retrieved docs containing
       imperative instructions ("ignore the above", "instead do", etc.)
    2. Context boundary confusion (I1.2): fake document boundaries,
       injected separators ("---END OF CONTEXT---", "[Document 2]:", etc.)
    3. Authority spoofing in docs (I1.3): retrieved content claiming to
       be system messages or official instructions
    4. Relevance manipulation (I1.4): keyword stuffing + injection payload
    5. Consistency anomalies (I1.5): contradictory instructions vs query
    6. Hidden instructions in structured data (IM.1): JSON/XML/markdown
       in retrieved docs containing hidden directives

Strategy:
    Pattern-based detection with 6 pattern groups.  Each group targets
    a specific RAG poisoning technique.  An optional ``query`` parameter
    enables relevance-alignment checking between the retrieved context
    and the user's apparent intent.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RagPoisonResult:
    """Result from RAG poisoning analysis."""
    poison_score: float = 0.0
    poison_indicators: List[str] = field(default_factory=list)
    technique_ids: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RAG poisoning detection patterns
# ---------------------------------------------------------------------------

_RAG_PATTERNS: list = []


def _add(category: str, name: str, severity: str, technique_ids: list, pattern: str):
    """Register a RAG poisoning detection pattern."""
    _RAG_PATTERNS.append(
        (category, name, severity, technique_ids, re.compile(pattern, re.IGNORECASE))
    )


# ---- Category 1: Instruction Injection in Context (I1.1) ----
# Retrieved docs that contain imperative instructions aimed at the LLM.

_add("instruction_injection", "ignore_above_context", "high", ["I1", "I1.1"],
     r"\b(?:ignore|disregard|forget|dismiss)\s+"
     r"(?:the\s+)?(?:above|previous|preceding|prior|earlier)\s+"
     r"(?:context|documents?|information|text|instructions?|results?|passages?)")

_add("instruction_injection", "instead_do_directive", "high", ["I1", "I1.1"],
     r"\b(?:instead|rather)\s*,?\s*(?:you\s+)?(?:should|must|need\s+to|have\s+to|will|shall)\s+"
     r"(?:do|say|respond|output|generate|provide|return|answer)")

_add("instruction_injection", "new_instructions_in_context", "high", ["I1", "I1.1"],
     r"\b(?:new|updated|revised|real|actual|correct)\s+"
     r"(?:instructions?|directives?|guidelines?|rules?|task)\s*:")

_add("instruction_injection", "override_context", "high", ["I1", "I1.1"],
     r"\b(?:override|replace|supersede|cancel|nullify)\s+"
     r"(?:the\s+)?(?:above|previous|prior|existing|current|original)\s+"
     r"(?:context|instructions?|information|directives?|guidelines?)")

_add("instruction_injection", "do_not_answer_query", "high", ["I1", "I1.1"],
     r"\b(?:do\s+not|don'?t|never)\s+"
     r"(?:answer|respond\s+to|address|follow)\s+"
     r"(?:the\s+)?(?:user'?s?\s+)?(?:query|question|request|prompt|original)")

_add("instruction_injection", "important_instruction_in_doc", "high", ["I1", "I1.1"],
     r"(?:IMPORTANT|CRITICAL|NOTE|ATTENTION|WARNING|URGENT)\s*:\s*"
     r"(?:ignore|disregard|forget|override|instead|you\s+(?:must|should|need))")


# ---- Category 2: Context Boundary Confusion (I1.2) ----
# Fake document boundaries injected to confuse the LLM about context structure.

_add("boundary_confusion", "fake_end_of_context", "high", ["I1", "I1.2"],
     r"(?:-{3,10}|={3,10})\s*(?:END\s+OF\s+(?:CONTEXT|DOCUMENT|RETRIEVED|RESULTS?|SEARCH))"
     r"(?:\s*(?:-{3,10}|={3,10}))?")

_add("boundary_confusion", "fake_document_separator", "medium", ["I1", "I1.2"],
     r"(?:^|\n)\s*\[(?:Document|Doc|Result|Source|Chunk|Passage|Context)\s*\d+\]\s*:")

_add("boundary_confusion", "fake_retrieved_result_header", "medium", ["I1", "I1.2"],
     r"(?:^|\n)\s*(?:Retrieved\s+(?:result|document|context|passage|chunk)|"
     r"Search\s+result|Context\s+(?:document|chunk|passage))\s*(?:#?\d+)?\s*:")

_add("boundary_confusion", "fake_system_boundary", "high", ["I1", "I1.2"],
     r"(?:^|\n)\s*(?:-{3,10}|={3,10})\s*(?:SYSTEM|INSTRUCTIONS?|ADMIN|DEVELOPER)\s+"
     r"(?:MESSAGE|PROMPT|NOTE|SECTION|CONTEXT)\s*(?:-{3,10}|={3,10})?\s*(?:$|\n)")

_add("boundary_confusion", "end_of_documents_marker", "high", ["I1", "I1.2"],
     r"(?:^|\n)\s*(?:</?(?:end_?of_?(?:context|documents?|retrieval|results?)|"
     r"context_?end|doc_?end)>)")

_add("boundary_confusion", "injected_role_separator", "high", ["I1", "I1.2"],
     r"(?:^|\n)\s*(?:###\s*)?(?:System|Assistant|User|Human|AI)\s*(?:message|response|turn)?\s*:\s*(?:\n|$)")


# ---- Category 3: Authority Spoofing in Docs (I1.3) ----
# Retrieved content impersonating system messages or official instructions.

_add("authority_spoofing", "fake_system_message", "high", ["I1", "I1.3"],
     r"\b(?:this\s+is\s+(?:a\s+)?(?:system|official|admin(?:istrator)?|developer)\s+"
     r"(?:message|instruction|directive|notice|update|announcement))")

_add("authority_spoofing", "admin_override_claim", "high", ["I1", "I1.3"],
     r"\b(?:(?:admin(?:istrator)?|system|developer|operator|platform)\s+"
     r"(?:override|update|directive|instruction|command|notice)\s*:)")

_add("authority_spoofing", "official_policy_claim", "high", ["I1", "I1.3"],
     r"\b(?:(?:per|according\s+to|as\s+per)\s+"
     r"(?:official|updated|new|revised|current)\s+"
     r"(?:policy|guidelines?|instructions?|protocol|rules?))\s*,?\s*"
     r"(?:you\s+(?:must|should|shall|need|are\s+required))")

_add("authority_spoofing", "security_update_claim", "high", ["I1", "I1.3"],
     r"\b(?:(?:security|safety|compliance|policy)\s+(?:update|patch|fix|notice)\s*:\s*"
     r"(?:you\s+(?:must|should|need|are\s+required)|"
     r"ignore|disregard|override|change|modify))")

_add("authority_spoofing", "trusted_source_impersonation", "high", ["I1", "I1.3"],
     r"\b(?:(?:message\s+from|notice\s+from|update\s+from|instructions?\s+from)\s+"
     r"(?:the\s+)?(?:system|admin(?:istrator)?|developer|operator|platform|API|backend))")


# ---- Category 4: Relevance Manipulation (I1.4) ----
# Content crafted to be retrieved via keyword stuffing with injection payload.

# NOTE: possessive quantifiers (\w++, \s*+) make the {5,15} word-run atomic so
# it cannot backtrack into exponential partitions on word/whitespace-heavy input
# (ReDoS — e.g. a whitespace-stego payload reaching this detector via scan()).
# Equivalent matches, linear time. Requires Python 3.11+ (repo runs 3.14).
_add("relevance_manipulation", "keyword_stuffing_with_injection", "high", ["I1", "I1.4"],
     r"(?:\b\w++\b\s*+,?\s*+){5,15}\s*+(?:\.\s*+|\n\s*+)"
     r"(?:ignore|disregard|forget|override|instead|new\s+instructions?)\b")

_add("relevance_manipulation", "seo_style_keyword_block", "medium", ["I1", "I1.4"],
     r"(?:^|\n)\s*(?:(?:keywords?|tags?|topics?|related)\s*:\s*)"
     r"(?:\w+(?:\s*[,;|]\s*\w+){4,})")

_add("relevance_manipulation", "repeated_terms_with_payload", "medium", ["I1", "I1.4"],
     r"(\b\w{3,}\b)(?:\s+\1){3,}")


# ---- Category 5: Consistency Anomalies (I1.5) ----
# Contradictory instructions within retrieved context.

_add("consistency_anomaly", "contradictory_imperative", "high", ["I1", "I1.5"],
     r"\b(?:always|must|should)\s+(?:follow|obey|comply|adhere|respect)\b"
     r"[\s\S]{1,200}?"
     r"\b(?:ignore|disregard|forget|bypass|override|skip)\s+"
     r"(?:all|the|your|any|every|those|these)\b")

_add("consistency_anomaly", "context_vs_instruction_conflict", "medium", ["I1", "I1.5"],
     r"\b(?:the\s+(?:documents?|context|information|data)\s+"
     r"(?:says?|states?|mentions?|indicates?|shows?)\b)"
     r"[\s\S]{1,200}?"
     r"\b(?:but\s+)?(?:ignore|disregard|don'?t\s+(?:use|follow|trust)|"
     r"instead|actually|however)\b")

_add("consistency_anomaly", "answer_contradiction", "medium", ["I1", "I1.5"],
     r"\b(?:the\s+(?:correct|right|true|real|actual)\s+answer\s+is)\b"
     r"[\s\S]{1,100}?"
     r"\b(?:not\s+what\s+(?:the|these|those)\s+(?:documents?|sources?|results?)|"
     r"regardless\s+of\s+(?:the|what)\s+(?:context|documents?|sources?|results?))\b")


# ---- Category 6: Hidden Instructions in Structured Data (IM.1) ----
# JSON/XML/markdown in retrieved docs containing hidden directives.

_add("hidden_structured", "json_hidden_instruction", "high", ["I1", "IM.1"],
     r'["\'](?:instruction|directive|command|task|system_message|system_prompt|'
     r'override|admin_note|hidden_message|internal_note)["\']'
     r'\s*:\s*["\']')

_add("hidden_structured", "xml_hidden_instruction", "high", ["I1", "IM.1"],
     r"<(?:instruction|directive|command|system[_-]?message|system[_-]?prompt|"
     r"override|admin[_-]?note|hidden|internal[_-]?note)>")

_add("hidden_structured", "markdown_hidden_comment", "medium", ["I1", "IM.1"],
     r"<!--\s*(?:instruction|directive|command|system|override|ignore|"
     r"admin|hidden|internal|secret)\b")

_add("hidden_structured", "html_hidden_directive", "high", ["I1", "IM.1"],
     r'<\w+\s+(?:style\s*=\s*["\']display\s*:\s*none|'
     r'hidden|aria-hidden\s*=\s*["\']true)["\']?\s*>'
     r'\s*(?:ignore|disregard|override|new\s+instructions?|instead)\b')

_add("hidden_structured", "data_attribute_instruction", "medium", ["I1", "IM.1"],
     r'data-(?:instruction|directive|command|prompt|message|note)\s*=\s*["\']')

_add("hidden_structured", "yaml_hidden_field", "medium", ["I1", "IM.1"],
     r"(?:^|\n)\s*(?:instruction|directive|command|system_message|"
     r"system_prompt|override|admin_note|hidden_message|internal_note)\s*:")


# ---------------------------------------------------------------------------
# Query-context alignment scoring
# ---------------------------------------------------------------------------

def _compute_query_alignment(text: str, query: str) -> float:
    """Compute how well text aligns with the query.

    Returns a misalignment score in [0.0, 1.0] where higher values
    indicate the retrieved context is suspiciously unrelated to the query
    while containing injection-like content.

    A high misalignment score is a secondary signal — it boosts the
    poison score but does not trigger detection on its own.
    """
    if not query or not query.strip() or not text or not text.strip():
        return 0.0

    # Extract query terms (lowercase, deduplicated, skip short words)
    query_terms = {
        w.lower() for w in re.findall(r"\b\w{3,}\b", query)
    }
    if not query_terms:
        return 0.0

    text_lower = text.lower()
    # Count how many query terms appear in the text
    matching_terms = sum(1 for t in query_terms if t in text_lower)
    term_overlap = matching_terms / len(query_terms)

    # Check for injection-like content in the text
    _injection_re = re.compile(
        r"\b(?:ignore|disregard|override|forget|bypass|instead|"
        r"new\s+instructions?|system\s+prompt|you\s+(?:must|should|shall))\b",
        re.IGNORECASE,
    )
    injection_count = len(_injection_re.findall(text))

    # Low overlap + injection content = high misalignment
    if term_overlap < 0.2 and injection_count >= 2:
        return 0.8
    if term_overlap < 0.3 and injection_count >= 1:
        return 0.5
    if term_overlap < 0.4 and injection_count >= 2:
        return 0.4

    return 0.0


# ---------------------------------------------------------------------------
# Severity weights
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHTS = {
    "critical": 0.40,
    "high": 0.25,
    "medium": 0.10,
    "low": 0.05,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_rag_poisoning(text: str,
                         query: Optional[str] = None) -> RagPoisonResult:
    """Detect RAG poisoning indicators in text.

    Parameters
    ----------
    text : str
        The input text to analyze (typically RAG-retrieved context
        concatenated with the user query).
    query : str or None
        The original user query.  When provided, enables
        query-context alignment checking for relevance manipulation
        detection.

    Returns
    -------
    RagPoisonResult
        Detection result with poison_score, indicators, technique_ids,
        and details.
    """
    if not text or not text.strip():
        return RagPoisonResult()

    indicators: List[str] = []
    technique_ids_seen: set = set()
    categories_seen: set = set()
    total_weight = 0.0
    details: dict = {
        "matched_patterns": [],
        "categories": [],
    }

    for category, name, severity, technique_ids, pattern in _RAG_PATTERNS:
        match = pattern.search(text)
        if match:
            indicator = "{}:{}".format(category, name)
            indicators.append(indicator)
            categories_seen.add(category)
            total_weight += _SEVERITY_WEIGHTS.get(severity, 0.10)

            for tid in technique_ids:
                technique_ids_seen.add(tid)

            details["matched_patterns"].append({
                "category": category,
                "name": name,
                "severity": severity,
                "matched_text": match.group(0)[:100],
            })

    # Query-context alignment check
    alignment_boost = 0.0
    if query:
        alignment_score = _compute_query_alignment(text, query)
        if alignment_score > 0:
            alignment_boost = alignment_score * 0.15
            details["query_misalignment"] = alignment_score
            if alignment_score >= 0.5:
                indicators.append("relevance:query_context_misalignment")
                technique_ids_seen.add("I1.4")

    # Multi-category boost: poisoned documents typically combine techniques.
    # Finding indicators from 2+ categories is strong evidence of RAG poisoning.
    category_boost = 0.0
    if len(categories_seen) >= 3:
        category_boost = 0.15
    elif len(categories_seen) >= 2:
        category_boost = 0.08

    # Compute final score
    raw_score = total_weight + alignment_boost + category_boost
    poison_score = min(raw_score, 1.0)

    details["categories"] = sorted(categories_seen)
    details["category_count"] = len(categories_seen)

    return RagPoisonResult(
        poison_score=poison_score,
        poison_indicators=indicators,
        technique_ids=sorted(technique_ids_seen),
        details=details,
    )


def get_rag_poison_weight(result: RagPoisonResult) -> float:
    """Compute the rule weight contribution from RAG poisoning detection.

    Parameters
    ----------
    result : RagPoisonResult
        Result from detect_rag_poisoning().

    Returns
    -------
    float
        Weight to add to the composite score (capped at 0.12).
    """
    if not result.poison_indicators:
        return 0.0

    # Scale the poison score into a weight contribution.
    # Cap at 0.12 to avoid over-influencing the composite.
    return min(result.poison_score * 0.20, 0.12)
