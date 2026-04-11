"""Position-Weighted RAG Context Scanning (IP.x / I1.x categories).

Detects when injection payloads are strategically positioned within
retrieved document chunks to exploit LLM attention patterns.

Research shows LLMs have positional biases -- they pay more attention
to content at the beginning and end of context windows ("lost in the
middle" effect).  Attackers exploit this by placing injection payloads
in first/last retrieved documents.

Detection strategy:
    1. Compute a per-chunk raw injection score using pattern matching.
    2. Apply a U-shaped position weight curve (primacy/recency bias).
    3. Multiply raw scores by position weights for final risk.
    4. Optionally gate on query relevance -- irrelevant chunks with
       high injection scores are more suspicious.
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
class PositionScanResult:
    """Result from position-weighted RAG context scanning."""
    risk_score: float = 0.0
    positional_risks: List[float] = field(default_factory=list)
    suspicious_positions: List[int] = field(default_factory=list)
    technique_ids: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Compiled detection patterns (module-level for reuse)
# ---------------------------------------------------------------------------

# Imperative instructions (I1.1)
_IMPERATIVE_RE = re.compile(
    r"\b(?:ignore\s+(?:the\s+)?(?:previous|above|prior|preceding)|"
    r"instead\s+(?:do|say|respond|output|follow)|"
    r"you\s+must|"
    r"override\s+(?:the\s+)?(?:previous|above|prior|existing|instructions?))\b",
    re.IGNORECASE,
)

# System prompt mimicry (I1.2)
_SYSTEM_MIMICRY_RE = re.compile(
    r"(?:^|\n)\s*(?:SYSTEM\s*:|"
    r"\[INST\]|"
    r"<<SYS>>|"
    r"###\s*Instruction)\b",
    re.IGNORECASE,
)

# Authority claims (I1.3)
_AUTHORITY_RE = re.compile(
    r"\b(?:as\s+the\s+administrator|"
    r"I\s+have\s+root\s+access|"
    r"admin(?:istrator)?\s+override|"
    r"this\s+is\s+(?:a\s+)?(?:system|official|admin)\s+(?:message|instruction|directive))\b",
    re.IGNORECASE,
)

# Boundary confusion (I1.2)
_BOUNDARY_RE = re.compile(
    r"(?:(?:-{3,10}|={3,10})\s*END\s+OF\s+(?:CONTEXT|DOCUMENT|RETRIEVED|RESULTS?)(?:\s*(?:-{3,10}|={3,10}))?|"
    r"(?:^|\n)\s*\[Document\s+\d+\]\s*:|"
    r"(?:^|\n)\s*(?:###\s*)?(?:System|Assistant|User|Human)\s*:\s*(?:\n|$))",
    re.IGNORECASE,
)

# Pattern weights for scoring
_PATTERN_WEIGHTS = [
    (_IMPERATIVE_RE, 0.35, "I1.1"),
    (_SYSTEM_MIMICRY_RE, 0.30, "I1.2"),
    (_AUTHORITY_RE, 0.25, "I1.3"),
    (_BOUNDARY_RE, 0.30, "I1.2"),
]

# Threshold above which a chunk is flagged as suspicious
_SUSPICIOUS_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_position_weights(n: int) -> List[float]:
    """Generate a U-shaped weight curve for *n* chunks.

    Positions 0, 1, N-1, N-2 get higher weights (primacy/recency).
    Middle positions get a baseline weight.
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    if n == 2:
        return [1.0, 1.0]

    weights = [0.0] * n
    # Edges get high weight
    weights[0] = 1.0
    weights[-1] = 1.0
    if n >= 3:
        weights[1] = 0.7
    if n >= 4:
        weights[-2] = 0.7
    # Fill middle positions
    for i in range(2, n - 2):
        weights[i] = 0.3
    return weights


def _score_chunk(text: str) -> tuple:
    """Score a single chunk for injection signals.

    Returns (raw_score, set_of_technique_ids).
    """
    if not text or not text.strip():
        return 0.0, set()

    score = 0.0
    techniques: set = set()

    for pattern, weight, technique_id in _PATTERN_WEIGHTS:
        matches = pattern.findall(text)
        if matches:
            # Diminishing returns for multiple matches of same pattern
            contribution = weight * min(len(matches), 3) / 1.0
            # But cap each pattern's contribution
            contribution = min(contribution, weight * 1.5)
            score += contribution
            techniques.add(technique_id)

    return min(score, 1.0), techniques


def _query_word_overlap(text: str, query: str) -> float:
    """Compute word overlap ratio between text and query.

    Returns a value in [0, 1] indicating what fraction of query words
    appear in the text.
    """
    query_terms = {w.lower() for w in re.findall(r"\b\w{3,}\b", query)}
    if not query_terms:
        return 1.0  # No query terms means we can't judge relevance

    text_lower = text.lower()
    matching = sum(1 for t in query_terms if t in text_lower)
    return matching / len(query_terms)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def position_weighted_scan(
    chunks: List[str],
    query: str = "",
    position_weights: Optional[List[float]] = None,
) -> PositionScanResult:
    """Scan RAG context chunks with position-weighted injection detection.

    Parameters
    ----------
    chunks : list of str
        Retrieved document chunks in their retrieval order.
    query : str
        The original user query.  When provided, enables relevance
        gating -- irrelevant chunks with high injection scores are
        flagged as more suspicious.
    position_weights : list of float or None
        Custom per-chunk position weights.  Must have len == len(chunks).
        If None, a default U-shaped curve is used.

    Returns
    -------
    PositionScanResult
        Detection result with risk_score, per-chunk positional_risks,
        suspicious_positions, technique_ids, and details.
    """
    n = len(chunks)
    if n == 0:
        return PositionScanResult()

    # Resolve position weights
    if position_weights is not None:
        if len(position_weights) != n:
            logger.warning(
                "position_weights length %d != chunks length %d; "
                "falling back to default curve",
                len(position_weights), n,
            )
            weights = _default_position_weights(n)
        else:
            weights = list(position_weights)
    else:
        weights = _default_position_weights(n)

    # Score each chunk
    positional_risks: List[float] = []
    suspicious_positions: List[int] = []
    all_techniques: set = set()
    chunk_details: list = []

    for i, chunk in enumerate(chunks):
        raw_score, techniques = _score_chunk(chunk)

        # Apply position weight
        weighted_score = raw_score * weights[i]

        # Query relevance gate
        relevance_boost = 0.0
        if query and query.strip() and raw_score > _SUSPICIOUS_THRESHOLD:
            overlap = _query_word_overlap(chunk, query)
            if overlap < 0.2:
                # High injection + low relevance = very suspicious
                relevance_boost = 0.3
                techniques.add("I1.4")
            elif overlap < 0.4:
                relevance_boost = 0.15
                techniques.add("I1.4")

        final_score = min(weighted_score + relevance_boost, 1.0)
        positional_risks.append(final_score)

        if final_score > _SUSPICIOUS_THRESHOLD:
            suspicious_positions.append(i)
            all_techniques.update(techniques)
            # Check for positional exploitation
            if weights[i] >= 0.7 and raw_score > _SUSPICIOUS_THRESHOLD:
                all_techniques.add("IP.1")

        chunk_details.append({
            "position": i,
            "raw_score": round(raw_score, 4),
            "position_weight": round(weights[i], 4),
            "weighted_score": round(final_score, 4),
            "techniques": sorted(techniques) if techniques else [],
        })

    # Overall risk = max of position-weighted scores
    risk_score = max(positional_risks) if positional_risks else 0.0

    details = {
        "chunk_count": n,
        "position_weights": [round(w, 4) for w in weights],
        "chunk_details": chunk_details,
    }

    return PositionScanResult(
        risk_score=round(risk_score, 4),
        positional_risks=[round(r, 4) for r in positional_risks],
        suspicious_positions=suspicious_positions,
        technique_ids=sorted(all_techniques),
        details=details,
    )
