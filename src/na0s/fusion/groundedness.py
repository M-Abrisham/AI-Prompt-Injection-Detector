"""Layer 6: Self-RAG groundedness check — verify MALICIOUS verdicts are grounded in evidence.

A MALICIOUS verdict backed by only a single evidence source has a higher
chance of being a false positive.  This module counts independent evidence
sources and recommends review when insufficient corroboration is found.
"""

from __future__ import annotations

from ..scan_result import ScanResult


def verify_verdict_grounded(
    scan_result: ScanResult,
    min_sources: int = 2,
) -> dict:
    """Check whether a scan verdict is grounded in multiple evidence sources.

    Independent evidence sources counted:
    * ``ml`` — ML confidence > 0.6
    * ``rules`` — at least one rule hit
    * ``anomaly`` — at least one anomaly flag
    * ``embedding`` — embedding score > 0.1
    * ``techniques`` — at least one technique tag (excluding cascade-stage tags)

    Parameters
    ----------
    scan_result : ScanResult
        The scan result to evaluate.
    min_sources : int
        Minimum number of independent sources required to consider the
        verdict *grounded* (default 2).

    Returns
    -------
    dict
        ``{"grounded": bool, "source_count": int, "sources": list[str],
        "recommendation": str}``
    """
    sources: list[str] = []

    # 1. ML confidence
    if scan_result.ml_confidence > 0.6:
        sources.append("ml")

    # 2. Rule hits
    if len(scan_result.rule_hits) > 0:
        sources.append("rules")

    # 3. Anomaly flags
    if len(scan_result.anomaly_flags) > 0:
        sources.append("anomaly")

    # 4. Embedding score
    if scan_result.embedding_score > 0.1:
        sources.append("embedding")

    # 5. Technique tags (exclude cascade-stage tags like "cascade:weighted")
    real_technique_tags = [
        t for t in scan_result.technique_tags
        if not t.startswith("cascade:")
    ]
    if len(real_technique_tags) > 0:
        sources.append("techniques")

    source_count = len(sources)
    grounded = source_count >= min_sources

    # Recommendation logic
    if scan_result.is_malicious:
        if grounded:
            recommendation = "confirmed"
        else:
            recommendation = "review"
    else:
        # SAFE verdicts: grounded = confident safe; not grounded = may need check
        if grounded:
            recommendation = "confirmed"
        else:
            recommendation = "confirmed"  # safe with few signals is normal

    return {
        "grounded": grounded,
        "source_count": source_count,
        "sources": sources,
        "recommendation": recommendation,
    }
