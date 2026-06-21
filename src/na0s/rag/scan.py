"""Route RAG-retrieved chunks through the Na0S detection pipeline.

Retrieved context is an indirect prompt-injection channel: a poisoned
document chunk pulled from a vector store can carry a worm / jailbreak
payload that the model then ingests as if it were trusted context.

Today the worm (IM1.6) signal — and the rest of the rule stack — only runs
on retrieved content *if the host happens to concatenate the chunks into the
single string it later passes to* :func:`na0s.predict.scan`.  Hosts that join
with custom separators, score chunks individually, or feed them straight to
the model never get the worm scan.  :func:`scan_retrieved_chunks` makes that
coverage explicit: it scans every chunk through ``predict.scan()`` and
aggregates, so the worm signal provably runs on retrieved content regardless
of how the host assembles the final prompt.

.. note::

    EMAIL and WEB ingestion remain the integrator's responsibility.  Na0S
    holds no email or web content of its own — inbound message bodies and
    fetched pages must be passed through :func:`na0s.predict.scan` (or these
    channel helpers) by the host application before they reach the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Sequence

logger = logging.getLogger(__name__)

# Keys commonly used to carry chunk text in RAG frameworks (LangChain
# Document.page_content, LlamaIndex node.text, raw dict {"content": ...}).
_TEXT_KEYS = ("text", "content", "page_content", "chunk", "passage", "document")


@dataclass
class ChunkFinding:
    """One retrieved chunk's scan verdict, tagged with its retrieval index."""

    index: int
    is_malicious: bool
    risk_score: float
    label: str
    rule_hits: List[str] = field(default_factory=list)
    technique_tags: List[str] = field(default_factory=list)
    text_preview: str = ""


@dataclass
class ChunksScanResult:
    """Aggregated verdict across every retrieved chunk.

    Attributes
    ----------
    is_malicious : bool
        True if *any* chunk was flagged malicious.
    risk_score : float
        The maximum per-chunk risk score (worst chunk wins).
    label : str
        ``"malicious"`` if any chunk tripped, else ``"safe"``.
    rule_hits : list[str]
        Union of rule hits across all chunks (e.g. ``worm:self_replication``).
    technique_tags : list[str]
        Union of technique tags across all chunks (e.g. ``IM1.6``).
    findings : list[ChunkFinding]
        Per-chunk detail, including benign chunks, for forensics.
    chunk_count : int
        Number of chunks supplied.
    scanned_count : int
        Number of non-empty chunks actually run through ``scan()``.
    malicious_indices : list[int]
        Retrieval indices of the chunks that were flagged.
    """

    is_malicious: bool = False
    risk_score: float = 0.0
    label: str = "safe"
    rule_hits: List[str] = field(default_factory=list)
    technique_tags: List[str] = field(default_factory=list)
    findings: List[ChunkFinding] = field(default_factory=list)
    chunk_count: int = 0
    scanned_count: int = 0
    malicious_indices: List[int] = field(default_factory=list)


def _chunk_text(chunk: Any) -> str:
    """Coerce a retrieved chunk into its text content.

    Accepts plain strings, dicts with a known text key, and objects with a
    ``text`` / ``page_content`` / ``content`` attribute (LangChain /
    LlamaIndex style).  Falls back to ``str(chunk)`` only for non-empty
    scalars; returns ``""`` for ``None``.
    """
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for key in _TEXT_KEYS:
            val = chunk.get(key)
            if isinstance(val, str) and val:
                return val
        return ""
    for key in _TEXT_KEYS:
        val = getattr(chunk, key, None)
        if isinstance(val, str) and val:
            return val
    return ""


def scan_retrieved_chunks(chunks: Sequence[Any]) -> ChunksScanResult:
    """Scan each RAG-retrieved chunk through :func:`na0s.predict.scan`.

    Runs every non-empty chunk through the full Na0S pipeline (the same rule
    stack as direct user input, including the worm / IM1.6 self-replication
    signal) and aggregates the per-chunk verdicts into one result
    (any-malicious / max-risk).

    Fail-safe: a per-chunk scan error is logged and that chunk skipped rather
    than crashing the host.

    Parameters
    ----------
    chunks : sequence
        Retrieved document chunks in retrieval order.  Each item may be a
        plain string, a dict with a ``text`` / ``content`` / ``page_content``
        key, or an object exposing one of those as an attribute.

    Returns
    -------
    ChunksScanResult
        Aggregated verdict plus per-chunk findings.

    .. note::

        EMAIL/WEB ingestion is the integrator's responsibility — Na0S never
        fetches mail or pages; pass those bodies through ``scan()`` yourself.
    """
    result = ChunksScanResult()

    if not chunks:
        return result

    result.chunk_count = len(chunks)

    for idx, chunk in enumerate(chunks):
        text = _chunk_text(chunk)
        if not text.strip():
            continue

        try:
            # Lazy import to avoid an import cycle (predict imports rag).
            from na0s.predict import scan as na0s_scan

            scan_res = na0s_scan(text)
        except Exception as exc:
            logger.debug(
                "scan_retrieved_chunks: scan() failed for chunk %d: %s",
                idx, exc,
            )
            continue

        result.scanned_count += 1

        finding = ChunkFinding(
            index=idx,
            is_malicious=bool(scan_res.is_malicious),
            risk_score=float(scan_res.risk_score),
            label=str(scan_res.label),
            rule_hits=list(scan_res.rule_hits or []),
            technique_tags=list(scan_res.technique_tags or []),
            text_preview=text[:120],
        )
        result.findings.append(finding)

        result.risk_score = max(result.risk_score, finding.risk_score)
        if finding.is_malicious:
            result.is_malicious = True
            result.malicious_indices.append(idx)
        for hit in finding.rule_hits:
            if hit not in result.rule_hits:
                result.rule_hits.append(hit)
        for tag in finding.technique_tags:
            if tag not in result.technique_tags:
                result.technique_tags.append(tag)

    result.label = "malicious" if result.is_malicious else "safe"

    if result.is_malicious:
        logger.warning(
            "scan_retrieved_chunks: flagged retrieved context "
            "(risk=%.2f, chunks=%s/%d, hits=%s)",
            result.risk_score, result.malicious_indices, result.chunk_count,
            result.rule_hits,
        )

    return result
