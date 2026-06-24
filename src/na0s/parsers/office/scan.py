"""Route extracted office-document text into the Na0S detection pipeline.

The office extractors (:mod:`na0s.parsers.office.router`) pull hidden text
out of comments, tracked changes, hidden sheets, speaker notes, metadata,
custom properties, etc.  Until this helper existed, that extracted text was
*never scored* — nothing fed ``artifact.text`` into :func:`na0s.predict.scan`,
so a prompt-injection / worm payload hiding in a DOCX comment passed straight
through the detector untouched.

:func:`scan_document` closes that ingestion-channel gap: it runs
``router.extract(data)`` and pushes each artifact's text through
``predict.scan()``, then aggregates the per-artifact verdicts into a single
result (any-malicious / max-risk).  This is the same pattern the visual
injection detector uses for OCR-extracted text
(:mod:`na0s.detectors.visual_injection`).

.. note::

    EMAIL and WEB ingestion remain the integrator's responsibility.  Na0S
    holds no email or web content of its own — inbound message bodies and
    fetched pages must be passed through :func:`na0s.predict.scan` (or these
    channel helpers) by the host application before they reach the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from na0s.parsers.office.base import ExtractedArtifact

logger = logging.getLogger(__name__)


@dataclass
class ArtifactFinding:
    """One artifact's scan verdict, tagged with where it was hiding."""

    location: str
    is_malicious: bool
    risk_score: float
    label: str
    rule_hits: List[str] = field(default_factory=list)
    technique_tags: List[str] = field(default_factory=list)
    text_preview: str = ""


@dataclass
class DocumentScanResult:
    """Aggregated verdict across every text surface in an office document.

    Attributes
    ----------
    is_malicious : bool
        True if *any* artifact was flagged malicious.
    risk_score : float
        The maximum per-artifact risk score (worst surface wins).
    label : str
        ``"malicious"`` if any artifact tripped, else ``"safe"``.
    rule_hits : list[str]
        Union of rule hits across all artifacts (e.g. ``worm:self_replication``).
    technique_tags : list[str]
        Union of technique tags across all artifacts (e.g. ``IM1.6``).
    findings : list[ArtifactFinding]
        Per-artifact detail, including benign ones, for forensics.
    artifact_count : int
        Number of artifacts the extractor produced.
    scanned_count : int
        Number of non-empty artifacts actually run through ``scan()``.
    """

    is_malicious: bool = False
    risk_score: float = 0.0
    label: str = "safe"
    rule_hits: List[str] = field(default_factory=list)
    technique_tags: List[str] = field(default_factory=list)
    findings: List[ArtifactFinding] = field(default_factory=list)
    artifact_count: int = 0
    scanned_count: int = 0


def scan_document(data: bytes) -> DocumentScanResult:
    """Extract an office document and scan every hidden text surface.

    Detects the format from magic bytes, extracts all artifacts via
    :func:`na0s.parsers.office.router.extract`, and routes each non-empty
    ``artifact.text`` through :func:`na0s.predict.scan`.  The returned
    :class:`DocumentScanResult` reports the worst surface (max risk /
    any-malicious) and keeps per-artifact findings.

    Fail-safe: if extraction or a per-artifact scan raises, the error is
    logged and that surface is skipped rather than crashing the host.  A
    document the parser cannot decode (unknown format, malformed ZIP) yields
    an empty, benign result.

    Parameters
    ----------
    data : bytes
        Raw office-document bytes (DOCX/XLSX/PPTX/ODF/OLE).

    Returns
    -------
    DocumentScanResult
        Aggregated verdict plus per-artifact findings.

    .. note::

        EMAIL/WEB ingestion is the integrator's responsibility — Na0S never
        fetches mail or pages; pass those bodies through ``scan()`` yourself.
    """
    # Lazy imports to avoid an import cycle (predict imports many detectors).
    from na0s.parsers.office.router import extract

    result = DocumentScanResult()

    try:
        artifacts: List[ExtractedArtifact] = extract(data)
    except Exception as exc:
        # Unknown format, malformed ZIP, unsupported document, etc.
        logger.debug("scan_document: extraction failed: %s", exc)
        return result

    result.artifact_count = len(artifacts)

    for artifact in artifacts:
        text = getattr(artifact, "text", "") or ""
        if not text.strip():
            continue

        try:
            from na0s.predict import scan as na0s_scan

            scan_res = na0s_scan(text)
        except Exception as exc:
            logger.debug(
                "scan_document: scan() failed for %s: %s",
                getattr(artifact, "location", "?"), exc,
            )
            continue

        result.scanned_count += 1

        finding = ArtifactFinding(
            location=getattr(artifact, "location", ""),
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
        for hit in finding.rule_hits:
            if hit not in result.rule_hits:
                result.rule_hits.append(hit)
        for tag in finding.technique_tags:
            if tag not in result.technique_tags:
                result.technique_tags.append(tag)

    result.label = "malicious" if result.is_malicious else "safe"

    if result.is_malicious:
        logger.warning(
            "scan_document: flagged office document "
            "(risk=%.2f, surfaces=%d/%d, hits=%s)",
            result.risk_score, result.scanned_count, result.artifact_count,
            result.rule_hits,
        )

    return result
