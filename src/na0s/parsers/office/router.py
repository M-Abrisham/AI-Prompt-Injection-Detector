"""Office document router — detect format and dispatch to the right extractor.

The router is the single entry point for consumers.  Given raw bytes, it
detects the format via magic bytes and delegates to the format-specific
extractor.  Consumers should never import individual extractors directly.

Usage::

    from na0s.parsers.office.router import extract

    artifacts = extract(raw_bytes)
    for a in artifacts:
        print(a.location, a.text[:80])
"""

from __future__ import annotations

import logging
from typing import List

from na0s.parsers.office.base import (
    ExtractedArtifact,
    UnsupportedDocumentError,
    detect_format,
)

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading all extractors at module level.
# Each extractor module may import heavy XML/OLE libraries.
_EXTRACTOR_REGISTRY: dict[str, str] = {
    "docx": "na0s.parsers.office.docx_extractor",
    "xlsx": "na0s.parsers.office.xlsx_extractor",
    "pptx": "na0s.parsers.office.pptx_extractor",
    "odt":  "na0s.parsers.office.odf_extractor",
    "ods":  "na0s.parsers.office.odf_extractor",
    "odp":  "na0s.parsers.office.odf_extractor",
    "ole":  "na0s.parsers.office.ole_extractor",
}


def extract(data: bytes) -> List[ExtractedArtifact]:
    """Extract all hidden-text artifacts from an office document.

    Detects the format from magic bytes (no file extension needed),
    dispatches to the appropriate extractor, and returns a list of
    ``ExtractedArtifact`` objects ready for the Na0S detection pipeline.

    Parameters
    ----------
    data : bytes
        Raw document bytes.

    Returns
    -------
    list[ExtractedArtifact]
        One artifact per user-controllable text surface found.

    Raises
    ------
    UnsupportedDocumentError
        If the format is recognized but explicitly unsupported
        (e.g., Apple .pages).
    ValueError
        If the format cannot be detected from magic bytes.
    """
    fmt = detect_format(data)
    if fmt is None:
        raise ValueError(
            "Cannot detect office document format from magic bytes. "
            "Supported: DOCX, XLSX, PPTX, ODT, ODS, ODP, OLE (.doc/.xls/.ppt)."
        )

    module_path = _EXTRACTOR_REGISTRY.get(fmt)
    if module_path is None:
        raise ValueError("No extractor registered for format: {!r}".format(fmt))

    # Lazy-import the extractor module
    import importlib
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            "Extractor module {!r} could not be imported: {}. "
            "Check that all dependencies are installed.".format(module_path, exc)
        ) from exc

    # Every extractor module must expose an `Extractor` class that
    # subclasses OfficeExtractor.
    extractor_cls = getattr(mod, "Extractor", None)
    if extractor_cls is None:
        raise AttributeError(
            "Extractor module {!r} does not expose an 'Extractor' class.".format(
                module_path,
            )
        )

    extractor = extractor_cls()
    artifacts = extractor.extract(data)

    logger.debug(
        "office router: format=%s, artifacts=%d",
        fmt, len(artifacts),
    )

    return artifacts
