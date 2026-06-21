"""Office document parser — deep extraction of hidden injection surfaces.

Extracts text from locations that naive extractors miss: comments,
tracked changes, hidden sheets, speaker notes, metadata, custom
properties, headers/footers, defined names, etc.

Usage::

    from na0s.parsers.office import extract, detect_format

    fmt = detect_format(raw_bytes)        # "docx", "xlsx", etc.
    artifacts = extract(raw_bytes)        # list[ExtractedArtifact]
    for a in artifacts:
        result = scan(a.text)             # feed into Na0S pipeline
"""

from na0s.parsers.office.base import (
    ExtractedArtifact,
    OfficeExtractor,
    UnsupportedDocumentError,
    detect_format,
)
from na0s.parsers.office.router import extract
from na0s.parsers.office.scan import (
    ArtifactFinding,
    DocumentScanResult,
    scan_document,
)

__all__ = [
    "ExtractedArtifact",
    "OfficeExtractor",
    "UnsupportedDocumentError",
    "detect_format",
    "extract",
    "scan_document",
    "DocumentScanResult",
    "ArtifactFinding",
]
