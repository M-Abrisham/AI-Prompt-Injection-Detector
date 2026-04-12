"""Base classes and utilities for the Office document parser suite.

Provides:
- ``ExtractedArtifact`` — normalized container for extracted text + location
- ``OfficeExtractor`` — abstract base class for format-specific extractors
- ``detect_format()`` — magic-byte-based format detection (no file extension)
- ``UnsupportedDocumentError`` — raised for formats we explicitly reject
- Zip-bomb and malformed-XML safety constants
"""

from __future__ import annotations

import io
import logging
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------

# Maximum decompressed size from a ZIP archive (zip-bomb protection).
# 200 MB is generous for any legitimate office document.
MAX_DECOMPRESSED_BYTES: int = 200 * 1024 * 1024

# Maximum number of files inside a ZIP (zip-bomb: millions of tiny entries).
MAX_ZIP_ENTRIES: int = 10_000

# Maximum XML file size to parse (prevents billion-laughs from a single entry).
MAX_XML_BYTES: int = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExtractedArtifact:
    """A single piece of text extracted from an office document.

    Each artifact represents one user-controllable text surface — a cell,
    a comment, a tracked-change insertion, a metadata field, etc.  The
    ``location`` string lets the downstream rules pipeline report *where*
    in the document an injection payload was hiding.

    Attributes
    ----------
    location : str
        Format-specific location tag, e.g.
        ``"docx:comments/comment[3]"``,
        ``"xlsx:sheet2[veryHidden]/A1"``,
        ``"pptx:slide1/notes"``,
        ``"odt:meta/user-defined[secret]"``.
    text : str
        The extracted text content.  May be empty if the location exists
        but contains no text (still worth reporting for structural probes).
    metadata : dict
        Optional key-value pairs (author, date, cell address, etc.).
    """

    location: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return "ExtractedArtifact(location={!r}, text={!r})".format(
            self.location, preview,
        )


class UnsupportedDocumentError(Exception):
    """Raised when the document format is recognized but not supported.

    For example, Apple .pages files are valid ZIP archives but we
    explicitly do not parse them.  Callers should catch this and
    surface a clear error to the operator rather than silently returning
    zero artifacts.
    """


# ---------------------------------------------------------------------------
# Magic-byte format detection
# ---------------------------------------------------------------------------

# OLE Compound File Binary Format (legacy .doc, .xls, .ppt)
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# PK ZIP header (OOXML: docx/xlsx/pptx, ODF: odt/ods/odp)
_PK_MAGIC = b"PK\x03\x04"


def detect_format(data: bytes) -> Optional[str]:
    """Detect office document format from magic bytes.

    Inspects the first few bytes and, for ZIP-based formats, peeks at
    internal paths to distinguish OOXML sub-formats and ODF.

    Parameters
    ----------
    data : bytes
        Raw document bytes (at least the first 8 bytes are needed).

    Returns
    -------
    str or None
        One of ``"docx"``, ``"xlsx"``, ``"pptx"``, ``"odt"``, ``"ods"``,
        ``"odp"``, ``"ole"``, or ``None`` if the format is not recognized.

    Raises
    ------
    UnsupportedDocumentError
        If the file is a recognized-but-unsupported format (e.g. .pages).
    """
    if len(data) < 4:
        return None

    # --- OLE Compound Binary (legacy Office) ---------------------------------
    if data[:8] == _OLE_MAGIC:
        return "ole"

    # --- ZIP-based formats (OOXML + ODF) -------------------------------------
    if data[:4] != _PK_MAGIC:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            names = set(zf.namelist())

            # Apple .pages — explicitly reject
            if "Index/Document.iwa" in names or any(
                n.startswith("Index/") for n in names
            ):
                raise UnsupportedDocumentError(
                    "Apple .pages format is not supported. "
                    "Export to DOCX or PDF before scanning."
                )

            # OOXML detection by content-type marker
            if "[Content_Types].xml" in names:
                if any(n.startswith("word/") for n in names):
                    return "docx"
                if any(n.startswith("xl/") for n in names):
                    return "xlsx"
                if any(n.startswith("ppt/") for n in names):
                    return "pptx"
                # Unknown OOXML variant
                return None

            # ODF detection by mimetype file
            if "mimetype" in names:
                try:
                    mime = zf.read("mimetype").decode("ascii", errors="replace").strip()
                except Exception:
                    mime = ""
                if "opendocument.text" in mime:
                    return "odt"
                if "opendocument.spreadsheet" in mime:
                    return "ods"
                if "opendocument.presentation" in mime:
                    return "odp"

            # ODF fallback: check for content.xml (some ODF files lack mimetype)
            if "content.xml" in names and "meta.xml" in names:
                return "odt"  # default to text if ambiguous

    except UnsupportedDocumentError:
        raise  # re-raise .pages rejection
    except (zipfile.BadZipFile, Exception) as exc:
        logger.debug("detect_format: ZIP inspection failed: %s", exc)
        return None

    return None


# ---------------------------------------------------------------------------
# Safe ZIP extraction helper
# ---------------------------------------------------------------------------

def safe_read_zip_entry(
    zf: zipfile.ZipFile,
    entry_name: str,
    *,
    max_bytes: int = MAX_XML_BYTES,
) -> Optional[bytes]:
    """Read a single ZIP entry with size guards.

    Returns None if the entry doesn't exist or exceeds the size limit.
    """
    try:
        info = zf.getinfo(entry_name)
    except KeyError:
        return None

    if info.file_size > max_bytes:
        logger.warning(
            "office parser: skipping %s (%.1f MB > %.1f MB limit)",
            entry_name,
            info.file_size / (1024 * 1024),
            max_bytes / (1024 * 1024),
        )
        return None

    try:
        return zf.read(entry_name)
    except Exception as exc:
        logger.debug("office parser: failed to read %s: %s", entry_name, exc)
        return None


def validate_zip_safety(zf: zipfile.ZipFile) -> list[str]:
    """Check a ZipFile for zip-bomb indicators.

    Returns a list of warning strings (empty = safe).
    """
    warnings: list[str] = []

    if len(zf.infolist()) > MAX_ZIP_ENTRIES:
        warnings.append(
            "ZIP has {} entries (limit {}); possible zip bomb".format(
                len(zf.infolist()), MAX_ZIP_ENTRIES,
            )
        )

    total_size = sum(info.file_size for info in zf.infolist())
    if total_size > MAX_DECOMPRESSED_BYTES:
        warnings.append(
            "ZIP decompressed size {:.1f} MB > {:.1f} MB limit; possible zip bomb".format(
                total_size / (1024 * 1024),
                MAX_DECOMPRESSED_BYTES / (1024 * 1024),
            )
        )

    return warnings


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class OfficeExtractor(ABC):
    """Abstract base class for format-specific office document extractors.

    Subclasses implement ``extract()`` which takes raw document bytes
    and returns a list of ``ExtractedArtifact`` objects — one per
    user-controllable text surface found in the document.

    The base class provides shared zip-safety checks and XML parsing
    utilities.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Short format identifier (e.g. ``"docx"``, ``"xlsx"``)."""

    @abstractmethod
    def extract(self, data: bytes) -> List[ExtractedArtifact]:
        """Extract all text artifacts from the document.

        Parameters
        ----------
        data : bytes
            Raw document bytes.

        Returns
        -------
        list[ExtractedArtifact]
            One artifact per user-controllable text surface.  May be
            empty if the document has no extractable text.

        Raises
        ------
        UnsupportedDocumentError
            If the document is recognized but cannot be parsed (e.g.,
            encrypted OLE, password-protected OOXML).
        """
