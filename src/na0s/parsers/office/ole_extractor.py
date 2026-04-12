"""OLE legacy extractor — handles .doc, .xls, .ppt (OLE Compound Binary Format).

Extracts user-controllable text surfaces from legacy Office documents using
the ``olefile`` library for OLE stream access, with optional ``oletools``
(VBA macros) and ``xlrd`` (cell values) support.

Three-tier extraction strategy:
  Tier 1 — SummaryInformation, DocumentSummaryInformation, stream listing
  Tier 2 — VBA macros, .ppt slide text, .xls sheet names/visibility
  Tier 3 — raw string fallback for .doc body text, xlrd cell values
"""

from __future__ import annotations

import logging
import re
import struct
from typing import Any, Dict, List, Optional

from na0s.parsers.office.base import (
    ExtractedArtifact,
    OfficeExtractor,
    UnsupportedDocumentError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import olefile  # type: ignore[import-untyped]

    _HAS_OLEFILE = True
except ImportError:
    _HAS_OLEFILE = False

try:
    from oletools.olevba import VBA_Parser  # type: ignore[import-untyped]

    _HAS_OLETOOLS = True
except ImportError:
    _HAS_OLETOOLS = False

try:
    import xlrd  # type: ignore[import-untyped]

    _HAS_XLRD = True
except ImportError:
    _HAS_XLRD = False


# ---------------------------------------------------------------------------
# Summary metadata fields to scan
# ---------------------------------------------------------------------------

_SUMMARY_FIELDS = (
    "title",
    "subject",
    "author",
    "keywords",
    "comments",
    "last_saved_by",
    "creating_application",
)

_DOCSUMMARY_FIELDS = (
    "category",
    "manager",
    "company",
    "content_type",
    "content_status",
)


# ---------------------------------------------------------------------------
# Extractor implementation
# ---------------------------------------------------------------------------


class Extractor(OfficeExtractor):
    """Extract text artifacts from legacy OLE (.doc/.xls/.ppt) documents."""

    @property
    def format_name(self) -> str:
        return "ole"

    # -- public API ----------------------------------------------------------

    def extract(self, data: bytes) -> List[ExtractedArtifact]:
        if not _HAS_OLEFILE:
            raise UnsupportedDocumentError(
                "olefile is required for legacy .doc/.xls/.ppt parsing"
            )

        ole = olefile.OleFileIO(data)
        try:
            self._check_encryption(ole)

            artifacts: List[ExtractedArtifact] = []

            # Tier 1
            self._extract_summary_info(ole, artifacts)
            self._extract_docsummary_info(ole, artifacts)
            self._extract_custom_properties(ole, artifacts)
            self._extract_stream_listing(ole, artifacts)

            # Tier 2
            self._extract_vba_macros(ole, data, artifacts)
            self._extract_ppt_slide_text(ole, artifacts)
            self._extract_xls_sheet_names(ole, artifacts)

            # Tier 3
            self._extract_xls_cell_values(data, artifacts)
            self._extract_raw_strings(ole, artifacts)

            return artifacts
        finally:
            ole.close()

    # -- encryption detection ------------------------------------------------

    @staticmethod
    def _check_encryption(ole: Any) -> None:
        """Raise if the OLE document is encrypted."""
        if ole.exists("\x06DataSpaces") or ole.exists("EncryptedPackage"):
            raise UnsupportedDocumentError(
                "Encrypted OLE document; decrypt before scanning"
            )

    # -- Tier 1 --------------------------------------------------------------

    @staticmethod
    def _extract_summary_info(
        ole: Any,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract SummaryInformation metadata fields."""
        try:
            meta = ole.get_metadata()
        except Exception:
            logger.debug("ole_extractor: failed to read SummaryInformation")
            return

        for field_name in _SUMMARY_FIELDS:
            value = getattr(meta, field_name, None)
            if value is None:
                continue
            text = _decode_meta_value(value)
            if text:
                artifacts.append(
                    ExtractedArtifact(
                        location="ole:summary/{}".format(field_name),
                        text=text,
                    )
                )

    @staticmethod
    def _extract_docsummary_info(
        ole: Any,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract standard DocumentSummaryInformation fields."""
        try:
            meta = ole.get_metadata()
        except Exception:
            return

        for field_name in _DOCSUMMARY_FIELDS:
            value = getattr(meta, field_name, None)
            if value is None:
                continue
            text = _decode_meta_value(value)
            if text:
                artifacts.append(
                    ExtractedArtifact(
                        location="ole:summary/{}".format(field_name),
                        text=text,
                    )
                )

    @staticmethod
    def _extract_custom_properties(
        ole: Any,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract custom document properties (section 1 of DocSummaryInfo)."""
        stream_name = "\x05DocumentSummaryInformation"
        if not ole.exists(stream_name):
            return

        # Section 1 holds the user-defined custom properties.
        # olefile API: getproperties(stream_name, ..., section=N)
        try:
            custom_props: Optional[Dict[Any, Any]] = ole.getproperties(
                stream_name, convert_time=True, no_conversion=[], section=1,
            )
        except Exception:
            custom_props = None

        if not custom_props:
            return

        for key, value in custom_props.items():
            if isinstance(key, int):
                # Numeric property IDs from olefile — skip codepage markers, etc.
                continue
            text = _decode_meta_value(value)
            if text:
                artifacts.append(
                    ExtractedArtifact(
                        location="ole:custom_properties/{}".format(key),
                        text=text,
                    )
                )

    @staticmethod
    def _extract_stream_listing(
        ole: Any,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Report all OLE stream names."""
        try:
            stream_paths = ole.listdir(streams=True, storages=True)
        except Exception:
            return

        joined = ["/".join(parts) for parts in stream_paths]
        if joined:
            artifacts.append(
                ExtractedArtifact(
                    location="ole:streams",
                    text="\n".join(sorted(joined)),
                    metadata={"count": str(len(joined))},
                )
            )

    # -- Tier 2 --------------------------------------------------------------

    @staticmethod
    def _extract_vba_macros(
        ole: Any,
        data: bytes,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract VBA macro source code (oletools) or flag VBA presence."""
        if _HAS_OLETOOLS:
            try:
                vba_parser = VBA_Parser(filename="inmemory.bin", data=data)
                try:
                    if vba_parser.detect_vba_macros():
                        for filename, stream_path, vba_filename, vba_code in (
                            vba_parser.extract_macros()
                        ):
                            if vba_code:
                                artifacts.append(
                                    ExtractedArtifact(
                                        location="ole:vba/{}".format(vba_filename),
                                        text=vba_code,
                                        metadata={
                                            "stream_path": stream_path,
                                            "filename": filename,
                                        },
                                    )
                                )
                finally:
                    vba_parser.close()
            except Exception:
                logger.debug("ole_extractor: oletools VBA extraction failed", exc_info=True)
            return

        # Fallback: detect VBA storage presence without decompiling.
        vba_storages: List[str] = []
        try:
            for parts in ole.listdir(streams=True, storages=True):
                joined = "/".join(parts)
                if "VBA" in parts or "_VBA_PROJECT_CUR" in parts:
                    vba_storages.append(joined)
        except Exception:
            return

        if vba_storages:
            artifacts.append(
                ExtractedArtifact(
                    location="ole:vba/_detected",
                    text="VBA project storage detected (oletools not available for decompilation)",
                    metadata={"vba_streams": ", ".join(vba_storages)},
                )
            )

    @staticmethod
    def _extract_ppt_slide_text(
        ole: Any,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Scan PowerPoint Document stream for TextCharsAtom/TextBytesAtom."""
        stream_name = "PowerPoint Document"
        if not ole.exists(stream_name):
            return

        try:
            data = ole.openstream(stream_name).read()
        except Exception:
            logger.debug("ole_extractor: failed to read PowerPoint Document stream")
            return

        texts: List[str] = []
        pos = 0
        while pos < len(data) - 8:
            try:
                rec_ver_inst, rec_type, rec_len = struct.unpack_from(
                    "<HHI", data, pos,
                )
            except struct.error:
                break

            # Guard against corrupt/huge record lengths
            if rec_len > len(data) - pos - 8:
                break

            if rec_type == 0x0FA0:  # TextCharsAtom (UTF-16LE)
                text = data[pos + 8 : pos + 8 + rec_len].decode(
                    "utf-16-le", errors="replace",
                )
                if text.strip():
                    texts.append(text)
            elif rec_type == 0x0FA8:  # TextBytesAtom (single-byte)
                text = data[pos + 8 : pos + 8 + rec_len].decode(
                    "latin-1", errors="replace",
                )
                if text.strip():
                    texts.append(text)

            pos += 8 + rec_len

        for idx, text in enumerate(texts, 1):
            artifacts.append(
                ExtractedArtifact(
                    location="ole:ppt/slide_text[{}]".format(idx),
                    text=text,
                )
            )

    @staticmethod
    def _extract_xls_sheet_names(
        ole: Any,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Parse BoundSheet8 records from Workbook stream."""
        # Try both stream names (BIFF8 vs BIFF5)
        stream_name: Optional[str] = None
        for candidate in ("Workbook", "Book"):
            if ole.exists(candidate):
                stream_name = candidate
                break
        if stream_name is None:
            return

        try:
            data = ole.openstream(stream_name).read()
        except Exception:
            logger.debug("ole_extractor: failed to read %s stream", stream_name)
            return

        _VISIBILITY = {0: "visible", 1: "hidden", 2: "veryHidden"}
        _SHEET_TYPE = {0: "worksheet", 1: "macro", 2: "chart", 6: "vb_module"}

        sheets: List[Dict[str, str]] = []
        pos = 0
        while pos < len(data) - 4:
            try:
                rec_type, rec_len = struct.unpack_from("<HH", data, pos)
            except struct.error:
                break

            if rec_type == 0x0000 and rec_len == 0:
                break  # end of records sentinel

            if rec_type == 0x0085 and rec_len >= 8:  # BoundSheet8
                end = pos + 4 + rec_len
                if end <= len(data):
                    hs_state = data[pos + 8] & 0x03
                    dt = data[pos + 9]
                    name_len = data[pos + 10]
                    name_flags = data[pos + 11] if pos + 11 < len(data) else 0

                    if name_flags & 0x01:  # Unicode
                        name_end = pos + 12 + name_len * 2
                        if name_end <= end:
                            name = data[pos + 12 : name_end].decode(
                                "utf-16-le", errors="replace",
                            )
                        else:
                            name = "<unreadable>"
                    else:
                        name_end = pos + 12 + name_len
                        if name_end <= end:
                            name = data[pos + 12 : name_end].decode(
                                "latin-1", errors="replace",
                            )
                        else:
                            name = "<unreadable>"

                    sheets.append({
                        "name": name,
                        "visibility": _VISIBILITY.get(hs_state, "unknown"),
                        "type": _SHEET_TYPE.get(dt, "unknown"),
                    })

            pos += 4 + rec_len

        if sheets:
            lines = []
            for s in sheets:
                lines.append(
                    "{} [{}] ({})".format(s["name"], s["visibility"], s["type"])
                )
            artifacts.append(
                ExtractedArtifact(
                    location="ole:xls/sheet_names",
                    text="\n".join(lines),
                    metadata={"count": str(len(sheets))},
                )
            )

    # -- Tier 3 --------------------------------------------------------------

    @staticmethod
    def _extract_xls_cell_values(
        data: bytes,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract cell values using xlrd if available."""
        if not _HAS_XLRD:
            return

        try:
            book = xlrd.open_workbook(file_contents=data)
        except Exception:
            logger.debug("ole_extractor: xlrd failed to open workbook", exc_info=True)
            return

        for sheet_idx in range(book.nsheets):
            sheet = book.sheet_by_index(sheet_idx)
            for row_idx in range(sheet.nrows):
                for col_idx in range(sheet.ncols):
                    cell = sheet.cell(row_idx, col_idx)
                    if cell.ctype == xlrd.XL_CELL_TEXT and cell.value:
                        artifacts.append(
                            ExtractedArtifact(
                                location="ole:xls/{}!R{}C{}".format(
                                    sheet.name, row_idx + 1, col_idx + 1,
                                ),
                                text=str(cell.value),
                            )
                        )

    @staticmethod
    def _extract_raw_strings(
        ole: Any,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Fallback: extract printable UTF-16LE strings (>= 20 chars) from all streams.

        Targeted at .doc body text which is too complex to parse properly.
        Only runs if no dedicated parser produced body-level text.
        """
        # Only do raw extraction for streams likely to hold document body text.
        # Skip if we already have PPT slide text or XLS cell values.
        target_streams = []
        for parts in ole.listdir(streams=True, storages=False):
            joined = "/".join(parts)
            # Focus on main content streams
            if joined in ("WordDocument", "1Table", "0Table"):
                target_streams.append(joined)

        if not target_streams:
            return

        min_length = 20
        utf16_pattern = re.compile(
            rb"(?:[\x20-\x7e]\x00){" + str(min_length).encode() + rb",}"
        )

        collected: List[str] = []
        for stream_name in target_streams:
            try:
                stream_data = ole.openstream(stream_name).read()
            except Exception:
                continue

            for match in utf16_pattern.finditer(stream_data):
                text = match.group().decode("utf-16-le", errors="replace")
                if text.strip():
                    collected.append(text)

        if collected:
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique: List[str] = []
            for t in collected:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)

            artifacts.append(
                ExtractedArtifact(
                    location="ole:raw_strings",
                    text="\n".join(unique),
                    metadata={
                        "note": "Unstructured UTF-16LE string extraction (fallback)",
                        "string_count": str(len(unique)),
                    },
                )
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_meta_value(value: Any) -> str:
    """Decode an OLE metadata value to a string, handling bytes."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1", errors="replace")
    if isinstance(value, str):
        return value
    if value is not None:
        return str(value)
    return ""
