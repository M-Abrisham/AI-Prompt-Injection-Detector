"""XLSX deep extractor -- extracts text from all user-controllable surfaces.

Parses XLSX (Office Open XML Spreadsheet) files using only stdlib
``zipfile`` and ``xml.etree.ElementTree``.  Extracts text from:

- Shared strings (``xl/sharedStrings.xml``)
- Inline strings in sheet cells (``<is>/<t>``)
- Cell comments (``xl/comments*.xml``)
- Hidden and very-hidden sheets (full content extraction)
- Defined names / named ranges
- Cell formulas (``<f>`` elements)
- Data validation messages (prompt / error text)
- Header/footer strings
- Core properties (``docProps/core.xml``)
- Custom properties (``docProps/custom.xml``)
- Hyperlink display text and tooltips
"""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, List, Optional

from na0s.parsers.office.base import (
    ExtractedArtifact,
    OfficeExtractor,
    safe_read_zip_entry,
    validate_zip_safety,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML namespaces
# ---------------------------------------------------------------------------

_NS_SPREADSHEET = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS_DC = "http://purl.org/dc/elements/1.1/"
_NS_CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
_NS_CUSTOM = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
_NS_VT = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"

_NSMAP = {
    "s": _NS_SPREADSHEET,
    "r": _NS_REL,
    "rel": _NS_RELS,
    "dc": _NS_DC,
    "cp": _NS_CP,
    "cust": _NS_CUSTOM,
    "vt": _NS_VT,
}


def _tag(ns_prefix: str, local: str) -> str:
    """Build a fully-qualified ``{namespace}local`` tag string."""
    return "{%s}%s" % (_NSMAP[ns_prefix], local)


def _parse_xml_safe(raw: bytes) -> Optional[ET.Element]:
    """Parse XML bytes, returning None on any malformed-XML error."""
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        logger.debug("xlsx_extractor: XML parse error: %s", exc)
        return None


def _collect_text(element: ET.Element) -> str:
    """Recursively collect all text from <t> elements within *element*.

    Handles both plain ``<si><t>text</t></si>`` and rich-text
    ``<si><r><t>part</t></r><r><t>part</t></r></si>`` structures.
    """
    parts: list[str] = []
    for t_el in element.iter(_tag("s", "t")):
        if t_el.text:
            parts.append(t_el.text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Sheet metadata
# ---------------------------------------------------------------------------

class _SheetInfo:
    """Metadata for a single worksheet parsed from workbook.xml + rels."""

    __slots__ = ("name", "sheet_id", "r_id", "state", "xml_path", "index")

    def __init__(
        self,
        name: str,
        sheet_id: str,
        r_id: str,
        state: str,
        xml_path: str,
        index: int,
    ) -> None:
        self.name = name
        self.sheet_id = sheet_id
        self.r_id = r_id
        self.state = state  # "visible", "hidden", or "veryHidden"
        self.xml_path = xml_path  # e.g. "xl/worksheets/sheet1.xml"
        self.index = index  # 1-based sheet number for location tags

    @property
    def location_prefix(self) -> str:
        """Build location prefix like ``xlsx:sheet2[hidden]``."""
        tag = "xlsx:sheet%d" % self.index
        if self.state != "visible":
            tag += "[%s]" % self.state
        return tag


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class Extractor(OfficeExtractor):
    """Extract all user-controllable text surfaces from an XLSX file."""

    @property
    def format_name(self) -> str:
        return "xlsx"

    # -- public API --------------------------------------------------------

    def extract(self, data: bytes) -> List[ExtractedArtifact]:
        artifacts: list[ExtractedArtifact] = []

        try:
            zf = zipfile.ZipFile(io.BytesIO(data), "r")
        except (zipfile.BadZipFile, Exception) as exc:
            logger.warning("xlsx_extractor: not a valid ZIP: %s", exc)
            return artifacts

        with zf:
            # Zip-bomb safety check
            warnings = validate_zip_safety(zf)
            for w in warnings:
                logger.warning("xlsx_extractor: %s", w)
            if warnings:
                return artifacts

            # Parse workbook structure
            sheets = self._parse_workbook(zf)
            shared_strings = self._parse_shared_strings(zf)

            # Emit shared strings as standalone artifacts
            self._emit_shared_strings(shared_strings, artifacts)

            # Per-sheet extraction
            for sheet in sheets:
                self._extract_sheet(zf, sheet, shared_strings, artifacts)

            # Workbook-level extraction
            self._extract_defined_names(zf, artifacts)
            self._extract_comments(zf, sheets, artifacts)
            self._extract_core_properties(zf, artifacts)
            self._extract_custom_properties(zf, artifacts)

        return artifacts

    # -- workbook structure ------------------------------------------------

    def _parse_workbook(self, zf: zipfile.ZipFile) -> list[_SheetInfo]:
        """Parse xl/workbook.xml and its rels to build sheet metadata."""
        sheets: list[_SheetInfo] = []

        raw = safe_read_zip_entry(zf, "xl/workbook.xml")
        if raw is None:
            return sheets
        root = _parse_xml_safe(raw)
        if root is None:
            return sheets

        # Build relationship-id -> target mapping from rels
        rid_map = self._parse_rels(zf, "xl/_rels/workbook.xml.rels")

        sheets_el = root.find(_tag("s", "sheets"))
        if sheets_el is None:
            return sheets

        for idx, sheet_el in enumerate(
            sheets_el.findall(_tag("s", "sheet")), start=1
        ):
            name = sheet_el.get("name", "Sheet%d" % idx)
            sheet_id = sheet_el.get("sheetId", str(idx))
            r_id = sheet_el.get(_tag("r", "id"), "")
            state = sheet_el.get("state", "visible")

            # Resolve r:id to the actual XML path
            target = rid_map.get(r_id, "")
            if target and not target.startswith("xl/"):
                # Relative paths in rels are relative to the source part
                xml_path = "xl/" + target.lstrip("/")
            else:
                xml_path = target

            sheets.append(
                _SheetInfo(
                    name=name,
                    sheet_id=sheet_id,
                    r_id=r_id,
                    state=state,
                    xml_path=xml_path,
                    index=idx,
                )
            )

        return sheets

    def _parse_rels(
        self, zf: zipfile.ZipFile, rels_path: str
    ) -> Dict[str, str]:
        """Parse a ``.rels`` file into {rId -> Target} mapping."""
        rid_map: dict[str, str] = {}
        raw = safe_read_zip_entry(zf, rels_path)
        if raw is None:
            return rid_map
        root = _parse_xml_safe(raw)
        if root is None:
            return rid_map

        for rel in root.iter(_tag("rel", "Relationship")):
            rid = rel.get("Id", "")
            target = rel.get("Target", "")
            if rid:
                rid_map[rid] = target
        return rid_map

    # -- shared strings ----------------------------------------------------

    def _parse_shared_strings(self, zf: zipfile.ZipFile) -> list[str]:
        """Parse the shared string table, returning indexed list of strings."""
        strings: list[str] = []
        raw = safe_read_zip_entry(zf, "xl/sharedStrings.xml")
        if raw is None:
            return strings
        root = _parse_xml_safe(raw)
        if root is None:
            return strings

        for si in root.findall(_tag("s", "si")):
            strings.append(_collect_text(si))

        return strings

    def _emit_shared_strings(
        self,
        shared_strings: list[str],
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Emit an artifact for each non-empty shared string."""
        for idx, text in enumerate(shared_strings):
            if text.strip():
                artifacts.append(
                    ExtractedArtifact(
                        location="xlsx:sharedStrings[%d]" % idx,
                        text=text,
                    )
                )

    # -- per-sheet extraction ----------------------------------------------

    def _extract_sheet(
        self,
        zf: zipfile.ZipFile,
        sheet: _SheetInfo,
        shared_strings: list[str],
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Extract all text surfaces from a single worksheet XML."""
        if not sheet.xml_path:
            return
        raw = safe_read_zip_entry(zf, sheet.xml_path)
        if raw is None:
            return
        root = _parse_xml_safe(raw)
        if root is None:
            return

        prefix = sheet.location_prefix

        # --- Cells: shared string refs, inline strings, formulas ----------
        for row_el in root.iter(_tag("s", "row")):
            for c_el in row_el.iter(_tag("s", "c")):
                cell_ref = c_el.get("r", "?")
                cell_type = c_el.get("t", "")

                # Inline strings
                if cell_type == "inlineStr":
                    is_el = c_el.find(_tag("s", "is"))
                    if is_el is not None:
                        text = _collect_text(is_el)
                        if text.strip():
                            artifacts.append(
                                ExtractedArtifact(
                                    location="%s/%s" % (prefix, cell_ref),
                                    text=text,
                                    metadata={"type": "inlineStr"},
                                )
                            )

                # Shared string reference
                elif cell_type == "s":
                    v_el = c_el.find(_tag("s", "v"))
                    if v_el is not None and v_el.text is not None:
                        try:
                            ss_idx = int(v_el.text)
                        except (ValueError, TypeError):
                            ss_idx = -1
                        if 0 <= ss_idx < len(shared_strings):
                            text = shared_strings[ss_idx]
                            if text.strip():
                                artifacts.append(
                                    ExtractedArtifact(
                                        location="%s/%s" % (prefix, cell_ref),
                                        text=text,
                                        metadata={
                                            "type": "sharedString",
                                            "ssi": str(ss_idx),
                                        },
                                    )
                                )

                # Formula (check regardless of cell type)
                f_el = c_el.find(_tag("s", "f"))
                if f_el is not None and f_el.text and f_el.text.strip():
                    artifacts.append(
                        ExtractedArtifact(
                            location="%s/%s" % (prefix, cell_ref),
                            text=f_el.text,
                            metadata={"type": "formula"},
                        )
                    )

        # --- Data validation messages -------------------------------------
        for dv_el in root.iter(_tag("s", "dataValidation")):
            sqref = dv_el.get("sqref", "?")
            for attr_name in ("prompt", "promptTitle", "error", "errorTitle"):
                val = dv_el.get(attr_name, "")
                if val.strip():
                    artifacts.append(
                        ExtractedArtifact(
                            location="%s/dataValidation[%s]" % (prefix, sqref),
                            text=val,
                            metadata={"field": attr_name},
                        )
                    )
            # Also extract formula1/formula2 inside dataValidation
            for ftag in ("formula1", "formula2"):
                f_el = dv_el.find(_tag("s", ftag))
                if f_el is not None and f_el.text and f_el.text.strip():
                    artifacts.append(
                        ExtractedArtifact(
                            location="%s/dataValidation[%s]" % (prefix, sqref),
                            text=f_el.text,
                            metadata={"field": ftag},
                        )
                    )

        # --- Header / footer strings --------------------------------------
        hf_el = root.find(_tag("s", "headerFooter"))
        if hf_el is not None:
            for child_tag in (
                "oddHeader", "oddFooter",
                "evenHeader", "evenFooter",
                "firstHeader", "firstFooter",
            ):
                child = hf_el.find(_tag("s", child_tag))
                if child is not None and child.text and child.text.strip():
                    artifacts.append(
                        ExtractedArtifact(
                            location="%s/headerFooter/%s" % (prefix, child_tag),
                            text=child.text,
                        )
                    )

        # --- Hyperlinks ---------------------------------------------------
        self._extract_hyperlinks(zf, sheet, root, artifacts)

    # -- hyperlinks --------------------------------------------------------

    def _extract_hyperlinks(
        self,
        zf: zipfile.ZipFile,
        sheet: _SheetInfo,
        sheet_root: ET.Element,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Extract hyperlink display text, tooltips, and target URLs."""
        prefix = sheet.location_prefix

        # Build rels map for this sheet to resolve hyperlink targets
        # e.g. xl/worksheets/_rels/sheet1.xml.rels
        rid_map: dict[str, str] = {}
        if sheet.xml_path:
            parts = sheet.xml_path.rsplit("/", 1)
            if len(parts) == 2:
                rels_path = "%s/_rels/%s.rels" % (parts[0], parts[1])
            else:
                rels_path = "_rels/%s.rels" % parts[0]
            rid_map = self._parse_rels(zf, rels_path)

        for hl_el in sheet_root.iter(_tag("s", "hyperlink")):
            cell_ref = hl_el.get("ref", "?")
            display = hl_el.get("display", "")
            tooltip = hl_el.get("tooltip", "")
            r_id = hl_el.get(_tag("r", "id"), "")
            target = rid_map.get(r_id, "") if r_id else ""

            texts: list[str] = []
            meta: dict[str, str] = {}

            if display.strip():
                texts.append(display)
                meta["display"] = display
            if tooltip.strip():
                texts.append(tooltip)
                meta["tooltip"] = tooltip
            if target.strip():
                meta["target"] = target

            if texts or target.strip():
                artifacts.append(
                    ExtractedArtifact(
                        location="%s/hyperlink[%s]" % (prefix, cell_ref),
                        text=" | ".join(texts) if texts else target,
                        metadata=meta,
                    )
                )

    # -- comments ----------------------------------------------------------

    def _extract_comments(
        self,
        zf: zipfile.ZipFile,
        sheets: list[_SheetInfo],
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Extract cell comments from xl/comments*.xml files.

        Comments are linked to sheets via sheet rels.  We check each
        sheet's rels for comment relationships, then fall back to
        scanning for unlinked comment files.
        """
        comment_files_seen: set[str] = set()

        for sheet in sheets:
            if not sheet.xml_path:
                continue
            parts = sheet.xml_path.rsplit("/", 1)
            if len(parts) == 2:
                rels_path = "%s/_rels/%s.rels" % (parts[0], parts[1])
            else:
                rels_path = "_rels/%s.rels" % parts[0]

            rid_map = self._parse_rels(zf, rels_path)
            for _rid, target in rid_map.items():
                if "comment" not in target.lower():
                    continue
                # Resolve relative path
                if target.startswith("/"):
                    comment_path = target.lstrip("/")
                elif target.startswith(".."):
                    # From xl/worksheets/ up one level -> xl/
                    comment_path = "xl/" + target.split("../", 1)[-1]
                elif not target.startswith("xl/"):
                    base = parts[0] if len(parts) == 2 else ""
                    comment_path = (base + "/" + target) if base else target
                else:
                    comment_path = target

                if comment_path in comment_files_seen:
                    continue
                comment_files_seen.add(comment_path)
                self._parse_comment_file(zf, comment_path, sheet, artifacts)

        # Fallback: scan for any comment files not yet processed
        for entry_name in zf.namelist():
            if (
                entry_name.startswith("xl/")
                and "comment" in entry_name.lower()
                and entry_name.endswith(".xml")
                and entry_name not in comment_files_seen
            ):
                comment_files_seen.add(entry_name)
                fallback_sheet = sheets[0] if sheets else None
                m = re.search(r"comments?(\d+)", entry_name)
                if m:
                    idx = int(m.group(1))
                    if 1 <= idx <= len(sheets):
                        fallback_sheet = sheets[idx - 1]
                if fallback_sheet:
                    self._parse_comment_file(
                        zf, entry_name, fallback_sheet, artifacts
                    )

    def _parse_comment_file(
        self,
        zf: zipfile.ZipFile,
        path: str,
        sheet: _SheetInfo,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Parse a single xl/comments*.xml file."""
        raw = safe_read_zip_entry(zf, path)
        if raw is None:
            return
        root = _parse_xml_safe(raw)
        if root is None:
            return

        # Extract author names
        authors: list[str] = []
        authors_el = root.find(_tag("s", "authors"))
        if authors_el is not None:
            for author_el in authors_el.findall(_tag("s", "author")):
                authors.append(author_el.text or "")

        # Extract comment text
        for comment_el in root.iter(_tag("s", "comment")):
            cell_ref = comment_el.get("ref", "?")
            text_el = comment_el.find(_tag("s", "text"))
            if text_el is None:
                continue
            text = _collect_text(text_el)
            if not text.strip():
                continue

            meta: dict[str, str] = {}
            author_id_str = comment_el.get("authorId", "")
            if author_id_str:
                try:
                    author_idx = int(author_id_str)
                    if 0 <= author_idx < len(authors):
                        meta["author"] = authors[author_idx]
                except (ValueError, TypeError):
                    pass

            artifacts.append(
                ExtractedArtifact(
                    location="xlsx:comments/sheet%d[%s]" % (
                        sheet.index, cell_ref
                    ),
                    text=text,
                    metadata=meta,
                )
            )

        # Also emit author names as artifacts (injectable text)
        for i, author in enumerate(authors):
            if author.strip():
                artifacts.append(
                    ExtractedArtifact(
                        location="xlsx:comments/sheet%d/author[%d]" % (
                            sheet.index, i
                        ),
                        text=author,
                        metadata={"type": "commentAuthor"},
                    )
                )

    # -- defined names -----------------------------------------------------

    def _extract_defined_names(
        self,
        zf: zipfile.ZipFile,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Extract defined names from xl/workbook.xml."""
        raw = safe_read_zip_entry(zf, "xl/workbook.xml")
        if raw is None:
            return
        root = _parse_xml_safe(raw)
        if root is None:
            return

        dn_container = root.find(_tag("s", "definedNames"))
        if dn_container is None:
            return

        for dn in dn_container.findall(_tag("s", "definedName")):
            name = dn.get("name", "?")
            hidden = dn.get("hidden", "0")
            text = dn.text or ""
            if not text.strip():
                continue

            meta: dict[str, str] = {"name": name}
            if hidden == "1":
                meta["hidden"] = "true"

            artifacts.append(
                ExtractedArtifact(
                    location="xlsx:definedNames/%s" % name,
                    text=text,
                    metadata=meta,
                )
            )

    # -- core properties ---------------------------------------------------

    def _extract_core_properties(
        self,
        zf: zipfile.ZipFile,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Extract Dublin Core / OPC metadata from docProps/core.xml."""
        raw = safe_read_zip_entry(zf, "docProps/core.xml")
        if raw is None:
            return
        root = _parse_xml_safe(raw)
        if root is None:
            return

        # Dublin Core elements
        dc_fields = ("title", "subject", "creator", "description", "language")
        for field_name in dc_fields:
            el = root.find("{%s}%s" % (_NS_DC, field_name))
            if el is not None and el.text and el.text.strip():
                artifacts.append(
                    ExtractedArtifact(
                        location="xlsx:coreProperties/%s" % field_name,
                        text=el.text,
                    )
                )

        # OPC core-properties namespace fields
        cp_fields = ("keywords", "category", "lastModifiedBy")
        for field_name in cp_fields:
            el = root.find("{%s}%s" % (_NS_CP, field_name))
            if el is not None and el.text and el.text.strip():
                artifacts.append(
                    ExtractedArtifact(
                        location="xlsx:coreProperties/%s" % field_name,
                        text=el.text,
                    )
                )

    # -- custom properties -------------------------------------------------

    def _extract_custom_properties(
        self,
        zf: zipfile.ZipFile,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Extract key-value pairs from docProps/custom.xml."""
        raw = safe_read_zip_entry(zf, "docProps/custom.xml")
        if raw is None:
            return
        root = _parse_xml_safe(raw)
        if root is None:
            return

        for prop in root.findall("{%s}property" % _NS_CUSTOM):
            prop_name = prop.get("name", "?")
            # Value can be in various vt: sub-elements
            value_text = ""
            for child in prop:
                if child.text and child.text.strip():
                    value_text = child.text
                    break

            if value_text.strip():
                artifacts.append(
                    ExtractedArtifact(
                        location="xlsx:customProperties/%s" % prop_name,
                        text=value_text,
                        metadata={"name": prop_name},
                    )
                )
