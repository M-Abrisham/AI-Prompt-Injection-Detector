"""ODF deep extractor for ODT, ODS, and ODP documents.

Extracts text from all user-controllable surfaces including body text,
tracked changes, annotations, hidden text, metadata, text boxes,
footnotes/endnotes, headers/footers, scripts, spreadsheet cells,
hidden sheets, named ranges, slide text, speaker notes, and hidden slides.

Uses only stdlib ``zipfile`` and ``xml.etree.ElementTree``.
"""

from __future__ import annotations

import io
import logging
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
# ODF XML namespaces
# ---------------------------------------------------------------------------

NS: Dict[str, str] = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "presentation": "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
    "meta": "urn:oasis:names:tc:opendocument:xmlns:meta:1.0",
    "dc": "http://purl.org/dc/elements/1.1/",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
    "xlink": "http://www.w3.org/1999/xlink",
    "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
    "form": "urn:oasis:names:tc:opendocument:xmlns:form:1.0",
}

# Register namespace prefixes so ET doesn't invent ns0/ns1/... aliases.
for _prefix, _uri in NS.items():
    ET.register_namespace(_prefix, _uri)


def _tag(prefix: str, local: str) -> str:
    """Build a Clark-notation tag from a namespace prefix and local name."""
    return "{%s}%s" % (NS[prefix], local)


def _collect_text(elem: Optional[ET.Element]) -> str:
    """Recursively collect all text content under *elem*.

    Joins ``elem.text`` and ``tail`` from all descendants, returning a
    single stripped string.  Returns ``""`` for ``None`` input.
    """
    if elem is None:
        return ""
    parts: list[str] = []
    for node in elem.iter():
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    return " ".join(parts).strip()


def _collect_text_no_tail(
    elem: Optional[ET.Element],
    skip_tags: Optional[set[str]] = None,
) -> str:
    """Like ``_collect_text`` but skips the top-level element's tail.

    Useful when extracting text from an annotation or note that is
    inline within a paragraph -- we don't want the surrounding paragraph
    text leaking in via ``tail``.

    If *skip_tags* is provided, subtrees rooted at elements with those
    tags are excluded (but their ``tail`` text is still collected since
    it belongs to the parent context).
    """
    if elem is None:
        return ""
    parts: list[str] = []
    _walk_text(elem, parts, skip_tags, is_root=True)
    return " ".join(parts).strip()


def _walk_text(
    elem: ET.Element,
    parts: list[str],
    skip_tags: Optional[set[str]],
    is_root: bool = False,
) -> None:
    """Recursive text walker with subtree-skip support."""
    if not is_root and skip_tags and elem.tag in skip_tags:
        # Skip the subtree but keep tail (tail belongs to parent)
        if elem.tail:
            parts.append(elem.tail)
        return
    if is_root:
        # Root element: collect text but not tail
        if elem.text:
            parts.append(elem.text)
    else:
        if elem.text:
            parts.append(elem.text)
        if elem.tail:
            parts.append(elem.tail)
    for child in elem:
        _walk_text(child, parts, skip_tags, is_root=False)


# Tags whose subtrees should be excluded when collecting body paragraph
# text (their content is extracted separately as annotations, notes, etc.)
_INLINE_SKIP_TAGS: set[str] = {
    _tag("office", "annotation"),
    _tag("text", "note"),
    _tag("text", "hidden-text"),
    _tag("text", "hidden-paragraph"),
}


def _safe_parse_xml(raw: Optional[bytes]) -> Optional[ET.Element]:
    """Parse XML bytes, returning the root element or ``None`` on failure."""
    if raw is None:
        return None
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        logger.debug("odf extractor: XML parse error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Sub-format detection
# ---------------------------------------------------------------------------

_MIME_TO_FORMAT: Dict[str, str] = {
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
    # Template variants map to the same extractors.
    "application/vnd.oasis.opendocument.text-template": "odt",
    "application/vnd.oasis.opendocument.spreadsheet-template": "ods",
    "application/vnd.oasis.opendocument.presentation-template": "odp",
}


def _detect_subformat(
    zf: zipfile.ZipFile, content_root: Optional[ET.Element],
) -> str:
    """Detect odt/ods/odp from the ZIP's ``mimetype`` file or XML namespaces."""
    mime_bytes = safe_read_zip_entry(zf, "mimetype", max_bytes=256)
    if mime_bytes is not None:
        mime = mime_bytes.decode("ascii", errors="replace").strip()
        fmt = _MIME_TO_FORMAT.get(mime)
        if fmt:
            return fmt

    # Fallback: inspect content.xml body element
    if content_root is not None:
        body = content_root.find(_tag("office", "body"))
        if body is not None:
            if body.find(_tag("office", "spreadsheet")) is not None:
                return "ods"
            if body.find(_tag("office", "presentation")) is not None:
                return "odp"

    return "odt"  # default to text


# ---------------------------------------------------------------------------
# Column-index helper
# ---------------------------------------------------------------------------

def _col_letter(index: int) -> str:
    """Convert a 0-based column index to a spreadsheet column letter (A, B, ..., Z, AA, ...)."""
    result = ""
    i = index
    while True:
        result = chr(65 + i % 26) + result
        i = i // 26 - 1
        if i < 0:
            break
    return result


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor(OfficeExtractor):
    """Deep extractor for ODF documents (ODT, ODS, ODP)."""

    @property
    def format_name(self) -> str:
        return "odf"

    def extract(self, data: bytes) -> List[ExtractedArtifact]:
        """Extract all text artifacts from an ODF document."""
        artifacts: List[ExtractedArtifact] = []

        try:
            zf = zipfile.ZipFile(io.BytesIO(data), "r")
        except (zipfile.BadZipFile, Exception) as exc:
            logger.warning("odf extractor: cannot open ZIP: %s", exc)
            return artifacts

        with zf:
            # Zip-bomb safety
            warnings = validate_zip_safety(zf)
            if warnings:
                for w in warnings:
                    logger.warning("odf extractor: %s", w)
                return artifacts

            # Parse key XML files
            content_root = _safe_parse_xml(
                safe_read_zip_entry(zf, "content.xml"),
            )
            meta_root = _safe_parse_xml(
                safe_read_zip_entry(zf, "meta.xml"),
            )
            styles_root = _safe_parse_xml(
                safe_read_zip_entry(zf, "styles.xml"),
            )

            if content_root is None:
                logger.debug("odf extractor: no content.xml found")
                return artifacts

            fmt = _detect_subformat(zf, content_root)

            # --- Shared extractions (all sub-formats) ---
            self._extract_metadata(meta_root, fmt, artifacts)
            self._extract_scripts(content_root, fmt, artifacts)

            # --- Format-specific extractions ---
            if fmt == "odt":
                # ODT: global annotation pass (annotations inline in body)
                self._extract_annotations_global(content_root, fmt, artifacts)
                self._extract_odt(content_root, styles_root, artifacts)
            elif fmt == "ods":
                # ODS: sheet-specific annotation pass (skips global to
                # avoid duplicates; sheet context is more informative)
                self._extract_ods(content_root, artifacts)
            elif fmt == "odp":
                self._extract_annotations_global(content_root, fmt, artifacts)
                self._extract_odp(content_root, artifacts)

        return artifacts

    # ------------------------------------------------------------------
    # Metadata (shared across all ODF sub-formats)
    # ------------------------------------------------------------------

    def _extract_metadata(
        self,
        meta_root: Optional[ET.Element],
        fmt: str,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        if meta_root is None:
            return

        office_meta = meta_root.find(_tag("office", "meta"))
        if office_meta is None:
            # Some files have meta elements directly under root
            office_meta = meta_root

        dc_fields = {
            "dc:title": _tag("dc", "title"),
            "dc:description": _tag("dc", "description"),
            "dc:subject": _tag("dc", "subject"),
            "dc:creator": _tag("dc", "creator"),
        }
        for label, tag in dc_fields.items():
            elem = office_meta.find(tag)
            if elem is not None:
                text = (elem.text or "").strip()
                if text:
                    artifacts.append(ExtractedArtifact(
                        location="%s:meta/%s" % (fmt, label),
                        text=text,
                    ))

        # meta:keyword (can appear multiple times)
        for i, kw in enumerate(
            office_meta.findall(_tag("meta", "keyword")), start=1,
        ):
            text = (kw.text or "").strip()
            if text:
                artifacts.append(ExtractedArtifact(
                    location="%s:meta/keyword[%d]" % (fmt, i),
                    text=text,
                ))

        # meta:initial-creator
        ic = office_meta.find(_tag("meta", "initial-creator"))
        if ic is not None:
            text = (ic.text or "").strip()
            if text:
                artifacts.append(ExtractedArtifact(
                    location="%s:meta/initial-creator" % fmt,
                    text=text,
                ))

        # User-defined metadata
        for i, ud in enumerate(
            office_meta.findall(_tag("meta", "user-defined")), start=1,
        ):
            name = ud.get(_tag("meta", "name"), "unknown")
            text = (ud.text or "").strip()
            if text:
                artifacts.append(ExtractedArtifact(
                    location="%s:meta/user-defined[%s]" % (fmt, name),
                    text=text,
                    metadata={"name": name},
                ))

    # ------------------------------------------------------------------
    # Scripts (shared)
    # ------------------------------------------------------------------

    def _extract_scripts(
        self,
        content_root: ET.Element,
        fmt: str,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        scripts_elem = content_root.find(_tag("office", "scripts"))
        if scripts_elem is None:
            return
        for i, script in enumerate(
            scripts_elem.findall(_tag("office", "script")), start=1,
        ):
            lang = script.get(_tag("office", "language"), "unknown")
            text = _collect_text_no_tail(script)
            if text:
                artifacts.append(ExtractedArtifact(
                    location="%s:script[%d]" % (fmt, i),
                    text=text,
                    metadata={"language": lang},
                ))

    # ------------------------------------------------------------------
    # Annotations found anywhere in the content tree (shared)
    # ------------------------------------------------------------------

    # Tags to skip when collecting annotation body text (metadata, not content).
    _ANNOTATION_SKIP_TAGS: set[str] = {
        _tag("dc", "creator"),
        _tag("dc", "date"),
    }

    def _extract_annotations_global(
        self,
        content_root: ET.Element,
        fmt: str,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        for i, ann in enumerate(
            content_root.iter(_tag("office", "annotation")), start=1,
        ):
            text = _collect_text_no_tail(ann, self._ANNOTATION_SKIP_TAGS)
            creator_el = ann.find(_tag("dc", "creator"))
            creator = (creator_el.text or "").strip() if creator_el is not None else ""
            meta: Dict[str, str] = {}
            if creator:
                meta["author"] = creator
            if text:
                artifacts.append(ExtractedArtifact(
                    location="%s:annotation[%d]" % (fmt, i),
                    text=text,
                    metadata=meta,
                ))

    # ==================================================================
    # ODT-specific extraction
    # ==================================================================

    def _extract_odt(
        self,
        content_root: ET.Element,
        styles_root: Optional[ET.Element],
        artifacts: List[ExtractedArtifact],
    ) -> None:
        body = content_root.find(_tag("office", "body"))
        if body is None:
            return
        text_body = body.find(_tag("office", "text"))
        if text_body is None:
            return

        self._extract_odt_body_text(text_body, artifacts)
        self._extract_odt_tracked_changes(text_body, artifacts)
        self._extract_odt_hidden_text(text_body, artifacts)
        self._extract_odt_text_boxes(text_body, artifacts)
        self._extract_odt_footnotes(text_body, artifacts)
        self._extract_odt_headers_footers(styles_root, artifacts)

    def _extract_odt_body_text(
        self,
        text_body: ET.Element,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract paragraphs and headings from ODT body."""
        p_idx = 0
        h_idx = 0
        for child in text_body:
            tag = child.tag
            if tag == _tag("text", "p"):
                p_idx += 1
                text = _collect_text_no_tail(child, _INLINE_SKIP_TAGS)
                if text:
                    artifacts.append(ExtractedArtifact(
                        location="odt:body/p[%d]" % p_idx,
                        text=text,
                    ))
            elif tag == _tag("text", "h"):
                h_idx += 1
                text = _collect_text_no_tail(child, _INLINE_SKIP_TAGS)
                if text:
                    artifacts.append(ExtractedArtifact(
                        location="odt:body/h[%d]" % h_idx,
                        text=text,
                    ))
            elif tag == _tag("text", "section"):
                self._extract_odt_section(child, artifacts)

    def _extract_odt_section(
        self,
        section: ET.Element,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract text from sections, noting hidden ones."""
        display = section.get(_tag("text", "display"), "true")
        name = section.get(_tag("text", "name"), "")
        hidden = display in ("none", "condition")
        label_prefix = "odt:section[%s%s]" % (
            name,
            "/hidden" if hidden else "",
        )
        for i, p in enumerate(section.findall(_tag("text", "p")), start=1):
            text = _collect_text_no_tail(p)
            if text:
                artifacts.append(ExtractedArtifact(
                    location="%s/p[%d]" % (label_prefix, i),
                    text=text,
                    metadata={"hidden": str(hidden)},
                ))

    def _extract_odt_tracked_changes(
        self,
        text_body: ET.Element,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract text from tracked changes (insertions and deletions)."""
        tc = text_body.find(_tag("text", "tracked-changes"))
        if tc is None:
            return
        ins_idx = 0
        del_idx = 0
        for region in tc.findall(_tag("text", "changed-region")):
            # Insertions
            for insertion in region.findall(_tag("text", "insertion")):
                ins_idx += 1
                text = _collect_text_no_tail(insertion)
                if text:
                    artifacts.append(ExtractedArtifact(
                        location="odt:tracked-changes/insertion[%d]" % ins_idx,
                        text=text,
                    ))
            # Deletions
            for deletion in region.findall(_tag("text", "deletion")):
                del_idx += 1
                text = _collect_text_no_tail(deletion)
                if text:
                    artifacts.append(ExtractedArtifact(
                        location="odt:tracked-changes/deletion[%d]" % del_idx,
                        text=text,
                    ))

    def _extract_odt_hidden_text(
        self,
        text_body: ET.Element,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract hidden-text fields and hidden-paragraph elements."""
        for i, ht in enumerate(
            text_body.iter(_tag("text", "hidden-text")), start=1,
        ):
            val = ht.get(_tag("text", "string-value"), "")
            text = val.strip() or _collect_text_no_tail(ht)
            if text:
                artifacts.append(ExtractedArtifact(
                    location="odt:hidden-text[%d]" % i,
                    text=text,
                ))

        for i, hp in enumerate(
            text_body.iter(_tag("text", "hidden-paragraph")), start=1,
        ):
            text = _collect_text_no_tail(hp)
            if text:
                artifacts.append(ExtractedArtifact(
                    location="odt:hidden-paragraph[%d]" % i,
                    text=text,
                ))

    def _extract_odt_text_boxes(
        self,
        text_body: ET.Element,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract text from draw:text-box elements."""
        for i, tb in enumerate(
            text_body.iter(_tag("draw", "text-box")), start=1,
        ):
            text = _collect_text_no_tail(tb)
            if text:
                artifacts.append(ExtractedArtifact(
                    location="odt:text-box[%d]" % i,
                    text=text,
                ))

    def _extract_odt_footnotes(
        self,
        text_body: ET.Element,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract text from footnotes and endnotes."""
        fn_idx = 0
        en_idx = 0
        for note in text_body.iter(_tag("text", "note")):
            note_class = note.get(_tag("text", "note-class"), "footnote")
            note_body = note.find(_tag("text", "note-body"))
            text = _collect_text_no_tail(note_body) if note_body is not None else ""
            if text:
                if note_class == "endnote":
                    en_idx += 1
                    loc = "odt:endnote[%d]" % en_idx
                else:
                    fn_idx += 1
                    loc = "odt:footnote[%d]" % fn_idx
                artifacts.append(ExtractedArtifact(
                    location=loc,
                    text=text,
                ))

    def _extract_odt_headers_footers(
        self,
        styles_root: Optional[ET.Element],
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract text from headers and footers in styles.xml."""
        if styles_root is None:
            return

        header_tags = [
            _tag("style", "header"),
            _tag("style", "header-left"),
            _tag("style", "header-first"),
        ]
        footer_tags = [
            _tag("style", "footer"),
            _tag("style", "footer-left"),
            _tag("style", "footer-first"),
        ]

        hdr_idx = 0
        ftr_idx = 0
        for master_page in styles_root.iter(_tag("style", "master-page")):
            page_name = master_page.get(
                _tag("style", "name"), "unknown",
            )
            for htag in header_tags:
                for hdr in master_page.findall(htag):
                    text = _collect_text_no_tail(hdr)
                    if text:
                        hdr_idx += 1
                        artifacts.append(ExtractedArtifact(
                            location="odt:header[%d]" % hdr_idx,
                            text=text,
                            metadata={"master-page": page_name},
                        ))
            for ftag in footer_tags:
                for ftr in master_page.findall(ftag):
                    text = _collect_text_no_tail(ftr)
                    if text:
                        ftr_idx += 1
                        artifacts.append(ExtractedArtifact(
                            location="odt:footer[%d]" % ftr_idx,
                            text=text,
                            metadata={"master-page": page_name},
                        ))

    # ==================================================================
    # ODS-specific extraction
    # ==================================================================

    def _extract_ods(
        self,
        content_root: ET.Element,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        body = content_root.find(_tag("office", "body"))
        if body is None:
            return
        spreadsheet = body.find(_tag("office", "spreadsheet"))
        if spreadsheet is None:
            return

        # Build a set of hidden table style names from automatic styles
        hidden_styles = self._find_hidden_table_styles(content_root)

        for sheet_idx, table in enumerate(
            spreadsheet.findall(_tag("table", "table")), start=1,
        ):
            sheet_name = table.get(_tag("table", "name"), "Sheet%d" % sheet_idx)
            style_name = table.get(_tag("table", "style-name"), "")
            is_hidden = style_name in hidden_styles

            sheet_label = "%s%s" % (
                sheet_name,
                "[hidden]" if is_hidden else "",
            )

            self._extract_ods_cells(table, sheet_label, artifacts)
            self._extract_ods_annotations(table, sheet_label, artifacts)

        # Named ranges / named expressions
        self._extract_ods_named_ranges(spreadsheet, artifacts)

    def _find_hidden_table_styles(
        self, content_root: ET.Element,
    ) -> set[str]:
        """Return a set of style names whose table-properties have display=false."""
        hidden: set[str] = set()
        auto_styles = content_root.find(_tag("office", "automatic-styles"))
        if auto_styles is None:
            return hidden
        for style in auto_styles.findall(_tag("style", "style")):
            family = style.get(_tag("style", "family"), "")
            if family != "table":
                continue
            tp = style.find(_tag("style", "table-properties"))
            if tp is not None:
                display = tp.get(_tag("table", "display"), "true")
                if display == "false":
                    sname = style.get(_tag("style", "name"), "")
                    if sname:
                        hidden.add(sname)
        return hidden

    def _extract_ods_cells(
        self,
        table: ET.Element,
        sheet_label: str,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract cell text content from a table."""
        row_idx = 0
        for row in table.iter(_tag("table", "table-row")):
            row_repeat = int(
                row.get(_tag("table", "number-rows-repeated"), "1"),
            )
            row_hidden = row.get(_tag("table", "visibility"), "visible") != "visible"

            col_idx = 0
            has_content = False
            for cell in row:
                if cell.tag == _tag("table", "covered-table-cell"):
                    col_repeat = int(
                        cell.get(_tag("table", "number-columns-repeated"), "1"),
                    )
                    col_idx += col_repeat
                    continue
                if cell.tag != _tag("table", "table-cell"):
                    continue

                col_repeat = int(
                    cell.get(_tag("table", "number-columns-repeated"), "1"),
                )

                # Collect cell text from <text:p> children
                cell_texts: list[str] = []
                for p in cell.findall(_tag("text", "p")):
                    t = _collect_text_no_tail(p)
                    if t:
                        cell_texts.append(t)

                # Also check office:string-value attribute
                str_val = cell.get(
                    "{%s}string-value" % NS["office"], "",
                ).strip()

                cell_text = " ".join(cell_texts)
                combined = cell_text
                if str_val and str_val != cell_text:
                    combined = "%s | attr:%s" % (cell_text, str_val) if cell_text else str_val

                if combined:
                    has_content = True
                    cell_ref = "%s%d" % (_col_letter(col_idx), row_idx + 1)
                    meta: Dict[str, str] = {}
                    if row_hidden:
                        meta["row_hidden"] = "true"
                    artifacts.append(ExtractedArtifact(
                        location="ods:%s/%s" % (sheet_label, cell_ref),
                        text=combined,
                        metadata=meta,
                    ))

                col_idx += col_repeat

            if has_content:
                # Only advance row index for rows that had data
                # (to avoid huge row gaps from repeated empty rows)
                row_idx += 1
            else:
                row_idx += row_repeat

    def _extract_ods_annotations(
        self,
        table: ET.Element,
        sheet_label: str,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract cell annotations (comments) from a table.

        Note: annotations at the global level are already captured by
        ``_extract_annotations_global``, but we do a table-level pass
        here to produce sheet-specific location tags.
        """
        # Annotations are children of table-cell; iterate cells
        ann_idx = 0
        for ann in table.iter(_tag("office", "annotation")):
            ann_idx += 1
            text = _collect_text_no_tail(ann, self._ANNOTATION_SKIP_TAGS)
            if text:
                artifacts.append(ExtractedArtifact(
                    location="ods:%s/annotation[%d]" % (sheet_label, ann_idx),
                    text=text,
                ))

    def _extract_ods_named_ranges(
        self,
        spreadsheet: ET.Element,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract named ranges and named expressions."""
        named_exprs = spreadsheet.find(_tag("table", "named-expressions"))
        if named_exprs is None:
            return
        for i, nr in enumerate(
            named_exprs.findall(_tag("table", "named-range")), start=1,
        ):
            name = nr.get(_tag("table", "name"), "")
            cell_range = nr.get(_tag("table", "cell-range-address"), "")
            if name:
                artifacts.append(ExtractedArtifact(
                    location="ods:named-range[%d]" % i,
                    text=name,
                    metadata={"range": cell_range},
                ))

        for i, ne in enumerate(
            named_exprs.findall(_tag("table", "named-expression")), start=1,
        ):
            name = ne.get(_tag("table", "name"), "")
            expr = ne.get(_tag("table", "expression"), "")
            if name or expr:
                artifacts.append(ExtractedArtifact(
                    location="ods:named-expression[%d]" % i,
                    text="%s=%s" % (name, expr) if expr else name,
                    metadata={"name": name, "expression": expr},
                ))

    # ==================================================================
    # ODP-specific extraction
    # ==================================================================

    def _extract_odp(
        self,
        content_root: ET.Element,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        body = content_root.find(_tag("office", "body"))
        if body is None:
            return
        presentation = body.find(_tag("office", "presentation"))
        if presentation is None:
            return

        for slide_idx, page in enumerate(
            presentation.findall(_tag("draw", "page")), start=1,
        ):
            page_name = page.get(_tag("draw", "name"), "slide%d" % slide_idx)
            visibility = page.get(
                _tag("presentation", "visibility"), "visible",
            )
            is_hidden = visibility == "hidden"

            slide_label = "%s%s" % (
                page_name,
                "[hidden]" if is_hidden else "",
            )

            self._extract_odp_slide_text(page, slide_label, slide_idx, artifacts)
            self._extract_odp_notes(page, slide_label, slide_idx, artifacts)

    def _extract_odp_slide_text(
        self,
        page: ET.Element,
        slide_label: str,
        slide_idx: int,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract text content from a slide (draw:page)."""
        # Collect text from all frames / text-boxes on the slide,
        # excluding presentation:notes children
        text_parts: list[str] = []
        for child in page:
            # Skip notes elements -- handled separately
            if child.tag == _tag("presentation", "notes"):
                continue
            part = _collect_text_no_tail(child)
            if part:
                text_parts.append(part)

        combined = " ".join(text_parts)
        if combined:
            artifacts.append(ExtractedArtifact(
                location="odp:%s/text" % slide_label,
                text=combined,
            ))

    def _extract_odp_notes(
        self,
        page: ET.Element,
        slide_label: str,
        slide_idx: int,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        """Extract speaker notes from a slide."""
        for notes in page.findall(_tag("presentation", "notes")):
            text = _collect_text_no_tail(notes)
            if text:
                artifacts.append(ExtractedArtifact(
                    location="odp:%s/notes" % slide_label,
                    text=text,
                ))
