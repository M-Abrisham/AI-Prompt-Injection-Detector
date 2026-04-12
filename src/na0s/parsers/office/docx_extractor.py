"""DOCX deep text extractor — extracts text from all user-controllable surfaces.

Parses OOXML (ECMA-376) DOCX files using only stdlib ``zipfile`` and
``xml.etree.ElementTree``.  Covers 19 hiding spots documented in the
research inventory (``docs/research/hiding_spots_docx.md``).
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
# OOXML namespace map
# ---------------------------------------------------------------------------

_NS: Dict[str, str] = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "dc": "http://purl.org/dc/elements/1.1/",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "vt": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
    "cust": "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties",
}

# Convenience tags (fully qualified)
_W = "{" + _NS["w"] + "}"
_R = "{" + _NS["r"] + "}"
_DC = "{" + _NS["dc"] + "}"
_CP = "{" + _NS["cp"] + "}"
_VT = "{" + _NS["vt"] + "}"
_MC = "{" + _NS["mc"] + "}"
_EP = "{" + _NS["ep"] + "}"
_CUST = "{" + _NS["cust"] + "}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_xml(raw: bytes) -> Optional[ET.Element]:
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        logger.debug("docx: malformed XML, skipping: %s", exc)
        return None


def _collect_text(node: ET.Element, tag: str = f"{_W}t") -> str:
    parts = []
    for t in node.iter(tag):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def _collect_del_text(node: ET.Element) -> str:
    parts = []
    for t in node.iter(f"{_W}delText"):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def _run_has_vanish(run: ET.Element) -> bool:
    rpr = run.find(f"{_W}rPr")
    if rpr is None:
        return False
    vanish = rpr.find(f"{_W}vanish")
    if vanish is None:
        return False
    val = vanish.get(f"{_W}val")
    # <w:vanish/> means true; <w:vanish w:val="false"/> means false
    return val is None or val.lower() not in ("false", "0", "off")


def _run_is_white_or_tiny(run: ET.Element) -> bool:
    rpr = run.find(f"{_W}rPr")
    if rpr is None:
        return False
    color = rpr.find(f"{_W}color")
    sz = rpr.find(f"{_W}sz")
    is_white = (
        color is not None
        and (color.get(f"{_W}val") or "").upper() in ("FFFFFF", "FFF")
    )
    is_tiny = sz is not None and (sz.get(f"{_W}val") or "0") in ("1", "2")
    return is_white or is_tiny


def _extract_runs_text(root: ET.Element) -> str:
    return _collect_text(root)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor(OfficeExtractor):
    @property
    def format_name(self) -> str:
        return "docx"

    def extract(self, data: bytes) -> List[ExtractedArtifact]:
        artifacts: List[ExtractedArtifact] = []

        try:
            zf = zipfile.ZipFile(io.BytesIO(data), "r")
        except zipfile.BadZipFile as exc:
            logger.warning("docx: invalid ZIP: %s", exc)
            return artifacts

        with zf:
            warnings = validate_zip_safety(zf)
            if warnings:
                for w in warnings:
                    logger.warning("docx: %s", w)
                return artifacts

            names = set(zf.namelist())

            # 1. Body text
            self._extract_body(zf, artifacts)

            # 2. Comments
            self._extract_comments(zf, artifacts)

            # 5. Headers
            for name in sorted(names):
                if name.startswith("word/header") and name.endswith(".xml"):
                    self._extract_header_footer(zf, name, "header", artifacts)

            # 6. Footers
            for name in sorted(names):
                if name.startswith("word/footer") and name.endswith(".xml"):
                    self._extract_header_footer(zf, name, "footer", artifacts)

            # 7. Footnotes
            self._extract_notes(zf, "word/footnotes.xml", "footnote", artifacts)

            # 8. Endnotes
            self._extract_notes(zf, "word/endnotes.xml", "endnote", artifacts)

            # 11. Core properties
            self._extract_core_props(zf, artifacts)

            # 12. Custom properties
            self._extract_custom_props(zf, artifacts)

            # 13. App properties
            self._extract_app_props(zf, artifacts)

            # 17. Custom XML parts
            for name in sorted(names):
                if name.startswith("customXml/item") and name.endswith(".xml"):
                    self._extract_custom_xml(zf, name, artifacts)

        return artifacts

    # -- Body text (1) + inline elements (3,4,9,10,14,15,16,18,19) ----------

    def _extract_body(
        self, zf: zipfile.ZipFile, artifacts: List[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, "word/document.xml")
        if raw is None:
            return
        root = _parse_xml(raw)
        if root is None:
            return
        logger.debug("docx: extracting body text from word/document.xml")

        body = root.find(f"{_W}body")
        if body is None:
            body = root  # fallback: search entire tree

        # 1. Body paragraphs — full text
        body_text = _collect_text(body)
        if body_text:
            artifacts.append(ExtractedArtifact(
                location="docx:body",
                text=body_text,
            ))

        # Now extract inline elements from the full document tree
        self._extract_inline_elements(root, "document", artifacts)

    def _extract_inline_elements(
        self,
        root: ET.Element,
        part_label: str,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        # 3. Tracked insertions
        for idx, ins in enumerate(root.iter(f"{_W}ins")):
            text = _collect_text(ins)
            if text:
                author = ins.get(f"{_W}author", "")
                artifacts.append(ExtractedArtifact(
                    location=f"docx:{part_label}/insertion[{idx}]",
                    text=text,
                    metadata={"author": author} if author else {},
                ))

        # 4. Tracked deletions
        for idx, deletion in enumerate(root.iter(f"{_W}del")):
            text = _collect_del_text(deletion)
            if text:
                author = deletion.get(f"{_W}author", "")
                artifacts.append(ExtractedArtifact(
                    location=f"docx:{part_label}/deletion[{idx}]",
                    text=text,
                    metadata={"author": author} if author else {},
                ))

        # 9. Text boxes (txbxContent)
        for idx, txbx in enumerate(root.iter(f"{_W}txbxContent")):
            text = _collect_text(txbx)
            if text:
                artifacts.append(ExtractedArtifact(
                    location=f"docx:{part_label}/textbox[{idx}]",
                    text=text,
                ))

        # 10. Hidden text (vanish)
        for run in root.iter(f"{_W}r"):
            if _run_has_vanish(run):
                text = _collect_text(run)
                if text:
                    artifacts.append(ExtractedArtifact(
                        location=f"docx:{part_label}/hidden-text",
                        text=text,
                    ))

        # 14. Hyperlinks
        for idx, hyperlink in enumerate(root.iter(f"{_W}hyperlink")):
            text = _collect_text(hyperlink)
            if text:
                r_id = hyperlink.get(f"{_R}id", "")
                artifacts.append(ExtractedArtifact(
                    location=f"docx:{part_label}/hyperlink[{idx}]",
                    text=text,
                    metadata={"r:id": r_id} if r_id else {},
                ))

        # 15. Field codes — instrText
        for idx, instr in enumerate(root.iter(f"{_W}instrText")):
            if instr.text and instr.text.strip():
                artifacts.append(ExtractedArtifact(
                    location=f"docx:{part_label}/field-code[{idx}]",
                    text=instr.text.strip(),
                ))

        # 15b. Simple fields — fldSimple
        for idx, fld in enumerate(root.iter(f"{_W}fldSimple")):
            instr_val = fld.get(f"{_W}instr", "")
            if instr_val.strip():
                artifacts.append(ExtractedArtifact(
                    location=f"docx:{part_label}/field-simple[{idx}]",
                    text=instr_val.strip(),
                ))

        # 16. Structured Document Tags (content controls)
        for idx, sdt in enumerate(root.iter(f"{_W}sdt")):
            sdt_pr = sdt.find(f"{_W}sdtPr")
            if sdt_pr is not None:
                tag_el = sdt_pr.find(f"{_W}tag")
                tag_val = tag_el.get(f"{_W}val", "") if tag_el is not None else ""
                alias_el = sdt_pr.find(f"{_W}alias")
                alias_val = alias_el.get(f"{_W}val", "") if alias_el is not None else ""
                if tag_val:
                    artifacts.append(ExtractedArtifact(
                        location=f"docx:{part_label}/sdt[{idx}]/tag",
                        text=tag_val,
                    ))
                if alias_val:
                    artifacts.append(ExtractedArtifact(
                        location=f"docx:{part_label}/sdt[{idx}]/alias",
                        text=alias_val,
                    ))

        # 18. Smart tags
        for idx, stag in enumerate(root.iter(f"{_W}smartTag")):
            spr = stag.find(f"{_W}smartTagPr")
            if spr is not None:
                for attr in spr.iter(f"{_W}attr"):
                    name = attr.get(f"{_W}name", "")
                    val = attr.get(f"{_W}val", "")
                    if val:
                        artifacts.append(ExtractedArtifact(
                            location=f"docx:{part_label}/smarttag[{idx}]/{name}",
                            text=val,
                        ))

        # 19. White/tiny text
        for run in root.iter(f"{_W}r"):
            if _run_is_white_or_tiny(run):
                text = _collect_text(run)
                if text:
                    artifacts.append(ExtractedArtifact(
                        location=f"docx:{part_label}/white-or-tiny-text",
                        text=text,
                    ))

    # -- Comments (2) -------------------------------------------------------

    def _extract_comments(
        self, zf: zipfile.ZipFile, artifacts: List[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, "word/comments.xml")
        if raw is None:
            return
        root = _parse_xml(raw)
        if root is None:
            return
        logger.debug("docx: extracting comments")

        for comment in root.iter(f"{_W}comment"):
            c_id = comment.get(f"{_W}id", "?")
            author = comment.get(f"{_W}author", "")
            text = _collect_text(comment)
            if text:
                artifacts.append(ExtractedArtifact(
                    location=f"docx:comments/comment[{c_id}]",
                    text=text,
                    metadata={"author": author} if author else {},
                ))

    # -- Headers / Footers (5, 6) -------------------------------------------

    def _extract_header_footer(
        self,
        zf: zipfile.ZipFile,
        entry: str,
        kind: str,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, entry)
        if raw is None:
            return
        root = _parse_xml(raw)
        if root is None:
            return

        # Derive label: "word/header1.xml" -> "header1"
        label = entry.replace("word/", "").replace(".xml", "")
        logger.debug("docx: extracting %s", label)

        text = _collect_text(root)
        if text:
            artifacts.append(ExtractedArtifact(
                location=f"docx:{label}",
                text=text,
            ))

        # Also extract inline elements (textboxes, hidden text, etc.)
        self._extract_inline_elements(root, label, artifacts)

    # -- Footnotes / Endnotes (7, 8) ----------------------------------------

    def _extract_notes(
        self,
        zf: zipfile.ZipFile,
        entry: str,
        note_tag: str,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, entry)
        if raw is None:
            return
        root = _parse_xml(raw)
        if root is None:
            return
        logger.debug("docx: extracting %ss", note_tag)

        for note in root.iter(f"{_W}{note_tag}"):
            n_type = note.get(f"{_W}type", "")
            # Skip system-generated separator notes
            if n_type in ("separator", "continuationSeparator", "continuationNotice"):
                continue
            n_id = note.get(f"{_W}id", "?")
            text = _collect_text(note)
            if text:
                artifacts.append(ExtractedArtifact(
                    location=f"docx:{note_tag}s/{note_tag}[{n_id}]",
                    text=text,
                ))

    # -- Core properties (11) -----------------------------------------------

    def _extract_core_props(
        self, zf: zipfile.ZipFile, artifacts: List[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, "docProps/core.xml")
        if raw is None:
            return
        root = _parse_xml(raw)
        if root is None:
            return
        logger.debug("docx: extracting core properties")

        fields = [
            (f"{_DC}title", "dc:title"),
            (f"{_DC}subject", "dc:subject"),
            (f"{_DC}description", "dc:description"),
            (f"{_DC}creator", "dc:creator"),
            (f"{_CP}keywords", "cp:keywords"),
            (f"{_CP}lastModifiedBy", "cp:lastModifiedBy"),
            (f"{_CP}category", "cp:category"),
        ]
        for tag, label in fields:
            el = root.find(tag)
            if el is not None and el.text and el.text.strip():
                artifacts.append(ExtractedArtifact(
                    location=f"docx:core/{label}",
                    text=el.text.strip(),
                ))

    # -- Custom properties (12) ---------------------------------------------

    def _extract_custom_props(
        self, zf: zipfile.ZipFile, artifacts: List[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, "docProps/custom.xml")
        if raw is None:
            return
        root = _parse_xml(raw)
        if root is None:
            return
        logger.debug("docx: extracting custom properties")

        for idx, prop in enumerate(root.iter(f"{_CUST}property")):
            prop_name = prop.get("name", f"property[{idx}]")
            # Value can be in various vt:* child elements
            value_text = ""
            for child in prop:
                if child.text and child.text.strip():
                    value_text = child.text.strip()
                    break
            if value_text:
                artifacts.append(ExtractedArtifact(
                    location=f"docx:custom/{prop_name}",
                    text=value_text,
                ))
            # The property name itself can contain injected text
            if prop_name and prop_name != f"property[{idx}]":
                artifacts.append(ExtractedArtifact(
                    location=f"docx:custom/property[{idx}]/name",
                    text=prop_name,
                ))

    # -- App properties (13) ------------------------------------------------

    def _extract_app_props(
        self, zf: zipfile.ZipFile, artifacts: List[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, "docProps/app.xml")
        if raw is None:
            return
        root = _parse_xml(raw)
        if root is None:
            return
        logger.debug("docx: extracting app properties")

        fields = ["Company", "Manager", "HyperlinkBase"]
        for field_name in fields:
            el = root.find(f"{_EP}{field_name}")
            if el is not None and el.text and el.text.strip():
                artifacts.append(ExtractedArtifact(
                    location=f"docx:app/{field_name}",
                    text=el.text.strip(),
                ))

    # -- Custom XML parts (17) ----------------------------------------------

    def _extract_custom_xml(
        self,
        zf: zipfile.ZipFile,
        entry: str,
        artifacts: List[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, entry)
        if raw is None:
            return
        root = _parse_xml(raw)
        if root is None:
            return

        label = entry.replace(".xml", "").replace("/", ".")
        logger.debug("docx: extracting custom XML part %s", entry)

        # Collect all text content from arbitrary XML
        texts: List[str] = []
        for el in root.iter():
            if el.text and el.text.strip():
                texts.append(el.text.strip())
            if el.tail and el.tail.strip():
                texts.append(el.tail.strip())

        combined = " ".join(texts)
        if combined:
            artifacts.append(ExtractedArtifact(
                location=f"docx:{label}",
                text=combined,
            ))
