"""PPTX deep extractor — extracts all user-controllable text surfaces.

Extracts text from slides, speaker notes, comments (legacy + modern),
slide masters, slide layouts, hidden slides, alt text, hyperlink tooltips,
table cells, core/custom properties, and user-defined tags.

Uses only stdlib ``zipfile`` + ``xml.etree.ElementTree``.  No ``python-pptx``.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set

from na0s.parsers.office.base import (
    ExtractedArtifact,
    OfficeExtractor,
    safe_read_zip_entry,
    validate_zip_safety,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "dc": "http://purl.org/dc/elements/1.1/",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dcterms": "http://purl.org/dc/terms/",
    "cust": "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties",
    "vt": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
    "p188": "http://schemas.microsoft.com/office/powerpoint/2018/8/main",
}

# Pre-register namespaces so ET doesn't generate ns0/ns1 prefixes in logs
for _prefix, _uri in _NS.items():
    ET.register_namespace(_prefix, _uri)


def _safe_parse_xml(raw: bytes) -> Optional[ET.Element]:
    """Parse XML bytes, returning None on any malformed-XML error."""
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        logger.debug("pptx extractor: XML parse error: %s", exc)
        return None


def _collect_text(element: ET.Element, ns_a: str) -> str:
    """Collect all <a:t> text within an element, joining with newlines per <a:p>."""
    paragraphs: list[str] = []
    for p_elem in element.iter("{%s}p" % ns_a):
        runs: list[str] = []
        for t_elem in p_elem.iter("{%s}t" % ns_a):
            if t_elem.text:
                runs.append(t_elem.text)
        if runs:
            paragraphs.append("".join(runs))
    return "\n".join(paragraphs)


def _extract_number(name: str, prefix: str) -> Optional[int]:
    """Extract the numeric suffix from a path like 'ppt/slides/slide3.xml' -> 3."""
    m = re.search(re.escape(prefix) + r"(\d+)\.xml", name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor(OfficeExtractor):
    """Deep extractor for PPTX (Office Open XML PresentationML) files."""

    @property
    def format_name(self) -> str:
        return "pptx"

    def extract(self, data: bytes) -> List[ExtractedArtifact]:
        """Extract all text artifacts from a PPTX document."""
        artifacts: list[ExtractedArtifact] = []

        try:
            zf = zipfile.ZipFile(io.BytesIO(data), "r")
        except (zipfile.BadZipFile, Exception) as exc:
            logger.warning("pptx extractor: cannot open ZIP: %s", exc)
            return artifacts

        with zf:
            # Zip-bomb safety check
            warnings = validate_zip_safety(zf)
            if warnings:
                for w in warnings:
                    logger.warning("pptx extractor: %s", w)
                return artifacts

            names = set(zf.namelist())

            # Determine which slides are hidden
            hidden_slides = self._detect_hidden_slides(zf, names)

            # Build slide-number -> notes-slide mapping from rels
            slide_notes_map = self._build_slide_notes_map(zf, names)

            # Build slide-number -> comments mapping
            slide_comments_map = self._build_slide_comments_map(zf, names)

            # --- Slides (text, alt text, hyperlink tooltips, tables) ---
            self._extract_slides(zf, names, hidden_slides, artifacts)

            # --- Speaker notes (#1 priority) ---
            self._extract_notes(zf, names, slide_notes_map, hidden_slides, artifacts)

            # --- Legacy comments ---
            self._extract_legacy_comments(zf, names, slide_comments_map, hidden_slides, artifacts)

            # --- Modern comments (P188) ---
            self._extract_modern_comments(zf, names, artifacts)

            # --- Slide masters ---
            self._extract_part_text(
                zf, names, "ppt/slideMasters/slideMaster", "pptx:slideMaster", artifacts,
            )

            # --- Slide layouts ---
            self._extract_part_text(
                zf, names, "ppt/slideLayouts/slideLayout", "pptx:slideLayout", artifacts,
            )

            # --- Core properties ---
            self._extract_core_properties(zf, artifacts)

            # --- Custom properties ---
            self._extract_custom_properties(zf, artifacts)

            # --- User-defined tags ---
            self._extract_tags(zf, names, artifacts)

        return artifacts

    # ------------------------------------------------------------------
    # Hidden-slide detection
    # ------------------------------------------------------------------

    def _detect_hidden_slides(
        self, zf: zipfile.ZipFile, names: Set[str],
    ) -> Set[int]:
        """Return set of slide numbers that have show='0'."""
        hidden: set[int] = set()
        ns_p = _NS["p"]

        for name in sorted(names):
            num = _extract_number(name, "ppt/slides/slide")
            if num is None:
                continue

            raw = safe_read_zip_entry(zf, name)
            if raw is None:
                continue

            root = _safe_parse_xml(raw)
            if root is None:
                continue

            # Root tag should be <p:sld>; check show attribute
            show = root.get("show", "1")
            if show == "0":
                hidden.add(num)

        return hidden

    # ------------------------------------------------------------------
    # Relationship helpers
    # ------------------------------------------------------------------

    def _parse_rels(self, zf: zipfile.ZipFile, rels_path: str) -> List[Dict[str, str]]:
        """Parse a .rels file and return list of relationship dicts."""
        raw = safe_read_zip_entry(zf, rels_path)
        if raw is None:
            return []
        root = _safe_parse_xml(raw)
        if root is None:
            return []
        results = []
        for rel in root.iter("{%s}Relationship" % _NS["rel"]):
            results.append({
                "id": rel.get("Id", ""),
                "type": rel.get("Type", ""),
                "target": rel.get("Target", ""),
            })
        return results

    def _build_slide_notes_map(
        self, zf: zipfile.ZipFile, names: Set[str],
    ) -> Dict[int, str]:
        """Map slide number -> notes slide ZIP path."""
        mapping: dict[int, str] = {}
        notes_rel_type = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
        )
        for name in sorted(names):
            num = _extract_number(name, "ppt/slides/slide")
            if num is None:
                continue
            rels_path = "ppt/slides/_rels/slide{}.xml.rels".format(num)
            for rel in self._parse_rels(zf, rels_path):
                if rel["type"] == notes_rel_type:
                    target = rel["target"]
                    # Target is relative to ppt/slides/
                    if target.startswith("../"):
                        target = "ppt/" + target[3:]
                    elif not target.startswith("ppt/"):
                        target = "ppt/slides/" + target
                    mapping[num] = target
        return mapping

    def _build_slide_comments_map(
        self, zf: zipfile.ZipFile, names: Set[str],
    ) -> Dict[int, List[str]]:
        """Map slide number -> list of comment file ZIP paths."""
        mapping: dict[int, list[str]] = {}
        comment_rel_type = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
        )
        for name in sorted(names):
            num = _extract_number(name, "ppt/slides/slide")
            if num is None:
                continue
            rels_path = "ppt/slides/_rels/slide{}.xml.rels".format(num)
            for rel in self._parse_rels(zf, rels_path):
                if rel["type"] == comment_rel_type:
                    target = rel["target"]
                    if target.startswith("../"):
                        target = "ppt/" + target[3:]
                    elif not target.startswith("ppt/"):
                        target = "ppt/slides/" + target
                    mapping.setdefault(num, []).append(target)
        return mapping

    # ------------------------------------------------------------------
    # Slide text extraction (body text, alt text, tooltips, tables)
    # ------------------------------------------------------------------

    def _extract_slides(
        self,
        zf: zipfile.ZipFile,
        names: Set[str],
        hidden_slides: Set[int],
        artifacts: list[ExtractedArtifact],
    ) -> None:
        ns_a = _NS["a"]
        ns_p = _NS["p"]

        for name in sorted(names):
            num = _extract_number(name, "ppt/slides/slide")
            if num is None:
                continue

            raw = safe_read_zip_entry(zf, name)
            if raw is None:
                continue

            root = _safe_parse_xml(raw)
            if root is None:
                continue

            hidden_tag = "[hidden]" if num in hidden_slides else ""
            slide_prefix = "pptx:slide{}{}".format(num, hidden_tag)

            # --- Body text (all <a:t> in the slide) ---
            body_text = _collect_text(root, ns_a)
            if body_text:
                artifacts.append(ExtractedArtifact(
                    location="{}/text".format(slide_prefix),
                    text=body_text,
                ))

            # --- Alt text (descr attribute on cNvPr) ---
            self._extract_alt_text(root, slide_prefix, artifacts)

            # --- Hyperlink tooltips ---
            self._extract_hyperlink_tooltips(root, slide_prefix, artifacts)

    def _extract_alt_text(
        self,
        root: ET.Element,
        slide_prefix: str,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Extract descr and title attributes from cNvPr elements."""
        ns_p = _NS["p"]
        img_counter = 0

        # Search for cNvPr in both p: and default namespaces
        for cNvPr in root.iter("{%s}cNvPr" % ns_p):
            img_counter += 1
            descr = cNvPr.get("descr", "").strip()
            title = cNvPr.get("title", "").strip()
            name = cNvPr.get("name", "")

            combined = ""
            if title and descr:
                combined = "{}: {}".format(title, descr)
            elif descr:
                combined = descr
            elif title:
                combined = title

            if combined:
                artifacts.append(ExtractedArtifact(
                    location="{}/altText[{}]".format(slide_prefix, name or "shape{}".format(img_counter)),
                    text=combined,
                    metadata={"name": name} if name else {},
                ))

    def _extract_hyperlink_tooltips(
        self,
        root: ET.Element,
        slide_prefix: str,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Extract tooltip attributes from hlinkClick and hlinkHover elements."""
        ns_a = _NS["a"]
        link_counter = 0

        for tag_suffix in ("hlinkClick", "hlinkHover"):
            for hlink in root.iter("{%s}%s" % (ns_a, tag_suffix)):
                link_counter += 1
                tooltip = hlink.get("tooltip", "").strip()
                if tooltip:
                    artifacts.append(ExtractedArtifact(
                        location="{}/hyperlink[{}]".format(slide_prefix, link_counter),
                        text=tooltip,
                        metadata={"type": tag_suffix},
                    ))

    # ------------------------------------------------------------------
    # Speaker notes
    # ------------------------------------------------------------------

    def _extract_notes(
        self,
        zf: zipfile.ZipFile,
        names: Set[str],
        slide_notes_map: Dict[int, str],
        hidden_slides: Set[int],
        artifacts: list[ExtractedArtifact],
    ) -> None:
        ns_a = _NS["a"]

        for slide_num, notes_path in sorted(slide_notes_map.items()):
            raw = safe_read_zip_entry(zf, notes_path)
            if raw is None:
                continue

            root = _safe_parse_xml(raw)
            if root is None:
                continue

            text = _collect_text(root, ns_a)
            if text:
                hidden_tag = "[hidden]" if slide_num in hidden_slides else ""
                artifacts.append(ExtractedArtifact(
                    location="pptx:slide{}{}/notes".format(slide_num, hidden_tag),
                    text=text,
                ))

    # ------------------------------------------------------------------
    # Legacy comments
    # ------------------------------------------------------------------

    def _extract_legacy_comments(
        self,
        zf: zipfile.ZipFile,
        names: Set[str],
        slide_comments_map: Dict[int, List[str]],
        hidden_slides: Set[int],
        artifacts: list[ExtractedArtifact],
    ) -> None:
        ns_p = _NS["p"]

        for slide_num, comment_paths in sorted(slide_comments_map.items()):
            for comment_path in comment_paths:
                raw = safe_read_zip_entry(zf, comment_path)
                if raw is None:
                    continue

                root = _safe_parse_xml(raw)
                if root is None:
                    continue

                hidden_tag = "[hidden]" if slide_num in hidden_slides else ""
                slide_prefix = "pptx:slide{}{}".format(slide_num, hidden_tag)

                comment_idx = 0
                for cm in root.iter("{%s}cm" % ns_p):
                    comment_idx += 1
                    text_elem = cm.find("{%s}text" % ns_p)
                    if text_elem is not None and text_elem.text:
                        author_id = cm.get("authorId", "")
                        artifacts.append(ExtractedArtifact(
                            location="{}/comment[{}]".format(slide_prefix, comment_idx),
                            text=text_elem.text,
                            metadata={"authorId": author_id} if author_id else {},
                        ))

    # ------------------------------------------------------------------
    # Modern comments (P188 namespace)
    # ------------------------------------------------------------------

    def _extract_modern_comments(
        self,
        zf: zipfile.ZipFile,
        names: Set[str],
        artifacts: list[ExtractedArtifact],
    ) -> None:
        ns_p188 = _NS["p188"]
        ns_a = _NS["a"]

        # Modern comments can appear in various paths; scan all ZIP entries
        for name in sorted(names):
            if "comment" not in name.lower():
                continue
            # Skip legacy comments (already handled)
            if re.match(r"ppt/comments/comment\d+\.xml$", name):
                continue

            raw = safe_read_zip_entry(zf, name)
            if raw is None:
                continue

            root = _safe_parse_xml(raw)
            if root is None:
                continue

            comment_idx = 0
            # Look for p188:cm elements anywhere in the tree
            for cm in root.iter("{%s}cm" % ns_p188):
                comment_idx += 1
                text = _collect_text(cm, ns_a)
                if text:
                    artifacts.append(ExtractedArtifact(
                        location="pptx:modernComment[{}]".format(comment_idx),
                        text=text,
                    ))

            # Also handle the case where the root itself is a p188:cm
            if root.tag == "{%s}cm" % ns_p188:
                text = _collect_text(root, ns_a)
                if text and comment_idx == 0:
                    artifacts.append(ExtractedArtifact(
                        location="pptx:modernComment[1]",
                        text=text,
                    ))

    # ------------------------------------------------------------------
    # Generic part text extraction (masters, layouts)
    # ------------------------------------------------------------------

    def _extract_part_text(
        self,
        zf: zipfile.ZipFile,
        names: Set[str],
        path_prefix: str,
        location_prefix: str,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        """Extract all <a:t> text from parts matching path_prefix + N.xml."""
        ns_a = _NS["a"]

        for name in sorted(names):
            num = _extract_number(name, path_prefix)
            if num is None:
                continue

            raw = safe_read_zip_entry(zf, name)
            if raw is None:
                continue

            root = _safe_parse_xml(raw)
            if root is None:
                continue

            text = _collect_text(root, ns_a)
            if text:
                artifacts.append(ExtractedArtifact(
                    location="{}{}/text".format(location_prefix, num),
                    text=text,
                ))

    # ------------------------------------------------------------------
    # Core properties (docProps/core.xml)
    # ------------------------------------------------------------------

    def _extract_core_properties(
        self,
        zf: zipfile.ZipFile,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, "docProps/core.xml")
        if raw is None:
            return

        root = _safe_parse_xml(raw)
        if root is None:
            return

        # Dublin Core and CP fields to extract
        dc_fields = [
            ("dc", "title"),
            ("dc", "subject"),
            ("dc", "creator"),
            ("dc", "description"),
        ]
        cp_fields = [
            ("cp", "lastModifiedBy"),
            ("cp", "keywords"),
            ("cp", "category"),
        ]

        for ns_prefix, field_name in dc_fields + cp_fields:
            ns_uri = _NS.get(ns_prefix, "")
            elem = root.find("{%s}%s" % (ns_uri, field_name))
            if elem is not None and elem.text and elem.text.strip():
                artifacts.append(ExtractedArtifact(
                    location="pptx:core/{}:{}".format(ns_prefix, field_name),
                    text=elem.text.strip(),
                ))

    # ------------------------------------------------------------------
    # Custom properties (docProps/custom.xml)
    # ------------------------------------------------------------------

    def _extract_custom_properties(
        self,
        zf: zipfile.ZipFile,
        artifacts: list[ExtractedArtifact],
    ) -> None:
        raw = safe_read_zip_entry(zf, "docProps/custom.xml")
        if raw is None:
            return

        root = _safe_parse_xml(raw)
        if root is None:
            return

        ns_cust = _NS["cust"]
        ns_vt = _NS["vt"]

        for prop in root.iter("{%s}property" % ns_cust):
            prop_name = prop.get("name", "").strip()
            if not prop_name:
                continue

            # Value can be in various vt: child elements
            value = ""
            for child in prop:
                if child.text and child.text.strip():
                    value = child.text.strip()
                    break

            if value:
                artifacts.append(ExtractedArtifact(
                    location="pptx:custom/{}".format(prop_name),
                    text=value,
                ))

    # ------------------------------------------------------------------
    # User-defined tags (ppt/tags/tagN.xml)
    # ------------------------------------------------------------------

    def _extract_tags(
        self,
        zf: zipfile.ZipFile,
        names: Set[str],
        artifacts: list[ExtractedArtifact],
    ) -> None:
        ns_p = _NS["p"]

        for name in sorted(names):
            num = _extract_number(name, "ppt/tags/tag")
            if num is None:
                continue

            raw = safe_read_zip_entry(zf, name)
            if raw is None:
                continue

            root = _safe_parse_xml(raw)
            if root is None:
                continue

            for tag in root.iter("{%s}tag" % ns_p):
                tag_name = tag.get("name", "").strip()
                tag_val = tag.get("val", "").strip()
                if tag_name or tag_val:
                    artifacts.append(ExtractedArtifact(
                        location="pptx:tags/tag{}".format(num),
                        text="{}: {}".format(tag_name, tag_val) if tag_name and tag_val else (tag_name or tag_val),
                        metadata={"name": tag_name, "val": tag_val},
                    ))
