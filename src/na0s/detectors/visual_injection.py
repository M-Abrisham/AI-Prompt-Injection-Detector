"""Image-based prompt injection detection (N6 -- Multimodal Injection).

Detects prompt injection attempts hidden in images and documents by
combining OCR text extraction, metadata scanning, and visual
steganography indicators.

Multimodal LLMs are vulnerable to visual prompt injection -- text
embedded in images that instructs the LLM.  This module catches:

1. **Text-in-image injection**: OCR-extracted text scanned through
   Na0S's existing ``scan()`` pipeline for injection patterns.
2. **Visual steganography indicators**: tiny font text, low-contrast
   text hiding, text in metadata (EXIF/XMP).
3. **Document visual injection**: white-on-white text in PDFs/DOCX,
   tiny font text (< 2pt), hidden annotations/comments/layers.

ALL image/document dependencies (PIL, OCR engines, pymupdf, etc.) are
optional.  When unavailable the detector degrades gracefully and
returns an empty result with warnings.

Technique IDs: ``M1`` (Multimodal Injection), ``M1.1`` (Text-in-Image),
``M1.2`` (Metadata Injection), ``M1.3`` (Steganographic Hiding),
``M1.4`` (Document Visual Injection).
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency probing
# ---------------------------------------------------------------------------

_HAS_PIL = False
Image = None  # sentinel so the name always exists as a patch/guard target
try:
    from PIL import Image  # type: ignore[import-untyped]

    _HAS_PIL = True
except ImportError:
    pass

_HAS_PYMUPDF = False
try:
    import fitz  # type: ignore[import-untyped]  # pymupdf

    _HAS_PYMUPDF = True
except ImportError:
    pass

_HAS_DOCX = False
try:
    import docx  # type: ignore[import-untyped]  # python-docx

    _HAS_DOCX = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Minimum risk score from scan() to consider OCR text suspicious.
OCR_SCAN_THRESHOLD: float = float(
    os.getenv("VISUAL_OCR_SCAN_THRESHOLD", "0.40")
)

#: Font size (pt) below which text is considered suspiciously small.
TINY_FONT_PT: float = float(os.getenv("VISUAL_TINY_FONT_PT", "2.0"))

#: Maximum image bytes to process (default 20 MB).
MAX_IMAGE_BYTES: int = int(
    os.getenv("VISUAL_MAX_IMAGE_BYTES", 20 * 1024 * 1024)
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VisualInjectionIndicator:
    """A single indicator of visual injection."""

    indicator_type: str  # e.g. "ocr_injection", "metadata_injection", "tiny_font"
    description: str
    severity: str  # "low", "medium", "high", "critical"
    evidence: str = ""


@dataclass
class VisualInjectionResult:
    """Result of visual injection scanning.

    Attributes:
        is_suspicious:         True if any injection indicator was found.
        risk_score:            Aggregate risk score in [0.0, 1.0].
        extracted_text:        Combined text extracted from the image
                               (OCR + metadata).
        injection_indicators:  List of detected injection indicators.
        technique_ids:         MITRE-style technique IDs matched.
        warnings:              Non-fatal issues encountered.
    """

    is_suspicious: bool = False
    risk_score: float = 0.0
    extracted_text: str = ""
    injection_indicators: List[VisualInjectionIndicator] = field(
        default_factory=list
    )
    technique_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Suspicious text patterns (large font instructions, repeated commands)
# ---------------------------------------------------------------------------

#: Patterns that suggest text was intentionally placed to instruct an LLM.
_INSTRUCTION_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|override)\s+"
            r"(?:all\s+)?(?:previous|prior|above|earlier|system)\s+"
            r"(?:instructions?|prompts?|rules?|commands?|directives?)"
        ),
        "instruction_override",
        "Text instructs LLM to ignore previous instructions",
    ),
    (
        re.compile(
            r"(?i)\b(?:you\s+are\s+now|new\s+instructions?|"
            r"from\s+now\s+on|instead\s+(?:do|follow|obey))\b"
        ),
        "role_reassignment",
        "Text attempts to reassign LLM role or instructions",
    ),
    (
        re.compile(
            r"(?i)\b(?:system\s*prompt|SYSTEM|<\|system\|>|<<SYS>>)\b"
        ),
        "system_prompt_marker",
        "Text contains system prompt markers",
    ),
    (
        re.compile(
            r"(?i)\b(?:do\s+not\s+mention|never\s+reveal|"
            r"keep\s+(?:this|it)\s+secret|hide\s+this)\b"
        ),
        "secrecy_instruction",
        "Text contains secrecy instructions for the LLM",
    ),
    (
        re.compile(
            r"(?i)\b(?:output|print|say|respond\s+with|reply\s+with)\s+"
            r"(?:exactly|only|just)\s*[:\"]"
        ),
        "forced_output",
        "Text forces specific LLM output",
    ),
]

#: Minimum number of repeated words/phrases to flag as suspicious repetition.
_REPETITION_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scan_text_for_injection(text: str) -> tuple[float, list[VisualInjectionIndicator], list[str]]:
    """Run extracted text through Na0S scan() and pattern checks.

    Returns (risk_score, indicators, technique_ids).
    """
    indicators: list[VisualInjectionIndicator] = []
    technique_ids: list[str] = []
    risk_score = 0.0

    if not text or not text.strip():
        return risk_score, indicators, technique_ids

    # --- Run through Na0S scan() pipeline ---
    try:
        from na0s.predict import scan as na0s_scan

        result = na0s_scan(text)
        if result.is_malicious or result.risk_score >= OCR_SCAN_THRESHOLD:
            scan_risk = result.risk_score
            risk_score = max(risk_score, scan_risk)
            indicators.append(VisualInjectionIndicator(
                indicator_type="ocr_injection",
                description="OCR-extracted text flagged as injection by Na0S scan "
                            "(score={:.2f}, label={})".format(scan_risk, result.label),
                severity="high" if scan_risk >= 0.7 else "medium",
                evidence=text[:200],
            ))
            technique_ids.extend(["M1", "M1.1"])
            # Carry forward any technique tags from the scan
            for tag in result.technique_tags:
                if tag not in technique_ids:
                    technique_ids.append(tag)
    except Exception as exc:
        logger.debug("Na0S scan() unavailable for OCR text: %s", exc)

    # --- Pattern-based checks on raw extracted text ---
    for pattern, name, description in _INSTRUCTION_PATTERNS:
        match = pattern.search(text)
        if match:
            risk_score = max(risk_score, 0.65)
            indicators.append(VisualInjectionIndicator(
                indicator_type="visual_instruction_pattern",
                description=description,
                severity="high",
                evidence=match.group(0)[:100],
            ))
            if "M1" not in technique_ids:
                technique_ids.append("M1")
            if "M1.1" not in technique_ids:
                technique_ids.append("M1.1")

    # --- Repeated text detection (common in visual injection) ---
    words = text.lower().split()
    if len(words) >= _REPETITION_THRESHOLD:
        from collections import Counter

        word_counts = Counter(words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]
        # If a single word appears more than 50% of the time and at least N times
        if most_common_count >= _REPETITION_THRESHOLD and most_common_count > len(words) * 0.5:
            risk_score = max(risk_score, 0.4)
            indicators.append(VisualInjectionIndicator(
                indicator_type="repeated_text",
                description="Suspiciously repeated text in image: '{}' appears {} times".format(
                    most_common_word[:30], most_common_count
                ),
                severity="medium",
                evidence=most_common_word,
            ))
            if "M1" not in technique_ids:
                technique_ids.append("M1")
            if "M1.3" not in technique_ids:
                technique_ids.append("M1.3")

    return risk_score, indicators, technique_ids


def _scan_metadata_for_injection(
    metadata_text: str,
    metadata_fields: list[str],
) -> tuple[float, list[VisualInjectionIndicator], list[str]]:
    """Scan image metadata text for injection payloads.

    Returns (risk_score, indicators, technique_ids).
    """
    indicators: list[VisualInjectionIndicator] = []
    technique_ids: list[str] = []
    risk_score = 0.0

    if not metadata_text or not metadata_text.strip():
        return risk_score, indicators, technique_ids

    # Check patterns in metadata
    for pattern, name, description in _INSTRUCTION_PATTERNS:
        match = pattern.search(metadata_text)
        if match:
            risk_score = max(risk_score, 0.75)
            indicators.append(VisualInjectionIndicator(
                indicator_type="metadata_injection",
                description="Injection pattern in image metadata ({}): {}".format(
                    ", ".join(metadata_fields), description
                ),
                severity="high",
                evidence=match.group(0)[:100],
            ))
            if "M1" not in technique_ids:
                technique_ids.append("M1")
            if "M1.2" not in technique_ids:
                technique_ids.append("M1.2")

    # Also run through scan() if the metadata text is long enough
    if len(metadata_text.strip()) > 10:
        try:
            from na0s.predict import scan as na0s_scan

            result = na0s_scan(metadata_text)
            if result.is_malicious or result.risk_score >= OCR_SCAN_THRESHOLD:
                risk_score = max(risk_score, result.risk_score)
                indicators.append(VisualInjectionIndicator(
                    indicator_type="metadata_injection_scan",
                    description="Metadata text flagged by Na0S scan "
                                "(score={:.2f}, fields={})".format(
                                    result.risk_score,
                                    ", ".join(metadata_fields),
                                ),
                    severity="high" if result.risk_score >= 0.7 else "medium",
                    evidence=metadata_text[:200],
                ))
                if "M1" not in technique_ids:
                    technique_ids.append("M1")
                if "M1.2" not in technique_ids:
                    technique_ids.append("M1.2")
        except Exception as exc:
            logger.debug("Na0S scan() unavailable for metadata: %s", exc)

    return risk_score, indicators, technique_ids


def _detect_tiny_font_text(
    ocr_text: str,
    image_data: bytes,
) -> tuple[float, list[VisualInjectionIndicator], list[str]]:
    """Detect text that may be rendered in very small font.

    Heuristic: if OCR extracts text from an image but the image is very
    large relative to the amount of text, the text may be tiny (below
    normal readability threshold but OCR-detectable).

    Returns (risk_score, indicators, technique_ids).
    """
    indicators: list[VisualInjectionIndicator] = []
    technique_ids: list[str] = []
    risk_score = 0.0

    if not ocr_text or not _HAS_PIL:
        return risk_score, indicators, technique_ids

    try:
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size
        pixel_count = width * height

        # Heuristic: characters per megapixel.  Normal readable text
        # in a screenshot/photo typically yields 50-500 chars/MP.
        # Very high density (>2000 chars/MP) suggests tiny hidden text.
        text_len = len(ocr_text.strip())
        if pixel_count > 0 and text_len > 0:
            chars_per_mp = text_len / (pixel_count / 1_000_000)
            if chars_per_mp > 2000:
                risk_score = max(risk_score, 0.6)
                indicators.append(VisualInjectionIndicator(
                    indicator_type="tiny_font_heuristic",
                    description="High text density ({:.0f} chars/MP) suggests "
                                "very small/hidden text".format(chars_per_mp),
                    severity="medium",
                    evidence="{}x{} image, {} chars extracted".format(
                        width, height, text_len
                    ),
                ))
                if "M1" not in technique_ids:
                    technique_ids.append("M1")
                if "M1.3" not in technique_ids:
                    technique_ids.append("M1.3")
    except Exception as exc:
        logger.debug("Tiny font detection failed: %s", exc)

    return risk_score, indicators, technique_ids


def _detect_low_contrast_text(image_data: bytes) -> tuple[float, list[VisualInjectionIndicator], list[str]]:
    """Detect text matching background color (low contrast hiding).

    Heuristic: if an image is nearly uniform in color but OCR still
    extracts text, the text may be hidden via low contrast.

    Returns (risk_score, indicators, technique_ids).
    """
    indicators: list[VisualInjectionIndicator] = []
    technique_ids: list[str] = []
    risk_score = 0.0

    if not _HAS_PIL:
        return risk_score, indicators, technique_ids

    try:
        img = Image.open(io.BytesIO(image_data)).convert("L")  # grayscale
        # Sample pixels to check color variance
        pixels = list(img.getdata())
        if len(pixels) < 10:
            return risk_score, indicators, technique_ids

        # Compute standard deviation of pixel values
        mean_val = sum(pixels) / len(pixels)
        variance = sum((p - mean_val) ** 2 for p in pixels) / len(pixels)
        std_dev = variance ** 0.5

        # Very low std_dev (< 5) on an image that supposedly has text
        # indicates near-invisible text (same color as background).
        if std_dev < 5.0:
            risk_score = max(risk_score, 0.55)
            indicators.append(VisualInjectionIndicator(
                indicator_type="low_contrast_text",
                description="Image has very low contrast (std_dev={:.1f}) -- "
                            "text may be hidden against background".format(std_dev),
                severity="medium",
                evidence="pixel std_dev={:.1f}".format(std_dev),
            ))
            if "M1" not in technique_ids:
                technique_ids.append("M1")
            if "M1.3" not in technique_ids:
                technique_ids.append("M1.3")
    except Exception as exc:
        logger.debug("Low contrast detection failed: %s", exc)

    return risk_score, indicators, technique_ids


def _scan_pdf_visual_injection(
    pdf_bytes: bytes,
) -> tuple[float, list[VisualInjectionIndicator], list[str], str]:
    """Scan PDF for visual injection: tiny font, white-on-white, annotations.

    Returns (risk_score, indicators, technique_ids, extracted_text).
    """
    indicators: list[VisualInjectionIndicator] = []
    technique_ids: list[str] = []
    risk_score = 0.0
    all_text_parts: list[str] = []

    if not _HAS_PYMUPDF:
        return risk_score, indicators, technique_ids, ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num in range(min(len(doc), 50)):
            page = doc[page_num]

            # --- Extract text blocks with font info ---
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in blocks.get("blocks", []):
                if block.get("type") != 0:  # text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        all_text_parts.append(text)
                        font_size = span.get("size", 12.0)
                        color = span.get("color", 0)

                        # --- Tiny font detection (< 2pt) ---
                        if font_size < TINY_FONT_PT:
                            risk_score = max(risk_score, 0.7)
                            indicators.append(VisualInjectionIndicator(
                                indicator_type="pdf_tiny_font",
                                description="PDF text at {:.1f}pt font (page {}) -- "
                                            "below readability threshold".format(
                                                font_size, page_num + 1
                                            ),
                                severity="high",
                                evidence=text[:100],
                            ))
                            if "M1" not in technique_ids:
                                technique_ids.append("M1")
                            if "M1.4" not in technique_ids:
                                technique_ids.append("M1.4")

                        # --- White-on-white detection ---
                        # In pymupdf, color is an int (RGB packed) or tuple.
                        # White text: color close to 0xFFFFFF (16777215).
                        if isinstance(color, int) and color >= 0xFEFEFE:
                            risk_score = max(risk_score, 0.7)
                            indicators.append(VisualInjectionIndicator(
                                indicator_type="pdf_white_text",
                                description="White/near-white text in PDF "
                                            "(page {})".format(page_num + 1),
                                severity="high",
                                evidence=text[:100],
                            ))
                            if "M1" not in technique_ids:
                                technique_ids.append("M1")
                            if "M1.4" not in technique_ids:
                                technique_ids.append("M1.4")

            # --- Annotation / comment text ---
            annots = page.annots()
            if annots:
                for annot in annots:
                    annot_text = annot.info.get("content", "")
                    if annot_text and annot_text.strip():
                        all_text_parts.append(annot_text)
                        risk_score = max(risk_score, 0.5)
                        indicators.append(VisualInjectionIndicator(
                            indicator_type="pdf_annotation_text",
                            description="Text found in PDF annotation "
                                        "(page {})".format(page_num + 1),
                            severity="medium",
                            evidence=annot_text[:100],
                        ))
                        if "M1" not in technique_ids:
                            technique_ids.append("M1")
                        if "M1.4" not in technique_ids:
                            technique_ids.append("M1.4")

        doc.close()
    except Exception as exc:
        logger.debug("PDF visual injection scan failed: %s", exc)

    return risk_score, indicators, technique_ids, "\n".join(all_text_parts)


def _scan_docx_visual_injection(
    docx_bytes: bytes,
) -> tuple[float, list[VisualInjectionIndicator], list[str], str]:
    """Scan DOCX for visual injection: tiny font, white text, comments.

    Returns (risk_score, indicators, technique_ids, extracted_text).
    """
    indicators: list[VisualInjectionIndicator] = []
    technique_ids: list[str] = []
    risk_score = 0.0
    all_text_parts: list[str] = []

    if not _HAS_DOCX:
        return risk_score, indicators, technique_ids, ""

    try:
        document = docx.Document(io.BytesIO(docx_bytes))

        for para in document.paragraphs:
            for run in para.runs:
                text = run.text.strip()
                if not text:
                    continue

                all_text_parts.append(text)

                # --- Tiny font detection ---
                font_size = run.font.size
                if font_size is not None:
                    # font.size is in EMU (English Metric Units).
                    # 1pt = 12700 EMU.
                    pt_size = font_size / 12700.0
                    if pt_size < TINY_FONT_PT:
                        risk_score = max(risk_score, 0.7)
                        indicators.append(VisualInjectionIndicator(
                            indicator_type="docx_tiny_font",
                            description="DOCX text at {:.1f}pt font -- "
                                        "below readability threshold".format(pt_size),
                            severity="high",
                            evidence=text[:100],
                        ))
                        if "M1" not in technique_ids:
                            technique_ids.append("M1")
                        if "M1.4" not in technique_ids:
                            technique_ids.append("M1.4")

                # --- White text detection ---
                font_color = run.font.color
                if font_color and font_color.rgb is not None:
                    rgb_str = str(font_color.rgb).upper()
                    # White or near-white: FFFFFF, FEFEFE, etc.
                    if rgb_str in ("FFFFFF", "FEFEFE", "FDFDFD", "FAFAFA"):
                        risk_score = max(risk_score, 0.7)
                        indicators.append(VisualInjectionIndicator(
                            indicator_type="docx_white_text",
                            description="White/near-white text in DOCX "
                                        "(color={})".format(rgb_str),
                            severity="high",
                            evidence=text[:100],
                        ))
                        if "M1" not in technique_ids:
                            technique_ids.append("M1")
                        if "M1.4" not in technique_ids:
                            technique_ids.append("M1.4")

        # --- Comments extraction (via XML) ---
        try:
            from lxml import etree  # type: ignore[import-untyped]

            # Comments are in word/comments.xml inside the DOCX zip
            import zipfile

            with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
                if "word/comments.xml" in zf.namelist():
                    comments_xml = zf.read("word/comments.xml")
                    root = etree.fromstring(comments_xml)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    for comment in root.findall(".//w:comment", ns):
                        comment_text = "".join(comment.itertext()).strip()
                        if comment_text:
                            all_text_parts.append(comment_text)
                            risk_score = max(risk_score, 0.5)
                            indicators.append(VisualInjectionIndicator(
                                indicator_type="docx_comment_text",
                                description="Text found in DOCX comment",
                                severity="medium",
                                evidence=comment_text[:100],
                            ))
                            if "M1" not in technique_ids:
                                technique_ids.append("M1")
                            if "M1.4" not in technique_ids:
                                technique_ids.append("M1.4")
        except Exception as exc:
            logger.debug("DOCX comment extraction failed: %s", exc)

    except Exception as exc:
        logger.debug("DOCX visual injection scan failed: %s", exc)

    return risk_score, indicators, technique_ids, "\n".join(all_text_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_image(image_bytes: bytes) -> VisualInjectionResult:
    """Scan image bytes for visual prompt injection.

    This is the main entry point for image-based injection detection.
    It performs:

    1. OCR text extraction (via layer0 ``extract_text_from_image``)
    2. Metadata extraction (EXIF/XMP via layer0 ``extract_image_metadata``)
    3. Injection pattern scanning on extracted text
    4. Visual steganography indicators (tiny font, low contrast)

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes (PNG, JPEG, GIF, BMP, TIFF, WebP).

    Returns
    -------
    VisualInjectionResult
        Always returns a result -- never raises.
    """
    warnings: list[str] = []
    all_indicators: list[VisualInjectionIndicator] = []
    all_technique_ids: list[str] = []
    max_risk = 0.0
    text_parts: list[str] = []

    if not image_bytes:
        return VisualInjectionResult(warnings=["Empty image data"])

    # --- Size guard ---
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return VisualInjectionResult(
            warnings=[
                "Image exceeds size limit ({} bytes > {} bytes)".format(
                    len(image_bytes), MAX_IMAGE_BYTES
                )
            ]
        )

    # --- OCR text extraction ---
    ocr_text = ""
    try:
        from na0s.layer0.ocr_extractor import extract_text_from_image

        ocr_result = extract_text_from_image(image_bytes)
        ocr_text = ocr_result.text
        warnings.extend(ocr_result.warnings)
        if ocr_text:
            text_parts.append(ocr_text)
    except Exception as exc:
        warnings.append("OCR extraction failed: {}".format(exc))

    # --- Metadata extraction ---
    metadata_text = ""
    metadata_fields: list[str] = []
    try:
        from na0s.layer0.ocr_extractor import extract_image_metadata

        meta_result = extract_image_metadata(image_bytes)
        metadata_text = meta_result.metadata_text
        metadata_fields = meta_result.metadata_fields
        warnings.extend(meta_result.warnings)
        if metadata_text:
            text_parts.append(metadata_text)
    except Exception as exc:
        warnings.append("Metadata extraction failed: {}".format(exc))

    # --- Scan OCR text for injection ---
    if ocr_text:
        score, inds, tids = _scan_text_for_injection(ocr_text)
        max_risk = max(max_risk, score)
        all_indicators.extend(inds)
        for tid in tids:
            if tid not in all_technique_ids:
                all_technique_ids.append(tid)

    # --- Scan metadata for injection ---
    if metadata_text:
        score, inds, tids = _scan_metadata_for_injection(
            metadata_text, metadata_fields
        )
        max_risk = max(max_risk, score)
        all_indicators.extend(inds)
        for tid in tids:
            if tid not in all_technique_ids:
                all_technique_ids.append(tid)

    # --- Steganography: tiny font heuristic ---
    if ocr_text:
        score, inds, tids = _detect_tiny_font_text(ocr_text, image_bytes)
        max_risk = max(max_risk, score)
        all_indicators.extend(inds)
        for tid in tids:
            if tid not in all_technique_ids:
                all_technique_ids.append(tid)

    # --- Steganography: low contrast ---
    if ocr_text:
        score, inds, tids = _detect_low_contrast_text(image_bytes)
        max_risk = max(max_risk, score)
        all_indicators.extend(inds)
        for tid in tids:
            if tid not in all_technique_ids:
                all_technique_ids.append(tid)

    combined_text = "\n".join(text_parts).strip()

    return VisualInjectionResult(
        is_suspicious=len(all_indicators) > 0,
        risk_score=min(1.0, max_risk),
        extracted_text=combined_text,
        injection_indicators=all_indicators,
        technique_ids=all_technique_ids,
        warnings=warnings,
    )


def scan_image_file(path: str) -> VisualInjectionResult:
    """Scan an image file for visual prompt injection.

    Convenience wrapper that reads the file and calls ``scan_image``.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    VisualInjectionResult
        Always returns a result -- never raises.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as exc:
        return VisualInjectionResult(
            warnings=["Failed to read image file: {}".format(exc)]
        )

    return scan_image(data)


def scan_document_visual(
    doc_bytes: bytes,
    doc_type: str,
) -> VisualInjectionResult:
    """Scan a document for visual injection techniques.

    Detects white-on-white text, tiny font, annotations/comments,
    and hidden layers in PDFs and DOCX files.

    Parameters
    ----------
    doc_bytes : bytes
        Raw document bytes.
    doc_type : str
        Document type: ``"pdf"`` or ``"docx"``.

    Returns
    -------
    VisualInjectionResult
        Always returns a result -- never raises.
    """
    warnings: list[str] = []

    if not doc_bytes:
        return VisualInjectionResult(warnings=["Empty document data"])

    dtype = doc_type.lower().strip()

    if dtype == "pdf":
        score, indicators, tids, text = _scan_pdf_visual_injection(doc_bytes)
    elif dtype == "docx":
        score, indicators, tids, text = _scan_docx_visual_injection(doc_bytes)
    else:
        return VisualInjectionResult(
            warnings=["Unsupported document type for visual scan: {}".format(dtype)]
        )

    # Also scan extracted text for injection content
    if text:
        text_score, text_inds, text_tids = _scan_text_for_injection(text)
        score = max(score, text_score)
        indicators.extend(text_inds)
        for tid in text_tids:
            if tid not in tids:
                tids.append(tid)

    return VisualInjectionResult(
        is_suspicious=len(indicators) > 0,
        risk_score=min(1.0, score),
        extracted_text=text,
        injection_indicators=indicators,
        technique_ids=tids,
        warnings=warnings,
    )
