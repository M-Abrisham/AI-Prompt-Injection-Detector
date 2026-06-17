"""OCR text extraction and EXIF/XMP metadata extraction from images.

Extracts text from image data (PNG, JPEG, GIF, BMP, TIFF, WebP) using
OCR engines.  Engine priority: EasyOCR > pytesseract > none.

Also extracts text from EXIF tags (ImageDescription, UserComment,
XPComment, XPTitle, XPSubject) and XMP metadata (dc:description,
dc:title).  Image metadata can carry injection payloads invisible
to OCR -- extracting it ensures the full-text scan covers hidden text.

ALL imports are optional.  When no OCR library is installed the module
still loads and ``extract_text_from_image`` returns an empty
``OCRResult`` with ``engine="none"`` so the rest of the pipeline
continues without error.
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency probing
# ---------------------------------------------------------------------------

_HAS_PIL = False
try:
    from PIL import Image  # type: ignore[import-untyped]

    _HAS_PIL = True
except ImportError:
    pass

_HAS_EASYOCR = False
try:
    import easyocr  # type: ignore[import-untyped]

    _HAS_EASYOCR = True
except ImportError:
    pass

_HAS_TESSERACT = False
try:
    import pytesseract  # type: ignore[import-untyped]

    _HAS_TESSERACT = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from na0s._env import safe_int_env

#: Maximum allowed image size in bytes (default 10 MB, env-configurable).
MAX_IMAGE_BYTES: int = safe_int_env("L0_MAX_IMAGE_BYTES", 10 * 1024 * 1024, lo=1)

#: Maximum metadata text bytes to extract (default 64 KB, env-configurable).
#: Prevents DoS from images with huge EXIF/XMP metadata blocks.
MAX_METADATA_TEXT_BYTES: int = safe_int_env("L0_MAX_METADATA_TEXT_BYTES", 65536, lo=1)

#: Supported image magic-byte signatures.
_IMAGE_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II\x2a\x00", "tiff"),  # little-endian TIFF
    (b"MM\x00\x2a", "tiff"),  # big-endian TIFF
    (b"RIFF", "webp"),        # WebP is a RIFF container
]

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class OCRResult:
    """Result of an OCR extraction attempt.

    Attributes:
        text:       Extracted text (empty string if extraction failed).
        confidence: Engine-reported confidence [0.0, 1.0].  0.0 when no
                    extraction was performed.
        engine:     Name of the engine that produced the result
                    (``"easyocr"``, ``"tesseract"``, or ``"none"``).
        language:   Language code used/detected (e.g. ``"en"``).
        warnings:   Non-fatal issues encountered during extraction.
    """

    text: str = ""
    confidence: float = 0.0
    engine: str = "none"
    language: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImageMetadataResult:
    """Result of EXIF/XMP metadata text extraction from an image.

    Attributes:
        metadata_text:    Combined text from all metadata fields found.
        metadata_fields:  Names of the metadata fields that contained text.
        has_metadata_text: True if any metadata text was extracted.
        warnings:         Non-fatal issues encountered during extraction.
    """

    metadata_text: str = ""
    metadata_fields: list[str] = field(default_factory=list)
    has_metadata_text: bool = False
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EXIF tag IDs of interest (text-carrying tags)
# ---------------------------------------------------------------------------

#: EXIF tag ID -> human-readable name.
#: Covers all standard text-carrying tags an attacker could use to smuggle
#: injection payloads invisible to OCR.  Verified via PIL round-trip.
#:
#: Security audit (2026-02-18) findings:
#:   - 40093 is XPAuthor, NOT XPSubject (was mislabelled)
#:   - 40094 (XPKeywords) and 40095 (XPSubject) were missing entirely
#:   - Artist/Copyright/Software/DocumentName are commonly editable via
#:     ExifTool and can carry arbitrary text payloads
_EXIF_TEXT_TAGS: dict[int, str] = {
    269: "DocumentName",
    270: "ImageDescription",
    305: "Software",
    315: "Artist",
    33432: "Copyright",
    37510: "UserComment",
    40091: "XPTitle",
    40092: "XPComment",
    40093: "XPAuthor",      # FIXED: was incorrectly "XPSubject"
    40094: "XPKeywords",    # NEW: was missing
    40095: "XPSubject",     # NEW: this is the real XPSubject
}

#: Regex to find XMP metadata blocks in raw bytes.
_XMP_BLOCK_RE = re.compile(
    rb"<x:xmpmeta[^>]*>(.+?)</x:xmpmeta>", re.DOTALL
)

#: Regex to extract ALL rdf:li entries from a dc: field (multi-language).
#: Captures both plain text and CDATA-wrapped content.
#: Uses findall() so all language variants are extracted, not just the first.
_XMP_RDF_LI_RE = re.compile(
    rb"<rdf:li[^>]*>"
    rb"(?:<!\[CDATA\[(.*?)\]\]>|([^<]+))"
    rb"</rdf:li>",
    re.DOTALL,
)

#: Regex to extract the dc:description block (which contains rdf:li entries).
_XMP_DC_DESC_RE = re.compile(
    rb"<dc:description[^>]*>(.*?)</dc:description>",
    re.DOTALL,
)

#: Regex to extract the dc:title block.
_XMP_DC_TITLE_RE = re.compile(
    rb"<dc:title[^>]*>(.*?)</dc:title>",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_image_format(data: bytes) -> str | None:
    """Return the image format string if *data* matches a known signature.

    Returns ``None`` if the bytes do not match any known image header.
    """
    for sig, fmt in _IMAGE_SIGNATURES:
        if fmt == "webp":
            # WebP: RIFF....WEBP
            if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                return "webp"
        elif data[: len(sig)] == sig:
            return fmt
    return None


def extract_image_metadata(image_data: bytes) -> ImageMetadataResult:
    """Extract text from EXIF tags and XMP metadata in image bytes.

    Image metadata fields such as ``ImageDescription``, ``UserComment``,
    ``XPComment``, ``XPTitle``, ``XPSubject`` (EXIF) and ``dc:description``,
    ``dc:title`` (XMP) can carry injection payloads invisible to OCR.

    This function extracts any text found in those fields so the
    downstream pipeline can scan it for malicious content.

    Parameters
    ----------
    image_data:
        Raw bytes of the image file.

    Returns
    -------
    ImageMetadataResult
        Always returns a result -- never raises.  If PIL is not
        installed or the image has no text metadata, the result is
        empty.
    """
    if not image_data:
        return ImageMetadataResult()

    texts: list[str] = []
    fields_found: list[str] = []
    warnings: list[str] = []

    # --- EXIF extraction via PIL -----------------------------------------
    if _HAS_PIL:
        try:
            img = Image.open(io.BytesIO(image_data))
            exif_data = img.getexif()
            if exif_data:
                for tag_id, tag_name in _EXIF_TEXT_TAGS.items():
                    value = exif_data.get(tag_id)
                    if value is not None:
                        text = _decode_exif_value(value)
                        if text:
                            texts.append(text)
                            fields_found.append("exif:{}".format(tag_name))
        except Exception as exc:
            warnings.append("EXIF extraction error: {}".format(exc))
    else:
        warnings.append(
            "Pillow (PIL) not installed -- EXIF extraction skipped"
        )

    # --- XMP extraction from raw bytes -----------------------------------
    # XMP is embedded as XML in the image file; we can search for it
    # directly in the raw bytes without needing PIL.
    try:
        xmp_texts, xmp_fields = _extract_xmp_text(image_data)
        texts.extend(xmp_texts)
        fields_found.extend(xmp_fields)
    except Exception as exc:
        warnings.append("XMP extraction error: {}".format(exc))

    combined = "\n".join(texts).strip()

    # Truncate to prevent DoS from oversized metadata
    if len(combined) > MAX_METADATA_TEXT_BYTES:
        combined = combined[:MAX_METADATA_TEXT_BYTES]
        warnings.append(
            "Metadata text truncated to {} bytes".format(MAX_METADATA_TEXT_BYTES)
        )

    return ImageMetadataResult(
        metadata_text=combined,
        metadata_fields=fields_found,
        has_metadata_text=bool(combined),
        warnings=warnings,
    )


def extract_text_from_image(
    image_data: bytes,
    *,
    language: str = "en",
    max_bytes: int | None = None,
) -> OCRResult:
    """Extract text from raw image bytes.

    Parameters
    ----------
    image_data:
        Raw bytes of the image file.
    language:
        Language hint for the OCR engine (ISO 639-1, default ``"en"``).
    max_bytes:
        Override for the maximum allowed image size.  ``None`` uses the
        module-level ``MAX_IMAGE_BYTES`` constant.

    Returns
    -------
    OCRResult
        Always returns a result -- never raises.  If no library is
        available or the image cannot be processed, the ``text`` field
        is empty and ``engine`` is ``"none"``.
    """
    effective_max = max_bytes if max_bytes is not None else MAX_IMAGE_BYTES
    result_warnings: list[str] = []

    # --- Guard: size limit ---------------------------------------------------
    if len(image_data) > effective_max:
        return OCRResult(
            text="",
            confidence=0.0,
            engine="none",
            language=language,
            warnings=[
                "Image exceeds size limit ({} bytes > {} bytes)".format(
                    len(image_data), effective_max
                )
            ],
        )

    # --- Guard: PIL is required to decode image bytes ------------------------
    if not _HAS_PIL:
        return OCRResult(
            text="",
            confidence=0.0,
            engine="none",
            language=language,
            warnings=["Pillow (PIL) is not installed -- cannot decode images"],
        )

    # --- Guard: at least one OCR engine must be installed --------------------
    if not _HAS_EASYOCR and not _HAS_TESSERACT:
        return OCRResult(
            text="",
            confidence=0.0,
            engine="none",
            language=language,
            warnings=[
                "No OCR engine installed (install easyocr or pytesseract)"
            ],
        )

    # --- Load image via PIL --------------------------------------------------
    try:
        img = Image.open(io.BytesIO(image_data))
        img.load()  # force decode so corrupt data is caught here
    except Exception as exc:
        return OCRResult(
            text="",
            confidence=0.0,
            engine="none",
            language=language,
            warnings=["Failed to decode image: {}".format(exc)],
        )

    # --- Try EasyOCR first (better accuracy) ---------------------------------
    if _HAS_EASYOCR:
        try:
            reader = easyocr.Reader([language], gpu=False, verbose=False)
            import numpy as np  # type: ignore[import-untyped]

            img_array = np.array(img.convert("RGB"))
            results = reader.readtext(img_array)

            if results:
                texts = [entry[1] for entry in results]
                confidences = [entry[2] for entry in results]
                avg_confidence = sum(confidences) / len(confidences)
                return OCRResult(
                    text="\n".join(texts),
                    confidence=round(avg_confidence, 4),
                    engine="easyocr",
                    language=language,
                    warnings=result_warnings,
                )
            else:
                result_warnings.append("EasyOCR returned no text")
                if not _HAS_TESSERACT:
                    return OCRResult(
                        text="",
                        confidence=0.0,
                        engine="easyocr",
                        language=language,
                        warnings=result_warnings,
                    )
        except Exception as exc:
            logger.warning("EasyOCR failed, trying Tesseract: %s", exc)
            result_warnings.append("EasyOCR error: {}".format(exc))
            if not _HAS_TESSERACT:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    engine="none",
                    language=language,
                    warnings=result_warnings,
                )

    # --- Fall back to pytesseract --------------------------------------------
    if _HAS_TESSERACT:
        try:
            tess_lang = _iso_to_tesseract_lang(language)
            text = pytesseract.image_to_string(img, lang=tess_lang)
            confidence = _tesseract_confidence(img, tess_lang)
            return OCRResult(
                text=text.strip(),
                confidence=confidence,
                engine="tesseract",
                language=language,
                warnings=result_warnings,
            )
        except Exception as exc:
            result_warnings.append("Tesseract error: {}".format(exc))
            return OCRResult(
                text="",
                confidence=0.0,
                engine="none",
                language=language,
                warnings=result_warnings,
            )

    # Should not reach here, but just in case
    return OCRResult(
        text="",
        confidence=0.0,
        engine="none",
        language=language,
        warnings=result_warnings + ["No OCR engine produced output"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal ISO 639-1 -> Tesseract language mapping
_ISO_TO_TESS: dict[str, str] = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
    "ar": "ara",
    "ru": "rus",
    "hi": "hin",
    "nl": "nld",
    "sv": "swe",
    "pl": "pol",
    "tr": "tur",
}


def _iso_to_tesseract_lang(iso_code: str) -> str:
    """Convert ISO 639-1 language code to Tesseract language code."""
    return _ISO_TO_TESS.get(iso_code, iso_code)


def _tesseract_confidence(img, lang: str) -> float:
    """Compute average word-level confidence from Tesseract OSD data.

    Returns 0.0 if detailed data cannot be obtained.
    """
    try:
        data = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT
        )
        confs = [
            int(c)
            for c in data.get("conf", [])
            if str(c).lstrip("-").isdigit() and int(c) >= 0
        ]
        if confs:
            return round(sum(confs) / len(confs) / 100.0, 4)
    except Exception:
        logger.debug("Tesseract confidence calculation failed", exc_info=True)
    return 0.0


def _decode_exif_value(value) -> str:
    """Decode an EXIF tag value to a string.

    EXIF values can be ``str``, ``bytes``, ``int``, or other types.
    Windows XP tags (XPTitle, XPComment, XPSubject) are stored as
    UTF-16LE encoded bytes.  ``UserComment`` may have an 8-byte
    charset prefix (ASCII / JIS / Unicode / Undefined).
    """
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, bytes):
        # UserComment: first 8 bytes are charset identifier
        if len(value) > 8:
            charset_id = value[:8]
            if charset_id == b"UNICODE\x00":
                try:
                    return value[8:].decode("utf-16le", errors="replace").strip("\x00 ")
                except Exception:
                    pass
            elif charset_id == b"ASCII\x00\x00\x00":
                try:
                    return value[8:].decode("ascii", errors="replace").strip("\x00 ")
                except Exception:
                    pass
            elif charset_id == b"JIS\x00\x00\x00\x00\x00":
                try:
                    return value[8:].decode("iso2022_jp", errors="replace").strip("\x00 ")
                except Exception:
                    pass
            elif charset_id == b"\x00\x00\x00\x00\x00\x00\x00\x00":
                # Undefined charset — try UTF-8 then latin-1
                try:
                    return value[8:].decode("utf-8", errors="strict").strip("\x00 ")
                except Exception:
                    try:
                        return value[8:].decode("latin-1", errors="replace").strip("\x00 ")
                    except Exception:
                        pass

        # Detect UTF-16LE: if every other byte is \x00, try UTF-16LE first.
        # This handles Windows XP tags which are UTF-16LE encoded.
        if len(value) >= 2 and b"\x00" in value:
            # Heuristic: if >30% of bytes are \x00, likely UTF-16LE
            null_ratio = value.count(b"\x00"[0]) / len(value)
            if null_ratio > 0.3:
                try:
                    decoded = value.decode("utf-16le", errors="replace").strip("\x00 ")
                    if decoded:
                        return decoded
                except Exception:
                    pass

        # Try UTF-8 (common for ImageDescription and similar)
        try:
            decoded_utf8 = value.decode("utf-8", errors="strict").strip("\x00 ")
            if decoded_utf8:
                return decoded_utf8
        except (UnicodeDecodeError, Exception):
            pass

        # Windows XP tags are UTF-16LE, often null-terminated (fallback)
        try:
            decoded = value.decode("utf-16le", errors="replace").strip("\x00 ")
            if decoded:
                return decoded
        except Exception:
            pass

        # Last resort: UTF-8 with replacement chars
        try:
            return value.decode("utf-8", errors="replace").strip("\x00 ")
        except Exception:
            pass

    if isinstance(value, (int, float)):
        return ""  # numeric tags are not text

    return str(value).strip() if value else ""


def _extract_xmp_text(raw_bytes: bytes) -> tuple[list[str], list[str]]:
    """Extract text from XMP metadata embedded in raw image bytes.

    XMP is an XML block that can appear in JPEG, TIFF, PNG, and other
    formats.  We search for ``<x:xmpmeta>`` blocks and extract
    ``dc:description`` and ``dc:title`` values.

    Returns
    -------
    tuple of (list[str], list[str])
        ``(extracted_texts, field_names)``
    """
    texts: list[str] = []
    fields: list[str] = []

    xmp_match = _XMP_BLOCK_RE.search(raw_bytes)
    if not xmp_match:
        return texts, fields

    xmp_block = xmp_match.group(0)

    # Extract dc:description (all language variants + CDATA)
    desc_match = _XMP_DC_DESC_RE.search(xmp_block)
    if desc_match:
        desc_texts = _extract_rdf_li_texts(desc_match.group(1))
        if desc_texts:
            texts.extend(desc_texts)
            fields.append("xmp:dc:description")

    # Extract dc:title (all language variants + CDATA)
    title_match = _XMP_DC_TITLE_RE.search(xmp_block)
    if title_match:
        title_texts = _extract_rdf_li_texts(title_match.group(1))
        if title_texts:
            texts.extend(title_texts)
            fields.append("xmp:dc:title")

    return texts, fields


def _extract_rdf_li_texts(rdf_block: bytes) -> list[str]:
    """Extract text from all ``<rdf:li>`` entries in a dc: field block.

    Handles both plain text and ``<![CDATA[...]]>`` wrapped content.
    Returns a list of non-empty strings (one per language variant).
    """
    results: list[str] = []
    for cdata_text, plain_text in _XMP_RDF_LI_RE.findall(rdf_block):
        raw = cdata_text or plain_text
        if raw:
            try:
                decoded = raw.decode("utf-8", errors="replace").strip()
                if decoded:
                    results.append(decoded)
            except Exception:
                pass
    return results
