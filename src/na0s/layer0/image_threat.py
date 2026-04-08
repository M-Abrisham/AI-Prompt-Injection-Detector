"""Image threat detection beyond OCR text extraction.

Detects threats embedded in images that text extraction alone would miss:
- Steganographic markers (LSB patterns, trailer data after IEND/EOI)
- Polyglot files (image + HTML/JS/ZIP/PDF/ELF)
- EXIF/XMP metadata injection (script tags, SQL, shell commands)
- Adversarial perturbation markers (unusual pixel distributions)

Complements ``ocr_extractor.py`` which handles text extraction.
All imports are optional -- the module degrades gracefully when PIL
is not installed, performing only byte-level checks.
"""

from __future__ import annotations

import io
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

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

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImageThreatResult:
    """Result of image threat detection.

    Attributes:
        threat_score: Aggregate threat score in [0.0, 1.0].
        threats:      Short labels for each detected threat.
        details:      Per-detector detail dicts keyed by detector name.
    """

    threat_score: float = 0.0
    threats: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants / patterns
# ---------------------------------------------------------------------------

# PNG IEND chunk marker (4-byte length 0 + IEND tag + CRC)
_PNG_IEND = b"IEND"
_PNG_IEND_FULL = b"\x00\x00\x00\x00IEND\xae\x42\x60\x82"

# JPEG end-of-image marker
_JPEG_EOI = b"\xff\xd9"

# Polyglot signatures to search for after image data
_POLYGLOT_SIGS: list[tuple[bytes, str]] = [
    (b"<html", "html"),
    (b"<HTML", "html"),
    (b"<script", "javascript"),
    (b"<SCRIPT", "javascript"),
    (b"PK\x03\x04", "zip"),
    (b"%PDF", "pdf"),
    (b"\x7fELF", "elf"),
]

# EXIF injection patterns (case-insensitive matching on decoded text)
_EXIF_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"<script", re.IGNORECASE), "script_tag"),
    (re.compile(r"javascript\s*:", re.IGNORECASE), "javascript_uri"),
    (re.compile(r"onerror\s*=", re.IGNORECASE), "event_handler"),
    (re.compile(r"onload\s*=", re.IGNORECASE), "event_handler"),
    (re.compile(r"\bSELECT\b.*\bFROM\b", re.IGNORECASE | re.DOTALL), "sql_select"),
    (re.compile(r"\bUNION\b.*\bSELECT\b", re.IGNORECASE | re.DOTALL), "sql_union"),
    (re.compile(r"\bDROP\b\s+\b(TABLE|DATABASE)\b", re.IGNORECASE), "sql_drop"),
    (re.compile(r"\$\("), "shell_subshell"),
    (re.compile(r"`[^`]+`"), "shell_backtick"),
    (re.compile(r";\s*rm\b"), "shell_rm"),
    (re.compile(r"\|\s*nc\b"), "shell_netcat"),
]

# EXIF text tag IDs we inspect for injection
_EXIF_TEXT_TAGS: dict[int, str] = {
    269: "DocumentName",
    270: "ImageDescription",
    305: "Software",
    315: "Artist",
    33432: "Copyright",
    37510: "UserComment",
    40091: "XPTitle",
    40092: "XPComment",
    40093: "XPAuthor",
    40094: "XPKeywords",
    40095: "XPSubject",
}

# Max bytes we scan for trailer / polyglot data past the image end marker
_MAX_TRAILER_SCAN = 64 * 1024  # 64 KB


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_image_threat(data: bytes, filename: str = "") -> ImageThreatResult:
    """Analyse raw image bytes for embedded threats.

    Parameters
    ----------
    data:
        Raw bytes of the image file.
    filename:
        Optional filename hint (used for logging, not for detection).

    Returns
    -------
    ImageThreatResult
        Always returns a result -- never raises.
    """
    if not data:
        return ImageThreatResult()

    threats: List[str] = []
    details: dict = {}
    scores: list[float] = []

    # --- Polyglot detection (bytes only) ------------------------------------
    poly_result = _detect_polyglot(data)
    if poly_result:
        threats.append("polyglot_file")
        details["polyglot"] = poly_result
        scores.append(0.9)

    # --- Steganographic trailer detection (bytes only) ----------------------
    stego_result = _detect_stego_trailer(data)
    if stego_result:
        threats.append("stego_trailer")
        details["stego_trailer"] = stego_result
        scores.append(0.7)

    # --- EXIF injection detection -------------------------------------------
    exif_result = _detect_exif_injection(data)
    if exif_result:
        threats.append("exif_injection")
        details["exif_injection"] = exif_result
        scores.append(0.8)

    # --- File size vs dimensions ratio (PIL required) -----------------------
    ratio_result = _detect_size_dimension_anomaly(data)
    if ratio_result:
        threats.append("size_dimension_anomaly")
        details["size_dimension_anomaly"] = ratio_result
        scores.append(0.5)

    # --- Adversarial perturbation markers (PIL required) --------------------
    adv_result = _detect_adversarial(data)
    if adv_result:
        threats.append("adversarial_perturbation")
        details["adversarial"] = adv_result
        scores.append(0.6)

    # Aggregate: take the max individual score (threats are independent)
    threat_score = max(scores) if scores else 0.0

    return ImageThreatResult(
        threat_score=round(threat_score, 4),
        threats=threats,
        details=details,
    )


# ---------------------------------------------------------------------------
# Polyglot detection
# ---------------------------------------------------------------------------


def _detect_polyglot(data: bytes) -> dict | None:
    """Check if image bytes also contain another file format."""
    found: list[dict] = []

    for sig, fmt in _POLYGLOT_SIGS:
        # Skip if the signature is at byte 0 (that IS the file format)
        idx = data.find(sig, 1)
        if idx > 0:
            found.append({
                "format": fmt,
                "offset": idx,
                "signature": sig[:16].hex(),
            })

    if found:
        return {"formats_detected": [f["format"] for f in found], "matches": found}
    return None


# ---------------------------------------------------------------------------
# Steganographic trailer detection
# ---------------------------------------------------------------------------


def _detect_stego_trailer(data: bytes) -> dict | None:
    """Detect data appended after the image end marker."""
    results: dict = {}

    # --- PNG: data after IEND chunk -----------------------------------------
    iend_pos = data.find(_PNG_IEND_FULL)
    if iend_pos >= 0:
        end_of_image = iend_pos + len(_PNG_IEND_FULL)
        trailer_len = len(data) - end_of_image
        if trailer_len > 0:
            results["png_trailer"] = {
                "iend_offset": iend_pos,
                "trailer_bytes": trailer_len,
                "trailer_preview": data[end_of_image:end_of_image + 32].hex(),
            }

    # --- JPEG: data after EOI (FF D9) --------------------------------------
    if data[:3] == b"\xff\xd8\xff":
        # Search backwards for the last FFD9
        eoi_pos = data.rfind(_JPEG_EOI)
        if eoi_pos >= 2:
            end_of_image = eoi_pos + 2
            trailer_len = len(data) - end_of_image
            if trailer_len > 0:
                results["jpeg_trailer"] = {
                    "eoi_offset": eoi_pos,
                    "trailer_bytes": trailer_len,
                    "trailer_preview": data[end_of_image:end_of_image + 32].hex(),
                }

    if results:
        return results
    return None


# ---------------------------------------------------------------------------
# EXIF injection detection
# ---------------------------------------------------------------------------


def _detect_exif_injection(data: bytes) -> dict | None:
    """Detect malicious content in EXIF metadata fields."""
    injections: list[dict] = []

    # --- PIL-based EXIF extraction ------------------------------------------
    if _HAS_PIL:
        try:
            img = Image.open(io.BytesIO(data))
            exif_data = img.getexif()
            if exif_data:
                for tag_id, tag_name in _EXIF_TEXT_TAGS.items():
                    value = exif_data.get(tag_id)
                    if value is None:
                        continue
                    text = _exif_value_to_str(value)
                    if not text:
                        continue
                    for pattern, label in _EXIF_INJECTION_PATTERNS:
                        if pattern.search(text):
                            injections.append({
                                "field": tag_name,
                                "pattern": label,
                                "snippet": text[:200],
                            })
        except Exception as exc:
            logger.debug("EXIF injection scan error: %s", exc)

    # --- Raw-bytes XMP scan (works without PIL) -----------------------------
    xmp_start = data.find(b"<x:xmpmeta")
    if xmp_start >= 0:
        xmp_end = data.find(b"</x:xmpmeta>", xmp_start)
        if xmp_end >= 0:
            xmp_block = data[xmp_start:xmp_end + len(b"</x:xmpmeta>")]
            try:
                xmp_text = xmp_block.decode("utf-8", errors="replace")
            except Exception:
                xmp_text = ""
            if xmp_text:
                for pattern, label in _EXIF_INJECTION_PATTERNS:
                    if pattern.search(xmp_text):
                        injections.append({
                            "field": "xmp",
                            "pattern": label,
                            "snippet": xmp_text[:200],
                        })

    if injections:
        return {"injections": injections}
    return None


def _exif_value_to_str(value) -> str:
    """Convert an EXIF tag value to a string for injection scanning."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        # Try UTF-16LE (Windows XP tags), then UTF-8
        if len(value) >= 2 and value.count(0) / len(value) > 0.3:
            try:
                return value.decode("utf-16le", errors="replace").strip("\x00 ")
            except Exception:
                pass
        try:
            return value.decode("utf-8", errors="replace").strip("\x00 ")
        except Exception:
            return ""
    if isinstance(value, (int, float)):
        return ""
    return str(value) if value else ""


# ---------------------------------------------------------------------------
# Size / dimension anomaly detection
# ---------------------------------------------------------------------------


def _detect_size_dimension_anomaly(data: bytes) -> dict | None:
    """Flag files whose size is abnormally large relative to pixel count.

    A 100x100 PNG should not be 5 MB -- that suggests hidden data.
    Requires PIL.
    """
    if not _HAS_PIL:
        return None

    try:
        img = Image.open(io.BytesIO(data))
        width, height = img.size
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None

    pixel_count = width * height
    file_size = len(data)

    # Expected uncompressed size: ~3 bytes/pixel for RGB, ~4 for RGBA.
    # Compressed images are typically 5-20x smaller.
    # Flag if file size > 10 bytes per pixel (very generous threshold).
    bytes_per_pixel = file_size / pixel_count
    if bytes_per_pixel > 10.0:
        return {
            "width": width,
            "height": height,
            "file_size": file_size,
            "bytes_per_pixel": round(bytes_per_pixel, 2),
            "reason": "file_size_exceeds_expected_ratio",
        }

    return None


# ---------------------------------------------------------------------------
# Adversarial perturbation detection
# ---------------------------------------------------------------------------


def _detect_adversarial(data: bytes) -> dict | None:
    """Detect statistical signs of adversarial perturbation in pixel data.

    Checks for unusual pixel value distributions using entropy analysis.
    Requires PIL.
    """
    if not _HAS_PIL:
        return None

    try:
        img = Image.open(io.BytesIO(data))
        # Convert to grayscale for simpler analysis
        gray = img.convert("L")
        pixels = list(gray.getdata())
    except Exception:
        return None

    if len(pixels) < 64:
        return None

    # --- Pixel value entropy ------------------------------------------------
    counts = Counter(pixels)
    total = len(pixels)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )

    # --- High-frequency noise estimation ------------------------------------
    # Compare adjacent pixel differences; adversarial images tend to have
    # uniformly distributed small perturbations.
    width = img.size[0]
    diffs: list[int] = []
    for i in range(len(pixels) - 1):
        # Skip row boundaries for horizontal diff
        if (i + 1) % width != 0:
            diffs.append(abs(pixels[i + 1] - pixels[i]))

    if not diffs:
        return None

    avg_diff = sum(diffs) / len(diffs)
    diff_counts = Counter(diffs)
    diff_entropy = -sum(
        (c / len(diffs)) * math.log2(c / len(diffs))
        for c in diff_counts.values()
        if c > 0
    )

    # Thresholds tuned for natural images:
    # - Pixel entropy > 7.9 (near maximum 8.0) suggests uniform distribution
    # - Diff entropy > 7.0 with low avg_diff suggests adversarial noise
    suspicious = False
    reasons: list[str] = []

    if entropy > 7.9:
        suspicious = True
        reasons.append("near_uniform_pixel_distribution")

    if diff_entropy > 7.0 and avg_diff < 5.0:
        suspicious = True
        reasons.append("high_frequency_low_amplitude_noise")

    if suspicious:
        return {
            "pixel_entropy": round(entropy, 4),
            "diff_entropy": round(diff_entropy, 4),
            "avg_adjacent_diff": round(avg_diff, 4),
            "reasons": reasons,
        }

    return None
