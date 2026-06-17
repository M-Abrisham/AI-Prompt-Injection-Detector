"""Binary content-type detection via magic byte signatures.

Provides tiered file-type detection for injection-relevant formats.
No external dependencies — all detection is manual byte comparison.

Tiers:
    CRITICAL — Executables: always reject (MZ, ELF, Mach-O, Mach-O Universal, Java class, WASM, shebang)
    HIGH     — Documents (PDF, RTF, OOXML, OLE2), Archives (ZIP, GZIP, 7z, RAR, BZIP2, XZ, LZMA),
               Images (PNG, JPEG, GIF, BMP, TIFF, WebP, PSD, ICO)
    Polyglot — Secondary signature scan after primary match (PDF+ZIP, JPEG+ZIP, etc.)
    MEDIUM   — Audio (WAV, MP3, FLAC, OGG, AAC, MIDI, AIFF),
               Video (AVI, WebM/MKV, MP4, FLV, WMV)
"""

import base64
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContentTypeResult:
    """Result of binary content-type detection.

    Attributes:
        detected_type: Short format name (e.g. "pdf", "png", "exe_pe") or "".
        mime_type: Approximate MIME string (e.g. "application/pdf") or "".
        tier: "CRITICAL", "HIGH", "MEDIUM", or "".
        category: Broad category flag such as "embedded_executable".
        flags: List of anomaly flags to propagate
               (e.g. ["embedded_pdf", "embedded_document"]).
        reject: True when the file should be rejected outright.
        reject_reason: Human-readable rejection reason (empty when not rejected).
    """

    detected_type: str = ""
    mime_type: str = ""
    tier: str = ""
    category: str = ""
    flags: list = field(default_factory=list)
    reject: bool = False
    reject_reason: str = ""


# ---------------------------------------------------------------------------
# Signature tables
# ---------------------------------------------------------------------------
# Each entry: (byte_prefix, offset, detected_type, mime, tier, category, sub_flag)
# offset = starting byte position for comparison (usually 0).
# RIFF/FORM containers and PK archives are handled by dedicated helpers.

_SIGNATURES = [
    # ---- CRITICAL: Executables ----
    (b"MZ",                0, "exe_pe",       "application/x-dosexec",     "CRITICAL", "embedded_executable", "embedded_exe"),
    (b"\x7fELF",           0, "exe_elf",      "application/x-elf",         "CRITICAL", "embedded_executable", "embedded_elf"),
    (b"\xfe\xed\xfa\xce",  0, "exe_macho32",  "application/x-mach-binary", "CRITICAL", "embedded_executable", "embedded_macho"),
    (b"\xfe\xed\xfa\xcf",  0, "exe_macho64",  "application/x-mach-binary", "CRITICAL", "embedded_executable", "embedded_macho"),
    (b"\xce\xfa\xed\xfe",  0, "exe_macho32r", "application/x-mach-binary", "CRITICAL", "embedded_executable", "embedded_macho"),
    (b"\xcf\xfa\xed\xfe",  0, "exe_macho64r", "application/x-mach-binary", "CRITICAL", "embedded_executable", "embedded_macho"),
    # NOTE: 0xCAFEBABE (java_class / macho_universal) handled above linear scan
    (b"\x00asm",           0, "wasm",         "application/wasm",          "CRITICAL", "embedded_executable", "embedded_wasm"),
    (b"#!/",               0, "shebang",      "text/x-shellscript",        "CRITICAL", "embedded_executable", "embedded_shebang"),

    # ---- HIGH: Documents ----
    (b"%PDF",              0, "pdf",  "application/pdf", "HIGH", "embedded_document", "embedded_pdf"),
    (b"{\\rtf",            0, "rtf",  "application/rtf", "HIGH", "embedded_document", "embedded_rtf"),
    # OLE2 compound (legacy DOC/XLS/PPT)
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", 0, "ole2", "application/x-ole-storage", "HIGH", "embedded_document", "embedded_ole2"),

    # ---- HIGH: Images ----
    (b"\x89PNG\r\n\x1a\n", 0, "png",     "image/png",                 "HIGH", "embedded_image", "embedded_png"),
    (b"\xff\xd8\xff",      0, "jpeg",    "image/jpeg",                "HIGH", "embedded_image", "embedded_jpeg"),
    (b"GIF87a",            0, "gif",     "image/gif",                 "HIGH", "embedded_image", "embedded_gif"),
    (b"GIF89a",            0, "gif",     "image/gif",                 "HIGH", "embedded_image", "embedded_gif"),
    (b"BM",                0, "bmp",     "image/bmp",                 "HIGH", "embedded_image", "embedded_bmp"),
    (b"II*\x00",           0, "tiff_le", "image/tiff",                "HIGH", "embedded_image", "embedded_tiff"),
    (b"MM\x00*",           0, "tiff_be", "image/tiff",                "HIGH", "embedded_image", "embedded_tiff"),
    (b"8BPS",              0, "psd",     "image/vnd.adobe.photoshop", "HIGH", "embedded_image", "embedded_psd"),
    (b"\x00\x00\x01\x00",  0, "ico",     "image/x-icon",              "HIGH", "embedded_image", "embedded_ico"),

    # ---- HIGH: Archives ----
    (b"\x1f\x8b\x08",     0, "gzip",  "application/gzip",             "HIGH", "embedded_archive", "embedded_gzip"),
    (b"7z\xbc\xaf'\x1c",  0, "7z",    "application/x-7z-compressed",  "HIGH", "embedded_archive", "embedded_7z"),
    (b"Rar!\x1a\x07",     0, "rar",   "application/x-rar-compressed", "HIGH", "embedded_archive", "embedded_rar"),
    (b"BZh",               0, "bzip2", "application/x-bzip2",          "HIGH", "embedded_archive", "embedded_bzip2"),
    (b"\xfd7zXZ\x00",      0, "xz",    "application/x-xz",             "HIGH", "embedded_archive", "embedded_xz"),
    (b"\x5d\x00\x00",      0, "lzma",  "application/x-lzma",           "HIGH", "embedded_archive", "embedded_lzma"),

    # ---- MEDIUM: Audio ----
    (b"ID3",               0, "mp3_id3",  "audio/mpeg",  "MEDIUM", "embedded_audio", "embedded_mp3"),
    (b"\xff\xfb",          0, "mp3_sync", "audio/mpeg",  "MEDIUM", "embedded_audio", "embedded_mp3"),
    (b"\xff\xf3",          0, "mp3_sync", "audio/mpeg",  "MEDIUM", "embedded_audio", "embedded_mp3"),
    (b"\xff\xf2",          0, "mp3_sync", "audio/mpeg",  "MEDIUM", "embedded_audio", "embedded_mp3"),
    (b"fLaC",              0, "flac",     "audio/flac",  "MEDIUM", "embedded_audio", "embedded_flac"),
    (b"OggS",              0, "ogg",      "audio/ogg",   "MEDIUM", "embedded_audio", "embedded_ogg"),
    (b"\xff\xf1",          0, "aac",      "audio/aac",   "MEDIUM", "embedded_audio", "embedded_aac"),
    (b"\xff\xf9",          0, "aac",      "audio/aac",   "MEDIUM", "embedded_audio", "embedded_aac"),
    (b"MThd",              0, "midi",     "audio/midi",  "MEDIUM", "embedded_audio", "embedded_midi"),

    # ---- MEDIUM: Video ----
    (b"\x1aE\xdf\xa3",    0, "ebml", "video/webm",      "MEDIUM", "embedded_video", "embedded_webm"),
    (b"FLV\x01",          0, "flv",  "video/x-flv",     "MEDIUM", "embedded_video", "embedded_flv"),
    (b"0&\xb2u",          0, "wmv",  "video/x-ms-wmv",  "MEDIUM", "embedded_video", "embedded_wmv"),
]

# RIFF container: bytes 0-3 = "RIFF", bytes 8-11 = sub-type
_RIFF_SUBTYPES = {
    b"WAVE": ("wav",  "audio/wav",       "MEDIUM", "embedded_audio", "embedded_wav"),
    b"AVI ": ("avi",  "video/x-msvideo", "MEDIUM", "embedded_video", "embedded_avi"),
    b"WEBP": ("webp", "image/webp",      "HIGH",   "embedded_image", "embedded_webp"),
}

# AIFF container: bytes 0-3 = "FORM", bytes 8-11 = "AIFF" or "AIFC"
_FORM_SUBTYPES = {
    b"AIFF": ("aiff", "audio/aiff", "MEDIUM", "embedded_audio", "embedded_aiff"),
    b"AIFC": ("aiff", "audio/aiff", "MEDIUM", "embedded_audio", "embedded_aiff"),
}

# PK (ZIP-based) sub-type detection: look for internal paths in first 2 KB
_PK_HEADER = b"PK\x03\x04"
_PK_OOXML_MARKERS = [
    (b"word/",           "docx",  "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    (b"xl/",             "xlsx",  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    (b"ppt/",            "pptx",  "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    (b"[Content_Types]", "ooxml", "application/vnd.openxmlformats-officedocument"),
    (b"content.xml",     "odf",   "application/vnd.oasis.opendocument"),
    (b"META-INF/",       "jar",   "application/java-archive"),
]

# MP4 / MOV: "ftyp" box at offset 4
_FTYP_MARKER = b"ftyp"


# ---------------------------------------------------------------------------
# Base64 / Data URI detection (string-level)
# ---------------------------------------------------------------------------

# Matches base64 blobs >= 64 chars (likely binary payload, not short tokens)
_BASE64_BLOB_RE = re.compile(
    r"(?:[A-Za-z0-9+/]{4}){16,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?",
)

# Data URI scheme:  data:[<mediatype>][;base64],<data>
_DATA_URI_RE = re.compile(
    r"data:([a-zA-Z0-9][a-zA-Z0-9!#$&\-^_.+]*"
    r"(?:/[a-zA-Z0-9][a-zA-Z0-9!#$&\-^_.+]*)?)?"
    r"(?:;[a-zA-Z0-9\-]+=[a-zA-Z0-9\-]+)*"
    r";base64,"
    r"([A-Za-z0-9+/=]{20,})",
    re.ASCII,
)


# ---------------------------------------------------------------------------
# Base64 decode + re-scan safety limits
# ---------------------------------------------------------------------------

# Maximum encoded base64 string length to attempt decoding (bytes).
# Base64 expands ~33%, so 1.5 MB encoded ≈ ~1 MB decoded.
_BASE64_MAX_ENCODED_LEN = 1_500_000

# Maximum decoded payload size we will inspect (1 MB).
_BASE64_MAX_DECODED_LEN = 1_048_576

# Category ➔ flag-suffix mapping used by _decode_and_rescan().
_CATEGORY_TO_HIDDEN_SUFFIX = {
    "embedded_executable": "executable",
    "embedded_document": "document",
    "embedded_image": "image",
    "embedded_archive": "archive",
    "embedded_audio": "audio",
    "embedded_video": "video",
}


def _decode_and_rescan(b64_string):
    """Decode a base64 string and run detect_content_type on the result.

    Returns a list of ``base64_hidden_*`` flags.  Returns an empty list
    when decoding fails, the payload exceeds safety limits, or no binary
    signature is found in the decoded bytes.
    """
    if len(b64_string) > _BASE64_MAX_ENCODED_LEN:
        return ["base64_payload_too_large"]

    try:
        decoded = base64.b64decode(b64_string, validate=True)
    except Exception:
        logger.debug("Base64 decode+rescan failed", exc_info=True)
        return []

    if len(decoded) > _BASE64_MAX_DECODED_LEN:
        return ["base64_payload_too_large"]

    result = detect_content_type(decoded)
    if not result.detected_type:
        return []

    flags = []
    # Map category to a human-friendly hidden-content flag
    suffix = _CATEGORY_TO_HIDDEN_SUFFIX.get(result.category, result.detected_type)
    flags.append("base64_hidden_{}".format(suffix))

    # CRITICAL tier always gets the executable flag
    if result.tier == "CRITICAL":
        if "base64_hidden_executable" not in flags:
            flags.append("base64_hidden_executable")

    return flags


# ---------------------------------------------------------------------------
# Core detection function (bytes input)
# ---------------------------------------------------------------------------

def detect_content_type(data):
    """Detect the real content type of raw bytes via magic-byte signatures.

    Checks tiers in priority order: CRITICAL > HIGH > MEDIUM.
    Returns a ContentTypeResult; ``result.detected_type`` is the empty
    string when no known binary signature is found (i.e. likely text).

    Args:
        data: bytes or bytearray to inspect.
    """
    if not isinstance(data, (bytes, bytearray)):
        return ContentTypeResult()

    if len(data) < 2:
        return ContentTypeResult()

    result = None

    # --- RIFF container (needs 12 bytes) ---
    if len(data) >= 12 and data[:4] == b"RIFF":
        sub = data[8:12]
        if sub in _RIFF_SUBTYPES:
            dtype, mime, tier, cat, subflag = _RIFF_SUBTYPES[sub]
            result = ContentTypeResult(
                detected_type=dtype, mime_type=mime, tier=tier,
                category=cat, flags=[subflag, cat],
            )
        else:
            # Unknown RIFF sub-type -- still flag it
            result = ContentTypeResult(
                detected_type="riff_unknown",
                mime_type="application/octet-stream",
                tier="MEDIUM", category="embedded_archive",
                flags=["embedded_riff_unknown", "embedded_archive"],
            )

    # --- FORM container (AIFF) ---
    if result is None and len(data) >= 12 and data[:4] == b"FORM":
        sub = data[8:12]
        if sub in _FORM_SUBTYPES:
            dtype, mime, tier, cat, subflag = _FORM_SUBTYPES[sub]
            result = ContentTypeResult(
                detected_type=dtype, mime_type=mime, tier=tier,
                category=cat, flags=[subflag, cat],
            )

    # --- PK / ZIP-based (OOXML, ODF, JAR, plain ZIP) ---
    if result is None and len(data) >= 4 and data[:4] == _PK_HEADER:
        result = _check_pk_archive(data)

    # --- MP4 / MOV: "ftyp" at offset 4 ---
    if result is None and len(data) >= 8 and data[4:8] == _FTYP_MARKER:
        result = ContentTypeResult(
            detected_type="mp4", mime_type="video/mp4",
            tier="MEDIUM", category="embedded_video",
            flags=["embedded_mp4", "embedded_video"],
        )

    # --- TAR: "ustar" at offset 257 ---
    if result is None and len(data) >= 262 and data[257:262] == b"ustar":
        result = ContentTypeResult(
            detected_type="tar", mime_type="application/x-tar",
            tier="HIGH", category="embedded_archive",
            flags=["embedded_tar", "embedded_archive"],
        )

    # --- Java class vs Mach-O Universal (fat binary) disambiguation ---
    # Both share magic 0xCAFEBABE.  Bytes 6-7 are the Java major version
    # (45-66 for Java 1.1 through 22).  Mach-O fat stores the arch count
    # in bytes 4-7 (big-endian uint32); typical values (2-5) fall well
    # outside the Java major-version range.
    _CAFEBABE = b"\xca\xfe\xba\xbe"
    if result is None and len(data) >= 4 and data[:4] == _CAFEBABE:
        is_java = False
        if len(data) >= 8:
            major = int.from_bytes(data[6:8], "big")
            is_java = 45 <= major <= 66
        else:
            # Not enough bytes to disambiguate — default to java_class
            is_java = True

        if is_java:
            result = ContentTypeResult(
                detected_type="java_class",
                mime_type="application/java",
                tier="CRITICAL", category="embedded_executable",
                flags=["embedded_java_class", "embedded_executable"],
                reject=True,
                reject_reason="Executable binary detected (java_class)",
            )
        else:
            result = ContentTypeResult(
                detected_type="macho_universal",
                mime_type="application/x-mach-binary",
                tier="CRITICAL", category="embedded_executable",
                flags=["embedded_macho", "embedded_executable"],
                reject=True,
                reject_reason="Executable binary detected (macho_universal)",
            )

    # --- Linear scan of signature table ---
    if result is None:
        for sig, offset, dtype, mime, tier, cat, subflag in _SIGNATURES:
            end = offset + len(sig)
            if len(data) >= end and data[offset:end] == sig:
                # -- BMP secondary validation (BUG-CT-3) --
                # Real BMP has a 14-byte header; bytes 6-9 are reserved and
                # must be \x00\x00\x00\x00.  Without this check, any text
                # starting with "BM" (e.g. "BMP specs") is a false positive.
                if dtype == "bmp":
                    if len(data) >= 10 and data[6:10] != b"\x00\x00\x00\x00":
                        continue
                # -- ICO secondary validation (BUG-CT-3) --
                # Real ICO has bytes 4-5 = image count (LE uint16, 1-255).
                # A count of 0 or >255 is not a valid ICO file.
                if dtype == "ico":
                    if len(data) >= 6:
                        img_count = int.from_bytes(data[4:6], "little")
                        if img_count == 0 or img_count > 255:
                            continue

                reject = (tier == "CRITICAL")
                reason = ""
                if reject:
                    reason = "Executable binary detected ({})".format(dtype)
                result = ContentTypeResult(
                    detected_type=dtype, mime_type=mime, tier=tier,
                    category=cat, flags=[subflag, cat],
                    reject=reject, reject_reason=reason,
                )
                break

    # --- Polyglot detection: scan for secondary signatures ---
    if result is not None:
        return _check_polyglot(data, result)

    return ContentTypeResult()


# ---------------------------------------------------------------------------
# Polyglot detection — secondary signature scan
# ---------------------------------------------------------------------------

# Tier priority for upgrading (higher index = higher priority)
_TIER_PRIORITY = {"": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _check_polyglot(data, result):
    """Scan for secondary magic bytes that indicate a polyglot file.

    A polyglot file is one that is valid in two or more formats
    simultaneously (e.g. a PDF that is also a valid ZIP archive).
    Attackers use polyglots to bypass content-type filters.

    Modifies *result* in place: adds ``"polyglot_detected"`` to flags
    and upgrades tier to at least HIGH when a secondary signature is found.
    """
    if not result.detected_type:
        return result

    primary = result.detected_type

    # --- Check for embedded PK/ZIP signature after byte 4 ---
    # (Skip if primary is already a ZIP-based format)
    _ZIP_TYPES = {"zip", "docx", "xlsx", "pptx", "ooxml", "odf", "jar"}
    if primary not in _ZIP_TYPES and len(data) > 8:
        # Search for PK\x03\x04 starting after byte 4
        if _PK_HEADER in data[4:]:
            result.flags.append("polyglot_detected")

    # --- Check for embedded %PDF if primary is NOT already PDF ---
    if primary != "pdf" and len(data) > 4:
        if b"%PDF" in data:
            result.flags.append("polyglot_detected")

    # Upgrade tier if polyglot was detected
    if "polyglot_detected" in result.flags:
        primary_priority = _TIER_PRIORITY.get(result.tier, 0)
        high_priority = _TIER_PRIORITY["HIGH"]
        if primary_priority < high_priority:
            result.tier = "HIGH"

    return result


def _check_pk_archive(data):
    """Disambiguate PK/ZIP-based formats by checking internal paths."""
    search_window = data[:2048]
    for marker, dtype, mime in _PK_OOXML_MARKERS:
        if marker in search_window:
            subflag = "embedded_{}".format(dtype)
            cat = "embedded_archive" if dtype == "jar" else "embedded_document"
            return ContentTypeResult(
                detected_type=dtype, mime_type=mime,
                tier="HIGH", category=cat,
                flags=[subflag, cat],
            )
    # Plain ZIP (no recognized internal structure)
    return ContentTypeResult(
        detected_type="zip", mime_type="application/zip",
        tier="HIGH", category="embedded_archive",
        flags=["embedded_zip", "embedded_archive"],
    )


# ---------------------------------------------------------------------------
# String-input convenience (for text pipeline)
# ---------------------------------------------------------------------------

def sniff_binary(text):
    """Check if a text string begins with binary magic bytes.

    Encodes the first ~300 characters to bytes and runs detection.
    Also checks for base64-encoded binary blobs and data URIs.

    Returns a list of anomaly flags (empty if plain text).
    """
    flags = []

    # Check raw-byte signatures in the first 300 chars
    head = text[:300]
    raw = head.encode("utf-8", errors="replace")
    result = detect_content_type(raw)
    if result.detected_type:
        flags.extend(result.flags)

    # Check for base64 blobs in the full text
    b64_match = _BASE64_BLOB_RE.search(text)
    if b64_match:
        flags.append("base64_blob_detected")
        # Decode and re-scan to discover hidden content types
        hidden_flags = _decode_and_rescan(b64_match.group(0))
        flags.extend(hidden_flags)

    # Check for data URIs
    match = _DATA_URI_RE.search(text)
    if match:
        flags.append("data_uri_detected")
        media_type = match.group(1) or ""
        if media_type:
            flags.append("data_uri_type_{}".format(
                media_type.replace("/", "_").lower()
            ))
        # Decode the base64 payload from the data URI and re-scan
        b64_payload = match.group(2)
        if b64_payload:
            hidden_flags = _decode_and_rescan(b64_payload)
            flags.extend(hidden_flags)

    return flags
