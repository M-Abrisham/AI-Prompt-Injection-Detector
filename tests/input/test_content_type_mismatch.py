"""Tests for content-type mismatch detection.

Verifies that declared content types (from HTTP headers / file extensions)
are compared against magic-byte detected types, and that mismatches are
flagged appropriately.

Run: python -m unittest tests/test_content_type_mismatch.py -v
"""

import unittest
from unittest.mock import patch, MagicMock

from na0s.layer0.content_type import ContentTypeResult
from na0s.layer0.sanitizer import (
    _check_content_type_mismatch,
    _get_mime_family,
    _MIME_FAMILY_MAP,
    _GENERIC_TYPES,
    layer0_sanitize,
)


# ---------------------------------------------------------------------------
# Helper to build minimal byte payloads with magic bytes
# ---------------------------------------------------------------------------

def _pad(header, length=64):
    """Pad *header* with null bytes to *length*."""
    return header + b"\x00" * (length - len(header))


# Well-known magic byte headers
_MZ_EXE = _pad(b"MZ", 128)               # PE executable
_PDF_MAGIC = _pad(b"%PDF-1.4 ", 128)      # PDF document
_PNG_MAGIC = _pad(b"\x89PNG\r\n\x1a\n", 128)  # PNG image
_JPEG_MAGIC = _pad(b"\xff\xd8\xff\xe0", 128)  # JPEG image


# ===================================================================
# Unit tests for _get_mime_family
# ===================================================================

class TestGetMimeFamily(unittest.TestCase):
    """Test MIME type to family resolution."""

    def test_text_plain(self):
        self.assertEqual(_get_mime_family("text/plain"), "text")

    def test_text_html(self):
        self.assertEqual(_get_mime_family("text/html"), "text")

    def test_application_pdf(self):
        self.assertEqual(_get_mime_family("application/pdf"), "document")

    def test_image_png(self):
        self.assertEqual(_get_mime_family("image/png"), "image")

    def test_audio_mpeg(self):
        self.assertEqual(_get_mime_family("audio/mpeg"), "audio")

    def test_video_mp4(self):
        self.assertEqual(_get_mime_family("video/mp4"), "video")

    def test_executable_dosexec(self):
        self.assertEqual(_get_mime_family("application/x-dosexec"), "executable")

    def test_archive_zip(self):
        self.assertEqual(_get_mime_family("application/zip"), "archive")

    def test_octet_stream_returns_none(self):
        """application/octet-stream is generic and should return None."""
        self.assertIsNone(_get_mime_family("application/octet-stream"))

    def test_none_returns_none(self):
        self.assertIsNone(_get_mime_family(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_get_mime_family(""))

    def test_mime_with_parameters_stripped(self):
        """Parameters like charset should be stripped before lookup."""
        self.assertEqual(
            _get_mime_family("text/html; charset=utf-8"), "text"
        )

    def test_fallback_to_top_level_image(self):
        """Unknown image/* subtypes should fall back to 'image' family."""
        self.assertEqual(_get_mime_family("image/x-custom-format"), "image")

    def test_fallback_to_top_level_audio(self):
        self.assertEqual(_get_mime_family("audio/x-custom"), "audio")

    def test_fallback_to_top_level_video(self):
        self.assertEqual(_get_mime_family("video/x-custom"), "video")

    def test_fallback_to_top_level_text(self):
        self.assertEqual(_get_mime_family("text/x-custom"), "text")

    def test_unknown_application_returns_none(self):
        """Unknown application/* types not in the map should return None."""
        self.assertIsNone(_get_mime_family("application/x-unknown-thing"))


# ===================================================================
# Unit tests for _check_content_type_mismatch
# ===================================================================

class TestCheckContentTypeMismatch(unittest.TestCase):
    """Test the mismatch detection function directly."""

    def test_mismatch_text_declared_exe_detected(self):
        """Declared text/plain but magic bytes say executable -> mismatch."""
        metadata = {"content_type": "text/plain"}
        ct = ContentTypeResult(
            detected_type="exe_pe",
            mime_type="application/x-dosexec",
            tier="CRITICAL",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertIn("content_type_mismatch", flags)
        # Mismatch details stored in metadata
        self.assertIn("content_type_mismatch", metadata)
        details = metadata["content_type_mismatch"]
        self.assertEqual(details["declared_family"], "text")
        self.assertEqual(details["detected_family"], "executable")

    def test_mismatch_image_declared_pdf_detected(self):
        """Declared image/png but magic bytes say PDF -> mismatch."""
        metadata = {"content_type": "image/png"}
        ct = ContentTypeResult(
            detected_type="pdf",
            mime_type="application/pdf",
            tier="HIGH",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertIn("content_type_mismatch", flags)
        details = metadata["content_type_mismatch"]
        self.assertEqual(details["declared_family"], "image")
        self.assertEqual(details["detected_family"], "document")

    def test_mismatch_text_declared_image_detected(self):
        """Declared text/plain but magic bytes say PNG image -> mismatch."""
        metadata = {"content_type": "text/plain"}
        ct = ContentTypeResult(
            detected_type="png",
            mime_type="image/png",
            tier="HIGH",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertIn("content_type_mismatch", flags)

    def test_mismatch_document_declared_archive_detected(self):
        """Declared application/pdf but magic bytes say ZIP -> mismatch."""
        metadata = {"content_type": "application/pdf"}
        ct = ContentTypeResult(
            detected_type="zip",
            mime_type="application/zip",
            tier="HIGH",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertIn("content_type_mismatch", flags)

    def test_no_mismatch_when_types_agree(self):
        """Both declared and detected are PDF -> no mismatch."""
        metadata = {"content_type": "application/pdf"}
        ct = ContentTypeResult(
            detected_type="pdf",
            mime_type="application/pdf",
            tier="HIGH",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertEqual(flags, [])
        self.assertNotIn("content_type_mismatch", metadata)

    def test_no_mismatch_same_family_different_subtype(self):
        """Declared image/jpeg, detected image/png -> same family, no mismatch."""
        metadata = {"content_type": "image/jpeg"}
        ct = ContentTypeResult(
            detected_type="png",
            mime_type="image/png",
            tier="HIGH",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertEqual(flags, [])

    def test_no_mismatch_for_octet_stream_declared(self):
        """Declared application/octet-stream is generic -> no mismatch."""
        metadata = {"content_type": "application/octet-stream"}
        ct = ContentTypeResult(
            detected_type="exe_pe",
            mime_type="application/x-dosexec",
            tier="CRITICAL",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertEqual(flags, [])

    def test_no_mismatch_for_octet_stream_detected(self):
        """Detected application/octet-stream is generic -> no mismatch."""
        metadata = {"content_type": "text/plain"}
        ct = ContentTypeResult(
            detected_type="unknown",
            mime_type="application/octet-stream",
            tier="",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertEqual(flags, [])

    def test_no_mismatch_when_no_declared_type(self):
        """No declared content type in metadata -> no mismatch."""
        metadata = {}
        ct = ContentTypeResult(
            detected_type="pdf",
            mime_type="application/pdf",
            tier="HIGH",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertEqual(flags, [])

    def test_no_mismatch_when_declared_is_none(self):
        """Declared content type is None -> no mismatch."""
        metadata = {"content_type": None}
        ct = ContentTypeResult(
            detected_type="pdf",
            mime_type="application/pdf",
            tier="HIGH",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertEqual(flags, [])

    def test_no_mismatch_when_no_detected_type(self):
        """No detection result (empty ct_result) -> no mismatch."""
        metadata = {"content_type": "text/plain"}
        ct = ContentTypeResult()  # empty, no detected type
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertEqual(flags, [])

    def test_no_mismatch_when_ct_result_is_none(self):
        """ct_result is None -> no mismatch."""
        metadata = {"content_type": "text/plain"}
        flags = _check_content_type_mismatch(metadata, None)
        self.assertEqual(flags, [])

    def test_declared_with_charset_param(self):
        """Declared type has charset parameter, should still detect mismatch."""
        metadata = {"content_type": "text/plain; charset=utf-8"}
        ct = ContentTypeResult(
            detected_type="pdf",
            mime_type="application/pdf",
            tier="HIGH",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertIn("content_type_mismatch", flags)


# ===================================================================
# Integration tests through layer0_sanitize
# ===================================================================

class TestMismatchIntegration(unittest.TestCase):
    """Test mismatch detection through the full sanitize pipeline.

    Uses mocked input_loader to simulate HTTP responses with mismatched
    content types.
    """

    @patch("na0s.layer0.sanitizer.load_input")
    def test_url_text_declared_exe_detected(self, mock_load):
        """URL declares text/plain but serves EXE bytes -> mismatch + reject."""
        mock_load.return_value = (
            _MZ_EXE,
            {
                "source_type": "url",
                "source_path": "https://example.com/payload",
                "content_type": "text/plain",
                "file_size": len(_MZ_EXE),
            },
        )
        result = layer0_sanitize("https://example.com/payload")
        # EXE should be rejected
        self.assertTrue(result.rejected)
        # Mismatch flag should still be present (added before rejection)
        self.assertIn("content_type_mismatch", result.anomaly_flags)
        # Check metadata details
        mismatch = result.source_metadata.get("content_type_mismatch")
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["declared_family"], "text")
        self.assertEqual(mismatch["detected_family"], "executable")

    @patch("na0s.layer0.sanitizer.load_input")
    def test_url_image_declared_pdf_detected(self, mock_load):
        """URL declares image/png but serves PDF bytes -> mismatch flagged."""
        mock_load.return_value = (
            _PDF_MAGIC,
            {
                "source_type": "url",
                "source_path": "https://example.com/image.png",
                "content_type": "image/png",
                "file_size": len(_PDF_MAGIC),
            },
        )
        result = layer0_sanitize("https://example.com/image.png")
        self.assertIn("content_type_mismatch", result.anomaly_flags)
        mismatch = result.source_metadata.get("content_type_mismatch")
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["declared_family"], "image")
        self.assertEqual(mismatch["detected_family"], "document")

    @patch("na0s.layer0.sanitizer.load_input")
    def test_url_no_mismatch_when_types_agree(self, mock_load):
        """URL declares image/png and serves PNG bytes -> no mismatch."""
        mock_load.return_value = (
            _PNG_MAGIC,
            {
                "source_type": "url",
                "source_path": "https://example.com/image.png",
                "content_type": "image/png",
                "file_size": len(_PNG_MAGIC),
            },
        )
        result = layer0_sanitize("https://example.com/image.png")
        self.assertNotIn("content_type_mismatch", result.anomaly_flags)

    @patch("na0s.layer0.sanitizer.load_input")
    def test_url_no_mismatch_for_octet_stream(self, mock_load):
        """URL declares application/octet-stream -> no mismatch (generic)."""
        mock_load.return_value = (
            _PDF_MAGIC,
            {
                "source_type": "url",
                "source_path": "https://example.com/data.bin",
                "content_type": "application/octet-stream",
                "file_size": len(_PDF_MAGIC),
            },
        )
        result = layer0_sanitize("https://example.com/data.bin")
        self.assertNotIn("content_type_mismatch", result.anomaly_flags)

    @patch("na0s.layer0.sanitizer.load_input")
    def test_url_no_mismatch_when_no_declared_type(self, mock_load):
        """URL with no Content-Type header -> no mismatch."""
        mock_load.return_value = (
            _PDF_MAGIC,
            {
                "source_type": "url",
                "source_path": "https://example.com/unknown",
                "content_type": None,
                "file_size": len(_PDF_MAGIC),
            },
        )
        result = layer0_sanitize("https://example.com/unknown")
        self.assertNotIn("content_type_mismatch", result.anomaly_flags)

    @patch("na0s.layer0.sanitizer.load_input")
    def test_file_text_declared_exe_detected(self, mock_load):
        """File extension says text/plain but content is EXE -> mismatch."""
        mock_load.return_value = (
            _MZ_EXE,
            {
                "source_type": "file",
                "source_path": "/tmp/payload.txt",
                "content_type": "text/plain",
                "file_size": len(_MZ_EXE),
            },
        )
        # Simulate file path loading by passing a string that triggers loading
        result = layer0_sanitize("https://example.com/payload.txt")
        self.assertTrue(result.rejected)
        self.assertIn("content_type_mismatch", result.anomaly_flags)

    def test_raw_bytes_no_declared_type_no_mismatch(self):
        """Raw bytes input (no source_metadata) -> no mismatch possible."""
        # Passing raw bytes directly does NOT go through input_loader,
        # so there is no declared content_type to compare against.
        result = layer0_sanitize(_PDF_MAGIC)
        self.assertNotIn("content_type_mismatch", result.anomaly_flags)


# ===================================================================
# Edge cases
# ===================================================================

class TestMismatchEdgeCases(unittest.TestCase):
    """Edge cases for mismatch detection."""

    def test_text_family_variations_no_mismatch(self):
        """text/html declared, text/x-shellscript detected -> both 'text' family."""
        metadata = {"content_type": "text/html"}
        ct = ContentTypeResult(
            detected_type="shebang",
            mime_type="text/x-shellscript",
            tier="CRITICAL",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        # Both are "text" family -> no mismatch at the family level.
        # (Shebang is still flagged/rejected by the executable detection,
        #  but the content-type families match.)
        self.assertEqual(flags, [])

    def test_document_family_variations_no_mismatch(self):
        """application/msword declared, application/pdf detected -> both 'document'."""
        metadata = {"content_type": "application/msword"}
        ct = ContentTypeResult(
            detected_type="pdf",
            mime_type="application/pdf",
            tier="HIGH",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertEqual(flags, [])

    def test_mismatch_audio_declared_video_detected(self):
        """audio/mpeg declared, video/webm detected -> different families."""
        metadata = {"content_type": "audio/mpeg"}
        ct = ContentTypeResult(
            detected_type="ebml",
            mime_type="video/webm",
            tier="MEDIUM",
        )
        flags = _check_content_type_mismatch(metadata, ct)
        self.assertIn("content_type_mismatch", flags)

    def test_mismatch_details_structure(self):
        """Verify the structure of mismatch details stored in metadata."""
        metadata = {"content_type": "text/plain"}
        ct = ContentTypeResult(
            detected_type="pdf",
            mime_type="application/pdf",
            tier="HIGH",
        )
        _check_content_type_mismatch(metadata, ct)
        details = metadata["content_type_mismatch"]
        self.assertIn("declared_type", details)
        self.assertIn("declared_family", details)
        self.assertIn("detected_type", details)
        self.assertIn("detected_family", details)
        self.assertEqual(details["declared_type"], "text/plain")
        self.assertEqual(details["detected_type"], "application/pdf")


if __name__ == "__main__":
    unittest.main()
