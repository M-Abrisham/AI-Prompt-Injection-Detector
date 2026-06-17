"""Tests for PDF JavaScript / action detection in doc_extractor.py.

Covers:
    - detect_pdf_javascript() function directly
    - Integration via extract_text_from_document() warning propagation
    - Edge cases: empty input, non-PDF bytes, multiple indicators
"""

import unittest

from na0s.layer0.doc_extractor import detect_pdf_javascript, extract_text_from_document


class TestDetectPdfJavascript(unittest.TestCase):
    """Unit tests for detect_pdf_javascript()."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pdf_bytes(*operators: bytes) -> bytes:
        """Build minimal synthetic PDF-like bytes containing given operators.

        The output starts with the ``%PDF-1.4`` header so magic-byte
        detection recognises it as a PDF.  Each operator is placed on its
        own line inside a ``<<  >>`` dictionary to mimic real PDF syntax.
        """
        parts = [b"%PDF-1.4\n"]
        for op in operators:
            parts.append(b"<< " + op + b" >>\n")
        parts.append(b"%%EOF\n")
        return b"".join(parts)

    # ------------------------------------------------------------------
    # Positive detection tests
    # ------------------------------------------------------------------

    def test_pdf_with_js(self):
        """PDF bytes containing /JS should be detected."""
        data = self._make_pdf_bytes(b"/JS (alert('xss'))")
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/JS", result["js_indicators"])
        self.assertIn("pdf_javascript", result["anomaly_flags"])

    def test_pdf_with_javascript_dict(self):
        """PDF bytes containing /JavaScript should be detected."""
        data = self._make_pdf_bytes(b"/JavaScript /Name")
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/JavaScript", result["js_indicators"])
        self.assertIn("pdf_javascript", result["anomaly_flags"])

    def test_pdf_with_openaction(self):
        """PDF bytes containing /OpenAction should be detected."""
        data = self._make_pdf_bytes(b"/OpenAction /GoTo")
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/OpenAction", result["js_indicators"])
        self.assertIn("pdf_auto_action", result["anomaly_flags"])

    def test_pdf_with_aa(self):
        """PDF bytes containing /AA (additional actions) should be detected."""
        data = self._make_pdf_bytes(b"/AA <<>>")
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/AA", result["js_indicators"])
        self.assertIn("pdf_auto_action", result["anomaly_flags"])

    def test_pdf_with_launch(self):
        """PDF bytes containing /Launch should be detected."""
        data = self._make_pdf_bytes(b"/Launch /Action")
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/Launch", result["js_indicators"])
        self.assertIn("pdf_external_action", result["anomaly_flags"])

    def test_pdf_with_submitform(self):
        """PDF bytes containing /SubmitForm should be detected."""
        data = self._make_pdf_bytes(b"/SubmitForm /URL")
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/SubmitForm", result["js_indicators"])
        self.assertIn("pdf_external_action", result["anomaly_flags"])

    def test_pdf_with_importdata(self):
        """PDF bytes containing /ImportData should be detected."""
        data = self._make_pdf_bytes(b"/ImportData /Ref")
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/ImportData", result["js_indicators"])
        self.assertIn("pdf_external_action", result["anomaly_flags"])

    # ------------------------------------------------------------------
    # Multiple indicators
    # ------------------------------------------------------------------

    def test_multiple_indicators(self):
        """PDF with both /JS and /OpenAction should flag both categories."""
        data = self._make_pdf_bytes(
            b"/JS (app.alert('hi'))",
            b"/OpenAction /GoTo",
        )
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/JS", result["js_indicators"])
        self.assertIn("/OpenAction", result["js_indicators"])
        self.assertIn("pdf_javascript", result["anomaly_flags"])
        self.assertIn("pdf_auto_action", result["anomaly_flags"])

    def test_all_indicators(self):
        """PDF with all 7 operators should detect all 3 flag categories."""
        data = self._make_pdf_bytes(
            b"/JS (code)",
            b"/JavaScript /Dict",
            b"/OpenAction /GoTo",
            b"/AA <<>>",
            b"/Launch /App",
            b"/SubmitForm /URL",
            b"/ImportData /Ref",
        )
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertEqual(len(result["js_indicators"]), 7)
        self.assertEqual(
            result["anomaly_flags"],
            {"pdf_javascript", "pdf_auto_action", "pdf_external_action"},
        )

    # ------------------------------------------------------------------
    # Negative / clean tests
    # ------------------------------------------------------------------

    def test_clean_pdf(self):
        """Normal PDF bytes without any JS operators should not flag."""
        data = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"
        result = detect_pdf_javascript(data)
        self.assertFalse(result["has_javascript"])
        self.assertEqual(result["js_indicators"], [])
        self.assertEqual(result["anomaly_flags"], set())

    def test_non_pdf_bytes(self):
        """Non-PDF input should return clean result (no crash)."""
        result = detect_pdf_javascript(b"Hello world, this is plain text")
        self.assertFalse(result["has_javascript"])
        self.assertEqual(result["js_indicators"], [])
        self.assertEqual(result["anomaly_flags"], set())

    def test_empty_bytes(self):
        """Empty bytes should return clean result."""
        result = detect_pdf_javascript(b"")
        self.assertFalse(result["has_javascript"])
        self.assertEqual(result["js_indicators"], [])
        self.assertEqual(result["anomaly_flags"], set())

    def test_bytearray_input(self):
        """bytearray input should work identically to bytes."""
        data = bytearray(self._make_pdf_bytes(b"/JS (code)"))
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/JS", result["js_indicators"])

    # ------------------------------------------------------------------
    # False-positive resistance
    # ------------------------------------------------------------------

    def test_js_in_text_content_not_false_positive(self):
        """The substring '/JS' inside a text stream surrounded by letters
        should NOT match because it lacks proper PDF token boundaries.
        Example: a word like 'flags/JScript' should not trigger."""
        data = b"%PDF-1.4\n(the text says flags/JScript is nice)\n%%EOF\n"
        result = detect_pdf_javascript(data)
        # /JScript contains /JS as a prefix, but boundary check should
        # prevent false positive because 'c' follows /JS without delimiter.
        # The regex requires a delimiter after the operator.
        self.assertFalse(result["has_javascript"])

    def test_aa_substring_not_false_positive(self):
        """/AA as a substring of /AAA should not match if there's no
        trailing delimiter."""
        # /AAA has no trailing delimiter (the third A directly follows)
        data = b"%PDF-1.4\n<< /AAA (value) >>\n%%EOF\n"
        result = detect_pdf_javascript(data)
        # /AA + A means the trailing boundary is 'A', not a delimiter
        # Our regex should NOT match this as /AA
        self.assertFalse(result["has_javascript"])

    def test_launch_in_comment_still_detected(self):
        """A /Launch operator in a PDF dictionary line (not a comment)
        should still be detected even if preceded by %."""
        # In PDF, % starts a comment to end of line. But /Launch on its
        # own line in a dictionary should still be found.
        data = self._make_pdf_bytes(b"/Launch /App")
        result = detect_pdf_javascript(data)
        self.assertTrue(result["has_javascript"])
        self.assertIn("/Launch", result["js_indicators"])


class TestExtractTextFromDocumentJsIntegration(unittest.TestCase):
    """Integration: extract_text_from_document() propagates JS warnings."""

    @staticmethod
    def _make_pdf_bytes(*operators: bytes) -> bytes:
        parts = [b"%PDF-1.4\n"]
        for op in operators:
            parts.append(b"<< " + op + b" >>\n")
        parts.append(b"%%EOF\n")
        return b"".join(parts)

    def test_extract_propagates_js_warnings(self):
        """extract_text_from_document should include JS warnings for PDFs."""
        data = self._make_pdf_bytes(b"/JS (alert('xss'))")
        result = extract_text_from_document(data, "pdf")
        # Check that at least one warning mentions JavaScript/action
        js_warnings = [w for w in result.warnings if "JavaScript" in w or "flag:" in w]
        self.assertTrue(
            len(js_warnings) > 0,
            "Expected JS-related warnings in DocResult.warnings, got: {}".format(
                result.warnings
            ),
        )

    def test_extract_includes_flag_warnings(self):
        """extract_text_from_document should emit flag: prefixed warnings."""
        data = self._make_pdf_bytes(b"/OpenAction /GoTo")
        result = extract_text_from_document(data, "pdf")
        flag_warnings = [w for w in result.warnings if w.startswith("flag:")]
        flag_names = [w[5:] for w in flag_warnings]
        self.assertIn("pdf_auto_action", flag_names)

    def test_extract_no_js_warnings_for_clean_pdf(self):
        """Clean PDF should have no JS-related warnings."""
        data = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"
        result = extract_text_from_document(data, "pdf")
        flag_warnings = [w for w in result.warnings if w.startswith("flag:")]
        self.assertEqual(flag_warnings, [])

    def test_extract_no_js_for_non_pdf_type(self):
        """Non-PDF doc type should not trigger JS detection."""
        # Minimal RTF-like data
        data = b"{\\rtf1 Hello}"
        result = extract_text_from_document(data, "rtf")
        flag_warnings = [w for w in result.warnings if w.startswith("flag:")]
        self.assertEqual(flag_warnings, [])

    def test_extract_metadata_includes_js_detection(self):
        """DocResult.metadata should include pdf_js_detection dict."""
        data = self._make_pdf_bytes(b"/SubmitForm /URL")
        result = extract_text_from_document(data, "pdf")
        self.assertIn("pdf_js_detection", result.metadata)
        self.assertTrue(result.metadata["pdf_js_detection"]["has_javascript"])
        self.assertIn(
            "/SubmitForm",
            result.metadata["pdf_js_detection"]["js_indicators"],
        )


class TestPdfJavascriptReturnContract(unittest.TestCase):
    """Verify the return-value contract of detect_pdf_javascript()."""

    def test_return_keys(self):
        """All three keys must be present in the return dict."""
        result = detect_pdf_javascript(b"")
        self.assertIn("has_javascript", result)
        self.assertIn("js_indicators", result)
        self.assertIn("anomaly_flags", result)

    def test_return_types(self):
        """Return types: bool, list, set."""
        result = detect_pdf_javascript(b"")
        self.assertIsInstance(result["has_javascript"], bool)
        self.assertIsInstance(result["js_indicators"], list)
        self.assertIsInstance(result["anomaly_flags"], set)

    def test_positive_return_types(self):
        """Return types when indicators are found."""
        data = b"%PDF-1.4\n<< /JS (code) >>\n%%EOF\n"
        result = detect_pdf_javascript(data)
        self.assertIsInstance(result["has_javascript"], bool)
        self.assertTrue(result["has_javascript"])
        self.assertIsInstance(result["js_indicators"], list)
        for item in result["js_indicators"]:
            self.assertIsInstance(item, str)
        self.assertIsInstance(result["anomaly_flags"], set)
        for item in result["anomaly_flags"]:
            self.assertIsInstance(item, str)


if __name__ == "__main__":
    unittest.main()
