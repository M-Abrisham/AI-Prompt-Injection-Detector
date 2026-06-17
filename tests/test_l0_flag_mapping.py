"""Tests for L0 anomaly flag -> technique_id mappings in predict.py.

Verifies that all Layer 0 flags emitted by normalization.py, html_extractor.py,
encoding.py, and sanitizer.py are correctly mapped in predict.py's _L0_FLAG_MAP.

We read predict.py's source directly (like test_exif_metadata.py) to avoid
importing the full predict module, which requires model files.
"""
import os
import unittest


def _read_predict_source():
    """Read predict.py source once and cache it."""
    predict_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "na0s", "predict.py"
    )
    with open(predict_path, "r") as f:
        return f.read()


# Read once at module level for all tests
_PREDICT_SOURCE = _read_predict_source()


class TestNormalizationFlagMapping(unittest.TestCase):
    """Verify normalization.py stego/homoglyph flags are mapped."""

    def test_unicode_tag_stego_mapped_to_D5_8(self):
        # Tag-character smuggling is its own technique (was mislabeled D5.2).
        self.assertIn('"unicode_tag_stego": "D5.8"', _PREDICT_SOURCE)

    def test_variation_selector_stego_mapped_to_D5_9(self):
        # Variation-selector stego is its own technique (was mislabeled D5.2).
        self.assertIn('"variation_selector_stego": "D5.9"', _PREDICT_SOURCE)

    def test_mixed_script_homoglyphs_mapped_to_D5_3(self):
        self.assertIn('"mixed_script_homoglyphs": "D5.3"', _PREDICT_SOURCE)

    def test_mojibake_repaired_mapped_to_D5(self):
        self.assertIn('"mojibake_repaired": "D5"', _PREDICT_SOURCE)

    def test_ftfy_suspicious_correction_mapped_to_D5(self):
        self.assertIn('"ftfy_suspicious_correction": "D5"', _PREDICT_SOURCE)


class TestHtmlExtractorFlagMapping(unittest.TestCase):
    """Verify html_extractor.py depth flag is mapped."""

    def test_html_depth_exceeded_mapped_to_A1(self):
        self.assertIn('"html_depth_exceeded": "A1"', _PREDICT_SOURCE)


class TestEncodingBomFlagMapping(unittest.TestCase):
    """Verify encoding.py BOM flags are mapped to D4.

    encoding.py emits flags like 'bom_detected_utf-8-sig' dynamically.
    predict.py uses exact-match (.get()), so each variant needs its own entry.
    """

    def test_bom_detected_utf8sig_mapped_to_D4(self):
        self.assertIn('"bom_detected_utf-8-sig": "D4"', _PREDICT_SOURCE)

    def test_bom_detected_utf16le_mapped_to_D4(self):
        self.assertIn('"bom_detected_utf-16-le": "D4"', _PREDICT_SOURCE)

    def test_bom_detected_utf16be_mapped_to_D4(self):
        self.assertIn('"bom_detected_utf-16-be": "D4"', _PREDICT_SOURCE)

    def test_bom_detected_utf32le_mapped_to_D4(self):
        self.assertIn('"bom_detected_utf-32-le": "D4"', _PREDICT_SOURCE)

    def test_bom_detected_utf32be_mapped_to_D4(self):
        self.assertIn('"bom_detected_utf-32-be": "D4"', _PREDICT_SOURCE)

    def test_all_bom_variants_match_encoding_module(self):
        """Cross-check: every BOM encoding in encoding.py has a map entry."""
        # These are the exact encoding names from encoding.py _BOM_MAP
        bom_encodings = [
            "utf-8-sig", "utf-32-le", "utf-32-be", "utf-16-le", "utf-16-be",
        ]
        for enc in bom_encodings:
            flag = f"bom_detected_{enc}"
            with self.subTest(flag=flag):
                self.assertIn(f'"{flag}"', _PREDICT_SOURCE,
                              f"Missing _L0_FLAG_MAP entry for {flag}")


class TestSanitizerTimeoutFlagMapping(unittest.TestCase):
    """Verify sanitizer.py timeout flags are mapped to A1.1."""

    def test_timeout_normalize_mapped_to_A1_1(self):
        self.assertIn('"timeout_normalize": "A1.1"', _PREDICT_SOURCE)

    def test_timeout_html_mapped_to_A1_1(self):
        self.assertIn('"timeout_html": "A1.1"', _PREDICT_SOURCE)

    def test_timeout_tokenize_mapped_to_A1_1(self):
        self.assertIn('"timeout_tokenize": "A1.1"', _PREDICT_SOURCE)

    def test_timeout_pipeline_mapped_to_A1_1(self):
        self.assertIn('"timeout_pipeline": "A1.1"', _PREDICT_SOURCE)


class TestFlagMapConsumption(unittest.TestCase):
    """Verify the map is consumed correctly via .get() exact matching."""

    def test_map_uses_get_for_lookup(self):
        """Confirm predict.py uses _L0_FLAG_MAP.get(flag) for exact match."""
        self.assertIn("_L0_FLAG_MAP.get(flag)", _PREDICT_SOURCE)

    def test_all_new_flags_present_in_map_block(self):
        """Every flag we added must appear between _L0_FLAG_MAP = { and }."""
        new_flags = [
            "unicode_tag_stego",
            "variation_selector_stego",
            "mixed_script_homoglyphs",
            "mojibake_repaired",
            "ftfy_suspicious_correction",
            "html_depth_exceeded",
            "bom_detected_utf-8-sig",
            "bom_detected_utf-16-le",
            "bom_detected_utf-16-be",
            "bom_detected_utf-32-le",
            "bom_detected_utf-32-be",
            "timeout_normalize",
            "timeout_html",
            "timeout_tokenize",
            "timeout_pipeline",
        ]
        for flag in new_flags:
            with self.subTest(flag=flag):
                self.assertIn(f'"{flag}"', _PREDICT_SOURCE,
                              f"Flag '{flag}' missing from _L0_FLAG_MAP")


class TestFlagMapTechniqueIds(unittest.TestCase):
    """Verify each new flag maps to the correct technique ID."""

    _EXPECTED = {
        "unicode_tag_stego": "D5.8",
        "variation_selector_stego": "D5.9",
        "mixed_script_homoglyphs": "D5.3",
        "mojibake_repaired": "D5",
        "ftfy_suspicious_correction": "D5",
        "html_depth_exceeded": "A1",
        "bom_detected_utf-8-sig": "D4",
        "bom_detected_utf-16-le": "D4",
        "bom_detected_utf-16-be": "D4",
        "bom_detected_utf-32-le": "D4",
        "bom_detected_utf-32-be": "D4",
        "timeout_normalize": "A1.1",
        "timeout_html": "A1.1",
        "timeout_tokenize": "A1.1",
        "timeout_pipeline": "A1.1",
    }

    def test_all_mappings_correct(self):
        for flag, technique_id in self._EXPECTED.items():
            expected_entry = f'"{flag}": "{technique_id}"'
            with self.subTest(flag=flag, technique_id=technique_id):
                self.assertIn(expected_entry, _PREDICT_SOURCE,
                              f"Expected mapping {expected_entry} not found")


if __name__ == "__main__":
    unittest.main()
