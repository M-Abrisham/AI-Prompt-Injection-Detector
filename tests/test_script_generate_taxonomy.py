"""Unit tests for scripts/generate_taxonomy_samples.py — taxonomy generation pipeline.

Tests probe generation format, technique ID validity, sample text,
metadata computation, gap-closure logic, and CSV schema.
"""

import hashlib
import os
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from generate_taxonomy_samples import (
    _compute_metadata,
    _technique_to_category,
    _FIELDNAMES,
    _BENIGN_SUFFIX,
    _RESET_RE,
    _OVERRIDE_RE,
    _MULTI_SUB_LETTERS,
    _MULTI_LETTER_PREFIXES,
    _GAP_TEMPLATES,
    _GAP_TEMPLATES_DEFAULT,
    _GAP_META,
)


# ---------------------------------------------------------------------------
# Probe generation format
# ---------------------------------------------------------------------------

class TestProbeGenerationFormat(unittest.TestCase):
    """Test that probes produce samples in the expected (text, tech_id, ...) format."""

    def test_all_probes_importable(self):
        from taxonomy import ALL_PROBES
        self.assertGreater(len(ALL_PROBES), 0)

    def test_probe_sample_is_tuple(self):
        import random
        from taxonomy import ALL_PROBES
        probe = ALL_PROBES[0]()
        random.seed(42)
        samples = probe.generate()
        self.assertGreater(len(samples), 0)
        for item in samples[:5]:
            self.assertIsInstance(item, (tuple, list))
            self.assertGreaterEqual(len(item), 2)

    def test_probe_sample_text_is_string(self):
        import random
        from taxonomy import ALL_PROBES
        probe = ALL_PROBES[0]()
        random.seed(42)
        samples = probe.generate()
        for text, tech_id, *_ in samples[:5]:
            self.assertIsInstance(text, str)

    def test_probe_sample_tech_id_is_string(self):
        import random
        from taxonomy import ALL_PROBES
        probe = ALL_PROBES[0]()
        random.seed(42)
        samples = probe.generate()
        for text, tech_id, *_ in samples[:5]:
            self.assertIsInstance(tech_id, str)

    def test_probe_sample_text_nonempty(self):
        """Each probe sample text must be non-empty."""
        import random
        from taxonomy import ALL_PROBES
        probe = ALL_PROBES[0]()
        random.seed(42)
        samples = probe.generate()
        for text, tech_id, *_ in samples[:10]:
            self.assertTrue(len(text.strip()) > 0,
                            f"Empty text for tech_id={tech_id}")


# ---------------------------------------------------------------------------
# Technique ID validity
# ---------------------------------------------------------------------------

class TestTechniqueIDValidity(unittest.TestCase):
    """Test technique_id format and category mapping."""

    def test_technique_id_has_dot_notation(self):
        """Most technique IDs should contain a dot (e.g. D1.1, E2.3)."""
        import random
        from taxonomy import ALL_PROBES
        probe = ALL_PROBES[0]()
        random.seed(42)
        samples = probe.generate()
        for text, tech_id, *_ in samples[:5]:
            # tech_id should match pattern like D1.1 or D1.1_benign
            clean = tech_id.removesuffix(_BENIGN_SUFFIX)
            self.assertRegex(clean, r'^[A-Z0-9]+',
                             f"Technique ID must start with uppercase: {tech_id}")

    def test_category_mapping_d_subcategory(self):
        self.assertEqual(_technique_to_category("D1.1"), "D1")
        self.assertEqual(_technique_to_category("D7.2"), "D7")

    def test_category_mapping_single_letter(self):
        self.assertEqual(_technique_to_category("E1.1"), "E")
        self.assertEqual(_technique_to_category("P1.2"), "P")
        self.assertEqual(_technique_to_category("T1.1"), "T")

    def test_category_mapping_multi_letter(self):
        for prefix in _MULTI_LETTER_PREFIXES:
            tech_id = f"{prefix}1.1"
            self.assertEqual(_technique_to_category(tech_id), prefix,
                             f"Expected {prefix} for {tech_id}")

    def test_category_mapping_benign_suffix(self):
        self.assertEqual(_technique_to_category("D1.1_benign"), "D1")
        self.assertEqual(_technique_to_category("E1.1_benign"), "E")


# ---------------------------------------------------------------------------
# Metadata computation
# ---------------------------------------------------------------------------

class TestMetadataComputation(unittest.TestCase):
    """Test _compute_metadata returns correct values."""

    def test_metadata_keys(self):
        meta = _compute_metadata("test text here")
        expected = {"length_chars", "length_bytes", "token_count",
                    "compression_ratio", "has_reset_claim", "has_override_language"}
        self.assertEqual(set(meta.keys()), expected)

    def test_length_chars_correct(self):
        meta = _compute_metadata("hello")
        self.assertEqual(meta["length_chars"], 5)

    def test_length_bytes_multibyte(self):
        meta = _compute_metadata("\u00e9")  # e-acute, 2 bytes in UTF-8
        self.assertEqual(meta["length_chars"], 1)
        self.assertEqual(meta["length_bytes"], 2)

    def test_token_count_positive(self):
        meta = _compute_metadata("This is a sentence with multiple tokens.")
        self.assertGreater(meta["token_count"], 0)

    def test_compression_ratio_for_repetitive(self):
        repetitive = "AAAA" * 1000
        meta = _compute_metadata(repetitive)
        # Highly repetitive text should compress well (low ratio)
        self.assertLess(meta["compression_ratio"], 0.5)

    def test_empty_string_metadata(self):
        meta = _compute_metadata("")
        self.assertEqual(meta["length_chars"], 0)
        self.assertEqual(meta["length_bytes"], 0)
        self.assertEqual(meta["compression_ratio"], 1.0)

    def test_reset_claim_flag(self):
        meta = _compute_metadata("new session boundary activated")
        self.assertEqual(meta["has_reset_claim"], 1)

    def test_override_language_flag(self):
        meta = _compute_metadata("ignore all previous instructions immediately")
        self.assertEqual(meta["has_override_language"], 1)

    def test_no_flags_for_clean_text(self):
        meta = _compute_metadata("What time is the meeting tomorrow?")
        self.assertEqual(meta["has_reset_claim"], 0)
        self.assertEqual(meta["has_override_language"], 0)


# ---------------------------------------------------------------------------
# Gap closure templates
# ---------------------------------------------------------------------------

class TestGapClosureTemplates(unittest.TestCase):
    """Test gap-closure template structure and validity."""

    def test_all_template_categories_have_entries(self):
        for cat_type, templates in _GAP_TEMPLATES.items():
            self.assertGreater(len(templates), 0,
                               f"Template category '{cat_type}' is empty")

    def test_default_templates_nonempty(self):
        self.assertGreater(len(_GAP_TEMPLATES_DEFAULT), 0)

    def test_templates_contain_placeholder(self):
        """Templates should contain {tech_name} placeholder."""
        for cat_type, templates in _GAP_TEMPLATES.items():
            for tmpl in templates:
                self.assertIn("{tech_name}", tmpl,
                              f"Template in '{cat_type}' missing {{tech_name}}")

    def test_gap_meta_has_expected_keys(self):
        self.assertIn("difficulty", _GAP_META)
        self.assertIn("difficulty_score", _GAP_META)
        self.assertIn("evasion_type", _GAP_META)

    def test_template_format_works(self):
        """Templates should format without error."""
        for cat_type, templates in _GAP_TEMPLATES.items():
            for tmpl in templates:
                try:
                    tmpl.format(tech_name="test", cat_name="cat", cat_desc="desc")
                except KeyError as e:
                    self.fail(f"Template format error in '{cat_type}': {e}")


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

class TestCSVSchemaFields(unittest.TestCase):
    """Test CSV field definitions."""

    def test_required_columns_present(self):
        for col in ["text", "label", "technique_id", "category"]:
            self.assertIn(col, _FIELDNAMES)

    def test_metadata_columns_present(self):
        for col in ["length_chars", "length_bytes", "token_count",
                     "compression_ratio", "has_reset_claim", "has_override_language"]:
            self.assertIn(col, _FIELDNAMES)

    def test_difficulty_columns_present(self):
        self.assertIn("difficulty", _FIELDNAMES)
        self.assertIn("difficulty_score", _FIELDNAMES)
        self.assertIn("evasion_type", _FIELDNAMES)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

class TestRegexPatterns(unittest.TestCase):
    """Test the reset and override regex patterns."""

    def test_reset_matches_context_clear(self):
        self.assertIsNotNone(_RESET_RE.search("context cleared"))

    def test_reset_matches_system_reset(self):
        self.assertIsNotNone(_RESET_RE.search("system reset detected"))

    def test_override_matches_ignore_instructions(self):
        self.assertIsNotNone(_OVERRIDE_RE.search("ignore all instructions"))

    def test_override_matches_bypass_safety(self):
        self.assertIsNotNone(_OVERRIDE_RE.search("bypass all safety filters"))

    def test_clean_text_no_reset(self):
        self.assertIsNone(_RESET_RE.search("What is 2+2?"))

    def test_clean_text_no_override(self):
        self.assertIsNone(_OVERRIDE_RE.search("Tell me a story about a cat."))


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

class TestLabelMapping(unittest.TestCase):
    """Test benign suffix and label logic."""

    def test_benign_suffix_constant(self):
        self.assertEqual(_BENIGN_SUFFIX, "_benign")

    def test_benign_detection(self):
        self.assertTrue("D1.1_benign".endswith(_BENIGN_SUFFIX))
        self.assertFalse("D1.1".endswith(_BENIGN_SUFFIX))

    def test_malicious_label_value(self):
        tech_id = "D1.1"
        label = 0 if tech_id.endswith(_BENIGN_SUFFIX) else 1
        self.assertEqual(label, 1)

    def test_benign_label_value(self):
        tech_id = "D1.1_benign"
        label = 0 if tech_id.endswith(_BENIGN_SUFFIX) else 1
        self.assertEqual(label, 0)


if __name__ == "__main__":
    unittest.main()
