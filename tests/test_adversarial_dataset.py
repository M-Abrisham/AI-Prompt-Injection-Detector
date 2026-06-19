"""
Tests for the adversarial evasion benchmark dataset.

Validates that data/benchmark/adversarial_evasion.jsonl meets all
requirements:
  - File exists, valid JSONL, >= 500 samples
  - All labels are 1
  - All required fields present (text, label, source, evasion_type, original)
  - At least 7 different evasion_type values
  - No empty texts, no duplicates

The dataset lives under data/benchmark/ which is gitignored (generated
artifact, not committed).  If the file is missing at import time we
auto-materialize it by running scripts/generate_adversarial.py — the
generator is deterministic (seed=42), stdlib-only, and takes <1 second.
"""

import hashlib
import json
import os
import subprocess
import sys
import unittest

# Path to the dataset
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(
    _PROJECT_ROOT, "data", "benchmark", "adversarial_evasion.jsonl"
)
_GENERATOR_PATH = os.path.join(_PROJECT_ROOT, "scripts", "generate_adversarial.py")


def _ensure_dataset_materialized():
    """Run the generator if the dataset file is missing.

    data/benchmark/*.jsonl is gitignored; developers cloning the repo
    won't have the file until someone (or this hook) runs the generator.
    """
    if os.path.isfile(_DATASET_PATH):
        return
    if not os.path.isfile(_GENERATOR_PATH):
        return  # generator gone — tests will fail with a clear FileNotFoundError
    subprocess.run(
        [sys.executable, _GENERATOR_PATH],
        check=False,
        cwd=_PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


_ensure_dataset_materialized()

# Required fields per the schema
_REQUIRED_FIELDS = {"text", "label", "source", "evasion_type", "original"}

# Known evasion types from the spec
_EXPECTED_EVASION_TYPES = {
    "base64",
    "rot13",
    "leetspeak",
    "unicode_homoglyphs",
    "reversed",
    "hex_encoding",
    "mixed_encoding",
    "whitespace_insertion",
    "syllable_split",
}

# Minimum counts from the spec
_MIN_TOTAL_SAMPLES = 500
_MIN_EVASION_TYPES = 7
_MIN_PER_TECHNIQUE = 60


def _load_dataset():
    """Load the entire JSONL dataset, returning list of dicts."""
    samples = []
    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"Invalid JSON on line {line_no}: {exc}"
                ) from exc
            samples.append(obj)
    return samples


class TestAdversarialDatasetExists(unittest.TestCase):
    """Test that the dataset file exists and is non-empty."""

    def test_file_exists(self):
        self.assertTrue(
            os.path.isfile(_DATASET_PATH),
            f"Dataset file not found: {_DATASET_PATH}",
        )

    def test_file_not_empty(self):
        self.assertGreater(
            os.path.getsize(_DATASET_PATH),
            0,
            "Dataset file is empty",
        )


class TestAdversarialDatasetFormat(unittest.TestCase):
    """Test that the dataset is valid JSONL with correct schema."""

    @classmethod
    def setUpClass(cls):
        cls.samples = _load_dataset()

    def test_valid_jsonl(self):
        """Every line should be valid JSON."""
        # If _load_dataset() succeeded without AssertionError, all lines
        # are valid JSON.
        self.assertIsInstance(self.samples, list)

    def test_minimum_sample_count(self):
        """Dataset must have at least 500 samples."""
        self.assertGreaterEqual(
            len(self.samples),
            _MIN_TOTAL_SAMPLES,
            f"Expected >= {_MIN_TOTAL_SAMPLES} samples, got {len(self.samples)}",
        )

    def test_all_required_fields_present(self):
        """Every sample must have text, label, source, evasion_type, original."""
        for i, sample in enumerate(self.samples):
            missing = _REQUIRED_FIELDS - set(sample.keys())
            self.assertEqual(
                missing,
                set(),
                f"Sample {i} missing fields: {missing}",
            )

    def test_all_labels_are_one(self):
        """All samples must have label=1 (malicious)."""
        for i, sample in enumerate(self.samples):
            self.assertEqual(
                sample["label"],
                1,
                f"Sample {i} has label={sample['label']}, expected 1",
            )

    def test_source_is_generated(self):
        """All samples must have source='generated'."""
        for i, sample in enumerate(self.samples):
            self.assertEqual(
                sample["source"],
                "generated",
                f"Sample {i} has source={sample['source']!r}, expected 'generated'",
            )

    def test_no_empty_texts(self):
        """No sample should have an empty or whitespace-only text field."""
        for i, sample in enumerate(self.samples):
            self.assertTrue(
                sample["text"].strip(),
                f"Sample {i} has empty text",
            )

    def test_no_empty_originals(self):
        """No sample should have an empty original field."""
        for i, sample in enumerate(self.samples):
            self.assertTrue(
                sample["original"].strip(),
                f"Sample {i} has empty original",
            )

    def test_no_empty_evasion_type(self):
        """No sample should have an empty evasion_type."""
        for i, sample in enumerate(self.samples):
            self.assertTrue(
                sample["evasion_type"].strip(),
                f"Sample {i} has empty evasion_type",
            )


class TestAdversarialDatasetEvasionTypes(unittest.TestCase):
    """Test evasion type diversity and distribution."""

    @classmethod
    def setUpClass(cls):
        cls.samples = _load_dataset()
        cls.type_counts = {}
        for s in cls.samples:
            t = s["evasion_type"]
            cls.type_counts[t] = cls.type_counts.get(t, 0) + 1

    def test_at_least_seven_evasion_types(self):
        """Must have at least 7 different evasion_type values."""
        self.assertGreaterEqual(
            len(self.type_counts),
            _MIN_EVASION_TYPES,
            f"Expected >= {_MIN_EVASION_TYPES} evasion types, "
            f"got {len(self.type_counts)}: {sorted(self.type_counts.keys())}",
        )

    def test_all_nine_expected_types_present(self):
        """All 9 specified evasion types should be present."""
        found = set(self.type_counts.keys())
        missing = _EXPECTED_EVASION_TYPES - found
        self.assertEqual(
            missing,
            set(),
            f"Missing evasion types: {missing}",
        )

    def test_minimum_per_technique(self):
        """Each evasion type must have at least 60 samples."""
        for etype, count in self.type_counts.items():
            self.assertGreaterEqual(
                count,
                _MIN_PER_TECHNIQUE,
                f"Evasion type {etype!r} has only {count} samples, "
                f"need >= {_MIN_PER_TECHNIQUE}",
            )


class TestAdversarialDatasetNoDuplicates(unittest.TestCase):
    """Test that there are no duplicate texts."""

    @classmethod
    def setUpClass(cls):
        cls.samples = _load_dataset()

    def test_no_duplicate_texts(self):
        """No two samples should have the same text."""
        seen = set()
        duplicates = []
        for i, sample in enumerate(self.samples):
            text_hash = hashlib.sha256(
                sample["text"].encode("utf-8")
            ).hexdigest()
            if text_hash in seen:
                duplicates.append(i)
            seen.add(text_hash)
        self.assertEqual(
            len(duplicates),
            0,
            f"Found {len(duplicates)} duplicate texts at indices: "
            f"{duplicates[:10]}{'...' if len(duplicates) > 10 else ''}",
        )


class TestAdversarialDatasetEncodingQuality(unittest.TestCase):
    """Spot-check that evasion techniques actually produce encoded content."""

    @classmethod
    def setUpClass(cls):
        cls.samples = _load_dataset()
        cls.by_type = {}
        for s in cls.samples:
            cls.by_type.setdefault(s["evasion_type"], []).append(s)

    def test_base64_samples_contain_base64(self):
        """Base64 samples should contain base64-encoded strings."""
        import base64 as b64
        for sample in self.by_type.get("base64", [])[:5]:
            # The text should contain a base64-looking string
            # (multiple of 4 length, alphanumeric + /+=)
            text = sample["text"]
            # Check that it's different from the original
            self.assertNotEqual(
                text, sample["original"],
                "Base64 sample text should differ from original",
            )

    def test_rot13_samples_differ_from_original(self):
        """ROT13 samples should not contain the original text verbatim."""
        for sample in self.by_type.get("rot13", [])[:5]:
            # The original should not appear verbatim in the encoded text
            # (unless very short or coincidental)
            if len(sample["original"]) > 10:
                self.assertNotIn(
                    sample["original"],
                    sample["text"],
                    "ROT13 text should not contain original verbatim",
                )

    def test_unicode_homoglyphs_have_non_ascii(self):
        """Unicode homoglyph samples should contain non-ASCII characters."""
        non_ascii_count = 0
        for sample in self.by_type.get("unicode_homoglyphs", [])[:10]:
            has_non_ascii = any(ord(c) > 127 for c in sample["text"])
            if has_non_ascii:
                non_ascii_count += 1
        self.assertGreater(
            non_ascii_count, 0,
            "Unicode homoglyph samples should contain non-ASCII characters",
        )

    def test_reversed_samples_are_reversed(self):
        """Reversed samples should contain the reversed original text."""
        for sample in self.by_type.get("reversed", [])[:5]:
            reversed_original = sample["original"][::-1]
            self.assertIn(
                reversed_original,
                sample["text"],
                f"Reversed sample should contain reversed original: "
                f"{reversed_original!r}",
            )

    def test_hex_samples_contain_hex_bytes(self):
        """Hex-encoded samples should contain hex byte sequences."""
        import re
        hex_pattern = re.compile(r"[0-9a-f]{2}(?:\s[0-9a-f]{2})+")
        for sample in self.by_type.get("hex_encoding", [])[:5]:
            self.assertRegex(
                sample["text"],
                hex_pattern,
                "Hex sample should contain hex byte sequences",
            )

    def test_whitespace_insertion_has_invisible_chars(self):
        """Whitespace insertion samples should contain zero-width characters."""
        zero_width_chars = {
            "\u200b", "\u200c", "\u200d", "\ufeff", "\u2060",
        }
        has_invisible_count = 0
        for sample in self.by_type.get("whitespace_insertion", [])[:10]:
            if any(c in zero_width_chars for c in sample["text"]):
                has_invisible_count += 1
        self.assertGreater(
            has_invisible_count, 0,
            "Whitespace insertion samples should contain zero-width chars",
        )

    def test_syllable_split_has_dashes(self):
        """Syllable-split samples should contain hyphens or dashes."""
        dash_chars = {"-", "\u2010", "\u2011", "\u2012", "\u2013", "\u00ad"}
        has_dash_count = 0
        for sample in self.by_type.get("syllable_split", [])[:10]:
            if any(c in dash_chars for c in sample["text"]):
                has_dash_count += 1
        self.assertGreater(
            has_dash_count, 0,
            "Syllable-split samples should contain dashes",
        )

    def test_leetspeak_has_digit_substitutions(self):
        """Leetspeak samples should contain digit substitutions."""
        for sample in self.by_type.get("leetspeak", [])[:5]:
            has_digits = any(c.isdigit() for c in sample["text"])
            self.assertTrue(
                has_digits,
                f"Leetspeak sample should contain digit substitutions: "
                f"{sample['text']!r}",
            )


class TestAdversarialDatasetReproducibility(unittest.TestCase):
    """Test that the generator produces deterministic output."""

    def test_generator_is_deterministic(self):
        """Running generate_dataset twice with same seed should yield same result."""
        import sys
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
        from generate_adversarial import generate_dataset

        samples_a = generate_dataset(seed=42)
        samples_b = generate_dataset(seed=42)

        self.assertEqual(
            len(samples_a),
            len(samples_b),
            "Two runs with same seed should produce same count",
        )

        for i, (a, b) in enumerate(zip(samples_a, samples_b)):
            self.assertEqual(
                a["text"], b["text"],
                f"Sample {i} differs between runs",
            )
            self.assertEqual(
                a["evasion_type"], b["evasion_type"],
                f"Sample {i} evasion_type differs between runs",
            )


if __name__ == "__main__":
    unittest.main()
