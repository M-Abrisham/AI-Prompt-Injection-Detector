"""
Tests for the malicious holdout corpus at data/holdout/malicious_holdout.jsonl.

Validates:
1. File exists and is valid JSONL
2. Has at least 200 samples
3. All labels are 1 (malicious)
4. Covers at least 8 different technique categories
5. All records have required fields (text, label, source, category)
6. No empty text fields
7. No duplicate texts
8. Category codes conform to the Na0S taxonomy
9. Unicode and encoding samples are well-formed
10. Source field is consistently "holdout"
"""

import json
import os
import pathlib
import unittest
from collections import Counter

import pytest

# Resolve the corpus path relative to the project root.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CORPUS_PATH = _PROJECT_ROOT / "data" / "holdout" / "malicious_holdout.jsonl"

# Skip every test in this module when the holdout data artifact is not
# materialized in the working tree. Regenerate with
# ``scripts/gen_malicious_holdout.py`` (or ``scripts/gen_all_datasets.py``)
# or pull it from the data remote before running these tests.
pytestmark = pytest.mark.skipif(
    not _CORPUS_PATH.exists(),
    reason=(
        f"holdout data not materialized at {_CORPUS_PATH} "
        "- run scripts/gen_malicious_holdout.py or dvc pull"
    ),
)

# Required fields for every record.
_REQUIRED_FIELDS = {"text", "label", "source", "category"}

# Minimum number of samples required overall.
_MIN_SAMPLES = 200

# Minimum number of distinct technique categories.
_MIN_CATEGORIES = 8

# Expected categories from the Na0S taxonomy (at least these should appear).
_EXPECTED_CATEGORIES = {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "E1", "E2", "C1", "O1"}

# Minimum samples per major category.
_MIN_PER_CATEGORY = 10


def _load_corpus():
    """Load the corpus once and return a list of parsed records."""
    records = []
    with open(_CORPUS_PATH, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"Invalid JSON on line {line_num}: {exc}"
                ) from exc
            records.append(record)
    return records


class TestCorpusFileExists(unittest.TestCase):
    """Test 1: File exists and is valid JSONL."""

    def test_file_exists(self):
        self.assertTrue(
            _CORPUS_PATH.exists(),
            f"Corpus file not found at {_CORPUS_PATH}",
        )

    def test_file_is_not_empty(self):
        self.assertGreater(
            _CORPUS_PATH.stat().st_size,
            0,
            "Corpus file is empty",
        )

    def test_all_lines_are_valid_json(self):
        """Every non-blank line must parse as valid JSON."""
        with open(_CORPUS_PATH, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    self.fail(f"Line {line_num} is not valid JSON: {exc}")


class TestMinimumSampleCount(unittest.TestCase):
    """Test 2: Has at least 200 samples."""

    def test_at_least_200_samples(self):
        records = _load_corpus()
        self.assertGreaterEqual(
            len(records),
            _MIN_SAMPLES,
            f"Expected at least {_MIN_SAMPLES} samples, got {len(records)}",
        )


class TestAllLabelsMalicious(unittest.TestCase):
    """Test 3: All labels are 1 (malicious)."""

    def test_all_labels_are_one(self):
        records = _load_corpus()
        for idx, rec in enumerate(records):
            self.assertEqual(
                rec.get("label"),
                1,
                f"Record {idx} has label={rec.get('label')}, expected 1",
            )

    def test_label_is_integer(self):
        records = _load_corpus()
        for idx, rec in enumerate(records):
            self.assertIsInstance(
                rec.get("label"),
                int,
                f"Record {idx} label is not an integer: {type(rec.get('label'))}",
            )


class TestCategoryDiversity(unittest.TestCase):
    """Test 4: Covers at least 8 different technique categories."""

    def test_minimum_category_count(self):
        records = _load_corpus()
        categories = {rec.get("category") for rec in records}
        self.assertGreaterEqual(
            len(categories),
            _MIN_CATEGORIES,
            f"Expected at least {_MIN_CATEGORIES} categories, got {len(categories)}: {sorted(categories)}",
        )

    def test_expected_categories_present(self):
        """All expected Na0S taxonomy categories should be represented."""
        records = _load_corpus()
        categories = {rec.get("category") for rec in records}
        missing = _EXPECTED_CATEGORIES - categories
        self.assertEqual(
            len(missing),
            0,
            f"Missing expected categories: {sorted(missing)}",
        )

    def test_minimum_samples_per_major_category(self):
        """Each expected category should have at least _MIN_PER_CATEGORY samples."""
        records = _load_corpus()
        category_counts = Counter(rec.get("category") for rec in records)
        under_minimum = {
            cat: count
            for cat, count in category_counts.items()
            if cat in _EXPECTED_CATEGORIES and count < _MIN_PER_CATEGORY
        }
        self.assertEqual(
            len(under_minimum),
            0,
            f"Categories with fewer than {_MIN_PER_CATEGORY} samples: {under_minimum}",
        )


class TestRequiredFields(unittest.TestCase):
    """Test 5: All records have required fields (text, label, source, category)."""

    def test_required_fields_present(self):
        records = _load_corpus()
        for idx, rec in enumerate(records):
            for field in _REQUIRED_FIELDS:
                self.assertIn(
                    field,
                    rec,
                    f"Record {idx} missing required field '{field}'",
                )


class TestNoEmptyText(unittest.TestCase):
    """Test 6: No empty text fields."""

    def test_no_empty_text(self):
        records = _load_corpus()
        for idx, rec in enumerate(records):
            text = rec.get("text", "")
            self.assertIsInstance(text, str, f"Record {idx} text is not a string")
            self.assertGreater(
                len(text.strip()),
                0,
                f"Record {idx} has empty or whitespace-only text",
            )

    def test_text_minimum_length(self):
        """Each text should have a reasonable minimum length (at least 5 chars)."""
        records = _load_corpus()
        for idx, rec in enumerate(records):
            self.assertGreaterEqual(
                len(rec.get("text", "")),
                5,
                f"Record {idx} text is suspiciously short: {rec.get('text', '')!r}",
            )


class TestNoDuplicateTexts(unittest.TestCase):
    """Test 7: No duplicate texts."""

    def test_no_duplicate_texts(self):
        records = _load_corpus()
        texts = [rec.get("text") for rec in records]
        seen = set()
        duplicates = []
        for i, text in enumerate(texts):
            if text in seen:
                duplicates.append((i, text[:80]))
            seen.add(text)
        self.assertEqual(
            len(duplicates),
            0,
            f"Found {len(duplicates)} duplicate texts: {duplicates[:5]}",
        )


class TestCategoryCodes(unittest.TestCase):
    """Test 8: Category codes conform to the Na0S taxonomy."""

    _VALID_CATEGORIES = {
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
        "E1", "E2", "C1", "O1", "P1",
        # Sub-categories are also acceptable
        "D1.1", "D1.2", "D1.3", "D2.1", "D2.2", "D3.1", "D3.2",
        "D4.1", "D4.2", "D4.3", "D4.4", "D4.5",
        "D5.1", "D5.2", "D5.3", "D5.4", "D5.5",
        "D6.1", "D6.2", "D6.3", "D6.4", "D6.5", "D6.6",
        "D7.1", "D7.2", "D7.3", "D7.4",
        "D8.1", "D8.2", "D8.3", "D8.4",
        "E1.1", "E1.2", "E1.3", "E1.4", "E1.5", "E1.6",
        "E2.1", "E2.2", "E2.3", "E2.4", "E2.5",
        "C1.1", "C1.2", "C1.3", "C1.4", "C1.5",
        "O1.1", "O1.2", "O1.3", "O1.4", "O1.5",
        "O2.1", "O2.2", "O2.3",
        "P1.1", "P1.2", "P1.3", "P1.4", "P1.5", "P1.6",
    }

    def test_all_categories_are_valid(self):
        records = _load_corpus()
        invalid = set()
        for rec in records:
            cat = rec.get("category", "")
            if cat not in self._VALID_CATEGORIES:
                invalid.add(cat)
        self.assertEqual(
            len(invalid),
            0,
            f"Invalid category codes found: {sorted(invalid)}",
        )


class TestSourceField(unittest.TestCase):
    """Test 9: Source field is consistently 'generated'.

    The malicious holdout corpus is produced by scripts/gen_all_datasets.py,
    which (like gen_malicious_holdout.py) stamps every malicious record with
    source="generated". This asserts that single canonical value; the earlier
    expectation of "holdout" was stale — neither malicious-corpus generator
    emits it.
    """

    def test_source_is_generated(self):
        records = _load_corpus()
        sources = {rec.get("source") for rec in records}
        self.assertEqual(
            sources,
            {"generated"},
            f"Expected all sources to be 'generated', got: {sources}",
        )


class TestContentQuality(unittest.TestCase):
    """Test 10: Content quality checks."""

    def test_encoding_samples_are_real_encodings(self):
        """D4 base64 samples should contain real base64 strings."""
        import base64 as b64
        records = _load_corpus()
        d4_records = [r for r in records if r.get("category") == "D4"]
        base64_count = 0
        for rec in d4_records:
            text = rec["text"]
            # Look for base64-like strings (at least 16 chars of valid base64)
            import re
            b64_matches = re.findall(r'[A-Za-z0-9+/]{16,}={0,2}', text)
            for match in b64_matches:
                try:
                    decoded = b64.b64decode(match)
                    decoded.decode("utf-8")
                    base64_count += 1
                except Exception:
                    pass
        self.assertGreater(
            base64_count,
            0,
            "Expected at least one real base64-encoded payload in D4 samples",
        )

    def test_unicode_samples_contain_special_chars(self):
        """D5 samples should contain actual Unicode trickery."""
        records = _load_corpus()
        d5_records = [r for r in records if r.get("category") == "D5"]
        non_ascii_count = 0
        for rec in d5_records:
            text = rec["text"]
            if any(ord(c) > 127 for c in text):
                non_ascii_count += 1
        self.assertGreater(
            non_ascii_count,
            len(d5_records) // 2,
            f"Expected most D5 samples to contain non-ASCII characters, "
            f"got {non_ascii_count}/{len(d5_records)}",
        )

    def test_multilingual_samples_contain_non_english(self):
        """D6 samples should contain non-English text."""
        records = _load_corpus()
        d6_records = [r for r in records if r.get("category") == "D6"]
        non_ascii_count = 0
        for rec in d6_records:
            text = rec["text"]
            if any(ord(c) > 127 for c in text):
                non_ascii_count += 1
        self.assertGreater(
            non_ascii_count,
            len(d6_records) // 2,
            f"Expected most D6 samples to contain non-ASCII (non-English) characters, "
            f"got {non_ascii_count}/{len(d6_records)}",
        )

    def test_no_null_bytes(self):
        """No record should contain null bytes."""
        records = _load_corpus()
        for idx, rec in enumerate(records):
            self.assertNotIn(
                "\x00",
                rec.get("text", ""),
                f"Record {idx} contains a null byte",
            )


class TestCategoryDistribution(unittest.TestCase):
    """Test 11: Category distribution is reasonable."""

    def test_no_single_category_dominates(self):
        """No single category should have more than 25% of all samples."""
        records = _load_corpus()
        total = len(records)
        category_counts = Counter(rec.get("category") for rec in records)
        for cat, count in category_counts.items():
            pct = count / total
            self.assertLessEqual(
                pct,
                0.25,
                f"Category {cat} has {count}/{total} ({pct:.1%}) samples, "
                f"exceeding the 25% threshold",
            )


if __name__ == "__main__":
    unittest.main()
