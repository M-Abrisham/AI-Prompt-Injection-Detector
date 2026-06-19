"""Comprehensive tests for the safe-text holdout corpus.

Validates that data/holdout/safe_holdout.jsonl meets all requirements
specified for the holdout corpus:
  - Valid JSONL format
  - At least 500 samples
  - All labels are 0 (benign)
  - All 5 categories present
  - At least 90 samples per category
  - All records have required fields
  - No empty text fields
  - No duplicate texts
"""

import json
import os
import unittest
from collections import Counter
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_PATH = PROJECT_ROOT / "data" / "holdout" / "safe_holdout.jsonl"

# Skip every test in this module when the holdout data artifact is not
# materialized in the working tree. Regenerate with
# ``scripts/generate_safe_holdout.py`` (or ``scripts/gen_all_datasets.py``)
# or pull it from the data remote before running these tests.
pytestmark = pytest.mark.skipif(
    not HOLDOUT_PATH.exists(),
    reason=(
        f"holdout data not materialized at {HOLDOUT_PATH} "
        "- run scripts/generate_safe_holdout.py or dvc pull"
    ),
)

REQUIRED_FIELDS = {"text", "label", "source", "category"}
EXPECTED_CATEGORIES = {
    "instructional",
    "code",
    "customer_support",
    "creative_writing",
    "technical_docs",
}
MIN_TOTAL_SAMPLES = 500
MIN_SAMPLES_PER_CATEGORY = 90


# ---------------------------------------------------------------------------
# Helper: load all records once
# ---------------------------------------------------------------------------
def _load_records():
    """Load and parse all JSONL records from the holdout file."""
    records = []
    with open(HOLDOUT_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line"] = line_num
            records.append(record)
    return records


class TestHoldoutFileExists(unittest.TestCase):
    """Verify the holdout file exists and is readable."""

    def test_file_exists(self):
        self.assertTrue(
            HOLDOUT_PATH.exists(),
            f"Holdout file not found at {HOLDOUT_PATH}",
        )

    def test_file_is_not_empty(self):
        self.assertGreater(
            HOLDOUT_PATH.stat().st_size,
            0,
            "Holdout file exists but is empty",
        )


class TestValidJSONL(unittest.TestCase):
    """Verify every line is valid JSON."""

    def test_all_lines_valid_json(self):
        with open(HOLDOUT_PATH, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    self.fail(
                        f"Invalid JSON on line {line_num}: {exc}\n"
                        f"Content: {line[:200]}"
                    )


class TestSampleCount(unittest.TestCase):
    """Verify there are at least 500 samples."""

    def test_minimum_500_samples(self):
        records = _load_records()
        self.assertGreaterEqual(
            len(records),
            MIN_TOTAL_SAMPLES,
            f"Expected at least {MIN_TOTAL_SAMPLES} samples, found {len(records)}",
        )


class TestAllLabelsBenign(unittest.TestCase):
    """Verify all labels are 0 (benign)."""

    def test_all_labels_are_zero(self):
        records = _load_records()
        non_zero = [
            (r["_line"], r.get("label"))
            for r in records
            if r.get("label") != 0
        ]
        self.assertEqual(
            len(non_zero),
            0,
            f"Found {len(non_zero)} records with non-zero label: "
            f"{non_zero[:10]}",
        )

    def test_label_is_integer(self):
        records = _load_records()
        for r in records:
            self.assertIsInstance(
                r["label"],
                int,
                f"Line {r['_line']}: label should be int, got {type(r['label']).__name__}",
            )


class TestCategoriesPresent(unittest.TestCase):
    """Verify all required categories are present."""

    def test_all_expected_categories_present(self):
        records = _load_records()
        found_categories = {r.get("category") for r in records}
        missing = EXPECTED_CATEGORIES - found_categories
        self.assertEqual(
            len(missing),
            0,
            f"Missing categories: {missing}",
        )

    def test_no_unexpected_categories(self):
        records = _load_records()
        found_categories = {r.get("category") for r in records}
        unexpected = found_categories - EXPECTED_CATEGORIES
        self.assertEqual(
            len(unexpected),
            0,
            f"Unexpected categories found: {unexpected}",
        )


class TestCategoryCounts(unittest.TestCase):
    """Verify at least 90 samples per category."""

    def test_minimum_samples_per_category(self):
        records = _load_records()
        counts = Counter(r.get("category") for r in records)
        for category in EXPECTED_CATEGORIES:
            count = counts.get(category, 0)
            self.assertGreaterEqual(
                count,
                MIN_SAMPLES_PER_CATEGORY,
                f"Category '{category}' has {count} samples, "
                f"expected at least {MIN_SAMPLES_PER_CATEGORY}",
            )


class TestRequiredFields(unittest.TestCase):
    """Verify all records have the required fields."""

    def test_all_records_have_required_fields(self):
        records = _load_records()
        for r in records:
            record_keys = set(r.keys()) - {"_line"}
            missing = REQUIRED_FIELDS - record_keys
            self.assertEqual(
                len(missing),
                0,
                f"Line {r['_line']}: missing fields {missing}",
            )


class TestNoEmptyText(unittest.TestCase):
    """Verify no text fields are empty or whitespace-only."""

    def test_no_empty_text_fields(self):
        records = _load_records()
        empty = [r["_line"] for r in records if not r.get("text", "").strip()]
        self.assertEqual(
            len(empty),
            0,
            f"Found {len(empty)} records with empty text at lines: {empty[:20]}",
        )

    def test_text_is_non_trivial(self):
        """Every text should be at least 10 characters (not just 'hi')."""
        records = _load_records()
        short = [
            (r["_line"], r["text"][:50])
            for r in records
            if len(r.get("text", "").strip()) < 10
        ]
        self.assertEqual(
            len(short),
            0,
            f"Found {len(short)} records with very short text: {short[:10]}",
        )


class TestNoDuplicateTexts(unittest.TestCase):
    """Verify all text fields are unique."""

    def test_no_duplicate_texts(self):
        records = _load_records()
        texts = [r["text"] for r in records]
        seen = {}
        duplicates = []
        for i, text in enumerate(texts):
            if text in seen:
                duplicates.append(
                    (i + 1, seen[text], text[:80])
                )
            else:
                seen[text] = i + 1
        self.assertEqual(
            len(duplicates),
            0,
            f"Found {len(duplicates)} duplicate texts. "
            f"First few: {duplicates[:5]}",
        )


class TestSourceField(unittest.TestCase):
    """Verify source field values."""

    def test_all_sources_are_holdout(self):
        records = _load_records()
        non_holdout = [
            (r["_line"], r.get("source"))
            for r in records
            if r.get("source") != "holdout"
        ]
        self.assertEqual(
            len(non_holdout),
            0,
            f"Found {len(non_holdout)} records with non-'holdout' source: "
            f"{non_holdout[:10]}",
        )


class TestTextFieldTypes(unittest.TestCase):
    """Verify field types are correct."""

    def test_text_is_string(self):
        records = _load_records()
        for r in records:
            self.assertIsInstance(
                r["text"],
                str,
                f"Line {r['_line']}: text should be str",
            )

    def test_category_is_string(self):
        records = _load_records()
        for r in records:
            self.assertIsInstance(
                r["category"],
                str,
                f"Line {r['_line']}: category should be str",
            )

    def test_source_is_string(self):
        records = _load_records()
        for r in records:
            self.assertIsInstance(
                r["source"],
                str,
                f"Line {r['_line']}: source should be str",
            )


class TestHardNegativeCoverage(unittest.TestCase):
    """Verify that some samples contain typical false-positive trigger words
    in benign context (hard negatives)."""

    TRIGGER_WORDS = [
        "ignore",
        "system",
        "override",
        "instructions",
        "prompt",
        "role",
        "admin",
    ]

    def test_trigger_words_present_in_benign_samples(self):
        """At least some samples should contain false-positive trigger words."""
        records = _load_records()
        all_text = " ".join(r["text"].lower() for r in records)
        for word in self.TRIGGER_WORDS:
            self.assertIn(
                word,
                all_text,
                f"Trigger word '{word}' not found in any sample. "
                f"Hard negatives should include benign uses of this word.",
            )

    def test_multiple_samples_with_trigger_words(self):
        """There should be at least 5 samples containing trigger words."""
        records = _load_records()
        trigger_count = 0
        for r in records:
            text_lower = r["text"].lower()
            if any(word in text_lower for word in self.TRIGGER_WORDS):
                trigger_count += 1
        self.assertGreaterEqual(
            trigger_count,
            5,
            f"Only {trigger_count} samples contain trigger words. "
            f"Expected at least 5 hard-negative samples.",
        )


if __name__ == "__main__":
    unittest.main()
