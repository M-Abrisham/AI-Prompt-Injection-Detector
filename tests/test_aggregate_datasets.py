"""Tests for scripts/aggregate_datasets.py.

Validates dataset aggregation logic: text normalisation, hashing,
record normalisation, deduplication, stratified sampling, dataset
registry integrity, CLI parsing, and output format -- all without
making real network requests.
"""

import json
import os
import random
import re
import sys
import unittest

# Ensure the scripts directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from aggregate_datasets import (
    normalize_text,
    text_hash,
    normalize_records,
    deduplicate,
    stratified_sample,
    build_parser,
    DATASETS,
)


# ===================================================================
# 1. Import tests
# ===================================================================

class TestImport(unittest.TestCase):
    """The module should import without side effects."""

    def test_import(self):
        """Module imports without error and exposes key symbols."""
        import aggregate_datasets as mod
        for attr in ("normalize_text", "text_hash", "normalize_records",
                      "deduplicate", "stratified_sample", "build_parser",
                      "DATASETS", "main"):
            self.assertTrue(hasattr(mod, attr), f"Missing attribute: {attr}")

    def test_datasets_registry_not_empty(self):
        """DATASETS list has at least one entry."""
        self.assertIsInstance(DATASETS, list)
        self.assertGreater(len(DATASETS), 0)


# ===================================================================
# 2. normalize_text tests
# ===================================================================

class TestNormalizeText(unittest.TestCase):

    def test_normalize_strips_whitespace(self):
        self.assertEqual(normalize_text(" hello world "), "hello world")

    def test_normalize_lowercases(self):
        self.assertEqual(normalize_text("HELLO"), "hello")

    def test_normalize_collapses_spaces(self):
        self.assertEqual(normalize_text("hello   world"), "hello world")

    def test_normalize_handles_newlines(self):
        self.assertEqual(normalize_text("hello\n\nworld"), "hello world")

    def test_normalize_empty(self):
        self.assertEqual(normalize_text(""), "")


# ===================================================================
# 3. text_hash tests
# ===================================================================

class TestTextHash(unittest.TestCase):

    def test_hash_deterministic(self):
        """Same input always produces the same hash."""
        h1 = text_hash("hello world")
        h2 = text_hash("hello world")
        self.assertEqual(h1, h2)

    def test_hash_different_for_different_inputs(self):
        self.assertNotEqual(text_hash("a"), text_hash("b"))

    def test_hash_is_64_char_hex(self):
        h = text_hash("test string")
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))


# ===================================================================
# 4. normalize_records tests
# ===================================================================

class TestNormalizeRecords(unittest.TestCase):

    def _make_config(self, **overrides):
        """Create a minimal dataset config dict."""
        cfg = {
            "id": "test/dataset",
            "name": "test_dataset",
            "text_col": "text",
            "splits": ["train"],
            "source": "test_src",
        }
        cfg.update(overrides)
        return cfg

    def test_normalize_basic_with_label_col(self):
        rows = [
            {"text": "hello", "label": 0},
            {"text": "inject me", "label": 1},
        ]
        config = self._make_config(label_col="label")
        records = normalize_records(rows, config)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["label"], 0)
        self.assertEqual(records[1]["label"], 1)

    def test_normalize_with_fixed_label(self):
        rows = [
            {"text": "safe prompt 1"},
            {"text": "safe prompt 2"},
        ]
        config = self._make_config(fixed_label=1)
        records = normalize_records(rows, config)
        self.assertEqual(len(records), 2)
        for rec in records:
            self.assertEqual(rec["label"], 1)

    def test_normalize_with_label_map(self):
        rows = [
            {"text": "hello", "type": "benign"},
            {"text": "evil", "type": "malicious"},
        ]
        config = self._make_config(
            label_col="type",
            label_map={"benign": 0, "malicious": 1},
        )
        records = normalize_records(rows, config)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["label"], 0)
        self.assertEqual(records[1]["label"], 1)

    def test_normalize_skips_empty_text(self):
        rows = [
            {"text": "", "label": 0},
            {"text": "   ", "label": 0},
            {"text": None, "label": 0},
            {"text": "valid", "label": 0},
        ]
        config = self._make_config(label_col="label")
        records = normalize_records(rows, config)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "valid")

    def test_normalize_strips_text(self):
        rows = [{"text": "  hello world  ", "label": 0}]
        config = self._make_config(label_col="label")
        records = normalize_records(rows, config)
        self.assertEqual(records[0]["text"], "hello world")

    def test_normalize_adds_source(self):
        rows = [{"text": "hello", "label": 0}]
        config = self._make_config(label_col="label", source="my_source")
        records = normalize_records(rows, config)
        self.assertEqual(records[0]["source"], "my_source")

    def test_normalize_category_benign(self):
        rows = [{"text": "hello", "label": 0}]
        config = self._make_config(label_col="label")
        records = normalize_records(rows, config)
        self.assertEqual(records[0]["category"], "benign")

    def test_normalize_category_injection(self):
        rows = [{"text": "inject", "label": 1}]
        config = self._make_config(label_col="label")
        records = normalize_records(rows, config)
        self.assertEqual(records[0]["category"], "injection")


# ===================================================================
# 5. deduplicate tests
# ===================================================================

class TestDeduplicate(unittest.TestCase):

    def _rec(self, text, label=0, source="src"):
        return {"text": text, "label": label, "source": source, "category": "benign"}

    def test_dedup_no_duplicates(self):
        records = [self._rec("aaa"), self._rec("bbb"), self._rec("ccc")]
        deduped, stats = deduplicate(records)
        self.assertEqual(len(deduped), 3)

    def test_dedup_exact_duplicates(self):
        records = [self._rec("aaa"), self._rec("aaa")]
        deduped, stats = deduplicate(records)
        self.assertEqual(len(deduped), 1)

    def test_dedup_case_insensitive(self):
        """'HELLO' and 'hello' should deduplicate to one record."""
        records = [self._rec("HELLO"), self._rec("hello")]
        deduped, stats = deduplicate(records)
        self.assertEqual(len(deduped), 1)

    def test_dedup_preserves_first(self):
        """First occurrence is the one kept."""
        records = [
            {"text": "HELLO", "label": 0, "source": "first", "category": "benign"},
            {"text": "hello", "label": 1, "source": "second", "category": "injection"},
        ]
        deduped, stats = deduplicate(records)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["source"], "first")

    def test_dedup_returns_stats(self):
        """deduplicate returns (records, dup_counts_dict)."""
        records = [
            self._rec("aaa", source="s1"),
            self._rec("aaa", source="s2"),
            self._rec("bbb", source="s1"),
        ]
        deduped, dup_counts = deduplicate(records)
        self.assertIsInstance(deduped, list)
        self.assertIsInstance(dup_counts, dict)
        # One duplicate was from source "s2"
        self.assertEqual(dup_counts.get("s2", 0), 1)
        # Before = 3, after = 2
        self.assertEqual(len(deduped), 2)


# ===================================================================
# 6. stratified_sample tests
# ===================================================================

class TestStratifiedSample(unittest.TestCase):

    def _make_records(self, n0, n1):
        """Create records with *n0* benign and *n1* injection records."""
        records = []
        for i in range(n0):
            records.append({
                "text": f"benign_{i}", "label": 0,
                "source": "test", "category": "benign",
            })
        for i in range(n1):
            records.append({
                "text": f"inject_{i}", "label": 1,
                "source": "test", "category": "injection",
            })
        return records

    def test_sample_reduces_size(self):
        records = self._make_records(70, 30)
        rng = random.Random(42)
        sampled = stratified_sample(records, 10, rng)
        self.assertEqual(len(sampled), 10)

    def test_sample_preserves_ratio(self):
        """70/30 ratio maintained approximately after sampling."""
        records = self._make_records(700, 300)
        rng = random.Random(42)
        sampled = stratified_sample(records, 100, rng)

        n0 = sum(1 for r in sampled if r["label"] == 0)
        n1 = sum(1 for r in sampled if r["label"] == 1)
        # Allow some tolerance -- ratio should be roughly 70/30
        self.assertGreater(n0, n1, "Benign count should exceed injection count")
        self.assertGreaterEqual(n0, 55, f"Expected ~70 benign, got {n0}")
        self.assertGreaterEqual(n1, 20, f"Expected ~30 injection, got {n1}")

    def test_sample_deterministic(self):
        records = self._make_records(70, 30)
        sampled1 = stratified_sample(records, 10, random.Random(42))
        sampled2 = stratified_sample(records, 10, random.Random(42))
        self.assertEqual(
            [r["text"] for r in sampled1],
            [r["text"] for r in sampled2],
        )

    def test_sample_noop_when_under_limit(self):
        records = self._make_records(7, 3)
        rng = random.Random(42)
        sampled = stratified_sample(records, 100, rng)
        self.assertEqual(len(sampled), 10)


# ===================================================================
# 7. Dataset registry validation
# ===================================================================

class TestDatasetRegistry(unittest.TestCase):

    def test_all_datasets_have_required_fields(self):
        required = {"id", "name", "text_col", "splits", "source"}
        for ds in DATASETS:
            for field in required:
                self.assertIn(
                    field, ds,
                    f"Dataset '{ds.get('name', '?')}' missing field: {field}",
                )

    def test_all_datasets_have_label_info(self):
        """Each dataset must have either label_col or fixed_label."""
        for ds in DATASETS:
            has_label_col = ds.get("label_col") is not None
            has_fixed_label = "fixed_label" in ds
            self.assertTrue(
                has_label_col or has_fixed_label,
                f"Dataset '{ds['name']}' has neither label_col nor fixed_label",
            )

    def test_dataset_ids_are_valid_hf_format(self):
        """Each id should match 'owner/dataset-name' pattern."""
        pattern = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$")
        for ds in DATASETS:
            self.assertRegex(
                ds["id"], pattern,
                f"Dataset id '{ds['id']}' does not match HuggingFace format",
            )

    def test_dataset_names_unique(self):
        names = [ds["name"] for ds in DATASETS]
        self.assertEqual(len(names), len(set(names)), "Duplicate dataset names found")

    def test_dataset_sources_unique(self):
        sources = [ds["source"] for ds in DATASETS]
        self.assertEqual(
            len(sources), len(set(sources)),
            "Duplicate dataset sources found",
        )


# ===================================================================
# 8. CLI parser tests
# ===================================================================

class TestBuildParser(unittest.TestCase):

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.output_dir, "data/aggregated")
        self.assertTrue(args.skip_gated)

    def test_parser_force_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--force"])
        self.assertTrue(args.force)

    def test_parser_datasets_filter(self):
        parser = build_parser()
        args = parser.parse_args(["--datasets", "a,b"])
        self.assertEqual(args.datasets, "a,b")


# ===================================================================
# 9. Output format tests
# ===================================================================

class TestOutputFormat(unittest.TestCase):
    """Verify stats.json schema and merged JSONL validity using tmp_path."""

    def test_stats_json_schema(self):
        """stats.json written by merge_all has the expected keys."""
        import tempfile
        from unittest import mock

        import aggregate_datasets as mod

        # Mock fetch_dataset to return predictable records
        fake_records = [
            {"text": "hello", "label": 0, "source": "s1", "category": "benign"},
            {"text": "evil", "label": 1, "source": "s1", "category": "injection"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(mod, "fetch_dataset", return_value=fake_records):
                mod.merge_all(
                    output_dir=tmpdir,
                    datasets=[DATASETS[0]],
                    force=True,
                )

            stats_path = os.path.join(tmpdir, "stats.json")
            self.assertTrue(os.path.exists(stats_path), "stats.json not written")

            with open(stats_path, "r", encoding="utf-8") as fh:
                stats = json.load(fh)

            expected_keys = {
                "total_before_dedup",
                "total_after_dedup",
                "duplicates_removed",
                "errors",
            }
            for key in expected_keys:
                self.assertIn(key, stats, f"stats.json missing key: {key}")

    def test_merged_jsonl_valid(self):
        """Each line in the merged JSONL is valid JSON with required fields."""
        import tempfile
        from unittest import mock

        import aggregate_datasets as mod

        fake_records = [
            {"text": "hello", "label": 0, "source": "s1", "category": "benign"},
            {"text": "world", "label": 1, "source": "s1", "category": "injection"},
            {"text": "test", "label": 0, "source": "s1", "category": "benign"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(mod, "fetch_dataset", return_value=fake_records):
                mod.merge_all(
                    output_dir=tmpdir,
                    datasets=[DATASETS[0]],
                    force=True,
                )

            merged_path = os.path.join(tmpdir, "merged_train.jsonl")
            self.assertTrue(os.path.exists(merged_path), "merged_train.jsonl not written")

            with open(merged_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            self.assertGreater(len(lines), 0, "merged file is empty")
            for line in lines:
                obj = json.loads(line.strip())
                self.assertIsInstance(obj["text"], str)
                self.assertIn(obj["label"], (0, 1))
                self.assertIn("source", obj)
                self.assertIn("category", obj)

    def test_merged_jsonl_no_duplicates(self):
        """The merged JSONL file should contain no duplicate texts."""
        import tempfile
        from unittest import mock

        import aggregate_datasets as mod

        fake_records = [
            {"text": "hello", "label": 0, "source": "s1", "category": "benign"},
            {"text": "hello", "label": 0, "source": "s1", "category": "benign"},
            {"text": "world", "label": 1, "source": "s1", "category": "injection"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(mod, "fetch_dataset", return_value=fake_records):
                mod.merge_all(
                    output_dir=tmpdir,
                    datasets=[DATASETS[0]],
                    force=True,
                )

            merged_path = os.path.join(tmpdir, "merged_train.jsonl")
            with open(merged_path, "r", encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]

            texts = [r["text"] for r in records]
            self.assertEqual(len(texts), len(set(texts)),
                             "Merged file contains duplicate texts")


if __name__ == "__main__":
    unittest.main()
