"""Tests for scripts/taxonomy/_tags.py — all 7 issues."""

import logging
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.taxonomy._tags import (
    aggregate_by_taxonomy,
    clear_tag_cache,
    load_tags,
    summarize_groups,
)
import scripts.taxonomy._tags as _tags_mod


def _write_tsv(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _set_tags_path(path):
    """Point _tags module at a custom TSV file and clear cache."""
    _tags_mod._TAGS_PATH = path
    clear_tag_cache()


class _TempTSVMixin:
    """Mixin that restores _TAGS_PATH after each test."""

    def setUp(self):
        self._orig_path = _tags_mod._TAGS_PATH
        clear_tag_cache()

    def tearDown(self):
        _tags_mod._TAGS_PATH = self._orig_path
        clear_tag_cache()


# ---------------------------------------------------------------------------
# Issue 1 — Missing file error + brittle path
# ---------------------------------------------------------------------------

class TestMissingFile(_TempTSVMixin, unittest.TestCase):

    def test_missing_file_raises_with_context(self):
        from pathlib import Path
        _set_tags_path(Path("/tmp/nonexistent_tags_file.tsv"))
        with self.assertRaises(FileNotFoundError) as ctx:
            load_tags()
        self.assertIn("MISP tags file not found", str(ctx.exception))
        self.assertIn("TAGS_MISP_PATH", str(ctx.exception))

    def test_env_var_override(self):
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("test-tag:01\tTest description\n")
            tmp = f.name
        try:
            _set_tags_path(Path(tmp))
            tags = load_tags()
            self.assertIn("test-tag:01", tags)
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Issue 2 — Malformed lines warned (not silently skipped)
# ---------------------------------------------------------------------------

class TestMalformedLines(_TempTSVMixin, unittest.TestCase):

    def test_malformed_line_warns(self):
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("good-tag:01\tGood\n")
            f.write("bad line without tab\n")
            tmp = f.name
        try:
            _set_tags_path(Path(tmp))
            with self.assertLogs("scripts.taxonomy._tags", level="WARNING") as cm:
                tags = load_tags()
            self.assertIn("good-tag:01", tags)
            self.assertEqual(len(tags), 1)
            self.assertTrue(any("line 2" in msg for msg in cm.output))
        finally:
            os.unlink(tmp)

    def test_comment_and_blank_lines_skipped_silently(self):
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("# comment\n\ngood:01\tDesc\n")
            tmp = f.name
        try:
            _set_tags_path(Path(tmp))
            tags = load_tags()
            self.assertEqual(len(tags), 1)
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Issue 3 — Duplicate tags warn + keep first
# ---------------------------------------------------------------------------

class TestDuplicateTags(_TempTSVMixin, unittest.TestCase):

    def test_duplicate_warns_keeps_first(self):
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("dup:01\tFirst\n")
            f.write("dup:01\tSecond\n")
            tmp = f.name
        try:
            _set_tags_path(Path(tmp))
            with self.assertLogs("scripts.taxonomy._tags", level="WARNING") as cm:
                tags = load_tags()
            self.assertEqual(tags["dup:01"], "First")
            self.assertTrue(any("duplicate" in msg for msg in cm.output))
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Issue 4 — Prefix matching enforces delimiter
# ---------------------------------------------------------------------------

class TestNamespaceDelimiter(_TempTSVMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        # Create a temp TSV so aggregate_by_taxonomy can load tags
        from pathlib import Path
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False,
        )
        self._tmp.write("owasp-llm:2025:llm01\tLLM01 Prompt Injection\n")
        self._tmp.write("owasp-llm:2025:llm02\tLLM02 Insecure Output\n")
        self._tmp.write("risk-cards:lmrc:dos\tDenial of Service\n")
        self._tmp.close()
        _set_tags_path(Path(self._tmp.name))

    def tearDown(self):
        os.unlink(self._tmp.name)
        super().tearDown()

    def test_short_prefix_does_not_cross_contaminate(self):
        """namespace='owasp' must NOT match 'owasp-llm:...'."""
        results = [{
            "probe": "P1", "total": 10, "detected": 5, "attributed": 3,
            "tags": ["owasp-llm:2025:llm01"],
        }]
        groups = aggregate_by_taxonomy(results, "owasp")
        self.assertEqual(groups, {})

    def test_full_prefix_matches(self):
        results = [{
            "probe": "P1", "total": 10, "detected": 5, "attributed": 3,
            "tags": ["owasp-llm:2025:llm01"],
        }]
        groups = aggregate_by_taxonomy(results, "owasp-llm")
        self.assertIn("owasp-llm:2025:llm01", groups)

    def test_trailing_colon_in_namespace_works(self):
        results = [{
            "probe": "P1", "total": 10, "detected": 5, "attributed": 3,
            "tags": ["risk-cards:lmrc:dos"],
        }]
        groups = aggregate_by_taxonomy(results, "risk-cards:")
        self.assertIn("risk-cards:lmrc:dos", groups)


# ---------------------------------------------------------------------------
# Issue 5 — Unknown tag warns
# ---------------------------------------------------------------------------

class TestUnknownTagWarning(_TempTSVMixin, unittest.TestCase):

    def test_unknown_tag_warns_with_probe_name(self):
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("ns:known\tKnown tag\n")
            tmp = f.name
        try:
            _set_tags_path(Path(tmp))
            results = [{
                "probe": "TestProbe", "total": 5, "detected": 2, "attributed": 1,
                "tags": ["ns:unknown"],
            }]
            with self.assertLogs("scripts.taxonomy._tags", level="WARNING") as cm:
                groups = aggregate_by_taxonomy(results, "ns")
            self.assertTrue(any("ns:unknown" in msg for msg in cm.output))
            self.assertTrue(any("TestProbe" in msg for msg in cm.output))
            # Falls back to raw tag as description
            self.assertEqual(groups["ns:unknown"]["description"], "ns:unknown")
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Issue 6 — Summary stats + attribution passthrough
# ---------------------------------------------------------------------------

class TestSummaryAndAttribution(_TempTSVMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        # Create a temp TSV so aggregate_by_taxonomy can load tags
        from pathlib import Path
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False,
        )
        self._tmp.write("ns:tag1\tTag One\n")
        self._tmp.write("ns:tag2\tTag Two\n")
        self._tmp.close()
        _set_tags_path(Path(self._tmp.name))

    def tearDown(self):
        os.unlink(self._tmp.name)
        super().tearDown()

    def _make_groups(self):
        return {
            "ns:a": {
                "total": 10, "detected": 8, "attributed": 6,
                "missed": 2, "recall": 0.8, "attribution_rate": 0.75,
                "description": "A", "probes": ["P1"],
            },
            "ns:b": {
                "total": 20, "detected": 10, "attributed": 4,
                "missed": 10, "recall": 0.5, "attribution_rate": 0.4,
                "description": "B", "probes": ["P2"],
            },
        }

    def test_summarize_totals(self):
        s = summarize_groups(self._make_groups(), "ns")
        self.assertEqual(s["total"], 30)
        self.assertEqual(s["detected"], 18)
        self.assertEqual(s["attributed"], 10)
        self.assertEqual(s["missed"], 12)
        self.assertAlmostEqual(s["recall"], 18 / 30)
        self.assertAlmostEqual(s["attribution_rate"], 10 / 18)
        self.assertEqual(s["namespace"], "ns")
        self.assertEqual(s["tag_count"], 2)

    def test_summarize_empty_groups(self):
        s = summarize_groups({})
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["recall"], 0.0)
        self.assertEqual(s["attribution_rate"], 0.0)

    def test_aggregate_tracks_attribution(self):
        results = [{
            "probe": "P1", "total": 10, "detected": 7, "attributed": 5,
            "tags": ["ns:tag1"],
        }]
        groups = aggregate_by_taxonomy(results, "ns")
        g = groups["ns:tag1"]
        self.assertEqual(g["attributed"], 5)
        self.assertAlmostEqual(g["attribution_rate"], 5 / 7)

    def test_aggregate_missed_computed(self):
        results = [{
            "probe": "P1", "total": 10, "detected": 3, "attributed": 1,
            "tags": ["ns:tag1"],
        }]
        groups = aggregate_by_taxonomy(results, "ns")
        self.assertEqual(groups["ns:tag1"]["missed"], 7)

    def test_aggregate_zero_detected(self):
        results = [{
            "probe": "P1", "total": 10, "detected": 0, "attributed": 0,
            "tags": ["ns:tag1"],
        }]
        groups = aggregate_by_taxonomy(results, "ns")
        g = groups["ns:tag1"]
        self.assertEqual(g["recall"], 0.0)
        self.assertEqual(g["attribution_rate"], 0.0)

    def test_legacy_result_without_attributed(self):
        """Results from older evaluate() without 'attributed' key."""
        results = [{
            "probe": "P1", "total": 10, "detected": 5,
            "tags": ["ns:tag1"],
        }]
        groups = aggregate_by_taxonomy(results, "ns")
        self.assertEqual(groups["ns:tag1"]["attributed"], 0)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety(_TempTSVMixin, unittest.TestCase):

    def test_concurrent_loads(self):
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("t:01\tDesc\n")
            tmp = f.name
        try:
            _set_tags_path(Path(tmp))
            results = [None] * 10
            def load(i):
                results[i] = load_tags()
            threads = [threading.Thread(target=load, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # All threads get the same content (lru_cache may return
            # separate objects during concurrent first-call races, but the
            # data must be identical)
            for r in results:
                self.assertEqual(r, results[0])
        finally:
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
