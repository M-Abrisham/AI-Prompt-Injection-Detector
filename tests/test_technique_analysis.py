"""Tests for scripts/technique_analysis.py — per-technique performance analysis.

Validates:
1. Module imports work correctly
2. load_malicious_holdout parses JSONL correctly
3. load_evasion_dataset parses JSONL correctly
4. analyze_by_category computes correct recall from mocked scan results
5. analyze_by_evasion_type computes correct detection rate from mocked scan results
6. Output JSON format has required top-level keys
7. Per-category result contains all required fields
8. Per-evasion result contains all required fields
9. Recall is 1.0 when all samples are detected
10. Recall is 0.0 when no samples are detected
11. Mixed detection results produce correct counts
12. Empty dataset produces empty results dict
13. max_samples limits the number of samples loaded
14. print_category_table runs without error
15. print_evasion_table runs without error
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

# Ensure Na0S source is importable.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import technique_analysis as ta


# ---------------------------------------------------------------------------
# Helper: mock ScanResult (avoids importing na0s.scan_result at test time)
# ---------------------------------------------------------------------------

@dataclass
class MockScanResult:
    """Minimal ScanResult-like object for testing."""
    is_malicious: bool = False
    risk_score: float = 0.0
    technique_tags: list = field(default_factory=list)
    label: str = "safe"
    rule_hits: list = field(default_factory=list)
    elapsed_ms: float = 0.0


def _make_scan_fn(results_iter):
    """Create a scan function that returns pre-built results in order."""
    results = list(results_iter)
    call_idx = {"i": 0}

    def _scan(text, threshold=0.55):
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx < len(results):
            return results[idx]
        return MockScanResult()

    return _scan


# ---------------------------------------------------------------------------
# Temporary JSONL helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


class TestImports(unittest.TestCase):
    """Test that the analysis module imports cleanly."""

    def test_import_module(self):
        import technique_analysis  # noqa: F811
        self.assertTrue(hasattr(technique_analysis, "main"))

    def test_import_load_malicious_holdout(self):
        self.assertTrue(callable(ta.load_malicious_holdout))

    def test_import_load_evasion_dataset(self):
        self.assertTrue(callable(ta.load_evasion_dataset))

    def test_import_analyze_by_category(self):
        self.assertTrue(callable(ta.analyze_by_category))

    def test_import_analyze_by_evasion_type(self):
        self.assertTrue(callable(ta.analyze_by_evasion_type))


class TestLoadMaliciousHoldout(unittest.TestCase):
    """Test load_malicious_holdout JSONL parsing."""

    def test_loads_correct_fields(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _write_jsonl(f.name, [
                {"text": "attack 1", "label": 1, "category": "D1", "source": "holdout"},
                {"text": "attack 2", "label": 1, "category": "D3", "source": "holdout"},
            ])
            path = f.name
        try:
            samples = ta.load_malicious_holdout(path)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["text"], "attack 1")
            self.assertEqual(samples[0]["category"], "D1")
            self.assertEqual(samples[0]["label"], 1)
            self.assertEqual(samples[1]["category"], "D3")
        finally:
            os.unlink(path)

    def test_max_samples_limits(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            records = [{"text": f"sample {i}", "label": 1, "category": "D1"}
                       for i in range(50)]
            _write_jsonl(f.name, records)
            path = f.name
        try:
            samples = ta.load_malicious_holdout(path, max_samples=10)
            self.assertEqual(len(samples), 10)
        finally:
            os.unlink(path)

    def test_skips_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "a", "label": 1, "category": "D1"}\n')
            f.write('\n')
            f.write('{"text": "b", "label": 1, "category": "D2"}\n')
            path = f.name
        try:
            samples = ta.load_malicious_holdout(path)
            self.assertEqual(len(samples), 2)
        finally:
            os.unlink(path)


class TestLoadEvasionDataset(unittest.TestCase):
    """Test load_evasion_dataset JSONL parsing."""

    def test_loads_correct_fields(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _write_jsonl(f.name, [
                {"text": "enc text", "label": 1, "evasion_type": "base64"},
                {"text": "rot text", "label": 1, "evasion_type": "rot13"},
            ])
            path = f.name
        try:
            samples = ta.load_evasion_dataset(path)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["evasion_type"], "base64")
            self.assertEqual(samples[1]["evasion_type"], "rot13")
        finally:
            os.unlink(path)

    def test_max_samples_limits(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            records = [{"text": f"ev {i}", "label": 1, "evasion_type": "rot13"}
                       for i in range(30)]
            _write_jsonl(f.name, records)
            path = f.name
        try:
            samples = ta.load_evasion_dataset(path, max_samples=5)
            self.assertEqual(len(samples), 5)
        finally:
            os.unlink(path)


class TestAnalyzeByCategory(unittest.TestCase):
    """Test analyze_by_category with mocked scan results."""

    def test_all_detected_recall_1(self):
        """When all samples are detected, recall should be 1.0."""
        samples = [
            {"text": "atk1", "category": "D1", "label": 1},
            {"text": "atk2", "category": "D1", "label": 1},
            {"text": "atk3", "category": "D1", "label": 1},
        ]
        scan_fn = _make_scan_fn([
            MockScanResult(is_malicious=True, risk_score=0.9, technique_tags=["D1"]),
            MockScanResult(is_malicious=True, risk_score=0.8, technique_tags=["D1", "D1.1"]),
            MockScanResult(is_malicious=True, risk_score=0.7, technique_tags=["D1"]),
        ])
        results = ta.analyze_by_category(samples, scan_fn, threshold=0.55)
        self.assertEqual(results["D1"]["recall"], 1.0)
        self.assertEqual(results["D1"]["total"], 3)
        self.assertEqual(results["D1"]["detected"], 3)
        self.assertEqual(results["D1"]["missed"], 0)

    def test_none_detected_recall_0(self):
        """When no samples are detected, recall should be 0.0."""
        samples = [
            {"text": "atk1", "category": "D5", "label": 1},
            {"text": "atk2", "category": "D5", "label": 1},
        ]
        scan_fn = _make_scan_fn([
            MockScanResult(is_malicious=False, risk_score=0.2),
            MockScanResult(is_malicious=False, risk_score=0.3),
        ])
        results = ta.analyze_by_category(samples, scan_fn, threshold=0.55)
        self.assertEqual(results["D5"]["recall"], 0.0)
        self.assertEqual(results["D5"]["missed"], 2)

    def test_mixed_detection(self):
        """Mixed detection produces correct counts and recall."""
        samples = [
            {"text": "a", "category": "D3", "label": 1},
            {"text": "b", "category": "D3", "label": 1},
            {"text": "c", "category": "D3", "label": 1},
            {"text": "d", "category": "D3", "label": 1},
        ]
        scan_fn = _make_scan_fn([
            MockScanResult(is_malicious=True, risk_score=0.9, technique_tags=["D3"]),
            MockScanResult(is_malicious=False, risk_score=0.4),
            MockScanResult(is_malicious=True, risk_score=0.7, technique_tags=["D3", "D3.4"]),
            MockScanResult(is_malicious=False, risk_score=0.3),
        ])
        results = ta.analyze_by_category(samples, scan_fn, threshold=0.55)
        self.assertEqual(results["D3"]["total"], 4)
        self.assertEqual(results["D3"]["detected"], 2)
        self.assertEqual(results["D3"]["missed"], 2)
        self.assertEqual(results["D3"]["recall"], 0.5)

    def test_multiple_categories(self):
        """Results are grouped correctly by category."""
        samples = [
            {"text": "a", "category": "D1", "label": 1},
            {"text": "b", "category": "D8", "label": 1},
            {"text": "c", "category": "E1", "label": 1},
        ]
        scan_fn = _make_scan_fn([
            MockScanResult(is_malicious=True, risk_score=0.9, technique_tags=["D1"]),
            MockScanResult(is_malicious=True, risk_score=0.8, technique_tags=["D8"]),
            MockScanResult(is_malicious=False, risk_score=0.4),
        ])
        results = ta.analyze_by_category(samples, scan_fn, threshold=0.55)
        self.assertIn("D1", results)
        self.assertIn("D8", results)
        self.assertIn("E1", results)
        self.assertEqual(results["D1"]["recall"], 1.0)
        self.assertEqual(results["D8"]["recall"], 1.0)
        self.assertEqual(results["E1"]["recall"], 0.0)

    def test_technique_tags_counted(self):
        """technique_tags_seen aggregates tag counts correctly."""
        samples = [
            {"text": "a", "category": "D4", "label": 1},
            {"text": "b", "category": "D4", "label": 1},
        ]
        scan_fn = _make_scan_fn([
            MockScanResult(is_malicious=True, risk_score=0.8,
                           technique_tags=["D4", "D4.1"]),
            MockScanResult(is_malicious=True, risk_score=0.7,
                           technique_tags=["D4", "D4.5"]),
        ])
        results = ta.analyze_by_category(samples, scan_fn, threshold=0.55)
        tags = results["D4"]["technique_tags_seen"]
        self.assertEqual(tags["D4"], 2)
        self.assertEqual(tags.get("D4.1"), 1)
        self.assertEqual(tags.get("D4.5"), 1)

    def test_empty_dataset(self):
        """Empty dataset produces empty results."""
        results = ta.analyze_by_category([], _make_scan_fn([]), threshold=0.55)
        self.assertEqual(results, {})


class TestAnalyzeByEvasionType(unittest.TestCase):
    """Test analyze_by_evasion_type with mocked scan results."""

    def test_all_detected_rate_1(self):
        samples = [
            {"text": "enc", "evasion_type": "base64", "label": 1},
            {"text": "enc2", "evasion_type": "base64", "label": 1},
        ]
        scan_fn = _make_scan_fn([
            MockScanResult(is_malicious=True, risk_score=0.9),
            MockScanResult(is_malicious=True, risk_score=0.8),
        ])
        results = ta.analyze_by_evasion_type(samples, scan_fn, threshold=0.55)
        self.assertEqual(results["base64"]["detection_rate"], 1.0)
        self.assertEqual(results["base64"]["total"], 2)

    def test_none_detected_rate_0(self):
        samples = [
            {"text": "x", "evasion_type": "hex_encoding", "label": 1},
        ]
        scan_fn = _make_scan_fn([
            MockScanResult(is_malicious=False, risk_score=0.1),
        ])
        results = ta.analyze_by_evasion_type(samples, scan_fn, threshold=0.55)
        self.assertEqual(results["hex_encoding"]["detection_rate"], 0.0)

    def test_result_has_required_fields(self):
        """Each evasion result has all required fields."""
        samples = [{"text": "t", "evasion_type": "rot13", "label": 1}]
        scan_fn = _make_scan_fn([
            MockScanResult(is_malicious=True, risk_score=0.9, technique_tags=["D4.4"]),
        ])
        results = ta.analyze_by_evasion_type(samples, scan_fn, threshold=0.55)
        r = results["rot13"]
        required = {"total", "detected", "missed", "detection_rate",
                    "technique_tags_seen", "avg_latency_ms", "total_time_ms",
                    "missed_samples_preview"}
        self.assertTrue(required.issubset(set(r.keys())),
                        f"Missing keys: {required - set(r.keys())}")

    def test_empty_dataset(self):
        results = ta.analyze_by_evasion_type([], _make_scan_fn([]), threshold=0.55)
        self.assertEqual(results, {})


class TestCategoryResultFormat(unittest.TestCase):
    """Test per-category result dict has all required fields."""

    def test_required_fields_present(self):
        samples = [{"text": "atk", "category": "D2", "label": 1}]
        scan_fn = _make_scan_fn([
            MockScanResult(is_malicious=True, risk_score=0.8, technique_tags=["D2"]),
        ])
        results = ta.analyze_by_category(samples, scan_fn, threshold=0.55)
        r = results["D2"]
        required = {"total", "detected", "missed", "recall",
                    "technique_tags_seen", "avg_latency_ms", "total_time_ms",
                    "missed_samples_preview"}
        self.assertTrue(required.issubset(set(r.keys())),
                        f"Missing keys: {required - set(r.keys())}")


class TestPrintTables(unittest.TestCase):
    """Test that print functions do not raise errors."""

    def test_print_category_table(self):
        results = {
            "D1": {
                "total": 10, "detected": 8, "missed": 2, "recall": 0.8,
                "technique_tags_seen": {"D1": 8, "D1.1": 5},
                "avg_latency_ms": 30.0, "total_time_ms": 300.0,
                "missed_samples_preview": [],
            },
        }
        # Should not raise
        ta.print_category_table(results)

    def test_print_evasion_table(self):
        results = {
            "base64": {
                "total": 5, "detected": 4, "missed": 1, "detection_rate": 0.8,
                "technique_tags_seen": {"D4.1": 4},
                "avg_latency_ms": 45.0, "total_time_ms": 225.0,
                "missed_samples_preview": [],
            },
        }
        # Should not raise
        ta.print_evasion_table(results)


class TestOutputFormat(unittest.TestCase):
    """Test the JSON output file structure."""

    def test_output_json_has_required_keys(self):
        """Verify the output JSON from a real run has correct top-level keys."""
        output_path = _PROJECT_ROOT / "benchmarks" / "results" / "technique_analysis.json"
        if not output_path.exists():
            self.skipTest("Output file does not exist (run analysis first)")
        with open(output_path, "r") as fh:
            data = json.load(fh)
        required = {"timestamp", "version", "summary", "per_category", "per_evasion_type"}
        self.assertTrue(required.issubset(set(data.keys())),
                        f"Missing keys: {required - set(data.keys())}")
        # Verify summary has key fields
        summary = data["summary"]
        self.assertIn("overall_malicious_recall", summary)
        self.assertIn("overall_evasion_detection_rate", summary)
        self.assertIn("threshold", summary)


if __name__ == "__main__":
    unittest.main()
