"""Tests for the threshold sweep benchmark script.

Covers:
- Module import correctness
- JSONL loading
- Metrics computation (TP, TN, FP, FN, precision, recall, F1, FPR, accuracy)
- Edge cases (empty inputs, all-same predictions, zero denominators)
- Sweep logic with mocked scan()
- Output JSON structure
- CLI argument parsing
- Table formatting
- find_optimal helper
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so both na0s and scripts can be imported.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import threshold_sweep as ts


# ---------------------------------------------------------------------------
# 1. Test that the module imports correctly
# ---------------------------------------------------------------------------

class TestModuleImport:
    """Verify the script imports without errors and exposes expected symbols."""

    def test_import_module(self):
        """The module should import without errors."""
        assert hasattr(ts, "main")
        assert hasattr(ts, "scan")
        assert hasattr(ts, "load_jsonl")
        assert hasattr(ts, "compute_metrics")
        assert hasattr(ts, "run_sweep_on_dataset")
        assert hasattr(ts, "print_table")
        assert hasattr(ts, "find_optimal")
        assert hasattr(ts, "build_parser")

    def test_threshold_metrics_dataclass(self):
        """ThresholdMetrics should be a usable dataclass with to_dict()."""
        m = ts.ThresholdMetrics(threshold=0.55, dataset="test", tp=10, tn=20)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["threshold"] == 0.55
        assert d["tp"] == 10

    def test_default_thresholds(self):
        """DEFAULT_THRESHOLDS should contain the expected values."""
        assert ts.DEFAULT_THRESHOLDS == [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


# ---------------------------------------------------------------------------
# 2. Test JSONL loading
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    """Test the load_jsonl function."""

    def test_load_basic(self, tmp_path):
        """Load a simple JSONL file."""
        jsonl_path = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"text": "hello", "label": 0}),
            json.dumps({"text": "evil", "label": 1}),
            json.dumps({"text": "world", "label": 0}),
        ]
        jsonl_path.write_text("\n".join(lines) + "\n")

        samples = ts.load_jsonl(str(jsonl_path))
        assert len(samples) == 3
        assert samples[0]["text"] == "hello"
        assert samples[1]["label"] == 1

    def test_load_max_samples(self, tmp_path):
        """Respects --max-samples limit."""
        jsonl_path = tmp_path / "test.jsonl"
        lines = [json.dumps({"text": f"sample_{i}", "label": 0}) for i in range(20)]
        jsonl_path.write_text("\n".join(lines) + "\n")

        samples = ts.load_jsonl(str(jsonl_path), max_samples=5)
        assert len(samples) == 5

    def test_load_empty_lines_skipped(self, tmp_path):
        """Empty lines in JSONL should be skipped."""
        jsonl_path = tmp_path / "test.jsonl"
        content = '{"text": "a", "label": 0}\n\n{"text": "b", "label": 1}\n\n'
        jsonl_path.write_text(content)

        samples = ts.load_jsonl(str(jsonl_path))
        assert len(samples) == 2

    def test_load_max_samples_none_loads_all(self, tmp_path):
        """When max_samples is None, all samples should be loaded."""
        jsonl_path = tmp_path / "test.jsonl"
        lines = [json.dumps({"text": f"s{i}", "label": 0}) for i in range(10)]
        jsonl_path.write_text("\n".join(lines))

        samples = ts.load_jsonl(str(jsonl_path), max_samples=None)
        assert len(samples) == 10


# ---------------------------------------------------------------------------
# 3. Test metrics computation
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Test the compute_metrics function."""

    def test_perfect_classification(self):
        """All correct predictions should yield perfect metrics."""
        predictions = [True, True, False, False]
        labels = [1, 1, 0, 0]
        latencies = [10.0, 12.0, 8.0, 9.0]

        m = ts.compute_metrics(predictions, labels, 0.55, "test", latencies)

        assert m.tp == 2
        assert m.tn == 2
        assert m.fp == 0
        assert m.fn == 0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.fpr == 0.0
        assert m.accuracy == 1.0
        assert m.total_samples == 4
        assert m.avg_latency_ms == pytest.approx(9.75, abs=0.01)

    def test_all_false_positives(self):
        """All predictions malicious on safe data => FPR = 1.0, recall undefined."""
        predictions = [True, True, True]
        labels = [0, 0, 0]
        latencies = [5.0, 5.0, 5.0]

        m = ts.compute_metrics(predictions, labels, 0.5, "test", latencies)

        assert m.tp == 0
        assert m.fp == 3
        assert m.tn == 0
        assert m.fn == 0
        assert m.fpr == 1.0
        assert m.precision == 0.0  # 0/(0+3) = 0
        assert m.recall == 0.0    # no positives in ground truth
        assert m.f1 == 0.0

    def test_all_false_negatives(self):
        """All predictions safe on malicious data => recall = 0."""
        predictions = [False, False, False]
        labels = [1, 1, 1]
        latencies = [5.0, 5.0, 5.0]

        m = ts.compute_metrics(predictions, labels, 0.5, "test", latencies)

        assert m.tp == 0
        assert m.fn == 3
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_mixed_results(self):
        """Typical mixed results with some errors."""
        # 3 malicious, 2 safe ground truth
        predictions = [True, False, True, False, True]
        labels =      [1,    1,     0,    0,     1]
        latencies = [10.0] * 5

        m = ts.compute_metrics(predictions, labels, 0.55, "test", latencies)

        # TP=2 (idx 0,4), TN=1 (idx 3), FP=1 (idx 2), FN=1 (idx 1)
        assert m.tp == 2
        assert m.tn == 1
        assert m.fp == 1
        assert m.fn == 1
        assert m.precision == pytest.approx(2 / 3, abs=0.001)
        assert m.recall == pytest.approx(2 / 3, abs=0.001)
        assert m.fpr == pytest.approx(1 / 2, abs=0.001)
        assert m.accuracy == pytest.approx(3 / 5, abs=0.001)

    def test_empty_inputs(self):
        """Empty predictions list should return zeros."""
        m = ts.compute_metrics([], [], 0.55, "test", [])

        assert m.tp == 0
        assert m.tn == 0
        assert m.fp == 0
        assert m.fn == 0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.fpr == 0.0
        assert m.accuracy == 0.0
        assert m.avg_latency_ms == 0.0

    def test_threshold_stored(self):
        """The threshold value should be stored in the result."""
        m = ts.compute_metrics([True], [1], 0.42, "ds", [5.0])
        assert m.threshold == 0.42

    def test_dataset_name_stored(self):
        """The dataset name should be stored in the result."""
        m = ts.compute_metrics([True], [1], 0.5, "my_dataset", [5.0])
        assert m.dataset == "my_dataset"


# ---------------------------------------------------------------------------
# 4. Test sweep logic with mocked scan()
# ---------------------------------------------------------------------------

class TestRunSweep:
    """Test run_sweep_on_dataset with mocked scan."""

    @patch("threshold_sweep.scan")
    def test_sweep_returns_correct_count(self, mock_scan):
        """Should return one ThresholdMetrics per threshold."""
        mock_result = SimpleNamespace(is_malicious=True)
        mock_scan.return_value = mock_result

        samples = [{"text": "test", "label": 1}] * 5
        thresholds = [0.40, 0.50, 0.60]

        results = ts.run_sweep_on_dataset(samples, thresholds, "test_ds")

        assert len(results) == 3
        assert results[0].threshold == 0.40
        assert results[1].threshold == 0.50
        assert results[2].threshold == 0.60

    @patch("threshold_sweep.scan")
    def test_sweep_calls_scan_with_threshold(self, mock_scan):
        """scan() should be called with the correct threshold for each sample."""
        mock_result = SimpleNamespace(is_malicious=False)
        mock_scan.return_value = mock_result

        samples = [{"text": "hello", "label": 0}]
        thresholds = [0.45]

        ts.run_sweep_on_dataset(samples, thresholds, "test")

        # scan is called once for warmup + once per sample per threshold
        calls = mock_scan.call_args_list
        # Find the call with threshold=0.45
        threshold_calls = [c for c in calls if c[1].get("threshold") == 0.45]
        assert len(threshold_calls) == 1
        assert threshold_calls[0][0][0] == "hello"

    @patch("threshold_sweep.scan")
    def test_sweep_correct_metrics(self, mock_scan):
        """Sweep with known predictions should produce correct metrics."""
        # For threshold 0.5, scan returns malicious for all, giving:
        #   sample 0: label=1, pred=True => TP
        #   sample 1: label=0, pred=True => FP
        mock_result = SimpleNamespace(is_malicious=True)
        mock_scan.return_value = mock_result

        samples = [
            {"text": "attack", "label": 1},
            {"text": "safe", "label": 0},
        ]

        results = ts.run_sweep_on_dataset(samples, [0.50], "test")
        m = results[0]

        assert m.tp == 1
        assert m.fp == 1
        assert m.tn == 0
        assert m.fn == 0


# ---------------------------------------------------------------------------
# 5. Test output format
# ---------------------------------------------------------------------------

class TestOutputFormat:
    """Test JSON output structure."""

    @patch("threshold_sweep.scan")
    def test_main_produces_valid_json(self, mock_scan, tmp_path):
        """main() should produce a valid JSON file with expected keys."""
        mock_result = SimpleNamespace(is_malicious=False)
        mock_scan.return_value = mock_result

        # Create dummy datasets
        safe_path = tmp_path / "safe.jsonl"
        mal_path = tmp_path / "malicious.jsonl"
        out_dir = tmp_path / "results"

        safe_path.write_text(
            "\n".join(json.dumps({"text": f"safe_{i}", "label": 0}) for i in range(5))
        )
        mal_path.write_text(
            "\n".join(json.dumps({"text": f"mal_{i}", "label": 1}) for i in range(5))
        )

        argv = [
            "--max-samples", "3",
            "--safe-path", str(safe_path),
            "--malicious-path", str(mal_path),
            "--output-dir", str(out_dir),
            "--skip-adversarial",
            "--thresholds", "0.50", "0.55",
        ]

        result = ts.main(argv)

        assert "meta" in result
        assert "holdout" in result
        assert "adversarial" in result
        assert "best_holdout" in result

        # Verify JSON file was written
        json_path = out_dir / "threshold_sweep.json"
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["meta"]["thresholds"] == [0.50, 0.55]
        assert len(loaded["holdout"]) == 2  # two thresholds

    @patch("threshold_sweep.scan")
    def test_output_holdout_keys(self, mock_scan, tmp_path):
        """Each holdout entry should have all required metric keys."""
        mock_result = SimpleNamespace(is_malicious=True)
        mock_scan.return_value = mock_result

        safe_path = tmp_path / "safe.jsonl"
        mal_path = tmp_path / "mal.jsonl"
        out_dir = tmp_path / "results"

        safe_path.write_text(json.dumps({"text": "a", "label": 0}))
        mal_path.write_text(json.dumps({"text": "b", "label": 1}))

        result = ts.main([
            "--max-samples", "1",
            "--safe-path", str(safe_path),
            "--malicious-path", str(mal_path),
            "--output-dir", str(out_dir),
            "--skip-adversarial",
            "--thresholds", "0.55",
        ])

        entry = result["holdout"][0]
        required_keys = {
            "threshold", "dataset", "tp", "tn", "fp", "fn",
            "precision", "recall", "f1", "fpr", "accuracy",
            "total_samples", "avg_latency_ms",
        }
        assert required_keys <= set(entry.keys())


# ---------------------------------------------------------------------------
# 6. Test find_optimal
# ---------------------------------------------------------------------------

class TestFindOptimal:
    """Test the find_optimal helper."""

    def test_selects_highest_f1(self):
        """Should select the metrics with the highest F1 score."""
        results = [
            ts.ThresholdMetrics(threshold=0.40, f1=0.80),
            ts.ThresholdMetrics(threshold=0.50, f1=0.95),
            ts.ThresholdMetrics(threshold=0.60, f1=0.85),
        ]
        best = ts.find_optimal(results)
        assert best.threshold == 0.50
        assert best.f1 == 0.95

    def test_single_result(self):
        """Should work with a single result."""
        results = [ts.ThresholdMetrics(threshold=0.55, f1=0.90)]
        best = ts.find_optimal(results)
        assert best.threshold == 0.55


# ---------------------------------------------------------------------------
# 7. Test CLI argument parsing
# ---------------------------------------------------------------------------

class TestBuildParser:
    """Test CLI argument parsing."""

    def test_default_args(self):
        """Default arguments should match expected values."""
        parser = ts.build_parser()
        args = parser.parse_args([])
        assert args.thresholds == ts.DEFAULT_THRESHOLDS
        assert args.max_samples is None
        assert args.skip_adversarial is False

    def test_max_samples_arg(self):
        """--max-samples should be parsed correctly."""
        parser = ts.build_parser()
        args = parser.parse_args(["--max-samples", "42"])
        assert args.max_samples == 42

    def test_custom_thresholds(self):
        """--thresholds should accept multiple float values."""
        parser = ts.build_parser()
        args = parser.parse_args(["--thresholds", "0.30", "0.70"])
        assert args.thresholds == [0.30, 0.70]

    def test_skip_adversarial_flag(self):
        """--skip-adversarial flag should be True when set."""
        parser = ts.build_parser()
        args = parser.parse_args(["--skip-adversarial"])
        assert args.skip_adversarial is True


# ---------------------------------------------------------------------------
# 8. Test table formatting (no crash)
# ---------------------------------------------------------------------------

class TestPrintTable:
    """Test that print_table runs without errors."""

    def test_print_table_no_crash(self, capsys):
        """print_table should produce output without raising."""
        results = [
            ts.ThresholdMetrics(
                threshold=0.50, dataset="test", tp=10, tn=20, fp=2, fn=3,
                precision=0.833, recall=0.769, f1=0.800, fpr=0.091,
                accuracy=0.857, total_samples=35, avg_latency_ms=15.5,
            ),
        ]
        ts.print_table(results, "Test Results")
        captured = capsys.readouterr()
        assert "Test Results" in captured.out
        assert "0.50" in captured.out
        assert "0.8000" in captured.out


# ---------------------------------------------------------------------------
# 9. Test ThresholdMetrics.to_dict roundtrip
# ---------------------------------------------------------------------------

class TestThresholdMetricsRoundtrip:
    """Test serialization roundtrip for ThresholdMetrics."""

    def test_to_dict_roundtrip(self):
        """to_dict() output should be JSON-serializable and preserve values."""
        m = ts.ThresholdMetrics(
            threshold=0.55, dataset="holdout", tp=100, tn=450, fp=5, fn=10,
            precision=0.9524, recall=0.9091, f1=0.9302, fpr=0.0110,
            accuracy=0.9735, total_samples=565, avg_latency_ms=32.5,
        )
        d = m.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)

        assert loaded["threshold"] == 0.55
        assert loaded["tp"] == 100
        assert loaded["precision"] == 0.9524
        assert loaded["avg_latency_ms"] == 32.5
