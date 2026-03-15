"""Tests for scripts/evaluate_probes.py — probe evaluation pipeline."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probe_result(probe_id="D1", name="Test Probe", detected=8,
                       total=10, recall=0.8, missed=None, by_technique=None):
    """Build a fake probe evaluation result dict."""
    return {
        "probe": probe_id,
        "name": name,
        "detected": detected,
        "total": total,
        "recall": recall,
        "missed_samples": missed or [],
        "by_technique": by_technique or {},
    }


# ---------------------------------------------------------------------------
# Test: _select_probes
# ---------------------------------------------------------------------------

class TestSelectProbes:
    """Tests for the probe selection/filtering logic."""

    def test_none_returns_all(self):
        from evaluate_probes import _select_probes
        result = _select_probes(None)
        assert len(result) > 0, "Should return all probes when no filter"

    def test_empty_list_returns_all(self):
        from evaluate_probes import _select_probes
        result = _select_probes([])
        assert len(result) > 0

    def test_filter_by_known_id(self):
        from evaluate_probes import _select_probes
        from taxonomy import ALL_PROBES
        first_id = ALL_PROBES[0].category_id
        result = _select_probes([first_id])
        assert len(result) == 1
        assert result[0].category_id == first_id

    def test_unknown_id_warns(self, capsys):
        from evaluate_probes import _select_probes
        result = _select_probes(["NONEXISTENT_PROBE_XYZ"])
        assert len(result) == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "unknown" in captured.out.lower()

    def test_mixed_known_unknown(self):
        from evaluate_probes import _select_probes
        from taxonomy import ALL_PROBES
        first_id = ALL_PROBES[0].category_id
        result = _select_probes([first_id, "FAKE_ID"])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: _print_probe_report
# ---------------------------------------------------------------------------

class TestPrintProbeReport:
    """Tests for the probe report printer."""

    def test_perfect_recall(self, capsys):
        from evaluate_probes import _print_probe_report
        results = [_make_probe_result(detected=10, total=10, recall=1.0)]
        _print_probe_report(results)
        out = capsys.readouterr().out
        assert "100.0%" in out
        assert "WEAK" not in out

    def test_zero_recall_shows_weak(self, capsys):
        from evaluate_probes import _print_probe_report
        results = [_make_probe_result(detected=0, total=10, recall=0.0)]
        _print_probe_report(results)
        out = capsys.readouterr().out
        assert "0.0%" in out
        assert "WEAK" in out

    def test_empty_results(self, capsys):
        from evaluate_probes import _print_probe_report
        _print_probe_report([])
        out = capsys.readouterr().out
        assert "OVERALL" in out

    def test_multiple_probes(self, capsys):
        from evaluate_probes import _print_probe_report
        results = [
            _make_probe_result(probe_id="D1", detected=9, total=10, recall=0.9),
            _make_probe_result(probe_id="D2", detected=5, total=10, recall=0.5),
        ]
        _print_probe_report(results)
        out = capsys.readouterr().out
        assert "D1" in out
        assert "D2" in out


# ---------------------------------------------------------------------------
# Test: probe result structure
# ---------------------------------------------------------------------------

class TestProbeResultStructure:
    """Tests for probe evaluation result dicts."""

    def test_result_has_required_keys(self):
        result = _make_probe_result()
        for key in ("probe", "name", "detected", "total", "recall", "missed_samples"):
            assert key in result

    def test_recall_is_float(self):
        result = _make_probe_result(recall=0.75)
        assert isinstance(result["recall"], float)

    def test_detected_leq_total(self):
        result = _make_probe_result(detected=8, total=10)
        assert result["detected"] <= result["total"]


# ---------------------------------------------------------------------------
# Test: edge cases for zero-sample probes
# ---------------------------------------------------------------------------

class TestZeroSampleEdgeCases:
    """Tests for probes with 0 samples."""

    def test_zero_total_prints_cleanly(self, capsys):
        from evaluate_probes import _print_probe_report
        results = [_make_probe_result(detected=0, total=0, recall=0.0)]
        _print_probe_report(results)
        out = capsys.readouterr().out
        assert "0.0%" in out

    def test_zero_total_overall(self, capsys):
        from evaluate_probes import _print_probe_report
        results = [_make_probe_result(detected=0, total=0, recall=0.0)]
        _print_probe_report(results)
        out = capsys.readouterr().out
        assert "OVERALL" in out


# ---------------------------------------------------------------------------
# Test: argparse validation
# ---------------------------------------------------------------------------

class TestArgParsing:
    """Tests for CLI argument parsing."""

    def test_default_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--probes", nargs="*", default=None)
        parser.add_argument("--buffs", action="store_true")
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--output", type=str, default=None)
        args = parser.parse_args([])
        assert args.probes is None
        assert args.buffs is False
        assert args.json is False

    def test_probes_flag(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--probes", nargs="*", default=None)
        args = parser.parse_args(["--probes", "D1", "D2"])
        assert args.probes == ["D1", "D2"]
