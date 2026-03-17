"""Tests for scripts/regression_dashboard.py."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from regression_dashboard import (
    append_history,
    build_snapshot,
    cmd_compare,
    compute_deltas,
    load_history,
)


def _make_probe_results(probes):
    """Build a minimal probe_results list from a dict of {probe_id: recall}."""
    results = []
    for probe_id, recall in probes.items():
        results.append({
            "probe": probe_id,
            "name": "Test Probe {}".format(probe_id),
            "recall": recall,
            "attribution_rate": 0.8,
            "total": 100,
            "detected": int(recall * 100),
        })
    return results


class TestHistoryAppendWritesValidJsonl:
    """test_history_append_writes_valid_jsonl"""

    def test_appended_lines_are_valid_json_with_correct_schema(self, tmp_path):
        # Arrange — create mock probe results and write a history file
        probe_results = _make_probe_results({"P001": 0.95, "P002": 0.88})
        snapshot = build_snapshot(probe_results, is_baseline=False)
        history_path = str(tmp_path / "regression_history.jsonl")

        # Act — append twice to verify multi-line behaviour
        append_history(snapshot, path=history_path)
        append_history(snapshot, path=history_path)

        # Assert — each line is valid JSON with required top-level keys
        entries = load_history(history_path)
        assert len(entries) == 2

        required_keys = {
            "timestamp", "git_sha", "is_baseline",
            "per_probe", "overall", "latency_ms",
        }
        for entry in entries:
            assert required_keys.issubset(entry.keys())
            assert isinstance(entry["per_probe"], dict)
            assert "mean_recall" in entry["overall"]
            assert "mean_attribution_rate" in entry["overall"]
            # Verify per-probe schema
            for probe_id, metrics in entry["per_probe"].items():
                assert "recall" in metrics
                assert "attribution_rate" in metrics
                assert "sample_count" in metrics


class TestDeltaDetectsRegression:
    """test_delta_detects_regression"""

    def test_five_percent_recall_drop_flagged_as_regression(self):
        # Arrange — two snapshots where P001 recall drops by 5%
        prev_results = _make_probe_results({"P001": 0.95, "P002": 0.90})
        curr_results = _make_probe_results({"P001": 0.90, "P002": 0.90})

        prev_snapshot = build_snapshot(prev_results)
        curr_snapshot = build_snapshot(curr_results)

        # Act
        deltas = compute_deltas(curr_snapshot, prev_snapshot)

        # Assert
        by_probe = {d["probe"]: d for d in deltas}
        assert by_probe["P001"]["status"] == "REGRESSION"
        assert by_probe["P001"]["delta"] == pytest.approx(-0.05, abs=1e-6)
        assert by_probe["P002"]["status"] == "OK"

    def test_new_probe_flagged_as_new(self):
        prev_results = _make_probe_results({"P001": 0.95})
        curr_results = _make_probe_results({"P001": 0.95, "P003": 0.80})

        deltas = compute_deltas(
            build_snapshot(curr_results),
            build_snapshot(prev_results),
        )
        by_probe = {d["probe"]: d for d in deltas}
        assert by_probe["P003"]["status"] == "NEW"
        assert by_probe["P003"]["prev_recall"] is None


class TestMissingHistoryHandledGracefully:
    """test_missing_history_handled_gracefully"""

    def test_compare_with_no_history_file_does_not_crash(self, tmp_path, capsys):
        # Point to a non-existent history file
        history_path = str(tmp_path / "does_not_exist.jsonl")

        # Act — should not raise
        rc = cmd_compare(output_format="table", history_path=history_path)

        # Assert
        assert rc == 0
        captured = capsys.readouterr()
        assert "Not enough history" in captured.out

    def test_compare_with_single_entry_does_not_crash(self, tmp_path, capsys):
        # Create history with only 1 entry
        history_path = str(tmp_path / "regression_history.jsonl")
        probe_results = _make_probe_results({"P001": 0.95})
        snapshot = build_snapshot(probe_results)
        append_history(snapshot, path=history_path)

        # Act
        rc = cmd_compare(output_format="table", history_path=history_path)

        # Assert
        assert rc == 0
        captured = capsys.readouterr()
        assert "Not enough history" in captured.out
