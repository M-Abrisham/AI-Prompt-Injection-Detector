"""Tests for attribution metrics in evaluate_probes.py."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from evaluate_probes import _print_attribution_summary, _export_attribution


def _make_probe_result(by_technique, confusion=None):
    """Helper to build a minimal probe_result dict."""
    return {
        "probe": "P999",
        "name": "mock",
        "detected": 0,
        "total": 0,
        "recall": 0.0,
        "by_technique": by_technique,
        "confusion": confusion or {},
        "missed_samples": [],
    }


class TestWorst10Ranking:
    def test_worst_10_ranking_correct(self):
        """worst_10 should be sorted by recall ascending (worst first)."""
        probe_results = [
            _make_probe_result({
                "T1": {"detected": 9, "missed": 1, "attributed": 9},   # recall 0.9
                "T2": {"detected": 3, "missed": 7, "attributed": 3},   # recall 0.3
                "T3": {"detected": 5, "missed": 5, "attributed": 5},   # recall 0.5
                "T4": {"detected": 1, "missed": 9, "attributed": 1},   # recall 0.1
            }),
        ]
        result = _print_attribution_summary(probe_results)
        assert result["worst_10"] == ["T4", "T2", "T3", "T1"]

    def test_worst_10_caps_at_10(self):
        """worst_10 should contain at most 10 entries."""
        by_tech = {}
        for i in range(15):
            by_tech[f"T{i}"] = {"detected": i + 1, "missed": 15 - i, "attributed": i}
        probe_results = [_make_probe_result(by_tech)]
        result = _print_attribution_summary(probe_results)
        assert len(result["worst_10"]) == 10


class TestConfusionPairs:
    def test_confusion_pairs_sorted_by_count(self):
        """confusion_pairs should be sorted by count descending."""
        probe_results = [
            _make_probe_result(
                {"T1": {"detected": 10, "missed": 0, "attributed": 8}},
                confusion={"T1": {"T2": 5, "T3": 1, "T4": 3}},
            ),
        ]
        result = _print_attribution_summary(probe_results)
        pairs = result["techniques"]["T1"]["confusion_pairs"]
        counts = [p["count"] for p in pairs]
        assert counts == sorted(counts, reverse=True)
        assert counts == [5, 3, 1]

    def test_confusion_excludes_self_and_none(self):
        """confusion_pairs should exclude self-attribution and _none."""
        probe_results = [
            _make_probe_result(
                {"T1": {"detected": 10, "missed": 0, "attributed": 8}},
                confusion={"T1": {"T1": 8, "_none": 2, "T2": 3}},
            ),
        ]
        result = _print_attribution_summary(probe_results)
        pairs = result["techniques"]["T1"]["confusion_pairs"]
        predicted_ids = [p["predicted"] for p in pairs]
        assert "T1" not in predicted_ids
        assert "_none" not in predicted_ids
        assert "T2" in predicted_ids


class TestExportValidJson:
    def test_export_valid_json_schema(self, tmp_path, monkeypatch):
        """Exported file should be valid JSON with required keys."""
        attribution_data = {
            "techniques": {
                "T1": {
                    "technique_id": "T1",
                    "recall": 0.5,
                    "attribution_rate": 0.8,
                    "detected": 5,
                    "total": 10,
                    "confusion_pairs": [],
                }
            },
            "worst_10": ["T1"],
        }

        import evaluate_probes as ep

        # Make _export_attribution write into tmp_path by patching __file__
        fake_scripts_dir = os.path.join(str(tmp_path), "scripts")
        os.makedirs(fake_scripts_dir, exist_ok=True)
        monkeypatch.setattr(ep, "__file__", os.path.join(fake_scripts_dir, "evaluate_probes.py"))

        _export_attribution(attribution_data)

        out_path = os.path.join(str(tmp_path), "data", "evaluation", "attribution_metrics.json")
        assert os.path.exists(out_path)
        with open(out_path) as f:
            data = json.load(f)

        assert "generated_at" in data
        assert "techniques" in data
        assert "worst_10" in data
        assert data["worst_10"] == ["T1"]
        assert "T1" in data["techniques"]
