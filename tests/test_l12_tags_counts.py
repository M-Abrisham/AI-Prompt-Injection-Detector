"""Tests for per-probe counts and top-missed-technique aggregation in _tags.py."""

import pytest

from scripts.taxonomy._tags import (
    aggregation_summary,
    count_by_probe,
    top_missed_techniques,
)


# ---------------------------------------------------------------------------
# Helpers to build mock probe results
# ---------------------------------------------------------------------------

def _make_result(
    probe="D1",
    total=10,
    detected=8,
    missed=2,
    false_positives=0,
    by_technique=None,
    by_difficulty=None,
    by_evasion_type=None,
):
    """Return a minimal probe-result dict matching Probe.evaluate() shape."""
    return {
        "probe": probe,
        "name": f"Probe {probe}",
        "tags": [],
        "total": total,
        "detected": detected,
        "missed": missed,
        "false_positives": false_positives,
        "by_technique": by_technique or {},
        "by_difficulty": by_difficulty or {},
        "by_evasion_type": by_evasion_type or {},
        "scores": [],
    }


# ===== count_by_probe =====

class TestCountByProbe:
    def test_single_probe(self):
        results = [_make_result(probe="D1", total=10, detected=7, missed=3)]
        counts = count_by_probe(results)
        assert "D1" in counts
        assert counts["D1"]["total_samples"] == 10
        assert counts["D1"]["detected"] == 7
        assert counts["D1"]["missed"] == 3
        assert counts["D1"]["recall"] == pytest.approx(7 / 10)

    def test_multiple_probes(self):
        results = [
            _make_result(probe="D1", total=10, detected=8, missed=2),
            _make_result(probe="D5", total=20, detected=15, missed=5),
        ]
        counts = count_by_probe(results)
        assert len(counts) == 2
        assert counts["D5"]["total_samples"] == 20
        assert counts["D5"]["recall"] == pytest.approx(15 / 20)

    def test_duplicate_probe_ids_accumulate(self):
        results = [
            _make_result(probe="D1", total=5, detected=3, missed=2, false_positives=1),
            _make_result(probe="D1", total=5, detected=5, missed=0, false_positives=0),
        ]
        counts = count_by_probe(results)
        assert counts["D1"]["total_samples"] == 10
        assert counts["D1"]["detected"] == 8
        assert counts["D1"]["missed"] == 2
        assert counts["D1"]["false_positives"] == 1
        assert counts["D1"]["recall"] == pytest.approx(8 / 10)

    def test_empty_results(self):
        assert count_by_probe([]) == {}

    def test_all_detected(self):
        results = [_make_result(probe="E", total=5, detected=5, missed=0)]
        counts = count_by_probe(results)
        assert counts["E"]["recall"] == pytest.approx(1.0)
        assert counts["E"]["missed"] == 0

    def test_none_detected(self):
        results = [_make_result(probe="T", total=5, detected=0, missed=5)]
        counts = count_by_probe(results)
        assert counts["T"]["recall"] == pytest.approx(0.0)
        assert counts["T"]["missed"] == 5

    def test_false_positives_tracked(self):
        results = [_make_result(probe="D1", total=10, detected=8, missed=2, false_positives=3)]
        counts = count_by_probe(results)
        assert counts["D1"]["false_positives"] == 3


# ===== top_missed_techniques =====

class TestTopMissedTechniques:
    def test_ordering_by_missed_count(self):
        results = [_make_result(
            by_technique={
                "D1.1": {"detected": 5, "missed": 10},
                "D1.2": {"detected": 5, "missed": 20},
                "D1.3": {"detected": 5, "missed": 15},
            },
        )]
        top = top_missed_techniques(results, n=10)
        assert len(top) == 3
        assert top[0]["technique_id"] == "D1.2"
        assert top[0]["missed_count"] == 20
        assert top[1]["technique_id"] == "D1.3"
        assert top[2]["technique_id"] == "D1.1"

    def test_n_limits_output(self):
        by_tech = {f"T{i}": {"detected": 1, "missed": i} for i in range(1, 20)}
        results = [_make_result(by_technique=by_tech)]
        top = top_missed_techniques(results, n=5)
        assert len(top) == 5
        assert top[0]["missed_count"] == 19

    def test_empty_results(self):
        assert top_missed_techniques([], n=10) == []

    def test_no_misses_returns_empty(self):
        results = [_make_result(
            by_technique={"D1.1": {"detected": 10, "missed": 0}},
        )]
        assert top_missed_techniques(results) == []

    def test_miss_rate_calculation(self):
        results = [_make_result(
            by_technique={"D1.1": {"detected": 3, "missed": 7}},
        )]
        top = top_missed_techniques(results)
        assert len(top) == 1
        assert top[0]["miss_rate"] == pytest.approx(7 / 10)
        assert top[0]["total_count"] == 10

    def test_merges_across_probes(self):
        r1 = _make_result(probe="D1", by_technique={"T1": {"detected": 5, "missed": 3}})
        r2 = _make_result(probe="D5", by_technique={"T1": {"detected": 2, "missed": 4}})
        top = top_missed_techniques([r1, r2])
        assert len(top) == 1
        assert top[0]["technique_id"] == "T1"
        assert top[0]["missed_count"] == 7
        assert top[0]["total_count"] == 14

    def test_tiebreak_alphabetical(self):
        results = [_make_result(
            by_technique={
                "B": {"detected": 0, "missed": 5},
                "A": {"detected": 0, "missed": 5},
            },
        )]
        top = top_missed_techniques(results)
        assert top[0]["technique_id"] == "A"
        assert top[1]["technique_id"] == "B"


# ===== aggregation_summary =====

class TestAggregationSummary:
    def test_completeness_keys(self):
        results = [_make_result()]
        summary = aggregation_summary(results)
        assert "per_probe" in summary
        assert "top_missed" in summary
        assert "by_difficulty" in summary
        assert "by_evasion_type" in summary
        assert "overall" in summary

    def test_overall_totals(self):
        results = [
            _make_result(probe="D1", total=10, detected=8, missed=2, false_positives=1),
            _make_result(probe="D5", total=20, detected=15, missed=5, false_positives=2),
        ]
        summary = aggregation_summary(results)
        ov = summary["overall"]
        assert ov["total"] == 30
        assert ov["detected"] == 23
        assert ov["missed"] == 7
        assert ov["recall"] == pytest.approx(23 / 30)
        assert ov["false_positives"] == 3

    def test_difficulty_merge(self):
        r1 = _make_result(by_difficulty={
            "basic": {"detected": 5, "missed": 1, "total": 6},
            "advanced": {"detected": 2, "missed": 3, "total": 5},
        })
        r2 = _make_result(by_difficulty={
            "basic": {"detected": 4, "missed": 0, "total": 4},
        })
        summary = aggregation_summary([r1, r2])
        bd = summary["by_difficulty"]
        assert bd["basic"]["detected"] == 9
        assert bd["basic"]["total"] == 10
        assert bd["basic"]["recall"] == pytest.approx(9 / 10)
        assert bd["advanced"]["recall"] == pytest.approx(2 / 5)

    def test_evasion_type_merge(self):
        r1 = _make_result(by_evasion_type={
            "semantic": {"detected": 3, "missed": 2, "total": 5},
        })
        r2 = _make_result(by_evasion_type={
            "semantic": {"detected": 7, "missed": 3, "total": 10},
            "token": {"detected": 4, "missed": 1, "total": 5},
        })
        summary = aggregation_summary([r1, r2])
        be = summary["by_evasion_type"]
        assert be["semantic"]["detected"] == 10
        assert be["semantic"]["total"] == 15
        assert be["token"]["recall"] == pytest.approx(4 / 5)

    def test_empty_results(self):
        summary = aggregation_summary([])
        assert summary["per_probe"] == {}
        assert summary["top_missed"] == []
        assert summary["by_difficulty"] == {}
        assert summary["by_evasion_type"] == {}
        assert summary["overall"]["total"] == 0
        assert summary["overall"]["recall"] == 0.0
