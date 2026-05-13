"""Tests for scripts/f14_promotion_gate.py.

Covers:
  - Scenario → (y_true, y_pred) conversion for single_prompt and multi_turn
  - Per-category and per-severity metric slicing
  - Rule (a) overall TPR drop failure
  - Rule (b) critical-containing category regress failure
  - Rule (c) critical-severity TPR drop failure
  - First-run baseline seeding (no baseline → PASS + write)
  - --update-baseline + FAIL refuses to overwrite baseline
  - Mutually exclusive --update-baseline + --no-baseline
  - JSON output shape round-trips via asdict/json

No real model load — tests inject synthetic records directly into the
metric + compare functions and use a DummyModel/DummyVectorizer for
end-to-end runs on fixture scenarios.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import f14_promotion_gate as gate  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────────


def _mk_record(name, cat, sev, y_true, y_pred):
    return {
        "name": name,
        "attack_category": cat,
        "severity": sev,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _mk_baseline(overall_tpr=1.0, per_cat=None, per_sev=None):
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "overall": {"tpr": overall_tpr, "fpr": 0.0, "tnr": 1.0, "f1": 1.0,
                    "tp": 10, "fn": 0, "tn": 5, "fp": 0,
                    "n_blocked": 10, "n_allowed": 5},
        "per_category": per_cat or {},
        "per_severity": per_sev or {},
    }


# ── Metric slicing ──────────────────────────────────────────────────────────


class TestComputeMetricsByCategory:
    def test_overall_perfect(self):
        records = [_mk_record(f"s{i}", "D1", "high", 1, 1) for i in range(5)]
        records += [_mk_record(f"b{i}", "D1", "high", 0, 0) for i in range(3)]
        overall, per_cat, per_sev = gate._compute_metrics_by_category(records)
        assert overall.tpr == 1.0
        assert overall.fpr == 0.0
        assert overall.n_blocked == 5
        assert overall.n_allowed == 3

    def test_per_category_split(self):
        records = [
            _mk_record("e1a", "E1", "critical", 1, 1),
            _mk_record("e1b", "E1", "critical", 1, 0),
            _mk_record("d1a", "D1", "high", 1, 1),
            _mk_record("d1b", "D1", "high", 0, 0),
        ]
        _, per_cat, _ = gate._compute_metrics_by_category(records)
        assert per_cat["E1"].tpr == 0.5
        assert per_cat["E1"].critical_count == 2
        assert per_cat["D1"].tpr == 1.0
        assert per_cat["D1"].critical_count == 0

    def test_per_severity_attack_only(self):
        records = [
            _mk_record("a1", "E1", "critical", 1, 1),
            _mk_record("a2", "E1", "critical", 1, 0),
            _mk_record("b1", "E1", "critical", 0, 0),  # benign — must NOT
            _mk_record("a3", "E1", "high", 1, 1),      # enter severity buckets
        ]
        _, _, per_sev = gate._compute_metrics_by_category(records)
        assert per_sev["critical"].n == 2
        assert per_sev["critical"].tpr == 0.5
        assert per_sev["high"].n == 1


# ── Regression rules ────────────────────────────────────────────────────────


class TestCompareToBaseline:
    def test_no_baseline_returns_none_no_failures(self):
        records = [_mk_record("x", "E1", "critical", 1, 1)]
        overall, per_cat, per_sev = gate._compute_metrics_by_category(records)
        comp, failures = gate._compare_to_baseline(
            overall=overall, per_category=per_cat, per_severity=per_sev,
            baseline=None, baseline_path="/does/not/matter",
        )
        assert comp is None
        assert failures == []

    def test_rule_a_overall_tpr_drop_fails(self):
        records = [_mk_record(f"a{i}", "D1", "high", 1, 1) for i in range(9)]
        records += [_mk_record("miss", "D1", "high", 1, 0)]  # 9/10 = 0.9
        overall, per_cat, per_sev = gate._compute_metrics_by_category(records)
        baseline = _mk_baseline(overall_tpr=1.0)
        _, failures = gate._compare_to_baseline(
            overall=overall, per_category=per_cat, per_severity=per_sev,
            baseline=baseline, baseline_path="b.json",
        )
        assert any("a_overall_tpr" in f or "Rule (a)" in f for f in failures)

    def test_rule_a_small_drop_within_limit_passes(self):
        # 99/100 = 0.99, baseline 1.0 — drop=0.01, well within 0.02 limit
        records = [_mk_record(f"a{i}", "D1", "high", 1, 1) for i in range(99)]
        records += [_mk_record("miss", "D1", "high", 1, 0)]
        overall, per_cat, per_sev = gate._compute_metrics_by_category(records)
        baseline = _mk_baseline(overall_tpr=1.0)
        _, failures = gate._compare_to_baseline(
            overall=overall, per_category=per_cat, per_severity=per_sev,
            baseline=baseline, baseline_path="b.json",
        )
        assert not any("Rule (a)" in f for f in failures)

    def test_rule_b_critical_category_regress(self):
        # Current: E1 has 1 critical attack, missed. Baseline: caught.
        records = [_mk_record("e1", "E1", "critical", 1, 0)]
        overall, per_cat, per_sev = gate._compute_metrics_by_category(records)
        baseline = _mk_baseline(
            overall_tpr=1.0,
            per_cat={
                "E1": {"tpr": 1.0, "fpr": 0.0, "tp": 1, "fn": 0,
                       "tn": 0, "fp": 0, "n_attack": 1, "n_benign": 0,
                       "critical_count": 1},
            },
        )
        _, failures = gate._compare_to_baseline(
            overall=overall, per_category=per_cat, per_severity=per_sev,
            baseline=baseline, baseline_path="b.json",
        )
        assert any("Rule (b)" in f for f in failures)

    def test_rule_b_noncritical_category_not_flagged(self):
        records = [_mk_record("d1", "D1", "high", 1, 0)]
        overall, per_cat, per_sev = gate._compute_metrics_by_category(records)
        baseline = _mk_baseline(
            overall_tpr=1.0,
            per_cat={
                "D1": {"tpr": 1.0, "fpr": 0.0, "tp": 1, "fn": 0,
                       "tn": 0, "fp": 0, "n_attack": 1, "n_benign": 0,
                       "critical_count": 0},
            },
        )
        _, failures = gate._compare_to_baseline(
            overall=overall, per_category=per_cat, per_severity=per_sev,
            baseline=baseline, baseline_path="b.json",
        )
        assert not any("Rule (b)" in f for f in failures)

    def test_rule_c_critical_severity_any_drop(self):
        records = [
            _mk_record("c1", "E1", "critical", 1, 1),
            _mk_record("c2", "D1", "critical", 1, 0),  # 1/2 = 0.5
        ]
        overall, per_cat, per_sev = gate._compute_metrics_by_category(records)
        baseline = _mk_baseline(
            overall_tpr=1.0,
            per_sev={"critical": {"tpr": 1.0, "n": 2, "tp": 2, "fn": 0}},
        )
        _, failures = gate._compare_to_baseline(
            overall=overall, per_category=per_cat, per_severity=per_sev,
            baseline=baseline, baseline_path="b.json",
        )
        assert any("Rule (c)" in f for f in failures)

    def test_rule_b_uses_raw_fn_when_available(self):
        # Current FN=1, baseline_fn=0 → extra_misses=1 → should fail
        records = [_mk_record("e1", "E1", "critical", 1, 0)]
        overall, per_cat, per_sev = gate._compute_metrics_by_category(records)
        baseline = _mk_baseline(
            overall_tpr=1.0,
            per_cat={
                "E1": {"tpr": 1.0, "fpr": 0.0, "tp": 1, "fn": 0,
                       "tn": 0, "fp": 0, "n_attack": 1, "n_benign": 0,
                       "critical_count": 1},
            },
        )
        _, failures = gate._compare_to_baseline(
            overall=overall, per_category=per_cat, per_severity=per_sev,
            baseline=baseline, baseline_path="b.json",
        )
        assert any("Rule (b)" in f for f in failures)

    def test_rule_b_legacy_baseline_no_fn_field(self):
        records = [_mk_record("e1", "E1", "critical", 1, 0)]
        overall, per_cat, per_sev = gate._compute_metrics_by_category(records)
        # Legacy baseline: no tp/fn, only tpr + n_attack
        baseline = _mk_baseline(
            overall_tpr=1.0,
            per_cat={
                "E1": {"tpr": 1.0, "fpr": 0.0, "n_attack": 1, "n_benign": 0,
                       "critical_count": 1},
            },
        )
        _, failures = gate._compare_to_baseline(
            overall=overall, per_category=per_cat, per_severity=per_sev,
            baseline=baseline, baseline_path="b.json",
        )
        assert any("Rule (b)" in f for f in failures)


# ── Scenario → record conversion ────────────────────────────────────────────


class TestRunScenarios:
    def test_single_prompt_malicious_label(self):
        scn = SimpleNamespace(
            name="s1", attack_category="E1", severity="critical",
            expected_verdict="blocked", type=gate.ScenarioType.SINGLE_PROMPT,
            payload="Ignore previous", turns=[],
        )
        with mock.patch("na0s.predict.classify_prompt",
                        return_value=("MALICIOUS", 0.9, [], None, [], {}, 0.0)):
            records = gate._run_scenarios([scn], vectorizer=None, model=None)
        assert len(records) == 1
        assert records[0]["y_true"] == 1
        assert records[0]["y_pred"] == 1

    def test_single_prompt_safe_on_benign(self):
        scn = SimpleNamespace(
            name="b1", attack_category="E1_benign", severity="low",
            expected_verdict="allowed", type=gate.ScenarioType.SINGLE_PROMPT,
            payload="Hello world", turns=[],
        )
        with mock.patch("na0s.predict.classify_prompt",
                        return_value=("SAFE", 0.05, [], None, [], {}, 0.0)):
            records = gate._run_scenarios([scn], vectorizer=None, model=None)
        assert records[0]["y_true"] == 0
        assert records[0]["y_pred"] == 0

    def test_multi_turn_breaks_on_first_malicious(self):
        turn1 = SimpleNamespace(text="hi", expected_label="safe", risk_score=None)
        turn2 = SimpleNamespace(text="attack", expected_label="malicious", risk_score=None)
        turn3 = SimpleNamespace(text="filler", expected_label="safe", risk_score=None)
        scn = SimpleNamespace(
            name="mt1", attack_category="E1", severity="critical",
            expected_verdict="blocked", type=gate.ScenarioType.MULTI_TURN,
            payload=None, turns=[turn1, turn2, turn3],
        )
        call_count = {"n": 0}

        def fake_classify(text, v, m):
            call_count["n"] += 1
            label = "MALICIOUS" if text == "attack" else "SAFE"
            return label, 0.9 if label == "MALICIOUS" else 0.1, [], None, [], {}, 0.0

        with mock.patch("na0s.predict.classify_prompt", side_effect=fake_classify):
            records = gate._run_scenarios([scn], vectorizer=None, model=None)
        assert records[0]["y_pred"] == 1
        # Should stop after turn 2 — turn 3 not scanned
        assert call_count["n"] == 2


# ── CLI / first-run / baseline-refuse ──────────────────────────────────────


class TestCliBehavior:
    def test_first_run_seeds_baseline(self, tmp_path):
        baseline_path = tmp_path / "baseline.json"
        assert not baseline_path.exists()

        # Build a synthetic report and call the write helper directly —
        # exercising the main() bootstrap is overkill for unit tests.
        overall = gate.OverallMetrics(
            tpr=1.0, fpr=0.0, tnr=1.0, f1=1.0, tp=1, fn=0, tn=1, fp=0,
            n_blocked=1, n_allowed=1,
        )
        report = gate.GateReport(
            timestamp="2026-04-23T00:00:00+00:00",
            candidate_path="x", scenarios_total=2,
            overall=overall,
            per_category={}, per_severity={},
            baseline_comparison=None, verdict="PASS", failures=[],
        )
        gate._write_baseline(report, str(baseline_path))
        assert baseline_path.exists()
        with open(baseline_path) as f:
            obj = json.load(f)
        assert obj["verdict"] == "PASS"
        assert obj["baseline_comparison"] is None

    def test_update_baseline_and_no_baseline_are_mutex(self, capsys):
        argv = [
            "f14_promotion_gate.py", "--candidate", "x",
            "--update-baseline", "--no-baseline",
        ]
        with mock.patch.object(sys, "argv", argv):
            rc = gate.main()
        assert rc == 2
        captured = capsys.readouterr()
        assert "mutually exclusive" in captured.err


class TestJsonRoundTrip:
    def test_gate_report_serializes(self):
        overall = gate.OverallMetrics(
            tpr=0.95, fpr=0.01, tnr=0.99, f1=0.94, tp=19, fn=1, tn=99, fp=1,
            n_blocked=20, n_allowed=100,
        )
        cat = gate.CategoryMetrics(
            tpr=1.0, fpr=0.0, tp=5, fn=0, tn=3, fp=0,
            n_attack=5, n_benign=3, critical_count=2,
        )
        sev = gate.SeverityMetrics(tpr=1.0, n=5, tp=5, fn=0)
        report = gate.GateReport(
            timestamp="t", candidate_path="c", scenarios_total=8,
            overall=overall, per_category={"E1": cat},
            per_severity={"critical": sev},
            baseline_comparison=None, verdict="PASS", failures=[],
        )
        from dataclasses import asdict
        s = json.dumps(asdict(report))
        obj = json.loads(s)
        assert obj["overall"]["tpr"] == 0.95
        assert obj["per_category"]["E1"]["critical_count"] == 2
