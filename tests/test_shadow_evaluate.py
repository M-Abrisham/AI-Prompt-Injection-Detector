"""Tests for scripts/shadow_evaluate.py — shadow evaluation before model promotion.

Covers:
  - Metric computation (accuracy, precision, recall, F1, FPR, FNR)
  - Gate logic (pass/fail for F1, FPR, recall thresholds)
  - Disagreement detection
  - Comparison table formatting
  - Dataset loading (CSV, directory, missing columns, missing files)
  - End-to-end shadow_evaluate with mock models
  - Output format validation
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

# Force-load from scripts/shadow_evaluate.py (not scripts/data/shadow_evaluate.py)
# and register under "shadow_evaluate" so mock.patch("shadow_evaluate.xxx") works.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "shadow_evaluate",
    os.path.join(_PROJECT_ROOT, "scripts", "shadow_evaluate.py"),
)
_shadow_mod = _ilu.module_from_spec(_spec)
sys.modules["shadow_evaluate"] = _shadow_mod
_spec.loader.exec_module(_shadow_mod)

GATE_MAX_FPR_INCREASE = _shadow_mod.GATE_MAX_FPR_INCREASE
GATE_MAX_RECALL_DROP = _shadow_mod.GATE_MAX_RECALL_DROP
check_gates = _shadow_mod.check_gates
compute_metrics = _shadow_mod.compute_metrics
find_disagreements = _shadow_mod.find_disagreements
format_comparison_table = _shadow_mod.format_comparison_table
load_eval_dataset = _shadow_mod.load_eval_dataset
predict_with_model = _shadow_mod.predict_with_model
shadow_evaluate = _shadow_mod.shadow_evaluate


# ── Helpers ──────────────────────────────────────────────────────────

class FakeVectorizer:
    """Minimal sklearn-compatible vectorizer mock."""

    def transform(self, texts):
        return np.array([[1.0]] * len(texts))


class FakeModel:
    """Minimal sklearn-compatible model that returns fixed predictions."""

    def __init__(self, predictions: list[int]):
        self._preds = np.array(predictions)

    def predict(self, X):
        return self._preds[: len(X)]

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, 2))
        for i in range(n):
            if self._preds[i] == 1:
                proba[i] = [0.1, 0.9]
            else:
                proba[i] = [0.9, 0.1]
        return proba


def _make_eval_csv(tmp_dir: str, texts: list[str], labels: list[int], name: str = "eval.csv") -> str:
    """Write a small eval CSV and return its path."""
    df = pd.DataFrame({"text": texts, "label": labels})
    path = os.path.join(tmp_dir, name)
    df.to_csv(path, index=False)
    return path


# ── Metric computation ───────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["fpr"] == 0.0
        assert m["fnr"] == 0.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0
        assert m["fpr"] == 1.0
        assert m["fnr"] == 1.0
        assert m["recall"] == 0.0

    def test_mixed_predictions(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 0])
        m = compute_metrics(y_true, y_pred)
        # 1 FP out of 4 safe => FPR = 0.25
        assert abs(m["fpr"] - 0.25) < 1e-6
        # 1 FN out of 4 malicious => FNR = 0.25
        assert abs(m["fnr"] - 0.25) < 1e-6
        # Recall = 3/4 = 0.75
        assert abs(m["recall"] - 0.75) < 1e-6
        assert m["n_samples"] == 8
        assert m["n_safe"] == 4
        assert m["n_malicious"] == 4

    def test_no_positive_samples(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["fnr"] == 0.0  # no malicious samples, default 0
        assert m["n_malicious"] == 0

    def test_no_negative_samples(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 0, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["fpr"] == 0.0  # no safe samples, default 0
        assert m["n_safe"] == 0


# ── Gate logic ───────────────────────────────────────────────────────

class TestCheckGates:
    def test_all_gates_pass_identical_metrics(self):
        prod = {"f1": 0.90, "fpr": 0.05, "recall": 0.92}
        cand = {"f1": 0.90, "fpr": 0.05, "recall": 0.92}
        gates = check_gates(prod, cand)
        assert all(g["passed"] for g in gates)

    def test_all_gates_pass_candidate_better(self):
        prod = {"f1": 0.88, "fpr": 0.06, "recall": 0.90}
        cand = {"f1": 0.92, "fpr": 0.04, "recall": 0.95}
        gates = check_gates(prod, cand)
        assert all(g["passed"] for g in gates)

    def test_f1_regression_fails(self):
        prod = {"f1": 0.95, "fpr": 0.05, "recall": 0.92}
        cand = {"f1": 0.94, "fpr": 0.05, "recall": 0.92}
        gates = check_gates(prod, cand)
        f1_gate = [g for g in gates if "F1" in g["gate"]][0]
        assert not f1_gate["passed"]

    def test_fpr_increase_over_threshold_fails(self):
        prod = {"f1": 0.90, "fpr": 0.05, "recall": 0.92}
        cand = {"f1": 0.90, "fpr": 0.07, "recall": 0.92}  # +0.02 > 0.01 threshold
        gates = check_gates(prod, cand)
        fpr_gate = [g for g in gates if "FPR" in g["gate"]][0]
        assert not fpr_gate["passed"]

    def test_fpr_increase_at_boundary_passes(self):
        prod = {"f1": 0.90, "fpr": 0.05, "recall": 0.92}
        cand = {"f1": 0.90, "fpr": 0.06, "recall": 0.92}  # +0.01 == threshold
        gates = check_gates(prod, cand)
        fpr_gate = [g for g in gates if "FPR" in g["gate"]][0]
        assert fpr_gate["passed"]

    def test_recall_drop_over_threshold_fails(self):
        prod = {"f1": 0.90, "fpr": 0.05, "recall": 0.92}
        cand = {"f1": 0.90, "fpr": 0.05, "recall": 0.91}  # -0.01 > 0.005 threshold
        gates = check_gates(prod, cand)
        recall_gate = [g for g in gates if "Recall" in g["gate"]][0]
        assert not recall_gate["passed"]

    def test_recall_drop_at_boundary_passes(self):
        prod = {"f1": 0.90, "fpr": 0.05, "recall": 0.920}
        cand = {"f1": 0.90, "fpr": 0.05, "recall": 0.915}  # -0.005 == threshold
        gates = check_gates(prod, cand)
        recall_gate = [g for g in gates if "Recall" in g["gate"]][0]
        assert recall_gate["passed"]

    def test_multiple_gates_fail(self):
        prod = {"f1": 0.95, "fpr": 0.02, "recall": 0.95}
        cand = {"f1": 0.80, "fpr": 0.10, "recall": 0.80}  # all worse
        gates = check_gates(prod, cand)
        assert not any(g["passed"] for g in gates)

    def test_gate_count(self):
        """There should be exactly 3 gates."""
        prod = {"f1": 0.90, "fpr": 0.05, "recall": 0.92}
        cand = {"f1": 0.90, "fpr": 0.05, "recall": 0.92}
        gates = check_gates(prod, cand)
        assert len(gates) == 3


# ── Disagreement detection ───────────────────────────────────────────

class TestFindDisagreements:
    def test_no_disagreements(self):
        texts = pd.Series(["a", "b", "c"])
        y_true = np.array([0, 1, 0])
        y_prod = np.array([0, 1, 0])
        y_cand = np.array([0, 1, 0])
        result = find_disagreements(texts, y_true, y_prod, y_cand)
        assert result == []

    def test_disagreements_found(self):
        texts = pd.Series(["safe text", "ignore previous", "hello world"])
        y_true = np.array([0, 1, 0])
        y_prod = np.array([0, 1, 0])
        y_cand = np.array([1, 1, 1])  # candidate flags everything
        result = find_disagreements(texts, y_true, y_prod, y_cand)
        assert len(result) == 2
        assert result[0]["production_pred"] == 0
        assert result[0]["candidate_pred"] == 1
        assert "text_snippet" in result[0]
        assert "true_label" in result[0]

    def test_max_samples_cap(self):
        n = 100
        texts = pd.Series([f"text_{i}" for i in range(n)])
        y_true = np.zeros(n, dtype=int)
        y_prod = np.zeros(n, dtype=int)
        y_cand = np.ones(n, dtype=int)  # all disagree
        result = find_disagreements(texts, y_true, y_prod, y_cand, max_samples=5)
        assert len(result) == 5


# ── Comparison table formatting ──────────────────────────────────────

class TestFormatComparisonTable:
    def test_table_contains_all_metrics(self):
        m = {"accuracy": 0.9, "precision": 0.85, "recall": 0.88,
             "f1": 0.86, "fpr": 0.05, "fnr": 0.12, "n_samples": 100}
        table = format_comparison_table(m, m)
        for key in ("accuracy", "precision", "recall", "f1", "fpr", "fnr", "n_samples"):
            assert key in table

    def test_table_shows_delta(self):
        prod = {"accuracy": 0.9, "precision": 0.85, "recall": 0.88,
                "f1": 0.86, "fpr": 0.05, "fnr": 0.12, "n_samples": 100}
        cand = {"accuracy": 0.92, "precision": 0.87, "recall": 0.90,
                "f1": 0.88, "fpr": 0.04, "fnr": 0.10, "n_samples": 100}
        table = format_comparison_table(prod, cand)
        assert "Production" in table
        assert "Candidate" in table
        assert "Delta" in table


# ── Dataset loading ──────────────────────────────────────────────────

class TestLoadEvalDataset:
    def test_load_csv_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_eval_csv(tmp, ["hello", "ignore"], [0, 1])
            df = load_eval_dataset(path)
            assert len(df) == 2
            assert list(df.columns) >= ["text", "label"]

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_eval_csv(tmp, ["a", "b"], [0, 1], "part1.csv")
            _make_eval_csv(tmp, ["c", "d"], [1, 0], "part2.csv")
            df = load_eval_dataset(tmp)
            assert len(df) == 4

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_eval_dataset("/nonexistent/path/to/data.csv")

    def test_missing_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            pd.DataFrame({"foo": [1, 2]}).to_csv(path, index=False)
            with pytest.raises(ValueError, match="text.*label"):
                load_eval_dataset(path)

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                load_eval_dataset(tmp)


# ── predict_with_model ───────────────────────────────────────────────

class TestPredictWithModel:
    def test_returns_predictions(self):
        model = FakeModel([0, 1, 1, 0])
        vec = FakeVectorizer()
        texts = pd.Series(["a", "b", "c", "d"])
        preds = predict_with_model(model, vec, texts)
        np.testing.assert_array_equal(preds, [0, 1, 1, 0])


# ── End-to-end with mocks ───────────────────────────────────────────

class TestShadowEvaluateIntegration:
    def test_pass_verdict(self):
        """Candidate identical to production => PASS."""


        with tempfile.TemporaryDirectory() as tmp:
            holdout = _make_eval_csv(
                tmp,
                ["safe1", "safe2", "mal1", "mal2"],
                [0, 0, 1, 1],
            )
            output = os.path.join(tmp, "results.json")

            prod_model = FakeModel([0, 0, 1, 1])
            prod_vec = FakeVectorizer()
            cand_model = FakeModel([0, 0, 1, 1])
            cand_vec = FakeVectorizer()

            with mock.patch("shadow_evaluate.load_production_model", return_value=(prod_model, prod_vec, None)), \
                 mock.patch("shadow_evaluate.load_candidate_model", return_value=(cand_model, cand_vec, None)):
                report = shadow_evaluate(
                    candidate_path="/fake/candidate",
                    holdout_path=holdout,
                    output_path=output,
                )

            assert report["verdict"] == "PASS"
            assert len(report["failures"]) == 0
            assert os.path.isfile(output)

            with open(output) as f:
                saved = json.load(f)
            assert saved["verdict"] == "PASS"
            assert "production_metrics" in saved
            assert "candidate_metrics" in saved
            assert "gates" in saved
            assert "disagreements" in saved

    def test_fail_verdict_f1_regression(self):
        """Candidate with worse F1 => FAIL."""


        with tempfile.TemporaryDirectory() as tmp:
            holdout = _make_eval_csv(
                tmp,
                ["s1", "s2", "s3", "s4", "m1", "m2", "m3", "m4"],
                [0, 0, 0, 0, 1, 1, 1, 1],
            )
            output = os.path.join(tmp, "results.json")

            prod_model = FakeModel([0, 0, 0, 0, 1, 1, 1, 1])  # perfect
            prod_vec = FakeVectorizer()
            cand_model = FakeModel([0, 0, 0, 0, 1, 1, 0, 0])  # misses 2 malicious
            cand_vec = FakeVectorizer()

            with mock.patch("shadow_evaluate.load_production_model", return_value=(prod_model, prod_vec, None)), \
                 mock.patch("shadow_evaluate.load_candidate_model", return_value=(cand_model, cand_vec, None)):
                report = shadow_evaluate(
                    candidate_path="/fake/candidate",
                    holdout_path=holdout,
                    output_path=output,
                )

            assert report["verdict"] == "FAIL"
            assert "F1 no regression" in report["failures"]

    def test_fail_verdict_fpr_increase(self):
        """Candidate with much higher FPR => FAIL."""


        with tempfile.TemporaryDirectory() as tmp:
            # 10 safe, 2 malicious
            holdout = _make_eval_csv(
                tmp,
                [f"s{i}" for i in range(10)] + ["m1", "m2"],
                [0] * 10 + [1, 1],
            )
            output = os.path.join(tmp, "results.json")

            prod_model = FakeModel([0] * 10 + [1, 1])  # perfect
            prod_vec = FakeVectorizer()
            # Candidate: flags 3 safe as malicious => FPR = 0.3
            cand_model = FakeModel([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
            cand_vec = FakeVectorizer()

            with mock.patch("shadow_evaluate.load_production_model", return_value=(prod_model, prod_vec, None)), \
                 mock.patch("shadow_evaluate.load_candidate_model", return_value=(cand_model, cand_vec, None)):
                report = shadow_evaluate(
                    candidate_path="/fake/candidate",
                    holdout_path=holdout,
                    output_path=output,
                )

            assert report["verdict"] == "FAIL"
            assert "FPR increase" in report["failures"]

    def test_disagreements_in_report(self):
        """Report includes disagreement samples when models differ."""


        with tempfile.TemporaryDirectory() as tmp:
            holdout = _make_eval_csv(
                tmp,
                ["safe_text", "injection_text"],
                [0, 1],
            )
            output = os.path.join(tmp, "results.json")

            prod_model = FakeModel([0, 1])
            prod_vec = FakeVectorizer()
            cand_model = FakeModel([1, 1])  # disagrees on first
            cand_vec = FakeVectorizer()

            with mock.patch("shadow_evaluate.load_production_model", return_value=(prod_model, prod_vec, None)), \
                 mock.patch("shadow_evaluate.load_candidate_model", return_value=(cand_model, cand_vec, None)):
                report = shadow_evaluate(
                    candidate_path="/fake/candidate",
                    holdout_path=holdout,
                    output_path=output,
                )

            assert report["n_disagreements"] == 1
            assert len(report["disagreements"]) == 1
            assert report["disagreements"][0]["production_pred"] == 0
            assert report["disagreements"][0]["candidate_pred"] == 1

    def test_output_json_structure(self):
        """Verify all expected top-level keys in the output JSON."""


        with tempfile.TemporaryDirectory() as tmp:
            holdout = _make_eval_csv(tmp, ["a", "b"], [0, 1])
            output = os.path.join(tmp, "results.json")

            model = FakeModel([0, 1])
            vec = FakeVectorizer()

            with mock.patch("shadow_evaluate.load_production_model", return_value=(model, vec, None)), \
                 mock.patch("shadow_evaluate.load_candidate_model", return_value=(model, vec, None)):
                shadow_evaluate(
                    candidate_path="/fake",
                    holdout_path=holdout,
                    output_path=output,
                )

            with open(output) as f:
                saved = json.load(f)

            expected_keys = {
                "verdict", "production_metrics", "candidate_metrics",
                "gates", "failures", "disagreements", "n_disagreements",
                "holdout_path", "candidate_path",
            }
            assert expected_keys.issubset(set(saved.keys()))
