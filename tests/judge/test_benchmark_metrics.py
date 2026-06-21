"""Tests for the trustworthy-metrics wiring in ``scripts/benchmark.py``.

These assert the Build-1b contract for the benchmark CI harness:

  * the metrics dict gains bootstrap-CI fields for the rate metrics
    (tpr/tnr/precision/recall) without dropping any pre-existing key that the
    DVC ``benchmark`` stage / ``data/evaluation/benchmark_results.json`` and the
    competitor-wrapper tests depend on;
  * accuracy is DEMOTED — it is no longer a headline row in the printed table
    and instead appears in a clearly-labeled footnote alongside prevalence;
  * the rate metrics LEAD the table (TPR/TNR/precision/recall above accuracy);
  * a degenerate (single-class) AUC path prints an explicit WARNING rather than
    silently swallowing the skip.

All data here is a tiny synthetic ``ground_truth``/``prediction`` list — no real
dataset, model, or network is touched. ``scripts.benchmark`` only imports
``na0s`` lazily inside the tool runners, so importing the module is cheap and
hits no LLM/API.
"""

from __future__ import annotations

import importlib

import pytest

# ``scripts/__init__.py`` exists, so the harness is importable as a package.
# It imports ``na0s.judge.calibration`` at module load (best-effort) but defers
# the heavy ``na0s`` scan import to call time, so this is fast and offline.
bm = importlib.import_module("scripts.benchmark")


# A 6-sample synthetic fixture with both classes present and a mix of
# correct/incorrect predictions, so every rate metric is well-defined and the
# bootstrap has both positives and negatives to resample.
_MIXED_RECORDS = [
    {"ground_truth": 1, "prediction": 1, "score": 0.90, "latency_ms": 1.0},
    {"ground_truth": 1, "prediction": 0, "score": 0.40, "latency_ms": 1.0},
    {"ground_truth": 0, "prediction": 0, "score": 0.10, "latency_ms": 1.0},
    {"ground_truth": 0, "prediction": 1, "score": 0.60, "latency_ms": 1.0},
    {"ground_truth": 1, "prediction": 1, "score": 0.80, "latency_ms": 1.0},
    {"ground_truth": 0, "prediction": 0, "score": 0.20, "latency_ms": 1.0},
]

_THRESHOLD = 0.55  # matches the harness default; value is irrelevant to metrics


# ── CI fields on the metrics dict ───────────────────────────────────────────


class TestCIFields:
    """The metrics dict must gain bootstrap-CI fields for the rate metrics."""

    @pytest.mark.skipif(
        not bm._HAS_CALIBRATION,
        reason="na0s.judge.calibration unavailable (numpy missing) — no CIs",
    )
    def test_ci_keys_present(self):
        m = bm.compute_metrics(_MIXED_RECORDS, _THRESHOLD)
        for key in ("tpr_ci", "tnr_ci", "precision_ci", "recall_ci"):
            assert key in m, f"missing CI field: {key}"

    @pytest.mark.skipif(
        not bm._HAS_CALIBRATION,
        reason="na0s.judge.calibration unavailable (numpy missing) — no CIs",
    )
    def test_ci_is_ordered_pair_bracketing_point_estimate(self):
        m = bm.compute_metrics(_MIXED_RECORDS, _THRESHOLD)
        for metric, ci_key in (
            ("tpr", "tpr_ci"),
            ("tnr", "tnr_ci"),
            ("precision", "precision_ci"),
            ("recall", "recall_ci"),
        ):
            ci = m[ci_key]
            assert ci is not None, f"{ci_key} should be populated for mixed data"
            lo, hi = ci
            assert lo <= hi, f"{ci_key} not ordered: {ci}"
            # The percentile bootstrap interval should bracket the point estimate.
            assert lo <= m[metric] <= hi, (
                f"{metric}={m[metric]} outside its CI {ci}"
            )

    @pytest.mark.skipif(
        not bm._HAS_CALIBRATION,
        reason="na0s.judge.calibration unavailable (numpy missing) — no CIs",
    )
    def test_ci_level_field_reports_confidence(self):
        m = bm.compute_metrics(_MIXED_RECORDS, _THRESHOLD)
        # Default alpha=0.05 -> a 95% interval.
        assert m["ci_level"] == pytest.approx(0.95)

    def test_existing_keys_preserved(self):
        """ADD CI fields, do NOT break keys the DVC stage / other tests read."""
        m = bm.compute_metrics(_MIXED_RECORDS, _THRESHOLD)
        # The original benchmark schema (pre-Build-1b) — must all survive.
        for key in (
            "n_samples", "n_malicious", "n_safe",
            "tp", "tn", "fp", "fn",
            "precision", "recall", "f1", "fpr", "accuracy",
            "auc_roc", "auc_pr",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms",
            "p99_latency_ms", "throughput_per_sec", "threshold",
        ):
            assert key in m, f"regression: existing key dropped: {key}"

    def test_new_rate_and_prevalence_keys_added(self):
        m = bm.compute_metrics(_MIXED_RECORDS, _THRESHOLD)
        # tpr/tnr/prevalence are new explicit fields backing the headline table.
        for key in ("tpr", "tnr", "prevalence"):
            assert key in m, f"missing new field: {key}"
        # 3 malicious of 6 samples.
        assert m["prevalence"] == pytest.approx(0.5)
        # tpr = recall here (both == TP / (TP+FN)).
        assert m["tpr"] == pytest.approx(m["recall"])


# ── accuracy demoted from the headline ──────────────────────────────────────


class TestAccuracyDemoted:
    """Accuracy must not be the first/headline metric in the printed table."""

    def _capture_table(self, capsys):
        m = bm.compute_metrics(_MIXED_RECORDS, _THRESHOLD)
        bm.print_summary_table(m, "na0s", "synthetic.jsonl")
        return capsys.readouterr().out

    def test_accuracy_not_a_table_row(self, capsys):
        out = self._capture_table(capsys)
        # No "| Accuracy" table row anymore — it was demoted to a footnote.
        assert "| Accuracy" not in out, (
            "accuracy must not be a headline table row; demote to footnote"
        )

    def test_accuracy_appears_only_as_labeled_footnote_with_prevalence(self, capsys):
        out = self._capture_table(capsys)
        assert "Footnote" in out, "accuracy footnote label missing"
        # The footnote must show accuracy AND prevalence together so a
        # TN-dominated accuracy is not misread.
        assert "accuracy:" in out
        assert "prevalence=" in out

    def test_rate_metrics_lead_before_accuracy_footnote(self, capsys):
        out = self._capture_table(capsys)
        # The honest rate metrics must appear before the accuracy footnote.
        idx_tpr = out.find("TPR")
        idx_precision = out.find("Precision")
        idx_recall = out.find("Recall")
        idx_footnote = out.find("Footnote")
        assert idx_tpr != -1 and idx_precision != -1 and idx_recall != -1
        assert idx_footnote != -1
        assert idx_tpr < idx_footnote, "TPR must lead the accuracy footnote"
        assert idx_precision < idx_footnote
        assert idx_recall < idx_footnote

    def test_first_rate_row_is_tpr_not_accuracy(self, capsys):
        out = self._capture_table(capsys)
        # Among the rate rows, TPR (recall) is the first one printed.
        idx_tpr = out.find("TPR (recall)")
        idx_precision = out.find("| Precision")
        idx_recall = out.find("| Recall")
        assert idx_tpr != -1
        assert idx_tpr < idx_precision < idx_recall


# ── AUC skip emits an explicit warning (not silently swallowed) ──────────────


class TestAUCWarning:
    """A skipped AUC must print an explicit WARNING."""

    def test_single_class_dataset_warns(self, capsys):
        # All-positive dataset -> AUC undefined -> must warn, not silently skip.
        single_class = [
            {"ground_truth": 1, "prediction": 1, "score": 0.9, "latency_ms": 1.0},
            {"ground_truth": 1, "prediction": 0, "score": 0.4, "latency_ms": 1.0},
        ]
        m = bm.compute_metrics(single_class, _THRESHOLD)
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "AUC" in err
        assert "single-class" in err
        # AUC stays None (undefined) but the skip is now loud.
        assert m["auc_roc"] is None
        assert m["auc_pr"] is None

    def test_missing_sklearn_warns(self, capsys, monkeypatch):
        """If sklearn import fails, the AUC skip must WARN, not pass silently."""
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "sklearn.metrics" or name.startswith("sklearn"):
                raise ImportError("simulated: sklearn not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        m = bm.compute_metrics(_MIXED_RECORDS, _THRESHOLD)
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "sklearn" in err
        # No AUC computed, but the absence is surfaced.
        assert m["auc_roc"] is None
        assert m["auc_pr"] is None
