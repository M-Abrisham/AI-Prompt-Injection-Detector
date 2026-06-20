"""Tests for na0s.judge.calibration.

Known-answer tests for Rogan-Gladen + confusion metrics, statistical sanity for
the bootstrap CI, and a regression guard that the calibration report does NOT
headline accuracy (the "report recall, not accuracy" rule).
"""

from __future__ import annotations

import math

import pytest

from na0s.judge import calibration as cal


# ── Rogan-Gladen ────────────────────────────────────────────────────────────


class TestRoganGladen:
    def test_known_example(self):
        # (0.30 + 0.95 - 1) / (0.90 + 0.95 - 1) = 0.25 / 0.85
        assert cal.rogan_gladen(0.30, 0.90, 0.95) == pytest.approx(0.25 / 0.85, abs=1e-9)

    def test_perfect_judge_is_identity(self):
        for apparent in (0.0, 0.3, 1.0):
            assert cal.rogan_gladen(apparent, 1.0, 1.0) == pytest.approx(apparent)

    def test_degenerate_judge_raises_not_silent_zero(self):
        with pytest.raises(ValueError):
            cal.rogan_gladen(0.3, 0.5, 0.5)        # tpr+tnr-1 == 0
        with pytest.raises(ValueError):
            cal.rogan_gladen(0.3, 0.4, 0.5)        # tpr+tnr-1 < 0 (worse than chance)

    def test_clamps_below_floor_to_zero(self):
        # apparent below (1 - tnr) drives the raw estimate negative -> clamp 0.
        assert cal.rogan_gladen(0.0, 0.9, 0.95) == 0.0

    def test_clamps_above_one(self):
        assert cal.rogan_gladen(1.0, 0.9, 0.95) == 1.0


# ── confusion metrics ───────────────────────────────────────────────────────


class TestConfusionMetrics:
    def test_known_counts(self):
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        m = cal.confusion_metrics(y_true, y_pred)
        assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 1, 1)
        assert m["n"] == 5
        assert m["tpr"] == pytest.approx(2 / 3)
        assert m["tnr"] == pytest.approx(0.5)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["prevalence_apparent"] == pytest.approx(0.6)  # (tp+fp)/n

    def test_empty_is_zero_not_crash(self):
        m = cal.confusion_metrics([], [])
        assert m["n"] == 0 and m["tpr"] == 0.0


# ── bootstrap CI ────────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_ci_brackets_known_recall(self):
        # all positives; 80% predicted positive -> recall == 0.8
        y_true = [1] * 500
        y_pred = [1] * 400 + [0] * 100
        lo, hi = cal.bootstrap_ci(y_true, y_pred, cal.recall_stat, n_boot=1000, seed=0)
        assert lo <= 0.8 <= hi
        assert lo < hi  # a real interval, not a point

    def test_ci_narrows_with_n(self):
        small = cal.bootstrap_ci([1] * 50, [1] * 40 + [0] * 10, cal.recall_stat,
                                 n_boot=1000, seed=0)
        large = cal.bootstrap_ci([1] * 500, [1] * 400 + [0] * 100, cal.recall_stat,
                                 n_boot=1000, seed=0)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_empty_returns_nan(self):
        lo, hi = cal.bootstrap_ci([], [], cal.recall_stat)
        assert math.isnan(lo) and math.isnan(hi)

    def test_deterministic_with_seed(self):
        a = cal.bootstrap_ci([1] * 100, [1] * 70 + [0] * 30, cal.recall_stat, seed=7)
        b = cal.bootstrap_ci([1] * 100, [1] * 70 + [0] * 30, cal.recall_stat, seed=7)
        assert a == b


# ── top-level calibrate() + anti-accuracy guard ─────────────────────────────


class TestCalibrate:
    def test_report_has_no_accuracy_headline(self):
        res = cal.calibrate([1, 0, 1, 0], [1, 0, 0, 0])
        d = res.to_dict()
        assert "accuracy" not in d
        assert {"tpr", "tnr", "recall", "precision"} <= set(d)

    def test_corrected_prevalence_present_for_good_judge(self):
        # 100 items, true prevalence 0.5, a decent (not perfect) judge.
        y_true = [1] * 50 + [0] * 50
        # 45/50 TP (tpr=0.9), 47/50 TN (tnr=0.94)
        y_pred = [1] * 45 + [0] * 5 + [0] * 47 + [1] * 3
        res = cal.calibrate(y_true, y_pred, n_boot=500, seed=0)
        assert res.prevalence_corrected is not None
        # corrected should land near the true 0.5
        assert 0.4 <= res.prevalence_corrected <= 0.6
        lo, hi = res.prevalence_corrected_ci
        assert lo <= res.prevalence_corrected <= hi

    def test_degenerate_judge_corrected_is_none_not_crash(self):
        # tpr=0, tnr=0 -> Rogan-Gladen undefined -> None, no exception.
        res = cal.calibrate([1, 0], [0, 1], n_boot=100, seed=0)
        assert res.prevalence_corrected is None
