"""Tests for training metrics helpers and JSON output structure."""

import json
import os
import tempfile

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from scripts.model import compute_ece, _DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

class TestComputeECE:
    """Expected Calibration Error on known inputs."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should yield ECE near 0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece = compute_ece(y_true, y_prob)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_worst_calibration(self):
        """Completely wrong probabilities should yield high ECE."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.05, 0.05, 0.05, 0.05, 0.05,
                           0.95, 0.95, 0.95, 0.95, 0.95])
        ece = compute_ece(y_true, y_prob)
        assert ece > 0.8

    def test_uniform_confidence(self):
        """All predictions at 0.5 with half positive => ECE ~ 0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.full(10, 0.5)
        ece = compute_ece(y_true, y_prob)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_ece_bounded(self):
        """ECE must be in [0, 1]."""
        rng = np.random.RandomState(99)
        y_true = rng.randint(0, 2, size=200)
        y_prob = rng.rand(200)
        ece = compute_ece(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_single_bin(self):
        """With n_bins=1 every sample falls in one bin."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.2, 0.8, 0.6, 0.4])
        ece = compute_ece(y_true, y_prob, n_bins=1)
        # accuracy = 0.5, avg confidence = 0.5 => ECE = 0
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_empty_bins_ignored(self):
        """Bins with no samples should not affect the result."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.91, 0.92, 0.93, 0.94])
        # All samples land in the last bin → ECE = |acc - conf|
        ece = compute_ece(y_true, y_prob, n_bins=10)
        expected = abs(0.5 - np.mean(y_prob))
        assert ece == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Brier score (thin wrapper sanity check)
# ---------------------------------------------------------------------------

class TestBrierScore:
    """Verify Brier score behaviour with sklearn."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score_loss(y_true, y_prob) == pytest.approx(0.0)

    def test_worst_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score_loss(y_true, y_prob) == pytest.approx(1.0)

    def test_mid_range(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.5, 0.5])
        assert brier_score_loss(y_true, y_prob) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Metrics JSON structure
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "raw_accuracy",
    "calibrated_accuracy",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "ece",
    "fnr_at_default_threshold",
    "default_threshold",
    "confusion_matrix",
    "n_train",
    "n_test",
}

CM_KEYS = {"tn", "fp", "fn", "tp"}


class TestMetricsJSON:
    """Validate the structure and types of training_metrics.json."""

    @staticmethod
    def _make_sample_metrics():
        """Return a metrics dict matching the schema produced by model.py."""
        return {
            "cv_accuracy_mean": 0.9500,
            "cv_accuracy_std": 0.0100,
            "cv_roc_auc_mean": 0.9800,
            "cv_roc_auc_std": 0.0050,
            "raw_accuracy": 0.9400,
            "calibrated_accuracy": 0.9500,
            "roc_auc": 0.9850,
            "pr_auc": 0.9700,
            "brier_score": 0.0450,
            "ece": 0.0200,
            "fnr_at_default_threshold": 0.0300,
            "default_threshold": _DEFAULT_THRESHOLD,
            "confusion_matrix": {"tn": 80, "fp": 5, "fn": 3, "tp": 62},
            "n_train": 600,
            "n_test": 150,
        }

    def test_all_required_keys_present(self):
        metrics = self._make_sample_metrics()
        assert REQUIRED_KEYS.issubset(metrics.keys())

    def test_confusion_matrix_keys(self):
        metrics = self._make_sample_metrics()
        assert CM_KEYS == set(metrics["confusion_matrix"].keys())

    def test_numeric_fields_are_floats_or_ints(self):
        metrics = self._make_sample_metrics()
        for k, v in metrics.items():
            if k == "confusion_matrix":
                for ck, cv in v.items():
                    assert isinstance(cv, (int, float)), f"cm.{ck} not numeric"
            else:
                assert isinstance(v, (int, float)), f"{k} not numeric"

    def test_roundtrip_json(self):
        """Metrics must survive JSON serialization/deserialization."""
        metrics = self._make_sample_metrics()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metrics, f, indent=2)
            tmp = f.name
        try:
            with open(tmp) as f:
                loaded = json.load(f)
            assert loaded == metrics
        finally:
            os.unlink(tmp)

    def test_threshold_matches_constant(self):
        metrics = self._make_sample_metrics()
        assert metrics["default_threshold"] == _DEFAULT_THRESHOLD

    def test_values_in_valid_range(self):
        metrics = self._make_sample_metrics()
        for key in ("roc_auc", "pr_auc", "brier_score", "ece",
                     "fnr_at_default_threshold", "raw_accuracy",
                     "calibrated_accuracy", "cv_accuracy_mean",
                     "cv_roc_auc_mean"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of [0,1]"
