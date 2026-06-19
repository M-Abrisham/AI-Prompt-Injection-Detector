"""Tests for the memory-safe training-set cap in scripts/features.py.

The Auto-Retrain feature-extraction step OOMs on the full ~2.2M-row
combined_data.csv. ``apply_stratified_cap`` stratified-downsamples the
dataset to a memory-safe ceiling (``NA0S_MAX_TRAIN_ROWS``, default
``_DEFAULT_MAX_TRAIN_ROWS``) so feature extraction stays tractable on a
standard runner, while preserving both class proportions.

These tests exercise the REAL functions imported from scripts.features
(not a re-implementation), covering:
- downsample to the ceiling when ceiling < rows,
- both labels preserved (stratified),
- determinism (random_state=42),
- no-op when ceiling >= rows,
- env override via NA0S_MAX_TRAIN_ROWS (including 0 = disabled).

Run with:
    python -m pytest tests/test_features_train_cap.py -v
"""

import os
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.features import (
    apply_stratified_cap,
    resolve_max_train_rows,
    _DEFAULT_MAX_TRAIN_ROWS,
    _MAX_TRAIN_ROWS_ENV,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_dataset(n_safe, n_malicious):
    """Build a two-class DataFrame with distinct, unique texts per row."""
    safe = pd.DataFrame(
        {"text": [f"benign sample number {i}" for i in range(n_safe)], "label": 0}
    )
    mal = pd.DataFrame(
        {"text": [f"malicious payload number {i}" for i in range(n_malicious)], "label": 1}
    )
    return pd.concat([safe, mal], ignore_index=True)


# ---------------------------------------------------------------------------
# apply_stratified_cap: downsampling behaviour
# ---------------------------------------------------------------------------

class TestStratifiedCapDownsamples:
    """When ceiling < rows, downsample to exactly the ceiling."""

    def test_downsamples_to_ceiling(self):
        # 1200 rows (1000 safe + 200 malicious), ceiling 600 < 1200.
        df = _make_dataset(n_safe=1000, n_malicious=200)
        assert len(df) == 1200
        capped = apply_stratified_cap(df, max_rows=600)
        assert len(capped) == 600

    def test_both_labels_preserved(self):
        df = _make_dataset(n_safe=1000, n_malicious=200)
        capped = apply_stratified_cap(df, max_rows=600)
        labels = set(capped["label"].unique())
        assert labels == {0, 1}, f"Expected both classes, got {labels}"

    def test_class_proportions_preserved(self):
        # 5:1 ratio (1000:200). After capping to 600, expect ~500:100.
        df = _make_dataset(n_safe=1000, n_malicious=200)
        capped = apply_stratified_cap(df, max_rows=600)
        counts = capped["label"].value_counts()
        # Proportional allocation: 600 * 1000/1200 = 500, 600 * 200/1200 = 100
        assert counts[0] == 500
        assert counts[1] == 100

    def test_deterministic(self):
        df = _make_dataset(n_safe=1000, n_malicious=200)
        r1 = apply_stratified_cap(df, max_rows=600)
        r2 = apply_stratified_cap(df, max_rows=600)
        pd.testing.assert_frame_equal(
            r1.sort_values("text").reset_index(drop=True),
            r2.sort_values("text").reset_index(drop=True),
        )

    def test_exact_row_count_with_rounding_remainder(self):
        # Choose counts so proportional floors don't sum to the ceiling and
        # the remainder distribution must kick in to land exactly on max_rows.
        df = _make_dataset(n_safe=700, n_malicious=300)  # 1000 rows
        capped = apply_stratified_cap(df, max_rows=333)
        assert len(capped) == 333
        # Both classes still present after the floor+remainder allocation.
        assert set(capped["label"].unique()) == {0, 1}

    def test_tiny_minority_class_never_dropped(self):
        # Minority class of 3 rows must survive even an aggressive cap.
        df = _make_dataset(n_safe=10000, n_malicious=3)
        capped = apply_stratified_cap(df, max_rows=100)
        assert len(capped) == 100
        assert (capped["label"] == 1).sum() >= 1


# ---------------------------------------------------------------------------
# apply_stratified_cap: no-op cases
# ---------------------------------------------------------------------------

class TestStratifiedCapNoOp:
    """No downsampling when ceiling >= rows or cap is disabled."""

    def test_noop_when_ceiling_equals_rows(self):
        df = _make_dataset(n_safe=300, n_malicious=300)  # 600 rows
        capped = apply_stratified_cap(df, max_rows=600)
        assert len(capped) == 600
        # Returned unchanged (same object identity is fine, but verify content).
        pd.testing.assert_frame_equal(capped, df)

    def test_noop_when_ceiling_greater_than_rows(self):
        df = _make_dataset(n_safe=300, n_malicious=300)
        capped = apply_stratified_cap(df, max_rows=10000)
        pd.testing.assert_frame_equal(capped, df)

    def test_noop_when_cap_disabled_zero(self):
        df = _make_dataset(n_safe=1000, n_malicious=200)
        capped = apply_stratified_cap(df, max_rows=0)
        pd.testing.assert_frame_equal(capped, df)

    def test_noop_when_cap_negative(self):
        df = _make_dataset(n_safe=1000, n_malicious=200)
        capped = apply_stratified_cap(df, max_rows=-1)
        pd.testing.assert_frame_equal(capped, df)


# ---------------------------------------------------------------------------
# resolve_max_train_rows: env override
# ---------------------------------------------------------------------------

class TestResolveMaxTrainRows:
    """NA0S_MAX_TRAIN_ROWS override semantics."""

    def test_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_MAX_TRAIN_ROWS_ENV, None)
            assert resolve_max_train_rows() == _DEFAULT_MAX_TRAIN_ROWS

    def test_default_is_positive_and_reasonable(self):
        # Threshold sanity: the default must be a positive ceiling that is
        # well below the full ~2.2M corpus and above the rebalanced minimum.
        assert 0 < _DEFAULT_MAX_TRAIN_ROWS < 2_000_000

    def test_env_override_custom_value(self):
        with patch.dict(os.environ, {_MAX_TRAIN_ROWS_ENV: "50000"}):
            assert resolve_max_train_rows() == 50000

    def test_env_override_zero_disables(self):
        with patch.dict(os.environ, {_MAX_TRAIN_ROWS_ENV: "0"}):
            assert resolve_max_train_rows() == 0

    def test_env_override_invalid_falls_back_to_default(self):
        with patch.dict(os.environ, {_MAX_TRAIN_ROWS_ENV: "not-a-number"}):
            assert resolve_max_train_rows() == _DEFAULT_MAX_TRAIN_ROWS

    def test_env_override_empty_falls_back_to_default(self):
        with patch.dict(os.environ, {_MAX_TRAIN_ROWS_ENV: ""}):
            assert resolve_max_train_rows() == _DEFAULT_MAX_TRAIN_ROWS

    def test_resolved_value_feeds_cap_end_to_end(self):
        # Wiring check: resolved env value is the ceiling apply_stratified_cap
        # actually uses.
        df = _make_dataset(n_safe=1000, n_malicious=200)
        with patch.dict(os.environ, {_MAX_TRAIN_ROWS_ENV: "300"}):
            ceiling = resolve_max_train_rows()
            capped = apply_stratified_cap(df, ceiling)
        assert len(capped) == 300
        assert set(capped["label"].unique()) == {0, 1}
