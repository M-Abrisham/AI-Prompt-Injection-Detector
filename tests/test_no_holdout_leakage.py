"""Eval-leakage guard: the holdout/benchmark sets must never enter training.

`scripts/optimize_threshold.py` and `scripts/threshold_sweep.py` score the model
against `data/holdout/` and `data/benchmark/`.  `scripts/process_data.py` once
merged those very directories into the training CSV (`data/processed/
combined_data.csv`), so the model — and the fitted decision threshold — were
trained on the exact rows used to measure recall, inflating every metric and
making any threshold "calibration" dishonest.  This pins the fix (prerequisite
for GAP-03/GAP-01).
"""

import csv
import glob
import json
import os
import sys

import pytest

_WT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_WT_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_WT_ROOT, "scripts"))

import process_data  # noqa: E402


def test_training_dirs_exclude_holdout_and_benchmark():
    assert process_data.HOLDOUT_DIR not in process_data.TRAINING_JSONL_DIRS
    assert process_data.BENCHMARK_DIR not in process_data.TRAINING_JSONL_DIRS
    assert set(process_data.TRAINING_JSONL_DIRS) == {
        process_data.AGGREGATED_DIR,
        process_data.HARVEST_DIR,
    }


def test_merge_iterates_training_constant_only():
    import inspect
    src = inspect.getsource(process_data.merge_datasets)
    assert "for jsonl_dir in TRAINING_JSONL_DIRS" in src
    assert "HOLDOUT_DIR, BENCHMARK_DIR]" not in src


def _ids(path):
    out = set()
    if not os.path.isfile(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        sid = row.get("text") or row.get("prompt") or row.get("stable_id")
        if sid:
            out.add(str(sid).strip())
    return out


def test_no_holdout_string_in_combined_csv():
    combined = process_data.OUTPUT_PATH
    eval_ids = set()
    for d in (process_data.HOLDOUT_DIR, process_data.BENCHMARK_DIR):
        if os.path.isdir(d):
            for p in glob.glob(os.path.join(d, "*.jsonl")):
                eval_ids |= _ids(p)
    if not os.path.isfile(combined) or not eval_ids:
        pytest.skip("combined_data.csv or eval artifacts not present locally")
    train_texts = set()
    with open(combined, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t = row.get("text") or row.get("prompt") or ""
            if t:
                train_texts.add(t.strip())
    leaked = eval_ids & train_texts
    assert not leaked, f"{len(leaked)} eval rows leaked into training: {list(leaked)[:3]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
