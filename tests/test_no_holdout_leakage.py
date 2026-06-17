"""Eval-leakage guard: the holdout/benchmark sets must never enter training.

`scripts/technique_analysis.py` scores the model against `data/holdout/` and
`data/benchmark/` as OUT-OF-SAMPLE recall.  `scripts/process_data.py` once
merged those very directories into the training CSV (`data/processed/
combined_data.csv`), so the model was trained on the exact strings used to
measure its recall — inflating every per-category number.  These tests pin the
fix and prevent regressions.
"""

import csv
import json
import os
import sys

import pytest

_WT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_WT_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_WT_ROOT, "scripts"))

import process_data  # noqa: E402


def test_training_dirs_exclude_holdout_and_benchmark():
    """The training source list must not contain the eval directories."""
    assert process_data.HOLDOUT_DIR not in process_data.TRAINING_JSONL_DIRS
    assert process_data.BENCHMARK_DIR not in process_data.TRAINING_JSONL_DIRS
    # Only the legitimate training sources are allowed.
    assert set(process_data.TRAINING_JSONL_DIRS) == {
        process_data.AGGREGATED_DIR,
        process_data.HARVEST_DIR,
    }


def test_merge_iterates_training_constant_only():
    """Regression guard: the merge loop must iterate the safe TRAINING_JSONL_DIRS
    constant, not a literal list that could re-include the eval dirs."""
    import inspect

    src = inspect.getsource(process_data.merge_datasets)
    assert "for jsonl_dir in TRAINING_JSONL_DIRS" in src, (
        "merge_datasets must iterate TRAINING_JSONL_DIRS (which excludes the eval dirs)"
    )
    # The old literal list that included the eval dirs must be gone.
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
        sid = row.get("stable_id") or row.get("id") or row.get("text") or row.get("prompt")
        if sid:
            out.add(str(sid))
    return out


def test_no_holdout_string_in_combined_csv():
    """If the artifacts exist locally, assert zero holdout rows leaked in."""
    combined = process_data.OUTPUT_PATH
    holdout_ids = set()
    for d in (process_data.HOLDOUT_DIR, process_data.BENCHMARK_DIR):
        if os.path.isdir(d):
            import glob
            for p in glob.glob(os.path.join(d, "*.jsonl")):
                holdout_ids |= _ids(p)
    if not os.path.isfile(combined) or not holdout_ids:
        pytest.skip("combined_data.csv or holdout artifacts not present locally")
    train_texts = set()
    with open(combined, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t = row.get("text") or row.get("prompt") or ""
            if t:
                train_texts.add(t.strip())
    leaked = holdout_ids & train_texts
    assert not leaked, f"{len(leaked)} holdout rows leaked into training: {list(leaked)[:3]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
