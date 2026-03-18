"""Tests for scripts/merge_taxonomy.py."""

from __future__ import annotations

import csv
import json
import os
import sys

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "scripts"))
sys.path.insert(0, os.path.join(_project_root, "src"))

from merge_taxonomy import (
    _assign_splits,
    _load_taxonomy_jsonl,
    _parse_split_ratio,
    _text_hash,
    merge,
    main,
)


# ── Helpers ────────────────────────────────────────────────────────

def _write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "augmentation_type"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _make_sample(text, label="injection", technique_id="D1.1"):
    return {"text": text, "label": label, "technique_id": technique_id}


# ── _text_hash ─────────────────────────────────────────────────────

class TestTextHash:
    def test_deterministic(self):
        assert _text_hash("hello") == _text_hash("hello")

    def test_case_insensitive(self):
        assert _text_hash("Hello World") == _text_hash("hello world")

    def test_strips_whitespace(self):
        assert _text_hash("  hello  ") == _text_hash("hello")

    def test_different_texts_differ(self):
        assert _text_hash("aaa") != _text_hash("bbb")


# ── _parse_split_ratio ────────────────────────────────────────────

class TestParseSplitRatio:
    def test_valid(self):
        assert _parse_split_ratio("0.8,0.1,0.1") == (0.8, 0.1, 0.1)

    def test_invalid_count(self):
        with pytest.raises(ValueError, match="3 values"):
            _parse_split_ratio("0.5,0.5")

    def test_invalid_sum(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            _parse_split_ratio("0.5,0.5,0.5")


# ── _load_taxonomy_jsonl ──────────────────────────────────────────

class TestLoadTaxonomyJsonl:
    def test_loads_samples(self, tmp_path):
        path = str(tmp_path / "samples.jsonl")
        _write_jsonl(path, [_make_sample("a"), _make_sample("b")])
        loaded = _load_taxonomy_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["text"] == "a"

    def test_skips_empty_lines(self, tmp_path):
        path = str(tmp_path / "samples.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps(_make_sample("x")) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_sample("y")) + "\n")
        loaded = _load_taxonomy_jsonl(path)
        assert len(loaded) == 2


# ── _assign_splits ────────────────────────────────────────────────

class TestAssignSplits:
    def test_correct_proportions(self):
        samples = [_make_sample("s{}".format(i)) for i in range(100)]
        splits = _assign_splits(samples, (0.8, 0.1, 0.1), seed=42)
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_split_field_set(self):
        samples = [_make_sample("s{}".format(i)) for i in range(10)]
        splits = _assign_splits(samples, (0.8, 0.1, 0.1), seed=42)
        for s in splits["train"]:
            assert s["split"] == "train"
        for s in splits["val"]:
            assert s["split"] == "val"
        for s in splits["test"]:
            assert s["split"] == "test"

    def test_deterministic(self):
        samples = [_make_sample("s{}".format(i)) for i in range(50)]
        s1 = _assign_splits([dict(s) for s in samples], (0.8, 0.1, 0.1), seed=42)
        s2 = _assign_splits([dict(s) for s in samples], (0.8, 0.1, 0.1), seed=42)
        assert [s["text"] for s in s1["train"]] == [s["text"] for s in s2["train"]]


# ── merge() integration ───────────────────────────────────────────

class TestMerge:
    def test_dedup_against_existing(self, tmp_path):
        taxonomy_path = str(tmp_path / "tax.jsonl")
        csv_path = str(tmp_path / "existing.csv")
        out_dir = str(tmp_path / "out")

        _write_jsonl(taxonomy_path, [
            _make_sample("duplicate text"),
            _make_sample("unique text"),
        ])
        _write_csv(csv_path, [
            {"text": "duplicate text", "label": "1", "augmentation_type": ""},
        ])

        counts = merge(taxonomy_path, csv_path, out_dir)
        total = sum(counts.values())
        assert total == 1  # only "unique text" remains

    def test_internal_dedup(self, tmp_path):
        taxonomy_path = str(tmp_path / "tax.jsonl")
        out_dir = str(tmp_path / "out")

        _write_jsonl(taxonomy_path, [
            _make_sample("same text"),
            _make_sample("same text"),
            _make_sample("different"),
        ])

        counts = merge(taxonomy_path, None, out_dir)
        total = sum(counts.values())
        assert total == 2

    def test_output_files_created(self, tmp_path):
        taxonomy_path = str(tmp_path / "tax.jsonl")
        out_dir = str(tmp_path / "out")
        _write_jsonl(taxonomy_path, [_make_sample("t{}".format(i)) for i in range(20)])

        merge(taxonomy_path, None, out_dir)
        assert os.path.exists(os.path.join(out_dir, "train.jsonl"))
        assert os.path.exists(os.path.join(out_dir, "val.jsonl"))
        assert os.path.exists(os.path.join(out_dir, "test.jsonl"))

    def test_no_existing_path(self, tmp_path):
        taxonomy_path = str(tmp_path / "tax.jsonl")
        out_dir = str(tmp_path / "out")
        _write_jsonl(taxonomy_path, [_make_sample("hello")])

        counts = merge(taxonomy_path, None, out_dir)
        assert sum(counts.values()) == 1


# ── CLI / main() ──────────────────────────────────────────────────

class TestMainCli:
    def test_full_run(self, tmp_path):
        taxonomy_path = str(tmp_path / "tax.jsonl")
        out_dir = str(tmp_path / "out")
        _write_jsonl(taxonomy_path, [_make_sample("s{}".format(i)) for i in range(10)])

        counts = main([
            "--taxonomy", taxonomy_path,
            "--output-dir", out_dir,
            "--split-ratio", "0.8,0.1,0.1",
        ])
        assert sum(counts.values()) == 10
