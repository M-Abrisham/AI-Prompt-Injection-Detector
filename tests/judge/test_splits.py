"""Tests for na0s.judge.splits.

Covers the two guarantees the harness depends on:
  1. stratified_split preserves class AND category proportions, returns the
     same object types, is deterministic, and yields DISJOINT stable_id sets.
  2. the one-time-test guard: a stable, order-independent manifest hash, and a
     log that makes a second use of the same test slice fail (unless allowed).

No network, no LLM — pure bookkeeping logic, so nothing is mocked.
"""

from __future__ import annotations

from collections import Counter

import pytest

from na0s.dataset.schema import DataLabel, DataSplit, Na0SSample
from na0s.judge import splits


# ── fixtures / builders ──────────────────────────────────────────────────────


def _make_corpus(n_per_cell: int = 20) -> list[Na0SSample]:
    """A balanced corpus over (label, category) cells big enough to split.

    Three injection categories + a benign cell, each with n_per_cell samples,
    every text unique (so stable_ids are distinct).
    """
    samples: list[Na0SSample] = []
    cells = [
        (DataLabel.INJECTION, "D1"),
        (DataLabel.INJECTION, "E1.3"),
        (DataLabel.INJECTION, "C2"),
        (DataLabel.BENIGN, None),
    ]
    for label, cat in cells:
        for i in range(n_per_cell):
            tag = cat or "benign"
            s = Na0SSample(
                text=f"sample {tag} #{i} {'attack' if cat else 'hello'}",
                label=label,
            )
            # attack_category is not a native Na0SSample field; the extractor
            # falls back to functional_category, which IS native.
            s.functional_category = cat
            samples.append(s)
    return samples


def _label_int(s: Na0SSample) -> int:
    return 1 if s.label == DataLabel.INJECTION else 0


def _cell(s: Na0SSample) -> tuple[int, str]:
    return (_label_int(s), s.functional_category or "__none__")


# ── stratified_split ─────────────────────────────────────────────────────────


class TestStratifiedSplit:
    def test_returns_three_splits_covering_all_inputs(self):
        corpus = _make_corpus(20)
        res = splits.stratified_split(corpus, seed=0)
        assert set(res) == {"train", "dev", "test"}
        total = len(res["train"]) + len(res["dev"]) + len(res["test"])
        assert total == len(corpus)

    def test_returns_same_object_types(self):
        corpus = _make_corpus(20)
        res = splits.stratified_split(corpus, seed=0)
        for name in ("train", "dev", "test"):
            for s in res[name]:
                assert isinstance(s, Na0SSample)

    def test_disjoint_stable_id_sets(self):
        corpus = _make_corpus(20)
        res = splits.stratified_split(corpus, seed=0)
        tr = {s.stable_id for s in res["train"]}
        dv = {s.stable_id for s in res["dev"]}
        te = {s.stable_id for s in res["test"]}
        assert tr.isdisjoint(dv)
        assert tr.isdisjoint(te)
        assert dv.isdisjoint(te)
        # union covers every input id exactly once
        assert tr | dv | te == {s.stable_id for s in corpus}
        assert len(tr) + len(dv) + len(te) == len(corpus)

    def test_preserves_label_ratio_across_splits(self):
        # Imbalanced corpus: 60 injection, 20 benign -> prevalence 0.75.
        corpus = []
        for i in range(60):
            s = Na0SSample(text=f"inj {i} attack", label=DataLabel.INJECTION)
            s.functional_category = "D1"
            corpus.append(s)
        for i in range(20):
            s = Na0SSample(text=f"ben {i} hello", label=DataLabel.BENIGN)
            s.functional_category = None
            corpus.append(s)
        overall = sum(_label_int(s) for s in corpus) / len(corpus)  # 0.75

        res = splits.stratified_split(corpus, seed=0)
        for name in ("train", "dev", "test"):
            part = res[name]
            assert part, f"{name} split is empty"
            frac = sum(_label_int(s) for s in part) / len(part)
            # Tolerance: stratification keeps each split's prevalence close to
            # the 0.75 overall; small-N integer rounding allows modest drift.
            assert abs(frac - overall) <= 0.15, f"{name} prevalence {frac:.3f}"

    def test_preserves_category_presence_across_splits(self):
        corpus = _make_corpus(20)
        all_cats = {s.functional_category or "__none__" for s in corpus}
        res = splits.stratified_split(corpus, seed=0)
        for name in ("train", "dev", "test"):
            cats = {s.functional_category or "__none__" for s in res[name]}
            # Every category that exists in the corpus should appear in each
            # split, given each cell has 20 members (well above the splittable
            # floor of 3).
            assert cats == all_cats, f"{name} missing categories: {all_cats - cats}"

    def test_category_proportions_roughly_preserved(self):
        corpus = _make_corpus(20)
        overall = Counter(_cell(s) for s in corpus)
        n = len(corpus)
        res = splits.stratified_split(corpus, ratios=(0.7, 0.15, 0.15), seed=0)
        # test split should hold ~15% of EACH cell (+/- a couple for rounding).
        test_counts = Counter(_cell(s) for s in res["test"])
        for cell, cnt in overall.items():
            expected = cnt * 0.15
            assert abs(test_counts[cell] - expected) <= 2, (
                f"cell {cell}: test has {test_counts[cell]}, expected ~{expected}"
            )

    def test_deterministic_with_seed(self):
        corpus = _make_corpus(20)
        a = splits.stratified_split(corpus, seed=42)
        b = splits.stratified_split(corpus, seed=42)
        for name in ("train", "dev", "test"):
            assert [s.stable_id for s in a[name]] == [s.stable_id for s in b[name]]

    def test_different_seed_changes_partition(self):
        corpus = _make_corpus(20)
        a = splits.stratified_split(corpus, seed=0)
        b = splits.stratified_split(corpus, seed=1)
        # Membership of at least one split should differ between seeds.
        a_test = {s.stable_id for s in a["test"]}
        b_test = {s.stable_id for s in b["test"]}
        assert a_test != b_test

    def test_small_groups_round_robin_not_dropped(self):
        # Cells with only 1-2 members can't be SSS-split; round-robin must keep
        # them and still produce disjoint splits with no dropped samples.
        corpus = []
        # 2-member injection cell, 1-member benign cell, 1-member injection cell
        for i in range(2):
            s = Na0SSample(text=f"rare inj A {i} attack", label=DataLabel.INJECTION)
            s.functional_category = "RARE1"
            corpus.append(s)
        s = Na0SSample(text="lonely benign hello", label=DataLabel.BENIGN)
        s.functional_category = None
        corpus.append(s)
        s = Na0SSample(text="lonely inj attack", label=DataLabel.INJECTION)
        s.functional_category = "RARE2"
        corpus.append(s)

        res = splits.stratified_split(corpus, seed=0)
        total = len(res["train"]) + len(res["dev"]) + len(res["test"])
        assert total == len(corpus)  # nothing dropped
        ids = (
            {s.stable_id for s in res["train"]}
            | {s.stable_id for s in res["dev"]}
            | {s.stable_id for s in res["test"]}
        )
        assert ids == {s.stable_id for s in corpus}

    def test_empty_corpus_returns_empty_splits(self):
        res = splits.stratified_split([], seed=0)
        assert res == {"train": [], "dev": [], "test": []}

    def test_accepts_dict_samples_with_int_labels(self):
        # Duck-typed: dicts with int label + attack_category + stable_id.
        corpus = []
        for i in range(15):
            corpus.append({"label": 1, "attack_category": "D1", "stable_id": f"inj{i}"})
        for i in range(15):
            corpus.append({"label": 0, "attack_category": "BEN", "stable_id": f"ben{i}"})
        res = splits.stratified_split(corpus, seed=0)
        total = len(res["train"]) + len(res["dev"]) + len(res["test"])
        assert total == 30
        # disjoint
        seen = set()
        for name in ("train", "dev", "test"):
            for d in res[name]:
                assert d["stable_id"] not in seen
                seen.add(d["stable_id"])

    def test_invalid_ratios_raise(self):
        corpus = _make_corpus(5)
        with pytest.raises(ValueError):
            splits.stratified_split(corpus, ratios=(0.5, 0.5))  # only 2
        with pytest.raises(ValueError):
            splits.stratified_split(corpus, ratios=(0.0, 0.0, 0.0))  # sum 0


# ── DataSplit reconciliation helper ──────────────────────────────────────────


class TestDataSplitMapping:
    def test_dev_maps_to_val(self):
        assert splits.split_key_to_datasplit("dev") == DataSplit.VAL
        assert splits.split_key_to_datasplit("train") == DataSplit.TRAIN
        assert splits.split_key_to_datasplit("test") == DataSplit.TEST

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            splits.split_key_to_datasplit("validation")


# ── one-time-test guard ──────────────────────────────────────────────────────


class TestManifestHash:
    def test_hash_is_stable(self):
        corpus = _make_corpus(5)
        test = splits.stratified_split(corpus, seed=0)["test"]
        h1 = splits.test_manifest_hash(test)
        h2 = splits.test_manifest_hash(test)
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex

    def test_hash_is_order_independent(self):
        corpus = _make_corpus(5)
        test = splits.stratified_split(corpus, seed=0)["test"]
        forward = splits.test_manifest_hash(test)
        backward = splits.test_manifest_hash(list(reversed(test)))
        assert forward == backward

    def test_hash_changes_with_membership(self):
        corpus = _make_corpus(8)
        res = splits.stratified_split(corpus, seed=0)
        h_test = splits.test_manifest_hash(res["test"])
        h_dev = splits.test_manifest_hash(res["dev"])
        assert h_test != h_dev

    def test_hash_dedups_repeated_ids(self):
        corpus = _make_corpus(5)
        test = splits.stratified_split(corpus, seed=0)["test"]
        with_dupes = test + test  # same membership, duplicated
        assert splits.test_manifest_hash(with_dupes) == splits.test_manifest_hash(test)


class TestTestUseGuard:
    def test_check_passes_when_unused(self, tmp_path):
        log = tmp_path / "use.jsonl"
        # No log file yet -> nothing used -> no raise.
        splits.check_test_unused("deadbeef" * 8, log)

    def test_second_use_raises(self, tmp_path):
        log = tmp_path / "use.jsonl"
        manifest = "a" * 64
        # First use: check passes, then record.
        splits.check_test_unused(manifest, log)
        splits.record_test_use(manifest, log)
        # Second use of the SAME manifest must fail.
        with pytest.raises(RuntimeError):
            splits.check_test_unused(manifest, log)

    def test_allow_reuse_downgrades_to_noop(self, tmp_path):
        log = tmp_path / "use.jsonl"
        manifest = "b" * 64
        splits.record_test_use(manifest, log)
        # allow_reuse=True -> no raise even though it's logged.
        splits.check_test_unused(manifest, log, allow_reuse=True)

    def test_distinct_manifests_independent(self, tmp_path):
        log = tmp_path / "use.jsonl"
        m1, m2 = "c" * 64, "d" * 64
        splits.record_test_use(m1, log)
        # m2 was never used -> still passes.
        splits.check_test_unused(m2, log)
        # m1 is used -> fails.
        with pytest.raises(RuntimeError):
            splits.check_test_unused(m1, log)

    def test_record_creates_parent_dirs(self, tmp_path):
        log = tmp_path / "nested" / "deeper" / "use.jsonl"
        splits.record_test_use("e" * 64, log)
        assert log.exists()

    def test_directory_log_path_uses_default_filename(self, tmp_path):
        manifest = "f" * 64
        splits.record_test_use(manifest, tmp_path)  # pass a directory
        expected = tmp_path / splits._DEFAULT_LOG_NAME
        assert expected.exists()
        with pytest.raises(RuntimeError):
            splits.check_test_unused(manifest, tmp_path)

    def test_corrupt_log_line_does_not_mask_real_use(self, tmp_path):
        log = tmp_path / "use.jsonl"
        manifest = "0" * 64
        splits.record_test_use(manifest, log)
        # Append a garbage line.
        with log.open("a", encoding="utf-8") as fh:
            fh.write("{not valid json\n")
        # Guard still sees the real prior use despite the corrupt line.
        with pytest.raises(RuntimeError):
            splits.check_test_unused(manifest, log)


# ── integration: split -> manifest -> guard round-trip ───────────────────────


class TestEndToEnd:
    def test_split_then_guard_roundtrip(self, tmp_path):
        corpus = _make_corpus(20)
        res = splits.stratified_split(corpus, seed=0)
        manifest = splits.test_manifest_hash(res["test"])
        log = tmp_path / "test_use_log.jsonl"

        splits.check_test_unused(manifest, log)  # first time: fine
        splits.record_test_use(manifest, log)
        with pytest.raises(RuntimeError):
            splits.check_test_unused(manifest, log)  # one-time guard fires
