"""Tests for na0s.data_schema — label mappings, validation, edge cases."""

from __future__ import annotations

import pytest

from na0s.data_schema import (
    DataLabel,
    DataSplit,
    Na0SSample,
    validate_sample,
    validate_sample_strict,
)


# ── DataLabel enum ─────────────────────────────────────────────────

class TestDataLabel:
    def test_values(self):
        assert DataLabel.INJECTION == "injection"
        assert DataLabel.BENIGN == "benign"

    def test_is_string(self):
        assert isinstance(DataLabel.INJECTION, str)


# ── DataSplit enum ─────────────────────────────────────────────────

class TestDataSplit:
    def test_values(self):
        assert DataSplit.TRAIN == "train"
        assert DataSplit.VAL == "val"
        assert DataSplit.TEST == "test"


# ── from_legacy_csv_row ───────────────────────────────────────────

class TestFromLegacyCsvRow:
    """Test all documented label aliases (case-insensitive)."""

    @pytest.mark.parametrize("raw,expected", [
        ("injection", DataLabel.INJECTION),
        ("INJECTION", DataLabel.INJECTION),
        ("Injection", DataLabel.INJECTION),
        ("malicious", DataLabel.INJECTION),
        ("MALICIOUS", DataLabel.INJECTION),
        ("True", DataLabel.INJECTION),
        ("true", DataLabel.INJECTION),
        ("TRUE", DataLabel.INJECTION),
        ("1", DataLabel.INJECTION),
        ("benign", DataLabel.BENIGN),
        ("BENIGN", DataLabel.BENIGN),
        ("safe", DataLabel.BENIGN),
        ("Safe", DataLabel.BENIGN),
        ("False", DataLabel.BENIGN),
        ("false", DataLabel.BENIGN),
        ("0", DataLabel.BENIGN),
    ])
    def test_label_mapping(self, raw, expected):
        row = {"text": "hello", "label": raw}
        sample = Na0SSample.from_legacy_csv_row(row)
        assert sample.label == expected

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="Unrecognised label"):
            Na0SSample.from_legacy_csv_row({"text": "x", "label": "unknown"})

    def test_augmentation_type_preserved(self):
        row = {"text": "hi", "label": "1", "augmentation_type": "paraphrase"}
        sample = Na0SSample.from_legacy_csv_row(row)
        assert sample.augmentation_type == "paraphrase"

    def test_augmentation_type_empty_becomes_none(self):
        row = {"text": "hi", "label": "0", "augmentation_type": ""}
        sample = Na0SSample.from_legacy_csv_row(row)
        assert sample.augmentation_type is None

    def test_missing_text_gives_empty(self):
        row = {"label": "1"}
        sample = Na0SSample.from_legacy_csv_row(row)
        assert sample.text == ""


# ── Na0SSample defaults ───────────────────────────────────────────

class TestNa0SSampleDefaults:
    def test_default_language(self):
        s = Na0SSample(text="x", label=DataLabel.BENIGN)
        assert s.language == "en"

    def test_default_split(self):
        s = Na0SSample(text="x", label=DataLabel.BENIGN)
        assert s.split == DataSplit.TRAIN

    def test_default_is_duplicate(self):
        s = Na0SSample(text="x", label=DataLabel.BENIGN)
        assert s.is_duplicate is False

    def test_created_at_auto_set(self):
        s = Na0SSample(text="x", label=DataLabel.BENIGN)
        assert s.created_at is not None


# ── to_dict round-trip ────────────────────────────────────────────

class TestToDict:
    def test_round_trip_keys(self):
        s = Na0SSample(text="hello", label=DataLabel.INJECTION, technique_id="D1.3")
        d = s.to_dict()
        assert d["text"] == "hello"
        assert d["label"] == "injection"
        assert d["technique_id"] == "D1.3"
        assert d["language"] == "en"
        assert d["split"] == "train"
        assert d["is_duplicate"] is False


# ── validate_sample ───────────────────────────────────────────────

class TestValidateSample:
    def test_valid_sample_no_errors(self):
        s = Na0SSample(text="hello world", label=DataLabel.BENIGN)
        assert validate_sample(s) == []

    def test_empty_text(self):
        s = Na0SSample(text="", label=DataLabel.BENIGN)
        errs = validate_sample(s)
        assert any("empty" in e for e in errs)

    def test_whitespace_only_text(self):
        s = Na0SSample(text="   \t\n", label=DataLabel.BENIGN)
        errs = validate_sample(s)
        assert any("empty" in e for e in errs)

    def test_text_too_long(self):
        s = Na0SSample(text="x" * 60_000, label=DataLabel.BENIGN)
        errs = validate_sample(s)
        assert any("50 000" in e for e in errs)

    def test_quality_score_below_zero(self):
        s = Na0SSample(text="ok", label=DataLabel.BENIGN, quality_score=-0.1)
        errs = validate_sample(s)
        assert any("quality_score" in e for e in errs)

    def test_quality_score_above_one(self):
        s = Na0SSample(text="ok", label=DataLabel.BENIGN, quality_score=1.5)
        errs = validate_sample(s)
        assert any("quality_score" in e for e in errs)

    def test_quality_score_valid_bounds(self):
        for qs in [0.0, 0.5, 1.0]:
            s = Na0SSample(text="ok", label=DataLabel.BENIGN, quality_score=qs)
            assert validate_sample(s) == []

    def test_invalid_difficulty(self):
        s = Na0SSample(text="ok", label=DataLabel.BENIGN, difficulty="extreme")
        errs = validate_sample(s)
        assert any("difficulty" in e for e in errs)

    def test_valid_difficulties(self):
        for d in ("easy", "medium", "hard"):
            s = Na0SSample(text="ok", label=DataLabel.BENIGN, difficulty=d)
            assert validate_sample(s) == []

    def test_invalid_technique_id(self):
        s = Na0SSample(text="ok", label=DataLabel.BENIGN, technique_id="bad-id")
        errs = validate_sample(s)
        assert any("technique_id" in e for e in errs)

    def test_valid_technique_ids(self):
        for tid in ("D1", "D1.3", "I2.1", "M1.1", "A3.2.1"):
            s = Na0SSample(text="ok", label=DataLabel.BENIGN, technique_id=tid)
            assert validate_sample(s) == [], f"Failed for {tid}"

    def test_quality_score_none_ok(self):
        s = Na0SSample(text="ok", label=DataLabel.BENIGN, quality_score=None)
        assert validate_sample(s) == []


# ── validate_sample_strict ────────────────────────────────────────

class TestValidateSampleStrict:
    def test_missing_license(self):
        s = Na0SSample(text="ok", label=DataLabel.BENIGN, source="test")
        errs = validate_sample_strict(s)
        assert any("license" in e for e in errs)

    def test_missing_source(self):
        s = Na0SSample(text="ok", label=DataLabel.BENIGN, license="MIT")
        errs = validate_sample_strict(s)
        assert any("source" in e for e in errs)

    def test_strict_passes_when_all_present(self):
        s = Na0SSample(
            text="ok",
            label=DataLabel.BENIGN,
            source="test-src",
            license="MIT",
        )
        assert validate_sample_strict(s) == []

    def test_strict_inherits_base_checks(self):
        s = Na0SSample(text="", label=DataLabel.BENIGN, source="s", license="L")
        errs = validate_sample_strict(s)
        assert any("empty" in e for e in errs)
