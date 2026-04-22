"""Tests for scripts/data/multilingual_generator.py."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from multilingual_generator import (  # noqa: E402
    ALL_LANGUAGES,
    CATEGORY_REGISTRY,
    expand_templates,
    generate_all,
    main,
    write_jsonl,
)
from na0s.dataset.schema import DataLabel, Na0SSample  # noqa: E402


class TestLanguageCoverage:
    """All 10 target languages must have templates."""

    def test_all_10_languages_present(self):
        for lang in ALL_LANGUAGES:
            samples = expand_templates(lang)
            assert len(samples) > 0, f"No samples generated for language '{lang}'"

    def test_language_count_is_10(self):
        assert len(ALL_LANGUAGES) == 10


class TestCategoryCoverage:
    """Each language must cover all 5 attack categories."""

    @pytest.mark.parametrize("lang", ALL_LANGUAGES)
    def test_all_5_categories_per_language(self, lang: str):
        samples = expand_templates(lang)
        technique_ids = {s.technique_id for s in samples}
        expected = {"D1", "D2", "E1", "D3", "C1"}
        assert expected.issubset(technique_ids), (
            f"Language '{lang}' missing categories: {expected - technique_ids}"
        )


class TestSampleFields:
    """Output samples must have correct field values."""

    def test_language_field_matches(self):
        for lang in ["ar", "de", "fr"]:
            samples = expand_templates(lang)
            for s in samples[:10]:
                assert s.language == lang

    def test_label_is_injection(self):
        samples = expand_templates("zh")
        for s in samples[:20]:
            assert s.label == DataLabel.INJECTION

    def test_source_is_synthetic_multilingual(self):
        samples = expand_templates("ru")
        for s in samples[:20]:
            assert s.source == "synthetic_multilingual"

    def test_technique_ids_are_valid_format(self):
        """technique_id should match the pattern [A-Z]+\\d+(\\. \\d+)* ."""
        import re

        pattern = re.compile(r"^[A-Z]+\d+(\.\d+)*$")
        samples = expand_templates("ja")
        for s in samples:
            assert pattern.match(s.technique_id), f"Bad technique_id: {s.technique_id}"


class TestTemplateExpansion:
    """Template expansion must produce a meaningful number of samples."""

    def test_expansion_produces_nonzero_samples(self):
        for lang in ALL_LANGUAGES:
            samples = expand_templates(lang)
            assert len(samples) > 0

    def test_no_empty_text_fields(self):
        for lang in ALL_LANGUAGES:
            samples = expand_templates(lang)
            for s in samples:
                assert s.text and s.text.strip(), (
                    f"Empty text for lang={lang}, tid={s.technique_id}"
                )

    def test_reasonable_volume_per_language(self):
        """Each language should produce at least several hundred samples."""
        for lang in ALL_LANGUAGES:
            samples = expand_templates(lang)
            assert len(samples) >= 100, (
                f"Language '{lang}' only produced {len(samples)} samples"
            )


class TestDryRun:
    """--dry-run must not write any files."""

    def test_dry_run_no_files_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(["--languages", "ar", "--output-dir", tmpdir, "--dry-run"])
            # Dry run should NOT create any .jsonl files
            jsonl_files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            assert len(jsonl_files) == 0, f"Dry run wrote files: {jsonl_files}"


class TestJSONLOutput:
    """Written JSONL must be valid."""

    def test_valid_jsonl_output(self):
        samples = expand_templates("de")[:50]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            written = write_jsonl(samples, path)
            assert written == 50

            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    d = json.loads(line)
                    assert "text" in d
                    assert "label" in d
                    assert d["label"] == "injection"
                    assert "technique_id" in d
                    assert "language" in d

    def test_output_preserves_unicode(self):
        samples = expand_templates("ar")[:5]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ar.jsonl")
            write_jsonl(samples, path)
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    # Arabic text should contain Arabic characters
                    assert any("\u0600" <= c <= "\u06FF" for c in d["text"]), (
                        f"Expected Arabic characters in: {d['text'][:80]}"
                    )


class TestGenerateAll:
    """generate_all() integration check."""

    def test_generate_all_returns_all_languages(self):
        result = generate_all()
        assert set(result.keys()) == set(ALL_LANGUAGES)

    def test_generate_subset(self):
        result = generate_all(["ar", "fr"])
        assert set(result.keys()) == {"ar", "fr"}
