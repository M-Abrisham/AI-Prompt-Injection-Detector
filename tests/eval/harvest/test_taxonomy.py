"""Tests for TaxonomyValidator.

Each test is written to FAIL if the validator's behavior regresses (e.g. if
validate_code started returning True for everything, the junk-code test would
fail; if severity lookup broke, the severity test would fail).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from na0s.eval.harvest import TaxonomyValidator


def test_known_category_code_validates_true():
    v = TaxonomyValidator()
    # D1 is a real top-level category in data/taxonomy.yaml.
    assert v.validate_code("D1") is True


def test_known_technique_code_validates_true():
    v = TaxonomyValidator()
    # E1.1 is a real technique under category E.
    assert v.validate_code("E1.1") is True
    # C1MT.3 exercises a multi-char category prefix with a dotted technique.
    assert v.validate_code("C1MT.3") is True


def test_junk_code_validates_false():
    v = TaxonomyValidator()
    assert v.validate_code("ZZ9.9") is False
    assert v.validate_code("") is False
    assert v.validate_code("D1.999") is False


def test_get_severity_returns_taxonomy_value():
    v = TaxonomyValidator()
    # E1.1 is severity "high" in the taxonomy; D1.1 is "critical".
    assert v.get_severity("E1.1") == "high"
    assert v.get_severity("D1.1") == "critical"
    # Category-level severity also resolves.
    assert v.get_severity("D1") == "critical"


def test_get_severity_unknown_code_returns_none():
    v = TaxonomyValidator()
    assert v.get_severity("ZZ9.9") is None


def test_custom_taxonomy_path(tmp_path: Path):
    """A custom taxonomy file is honored (proves the default isn't hardcoded)."""
    custom = tmp_path / "tax.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "categories": {
                    "X": {
                        "severity": "low",
                        "techniques": {"X1.1": {"name": "n", "severity": "medium"}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    v = TaxonomyValidator(taxonomy_path=custom)
    assert v.validate_code("X") is True
    assert v.validate_code("X1.1") is True
    assert v.get_severity("X1.1") == "medium"
    assert v.get_severity("X") == "low"
    # A real taxonomy code is NOT in this custom file.
    assert v.validate_code("D1") is False


def test_missing_taxonomy_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TaxonomyValidator(taxonomy_path=tmp_path / "does_not_exist.yaml")
