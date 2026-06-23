"""Tests for na0s.eval.registry.taxonomy_labels.

The load-bearing invariant is REGISTRY-CANONICAL: every taxonomy_codes entry on
every source in the LIVE data/datasets.yaml is a canonical taxonomy code. This
is the gate that prevents an invented code entering the training corpus through
the registry — it fails loudly if a Phase-2 tag is mistyped.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from na0s.eval.harvest.taxonomy import TaxonomyValidator
from na0s.eval.registry import (
    CODE_SEPARATOR,
    RegistryCodeError,
    iter_source_codes,
    load_registry,
    parse_codes,
    serialize_codes,
    validate_registry_codes,
)


# ── REGISTRY-CANONICAL: the live registry must be clean ──────────────


def test_live_registry_all_codes_canonical():
    """Every taxonomy_codes entry on every live source is canonical."""
    registry = load_registry()
    errors = validate_registry_codes(registry)
    assert errors == [], (
        "non-canonical taxonomy_codes in data/datasets.yaml: "
        + "; ".join(f"{e.source}:{e.code} ({e.reason})" for e in errors)
    )


def test_live_registry_every_source_tagged():
    """All 72 live sources carry a non-empty taxonomy_codes list.

    Item A tags every source; an absent/empty tag would be a coverage hole, so
    we assert full tagging here (validate_registry_codes ALLOWS empty — that is
    a separate, deliberately-permissive contract for future untagged sources).
    """
    registry = load_registry()
    pairs = list(iter_source_codes(registry))
    assert len(pairs) == len(registry["sources"])
    untagged = [name for name, codes in pairs if not codes]
    assert untagged == [], f"sources with no taxonomy_codes: {untagged}"


def test_live_registry_codes_resolve_via_validator():
    """Each live code independently validates through TaxonomyValidator."""
    registry = load_registry()
    validator = TaxonomyValidator()
    for source, codes in iter_source_codes(registry):
        for code in codes:
            assert validator.validate_code(code), (
                f"{source}: {code!r} is not canonical"
            )


def test_every_source_has_at_least_one_canonical_or_benign_code():
    """Item A contract: every source carries >=1 canonical taxonomy code.

    The task spec phrases this as ">=1 canonical taxonomy_code (or explicit
    benign)". BEN (the benign-control sentinel) IS canonical in data/taxonomy.yaml
    (taxonomy.py docstring lines 19-21), so a benign corpus tagged ["BEN"]
    satisfies this directly. This single assertion pins the exact spec contract
    (non-empty AND every member canonical) per source, independent of the two
    split invariants above.
    """
    registry = load_registry()
    validator = TaxonomyValidator()
    offenders: list[str] = []
    for source, codes in iter_source_codes(registry):
        canonical = [c for c in codes if validator.validate_code(c)]
        if not canonical:
            offenders.append(f"{source}={codes!r}")
    assert offenders == [], (
        "sources lacking >=1 canonical taxonomy_code (or explicit BEN): "
        + ", ".join(offenders)
    )
    # BEN must itself be canonical, or the "(or explicit benign)" clause is hollow.
    assert validator.validate_code("BEN")


# ── serialize / parse round-trip (the single CSV-cell format owner) ──


@pytest.mark.parametrize(
    "codes",
    [[], ["BEN"], ["D2", "D1"], ["E", "D1", "CT"]],
)
def test_serialize_parse_round_trip(codes):
    assert parse_codes(serialize_codes(codes)) == codes


def test_serialize_empty_is_empty_string():
    assert serialize_codes([]) == ""


def test_parse_empty_and_none():
    assert parse_codes("") == []
    assert parse_codes(None) == []


def test_parse_drops_blank_fragments():
    # A trailing separator must not yield a phantom empty code.
    assert parse_codes(f"D1{CODE_SEPARATOR}") == ["D1"]


def test_serialize_rejects_separator_in_code():
    bad = f"D1{CODE_SEPARATOR}evil"
    with pytest.raises(ValueError):
        serialize_codes([bad])


# ── synthetic registry: reject vs allow ─────────────────────────────


def _write_registry(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "datasets.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_synthetic_non_canonical_code_flagged(tmp_path: Path):
    """A source with an invented code yields exactly one error (ZZ9.9 pattern)."""
    reg_path = _write_registry(
        tmp_path,
        """
        version: "1.0"
        sources:
          good:
            type: huggingface
            taxonomy_codes: ["D1"]
            output: "good.csv"
          bad:
            type: huggingface
            taxonomy_codes: ["ZZ9.9"]
            output: "bad.csv"
        """,
    )
    errors = validate_registry_codes(load_registry(reg_path))
    assert len(errors) == 1
    assert errors[0] == RegistryCodeError(
        source="bad", code="ZZ9.9", reason="not a known taxonomy code"
    )


def test_synthetic_empty_or_absent_codes_allowed(tmp_path: Path):
    """A source with no taxonomy_codes (or an explicit empty list) is no error."""
    reg_path = _write_registry(
        tmp_path,
        """
        version: "1.0"
        sources:
          untagged:
            type: huggingface
            output: "untagged.csv"
          explicit_empty:
            type: huggingface
            taxonomy_codes: []
            output: "empty.csv"
        """,
    )
    assert validate_registry_codes(load_registry(reg_path)) == []


def test_synthetic_blank_code_flagged(tmp_path: Path):
    """A present-but-blank code is flagged as malformed (not silently passed)."""
    reg_path = _write_registry(
        tmp_path,
        """
        version: "1.0"
        sources:
          blanky:
            type: huggingface
            taxonomy_codes: [""]
            output: "blank.csv"
        """,
    )
    errors = validate_registry_codes(load_registry(reg_path))
    assert len(errors) == 1
    assert errors[0].source == "blanky"
    assert errors[0].reason == "empty or blank code"


def test_synthetic_scalar_code_coerced(tmp_path: Path):
    """A bare scalar taxonomy_codes is coerced to a 1-element list (not invented)."""
    reg_path = _write_registry(
        tmp_path,
        """
        version: "1.0"
        sources:
          scalarish:
            type: huggingface
            taxonomy_codes: BEN
            output: "s.csv"
        """,
    )
    pairs = dict(iter_source_codes(load_registry(reg_path)))
    assert pairs["scalarish"] == ["BEN"]
    assert validate_registry_codes(load_registry(reg_path)) == []


def test_load_registry_rejects_no_sources(tmp_path: Path):
    reg_path = _write_registry(tmp_path, "version: \"1.0\"\n")
    with pytest.raises(ValueError):
        load_registry(reg_path)
