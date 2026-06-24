"""Tests for the ATLAS bridge + E1/BEN reconciliation in TaxonomyValidator.

These cover the taxonomy-aware-harvester additions WITHOUT touching the
existing test_taxonomy.py contract:

- ``resolve_to_na0s`` maps MITRE ATLAS ``AML.Txxxx`` IDs to Na0S codes when an
  optional mapping file is present, and returns None otherwise.
- ``validate_code`` accepts a *resolvable* ATLAS ID but its Na0S-code semantics
  are UNCHANGED (backward-compat for PR #437 ``_validated_technique`` and the
  extractor) — i.e. with no mapping file, an ATLAS ID is rejected.
- The E1 reconciliation: the bare mid-level ``E1`` now validates, fixing the
  live v0.1 prompt-exfiltration scenarios without migrating them.
- ``BEN`` is canonical (no longer the "deliberately rejected" sentinel).
- A malformed / junk-target mapping file cannot widen validation to junk and
  cannot break canonical Na0S-code validation.

Every test is written to FAIL on a regression (e.g. if a missing mapping file
suddenly made ATLAS IDs validate, or if E1 stopped validating).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from na0s.eval.harvest import TaxonomyValidator

# A real Na0S code that the default taxonomy.yaml always contains.
_REAL_NA0S_CODE = "D1.1"

# The live v0.1 scenario set. The hand-curated prompt-exfiltration packs file
# their attacks under the bare mid-level code ``attack_category: E1`` (not a
# leaf technique), which is exactly what the E1 reconciliation key restores.
_V01_SCENARIOS_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "eval" / "scenarios" / "v0.1"
)


def _write_mapping(tmp_path: Path, mapping: dict) -> Path:
    p = tmp_path / "atlas_to_na0s_mapping.yaml"
    p.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return p


# ── E1 / BEN reconciliation ──────────────────────────────────────────────


def test_bare_e1_midlevel_code_now_validates():
    """The intermediate E1 key fixes the live v0.1 exfiltration scenarios."""
    v = TaxonomyValidator()
    assert v.validate_code("E1") is True
    # Its leaf techniques still validate (E1.x family preserved).
    assert v.validate_code("E1.1") is True
    assert v.validate_code("E1.6") is True
    # Severity resolves for the new mid-level key.
    assert v.get_severity("E1") == "high"


def test_ben_is_canonical():
    """BEN is a real category now (was previously rejected as a sentinel)."""
    v = TaxonomyValidator()
    assert v.validate_code("BEN") is True
    assert v.validate_code("BEN.1") is True
    assert v.get_severity("BEN") == "low"


def test_phantom_codes_still_rejected():
    """C2/M1 are phantom and must NOT have been added by the reconciliation."""
    v = TaxonomyValidator()
    assert v.validate_code("C2") is False
    assert v.validate_code("M1") is False


def test_e1_does_not_disturb_existing_e_category():
    """Adding the E1 key leaves category E and its E2.x leaves intact."""
    v = TaxonomyValidator()
    assert v.validate_code("E") is True
    assert v.validate_code("E2.1") is True


# ── reconciliation: the LIVE v0.1 E1 scenarios all validate ──────────────


def test_live_v01_e1_scenarios_all_validate():
    """Every live v0.1 scenario tagged ``attack_category: E1`` validates.

    Before the reconciliation the bare mid-level ``E1`` was NOT in
    taxonomy.yaml (only the ``E1.x`` leaves were), so the hand-curated
    prompt-exfiltration scenarios failed their own taxonomy gate. This test
    loads the actual live scenario YAML via the real ScenarioLoader and asserts
    each E1-tagged scenario now passes ``validate_code`` — fixing them without
    migrating the files. It is anchored to the real corpus (currently 25 E1
    scenarios) and asserts a non-trivial floor so a corpus that lost its E1
    scenarios can't make this pass vacuously.
    """
    from na0s.eval.scenarios.loader import ScenarioLoader

    v = TaxonomyValidator()
    assert v.validate_code("E1") is True  # the keystone fix

    scenarios = ScenarioLoader(_V01_SCENARIOS_DIR).load_all()
    e1_scenarios = [s for s in scenarios if s.attack_category == "E1"]

    # The live exfiltration packs carry the bare E1 code; there must be a
    # meaningful number of them (real count is 25; floor guards against a
    # vacuous pass if the corpus is emptied/renamed).
    assert len(e1_scenarios) >= 25, (
        f"expected >=25 live E1 scenarios, found {len(e1_scenarios)}"
    )
    bad = [s.name for s in e1_scenarios if not v.validate_code(s.attack_category)]
    assert bad == [], f"E1 scenarios that fail taxonomy validation: {bad}"


def test_all_live_v01_attack_categories_validate_except_known_gaps():
    """Sanity: every live v0.1 attack_category validates, modulo a known gap.

    Guards the broader reconciliation: with E1 + BEN canonical, the only live
    code that still fails validation is ``E1_benign`` (a paired-benign sentinel
    distinct from the canonical ``BEN`` — out of scope for the E1 fix and left
    for a separate change). Pinning it here documents the one remaining gap so a
    future regression elsewhere surfaces loudly.
    """
    from na0s.eval.scenarios.loader import ScenarioLoader

    v = TaxonomyValidator()
    scenarios = ScenarioLoader(_V01_SCENARIOS_DIR).load_all()
    failing = sorted(
        {s.attack_category for s in scenarios if not v.validate_code(s.attack_category)}
    )
    # E1_benign is the single documented residual; everything else validates.
    assert failing == ["E1_benign"], (
        f"unexpected non-validating live attack_categories: {failing}"
    )


# ── ATLAS bridge: absent mapping (default, pre-bridge behavior) ──────────


def test_resolve_returns_none_without_mapping_file(tmp_path: Path):
    """With no mapping file, ATLAS IDs do not resolve."""
    v = TaxonomyValidator(atlas_mapping_path=tmp_path / "absent.yaml")
    assert v.resolve_to_na0s("AML.T0051") is None
    # Na0S codes never resolve via the ATLAS bridge (it only maps FROM ATLAS).
    assert v.resolve_to_na0s(_REAL_NA0S_CODE) is None


def test_validate_rejects_atlas_id_without_mapping(tmp_path: Path):
    """Backward-compat: with no mapping, an ATLAS ID is NOT a valid code."""
    v = TaxonomyValidator(atlas_mapping_path=tmp_path / "absent.yaml")
    assert v.validate_code("AML.T0051") is False
    # Canonical Na0S-code validation is unchanged.
    assert v.validate_code(_REAL_NA0S_CODE) is True
    assert v.validate_code("ZZ9.9") is False


# ── ATLAS bridge: present, well-formed mapping ───────────────────────────


def test_resolve_maps_atlas_to_na0s_when_present(tmp_path: Path):
    mapping = _write_mapping(tmp_path, {"AML.T0051": _REAL_NA0S_CODE})
    v = TaxonomyValidator(atlas_mapping_path=mapping)
    assert v.resolve_to_na0s("AML.T0051") == _REAL_NA0S_CODE


def test_validate_accepts_resolvable_atlas_id(tmp_path: Path):
    mapping = _write_mapping(tmp_path, {"AML.T0051": _REAL_NA0S_CODE})
    v = TaxonomyValidator(atlas_mapping_path=mapping)
    assert v.validate_code("AML.T0051") is True
    # Severity bridges through to the mapped Na0S code.
    assert v.get_severity("AML.T0051") == v.get_severity(_REAL_NA0S_CODE)


def test_atlas_subtechnique_suffix_resolves(tmp_path: Path):
    mapping = _write_mapping(tmp_path, {"AML.T0051.000": _REAL_NA0S_CODE})
    v = TaxonomyValidator(atlas_mapping_path=mapping)
    assert v.resolve_to_na0s("AML.T0051.000") == _REAL_NA0S_CODE
    assert v.validate_code("AML.T0051.000") is True


# ── ATLAS bridge: defensive — junk/malformed mapping cannot widen ────────


def test_mapping_to_unknown_na0s_code_is_dropped(tmp_path: Path):
    """A mapping target that is not a real Na0S code is ignored (no junk widening)."""
    mapping = _write_mapping(tmp_path, {"AML.T0051": "ZZ9.9"})
    v = TaxonomyValidator(atlas_mapping_path=mapping)
    assert v.resolve_to_na0s("AML.T0051") is None
    assert v.validate_code("AML.T0051") is False


def test_non_atlas_key_in_mapping_is_dropped(tmp_path: Path):
    """A non-ATLAS key cannot smuggle an alias for a Na0S code."""
    mapping = _write_mapping(tmp_path, {"NOT_ATLAS": _REAL_NA0S_CODE})
    v = TaxonomyValidator(atlas_mapping_path=mapping)
    # The bogus key never resolves and never validates.
    assert v.resolve_to_na0s("NOT_ATLAS") is None
    assert v.validate_code("NOT_ATLAS") is False


def test_malformed_mapping_file_does_not_break_validation(tmp_path: Path):
    """A malformed mapping file is ignored; Na0S validation still works."""
    bad = tmp_path / "atlas_to_na0s_mapping.yaml"
    bad.write_text("not: a: valid: mapping: [", encoding="utf-8")
    v = TaxonomyValidator(atlas_mapping_path=bad)
    assert v.validate_code(_REAL_NA0S_CODE) is True
    assert v.validate_code("AML.T0051") is False


def test_non_mapping_yaml_is_ignored(tmp_path: Path):
    """A YAML list (not a mapping) is ignored rather than crashing."""
    odd = tmp_path / "atlas_to_na0s_mapping.yaml"
    odd.write_text(yaml.safe_dump(["AML.T0051", _REAL_NA0S_CODE]), encoding="utf-8")
    v = TaxonomyValidator(atlas_mapping_path=odd)
    assert v.validate_code(_REAL_NA0S_CODE) is True
    assert v.resolve_to_na0s("AML.T0051") is None
