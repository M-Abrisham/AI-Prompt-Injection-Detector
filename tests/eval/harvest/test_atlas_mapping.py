"""Tests for the committed MITRE ATLAS -> Na0S mapping file.

``data/threat_intel_snapshots/atlas_to_na0s_mapping.yaml`` is the human-reviewed
bridge that anchors the taxonomy-aware harvester to MITRE ATLAS. It is consumed
by BOTH ``na0s.threat_intel.atlas_sync`` and the harvest
:class:`~na0s.eval.harvest.taxonomy.TaxonomyValidator`. If a mapping target
drifts off the canonical taxonomy (or a non-ATLAS key sneaks in), the harvester
would silently widen — or quietly drop — codes. These tests assert the file:

- exists and parses as a flat ``{AML.Txxxx: <na0s_code>}`` mapping;
- has ONLY real ATLAS-shaped keys;
- has ONLY canonical Na0S-code targets (the never-invent contract);
- is loaded in full by ``TaxonomyValidator`` (no entry silently dropped),
  so every committed ATLAS id actually resolves + validates.

Offline; reads only the committed file, no network.

Each test is written to FAIL on a regression (e.g. a junk target, a typo'd
ATLAS id, or an entry the validator silently drops).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from na0s.eval.harvest import TaxonomyValidator

# Repo root is four parents up from this test file
# (tests/eval/harvest/test_atlas_mapping.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAPPING_PATH = (
    _REPO_ROOT / "data" / "threat_intel_snapshots" / "atlas_to_na0s_mapping.yaml"
)

# MITRE ATLAS technique id, optionally with a dotted sub-technique suffix.
# Mirrors taxonomy._ATLAS_ID_RE so the file's keys are checked against the same
# shape the validator recognizes.
_ATLAS_ID_RE = re.compile(r"^AML\.T\d{4}(?:\.\d{3})?$")


@pytest.fixture(scope="module")
def raw_mapping() -> dict:
    """Parse the committed mapping file (str keys/values)."""
    assert _MAPPING_PATH.is_file(), f"ATLAS mapping file missing: {_MAPPING_PATH}"
    raw = yaml.safe_load(_MAPPING_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "mapping must be a flat {atlas_id: na0s_code} dict"
    return {str(k): str(v) for k, v in raw.items()}


@pytest.fixture(scope="module")
def validator() -> TaxonomyValidator:
    """A validator over the DEFAULT taxonomy + the committed mapping file."""
    return TaxonomyValidator()


# ── file shape ─────────────────────────────────────────────────────────────


def test_mapping_file_exists_and_is_non_empty(raw_mapping: dict):
    assert raw_mapping, "committed ATLAS mapping must not be empty"


def test_every_key_is_a_real_atlas_id(raw_mapping: dict):
    """No non-ATLAS key may smuggle an alias for a Na0S code."""
    bad = [k for k in raw_mapping if not _ATLAS_ID_RE.match(k)]
    assert bad == [], f"non-ATLAS keys in mapping: {bad}"


def test_no_duplicate_atlas_ids(raw_mapping: dict):
    """YAML would silently keep the last duplicate; assert there are none.

    (raw_mapping is already de-duped by the dict; re-read the raw text and
    count keys to catch a duplicate the dict would have hidden.)
    """
    text = _MAPPING_PATH.read_text(encoding="utf-8")
    keyed_lines = re.findall(r"^\s*\"?(AML\.T\d{4}(?:\.\d{3})?)\"?\s*:", text, re.M)
    assert len(keyed_lines) == len(set(keyed_lines)), (
        f"duplicate ATLAS ids: "
        f"{sorted(k for k in keyed_lines if keyed_lines.count(k) > 1)}"
    )


# ── the never-invent contract: every target is canonical ───────────────────


def test_every_mapped_target_is_canonical(
    raw_mapping: dict, validator: TaxonomyValidator
):
    """Every right-hand value must validate against data/taxonomy.yaml."""
    non_canonical = {
        atlas_id: code
        for atlas_id, code in raw_mapping.items()
        if not validator.validate_code(code)
    }
    assert non_canonical == {}, (
        f"mapping targets not in taxonomy.yaml: {non_canonical}"
    )


def test_no_phantom_targets(raw_mapping: dict):
    """The known phantom (non-canonical) codes must never be a target.

    E1 is now canonical, but C2/M1 are phantom — they would be rejected at
    extraction, so they must not appear as mapping homes.
    """
    targets = set(raw_mapping.values())
    assert "C2" not in targets
    assert "M1" not in targets


# ── the validator loads the file in full (no silent drops) ─────────────────


def test_validator_loads_every_entry(
    raw_mapping: dict, validator: TaxonomyValidator
):
    """No committed entry is silently dropped at load time.

    The validator drops entries whose key isn't ATLAS-shaped or whose target
    isn't canonical. Since the file passes both checks above, the loaded
    mapping must contain EVERY committed entry, unchanged.
    """
    loaded = validator._atlas_mapping  # noqa: SLF001 - asserting load fidelity
    assert loaded == raw_mapping


def test_every_committed_atlas_id_resolves_and_validates(
    raw_mapping: dict, validator: TaxonomyValidator
):
    """Each committed ATLAS id resolves to its target AND validate_code is True."""
    for atlas_id, code in raw_mapping.items():
        assert validator.resolve_to_na0s(atlas_id) == code, atlas_id
        assert validator.validate_code(atlas_id) is True, atlas_id
        # Severity bridges through to the mapped Na0S code.
        assert validator.get_severity(atlas_id) == validator.get_severity(code), (
            atlas_id
        )


def test_known_prompt_injection_anchors_present(raw_mapping: dict):
    """Spot-check the load-bearing prompt-injection anchors are mapped.

    These are the ATLAS ids the harvester relies on for the core surface; if a
    refactor dropped them, discovery tagging via ATLAS would silently regress.
    """
    # LLM Prompt Injection parent + direct/indirect sub-techniques.
    assert raw_mapping.get("AML.T0051") == "CT"
    assert raw_mapping.get("AML.T0051.000") == "D1"
    assert raw_mapping.get("AML.T0051.001") == "I1"
    # LLM Jailbreak, RAG Poisoning, Extract System Prompt.
    assert raw_mapping.get("AML.T0054") == "D2"
    assert raw_mapping.get("AML.T0070") == "IG"
    assert raw_mapping.get("AML.T0056") == "E"
