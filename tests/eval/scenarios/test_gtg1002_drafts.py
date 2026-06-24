"""Structural tests for the GTG-1002 synthesized DRAFT scenarios.

These tests assert the *shape* of the draft scenarios produced by the
incident-to-scenario pipeline for the Anthropic GTG-1002 "AI-orchestrated
cyber-espionage" write-up (Nov 2025). They guard the contract the F14
admission gate later relies on, WITHOUT re-running the gate:

* every scenario loads as MULTI_TURN with a valid expected_verdict;
* every ATTACK scenario carries a canonical Category-T / Category-IM taxonomy
  code (validate_code == True);
* every attack is paired with a benign sibling — resolved by stable_id, not by
  name (the G1 guard: ``attack.paired_benign_id`` must equal some benign
  scenario's ``stable_id``);
* the benign decomposition is honest: every early (pre-final) turn of an attack
  is labelled ``safe`` in isolation, which is exactly why Na0S cannot catch this
  today (each subtask is individually benign).

The draft file is loaded via :class:`ScenarioLoader` (the same path the gate and
the live loader use) so a schema break here surfaces as a load failure, not a
silently-skipped file. No network, no LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from na0s.eval.harvest.taxonomy import TaxonomyValidator
from na0s.eval.scenarios import Scenario, ScenarioType
from na0s.eval.scenarios.loader import load_scenarios_dir

# src/na0s/eval/scenarios/loader.py is not where the drafts live; resolve the
# drafts dir from THIS test file: tests/eval/scenarios/ -> repo root is 3 up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DRAFTS_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "_drafts"
_DRAFT_FILE = _DRAFTS_DIR / "2025-11-gtg-1002-synthesized.yaml"


def _is_benign(s: Scenario) -> bool:
    """A benign sibling is an ALLOWED, BEN-tagged over-refusal control."""
    return s.expected_verdict == "allowed" and s.attack_category == "BEN"


@pytest.fixture(scope="module")
def gtg_scenarios() -> list[Scenario]:
    assert _DRAFT_FILE.is_file(), f"missing GTG-1002 draft file: {_DRAFT_FILE}"
    # Load only this draft via a temp-isolated loader would re-copy the file;
    # instead load the drafts dir and select the GTG-1002 family by tag/name so
    # the test stays scoped to the artifact under test even if other drafts land
    # in the same directory later.
    all_drafts = load_scenarios_dir(_DRAFTS_DIR)
    gtg = [s for s in all_drafts if s.name.startswith("gtg1002")]
    assert gtg, "no gtg1002_* scenarios loaded from the drafts dir"
    return gtg


@pytest.fixture(scope="module")
def taxonomy() -> TaxonomyValidator:
    return TaxonomyValidator()


def _attacks(scenarios: list[Scenario]) -> list[Scenario]:
    return [s for s in scenarios if not _is_benign(s)]


def _benigns(scenarios: list[Scenario]) -> list[Scenario]:
    return [s for s in scenarios if _is_benign(s)]


# ── Sanity: the family loaded and splits into attack + benign halves ──


def test_loads_attacks_and_benign_siblings(gtg_scenarios: list[Scenario]):
    attacks = _attacks(gtg_scenarios)
    benigns = _benigns(gtg_scenarios)
    assert attacks, "expected at least one GTG-1002 attack scenario"
    assert benigns, "expected at least one GTG-1002 benign sibling"
    # One benign twin per attack (1:1 pairing in this draft set).
    assert len(benigns) == len(attacks), (
        f"expected 1 benign twin per attack; got {len(attacks)} attacks, "
        f"{len(benigns)} benigns"
    )


# ── Every scenario is MULTI_TURN with a valid verdict ────────────────


def test_every_scenario_is_multi_turn(gtg_scenarios: list[Scenario]):
    for s in gtg_scenarios:
        assert s.type == ScenarioType.MULTI_TURN, (
            f"{s.name}: expected MULTI_TURN, got {s.type}"
        )
        assert s.turns, f"{s.name}: MULTI_TURN scenario has no turns"


def test_every_expected_verdict_is_valid(gtg_scenarios: list[Scenario]):
    for s in gtg_scenarios:
        assert s.expected_verdict in ("blocked", "allowed"), (
            f"{s.name}: invalid expected_verdict {s.expected_verdict!r}"
        )
    # Attacks are blocked; benign twins are allowed (no accidental swap).
    for s in _attacks(gtg_scenarios):
        assert s.expected_verdict == "blocked", f"{s.name} should be blocked"
    for s in _benigns(gtg_scenarios):
        assert s.expected_verdict == "allowed", f"{s.name} should be allowed"


# ── Attack taxonomy: Category T / IM, canonical (validate_code True) ──


def test_attack_categories_are_canonical_T_or_IM(
    gtg_scenarios: list[Scenario], taxonomy: TaxonomyValidator
):
    for s in _attacks(gtg_scenarios):
        code = s.attack_category
        assert code.startswith("T") or code.startswith("IM"), (
            f"{s.name}: attack_category {code!r} is neither Category T "
            "(Agent/Tool Abuse) nor Category IM (Inter-Model Propagation)"
        )
        assert taxonomy.validate_code(code), (
            f"{s.name}: attack_category {code!r} is NOT a canonical taxonomy "
            "code (validate_code == False)"
        )


def test_both_target_categories_are_covered(gtg_scenarios: list[Scenario]):
    """The whole point of this draft set: fill the EMPTY T and IM categories."""
    codes = {s.attack_category for s in _attacks(gtg_scenarios)}
    assert any(c.startswith("T") for c in codes), (
        f"no Category-T attack present; got {sorted(codes)}"
    )
    assert any(c.startswith("IM") for c in codes), (
        f"no Category-IM attack present; got {sorted(codes)}"
    )


# ── G1 guard: benign sibling resolved by stable_id, NOT by name ──────


def test_every_attack_has_benign_sibling_by_stable_id(
    gtg_scenarios: list[Scenario],
):
    """For each attack, paired_benign_id must equal a benign sibling's stable_id.

    This is the G1 guard: pairing is resolved through the content-derived
    stable_id (which changes if the benign text is edited), never through a
    name string (which would silently survive a text edit and decouple the
    over-refusal control from its attack).
    """
    benign_by_sid = {s.stable_id: s for s in _benigns(gtg_scenarios)}
    assert benign_by_sid, "no benign siblings to resolve against"

    for attack in _attacks(gtg_scenarios):
        twin_id = attack.paired_benign_id
        assert twin_id, f"{attack.name}: no paired_benign_id set"
        # Resolve by ID, not by name.
        twin = benign_by_sid.get(twin_id)
        assert twin is not None, (
            f"{attack.name}: paired_benign_id {twin_id[:12]}... matches no "
            "benign sibling's stable_id"
        )
        # The resolved twin must be an allowed BEN control, and the id must be
        # the twin's *content* hash (re-derivable), proving it is not a name.
        assert _is_benign(twin), (
            f"{attack.name}: resolved twin {twin.name} is not an allowed BEN "
            "over-refusal control"
        )
        assert twin.stable_id == twin._compute_stable_id(), (
            f"{twin.name}: stable_id is not the canonical content hash"
        )
        assert twin_id != attack.name, "paired_benign_id must not be a name"


def test_benign_siblings_do_not_point_back(gtg_scenarios: list[Scenario]):
    """Benign twins are leaves — they don't declare a paired_benign_id."""
    for s in _benigns(gtg_scenarios):
        assert s.paired_benign_id is None, (
            f"{s.name}: benign sibling should not carry a paired_benign_id"
        )


# ── Honest decomposition: every EARLY turn is individually safe ──────


def test_early_attack_turns_are_individually_safe(
    gtg_scenarios: list[Scenario],
):
    """Each attack's pre-final turns are labelled ``safe`` in isolation.

    This encodes the load-bearing GTG-1002 technique (trusted persona +
    decomposition into individually-benign subtasks) and is exactly why the
    detection tests are xfail: no single early turn looks malicious.
    """
    for attack in _attacks(gtg_scenarios):
        assert len(attack.turns) >= 2, (
            f"{attack.name}: decomposition needs >= 2 turns"
        )
        early = attack.turns[:-1]
        for i, turn in enumerate(early):
            assert turn.expected_label == "safe", (
                f"{attack.name} turn[{i}]: expected early turn label 'safe', "
                f"got {turn.expected_label!r} — a benign-decomposition "
                "scenario must have individually-benign early turns"
            )
        # The pivot turn is the only malicious one.
        assert attack.turns[-1].expected_label == "malicious", (
            f"{attack.name}: final pivot turn should be labelled 'malicious'"
        )


def test_benign_siblings_are_all_safe_turns(gtg_scenarios: list[Scenario]):
    """A real authorized pentest: every turn of the benign twin is safe."""
    for s in _benigns(gtg_scenarios):
        for i, turn in enumerate(s.turns):
            assert turn.expected_label == "safe", (
                f"{s.name} turn[{i}]: benign sibling turn should be 'safe', "
                f"got {turn.expected_label!r}"
            )
