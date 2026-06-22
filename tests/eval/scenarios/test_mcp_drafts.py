"""Load + schema + admission tests for the synthesized MCP supply-chain DRAFTs.

These tests guard the DRAFT scenarios authored by
``scripts/synthesize_mcp_supply_chain_scenarios.py`` for the MCP tool
supply-chain attack classes the new ``na0s.mcp`` guard server defends against:

* tool poisoning  (hidden directive in a tool description) ......... IM5.3
* rug pull        (tool swapped after approval) ................... IM5.4
* typosquat       (look-alike tool name shadows a trusted one) .... T1.4
* cross-server    (one server's manifest hijacks another's tool) .. IM3.3

The same three contracts the F14 admission gate later relies on are asserted
WITHOUT first re-running the gate, then the gate itself is run end-to-end:

* every scenario loads as MULTI_TURN with a valid expected_verdict;
* every ATTACK carries a canonical taxonomy code (``validate_code == True``);
* every attack is paired with a benign sibling resolved by ``stable_id`` (the
  content hash), never by name;
* the decomposition is honest: every early (pre-final) turn of an attack is
  labelled ``safe`` in isolation — which is exactly why the DEFAULT stateless
  path cannot catch this and why the MCP guard server's manifest-inspecting
  ``check_tool_call`` is the dedicated seam for it.

The draft file is loaded via :class:`ScenarioLoader` (the same path the gate and
the live loader use) so a schema break surfaces as a load failure, not a
silently-skipped file. No network, no LLM.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from na0s.eval.harvest.taxonomy import TaxonomyValidator
from na0s.eval.scenarios import Scenario, ScenarioType
from na0s.eval.scenarios.admission_gate import (
    AdmissionReport,
    ScenarioAdmissionGate,
)
from na0s.eval.scenarios.loader import load_scenarios_dir

# tests/eval/scenarios/ -> repo root is 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DRAFTS_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "_drafts"
_DRAFT_FILE = _DRAFTS_DIR / "2026-06-mcp-supply-chain-synthesized.yaml"

# Canonical taxonomy codes the MCP supply-chain set must cover. Kept explicit so
# a silent code drift (e.g. a typo or a non-canonical phantom like C2/M1) trips
# the test rather than slipping through.
_EXPECTED_ATTACK_CODES = {"IM5.3", "IM5.4", "T1.4", "IM3.3"}


def _is_benign(s: Scenario) -> bool:
    """A benign sibling is an ALLOWED, BEN-tagged over-refusal control."""
    return s.expected_verdict == "allowed" and s.attack_category == "BEN"


def _mcp_family(scenarios: list[Scenario]) -> list[Scenario]:
    return [s for s in scenarios if s.name.startswith("mcp_")]


def _attacks(scenarios: list[Scenario]) -> list[Scenario]:
    return [s for s in scenarios if not _is_benign(s)]


def _benigns(scenarios: list[Scenario]) -> list[Scenario]:
    return [s for s in scenarios if _is_benign(s)]


@pytest.fixture(scope="module")
def mcp_scenarios() -> list[Scenario]:
    assert _DRAFT_FILE.is_file(), f"missing MCP draft file: {_DRAFT_FILE}"
    all_drafts = load_scenarios_dir(_DRAFTS_DIR)
    mcp = _mcp_family(all_drafts)
    assert mcp, "no mcp_* scenarios loaded from the drafts dir"
    return mcp


@pytest.fixture(scope="module")
def taxonomy() -> TaxonomyValidator:
    return TaxonomyValidator()


@pytest.fixture
def empty_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """An empty (but existing) training dir + live dir for offline decontam."""
    train = tmp_path / "train_empty"
    live = tmp_path / "live_empty"
    train.mkdir()
    live.mkdir()
    return train, live


def _stub_embed(text: str) -> list[float]:
    """Deterministic, derived-from-text embedding (no network, no model).

    Hashes the text into 8 float lanes. Distinct texts yield distinct vectors,
    so the cosine leg produces a *real* (non-degenerate) similarity that varies
    across drafts — exercising the wiring without faking a fixed cosine.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    lanes = struct.unpack("<8I", digest[:32])
    return [float(v % 1000) / 1000.0 for v in lanes]


def _mcp_results(report: AdmissionReport):
    res = [r for r in report.results if r.name.startswith("mcp_")]
    assert res, (
        "no mcp_* results in admission report; "
        f"got {[r.name for r in report.results]}"
    )
    return res


# ── Sanity: the family loaded and splits into attack + benign halves ──


def test_loads_attacks_and_benign_siblings(mcp_scenarios: list[Scenario]):
    attacks = _attacks(mcp_scenarios)
    benigns = _benigns(mcp_scenarios)
    assert attacks, "expected at least one MCP attack scenario"
    assert benigns, "expected at least one MCP benign sibling"
    # The task asks for 3-4 scenarios; this set authors 4, each with a twin.
    assert 3 <= len(attacks) <= 4, (
        f"expected 3-4 MCP attack scenarios, got {len(attacks)}"
    )
    assert len(benigns) == len(attacks), (
        f"expected 1 benign twin per attack; got {len(attacks)} attacks, "
        f"{len(benigns)} benigns"
    )


# ── Every scenario is MULTI_TURN with a valid verdict ────────────────


def test_every_scenario_is_multi_turn(mcp_scenarios: list[Scenario]):
    for s in mcp_scenarios:
        assert s.type == ScenarioType.MULTI_TURN, (
            f"{s.name}: expected MULTI_TURN, got {s.type}"
        )
        assert s.turns, f"{s.name}: MULTI_TURN scenario has no turns"


def test_every_expected_verdict_is_valid(mcp_scenarios: list[Scenario]):
    for s in mcp_scenarios:
        assert s.expected_verdict in ("blocked", "allowed"), (
            f"{s.name}: invalid expected_verdict {s.expected_verdict!r}"
        )
    for s in _attacks(mcp_scenarios):
        assert s.expected_verdict == "blocked", f"{s.name} should be blocked"
    for s in _benigns(mcp_scenarios):
        assert s.expected_verdict == "allowed", f"{s.name} should be allowed"


# ── Provenance: synthesized + folded-in origin/retrieved suffix ──────


def test_scenarios_are_synthesized_with_provenance(mcp_scenarios: list[Scenario]):
    """source is 'synthesized' and the provenance suffix is folded in.

    The extractor forces ``source='harvest_pipeline'`` then the author script
    re-stamps ``source='synthesized'`` (there is no synthesized enum in the
    builder). The origin URL + retrieval date live in the description suffix.
    """
    for s in mcp_scenarios:
        assert s.source == "synthesized", (
            f"{s.name}: expected source 'synthesized', got {s.source!r}"
        )
        assert "provenance:" in s.description, (
            f"{s.name}: provenance suffix not folded into description"
        )
        assert "retrieved=2026-06" in s.description, (
            f"{s.name}: missing retrieval date in provenance suffix"
        )


# ── Attack taxonomy: canonical codes, exactly the MCP supply-chain set ──


def test_attack_categories_are_canonical(
    mcp_scenarios: list[Scenario], taxonomy: TaxonomyValidator
):
    for s in _attacks(mcp_scenarios):
        code = s.attack_category
        assert taxonomy.validate_code(code), (
            f"{s.name}: attack_category {code!r} is NOT a canonical taxonomy "
            "code (validate_code == False)"
        )


def test_attack_codes_cover_the_mcp_supply_chain_set(
    mcp_scenarios: list[Scenario],
):
    """tool-poisoning, rug-pull, typosquat, cross-server are all represented."""
    codes = {s.attack_category for s in _attacks(mcp_scenarios)}
    assert codes == _EXPECTED_ATTACK_CODES, (
        f"MCP attack codes {sorted(codes)} != expected "
        f"{sorted(_EXPECTED_ATTACK_CODES)}"
    )


# ── G1 guard: benign sibling resolved by stable_id, NOT by name ──────


def test_every_attack_has_benign_sibling_by_stable_id(
    mcp_scenarios: list[Scenario],
):
    benign_by_sid = {s.stable_id: s for s in _benigns(mcp_scenarios)}
    assert benign_by_sid, "no benign siblings to resolve against"

    for attack in _attacks(mcp_scenarios):
        twin_id = attack.paired_benign_id
        assert twin_id, f"{attack.name}: no paired_benign_id set"
        twin = benign_by_sid.get(twin_id)
        assert twin is not None, (
            f"{attack.name}: paired_benign_id {twin_id[:12]}... matches no "
            "benign sibling's stable_id"
        )
        assert _is_benign(twin), (
            f"{attack.name}: resolved twin {twin.name} is not an allowed BEN "
            "over-refusal control"
        )
        # The id must be the twin's *content* hash (re-derivable), proving it is
        # not a name that would silently survive a text edit.
        assert twin.stable_id == twin._compute_stable_id(), (
            f"{twin.name}: stable_id is not the canonical content hash"
        )
        assert twin_id != attack.name, "paired_benign_id must not be a name"


def test_benign_siblings_do_not_point_back(mcp_scenarios: list[Scenario]):
    for s in _benigns(mcp_scenarios):
        assert s.paired_benign_id is None, (
            f"{s.name}: benign sibling should not carry a paired_benign_id"
        )


# ── Honest decomposition: every EARLY turn is individually safe ──────


def test_early_attack_turns_are_individually_safe(
    mcp_scenarios: list[Scenario],
):
    """Each attack's pre-final turns are labelled ``safe`` in isolation.

    The poisoned description / swapped definition / look-alike registration only
    becomes detectable at the manifest level — no single early conversational
    turn looks malicious, which is precisely the gap the MCP guard server fills.
    """
    for attack in _attacks(mcp_scenarios):
        assert len(attack.turns) >= 2, (
            f"{attack.name}: decomposition needs >= 2 turns"
        )
        for i, turn in enumerate(attack.turns[:-1]):
            assert turn.expected_label == "safe", (
                f"{attack.name} turn[{i}]: expected early turn label 'safe', "
                f"got {turn.expected_label!r} — a benign-decomposition scenario "
                "must have individually-benign early turns"
            )
        assert attack.turns[-1].expected_label == "malicious", (
            f"{attack.name}: final pivot turn should be labelled 'malicious'"
        )


def test_benign_siblings_are_all_safe_turns(mcp_scenarios: list[Scenario]):
    """A real authorized engagement: every turn of the benign twin is safe."""
    for s in _benigns(mcp_scenarios):
        for i, turn in enumerate(s.turns):
            assert turn.expected_label == "safe", (
                f"{s.name} turn[{i}]: benign sibling turn should be 'safe', "
                f"got {turn.expected_label!r}"
            )


def test_risk_scores_rise_monotonically(mcp_scenarios: list[Scenario]):
    """Risk rises (non-strictly) across each attack's decomposition."""
    for attack in _attacks(mcp_scenarios):
        scores = [t.risk_score for t in attack.turns]
        assert all(s is not None for s in scores), (
            f"{attack.name}: every turn must carry a risk_score"
        )
        assert scores == sorted(scores), (
            f"{attack.name}: risk_score must rise monotonically, got {scores}"
        )
        # The pivot must be the high-water mark.
        assert scores[-1] == max(scores), (
            f"{attack.name}: final pivot must be the peak risk, got {scores}"
        )


# ── Admission gate runs end-to-end (taxonomy + decontam + benign_twin) ──


def test_mcp_drafts_admit_against_empty_corpus(
    empty_corpus: tuple[Path, Path],
):
    """The load-bearing legs PASS for every MCP scenario against an empty corpus.

    Pointed at empty training + live dirs so the (synthesized/paraphrased) MCP
    set cannot collide with a real corpus — making the test deterministic and
    offline. Mirrors ``test_gtg1002_admission``.
    """
    train, live = empty_corpus
    gate = ScenarioAdmissionGate(training_dirs=[train], live_dir=live)
    report = gate.run(_DRAFTS_DIR)

    assert report.training_sample_count == 0, (
        "expected an empty training corpus for a deterministic test"
    )

    for res in _mcp_results(report):
        assert "schema" in res.checks_passed, f"{res.name}: schema — {res.reasons}"
        assert "taxonomy" in res.checks_passed, (
            f"{res.name}: taxonomy check did not pass — {res.reasons}"
        )
        assert "exact_decontam" in res.checks_passed, (
            f"{res.name}: exact_decontam did not pass — {res.reasons}"
        )
        assert "near_dup_decontam" in res.checks_passed, (
            f"{res.name}: near_dup_decontam did not pass — {res.reasons}"
        )
        assert "benign_twin" in res.checks_passed, (
            f"{res.name}: benign_twin did not pass — {res.reasons}"
        )
        assert res.status == "ADMIT", (
            f"{res.name}: expected ADMIT, got {res.status} — {res.reasons}"
        )


def test_mcp_semantic_decontam_leg_runs(empty_corpus: tuple[Path, Path]):
    """A stub embedding_fn makes the gate run + report the cosine leg.

    The CLI never supplies an embedding_fn, so the cosine path is otherwise
    dead. Assert (a) the embedding_cosine method is reported, and (b) supplying
    it does not flip any MCP scenario to REJECT (distinct texts -> sub-threshold
    cosine against an empty corpus).
    """
    train, live = empty_corpus
    gate = ScenarioAdmissionGate(
        training_dirs=[train],
        live_dir=live,
        embedding_fn=_stub_embed,
    )
    report = gate.run(_DRAFTS_DIR)

    assert any("minhash" in m.lower() for m in report.decontam_methods), (
        f"lexical leg not reported: {report.decontam_methods}"
    )
    assert any("embedding_cosine" in m for m in report.decontam_methods), (
        "semantic cosine leg did NOT run / was not reported — embedding_fn "
        f"wiring is broken. decontam_methods={report.decontam_methods}"
    )

    for res in _mcp_results(report):
        assert "near_dup_decontam" in res.checks_passed, (
            f"{res.name}: near_dup_decontam failed under the semantic leg — "
            f"{res.reasons}"
        )
        assert res.status == "ADMIT", (
            f"{res.name}: expected ADMIT with stub embedder, got {res.status}"
            f" — {res.reasons}"
        )
