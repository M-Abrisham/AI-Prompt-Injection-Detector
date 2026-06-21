"""Admission-gate tests for the GTG-1002 synthesized DRAFT scenarios.

Runs :meth:`ScenarioAdmissionGate.run` against the real GTG-1002 draft file and
asserts the load-bearing legs PASS for every scenario:

* ``taxonomy``       — every attack's Category-T / Category-IM code is canonical;
* ``exact_decontam`` — no draft stable_id collides with training/live corpora;
* ``benign_twin``    — every attack's paired_benign_id resolves to an allowed twin.

To keep the test deterministic and offline, the gate is pointed at EMPTY
training + live directories (the GTG-1002 family is synthesized/paraphrased, so
it must not collide with an empty corpus). The drafts directory itself is the
real one — we select the GTG-1002 results by name.

A second test supplies a STUB embedding_fn (deterministic, derived from the
text — no network, no model download) and asserts the semantic-cosine
decontam leg actually executes and is reported. This covers the G4 wiring the
``scripts/f14_admission_gate.py`` CLI does not exercise (the CLI never passes an
embedding_fn, so the cosine leg is dead unless tested here).
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from na0s.eval.scenarios.admission_gate import (
    AdmissionReport,
    ScenarioAdmissionGate,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DRAFTS_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "_drafts"


@pytest.fixture
def empty_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """An empty (but existing) training dir + live dir for offline decontam."""
    train = tmp_path / "train_empty"
    live = tmp_path / "live_empty"
    train.mkdir()
    live.mkdir()
    return train, live


def _gtg_results(report: AdmissionReport):
    res = [r for r in report.results if r.name.startswith("gtg1002")]
    assert res, (
        "no gtg1002_* results in admission report; "
        f"got {[r.name for r in report.results]}"
    )
    return res


def _stub_embed(text: str) -> list[float]:
    """Deterministic, derived-from-text embedding (no network, no model).

    Hashes the text into 8 float lanes. Distinct texts yield distinct vectors,
    so the cosine leg produces a *real* (non-degenerate) similarity that varies
    across drafts — this exercises the wiring without faking a fixed cosine.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # 8 lanes from the first 32 bytes (4 bytes -> 1 float via unpack).
    lanes = struct.unpack("<8I", digest[:32])
    # Scale into a stable small-magnitude vector.
    return [float(v % 1000) / 1000.0 for v in lanes]


# ── Core legs PASS for every GTG-1002 scenario ───────────────────────


def test_gtg1002_drafts_admit_against_empty_corpus(
    empty_corpus: tuple[Path, Path],
):
    train, live = empty_corpus
    gate = ScenarioAdmissionGate(training_dirs=[train], live_dir=live)
    report = gate.run(_DRAFTS_DIR)

    # Decontam ran against a truly empty corpus (honest reporting).
    assert report.training_sample_count == 0, (
        "expected an empty training corpus for a deterministic test"
    )

    for res in _gtg_results(report):
        assert "taxonomy" in res.checks_passed, (
            f"{res.name}: taxonomy check did not pass — {res.reasons}"
        )
        assert "exact_decontam" in res.checks_passed, (
            f"{res.name}: exact_decontam did not pass — {res.reasons}"
        )
        assert "benign_twin" in res.checks_passed, (
            f"{res.name}: benign_twin did not pass — {res.reasons}"
        )
        assert res.status == "ADMIT", (
            f"{res.name}: expected ADMIT, got {res.status} — {res.reasons}"
        )


def test_gtg1002_taxonomy_leg_passes_for_attacks(
    empty_corpus: tuple[Path, Path],
):
    """Every ATTACK (blocked) result clears the taxonomy leg specifically."""
    train, live = empty_corpus
    gate = ScenarioAdmissionGate(training_dirs=[train], live_dir=live)
    report = gate.run(_DRAFTS_DIR)
    attacks = [
        r for r in _gtg_results(report)
        # benign twins use BEN + are exempt; attacks are the rest
        if not r.name.endswith("__benign")
    ]
    assert attacks, "no attack results found"
    for res in attacks:
        assert "taxonomy" in res.checks_passed
        assert "taxonomy" not in res.checks_failed


def test_gtg1002_benign_twins_resolve(empty_corpus: tuple[Path, Path]):
    """Each attack's benign_twin leg passes (paired_benign_id -> allowed twin)."""
    train, live = empty_corpus
    gate = ScenarioAdmissionGate(training_dirs=[train], live_dir=live)
    report = gate.run(_DRAFTS_DIR)
    attacks = [
        r for r in _gtg_results(report) if not r.name.endswith("__benign")
    ]
    assert attacks
    for res in attacks:
        assert "benign_twin" in res.checks_passed, (
            f"{res.name}: benign_twin leg failed — {res.reasons}"
        )


# ── G4 wiring: the semantic cosine leg actually runs ─────────────────


def test_gtg1002_semantic_decontam_leg_runs(empty_corpus: tuple[Path, Path]):
    """A stub embedding_fn makes the gate run + report the cosine leg.

    The CLI never supplies an embedding_fn, so the cosine path is otherwise
    dead. Here we assert (a) the embedding_cosine method is reported in
    decontam_methods, and (b) supplying it does not flip any GTG-1002 scenario
    to REJECT (distinct texts -> sub-threshold cosine against an empty corpus).
    """
    train, live = empty_corpus
    gate = ScenarioAdmissionGate(
        training_dirs=[train],
        live_dir=live,
        embedding_fn=_stub_embed,
    )
    report = gate.run(_DRAFTS_DIR)

    # (a) the gate reports BOTH the lexical proxy AND the semantic cosine leg —
    # it never silently skips decontam.
    assert any("minhash" in m.lower() for m in report.decontam_methods), (
        f"lexical leg not reported: {report.decontam_methods}"
    )
    assert any("embedding_cosine" in m for m in report.decontam_methods), (
        "semantic cosine leg did NOT run / was not reported — embedding_fn "
        f"wiring is broken. decontam_methods={report.decontam_methods}"
    )

    # (b) with an empty corpus the cosine leg has nothing to match against, so
    # every GTG-1002 scenario still passes near_dup_decontam and ADMITs.
    for res in _gtg_results(report):
        assert "near_dup_decontam" in res.checks_passed, (
            f"{res.name}: near_dup_decontam failed under the semantic leg — "
            f"{res.reasons}"
        )
        assert res.status == "ADMIT", (
            f"{res.name}: expected ADMIT with stub embedder, got {res.status}"
            f" — {res.reasons}"
        )


def test_stub_embedder_is_deterministic_and_offline():
    """Guard the test's own embedder: same text -> same vector, distinct texts
    -> distinct vectors (so the cosine leg is non-degenerate, not a constant)."""
    v1 = _stub_embed("authorized pentest recon")
    v1_again = _stub_embed("authorized pentest recon")
    v2 = _stub_embed("exfiltrate to attacker endpoint")
    assert v1 == v1_again, "embedder must be deterministic"
    assert v1 != v2, "distinct texts must yield distinct vectors"
    assert len(v1) == 8
