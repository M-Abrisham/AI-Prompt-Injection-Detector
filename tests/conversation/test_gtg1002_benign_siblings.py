"""Over-refusal regression guard for the GTG-1002 benign siblings (Build-3).

The six GTG-1002 attack scenarios are paired with six *authorized-pentest*
benign twins (``*__benign``) that carry the SAME trusted persona AND the same
recon / fingerprint phases as the attacks. The ONLY separators are (a) a
TERMINAL-phase pivot (EXPLOIT / CRED_HARVEST / EXFIL) that the benign twins
never take and (b) explicit rules-of-engagement / scope language the benign
twins do carry.

This module is the FP-SAFETY counterpart to ``test_gtg1002_detection.py``
(which asserts all six attacks BLOCK). Here we replay each benign twin through
the real :func:`na0s.predict.scan` session path and assert the session ENDS
ALLOWED — ``is_malicious is False`` and ``multi_turn_recommendation != "block"``
and ``multi_turn_threat_level != "blocked"`` on the final turn. If a future
detector starts blocking a benign twin, this is the over-refusal regression that
catches it.

WHY "ENDS" (final-turn session verdict), not "no turn ever crosses the line"
----------------------------------------------------------------------------
The session VERDICT is the final accumulated state — that is what a deployed
SDK acts on, and the GTG hard constraint is that the authorized siblings stay
ALLOWED. One benign opening sentence (``tool_param_exfil__benign`` turn 0)
single-turn-scores just over the line in the degraded-embedding fallback
(no sentence-transformers installed -> TfidfCentroid). That is a pre-existing
base-classifier artifact, independent of the Build-3 T/IM/goal_decomposition
detectors (all three return 0 on it), and it does not flip the SESSION
recommendation, which stays ``continue_monitoring``. We assert on the session
verdict the task specifies and additionally assert that none of the Build-3
detectors contributed a ``tool_abuse:`` / ``inter_model:`` / goal_decomposition
hit to any benign turn — so a regression *in the new detectors* is still caught
turn-by-turn.

No network, no LLM: scan() runs the local pipeline; the singleton conversation
monitor is reset per scenario for isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from na0s.eval.scenarios import Scenario
from na0s.eval.scenarios.loader import load_scenarios_dir
from na0s.predict import _reset_conversation_monitor, scan

# tests/conversation/ -> repo root is 2 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRAFTS_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "_drafts"


def _is_benign(s: Scenario) -> bool:
    return s.expected_verdict == "allowed" and s.attack_category == "BEN"


def _load_gtg_benign() -> list[Scenario]:
    """Load the GTG-1002 BENIGN siblings (allowed) at import time for params."""
    if not _DRAFTS_DIR.is_dir():
        return []
    scenarios = load_scenarios_dir(_DRAFTS_DIR)
    return [
        s
        for s in scenarios
        if s.name.startswith("gtg1002") and _is_benign(s)
    ]


_GTG_BENIGN = _load_gtg_benign()


@pytest.fixture(autouse=True)
def _isolate_session():
    """Reset the singleton conversation monitor around each replay."""
    _reset_conversation_monitor()
    try:
        yield
    finally:
        _reset_conversation_monitor()


def _replay(scenario: Scenario, session_id: str):
    """Replay every turn under one session; return the list of per-turn results."""
    results = []
    for turn in scenario.turns:
        results.append(scan(turn.text, session_id=session_id))
    return results


def test_gtg1002_benign_siblings_were_loaded():
    """Guard: the benign-sibling params are non-empty and well-formed.

    A 0-param parametrize collects silently green, which would hide the very
    over-refusal guard this file exists to provide.
    """
    assert _GTG_BENIGN, (
        "no GTG-1002 benign siblings loaded — the over-refusal guard would be "
        f"silently empty (looked in {_DRAFTS_DIR})"
    )
    # There must be exactly six benign twins (one per attack).
    assert len(_GTG_BENIGN) == 6, (
        f"expected 6 GTG-1002 benign siblings, found {len(_GTG_BENIGN)}: "
        f"{[s.name for s in _GTG_BENIGN]}"
    )
    for s in _GTG_BENIGN:
        assert s.turns, f"{s.name}: no turns to replay"
        # Every benign-twin turn is labelled safe (no terminal pivot).
        for t in s.turns:
            assert t.expected_label == "safe", (
                f"{s.name}: a benign-twin turn is not labelled safe"
            )


@pytest.mark.parametrize(
    "scenario",
    _GTG_BENIGN,
    ids=[s.name for s in _GTG_BENIGN],
)
def test_gtg1002_benign_sibling_session_is_allowed(scenario: Scenario):
    """HARD CONSTRAINT: replaying an authorized benign twin must stay ALLOWED.

    Asserts the SESSION verdict (final-turn accumulated state):
      * not is_malicious,
      * multi_turn_recommendation != "block",
      * multi_turn_threat_level != "blocked",
      * escalation not detected.
    """
    session_id = f"gtg1002benign::{scenario.stable_id}"
    results = _replay(scenario, session_id)
    assert results, f"{scenario.name}: scan() returned nothing"
    final = results[-1]

    assert final.is_malicious is False, (
        f"{scenario.name}: benign sibling SESSION ended malicious "
        f"(risk={final.risk_score}, label={final.label})"
    )
    assert final.multi_turn_recommendation != "block", (
        f"{scenario.name}: benign sibling SESSION ended with a block "
        f"recommendation (mt_reco={final.multi_turn_recommendation})"
    )
    assert final.multi_turn_threat_level != "blocked", (
        f"{scenario.name}: benign sibling SESSION ended blocked "
        f"(mt_level={final.multi_turn_threat_level})"
    )
    assert not final.escalation_detected, (
        f"{scenario.name}: benign sibling SESSION flagged escalation"
    )


@pytest.mark.parametrize(
    "scenario",
    _GTG_BENIGN,
    ids=[s.name for s in _GTG_BENIGN],
)
def test_gtg1002_benign_sibling_no_new_detector_hit(scenario: Scenario):
    """No Build-3 detector (tool_abuse / inter_model / goal_decomposition) may
    fire on ANY benign-sibling turn.

    This is the turn-by-turn FP guard specific to the NEW detectors: even though
    one pre-existing base-classifier artifact can elevate a single benign
    opening sentence, the three Build-3 detectors must contribute nothing to any
    benign turn. A regression in their FP-safety is caught here regardless of the
    session verdict.
    """
    session_id = f"gtg1002benign_hits::{scenario.stable_id}"
    results = _replay(scenario, session_id)
    for turn, r in zip(scenario.turns, results):
        offending = [
            tag for tag in (r.technique_tags or [])
            if tag.startswith("tool_abuse:")
            or tag.startswith("inter_model:")
        ]
        assert not offending, (
            f"{scenario.name}: a Build-3 detector flagged a benign turn "
            f"({offending}): {turn.text[:80]!r}"
        )
        # goal_decomposition surfaces as a multi_turn alert, not a technique_tag.
        gd_alerts = [
            a for a in (r.multi_turn_alerts or [])
            if a.get("alert_type") == "goal_decomposition"
        ]
        assert not gd_alerts, (
            f"{scenario.name}: goal_decomposition alerted on a benign sibling "
            f"({[a.get('evidence') for a in gd_alerts]})"
        )
