"""Detection (regression) tests for the GTG-1002 synthesized scenarios.

These replay each GTG-1002 multi-turn scenario through the REAL scan entrypoint
(:func:`na0s.predict.scan`) with a ``session_id`` so the Layer-16 multi-turn /
session-aggregation path is engaged (the default stateless path has no
multi-turn coverage at all). Each test asserts the session is ultimately
BLOCKED.

WHY THESE ARE xfail
-------------------
Na0S CANNOT detect the GTG-1002 pattern today, and these tests TRACK that gap
rather than hide it:

* The load-bearing technique is a TRUSTED PERSONA ("authorized security firm,
  routine testing") plus decomposition into individually-benign subtasks. Every
  early turn scores ~0 in isolation, so the score-driven session aggregate never
  crosses the block threshold for a clean decomposition.
* The inter-model-propagation (Category IM) detector is a stub, and the
  agent/tool-abuse (Category T) signals are off the default scan path.

Building the detector is a SEPARATE follow-up (tracked in ROADMAP_V2), not this
item. Marking these ``xfail(strict=False)`` means: if a future detector starts
blocking them, the suite stays green (xpass) and we get a signal to flip the
marker off — the gap is never silently lost.

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

_XFAIL_REASON = (
    "GTG-1002 benign-decomposition is a known detection gap (IM stub + "
    "score-driven session aggregate); tracked in ROADMAP_V2"
)


def _is_benign(s: Scenario) -> bool:
    return s.expected_verdict == "allowed" and s.attack_category == "BEN"


def _load_gtg_attacks() -> list[Scenario]:
    """Load the GTG-1002 ATTACK scenarios (blocked) at import time for params."""
    if not _DRAFTS_DIR.is_dir():
        return []
    scenarios = load_scenarios_dir(_DRAFTS_DIR)
    return [
        s
        for s in scenarios
        if s.name.startswith("gtg1002") and not _is_benign(s)
    ]


_GTG_ATTACKS = _load_gtg_attacks()


def _replay(scenario: Scenario, session_id: str):
    """Replay every turn through scan() under one session; return last result."""
    last = None
    for turn in scenario.turns:
        last = scan(turn.text, session_id=session_id)
    return last


@pytest.fixture(autouse=True)
def _isolate_session():
    """Reset the singleton conversation monitor around each replay."""
    _reset_conversation_monitor()
    try:
        yield
    finally:
        _reset_conversation_monitor()


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
@pytest.mark.parametrize(
    "scenario",
    _GTG_ATTACKS,
    ids=[s.name for s in _GTG_ATTACKS],
)
def test_gtg1002_session_is_blocked(scenario: Scenario):
    """Replaying the full kill-chain under a session should BLOCK (it does not).

    Expected (eventual) behavior: the final pivot turn, in the context of the
    accumulated session, is flagged malicious. Today the clean benign
    decomposition keeps the aggregate below threshold, so this xfails.
    """
    session_id = f"gtg1002::{scenario.stable_id}"
    result = _replay(scenario, session_id)

    assert result is not None, f"{scenario.name}: scan() returned nothing"
    # A "block" is either a malicious single-turn verdict on the final pivot or
    # a session-level block recommendation folded into the result.
    blocked = bool(
        result.is_malicious
        or result.label == "malicious"
        or result.multi_turn_recommendation == "block"
        or result.multi_turn_threat_level == "blocked"
        or result.escalation_detected
    )
    assert blocked, (
        f"{scenario.name}: session NOT blocked "
        f"(risk={result.risk_score}, label={result.label}, "
        f"mt_reco={result.multi_turn_recommendation}, "
        f"mt_level={result.multi_turn_threat_level}, "
        f"escalation={result.escalation_detected})"
    )


def test_gtg1002_attack_scenarios_were_loaded():
    """Guard: the xfail params are non-empty (a 0-param parametrize is silent).

    Without this, an empty draft glob would make the parametrized test collect
    zero cases and report all-green, hiding the very gap we are tracking.
    """
    assert _GTG_ATTACKS, (
        "no GTG-1002 attack scenarios loaded — the xfail detection suite would "
        f"be silently empty (looked in {_DRAFTS_DIR})"
    )
    # Sanity: every loaded attack is multi-turn with a final malicious pivot.
    for s in _GTG_ATTACKS:
        assert s.turns, f"{s.name}: no turns to replay"
        assert s.turns[-1].expected_label == "malicious", (
            f"{s.name}: final turn is not the malicious pivot"
        )
