"""Detection (regression) tests for the GTG-1002 synthesized scenarios.

These replay each GTG-1002 multi-turn scenario through the REAL scan entrypoint
(:func:`na0s.predict.scan`) with a ``session_id`` so the Layer-16 multi-turn /
session-aggregation path is engaged (the default stateless path has no
multi-turn coverage at all). Each test asserts the session is ultimately
BLOCKED.

DETECTION IS NOW LIVE (Build-3)
-------------------------------
These were ``xfail`` while the GTG-1002 benign-decomposition pattern was a known
gap. The gap is now closed by three wired detectors:

* ``conversation.detectors.goal_decomposition`` — order-aware kill-chain
  coverage: alerts only on a TERMINAL phase (EXPLOIT / CRED_HARVEST / EXFIL)
  appearing AFTER an earlier recon phase, with a persona multiplier (keyed off
  the killchain PERSONA phase) lowering the distinct-phase floor to 3 so the two
  3-phase attacks (unauthorized_tool_invocation, tool_param_exfil) also fire.
* ``detectors.tool_abuse`` — the in-prose terminal pivot (privileged-target
  invocation, scope-defiance, exfil-to-external-host).
* ``detectors.inter_model`` — fabricated cross-model authority.

All six attacks now BLOCK at their terminal turn; the six benign siblings stay
ALLOWED (see ``test_gtg1002_benign_siblings`` for the over-refusal guard). The
xfail marker is removed: these are STRICT regression tests now.

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


@pytest.mark.parametrize(
    "scenario",
    _GTG_ATTACKS,
    ids=[s.name for s in _GTG_ATTACKS],
)
def test_gtg1002_session_is_blocked(scenario: Scenario):
    """Replaying the full kill-chain under a session must BLOCK.

    The final pivot turn, in the context of the accumulated session, is flagged
    malicious (goal_decomposition's terminal-after-recon pivot + the in-prose
    tool_abuse / inter_model signals). STRICT since Build-3.
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
