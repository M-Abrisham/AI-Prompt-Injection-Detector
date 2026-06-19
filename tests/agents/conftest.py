"""Shared fixtures for the agents test package.

Test isolation for the dogfood wiring
-------------------------------------
``gate_analyzer.diagnose_failures`` now routes untrusted canary-error text
through ``input_guard.scan_untrusted`` -> the real ``na0s.predict`` pipeline
(the dogfood). Without isolation, every gate-analyzer test that carries
non-empty ``errors`` would load the real ML models into the agents test
session — slow, a "no real deps in tests" violation (see CLAUDE.md), and it
pulls global model/cache state into a subsystem whose tests have no business
exercising the detector.

This autouse fixture stubs ``scan_untrusted`` with a benign passthrough by
default, so no agents test invokes the real detector. Tests that specifically
verify the dogfood flagging (``test_dogfood_*``) override it with their own
``patch(...)`` inside the test body, which nests over this stub.

NOTE: this is purely test hygiene. The C1/D1/D3/D4/E1 CI failures were NOT a
flake: they were deterministic process pollution from tests/ml/ injecting a
``MagicMock`` for ``sentence_transformers`` (guarded only on ``not in
sys.modules``), which shadowed the real package in CI and zeroed the embedding
classifier for every later detection test. Fixed in tests/ml/ (find_spec guard
+ tests/ml/conftest.py); see that commit.
"""

from unittest.mock import patch

import pytest

from na0s.agents.input_guard import GuardResult


@pytest.fixture(autouse=True)
def _stub_input_guard_scan():
    """Default-stub the dogfood scanner so agents tests never load the real ML pipeline."""

    def _passthrough(value, source="input"):
        text = "" if value is None else str(value)
        return GuardResult(
            text=text,
            flagged=False,
            risk_score=0.0,
            label="stub",
            technique_ids=[],
            safe_text=text,
        )

    with patch("na0s.agents.gate_analyzer.scan_untrusted", side_effect=_passthrough):
        yield
