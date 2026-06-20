"""GAP-07: lock the public risk_score [0,1] (and NaN/inf) contract.

Internal scoring can transiently go negative (safe-content is subtracted with
no lower clamp) and numpy scalars / NaN can leak through.  ScanResult.__post_init__
now enforces the contract at the single output boundary; these tests pin it.
"""

import math
import os
import sys

import pytest

_WT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _WT_SRC not in sys.path:
    sys.path.insert(0, _WT_SRC)
# NOTE: this module previously purged already-imported ``na0s.*`` from
# sys.modules at collection time to force a reload from _WT_SRC.  That nuked the
# shared module cache mid-session and corrupted singletons / module-level
# constants for every later test -- producing broad, order-dependent failures.
# Rely on PYTHONPATH / the installed package instead; never delete na0s modules here.

from na0s.scan_result import ScanResult  # noqa: E402


# ---------------------------------------------------------------------------
# 1. ScanResult unit invariant — bad construction is clamped
# ---------------------------------------------------------------------------

class TestScanResultInvariant:
    @pytest.mark.parametrize("bad,expected", [
        (-0.5, 0.0), (-0.0408, 0.0), (1.5, 1.0), (2.0, 1.0),
        (float("nan"), 0.0), (float("inf"), 0.0), (float("-inf"), 0.0),
        (0.42, 0.42), (0.0, 0.0), (1.0, 1.0),
    ])
    def test_risk_score_clamped(self, bad, expected):
        r = ScanResult(risk_score=bad)
        assert r.risk_score == pytest.approx(expected)
        assert 0.0 <= r.risk_score <= 1.0
        assert isinstance(r.risk_score, float)
        assert not math.isnan(r.risk_score)

    def test_cumulative_risk_clamped(self):
        assert ScanResult(cumulative_risk=-0.3).cumulative_risk == 0.0
        assert ScanResult(cumulative_risk=1.7).cumulative_risk == 1.0

    def test_numpy_scalar_normalized(self):
        np = pytest.importorskip("numpy")
        r = ScanResult(risk_score=np.float64(-0.04))
        assert r.risk_score == 0.0
        assert type(r.risk_score) is float  # not np.float64

    def test_to_json_serializable_after_clamp(self):
        import json
        r = ScanResult(risk_score=float("nan"))
        json.loads(r.to_json())  # must not raise (NaN would break strict JSON)


# ---------------------------------------------------------------------------
# 2-4. End-to-end scan() invariant (needs models)
# ---------------------------------------------------------------------------

try:
    from na0s import scan
    from na0s.models import get_model_path
    _SCAN_OK = os.path.isfile(get_model_path("model.pkl"))
except Exception:
    _SCAN_OK = False


@pytest.mark.skipif(not _SCAN_OK, reason="model files not available")
class TestScanEndToEndInvariant:
    @pytest.mark.parametrize("text", [
        "What is prompt injection?",
        "How does a phishing attack work?",
        "Explain how SQL injection is prevented",
        "Describe best practices for secure password storage.",
        "",
        "   ",
    ])
    def test_scan_risk_in_unit_interval(self, text):
        r = scan(text)
        assert 0.0 <= r.risk_score <= 1.0, f"out of range: {r.risk_score} for {text!r}"

    def test_multi_turn_fold_no_crash_on_benign(self):
        import uuid
        # The benign security-education input previously produced a negative
        # score that crashed add_turn() and silently dropped the turn.
        r = scan("What is prompt injection?", session_id="inv-" + uuid.uuid4().hex)
        assert r is not None
        assert 0.0 <= r.risk_score <= 1.0
        assert r.session_id != ""

    def test_safe_content_heavy_input_floors_at_zero(self):
        # Heavy safe-content phrasing maximizes the (up to 0.30) deduction.
        r = scan("This is a helpful, safe, educational explanation of how "
                 "cryptography keeps data secure for everyone. Thank you!")
        assert r.risk_score >= 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
