"""GAP-03: decision-threshold resolution — loud fallback, strict mode, FPR key.

The threshold was silently falling back to 0.55 when optimal_threshold.json is
absent (DEBUG-only log), hiding the calibration gap.  Now: WARNING on fallback,
optional hard-fail strict mode for CI, and the FPR-anchored key is preferred.
"""

import json
import os
import sys

import pytest

_WT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _WT_SRC not in sys.path:
    sys.path.insert(0, _WT_SRC)
# NOTE: this module previously purged already-imported ``na0s.*`` from
# sys.modules at collection time to force a reload from _WT_SRC.  That nuked the
# shared module cache mid-session and corrupted singletons / module-level
# constants (e.g. the resolved decision threshold) for every later test —
# producing broad, order-dependent failures across the suite.  Rely on
# PYTHONPATH / the installed package instead; never delete na0s modules here.

from na0s.fusion import voting  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cache():
    voting._reset_threshold_cache()
    yield
    voting._reset_threshold_cache()


def test_absent_artifact_warns_loudly_not_silent(caplog, monkeypatch):
    monkeypatch.delenv("DECISION_THRESHOLD", raising=False)
    monkeypatch.delenv("NA0S_REQUIRE_THRESHOLD_ARTIFACT", raising=False)
    monkeypatch.setattr(voting, "_THRESHOLD_JSON_PATH", "/nonexistent/optimal_threshold.json")
    with caplog.at_level("WARNING"):
        t = voting.get_decision_threshold()
    assert t == voting._FALLBACK_THRESHOLD == 0.55
    assert any("UNCALIBRATED fallback" in r.message for r in caplog.records)


def test_strict_mode_raises_when_artifact_absent(monkeypatch):
    monkeypatch.delenv("DECISION_THRESHOLD", raising=False)
    monkeypatch.setenv("NA0S_REQUIRE_THRESHOLD_ARTIFACT", "1")
    monkeypatch.setattr(voting, "_THRESHOLD_JSON_PATH", "/nonexistent/x.json")
    with pytest.raises(RuntimeError, match="optimal_threshold.json"):
        voting.get_decision_threshold()


def test_prefers_target_fpr_over_recall95(tmp_path, monkeypatch):
    p = tmp_path / "optimal_threshold.json"
    p.write_text(json.dumps({"recall95_threshold": 0.61, "target_fpr_threshold": 0.73}))
    monkeypatch.delenv("DECISION_THRESHOLD", raising=False)
    monkeypatch.delenv("NA0S_REQUIRE_THRESHOLD_ARTIFACT", raising=False)
    monkeypatch.setattr(voting, "_THRESHOLD_JSON_PATH", str(p))
    assert voting.get_decision_threshold() == pytest.approx(0.73)


def test_falls_back_to_recall95_when_no_fpr_key(tmp_path, monkeypatch):
    p = tmp_path / "optimal_threshold.json"
    p.write_text(json.dumps({"recall95_threshold": 0.61}))
    monkeypatch.delenv("DECISION_THRESHOLD", raising=False)
    monkeypatch.delenv("NA0S_REQUIRE_THRESHOLD_ARTIFACT", raising=False)
    monkeypatch.setattr(voting, "_THRESHOLD_JSON_PATH", str(p))
    assert voting.get_decision_threshold() == pytest.approx(0.61)


def test_threshold_json_path_points_at_repo_root_not_src():
    """GAP-03 regression: the refactor moved voting.py to src/na0s/fusion/ but
    the relative path stayed 2 levels up (-> src/data/processed), so the artifact
    was never found.  It must resolve to <repo>/data/processed."""
    p = os.path.normpath(voting._THRESHOLD_JSON_PATH)
    parts = p.split(os.sep)
    # .../data/processed/optimal_threshold.json with NO 'src' immediately before 'data'
    assert parts[-3:] == ["data", "processed", "optimal_threshold.json"]
    assert parts[-4] != "src", f"path wrongly resolves under src/: {p}"


def test_env_var_still_wins(tmp_path, monkeypatch):
    p = tmp_path / "optimal_threshold.json"
    p.write_text(json.dumps({"target_fpr_threshold": 0.73}))
    monkeypatch.setattr(voting, "_THRESHOLD_JSON_PATH", str(p))
    monkeypatch.setenv("DECISION_THRESHOLD", "0.42")
    assert voting.get_decision_threshold() == pytest.approx(0.42)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
