"""Tests for dynamic threshold loading (Layer 4, P0).

Verifies that the decision threshold is resolved in priority order:
  1. DECISION_THRESHOLD env var
  2. recall95_threshold from data/processed/optimal_threshold.json
  3. Hardcoded fallback (0.55)

Also verifies that predict.py and ensemble.py use the same dynamic value.
"""

import json
import os
import types

import pytest


# ---------------------------------------------------------------------------
# Helpers — reimport _voting after resetting cache + env
# ---------------------------------------------------------------------------

def _reimport_voting(monkeypatch, env_val=None, json_path_override=None,
                     json_content=None, json_missing=False):
    """Re-import ``na0s._voting`` with a clean threshold cache.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
    env_val : str or None
        Value for DECISION_THRESHOLD env var.  None => unset.
    json_path_override : str or None
        Override _THRESHOLD_JSON_PATH to point at a temp file.
    json_content : dict or None
        If given, write this dict as JSON to the temp path.
    json_missing : bool
        If True, ensure the JSON path does not exist.

    Returns the freshly-patched module.
    """
    import na0s._voting as voting

    # Reset the cache
    voting._reset_threshold_cache()

    # Env var
    if env_val is not None:
        monkeypatch.setenv("DECISION_THRESHOLD", env_val)
    else:
        monkeypatch.delenv("DECISION_THRESHOLD", raising=False)

    # JSON file
    if json_path_override is not None:
        monkeypatch.setattr(voting, "_THRESHOLD_JSON_PATH", json_path_override)

    if json_content is not None and json_path_override is not None:
        os.makedirs(os.path.dirname(json_path_override), exist_ok=True)
        with open(json_path_override, "w") as fh:
            json.dump(json_content, fh)

    if json_missing and json_path_override is not None:
        if os.path.exists(json_path_override):
            os.remove(json_path_override)

    return voting


# ---------------------------------------------------------------------------
# Test: Load from JSON file
# ---------------------------------------------------------------------------

class TestLoadFromJSON:
    """Threshold loads recall95_threshold from optimal_threshold.json."""

    def test_recall95_threshold_loaded(self, tmp_path, monkeypatch):
        json_file = str(tmp_path / "optimal_threshold.json")
        data = {
            "youden_threshold": 0.42,
            "recall95_threshold": 0.38,
        }
        voting = _reimport_voting(
            monkeypatch,
            json_path_override=json_file,
            json_content=data,
        )
        result = voting.get_decision_threshold()
        assert result == 0.38

    def test_youden_not_used_by_default(self, tmp_path, monkeypatch):
        """Ensure youden_threshold is NOT selected (we want recall95)."""
        json_file = str(tmp_path / "optimal_threshold.json")
        data = {
            "youden_threshold": 0.50,
            "recall95_threshold": 0.33,
        }
        voting = _reimport_voting(
            monkeypatch,
            json_path_override=json_file,
            json_content=data,
        )
        assert voting.get_decision_threshold() == 0.33

    def test_caches_after_first_call(self, tmp_path, monkeypatch):
        json_file = str(tmp_path / "optimal_threshold.json")
        data = {"recall95_threshold": 0.40}
        voting = _reimport_voting(
            monkeypatch,
            json_path_override=json_file,
            json_content=data,
        )
        first = voting.get_decision_threshold()
        # Overwrite file with different value
        with open(json_file, "w") as fh:
            json.dump({"recall95_threshold": 0.99}, fh)
        second = voting.get_decision_threshold()
        assert first == second == 0.40


# ---------------------------------------------------------------------------
# Test: Fallback when file missing
# ---------------------------------------------------------------------------

class TestFallback:
    """Falls back to 0.55 when JSON file doesn't exist."""

    def test_missing_file_returns_fallback(self, tmp_path, monkeypatch):
        nonexistent = str(tmp_path / "does_not_exist.json")
        voting = _reimport_voting(
            monkeypatch,
            json_path_override=nonexistent,
            json_missing=True,
        )
        assert voting.get_decision_threshold() == 0.55

    def test_corrupt_json_returns_fallback(self, tmp_path, monkeypatch):
        bad_file = str(tmp_path / "corrupt.json")
        with open(bad_file, "w") as fh:
            fh.write("not valid json {{{")
        voting = _reimport_voting(
            monkeypatch,
            json_path_override=bad_file,
        )
        assert voting.get_decision_threshold() == 0.55

    def test_missing_key_returns_fallback(self, tmp_path, monkeypatch):
        json_file = str(tmp_path / "partial.json")
        data = {"youden_threshold": 0.42}  # no recall95_threshold
        voting = _reimport_voting(
            monkeypatch,
            json_path_override=json_file,
            json_content=data,
        )
        assert voting.get_decision_threshold() == 0.55


# ---------------------------------------------------------------------------
# Test: Env var override
# ---------------------------------------------------------------------------

class TestEnvVarOverride:
    """DECISION_THRESHOLD env var overrides everything."""

    def test_env_var_overrides_json(self, tmp_path, monkeypatch):
        json_file = str(tmp_path / "optimal_threshold.json")
        data = {"recall95_threshold": 0.38}
        voting = _reimport_voting(
            monkeypatch,
            env_val="0.70",
            json_path_override=json_file,
            json_content=data,
        )
        assert voting.get_decision_threshold() == 0.70

    def test_env_var_overrides_fallback(self, tmp_path, monkeypatch):
        nonexistent = str(tmp_path / "nope.json")
        voting = _reimport_voting(
            monkeypatch,
            env_val="0.60",
            json_path_override=nonexistent,
            json_missing=True,
        )
        assert voting.get_decision_threshold() == 0.60

    def test_invalid_env_var_ignored(self, tmp_path, monkeypatch):
        nonexistent = str(tmp_path / "nope.json")
        voting = _reimport_voting(
            monkeypatch,
            env_val="not_a_float",
            json_path_override=nonexistent,
            json_missing=True,
        )
        # Should fall through to fallback
        assert voting.get_decision_threshold() == 0.55


# ---------------------------------------------------------------------------
# Test: Value propagation to predict.py and ensemble.py
# ---------------------------------------------------------------------------

class TestPropagation:
    """predict.py and ensemble.py consume the threshold from _voting.py."""

    def test_predict_imports_from_voting(self):
        """predict.py's DECISION_THRESHOLD comes from _voting.get_decision_threshold."""
        import na0s.predict as predict
        import na0s._voting as voting
        # Both should resolve to the same function
        assert predict.DECISION_THRESHOLD == voting.get_decision_threshold()

    def test_ensemble_imports_from_voting(self):
        """ensemble.py's _DECISION_THRESHOLD comes from _voting.get_decision_threshold."""
        import na0s.ensemble as ensemble
        import na0s._voting as voting
        assert ensemble._DECISION_THRESHOLD == voting.get_decision_threshold()

    def test_all_three_agree(self):
        """predict, _voting, and ensemble all report the same threshold."""
        import na0s.predict as predict
        import na0s._voting as voting
        import na0s.ensemble as ensemble
        threshold = voting.get_decision_threshold()
        assert predict.DECISION_THRESHOLD == threshold
        assert ensemble._DECISION_THRESHOLD == threshold

    def test_voting_weighted_decision_default_uses_dynamic(self, tmp_path, monkeypatch):
        """weighted_decision() default threshold param equals the dynamic value."""
        import na0s._voting as voting
        import inspect
        sig = inspect.signature(voting.weighted_decision)
        default = sig.parameters["threshold"].default
        assert default == voting.DECISION_THRESHOLD
