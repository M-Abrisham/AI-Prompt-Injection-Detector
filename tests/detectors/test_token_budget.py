"""Tests for the D8.1 token-budget / context-eviction detector."""

from __future__ import annotations

from na0s.detectors import token_budget as tb
from na0s.detectors.token_budget import (
    _DEFAULT_MODEL_WINDOW,
    _EVICTION_RATIO,
    _WINDOW_ENV_VAR,
    TokenBudgetResult,
    analyze_token_budget,
)


# ---------------------------------------------------------------------------
# Core detection semantics
# ---------------------------------------------------------------------------

def test_short_input_not_detected():
    """A trivially short input is clean: no detection, no boost."""
    result = analyze_token_budget("hi there")
    assert isinstance(result, TokenBudgetResult)
    assert result.detected is False
    assert result.boost == 0.0
    assert result.technique_ids == []
    assert result.token_count > 0
    assert result.model_window == _DEFAULT_MODEL_WINDOW


def test_input_exceeding_window_detected_d8_1():
    """Input engineered past the window triggers the D8.1 eviction signal."""
    # Far more tokens than the default 8192-token window.
    result = analyze_token_budget("word " * 40000)
    assert result.detected is True
    assert "D8.1" in result.technique_ids
    assert result.ratio >= _EVICTION_RATIO
    assert result.boost > 0.0
    assert result.reason  # human-readable explanation populated


def test_detect_boost_is_capped():
    """Even a wildly oversized input cannot exceed the boost ceiling."""
    result = analyze_token_budget("word " * 100000)
    assert result.detected is True
    assert result.boost == tb._MAX_DETECT_BOOST
    assert result.boost <= 0.25


def test_eviction_via_small_custom_window():
    """A small per-call window makes a modest input cross the eviction line."""
    result = analyze_token_budget("word " * 200, model_window=100)
    assert result.detected is True
    assert "D8.1" in result.technique_ids
    assert result.ratio > 1.0


# ---------------------------------------------------------------------------
# WATCH band (over guardrail budget, below eviction)
# ---------------------------------------------------------------------------

def test_over_guardrail_budget_is_watch_only():
    """Over guardrail budget but below eviction: WATCH, not a detection."""
    # ~5000 tokens vs default window 8192 / guardrail 4096:
    # over budget (4096) but below eviction (0.9 * 8192 = 7372).
    result = analyze_token_budget("word " * 5000)
    assert result.detected is False
    assert result.technique_ids == []
    assert 0.0 < result.boost < tb._MAX_DETECT_BOOST
    assert result.boost == tb._WATCH_BOOST
    assert "watch" in result.reason.lower()


# ---------------------------------------------------------------------------
# False-positive guard
# ---------------------------------------------------------------------------

def test_large_but_within_budget_does_not_over_trigger():
    """A large input that still fits the guardrail budget stays clean."""
    # ~3000 tokens < guardrail 4096 < window 8192.
    result = analyze_token_budget("alpha " * 3000)
    assert result.detected is False
    assert result.boost == 0.0
    assert result.technique_ids == []
    assert result.token_count > 1000  # genuinely large, not a tiny input


def test_explicit_guardrail_budget_respected():
    """An explicit guardrail_budget overrides the default fraction."""
    # 3000-ish tokens; raise the guardrail above it so it is clean.
    result = analyze_token_budget("alpha " * 3000, guardrail_budget=10000)
    assert result.detected is False
    assert result.boost == 0.0


# ---------------------------------------------------------------------------
# Environment-variable window override
# ---------------------------------------------------------------------------

def test_env_var_window_override_respected(monkeypatch):
    """NA0S_MODEL_CONTEXT_WINDOW overrides the default window."""
    monkeypatch.setenv(_WINDOW_ENV_VAR, "100")
    result = analyze_token_budget("word " * 200)
    assert result.model_window == 100
    assert result.detected is True


def test_explicit_window_arg_beats_env(monkeypatch):
    """An explicit model_window argument wins over the env var."""
    monkeypatch.setenv(_WINDOW_ENV_VAR, "100")
    result = analyze_token_budget("word " * 200, model_window=1_000_000)
    assert result.model_window == 1_000_000
    assert result.detected is False


def test_malformed_env_var_falls_back_to_default(monkeypatch):
    """A non-integer env value is ignored, not crashed on."""
    monkeypatch.setenv(_WINDOW_ENV_VAR, "not-a-number")
    result = analyze_token_budget("hi")
    assert result.model_window == _DEFAULT_MODEL_WINDOW


# ---------------------------------------------------------------------------
# tiktoken-absent fallback path
# ---------------------------------------------------------------------------

def test_heuristic_fallback_when_tiktoken_absent(monkeypatch):
    """With tiktoken unavailable, counting uses the heuristic and flags it."""
    # Force the no-tiktoken branch and reset the memoised encoder state.
    monkeypatch.setattr(tb, "_HAS_TIKTOKEN", False)
    monkeypatch.setattr(tb, "_ENCODER", None)
    monkeypatch.setattr(tb, "_ENCODER_TRIED", False)

    result = analyze_token_budget("hello world foo bar")
    assert result.estimated is True
    # chars//4 floored at word count -> at least the 4 words.
    assert result.token_count >= 4


def test_heuristic_fallback_still_detects(monkeypatch):
    """The fallback path still raises D8.1 on oversized input."""
    monkeypatch.setattr(tb, "_HAS_TIKTOKEN", False)
    monkeypatch.setattr(tb, "_ENCODER", None)
    monkeypatch.setattr(tb, "_ENCODER_TRIED", False)

    result = analyze_token_budget("word " * 40000)
    assert result.estimated is True
    assert result.detected is True
    assert "D8.1" in result.technique_ids


def test_encoder_load_failure_falls_back_to_heuristic(monkeypatch):
    """If tiktoken is present but the encoder fails to load, use heuristic."""
    class _Boom:
        @staticmethod
        def get_encoding(_name):
            raise RuntimeError("cold cache, offline")

    monkeypatch.setattr(tb, "_HAS_TIKTOKEN", True)
    monkeypatch.setattr(tb, "tiktoken", _Boom)
    monkeypatch.setattr(tb, "_ENCODER", None)
    monkeypatch.setattr(tb, "_ENCODER_TRIED", False)

    result = analyze_token_budget("hello world")
    assert result.estimated is True
    assert result.token_count >= 2


# ---------------------------------------------------------------------------
# Result-shape contract
# ---------------------------------------------------------------------------

def test_result_has_full_public_contract():
    """The dataclass exposes exactly the documented public fields."""
    result = analyze_token_budget("hi")
    for name in (
        "detected",
        "token_count",
        "estimated",
        "model_window",
        "ratio",
        "boost",
        "technique_ids",
        "reason",
    ):
        assert hasattr(result, name), f"missing field {name}"
