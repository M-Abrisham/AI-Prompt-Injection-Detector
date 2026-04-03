"""Tests for security fixes in layer16/state.py."""

from __future__ import annotations

import math

import pytest

from na0s.layer16.models import ConversationState
from na0s.layer16.state import (
    MAX_TURNS_PER_SESSION,
    MAX_TURN_TEXT_LENGTH,
    _turn_from_dict,
    add_turn,
    from_dict,
    update_cumulative_risk,
)


def _fresh_state() -> ConversationState:
    return ConversationState(session_id="test-session")


# ── CRITICAL-1: Turn cap enforcement ────────────────────────────


class TestTurnCap:
    def test_turns_capped_at_max(self):
        state = _fresh_state()
        for i in range(510):
            add_turn(state, f"turn {i}")
        assert len(state.turns) == MAX_TURNS_PER_SESSION

    def test_under_cap_not_trimmed(self):
        state = _fresh_state()
        for i in range(10):
            add_turn(state, f"turn {i}")
        assert len(state.turns) == 10


# ── CRITICAL-3: Text truncation ─────────────────────────────────


class TestTextTruncation:
    def test_long_text_truncated(self):
        state = _fresh_state()
        long_text = "A" * (MAX_TURN_TEXT_LENGTH + 1)
        turn = add_turn(state, long_text)
        assert len(turn.text) == MAX_TURN_TEXT_LENGTH

    def test_short_text_not_truncated(self):
        state = _fresh_state()
        turn = add_turn(state, "hello")
        assert turn.text == "hello"


# ── MEDIUM-1: NaN / Inf rejection ───────────────────────────────


class TestNanInfRejection:
    def test_add_turn_rejects_nan(self):
        state = _fresh_state()
        with pytest.raises(ValueError):
            add_turn(state, "hi", risk_score=float("nan"))

    def test_add_turn_rejects_inf(self):
        state = _fresh_state()
        with pytest.raises(ValueError):
            add_turn(state, "hi", risk_score=float("inf"))

    def test_add_turn_rejects_neg_inf(self):
        state = _fresh_state()
        with pytest.raises(ValueError):
            add_turn(state, "hi", risk_score=float("-inf"))

    def test_update_cumulative_risk_rejects_nan(self):
        state = _fresh_state()
        with pytest.raises(ValueError):
            update_cumulative_risk(state, float("nan"))

    def test_update_cumulative_risk_rejects_inf(self):
        state = _fresh_state()
        with pytest.raises(ValueError):
            update_cumulative_risk(state, float("inf"))

    def test_update_cumulative_risk_rejects_neg_inf(self):
        state = _fresh_state()
        with pytest.raises(ValueError):
            update_cumulative_risk(state, float("-inf"))


# ── HIGH-7: Deserialization type validation ──────────────────────


class TestDeserializationValidation:
    def test_turn_from_dict_rejects_non_number_risk(self):
        d = {
            "turn_id": "t1",
            "text": "hello",
            "role": "user",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "risk_score": "not-a-number",
            "label": "safe",
        }
        with pytest.raises(ValueError, match="risk_score must be a number"):
            _turn_from_dict(d)

    def test_turn_from_dict_rejects_non_string_text(self):
        d = {
            "turn_id": "t1",
            "text": 12345,
            "role": "user",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "risk_score": 0.5,
            "label": "safe",
        }
        with pytest.raises(ValueError, match="text must be a string"):
            _turn_from_dict(d)

    def test_turn_from_dict_rejects_non_string_label(self):
        d = {
            "turn_id": "t1",
            "text": "hello",
            "role": "user",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "risk_score": 0.5,
            "label": 999,
        }
        with pytest.raises(ValueError, match="label must be a string"):
            _turn_from_dict(d)

    def test_from_dict_rejects_non_number_cumulative_risk(self):
        d = {
            "session_id": "s1",
            "turns": [],
            "cumulative_risk": "bad",
            "peak_risk": 0.0,
            "cusum_score": 0.0,
            "active_alerts": [],
            "created_at": "2025-01-01T00:00:00+00:00",
            "last_activity": "2025-01-01T00:00:00+00:00",
        }
        with pytest.raises(ValueError, match="cumulative_risk must be a number"):
            from_dict(d)

    def test_from_dict_rejects_non_number_peak_risk(self):
        d = {
            "session_id": "s1",
            "turns": [],
            "cumulative_risk": 0.0,
            "peak_risk": "bad",
            "cusum_score": 0.0,
            "active_alerts": [],
            "created_at": "2025-01-01T00:00:00+00:00",
            "last_activity": "2025-01-01T00:00:00+00:00",
        }
        with pytest.raises(ValueError, match="peak_risk must be a number"):
            from_dict(d)

    def test_from_dict_rejects_non_number_cusum_score(self):
        d = {
            "session_id": "s1",
            "turns": [],
            "cumulative_risk": 0.0,
            "peak_risk": 0.0,
            "cusum_score": [],
            "active_alerts": [],
            "created_at": "2025-01-01T00:00:00+00:00",
            "last_activity": "2025-01-01T00:00:00+00:00",
        }
        with pytest.raises(ValueError, match="cusum_score must be a number"):
            from_dict(d)
