"""Tests for Layer 16 ConversationState functions."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from na0s.layer16.models import Alert, ConversationState, ConversationTurn
from na0s.layer16.state import (
    add_turn,
    from_dict,
    get_combined_text,
    get_risk_trend,
    get_window,
    is_escalating,
    to_dict,
)


def _make_state(session_id: str = "test-session") -> ConversationState:
    return ConversationState(session_id=session_id)


class TestAddTurn:
    def test_adds_turn_and_updates_activity(self) -> None:
        state = _make_state()
        before = state.last_activity
        turn = add_turn(state, "hello", risk_score=0.1, label="safe")
        assert state.turn_count == 1
        assert turn.text == "hello"
        assert turn.risk_score == 0.1
        assert turn.label == "safe"
        assert state.last_activity >= before
        assert len(turn.turn_id) > 0

    def test_adds_multiple_turns(self) -> None:
        state = _make_state()
        add_turn(state, "first")
        add_turn(state, "second")
        add_turn(state, "third")
        assert state.turn_count == 3

    def test_flags_default_empty(self) -> None:
        state = _make_state()
        turn = add_turn(state, "text")
        assert turn.flags == []

    def test_flags_provided(self) -> None:
        state = _make_state()
        turn = add_turn(state, "text", flags=["D7", "C1"])
        assert turn.flags == ["D7", "C1"]

    def test_turn_id_is_unique(self) -> None:
        state = _make_state()
        t1 = add_turn(state, "a")
        t2 = add_turn(state, "b")
        assert t1.turn_id != t2.turn_id

    def test_timestamp_is_utc(self) -> None:
        state = _make_state()
        turn = add_turn(state, "utc check")
        assert turn.timestamp.tzinfo is not None


class TestGetWindow:
    def test_all_turns_when_n_is_none(self) -> None:
        state = _make_state()
        for i in range(5):
            add_turn(state, f"turn {i}")
        window = get_window(state)
        assert len(window) == 5

    def test_last_n_turns(self) -> None:
        state = _make_state()
        for i in range(5):
            add_turn(state, f"turn {i}")
        window = get_window(state, n=3)
        assert len(window) == 3
        assert window[0].text == "turn 2"

    def test_n_larger_than_count(self) -> None:
        state = _make_state()
        add_turn(state, "only")
        window = get_window(state, n=10)
        assert len(window) == 1

    def test_empty_state(self) -> None:
        state = _make_state()
        assert get_window(state) == []


class TestGetRiskTrend:
    def test_returns_scores_in_order(self) -> None:
        state = _make_state()
        add_turn(state, "a", risk_score=0.1)
        add_turn(state, "b", risk_score=0.5)
        add_turn(state, "c", risk_score=0.9)
        assert get_risk_trend(state) == [0.1, 0.5, 0.9]

    def test_empty(self) -> None:
        state = _make_state()
        assert get_risk_trend(state) == []


class TestIsEscalating:
    def test_escalating_scores(self) -> None:
        state = _make_state()
        for score in [0.1, 0.3, 0.6, 0.9]:
            add_turn(state, "x", risk_score=score)
        assert is_escalating(state, threshold=0.15) is True

    def test_flat_scores(self) -> None:
        state = _make_state()
        for score in [0.5, 0.5, 0.5, 0.5]:
            add_turn(state, "x", risk_score=score)
        assert is_escalating(state) is False

    def test_decreasing_scores(self) -> None:
        state = _make_state()
        for score in [0.9, 0.7, 0.3, 0.1]:
            add_turn(state, "x", risk_score=score)
        assert is_escalating(state) is False

    def test_fewer_than_two_turns(self) -> None:
        state = _make_state()
        add_turn(state, "x", risk_score=0.9)
        assert is_escalating(state) is False

    def test_empty(self) -> None:
        state = _make_state()
        assert is_escalating(state) is False

    def test_custom_threshold(self) -> None:
        state = _make_state()
        # Gentle escalation: slope ~0.067
        for score in [0.1, 0.2, 0.3]:
            add_turn(state, "x", risk_score=score)
        assert is_escalating(state, threshold=0.05) is True
        assert is_escalating(state, threshold=0.15) is False


class TestGetCombinedText:
    def test_all_turns(self) -> None:
        state = _make_state()
        add_turn(state, "hello")
        add_turn(state, "world")
        assert get_combined_text(state) == "hello\nworld"

    def test_last_n(self) -> None:
        state = _make_state()
        add_turn(state, "a")
        add_turn(state, "b")
        add_turn(state, "c")
        assert get_combined_text(state, last_n=2) == "b\nc"

    def test_empty(self) -> None:
        state = _make_state()
        assert get_combined_text(state) == ""


class TestSerialization:
    def test_round_trip(self) -> None:
        state = _make_state("round-trip-id")
        add_turn(state, "first", risk_score=0.1, label="safe", flags=["D7"])
        add_turn(state, "second", risk_score=0.5, label="injection", flags=["C1"])
        state.active_alerts.append(
            Alert(
                alert_type="escalation",
                severity="high",
                confidence=0.9,
                description="Risk escalating",
                turn_range=(0, 1),
                evidence=["slope=0.4"],
            )
        )
        state.cumulative_risk = 0.6
        state.metadata["key"] = "value"

        d = to_dict(state)
        # Ensure it's JSON-serializable
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        restored = from_dict(d2)

        assert restored.session_id == "round-trip-id"
        assert restored.turn_count == 2
        assert restored.turns[0].text == "first"
        assert restored.turns[0].flags == ["D7"]
        assert restored.turns[1].risk_score == 0.5
        assert restored.cumulative_risk == 0.6
        assert len(restored.active_alerts) == 1
        assert restored.active_alerts[0].alert_type == "escalation"
        assert restored.active_alerts[0].turn_range == (0, 1)
        assert restored.metadata["key"] == "value"
        assert isinstance(restored.created_at, datetime)
        assert isinstance(restored.last_activity, datetime)

    def test_empty_state_round_trip(self) -> None:
        state = _make_state("empty")
        d = to_dict(state)
        restored = from_dict(d)
        assert restored.session_id == "empty"
        assert restored.turn_count == 0
        assert restored.active_alerts == []
