"""Tests for Layer 16 SlidingWindow."""

from __future__ import annotations

from datetime import datetime, timezone

from na0s.layer16.models import ConversationTurn
from na0s.layer16.sliding_window import SlidingWindow


def _make_turn(text: str, risk_score: float = 0.0) -> ConversationTurn:
    return ConversationTurn(
        turn_id=f"t-{text}",
        text=text,
        timestamp=datetime.now(timezone.utc),
        risk_score=risk_score,
    )


class TestAdd:
    def test_add_single(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_make_turn("hello"))
        assert sw.size == 1

    def test_add_multiple(self) -> None:
        sw = SlidingWindow(max_size=5)
        for i in range(3):
            sw.add(_make_turn(f"turn-{i}"))
        assert sw.size == 3


class TestOverflow:
    def test_evicts_oldest_when_full(self) -> None:
        sw = SlidingWindow(max_size=3)
        sw.add(_make_turn("a"))
        sw.add(_make_turn("b"))
        sw.add(_make_turn("c"))
        assert sw.is_full is True
        sw.add(_make_turn("d"))
        assert sw.size == 3
        texts = [t.text for t in sw.get_turns()]
        assert texts == ["b", "c", "d"]

    def test_is_full_property(self) -> None:
        sw = SlidingWindow(max_size=2)
        assert sw.is_full is False
        sw.add(_make_turn("a"))
        assert sw.is_full is False
        sw.add(_make_turn("b"))
        assert sw.is_full is True

    def test_size_never_exceeds_max(self) -> None:
        sw = SlidingWindow(max_size=3)
        for i in range(10):
            sw.add(_make_turn(f"t-{i}"))
        assert sw.size == 3


class TestGetTurns:
    def test_returns_list(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_make_turn("x"))
        turns = sw.get_turns()
        assert isinstance(turns, list)
        assert len(turns) == 1

    def test_empty_window(self) -> None:
        sw = SlidingWindow(max_size=5)
        assert sw.get_turns() == []

    def test_order_preserved(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_make_turn("first"))
        sw.add(_make_turn("second"))
        sw.add(_make_turn("third"))
        texts = [t.text for t in sw.get_turns()]
        assert texts == ["first", "second", "third"]


class TestCombinedText:
    def test_combined_text(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_make_turn("hello"))
        sw.add(_make_turn("world"))
        assert sw.get_combined_text() == "hello\n---\nworld"

    def test_empty_combined(self) -> None:
        sw = SlidingWindow(max_size=5)
        assert sw.get_combined_text() == ""


class TestRiskScores:
    def test_risk_scores(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_make_turn("a", risk_score=0.1))
        sw.add(_make_turn("b", risk_score=0.5))
        sw.add(_make_turn("c", risk_score=0.9))
        assert sw.get_risk_scores() == [0.1, 0.5, 0.9]

    def test_risk_scores_empty(self) -> None:
        sw = SlidingWindow(max_size=5)
        assert sw.get_risk_scores() == []

    def test_risk_scores_after_overflow(self) -> None:
        sw = SlidingWindow(max_size=2)
        sw.add(_make_turn("a", risk_score=0.1))
        sw.add(_make_turn("b", risk_score=0.5))
        sw.add(_make_turn("c", risk_score=0.9))
        assert sw.get_risk_scores() == [0.5, 0.9]


class TestProperties:
    def test_max_size(self) -> None:
        sw = SlidingWindow(max_size=7)
        assert sw.max_size == 7

    def test_default_max_size(self) -> None:
        sw = SlidingWindow()
        assert sw.max_size == 10
