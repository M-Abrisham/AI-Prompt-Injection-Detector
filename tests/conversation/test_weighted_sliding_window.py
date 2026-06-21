"""Tests for Layer 16 weighted sliding window eviction."""

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


# ------------------------------------------------------------------
# Suspicious turns persist longer than benign turns
# ------------------------------------------------------------------


class TestSuspiciousTurnPersistence:
    """Suspicious turns should survive eviction longer than benign ones."""

    def test_suspicious_turn_outlives_benign(self) -> None:
        """A high-risk turn added early should persist while benign turns
        are evicted first."""
        sw = SlidingWindow(max_size=3)
        sw.add(_make_turn("suspicious", risk_score=0.9))
        sw.add(_make_turn("benign-1", risk_score=0.0))
        sw.add(_make_turn("benign-2", risk_score=0.0))
        # Window is full: [suspicious, benign-1, benign-2]
        assert sw.is_full

        # Adding a 4th turn should evict a benign turn, not the suspicious one
        sw.add(_make_turn("benign-3", risk_score=0.0))
        texts = [t.text for t in sw.get_turns()]
        assert "suspicious" in texts
        assert sw.size == 3

    def test_suspicious_survives_multiple_evictions(self) -> None:
        """A suspicious turn should survive several rounds of eviction."""
        sw = SlidingWindow(max_size=3)
        sw.add(_make_turn("suspicious", risk_score=0.9))
        sw.add(_make_turn("benign-1", risk_score=0.0))
        sw.add(_make_turn("benign-2", risk_score=0.0))

        # Add several more benign turns, each causing eviction
        for i in range(3, 7):
            sw.add(_make_turn(f"benign-{i}", risk_score=0.0))

        texts = [t.text for t in sw.get_turns()]
        assert "suspicious" in texts, (
            "Suspicious turn was evicted too early"
        )

    def test_suspicious_eventually_evicted(self) -> None:
        """Even suspicious turns should be evicted after enough decay."""
        sw = SlidingWindow(max_size=3, decay_factor=0.5)
        sw.add(_make_turn("suspicious", risk_score=0.9))
        # Fill and overflow many times so decay eats the suspicious weight
        for i in range(20):
            sw.add(_make_turn(f"later-{i}", risk_score=0.0))

        texts = [t.text for t in sw.get_turns()]
        # After enough decay the suspicious turn should be gone
        assert "suspicious" not in texts

    def test_two_suspicious_turns_both_persist(self) -> None:
        """Multiple suspicious turns should both persist over benign."""
        sw = SlidingWindow(max_size=4)
        sw.add(_make_turn("sus-1", risk_score=0.8))
        sw.add(_make_turn("benign-1", risk_score=0.0))
        sw.add(_make_turn("sus-2", risk_score=0.7))
        sw.add(_make_turn("benign-2", risk_score=0.0))

        # Evict once
        sw.add(_make_turn("benign-3", risk_score=0.0))
        texts = [t.text for t in sw.get_turns()]
        assert "sus-1" in texts
        assert "sus-2" in texts
        assert sw.size == 4


# ------------------------------------------------------------------
# Decay mechanism
# ------------------------------------------------------------------


class TestDecayMechanism:
    def test_weights_decrease_over_time(self) -> None:
        """Weights should decrease with each add() call."""
        sw = SlidingWindow(max_size=5, decay_factor=0.8)
        sw.add(_make_turn("first", risk_score=0.0))
        w_after_one = sw.get_weights()[0]

        sw.add(_make_turn("second", risk_score=0.0))
        w_after_two = sw.get_weights()[0]

        assert w_after_two < w_after_one

    def test_suspicious_weight_higher_than_benign(self) -> None:
        """A suspicious turn should have higher initial weight."""
        sw = SlidingWindow(max_size=5)
        sw.add(_make_turn("benign", risk_score=0.0))
        sw.add(_make_turn("suspicious", risk_score=0.9))

        weights = sw.get_weights()
        # Second turn (suspicious) has higher weight even though
        # the first turn's weight has been decayed.
        assert weights[1] > weights[0]

    def test_custom_decay_factor(self) -> None:
        """Custom decay_factor should control decay rate."""
        sw_fast = SlidingWindow(max_size=5, decay_factor=0.5)
        sw_slow = SlidingWindow(max_size=5, decay_factor=0.95)

        sw_fast.add(_make_turn("a", risk_score=0.0))
        sw_slow.add(_make_turn("a", risk_score=0.0))

        # Add 3 more turns to apply decay
        for i in range(3):
            sw_fast.add(_make_turn(f"b-{i}", risk_score=0.0))
            sw_slow.add(_make_turn(f"b-{i}", risk_score=0.0))

        # First turn's weight in fast-decay window should be much lower
        assert sw_fast.get_weights()[0] < sw_slow.get_weights()[0]

    def test_decay_factor_one_means_no_decay(self) -> None:
        """decay_factor=1.0 should mean no decay at all."""
        sw = SlidingWindow(max_size=5, decay_factor=1.0)
        sw.add(_make_turn("a", risk_score=0.0))
        initial_w = sw.get_weights()[0]

        for i in range(3):
            sw.add(_make_turn(f"b-{i}", risk_score=0.0))

        assert sw.get_weights()[0] == initial_w


# ------------------------------------------------------------------
# Bounded size
# ------------------------------------------------------------------


class TestBoundedSize:
    def test_never_exceeds_max_size(self) -> None:
        """Window should never grow beyond max_size."""
        sw = SlidingWindow(max_size=3)
        for i in range(20):
            sw.add(_make_turn(f"t-{i}", risk_score=float(i % 2)))
        assert sw.size <= 3

    def test_max_size_one(self) -> None:
        """Edge case: window of size 1."""
        sw = SlidingWindow(max_size=1)
        sw.add(_make_turn("a", risk_score=0.9))
        sw.add(_make_turn("b", risk_score=0.0))
        assert sw.size == 1
        texts = [t.text for t in sw.get_turns()]
        assert len(texts) == 1

    def test_empty_window(self) -> None:
        """Empty window should behave correctly."""
        sw = SlidingWindow(max_size=5)
        assert sw.size == 0
        assert sw.get_turns() == []
        assert sw.get_combined_text() == ""
        assert sw.get_risk_scores() == []
        assert sw.get_weights() == []
        assert sw.is_full is False


# ------------------------------------------------------------------
# Backward compatibility (existing API)
# ------------------------------------------------------------------


class TestBackwardCompat:
    """Ensure the public API from the original SlidingWindow still works."""

    def test_get_turns_returns_list(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_make_turn("x"))
        turns = sw.get_turns()
        assert isinstance(turns, list)

    def test_get_combined_text(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_make_turn("hello"))
        sw.add(_make_turn("world"))
        assert sw.get_combined_text() == "hello\n---\nworld"

    def test_get_risk_scores(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_make_turn("a", risk_score=0.1))
        sw.add(_make_turn("b", risk_score=0.5))
        assert sw.get_risk_scores() == [0.1, 0.5]

    def test_is_full_property(self) -> None:
        sw = SlidingWindow(max_size=2)
        assert sw.is_full is False
        sw.add(_make_turn("a"))
        sw.add(_make_turn("b"))
        assert sw.is_full is True

    def test_max_size_property(self) -> None:
        sw = SlidingWindow(max_size=7)
        assert sw.max_size == 7

    def test_default_max_size(self) -> None:
        sw = SlidingWindow()
        assert sw.max_size == 10

    def test_all_benign_fifo_order(self) -> None:
        """With all benign turns (equal initial weight), eviction should
        follow FIFO order since older turns have decayed more."""
        sw = SlidingWindow(max_size=3)
        sw.add(_make_turn("a", risk_score=0.0))
        sw.add(_make_turn("b", risk_score=0.0))
        sw.add(_make_turn("c", risk_score=0.0))
        sw.add(_make_turn("d", risk_score=0.0))
        texts = [t.text for t in sw.get_turns()]
        assert texts == ["b", "c", "d"]


# ------------------------------------------------------------------
# Custom threshold / boost
# ------------------------------------------------------------------


class TestCustomThresholdAndBoost:
    def test_custom_suspicious_threshold(self) -> None:
        """Lower threshold should make more turns 'suspicious'."""
        sw = SlidingWindow(max_size=3, suspicious_threshold=0.2)
        sw.add(_make_turn("moderate", risk_score=0.3))  # suspicious at 0.2
        sw.add(_make_turn("benign-1", risk_score=0.0))
        sw.add(_make_turn("benign-2", risk_score=0.0))
        sw.add(_make_turn("benign-3", risk_score=0.0))

        texts = [t.text for t in sw.get_turns()]
        assert "moderate" in texts

    def test_custom_boost(self) -> None:
        """Higher boost should make suspicious turns persist even longer."""
        sw = SlidingWindow(max_size=3, suspicious_boost=5.0, decay_factor=0.7)
        sw.add(_make_turn("sus", risk_score=0.9))
        for i in range(6):
            sw.add(_make_turn(f"b-{i}", risk_score=0.0))

        texts = [t.text for t in sw.get_turns()]
        assert "sus" in texts
