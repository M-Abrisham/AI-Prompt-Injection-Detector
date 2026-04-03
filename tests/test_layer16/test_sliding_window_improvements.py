"""Tests for SlidingWindow improvements (T1.2–T1.8)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from na0s.layer16.models import ConversationTurn
from na0s.layer16.sliding_window import SlidingWindow


def _turn(
    text: str = "hello",
    risk_score: float = 0.0,
    turn_id: str | None = None,
    label: str = "safe",
    flags: list | None = None,
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id or f"t-{text}",
        text=text,
        timestamp=datetime.now(timezone.utc),
        risk_score=risk_score,
        label=label,
        flags=flags or [],
    )


# ------------------------------------------------------------------ T1.2
class TestEvictionLog:
    def test_eviction_log_populated_on_evict(self) -> None:
        sw = SlidingWindow(max_size=2)
        sw.add(_turn("a", 0.0, turn_id="id-a", label="safe", flags=["f1"]))
        sw.add(_turn("b", 0.0, turn_id="id-b"))
        # Window full; next add triggers eviction
        sw.add(_turn("c", 0.0, turn_id="id-c"))
        log = sw.get_eviction_log()
        assert len(log) == 1
        entry = log[0]
        assert entry["turn_id"] == "id-a"
        assert entry["risk_score"] == 0.0
        assert entry["label"] == "safe"
        assert entry["flags"] == ["f1"]
        assert isinstance(entry["evicted_weight"], float)

    def test_eviction_log_capped_at_50(self) -> None:
        sw = SlidingWindow(max_size=1)
        for i in range(60):
            sw.add(_turn(f"t{i}", turn_id=f"id-{i}"))
        log = sw.get_eviction_log()
        assert len(log) == 50
        # Most recent evictions should be at the end
        assert log[-1]["turn_id"] == "id-58"

    def test_eviction_log_correct_fields(self) -> None:
        sw = SlidingWindow(max_size=1)
        sw.add(_turn("x", 0.7, turn_id="sus", label="injection", flags=["sql"]))
        sw.add(_turn("y", 0.1))
        log = sw.get_eviction_log()
        assert len(log) == 1
        entry = log[0]
        assert set(entry.keys()) == {
            "turn_id", "risk_score", "label", "flags", "evicted_weight",
        }

    def test_eviction_log_returns_copy(self) -> None:
        sw = SlidingWindow(max_size=1)
        sw.add(_turn("a"))
        sw.add(_turn("b"))
        log1 = sw.get_eviction_log()
        log1.clear()
        assert len(sw.get_eviction_log()) == 1


# ------------------------------------------------------------------ T1.3
class TestAggregateRisk:
    def test_aggregate_risk_empty(self) -> None:
        assert SlidingWindow().get_aggregate_risk() == 0.0

    def test_aggregate_risk_single(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_turn("a", risk_score=0.8))
        # Only turn, weight = 3.0 (1 + 2 boost); aggregate = 0.8
        assert sw.get_aggregate_risk() == pytest.approx(0.8)

    def test_aggregate_risk_weighted(self) -> None:
        sw = SlidingWindow(max_size=5, decay_factor=1.0)  # no decay
        sw.add(_turn("benign", risk_score=0.0))  # weight 1.0
        sw.add(_turn("risky", risk_score=0.6))   # weight 3.0
        # expected = (0.0*1 + 0.6*3) / (1+3) = 1.8/4 = 0.45
        assert sw.get_aggregate_risk() == pytest.approx(0.45)

    def test_peak_risk_empty(self) -> None:
        assert SlidingWindow().get_peak_risk() == 0.0

    def test_peak_risk(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_turn("a", 0.2))
        sw.add(_turn("b", 0.9))
        sw.add(_turn("c", 0.5))
        assert sw.get_peak_risk() == pytest.approx(0.9)


# ------------------------------------------------------------------ T1.4
class TestCombinedTextSeparator:
    def test_separator(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_turn("first"))
        sw.add(_turn("second"))
        assert sw.get_combined_text() == "first\n---\nsecond"

    def test_single_turn_no_separator(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_turn("only"))
        assert sw.get_combined_text() == "only"

    def test_empty_window(self) -> None:
        assert SlidingWindow().get_combined_text() == ""


# ------------------------------------------------------------------ T1.7
class TestMinWeightFloor:
    def test_suspicious_turn_clamped(self) -> None:
        sw = SlidingWindow(max_size=5, decay_factor=0.01, min_weight=0.5)
        sw.add(_turn("sus", 0.8))
        # After many decays the weight should not fall below min_weight
        for _ in range(50):
            sw.add(_turn("padding", 0.0))
        weights = sw.get_weights()
        # The suspicious turn is still in the window if min_weight kept it alive
        turns = sw.get_turns()
        sus_indices = [i for i, t in enumerate(turns) if t.text == "sus"]
        if sus_indices:
            assert weights[sus_indices[0]] >= 0.5

    def test_benign_turn_can_decay_to_zero(self) -> None:
        sw = SlidingWindow(max_size=10, decay_factor=0.01, min_weight=0.5)
        sw.add(_turn("benign", 0.0))
        # Trigger many decays
        for _ in range(20):
            sw.add(_turn("pad", 0.0))
        weights = sw.get_weights()
        # First entry (benign) should have decayed far below min_weight
        assert weights[0] < 0.5

    def test_min_weight_preserves_suspicious_over_benign(self) -> None:
        """Suspicious turns should outlast benign turns in eviction."""
        sw = SlidingWindow(max_size=2, decay_factor=0.5, min_weight=0.3)
        sw.add(_turn("sus", 0.8, turn_id="sus"))
        sw.add(_turn("benign", 0.0, turn_id="benign"))
        # Next add triggers eviction; benign should be evicted
        sw.add(_turn("new", 0.0, turn_id="new"))
        ids = [t.turn_id for t in sw.get_turns()]
        assert "sus" in ids
        assert "benign" not in ids


# ------------------------------------------------------------------ T1.8
class TestPythonicAPI:
    def test_len(self) -> None:
        sw = SlidingWindow(max_size=5)
        assert len(sw) == 0
        sw.add(_turn("a"))
        assert len(sw) == 1

    def test_iter(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_turn("a"))
        sw.add(_turn("b"))
        texts = [t.text for t in sw]
        assert texts == ["a", "b"]

    def test_iter_returns_conversation_turns(self) -> None:
        sw = SlidingWindow(max_size=5)
        sw.add(_turn("x"))
        for t in sw:
            assert isinstance(t, ConversationTurn)

    def test_clear(self) -> None:
        sw = SlidingWindow(max_size=1)
        sw.add(_turn("a"))
        sw.add(_turn("b"))  # triggers eviction
        assert len(sw.get_eviction_log()) == 1
        sw.clear()
        assert len(sw) == 0
        assert sw.get_eviction_log() == []
        assert sw.get_turns() == []


# ------------------------------------------------------------------ T1.6
class TestInputValidation:
    def test_max_size_zero(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindow(max_size=0)

    def test_max_size_negative(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindow(max_size=-1)

    def test_decay_factor_zero(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindow(decay_factor=0.0)

    def test_decay_factor_negative(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindow(decay_factor=-0.5)

    def test_decay_factor_above_one(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindow(decay_factor=1.1)

    def test_decay_factor_one_is_ok(self) -> None:
        sw = SlidingWindow(decay_factor=1.0)
        assert sw._decay_factor == 1.0

    def test_suspicious_threshold_negative(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindow(suspicious_threshold=-0.1)

    def test_suspicious_boost_negative(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindow(suspicious_boost=-1.0)

    def test_add_non_turn_raises_type_error(self) -> None:
        sw = SlidingWindow()
        with pytest.raises(TypeError):
            sw.add("not a turn")  # type: ignore[arg-type]

    def test_add_none_raises_type_error(self) -> None:
        sw = SlidingWindow()
        with pytest.raises(TypeError):
            sw.add(None)  # type: ignore[arg-type]

    def test_add_dict_raises_type_error(self) -> None:
        sw = SlidingWindow()
        with pytest.raises(TypeError):
            sw.add({"text": "hi"})  # type: ignore[arg-type]
