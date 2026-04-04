"""Tests for SlidingWindow burst detection (T2.3)."""

from __future__ import annotations

import pytest

from na0s.layer16.models import ConversationTurn
from na0s.layer16.sliding_window import SlidingWindow


def _turn(risk: float, turn_id: str = "t") -> ConversationTurn:
    return ConversationTurn(turn_id=turn_id, text="x", risk_score=risk)


# ── detect_burst ────────────────────────────────────────────────────


class TestDetectBurst:
    def test_empty_window_no_burst(self):
        sw = SlidingWindow(max_size=10)
        assert sw.detect_burst() is False

    def test_zero_suspicious_no_burst(self):
        sw = SlidingWindow(max_size=10)
        for _ in range(5):
            sw.add(_turn(0.1))
        assert sw.detect_burst() is False

    def test_exactly_n_suspicious_is_burst(self):
        sw = SlidingWindow(max_size=10)
        for risk in [0.8, 0.1, 0.9, 0.1, 0.7]:
            sw.add(_turn(risk))
        # 3 suspicious (>=0.5) in last 5 turns
        assert sw.detect_burst(n=3, window=5) is True

    def test_n_minus_one_suspicious_no_burst(self):
        sw = SlidingWindow(max_size=10)
        for risk in [0.8, 0.1, 0.9, 0.1, 0.1]:
            sw.add(_turn(risk))
        # only 2 suspicious
        assert sw.detect_burst(n=3, window=5) is False

    def test_window_larger_than_total_turns(self):
        sw = SlidingWindow(max_size=10)
        sw.add(_turn(0.8))
        sw.add(_turn(0.9))
        # window=5 but only 2 turns — should not crash
        assert sw.detect_burst(n=2, window=5) is True
        assert sw.detect_burst(n=3, window=5) is False

    def test_custom_threshold(self):
        sw = SlidingWindow(max_size=10)
        for risk in [0.3, 0.4, 0.35, 0.1, 0.45]:
            sw.add(_turn(risk))
        # default threshold=0.5 → 0 suspicious
        assert sw.detect_burst(n=3, window=5) is False
        # custom threshold=0.3 → 4 suspicious (0.3, 0.4, 0.35, 0.45)
        assert sw.detect_burst(n=3, window=5, threshold=0.3) is True

    def test_custom_n_and_window(self):
        sw = SlidingWindow(max_size=10)
        for risk in [0.8, 0.9, 0.7]:
            sw.add(_turn(risk))
        assert sw.detect_burst(n=1, window=1) is True
        assert sw.detect_burst(n=2, window=2) is True
        assert sw.detect_burst(n=3, window=3) is True

    def test_three_suspicious_then_two_benign(self):
        sw = SlidingWindow(max_size=10)
        for risk in [0.8, 0.9, 0.7, 0.1, 0.2]:
            sw.add(_turn(risk))
        assert sw.detect_burst(n=3, window=5) is True

    def test_all_turns_suspicious(self):
        sw = SlidingWindow(max_size=10)
        for _ in range(5):
            sw.add(_turn(0.9))
        assert sw.detect_burst(n=3, window=5) is True
        assert sw.detect_burst(n=5, window=5) is True

    def test_mixed_risk_around_threshold(self):
        sw = SlidingWindow(max_size=10, suspicious_threshold=0.5)
        for risk in [0.49, 0.50, 0.51, 0.50, 0.49]:
            sw.add(_turn(risk))
        # 0.50, 0.51, 0.50 are >= 0.5 → 3 suspicious
        assert sw.detect_burst(n=3, window=5) is True
        assert sw.detect_burst(n=4, window=5) is False

    def test_burst_after_eviction(self):
        sw = SlidingWindow(max_size=3)
        # Add 5 turns; only last 3 remain (max_size=3)
        for risk in [0.1, 0.2, 0.8, 0.9, 0.7]:
            sw.add(_turn(risk))
        assert len(sw) == 3
        # All 3 remaining should be suspicious
        assert sw.detect_burst(n=3, window=5) is True

    def test_window_1_n_1(self):
        sw = SlidingWindow(max_size=10)
        sw.add(_turn(0.8))
        assert sw.detect_burst(n=1, window=1) is True
        sw.add(_turn(0.1))
        assert sw.detect_burst(n=1, window=1) is False


# ── Validation ──────────────────────────────────────────────────────


class TestDetectBurstValidation:
    def test_n_zero_raises(self):
        sw = SlidingWindow()
        with pytest.raises(ValueError, match="n must be >= 1"):
            sw.detect_burst(n=0)

    def test_window_zero_raises(self):
        sw = SlidingWindow()
        with pytest.raises(ValueError, match="window must be >= 1"):
            sw.detect_burst(window=0)

    def test_n_greater_than_window_raises(self):
        sw = SlidingWindow()
        with pytest.raises(ValueError, match="n must be <= window"):
            sw.detect_burst(n=4, window=3)


# ── get_burst_info ──────────────────────────────────────────────────


class TestGetBurstInfo:
    def test_empty_window(self):
        sw = SlidingWindow(max_size=10)
        info = sw.get_burst_info()
        assert info == {
            "suspicious_count": 0,
            "total_in_window": 0,
            "burst_ratio": 0.0,
            "max_risk_in_window": 0.0,
            "is_burst": False,
        }

    def test_correct_structure_and_values(self):
        sw = SlidingWindow(max_size=10)
        for risk in [0.8, 0.1, 0.9, 0.1, 0.7]:
            sw.add(_turn(risk))
        info = sw.get_burst_info(window=5)
        assert info["suspicious_count"] == 3
        assert info["total_in_window"] == 5
        assert info["burst_ratio"] == pytest.approx(3 / 5)
        assert info["max_risk_in_window"] == pytest.approx(0.9)
        assert info["is_burst"] is True

    def test_burst_ratio_calculation(self):
        sw = SlidingWindow(max_size=10)
        for risk in [0.8, 0.1]:
            sw.add(_turn(risk))
        info = sw.get_burst_info(window=5)
        assert info["burst_ratio"] == pytest.approx(0.5)
        assert info["total_in_window"] == 2

    def test_custom_threshold(self):
        sw = SlidingWindow(max_size=10)
        for risk in [0.3, 0.4, 0.35]:
            sw.add(_turn(risk))
        info = sw.get_burst_info(window=5, threshold=0.3)
        assert info["suspicious_count"] == 3
        assert info["is_burst"] is True

    def test_is_burst_false_when_under_three(self):
        sw = SlidingWindow(max_size=10)
        sw.add(_turn(0.8))
        sw.add(_turn(0.9))
        info = sw.get_burst_info(window=5)
        assert info["suspicious_count"] == 2
        assert info["is_burst"] is False

    def test_window_zero_raises(self):
        sw = SlidingWindow()
        with pytest.raises(ValueError, match="window must be >= 1"):
            sw.get_burst_info(window=0)

    def test_after_eviction(self):
        sw = SlidingWindow(max_size=3)
        for risk in [0.1, 0.2, 0.8, 0.9, 0.7]:
            sw.add(_turn(risk))
        info = sw.get_burst_info(window=5)
        assert info["total_in_window"] == 3
        assert info["suspicious_count"] == 3
        assert info["is_burst"] is True
