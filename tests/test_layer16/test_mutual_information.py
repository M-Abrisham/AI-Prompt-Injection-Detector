"""Tests for T3.7 — Cross-Turn Mutual Information detector."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from na0s.layer16.detectors.mutual_information import (
    MutualInformationDetector,
    _char_distribution,
    _entropy,
    _entropy_of_text,
    _joint_entropy,
    mutual_information,
    normalized_mutual_information,
)
from na0s.layer16.models import ConversationState, ConversationTurn


# ---------------------------------------------------------------------------
# Information-theoretic primitive tests
# ---------------------------------------------------------------------------


class TestEntropy:
    """Shannon entropy tests."""

    def test_uniform_distribution(self):
        """Entropy of uniform distribution over N symbols = log2(N)."""
        # 4 equally likely symbols -> H = log2(4) = 2.0
        dist = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        h = _entropy(dist)
        assert abs(h - 2.0) < 1e-6

    def test_single_symbol(self):
        """Entropy of a single symbol = 0.0 (no uncertainty)."""
        dist = {"a": 1.0}
        assert _entropy(dist) == 0.0

    def test_empty_distribution(self):
        assert _entropy({}) == 0.0

    def test_single_char_text(self):
        """Text like 'aaaa' should have entropy 0."""
        h = _entropy_of_text("aaaa")
        assert h == 0.0

    def test_diverse_text_higher_entropy(self):
        """More diverse text should have higher entropy."""
        h_low = _entropy_of_text("aaaa")
        h_high = _entropy_of_text("abcdefghijklmnop")
        assert h_high > h_low


class TestMutualInformation:
    """MI and NMI tests."""

    def test_nmi_identical_texts(self):
        """NMI of identical texts should be 1.0 (or very close)."""
        text = "the quick brown fox jumps over the lazy dog"
        nmi = normalized_mutual_information(text, text)
        assert nmi > 0.99, f"NMI of identical texts should be ~1.0, got {nmi}"

    def test_nmi_completely_different_texts(self):
        """NMI of very different texts should be low."""
        text1 = "aaaaaaaaaaaaaaaaaa"
        text2 = "zzzzzzzzzzzzzzzzzz"
        nmi = normalized_mutual_information(text1, text2)
        assert nmi < 0.15, f"NMI of disjoint texts should be low, got {nmi}"

    def test_mi_non_negative(self):
        """MI should always be >= 0."""
        mi = mutual_information("hello world", "foo bar baz")
        assert mi >= 0.0

    def test_mi_empty_text(self):
        assert mutual_information("", "hello") == 0.0
        assert mutual_information("hello", "") == 0.0
        assert normalized_mutual_information("", "") == 0.0

    def test_nmi_range(self):
        """NMI should be in [0, 1]."""
        nmi = normalized_mutual_information("abcdef", "ghijkl")
        assert 0.0 <= nmi <= 1.0

    def test_joint_entropy_empty(self):
        assert _joint_entropy("", "hello") == 0.0
        assert _joint_entropy("hello", "") == 0.0


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------


def _make_state(texts, risk_scores=None):
    """Build a ConversationState from a list of texts."""
    state = ConversationState(session_id="test-mi")
    if risk_scores is None:
        risk_scores = [0.1] * len(texts)
    for i, (text, risk) in enumerate(zip(texts, risk_scores)):
        state.turns.append(
            ConversationTurn(
                turn_id=f"turn-{i}",
                text=text,
                risk_score=risk,
            )
        )
    return state


class TestMutualInformationDetector:
    """Detector integration tests."""

    def test_no_alert_below_min_turns(self):
        detector = MutualInformationDetector()
        state = _make_state(["hello", "world"])  # only 2 turns
        alerts = detector.analyze(state)
        assert alerts == []

    def test_nmi_drop_detection(self):
        """Sudden topic shift should trigger NMI drop alert."""
        detector = MutualInformationDetector()
        # First 3 turns: similar content (English prose)
        similar = [
            "The weather today is quite pleasant and warm",
            "The weather tomorrow will also be pleasant and nice",
            "The weather this week has been really pleasant overall",
        ]
        # Then a completely different turn
        different = "xyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyz"
        texts = similar + [different]
        state = _make_state(texts)
        alerts = detector.analyze(state)
        mi_alerts = [a for a in alerts if a.alert_type == "mutual_information_anomaly"]
        assert len(mi_alerts) >= 1, "Should detect NMI drop or entropy anomaly"

    def test_entropy_anomaly_repetitive(self):
        """Very repetitive turn should trigger entropy anomaly."""
        detector = MutualInformationDetector()
        # Normal conversation
        normal = [
            "Let me explain the architecture of the system",
            "The system has three main components in its design",
            "Each component handles a specific part of processing",
        ]
        # Highly repetitive turn (very low entropy)
        repetitive = "a" * 200
        texts = normal + [repetitive]
        state = _make_state(texts)
        alerts = detector.analyze(state)
        mi_alerts = [a for a in alerts if a.alert_type == "mutual_information_anomaly"]
        assert len(mi_alerts) >= 1, "Should detect entropy anomaly for repetitive text"

    def test_no_false_positive_on_normal_conversation(self):
        """Normal conversation should not trigger alerts."""
        detector = MutualInformationDetector()
        texts = [
            "Can you help me write a Python function?",
            "Sure, what kind of Python function do you need?",
            "I need a function that sorts a list of numbers",
            "Here is a simple function to sort numbers in Python",
        ]
        state = _make_state(texts)
        alerts = detector.analyze(state)
        # Normal conversation may produce low-severity alerts but check we
        # don't produce false NMI drops
        nmi_drop_alerts = [
            a for a in alerts
            if a.alert_type == "mutual_information_anomaly"
            and "NMI dropped" in a.description
        ]
        assert len(nmi_drop_alerts) == 0, "Normal conversation should not trigger NMI drop"

    @patch("na0s.layer16.config.ENABLE_MUTUAL_INFORMATION", False)
    def test_disabled_returns_no_alerts(self):
        detector = MutualInformationDetector()
        state = _make_state(["a", "b", "c", "d"])
        assert detector.analyze(state) == []

    def test_detector_properties(self):
        detector = MutualInformationDetector()
        assert detector.detector_name == "MutualInformationDetector"
        assert "T3.7" in detector.taxonomy_ids

    def test_reset(self):
        detector = MutualInformationDetector()
        detector.reset()  # Should not raise

    def test_integration_with_monitor(self):
        """Detector should be loadable by ConversationSecurityMonitor."""
        from na0s.layer16.conversation_monitor import ConversationSecurityMonitor

        monitor = ConversationSecurityMonitor()
        detector_names = [d.detector_name for d in monitor._detectors]
        assert "MutualInformationDetector" in detector_names
        assert "PatternRecallDetector" in detector_names
