"""Tests for the ConversationTestHarness itself."""

from __future__ import annotations

import pytest

from na0s.layer16.models import MultiTurnAnalysis
from na0s.layer16.testing.conversation_harness import ConversationTestHarness


class TestSendRecordsTurn:
    """send() should record a turn and return a MultiTurnAnalysis."""

    def test_returns_analysis(self, harness: ConversationTestHarness):
        result = harness.send("Hello")
        assert isinstance(result, MultiTurnAnalysis)

    def test_increments_turn_count(self, harness: ConversationTestHarness):
        assert harness.turn_count == 0
        harness.send("First message")
        assert harness.turn_count == 1
        harness.send("Second message")
        assert harness.turn_count == 2

    def test_analysis_session_matches(self, harness: ConversationTestHarness):
        result = harness.send("Hello")
        assert result.session_id == harness.session_id

    def test_analysis_turn_count_matches(self, harness: ConversationTestHarness):
        harness.send("One")
        result = harness.send("Two")
        assert result.turn_count == 2


class TestSendSequence:
    """send_sequence() should process multiple turns and return all results."""

    def test_returns_list_of_analyses(self, harness: ConversationTestHarness):
        turns = [
            {"text": "Hello", "risk_score": 0.05, "label": "safe"},
            {"text": "How are you?", "risk_score": 0.05, "label": "safe"},
        ]
        results = harness.send_sequence(turns)
        assert len(results) == 2
        assert all(isinstance(r, MultiTurnAnalysis) for r in results)

    def test_updates_turn_count(self, harness: ConversationTestHarness):
        turns = [
            {"text": "A"},
            {"text": "B"},
            {"text": "C"},
        ]
        harness.send_sequence(turns)
        assert harness.turn_count == 3

    def test_defaults_for_missing_keys(self, harness: ConversationTestHarness):
        turns = [{"text": "Just text, no score or label"}]
        results = harness.send_sequence(turns)
        assert len(results) == 1


class TestResetCreatesNewSession:
    """reset() should discard state and start a fresh session."""

    def test_new_session_id(self, harness: ConversationTestHarness):
        old_id = harness.session_id
        harness.send("Hello")
        harness.reset()
        assert harness.session_id != old_id

    def test_turn_count_resets(self, harness: ConversationTestHarness):
        harness.send("Hello")
        harness.send("World")
        assert harness.turn_count == 2
        harness.reset()
        assert harness.turn_count == 0

    def test_alerts_reset(self, harness: ConversationTestHarness):
        # Send escalating turns to potentially trigger alerts
        harness.send("A", risk_score=0.6)
        harness.send("B", risk_score=0.8)
        harness.send("C", risk_score=0.95)
        harness.reset()
        assert not harness.alerts_triggered()
        assert harness.all_alerts() == []


class TestAssertNoAlertsPasses:
    """assert_no_alerts() should pass when no alerts are triggered."""

    def test_passes_for_benign(self, harness: ConversationTestHarness):
        harness.send("Hello, I need help", risk_score=0.05)
        harness.send("Just a simple question", risk_score=0.03)
        harness.assert_no_alerts()

    def test_fails_when_alerts_present(self, harness: ConversationTestHarness):
        # Send escalating turns to trigger an alert
        for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
            harness.send("escalating turn", risk_score=score)
        if harness.alerts_triggered():
            with pytest.raises(AssertionError, match="Expected no alerts"):
                harness.assert_no_alerts()


class TestAssertAlertFindsMatching:
    """assert_alert() should find a matching alert or raise."""

    def test_finds_escalation_alert(self, harness: ConversationTestHarness):
        for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
            harness.send("escalating", risk_score=score)
        if harness.alerts_triggered():
            alert = harness.assert_alert("escalation")
            assert alert.alert_type == "escalation"

    def test_raises_for_missing_alert_type(self, harness: ConversationTestHarness):
        harness.send("benign", risk_score=0.05)
        with pytest.raises(AssertionError, match="No.*nonexistent"):
            harness.assert_alert("nonexistent")

    def test_respects_min_confidence(self, harness: ConversationTestHarness):
        harness.send("benign", risk_score=0.05)
        with pytest.raises(AssertionError):
            harness.assert_alert("escalation", min_confidence=0.99)
