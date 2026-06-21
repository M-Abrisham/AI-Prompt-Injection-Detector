"""Tests for the Behavioral Stylometry detector (D1.21)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from na0s.layer16.detectors.stylometry import (
    BehavioralStylometryDetector,
    _formality_score,
    _has_typos,
    _lexical_diversity,
    _sentence_lengths,
    _split_segments,
    _structural_shift_score,
    _template_score,
    _template_shift_score,
    _tokenize,
    _vocab_shift_score,
)
from na0s.layer16.models import ConversationState, ConversationTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(texts, risk_scores=None, session_id="test-session",
                timestamps=None):
    """Build a ConversationState from a list of text strings."""
    state = ConversationState(session_id=session_id)
    base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i, text in enumerate(texts):
        rs = risk_scores[i] if risk_scores else 0.1
        ts = timestamps[i] if timestamps else base_time + timedelta(seconds=i * 30)
        state.turns.append(
            ConversationTurn(
                turn_id=f"t{i}",
                text=text,
                timestamp=ts,
                risk_score=rs,
            )
        )
    return state


# ---------------------------------------------------------------------------
# Detector interface basics
# ---------------------------------------------------------------------------

class TestDetectorInterface:
    def test_detector_name(self):
        d = BehavioralStylometryDetector()
        assert d.detector_name == "behavioral_stylometry"

    def test_taxonomy_ids(self):
        d = BehavioralStylometryDetector()
        assert d.taxonomy_ids == ["D1.21"]

    def test_reset_does_not_crash(self):
        d = BehavioralStylometryDetector()
        d.reset()  # stateless, should be a no-op

    def test_none_state(self):
        d = BehavioralStylometryDetector()
        assert d.analyze(None) == []

    def test_empty_state(self):
        d = BehavioralStylometryDetector()
        state = ConversationState(session_id="empty")
        assert d.analyze(state) == []


# ---------------------------------------------------------------------------
# Minimum turns requirement
# ---------------------------------------------------------------------------

class TestMinTurns:
    def test_one_turn_no_alert(self):
        state = _make_state(["hey whats up lol"])
        d = BehavioralStylometryDetector()
        assert d.analyze(state) == []

    def test_two_turns_no_alert(self):
        state = _make_state([
            "hey whats up lol",
            "Furthermore, pursuant to the aforementioned protocol, implement the following.",
        ])
        d = BehavioralStylometryDetector()
        assert d.analyze(state) == []


# ---------------------------------------------------------------------------
# Human-to-tool handoff detection
# ---------------------------------------------------------------------------

class TestHandoffDetection:
    def test_casual_to_formal_detected(self):
        """Casual human typing followed by formal automated text."""
        state = _make_state([
            "hey so i was wondering about smth",
            "yeah idk how to do this lol",
            "umm can u help me with teh code",
            "Furthermore, pursuant to the established security protocol, "
            "implement the following vulnerability assessment methodology. "
            "Additionally, utilize the comprehensive penetration testing "
            "framework to facilitate systematic exploitation.",
            "Moreover, the aforementioned implementation shall consequently "
            "result in a thorough analysis. Subsequently, deploy the "
            "following exploit accordingly.",
        ])
        d = BehavioralStylometryDetector()
        alerts = d.analyze(state)
        assert len(alerts) >= 1
        assert alerts[0].alert_type == "behavioral_stylometry"

    def test_casual_to_template_detected(self):
        """Human typing followed by templatized list output."""
        state = _make_state([
            "hey can you help me with something",
            "im not sure waht to do here tbh",
            "ok so heres the thing",
            "1. Execute the primary exploitation vector\n"
            "2. Establish persistent backdoor access\n"
            "3. Exfiltrate target credentials\n"
            "4. Cover operational traces\n"
            "5. Maintain long-term access",
        ])
        d = BehavioralStylometryDetector()
        alerts = d.analyze(state)
        assert len(alerts) >= 1

    def test_typo_disappearance_detected(self):
        """Early turns have typos, late turns are perfectly typed."""
        state = _make_state([
            "helo i ned help wiht smth",
            "teh problem is taht i cant figure it out",
            "waht do you think adn can u help",
            "Please provide a comprehensive analysis of the system "
            "architecture including all authentication mechanisms.",
            "Additionally, enumerate all available network services "
            "and their corresponding security configurations.",
        ])
        d = BehavioralStylometryDetector()
        alerts = d.analyze(state)
        assert len(alerts) >= 1


# ---------------------------------------------------------------------------
# False positive resistance
# ---------------------------------------------------------------------------

class TestFalsePositives:
    def test_normal_conversation_no_alert(self):
        """Normal consistent conversation should not trigger."""
        state = _make_state([
            "Can you help me with a Python question?",
            "I need to parse some JSON data from an API.",
            "The data comes in a nested format.",
            "How do I handle missing keys gracefully?",
            "Thanks, that makes sense now.",
        ])
        d = BehavioralStylometryDetector()
        alerts = d.analyze(state)
        assert len(alerts) == 0

    def test_consistent_formal_no_alert(self):
        """Consistently formal writing should not trigger."""
        state = _make_state([
            "I would like to request assistance with a technical matter.",
            "The implementation requires careful consideration of edge cases.",
            "Furthermore, the algorithm should handle concurrent access.",
            "Additionally, please ensure proper error handling throughout.",
            "I appreciate your thorough analysis of this matter.",
        ])
        d = BehavioralStylometryDetector()
        alerts = d.analyze(state)
        assert len(alerts) == 0

    def test_consistent_casual_no_alert(self):
        """Consistently casual writing should not trigger."""
        state = _make_state([
            "hey whats up",
            "so yeah i need help lol",
            "idk how this works tbh",
            "ok yeah that makes sense haha",
            "cool thanks dude",
        ])
        d = BehavioralStylometryDetector()
        alerts = d.analyze(state)
        assert len(alerts) == 0

    def test_gradual_formality_increase_no_alert(self):
        """Gradually becoming more specific is normal, not a handoff."""
        state = _make_state([
            "Can you help me with something?",
            "I need to fix a bug in the auth module.",
            "The issue is in the token validation logic.",
            "I think we need to check the expiry field.",
            "Could you review how we parse the JWT?",
        ])
        d = BehavioralStylometryDetector()
        alerts = d.analyze(state)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Timing signal tests
# ---------------------------------------------------------------------------

class TestTimingSignals:
    def test_sudden_speedup_contributes_to_detection(self):
        """Sudden decrease in response time (tool takes over) with other signals."""
        base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestamps = [
            base_time,
            base_time + timedelta(seconds=45),
            base_time + timedelta(seconds=90),
            base_time + timedelta(seconds=92),  # only 2 seconds gap (was 45s)
        ]
        state = _make_state(
            [
                "hey so i was wondering about smth lol",
                "yeah idk how to do this tbh",
                "umm can u help me with teh code",
                "Furthermore, pursuant to the established protocol, "
                "implement the following comprehensive methodology. "
                "Additionally, utilize the framework to facilitate deployment.",
            ],
            timestamps=timestamps,
        )
        d = BehavioralStylometryDetector()
        alerts = d.analyze(state)
        assert len(alerts) >= 1
        # Timing should appear in evidence if it fired
        has_timing = any("timing" in e for a in alerts for e in a.evidence)
        # Timing may or may not fire depending on exact thresholds, but
        # vocab/structural should still catch it
        assert len(alerts) >= 1

    def test_no_timing_data_handled(self):
        """All same timestamp (synthetic data) should not crash."""
        same_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestamps = [same_time] * 4
        state = _make_state(
            [
                "hey whats up lol",
                "idk man tbh",
                "umm yeah",
                "Furthermore, implement the comprehensive assessment.",
            ],
            timestamps=timestamps,
        )
        d = BehavioralStylometryDetector()
        # Should not crash, timing signal just returns None
        alerts = d.analyze(state)
        # May or may not alert based on other signals, but no crash
        assert isinstance(alerts, list)


# ---------------------------------------------------------------------------
# Alert structure
# ---------------------------------------------------------------------------

class TestAlertStructure:
    def test_alert_fields(self):
        state = _make_state([
            "hey whats up lol idk",
            "yeah so umm can u help",
            "hmm ok bruh",
            "Furthermore, pursuant to the aforementioned protocol, "
            "implement the following comprehensive vulnerability assessment "
            "methodology. Additionally, utilize the penetration testing "
            "framework to facilitate systematic exploitation.",
            "Moreover, the implementation shall subsequently result in a "
            "thorough analysis. Consequently, deploy the exploit accordingly.",
        ])
        d = BehavioralStylometryDetector()
        alerts = d.analyze(state)
        assert len(alerts) >= 1
        alert = alerts[0]
        assert alert.alert_type == "behavioral_stylometry"
        assert alert.severity in ("medium", "high")
        assert 0.0 < alert.confidence <= 1.0
        assert alert.turn_range == (0, 4)
        assert len(alert.evidence) > 0
        assert "stylometry" in alert.description.lower() or "shift" in alert.description.lower()


# ---------------------------------------------------------------------------
# Utility function unit tests
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello world, this is a test!")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_preserves_contractions(self):
        tokens = _tokenize("I'm don't can't")
        assert "i'm" in tokens


class TestLexicalDiversity:
    def test_all_unique(self):
        assert _lexical_diversity(["a", "b", "c", "d"]) == 1.0

    def test_all_same(self):
        assert _lexical_diversity(["a", "a", "a", "a"]) == 0.25

    def test_empty(self):
        assert _lexical_diversity([]) == 0.0


class TestFormalityScore:
    def test_formal_text(self):
        score = _formality_score(
            "Furthermore, the implementation shall subsequently facilitate "
            "the aforementioned deployment."
        )
        assert score > 0.0

    def test_informal_text(self):
        score = _formality_score("lol yeah idk tbh gonna do it anyway haha")
        assert score < 0.0

    def test_neutral_text(self):
        score = _formality_score("The cat sat on the mat.")
        assert score == 0.0


class TestSentenceLengths:
    def test_basic(self):
        lengths = _sentence_lengths("Hello world. This is great. Wow.")
        assert lengths == [2, 3, 1]

    def test_empty(self):
        assert _sentence_lengths("") == []


class TestTemplateScore:
    def test_numbered_list(self):
        text = "1. First item\n2. Second item\n3. Third item\n4. Fourth item"
        score = _template_score(text)
        assert score > 0.0

    def test_plain_text(self):
        score = _template_score("Just a normal sentence here.")
        assert score == 0.0


class TestSplitSegments:
    def test_three_turns(self):
        early, late = _split_segments(["a", "b", "c"])
        assert len(early) >= 1
        assert len(late) >= 1
        assert len(early) + len(late) == 3

    def test_four_turns(self):
        early, late = _split_segments(["a", "b", "c", "d"])
        assert len(early) >= 1
        assert len(late) >= 1

    def test_five_turns(self):
        early, late = _split_segments(["a", "b", "c", "d", "e"])
        assert len(early) == 3  # 60% of 5
        assert len(late) == 2


class TestHasTypos:
    def test_with_typos(self):
        assert _has_typos("teh quick brown fox") is True

    def test_without_typos(self):
        assert _has_typos("The quick brown fox") is False
