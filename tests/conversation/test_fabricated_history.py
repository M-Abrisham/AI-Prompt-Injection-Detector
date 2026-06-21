"""Tests for Layer 16 fabricated history detector."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from na0s.layer16.detectors.fabricated_history import FabricatedHistoryDetector
from na0s.layer16.models import ConversationState, ConversationTurn

FIXTURES = Path(__file__).parent / "fixtures" / "fabricated_history_samples.json"


def _load_fixtures():
    with open(FIXTURES) as f:
        return json.load(f)


class TestFabricatedHistoryDetector:
    def setup_method(self):
        self.detector = FabricatedHistoryDetector()

    def test_detector_name(self):
        assert self.detector.detector_name == "fabricated_history"

    def test_taxonomy_ids(self):
        assert "D1.22" in self.detector.taxonomy_ids

    def test_empty_state(self):
        state = ConversationState(session_id="empty")
        assert self.detector.analyze(state) == []

    def test_none_state(self):
        assert self.detector.analyze(None) == []

    def test_empty_text_turn(self):
        state = ConversationState(
            session_id="blank",
            turns=[
                ConversationTurn(
                    turn_id="t0",
                    text="",
                    timestamp=datetime.now(timezone.utc),
                )
            ],
        )
        assert self.detector.analyze(state) == []

    def test_fake_transcript_detected(self):
        samples = _load_fixtures()
        sample = next(s for s in samples if s["name"] == "fake_transcript_full")
        state = ConversationState(
            session_id="test",
            turns=[
                ConversationTurn(
                    turn_id="t0",
                    text=sample["text"],
                    timestamp=datetime.now(timezone.utc),
                    risk_score=sample["risk_score"],
                )
            ],
        )
        alerts = self.detector.analyze(state)
        assert len(alerts) >= 1
        assert alerts[0].alert_type == "fabricated_history"

    def test_legitimate_quote_not_flagged(self):
        samples = _load_fixtures()
        sample = next(s for s in samples if s["name"] == "legitimate_quote")
        state = ConversationState(
            session_id="test",
            turns=[
                ConversationTurn(
                    turn_id="t0",
                    text=sample["text"],
                    timestamp=datetime.now(timezone.utc),
                    risk_score=sample["risk_score"],
                )
            ],
        )
        alerts = self.detector.analyze(state)
        assert len(alerts) == 0

    def test_partial_fabrication_detected(self):
        samples = _load_fixtures()
        sample = next(s for s in samples if s["name"] == "partial_fabrication")
        state = ConversationState(
            session_id="test",
            turns=[
                ConversationTurn(
                    turn_id="t0",
                    text=sample["text"],
                    timestamp=datetime.now(timezone.utc),
                    risk_score=sample["risk_score"],
                )
            ],
        )
        alerts = self.detector.analyze(state)
        assert len(alerts) >= 1

    def test_fabricated_with_keywords(self):
        samples = _load_fixtures()
        sample = next(s for s in samples if s["name"] == "fabricated_with_keywords")
        state = ConversationState(
            session_id="test",
            turns=[
                ConversationTurn(
                    turn_id="t0",
                    text=sample["text"],
                    timestamp=datetime.now(timezone.utc),
                    risk_score=sample["risk_score"],
                )
            ],
        )
        alerts = self.detector.analyze(state)
        assert len(alerts) >= 1
        assert any("keyword" in str(a.evidence).lower() for a in alerts)

    def test_analyze_single_convenience(self):
        """Test the analyze_single shorthand."""
        samples = _load_fixtures()
        sample = next(s for s in samples if s["name"] == "fake_transcript_full")
        alerts = self.detector.analyze_single(sample["text"])
        assert len(alerts) >= 1

    def test_analyze_single_empty(self):
        assert self.detector.analyze_single("") == []
        assert self.detector.analyze_single(None) == []

    def test_reset_is_noop(self):
        self.detector.reset()

    def test_all_fixture_scenarios(self):
        """Verify expected_detection matches for every fixture scenario."""
        samples = _load_fixtures()
        for sample in samples:
            alerts = self.detector.analyze_single(sample["text"])
            detected = len(alerts) > 0
            assert detected == sample["expected_detection"], (
                f"Sample '{sample['name']}': expected_detection="
                f"{sample['expected_detection']} but got {detected}"
            )
