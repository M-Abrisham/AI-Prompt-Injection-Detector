"""Tests for Layer 16 Two-Tier Memory (T3.5): CompressedHistory warm tier."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from na0s.layer16.compressed_history import CompressedHistory, TurnSummary
from na0s.layer16.models import ConversationTurn
from na0s.layer16.sliding_window import SlidingWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(
    risk_score: float = 0.0,
    label: str = "safe",
    flags: list | None = None,
    text: str = "hello world",
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=str(uuid.uuid4()),
        text=text,
        role="user",
        timestamp=datetime.now(timezone.utc),
        risk_score=risk_score,
        label=label,
        flags=flags or [],
    )


# ---------------------------------------------------------------------------
# TurnSummary creation
# ---------------------------------------------------------------------------

class TestTurnSummary:
    def test_basic_creation(self):
        ts = TurnSummary(
            batch_id="abc",
            turn_count=5,
            avg_risk_score=0.3,
            max_risk_score=0.8,
            dominant_label="injection",
            technique_tags={"t1", "t2"},
            text_snippet="some text",
        )
        assert ts.turn_count == 5
        assert ts.avg_risk_score == 0.3
        assert ts.max_risk_score == 0.8
        assert ts.dominant_label == "injection"
        assert ts.technique_tags == {"t1", "t2"}
        assert ts.text_snippet == "some text"
        assert isinstance(ts.created_at, datetime)

    def test_batch_id_stored(self):
        ts = TurnSummary(
            batch_id="xyz-123",
            turn_count=1,
            avg_risk_score=0.0,
            max_risk_score=0.0,
            dominant_label="safe",
            technique_tags=set(),
            text_snippet="",
        )
        assert ts.batch_id == "xyz-123"


# ---------------------------------------------------------------------------
# Batch compression
# ---------------------------------------------------------------------------

class TestBatchCompression:
    def test_five_turns_produce_one_summary(self):
        ch = CompressedHistory(max_summaries=10, batch_size=5)
        results = []
        for i in range(5):
            result = ch.record_eviction(
                _make_turn(risk_score=0.1 * (i + 1), label="safe")
            )
            results.append(result)
        # Only the 5th call should return a summary
        assert all(r is None for r in results[:4])
        assert results[4] is not None
        summary = results[4]
        assert summary.turn_count == 5
        assert summary.avg_risk_score == pytest.approx(0.3)  # (0.1+0.2+0.3+0.4+0.5)/5
        assert summary.max_risk_score == pytest.approx(0.5)

    def test_dominant_label(self):
        ch = CompressedHistory(batch_size=5)
        for i in range(5):
            label = "injection" if i < 3 else "safe"
            ch.record_eviction(_make_turn(label=label))
        summaries = ch.get_summaries()
        assert len(summaries) == 1
        assert summaries[0].dominant_label == "injection"

    def test_technique_tags_union(self):
        ch = CompressedHistory(batch_size=3)
        ch.record_eviction(_make_turn(flags=["t1", "t2"]))
        ch.record_eviction(_make_turn(flags=["t2", "t3"]))
        ch.record_eviction(_make_turn(flags=["t4"]))
        summaries = ch.get_summaries()
        assert summaries[0].technique_tags == {"t1", "t2", "t3", "t4"}

    def test_text_snippet_from_highest_risk(self):
        ch = CompressedHistory(batch_size=3)
        ch.record_eviction(_make_turn(risk_score=0.1, text="low risk turn"))
        ch.record_eviction(_make_turn(risk_score=0.9, text="HIGH RISK TURN with lots of detail"))
        ch.record_eviction(_make_turn(risk_score=0.2, text="medium turn"))
        summary = ch.get_summaries()[0]
        assert summary.text_snippet.startswith("HIGH RISK TURN")

    def test_text_snippet_truncated_at_100(self):
        ch = CompressedHistory(batch_size=1)
        long_text = "A" * 200
        ch.record_eviction(_make_turn(risk_score=0.5, text=long_text))
        summary = ch.get_summaries()[0]
        assert len(summary.text_snippet) == 100


# ---------------------------------------------------------------------------
# Max summaries cap (FIFO eviction)
# ---------------------------------------------------------------------------

class TestMaxSummariesCap:
    def test_fifo_eviction(self):
        ch = CompressedHistory(max_summaries=3, batch_size=2)
        # Create 4 batches -> 8 turns, should keep only last 3 summaries
        for batch_idx in range(4):
            for _ in range(2):
                ch.record_eviction(
                    _make_turn(risk_score=0.1 * (batch_idx + 1))
                )
        summaries = ch.get_summaries()
        assert len(summaries) == 3
        # Oldest batch (risk=0.1) was evicted; second batch (risk=0.2) is now oldest
        assert summaries[0].avg_risk_score == pytest.approx(0.2)

    def test_never_exceeds_max(self):
        ch = CompressedHistory(max_summaries=2, batch_size=1)
        for i in range(20):
            ch.record_eviction(_make_turn(risk_score=0.05 * i))
        assert len(ch.get_summaries()) == 2


# ---------------------------------------------------------------------------
# Historical risk calculation
# ---------------------------------------------------------------------------

class TestHistoricalRisk:
    def test_empty_returns_zero(self):
        ch = CompressedHistory()
        assert ch.get_historical_risk() == 0.0

    def test_recent_weighted_higher(self):
        ch = CompressedHistory(max_summaries=10, batch_size=1)
        # Older batch: low risk
        ch.record_eviction(_make_turn(risk_score=0.1))
        # Newer batch: high risk
        ch.record_eviction(_make_turn(risk_score=0.9))
        # Weighted avg: (0.1*1 + 0.9*2) / (1+2) = 1.9/3 = 0.633...
        risk = ch.get_historical_risk()
        assert risk == pytest.approx(1.9 / 3.0)

    def test_single_summary(self):
        ch = CompressedHistory(batch_size=1)
        ch.record_eviction(_make_turn(risk_score=0.5))
        assert ch.get_historical_risk() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Technique tag aggregation
# ---------------------------------------------------------------------------

class TestTechniqueHistory:
    def test_aggregation_across_summaries(self):
        ch = CompressedHistory(batch_size=1)
        ch.record_eviction(_make_turn(flags=["sql_injection"]))
        ch.record_eviction(_make_turn(flags=["sql_injection", "xss"]))
        ch.record_eviction(_make_turn(flags=["xss"]))
        history = ch.get_technique_history()
        assert history["sql_injection"] == 2
        assert history["xss"] == 2

    def test_empty_returns_empty_dict(self):
        ch = CompressedHistory()
        assert ch.get_technique_history() == {}


# ---------------------------------------------------------------------------
# Integration with SlidingWindow
# ---------------------------------------------------------------------------

class TestSlidingWindowIntegration:
    def test_evictions_flow_to_warm_tier(self):
        ch = CompressedHistory(max_summaries=10, batch_size=3)
        sw = SlidingWindow(max_size=3, compressed_history=ch)

        # Add 6 turns -> first 3 will be evicted as the last 3 arrive
        for i in range(6):
            sw.add(_make_turn(risk_score=0.1 * i))

        # At least some evictions should have reached warm tier
        assert ch.pending_count > 0 or len(ch.get_summaries()) > 0

    def test_get_full_context(self):
        ch = CompressedHistory(max_summaries=10, batch_size=2)
        sw = SlidingWindow(max_size=2, compressed_history=ch)

        # Add 6 turns -> 4 evictions -> 2 summaries in warm tier
        for i in range(6):
            sw.add(_make_turn(risk_score=0.1, flags=["tag_a"]))

        ctx = sw.get_full_context()
        assert "hot_turns" in ctx
        assert "warm_summaries" in ctx
        assert "warm_historical_risk" in ctx
        assert "warm_technique_history" in ctx
        assert "total_turns_seen" in ctx
        assert len(ctx["hot_turns"]) == 2
        # 4 evictions with batch_size=2 -> 2 summaries
        assert len(ctx["warm_summaries"]) == 2
        assert ctx["total_turns_seen"] == 2 + 4  # hot + warm

    def test_full_context_without_warm_tier(self):
        sw = SlidingWindow(max_size=3)
        sw.add(_make_turn(risk_score=0.1))
        ctx = sw.get_full_context()
        assert ctx["warm_summaries"] == []
        assert ctx["warm_historical_risk"] == 0.0
        assert ctx["warm_technique_history"] == {}
        assert ctx["total_turns_seen"] == 1


# ---------------------------------------------------------------------------
# Total turns compressed tracking
# ---------------------------------------------------------------------------

class TestTotalTurnsCompressed:
    def test_tracks_all_compressed_turns(self):
        ch = CompressedHistory(batch_size=2)
        for _ in range(6):
            ch.record_eviction(_make_turn())
        # 6 turns / batch_size 2 = 3 summaries, each with 2 turns
        assert ch.total_turns_compressed == 6

    def test_zero_when_empty(self):
        ch = CompressedHistory()
        assert ch.total_turns_compressed == 0

    def test_pending_not_counted(self):
        ch = CompressedHistory(batch_size=5)
        ch.record_eviction(_make_turn())
        ch.record_eviction(_make_turn())
        assert ch.total_turns_compressed == 0
        assert ch.pending_count == 2


# ---------------------------------------------------------------------------
# Clear method
# ---------------------------------------------------------------------------

class TestClear:
    def test_clears_summaries_and_pending(self):
        ch = CompressedHistory(batch_size=3)
        for _ in range(5):
            ch.record_eviction(_make_turn())
        assert len(ch.get_summaries()) == 1
        assert ch.pending_count == 2
        ch.clear()
        assert len(ch.get_summaries()) == 0
        assert ch.pending_count == 0
        assert ch.total_turns_compressed == 0


# ---------------------------------------------------------------------------
# has_historical_risk
# ---------------------------------------------------------------------------

class TestHasHistoricalRisk:
    def test_no_summaries(self):
        ch = CompressedHistory()
        assert ch.has_historical_risk() is False

    def test_low_risk_below_threshold(self):
        ch = CompressedHistory(batch_size=1)
        ch.record_eviction(_make_turn(risk_score=0.2))
        assert ch.has_historical_risk(threshold=0.5) is False

    def test_high_risk_above_threshold(self):
        ch = CompressedHistory(batch_size=1)
        ch.record_eviction(_make_turn(risk_score=0.8))
        assert ch.has_historical_risk(threshold=0.5) is True


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_max_summaries(self):
        with pytest.raises(ValueError):
            CompressedHistory(max_summaries=0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            CompressedHistory(batch_size=0)

    def test_invalid_turn_type(self):
        ch = CompressedHistory()
        with pytest.raises(TypeError):
            ch.record_eviction("not a turn")  # type: ignore[arg-type]
