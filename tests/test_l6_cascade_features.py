"""Tests for Layer 6 cascade features: ChainIntegrityTracker, RRF Fusion, Groundedness."""

import pytest

from na0s.scan_result import ScanResult
from na0s.chain_integrity import ChainIntegrityTracker
from na0s.rrf_fusion import rrf_score, rrf_decision
from na0s.groundedness import verify_verdict_grounded


# ====================================================================
# Helper factory for ScanResult
# ====================================================================

def _make_result(**kwargs):
    defaults = dict(
        sanitized_text="test",
        is_malicious=False,
        risk_score=0.0,
        label="safe",
        rule_hits=[],
        anomaly_flags=[],
        ml_confidence=0.0,
        ml_label="safe",
        technique_tags=[],
        embedding_score=0.0,
    )
    defaults.update(kwargs)
    return ScanResult(**defaults)


# ====================================================================
# ChainIntegrityTracker
# ====================================================================

class TestChainIntegrityTracker:
    """Tests for trust score propagation across pipeline stages."""

    def test_initial_trust_default(self):
        tracker = ChainIntegrityTracker()
        assert tracker.get_trust_score() == 1.0

    def test_initial_trust_custom(self):
        tracker = ChainIntegrityTracker(initial_trust=0.8)
        assert tracker.get_trust_score() == 0.8

    def test_no_decay_for_clean_stage(self):
        tracker = ChainIntegrityTracker()
        result = _make_result(risk_score=0.1)
        tracker.record_stage("stage1", result)
        assert tracker.get_trust_score() == 1.0

    def test_decay_on_rule_hits(self):
        tracker = ChainIntegrityTracker(decay_rate=0.15)
        result = _make_result(rule_hits=["role_override"])
        tracker.record_stage("stage1", result)
        expected = round(1.0 * 0.85, 6)
        assert tracker.get_trust_score() == expected

    def test_decay_on_anomaly_flags(self):
        tracker = ChainIntegrityTracker(decay_rate=0.15)
        result = _make_result(anomaly_flags=["null_bytes"])
        tracker.record_stage("stage1", result)
        expected = round(1.0 * 0.85, 6)
        assert tracker.get_trust_score() == expected

    def test_decay_on_high_risk_score(self):
        tracker = ChainIntegrityTracker(decay_rate=0.15)
        result = _make_result(risk_score=0.7)
        tracker.record_stage("stage1", result)
        expected = round(1.0 * 0.85, 6)
        assert tracker.get_trust_score() == expected

    def test_triple_decay_when_all_signals(self):
        """When rule_hits, anomaly_flags, AND risk_score all trigger,
        trust decays three times (one per condition)."""
        tracker = ChainIntegrityTracker(decay_rate=0.15)
        result = _make_result(
            rule_hits=["role_override"],
            anomaly_flags=["null_bytes"],
            risk_score=0.8,
        )
        tracker.record_stage("stage1", result)
        expected = round(1.0 * 0.85 * 0.85 * 0.85, 6)
        assert tracker.get_trust_score() == expected

    def test_multi_stage_compounding_decay(self):
        tracker = ChainIntegrityTracker(decay_rate=0.2)
        r1 = _make_result(rule_hits=["r1"])
        r2 = _make_result(rule_hits=["r2"])
        tracker.record_stage("s1", r1)
        tracker.record_stage("s2", r2)
        expected = round(1.0 * 0.8 * 0.8, 6)
        assert tracker.get_trust_score() == expected

    def test_escalation_threshold(self):
        tracker = ChainIntegrityTracker(decay_rate=0.5)
        # After one triple-decay: 1.0 * 0.5^3 = 0.125 < 0.5
        result = _make_result(
            rule_hits=["x"], anomaly_flags=["y"], risk_score=0.9,
        )
        tracker.record_stage("s1", result)
        assert tracker.should_escalate() is True

    def test_no_escalation_when_trust_high(self):
        tracker = ChainIntegrityTracker()
        assert tracker.should_escalate() is False

    def test_reset(self):
        tracker = ChainIntegrityTracker(decay_rate=0.5)
        result = _make_result(rule_hits=["x"])
        tracker.record_stage("s1", result)
        assert tracker.get_trust_score() < 1.0
        assert len(tracker.get_history()) == 1
        tracker.reset()
        assert tracker.get_trust_score() == 1.0
        assert len(tracker.get_history()) == 0

    def test_history_records(self):
        tracker = ChainIntegrityTracker()
        r1 = _make_result(risk_score=0.1)
        r2 = _make_result(rule_hits=["x"], risk_score=0.8)
        tracker.record_stage("clean_stage", r1)
        tracker.record_stage("suspicious_stage", r2)
        history = tracker.get_history()
        assert len(history) == 2
        assert history[0]["stage_name"] == "clean_stage"
        assert history[0]["decay_reasons"] == []
        assert history[1]["stage_name"] == "suspicious_stage"
        assert "rule_hits" in history[1]["decay_reasons"]
        assert "high_risk_score" in history[1]["decay_reasons"]

    def test_invalid_initial_trust(self):
        with pytest.raises(ValueError):
            ChainIntegrityTracker(initial_trust=1.5)

    def test_invalid_decay_rate(self):
        with pytest.raises(ValueError):
            ChainIntegrityTracker(decay_rate=-0.1)


# ====================================================================
# RRF Fusion
# ====================================================================

class TestRRFScore:
    """Tests for Reciprocal Rank Fusion score computation."""

    def test_empty_signals(self):
        assert rrf_score({}) == 0.0

    def test_single_signal(self):
        score = rrf_score({"ml": 0.9})
        # Single signal → rank 1 → 1/(k+1) / (1/(k+1)) = 1.0
        assert score == 1.0

    def test_two_equal_signals(self):
        score = rrf_score({"ml": 0.5, "rules": 0.5})
        # Both tied: ranks assigned by sort order, but sum is the same
        # raw = 1/(61) + 1/(62) = 0.01639 + 0.01613 = 0.03252
        # max = 2 * 1/61 = 0.03279
        # normalized = 0.03252 / 0.03279 ≈ 0.991...
        assert 0.98 < score <= 1.0

    def test_score_normalized_to_01(self):
        score = rrf_score({"a": 0.1, "b": 0.5, "c": 0.9})
        assert 0.0 <= score <= 1.0

    def test_more_signals_still_bounded(self):
        signals = {f"s{i}": i * 0.1 for i in range(10)}
        score = rrf_score(signals)
        assert 0.0 <= score <= 1.0

    def test_custom_k(self):
        signals = {"ml": 0.9, "rules": 0.3}
        score_k10 = rrf_score(signals, k=10)
        score_k100 = rrf_score(signals, k=100)
        # Both should be valid but potentially different
        assert 0.0 <= score_k10 <= 1.0
        assert 0.0 <= score_k100 <= 1.0

    def test_rank_ordering(self):
        """Higher signal values should get better (lower) ranks."""
        # With a big gap, the score should be close to 1 because
        # all signals contribute
        score_high = rrf_score({"ml": 0.99, "rules": 0.98})
        score_low = rrf_score({"ml": 0.01, "rules": 0.02})
        # Both should produce similar RRF scores because RRF is
        # rank-based, not magnitude-based
        assert abs(score_high - score_low) < 0.05

    def test_rrf_is_rank_invariant(self):
        """RRF score depends only on ranks, not magnitudes."""
        s1 = rrf_score({"a": 100.0, "b": 50.0, "c": 1.0})
        s2 = rrf_score({"a": 0.9, "b": 0.5, "c": 0.1})
        assert s1 == s2


class TestRRFDecision:
    """Tests for RRF-based classification decisions."""

    def test_malicious_above_threshold(self):
        # Single strong signal → score = 1.0 > 0.55
        label, conf = rrf_decision({"ml": 0.95}, threshold=0.55)
        assert label == "MALICIOUS"
        assert conf >= 0.55

    def test_safe_below_threshold(self):
        # Need a case where RRF < threshold.
        # With default k=60 and threshold=0.55, any non-empty signal
        # produces score >= ~0.98 for single signals.
        # Use a very high threshold to test the SAFE path.
        label, conf = rrf_decision({"ml": 0.1}, threshold=1.1)
        assert label == "SAFE"

    def test_custom_threshold(self):
        label, conf = rrf_decision({"ml": 0.5}, threshold=0.5)
        assert label == "MALICIOUS"  # single signal → score 1.0 > 0.5

    def test_returns_tuple(self):
        result = rrf_decision({"ml": 0.5, "rules": 0.3})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_empty_signals_safe(self):
        label, conf = rrf_decision({})
        assert label == "SAFE"
        assert conf == 0.0


# ====================================================================
# Groundedness Check
# ====================================================================

class TestGroundedness:
    """Tests for Self-RAG groundedness verification."""

    def test_multi_source_grounded(self):
        result = _make_result(
            is_malicious=True,
            label="malicious",
            ml_confidence=0.85,
            rule_hits=["role_override"],
            anomaly_flags=["null_bytes"],
        )
        gcheck = verify_verdict_grounded(result)
        assert gcheck["grounded"] is True
        assert gcheck["source_count"] >= 2
        assert "ml" in gcheck["sources"]
        assert "rules" in gcheck["sources"]
        assert gcheck["recommendation"] == "confirmed"

    def test_single_source_not_grounded(self):
        result = _make_result(
            is_malicious=True,
            label="malicious",
            ml_confidence=0.85,
            rule_hits=[],
            anomaly_flags=[],
        )
        gcheck = verify_verdict_grounded(result)
        assert gcheck["grounded"] is False
        assert gcheck["source_count"] == 1
        assert gcheck["recommendation"] == "review"

    def test_no_sources(self):
        result = _make_result(
            is_malicious=True,
            label="malicious",
            ml_confidence=0.3,  # below 0.6 threshold
        )
        gcheck = verify_verdict_grounded(result)
        assert gcheck["grounded"] is False
        assert gcheck["source_count"] == 0
        assert gcheck["recommendation"] == "review"

    def test_embedding_counts_as_source(self):
        result = _make_result(
            is_malicious=True,
            label="malicious",
            ml_confidence=0.85,
            embedding_score=0.15,
        )
        gcheck = verify_verdict_grounded(result)
        assert "embedding" in gcheck["sources"]
        assert gcheck["grounded"] is True

    def test_technique_tags_count_as_source(self):
        result = _make_result(
            is_malicious=True,
            label="malicious",
            ml_confidence=0.85,
            technique_tags=["T0001"],
        )
        gcheck = verify_verdict_grounded(result)
        assert "techniques" in gcheck["sources"]
        assert gcheck["grounded"] is True

    def test_cascade_tags_excluded(self):
        """cascade:* tags should NOT count as technique evidence."""
        result = _make_result(
            is_malicious=True,
            label="malicious",
            ml_confidence=0.3,
            technique_tags=["cascade:weighted"],
        )
        gcheck = verify_verdict_grounded(result)
        assert "techniques" not in gcheck["sources"]

    def test_safe_verdict_always_confirmed(self):
        result = _make_result(is_malicious=False, label="safe")
        gcheck = verify_verdict_grounded(result)
        assert gcheck["recommendation"] == "confirmed"

    def test_custom_min_sources(self):
        result = _make_result(
            is_malicious=True,
            label="malicious",
            ml_confidence=0.85,
            rule_hits=["x"],
        )
        # With min_sources=3, two sources is not enough
        gcheck = verify_verdict_grounded(result, min_sources=3)
        assert gcheck["grounded"] is False
        assert gcheck["recommendation"] == "review"

    def test_all_five_sources(self):
        result = _make_result(
            is_malicious=True,
            label="malicious",
            ml_confidence=0.9,
            rule_hits=["role_override"],
            anomaly_flags=["null_bytes"],
            embedding_score=0.15,
            technique_tags=["T0001", "T0002"],
        )
        gcheck = verify_verdict_grounded(result)
        assert gcheck["source_count"] == 5
        assert gcheck["grounded"] is True
        assert set(gcheck["sources"]) == {"ml", "rules", "anomaly", "embedding", "techniques"}
