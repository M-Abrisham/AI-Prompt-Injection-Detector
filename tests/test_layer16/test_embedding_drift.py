"""Tests for EmbeddingDriftDetector (D1.23).

All tests mock the embedding computation so sentence-transformers is NOT
required in the test environment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import patch

import pytest

from na0s.layer16.detectors.embedding_drift import (
    EmbeddingDriftDetector,
    _cosine_similarity,
)
from na0s.layer16.models import ConversationState, ConversationTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(texts: List[str], session_id: str = "test") -> ConversationState:
    """Build a ConversationState with the given turn texts."""
    turns = [
        ConversationTurn(
            turn_id=str(i),
            text=t,
            role="user",
            timestamp=datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc),
        )
        for i, t in enumerate(texts)
    ]
    return ConversationState(session_id=session_id, turns=turns)


def _mock_embeddings(vectors: List[List[float]]):
    """Return a patcher that makes _get_embeddings return *vectors*."""
    def side_effect(self, texts):
        # Return as many vectors as texts requested (trim if needed)
        return vectors[: len(texts)]

    return patch.object(EmbeddingDriftDetector, "_get_embeddings", side_effect)


# Convenient embedding vectors for testing.
# Unit vectors along different axes give cosine similarity = 0.0
# Identical vectors give cosine similarity = 1.0
_VEC_A = [1.0, 0.0, 0.0]
_VEC_B = [0.0, 1.0, 0.0]
_VEC_C = [0.0, 0.0, 1.0]
_VEC_SIMILAR_A = [0.95, 0.05, 0.0]  # very similar to _VEC_A


# ---------------------------------------------------------------------------
# Tests: Interface
# ---------------------------------------------------------------------------


class TestDetectorInterface:
    def test_detector_name(self):
        d = EmbeddingDriftDetector()
        assert d.detector_name == "embedding_drift"

    def test_taxonomy_ids(self):
        d = EmbeddingDriftDetector()
        assert d.taxonomy_ids == ["D1.23"]

    def test_reset(self):
        d = EmbeddingDriftDetector()
        d.reset()  # should not raise


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_none_state(self):
        d = EmbeddingDriftDetector()
        assert d.analyze(None) == []

    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_empty_state(self):
        d = EmbeddingDriftDetector()
        state = ConversationState(session_id="empty")
        assert d.analyze(state) == []

    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_single_turn_no_alert(self):
        d = EmbeddingDriftDetector()
        state = _make_state(["hello"])
        assert d.analyze(state) == []

    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_two_turns_below_min(self):
        d = EmbeddingDriftDetector()
        state = _make_state(["hello", "world"])
        assert d.analyze(state) == []


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", False)
    def test_no_embeddings_returns_empty(self):
        d = EmbeddingDriftDetector()
        state = _make_state(["a", "b", "c", "d", "e"])
        assert d.analyze(state) == []

    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", False)
    def test_no_crash_without_embeddings(self):
        d = EmbeddingDriftDetector()
        state = _make_state(["a", "b", "c"])
        # Must not raise
        result = d.analyze(state)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: Feature flag disabled
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    @patch("na0s.layer16.detectors.embedding_drift.ENABLE_EMBEDDING_DRIFT", False)
    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_disabled_returns_empty(self):
        d = EmbeddingDriftDetector()
        state = _make_state(["a", "b", "c", "d", "e"])
        with _mock_embeddings([_VEC_A, _VEC_A, _VEC_A, _VEC_B, _VEC_C]):
            assert d.analyze(state) == []


# ---------------------------------------------------------------------------
# Tests: Sharp pivot detection
# ---------------------------------------------------------------------------


class TestSharpPivot:
    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_sharp_pivot_fires(self):
        """High similarity then orthogonal vector -> alert."""
        d = EmbeddingDriftDetector()
        state = _make_state(["t1", "t2", "t3"])
        # First two similar, third orthogonal -> cos sim ~0
        vecs = [_VEC_A, _VEC_SIMILAR_A, _VEC_B]
        with _mock_embeddings(vecs):
            alerts = d.analyze(state)
        # Should have at least a sharp pivot alert
        sharp = [a for a in alerts if "sharp_pivot" in str(a.evidence)]
        assert len(sharp) == 1
        assert sharp[0].alert_type == "embedding_drift"
        assert sharp[0].confidence >= 0.3

    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_no_pivot_when_all_similar(self):
        """All similar vectors -> no sharp pivot alert."""
        d = EmbeddingDriftDetector()
        state = _make_state(["t1", "t2", "t3"])
        # All identical -> cosine sim = 1.0
        vecs = [_VEC_A, _VEC_A, _VEC_A]
        with _mock_embeddings(vecs):
            alerts = d.analyze(state)
        sharp = [a for a in alerts if "sharp_pivot" in str(a.evidence)]
        assert len(sharp) == 0


# ---------------------------------------------------------------------------
# Tests: Sustained drift detection
# ---------------------------------------------------------------------------


class TestSustainedDrift:
    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_sustained_drift_fires(self):
        """All consecutive pairs orthogonal -> avg sim ~0 -> alert."""
        d = EmbeddingDriftDetector()
        state = _make_state(["t1", "t2", "t3", "t4"])
        # Alternating orthogonal vectors
        vecs = [_VEC_A, _VEC_B, _VEC_C, _VEC_A]
        with _mock_embeddings(vecs):
            alerts = d.analyze(state)
        sustained = [a for a in alerts if "sustained_drift" in str(a.evidence)]
        assert len(sustained) == 1
        assert sustained[0].alert_type == "embedding_drift"

    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_no_sustained_drift_when_similar(self):
        """All identical -> avg sim = 1.0 -> no sustained drift."""
        d = EmbeddingDriftDetector()
        state = _make_state(["t1", "t2", "t3", "t4"])
        vecs = [_VEC_A, _VEC_A, _VEC_A, _VEC_A]
        with _mock_embeddings(vecs):
            alerts = d.analyze(state)
        sustained = [a for a in alerts if "sustained_drift" in str(a.evidence)]
        assert len(sustained) == 0


# ---------------------------------------------------------------------------
# Tests: Alert structure
# ---------------------------------------------------------------------------


class TestAlertStructure:
    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_alert_type_and_evidence(self):
        d = EmbeddingDriftDetector()
        state = _make_state(["t1", "t2", "t3"])
        vecs = [_VEC_A, _VEC_A, _VEC_B]  # last pair orthogonal
        with _mock_embeddings(vecs):
            alerts = d.analyze(state)
        assert len(alerts) >= 1
        for alert in alerts:
            assert alert.alert_type == "embedding_drift"
            assert alert.severity in ("low", "medium", "high", "critical")
            assert 0.0 <= alert.confidence <= 1.0
            # Evidence should contain similarity values
            evidence_str = " ".join(alert.evidence)
            assert "similarity" in evidence_str.lower()

    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_turn_range_set(self):
        d = EmbeddingDriftDetector()
        state = _make_state(["t1", "t2", "t3"])
        vecs = [_VEC_A, _VEC_A, _VEC_B]
        with _mock_embeddings(vecs):
            alerts = d.analyze(state)
        for alert in alerts:
            assert isinstance(alert.turn_range, tuple)
            assert len(alert.turn_range) == 2


# ---------------------------------------------------------------------------
# Tests: Confidence calculation
# ---------------------------------------------------------------------------


class TestConfidence:
    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_higher_drift_higher_confidence(self):
        """Orthogonal pair (sim=0) should have higher confidence than
        a pair with sim=0.2."""
        d = EmbeddingDriftDetector()

        # Complete orthogonal -> sim = 0
        state1 = _make_state(["t1", "t2", "t3"])
        vecs1 = [_VEC_A, _VEC_A, _VEC_B]
        with _mock_embeddings(vecs1):
            alerts1 = d.analyze(state1)

        # Partially similar -> sim ~0.2
        partially = [0.2, 0.9, 0.0]  # sim with _VEC_A ~ 0.2
        state2 = _make_state(["t1", "t2", "t3"])
        vecs2 = [_VEC_A, _VEC_A, partially]
        with _mock_embeddings(vecs2):
            alerts2 = d.analyze(state2)

        sharp1 = [a for a in alerts1 if "sharp_pivot" in str(a.evidence)]
        sharp2 = [a for a in alerts2 if "sharp_pivot" in str(a.evidence)]

        assert len(sharp1) >= 1
        assert len(sharp2) >= 1
        assert sharp1[0].confidence >= sharp2[0].confidence


# ---------------------------------------------------------------------------
# Tests: Window configuration
# ---------------------------------------------------------------------------


class TestWindow:
    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    @patch("na0s.layer16.detectors.embedding_drift.DRIFT_WINDOW", 3)
    def test_window_limits_turns_examined(self):
        """With DRIFT_WINDOW=3, only last 3 turns should be examined."""
        d = EmbeddingDriftDetector()
        # 5 turns total: first 2 orthogonal, last 3 identical
        state = _make_state(["t1", "t2", "t3", "t4", "t5"])
        # We mock _get_embeddings; it receives only the windowed texts
        call_args = []

        def capture_get_embeddings(self_, texts):
            call_args.append(texts)
            # Return identical vectors for all
            return [_VEC_A] * len(texts)

        with patch.object(
            EmbeddingDriftDetector, "_get_embeddings", capture_get_embeddings
        ):
            alerts = d.analyze(state)

        # Should have been called with 3 texts (last 3 turns)
        assert len(call_args) == 1
        assert len(call_args[0]) == 3
        # All identical -> no alerts
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Tests: Cosine similarity edge cases
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0

    def test_both_zero_vectors(self):
        assert _cosine_similarity([0, 0], [0, 0]) == 0.0


# ---------------------------------------------------------------------------
# Tests: _get_embeddings returns None gracefully
# ---------------------------------------------------------------------------


class TestGetEmbeddingsNone:
    @patch("na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True)
    def test_embeddings_none_returns_empty(self):
        """If _get_embeddings returns None, analyze returns []."""
        d = EmbeddingDriftDetector()
        state = _make_state(["a", "b", "c"])
        with patch.object(
            EmbeddingDriftDetector, "_get_embeddings", return_value=None
        ):
            assert d.analyze(state) == []
