"""Behavioral tests for Layer 16 embedding drift detector (D1.23)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from na0s.layer16.detectors.embedding_drift import (
    EmbeddingDriftDetector,
    _fallback_embedding,
)
from na0s.layer16.models import ConversationState, ConversationTurn


def _make_state(texts):
    state = ConversationState(session_id="drift-test")
    for i, text in enumerate(texts):
        state.turns.append(
            ConversationTurn(
                turn_id=f"t{i}",
                text=text,
                timestamp=datetime.now(timezone.utc),
                risk_score=0.1,
            )
        )
    return state


class TestEmbeddingDriftDetector:
    def test_fallback_embedding_is_deterministic(self):
        a = _fallback_embedding("weather forecast in sf")
        b = _fallback_embedding("weather forecast in sf")
        assert a == b
        assert any(v != 0.0 for v in a)

    def test_get_embeddings_fallback_without_sentence_transformers(self, monkeypatch):
        import na0s.layer16.detectors.embedding_drift as drift

        monkeypatch.setattr(drift, "_HAS_EMBEDDINGS", False)
        d = EmbeddingDriftDetector()
        vectors = d._get_embeddings(["hello world", "different topic"])
        assert vectors is not None
        assert len(vectors) == 2
        assert len(vectors[0]) == len(vectors[1])

    def test_detects_sharp_and_baseline_shift(self):
        d = EmbeddingDriftDetector()
        state = _make_state(["a", "b", "c", "d"])

        def fake_embeddings(_texts):
            return [
                [1.0, 0.0],
                [0.95, 0.1],
                [0.9, 0.2],
                [-1.0, 0.0],
            ]

        d._get_embeddings = fake_embeddings  # type: ignore[method-assign]
        alerts = d.analyze(state)
        assert alerts
        evidence = [e for a in alerts for e in a.evidence]
        assert any("signal=sharp_pivot" in e or "signal=baseline_shift" in e for e in evidence)

    def test_no_alert_when_embeddings_stable(self):
        d = EmbeddingDriftDetector()
        state = _make_state(["a", "b", "c", "d"])

        def stable_embeddings(_texts):
            return [
                [1.0, 0.0],
                [0.99, 0.05],
                [0.98, 0.08],
                [0.97, 0.1],
            ]

        d._get_embeddings = stable_embeddings  # type: ignore[method-assign]
        assert d.analyze(state) == []

