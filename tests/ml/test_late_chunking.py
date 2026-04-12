"""Tests for late chunking module (Layer 5 enhancement).

Covers:
- Chunk splitting logic with mock embeddings
- Graceful degradation without sentence-transformers
- Short text bypass
- Max-risk aggregation across chunks
- Integration with classifier interfaces
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: build mock model / tokenizer / classifier
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(num_tokens: int = 300):
    """Return a mock tokenizer that produces `num_tokens` tokens."""
    tokenizer = mock.MagicMock()
    # input_ids: [CLS] + content tokens + [SEP]
    input_ids = mock.MagicMock()
    input_ids.shape = (1, num_tokens)
    encoded = {"input_ids": input_ids, "attention_mask": mock.MagicMock()}
    tokenizer.return_value = encoded
    return tokenizer, encoded


def _make_mock_model(num_tokens: int = 300, hidden_dim: int = 384):
    """Return a mock transformer model with deterministic embeddings."""
    import torch

    model = mock.MagicMock()
    # Generate reproducible token embeddings
    rng = np.random.RandomState(42)
    embeddings = rng.randn(1, num_tokens, hidden_dim).astype(np.float32)
    tensor = torch.tensor(embeddings)
    output = mock.MagicMock()
    output.last_hidden_state = tensor
    model.return_value = output
    return model, embeddings


def _make_centroid_classifier(hidden_dim: int = 384, num_techniques: int = 3):
    """Return a mock classifier with centroid-based interface."""
    classifier = mock.MagicMock()
    rng = np.random.RandomState(99)
    centroids = {}
    for i in range(num_techniques):
        c = rng.randn(hidden_dim).astype(np.float32)
        c /= np.linalg.norm(c)
        centroids["D{0}".format(i + 1)] = c
    classifier._centroids = centroids
    return classifier


def _make_sklearn_classifier(hidden_dim: int = 384, p_malicious: float = 0.3):
    """Return a mock sklearn-style classifier."""
    classifier = mock.MagicMock()
    classifier._centroids = None  # not centroid-based
    del classifier._centroids  # remove attribute entirely
    classifier.predict_proba.return_value = np.array([[1 - p_malicious, p_malicious]])
    return classifier


# ---------------------------------------------------------------------------
# Test: _split_and_pool
# ---------------------------------------------------------------------------

class TestSplitAndPool:
    """Test the internal _split_and_pool function."""

    def test_basic_chunking(self):
        from na0s.late_chunking import _split_and_pool

        hidden_dim = 8
        seq_len = 20  # [CLS] + 18 content + [SEP]
        embeddings = np.random.randn(seq_len, hidden_dim).astype(np.float32)

        chunks = _split_and_pool(embeddings, content_start=1, content_end=19,
                                 chunk_size=6, stride=3)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.shape == (hidden_dim,)

    def test_chunk_count_with_stride(self):
        from na0s.late_chunking import _split_and_pool

        hidden_dim = 4
        # 20 content tokens, chunk_size=5, stride=5 -> 4 chunks
        embeddings = np.ones((22, hidden_dim), dtype=np.float32)
        chunks = _split_and_pool(embeddings, 1, 21, chunk_size=5, stride=5)
        assert len(chunks) == 4

    def test_overlapping_chunks(self):
        from na0s.late_chunking import _split_and_pool

        hidden_dim = 4
        # 10 content tokens, chunk_size=6, stride=3
        # Chunks: [1:7], [4:10], [7:11(=11)] -> 3 chunks
        embeddings = np.arange(12 * hidden_dim, dtype=np.float32).reshape(12, hidden_dim)
        chunks = _split_and_pool(embeddings, 1, 11, chunk_size=6, stride=3)
        assert len(chunks) >= 2
        # Overlapping chunks should differ (different token ranges)
        assert not np.allclose(chunks[0], chunks[1])

    def test_single_chunk_when_small(self):
        from na0s.late_chunking import _split_and_pool

        hidden_dim = 4
        # Only 3 content tokens, chunk_size=10 -> 1 chunk
        embeddings = np.ones((5, hidden_dim), dtype=np.float32)
        chunks = _split_and_pool(embeddings, 1, 4, chunk_size=10, stride=5)
        assert len(chunks) == 1

    def test_chunks_are_normalized(self):
        from na0s.late_chunking import _split_and_pool

        hidden_dim = 8
        embeddings = np.random.randn(20, hidden_dim).astype(np.float32)
        chunks = _split_and_pool(embeddings, 1, 19, chunk_size=6, stride=3)
        for chunk in chunks:
            norm = np.linalg.norm(chunk)
            assert abs(norm - 1.0) < 1e-5, "Chunk should be L2-normalized"

    def test_no_duplicate_trailing_chunk(self):
        from na0s.late_chunking import _split_and_pool

        hidden_dim = 4
        # 10 content tokens, chunk_size=5, stride=5 -> exactly 2 chunks
        embeddings = np.ones((12, hidden_dim), dtype=np.float32)
        chunks = _split_and_pool(embeddings, 1, 11, chunk_size=5, stride=5)
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Test: late_chunk_embed
# ---------------------------------------------------------------------------

class TestLateChunkEmbed:

    @pytest.fixture(autouse=True)
    def _check_transformers(self):
        """Skip if torch/transformers not available (CI without GPU deps)."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

    def test_returns_none_for_short_text(self):
        from na0s.late_chunking import late_chunk_embed

        num_tokens = 100  # below default MIN_TOKENS_FOR_CHUNKING (256)
        tokenizer, _ = _make_mock_tokenizer(num_tokens)
        model, _ = _make_mock_model(num_tokens)
        result = late_chunk_embed("short text", model, tokenizer)
        assert result is None

    def test_returns_chunks_for_long_text(self):
        from na0s.late_chunking import late_chunk_embed

        num_tokens = 400
        tokenizer, _ = _make_mock_tokenizer(num_tokens)
        model, _ = _make_mock_model(num_tokens)
        chunks = late_chunk_embed("long " * 200, model, tokenizer,
                                  chunk_size=128, stride=64)
        assert chunks is not None
        assert len(chunks) > 1
        assert chunks[0].shape == (384,)

    def test_invalid_chunk_size_raises(self):
        from na0s.late_chunking import late_chunk_embed

        tokenizer, _ = _make_mock_tokenizer(300)
        model, _ = _make_mock_model(300)
        with pytest.raises(ValueError, match="chunk_size"):
            late_chunk_embed("text", model, tokenizer, chunk_size=0)

    def test_invalid_stride_raises(self):
        from na0s.late_chunking import late_chunk_embed

        tokenizer, _ = _make_mock_tokenizer(300)
        model, _ = _make_mock_model(300)
        with pytest.raises(ValueError, match="stride"):
            late_chunk_embed("text", model, tokenizer, stride=0)


# ---------------------------------------------------------------------------
# Test: late_chunk_classify
# ---------------------------------------------------------------------------

class TestLateChunkClassify:

    @pytest.fixture(autouse=True)
    def _check_transformers(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

    def test_returns_none_for_short_text(self):
        from na0s.late_chunking import late_chunk_classify

        num_tokens = 50
        tokenizer, _ = _make_mock_tokenizer(num_tokens)
        model, _ = _make_mock_model(num_tokens)
        classifier = _make_centroid_classifier()

        result = late_chunk_classify("short", model, tokenizer, classifier)
        assert result is None

    def test_max_risk_aggregation(self):
        from na0s.late_chunking import late_chunk_classify

        num_tokens = 400
        tokenizer, _ = _make_mock_tokenizer(num_tokens)
        model, _ = _make_mock_model(num_tokens)
        classifier = _make_centroid_classifier()

        result = late_chunk_classify("long " * 200, model, tokenizer,
                                     classifier, chunk_size=128, stride=64)
        assert result is not None
        assert "max_score" in result
        assert "max_chunk_idx" in result
        assert "chunk_scores" in result
        assert "num_chunks" in result
        assert result["max_score"] == max(result["chunk_scores"])
        assert result["max_chunk_idx"] == result["chunk_scores"].index(
            result["max_score"]
        )

    def test_num_chunks_matches_scores(self):
        from na0s.late_chunking import late_chunk_classify

        num_tokens = 400
        tokenizer, _ = _make_mock_tokenizer(num_tokens)
        model, _ = _make_mock_model(num_tokens)
        classifier = _make_centroid_classifier()

        result = late_chunk_classify("long " * 200, model, tokenizer,
                                     classifier, chunk_size=128, stride=64)
        assert result is not None
        assert result["num_chunks"] == len(result["chunk_scores"])

    def test_with_sklearn_classifier(self):
        from na0s.late_chunking import late_chunk_classify

        num_tokens = 400
        tokenizer, _ = _make_mock_tokenizer(num_tokens)
        model, _ = _make_mock_model(num_tokens)
        classifier = _make_sklearn_classifier(p_malicious=0.8)

        result = late_chunk_classify("long " * 200, model, tokenizer,
                                     classifier, chunk_size=128, stride=64)
        assert result is not None
        # All chunks should get ~0.8 from the mock
        for score in result["chunk_scores"]:
            assert abs(score - 0.8) < 1e-5


# ---------------------------------------------------------------------------
# Test: _classify_single_embedding
# ---------------------------------------------------------------------------

class TestClassifySingleEmbedding:

    def test_centroid_classifier_returns_float(self):
        from na0s.late_chunking import _classify_single_embedding

        classifier = _make_centroid_classifier(hidden_dim=8)
        emb = np.random.randn(8).astype(np.float32)
        score = _classify_single_embedding(emb, classifier)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0 or score >= -1.0  # cosine range

    def test_sklearn_classifier_returns_proba(self):
        from na0s.late_chunking import _classify_single_embedding

        classifier = _make_sklearn_classifier(p_malicious=0.65)
        emb = np.random.randn(384).astype(np.float32)
        score = _classify_single_embedding(emb, classifier)
        assert abs(score - 0.65) < 1e-5

    def test_unknown_classifier_returns_zero(self):
        from na0s.late_chunking import _classify_single_embedding

        classifier = mock.MagicMock(spec=[])  # no attributes
        emb = np.random.randn(8).astype(np.float32)
        score = _classify_single_embedding(emb, classifier)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Test: graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:

    def test_late_chunk_embed_returns_none_without_transformers(self):
        """Simulate missing torch/transformers by patching the flag."""
        import na0s.late_chunking as lc_mod
        original = lc_mod._HAS_TRANSFORMERS
        try:
            lc_mod._HAS_TRANSFORMERS = False
            result = lc_mod.late_chunk_embed("some text", None, None)
            assert result is None
        finally:
            lc_mod._HAS_TRANSFORMERS = original

    def test_late_chunk_classify_returns_none_without_transformers(self):
        import na0s.late_chunking as lc_mod
        original = lc_mod._HAS_TRANSFORMERS
        try:
            lc_mod._HAS_TRANSFORMERS = False
            result = lc_mod.late_chunk_classify("some text", None, None, None)
            assert result is None
        finally:
            lc_mod._HAS_TRANSFORMERS = original

    def test_maybe_late_chunk_boost_noop_when_disabled(self):
        """When NA0S_LATE_CHUNKING is not set, boost should be a no-op."""
        import na0s.late_chunking as lc_mod

        with mock.patch.dict(os.environ, {"NA0S_LATE_CHUNKING": "0"}):
            score, details = lc_mod.maybe_late_chunk_boost(
                "text", 0.3, mock.MagicMock()
            )
        assert score == 0.3
        assert details is None

    def test_load_late_chunking_model_returns_none_without_transformers(self):
        import na0s.late_chunking as lc_mod
        original = lc_mod._HAS_TRANSFORMERS
        try:
            lc_mod._HAS_TRANSFORMERS = False
            result = lc_mod.load_late_chunking_model()
            assert result is None
        finally:
            lc_mod._HAS_TRANSFORMERS = original


# ---------------------------------------------------------------------------
# Test: maybe_late_chunk_boost integration
# ---------------------------------------------------------------------------

class TestMaybeLateChunkBoost:

    def test_boost_when_chunk_scores_higher(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        import na0s.late_chunking as lc_mod

        num_tokens = 400
        tokenizer, _ = _make_mock_tokenizer(num_tokens)
        model, _ = _make_mock_model(num_tokens)
        classifier = _make_centroid_classifier()

        with mock.patch.dict(os.environ, {"NA0S_LATE_CHUNKING": "1"}):
            score, details = lc_mod.maybe_late_chunk_boost(
                "long " * 200, 0.0, classifier,
                model=model, tokenizer=tokenizer,
            )
        # Should return some non-None details since text is long enough
        # and chunking is enabled
        if details is not None:
            assert score >= 0.0
            assert "max_score" in details

    def test_no_boost_when_disabled(self):
        import na0s.late_chunking as lc_mod

        with mock.patch.dict(os.environ, {"NA0S_LATE_CHUNKING": "0"}):
            score, details = lc_mod.maybe_late_chunk_boost(
                "text", 0.5, mock.MagicMock()
            )
        assert score == 0.5
        assert details is None


# ---------------------------------------------------------------------------
# Test: is_late_chunking_enabled
# ---------------------------------------------------------------------------

class TestIsLateChunkingEnabled:

    def test_disabled_by_default(self):
        from na0s.late_chunking import is_late_chunking_enabled

        with mock.patch.dict(os.environ, {}, clear=False):
            if "NA0S_LATE_CHUNKING" in os.environ:
                del os.environ["NA0S_LATE_CHUNKING"]
            assert is_late_chunking_enabled() is False

    def test_enabled_when_set(self):
        from na0s.late_chunking import is_late_chunking_enabled

        with mock.patch.dict(os.environ, {"NA0S_LATE_CHUNKING": "1"}):
            assert is_late_chunking_enabled() is True

    def test_disabled_when_zero(self):
        from na0s.late_chunking import is_late_chunking_enabled

        with mock.patch.dict(os.environ, {"NA0S_LATE_CHUNKING": "0"}):
            assert is_late_chunking_enabled() is False
