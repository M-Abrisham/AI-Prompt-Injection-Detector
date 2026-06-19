"""Security-focused tests for embedding_drift detector.

Covers:
- NaN / Inf / zero-vector handling in _cosine_similarity
- Thread-safe double-checked locking on model cache
"""

from __future__ import annotations

import math
import threading
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Cosine similarity edge cases
# ---------------------------------------------------------------------------

from na0s.layer16.detectors.embedding_drift import (
    EmbeddingDriftDetector,
    _cosine_similarity,
)


class TestCosineSimilarityNaN:
    """MEDIUM-5: NaN inputs must not propagate."""

    def test_nan_in_first_vector(self) -> None:
        result = _cosine_similarity([float("nan"), 1.0], [1.0, 1.0])
        assert result == 0.0

    def test_nan_in_second_vector(self) -> None:
        result = _cosine_similarity([1.0, 1.0], [float("nan"), 1.0])
        assert result == 0.0

    def test_all_nan(self) -> None:
        result = _cosine_similarity([float("nan")], [float("nan")])
        assert result == 0.0


class TestCosineSimilarityInf:
    """Inf inputs must not propagate."""

    def test_inf_in_first_vector(self) -> None:
        result = _cosine_similarity([float("inf"), 1.0], [1.0, 1.0])
        assert result == 0.0

    def test_neg_inf_in_second_vector(self) -> None:
        result = _cosine_similarity([1.0, 1.0], [float("-inf"), 1.0])
        assert result == 0.0


class TestCosineSimilarityZeroVectors:
    """Zero-magnitude vectors must return 0.0, not raise."""

    def test_both_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_one_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# Thread-safe model loading
# ---------------------------------------------------------------------------


class TestModelLock:
    """HIGH-6: model cache race condition."""

    def test_model_lock_exists_and_is_lock(self) -> None:
        assert hasattr(EmbeddingDriftDetector, "_model_lock")
        assert isinstance(EmbeddingDriftDetector._model_lock, type(threading.Lock()))

    def test_double_checked_locking_loads_once(self) -> None:
        """Concurrent _load_model calls must create only one model instance."""
        # Reset class state
        original_model = EmbeddingDriftDetector._model
        EmbeddingDriftDetector._model = None

        fake_model = mock.MagicMock(name="FakeSentenceTransformer")
        call_count = 0
        barrier = threading.Barrier(4)

        def fake_constructor(name: str, **kwargs):
            # Accept the pinned-loader kwargs (revision/cache_folder/...) so the
            # construction succeeds on the first attempt without the loader's
            # TypeError fallback retry.
            nonlocal call_count
            call_count += 1
            return fake_model

        try:
            with mock.patch.dict(
                "na0s.layer16.detectors.embedding_drift.__builtins__",
                {},
            ):
                pass  # just checking import works
        except Exception:
            pass

        try:
            # _load_model now constructs via the shared pinned loader, which
            # uses the module-scope ``SentenceTransformer`` reference.  Patch
            # that reference so the pinned-loader call is intercepted; the
            # double-checked-locking contract (exactly one construction across
            # 4 threads) is unchanged.
            fake_st_class = mock.MagicMock(side_effect=fake_constructor)
            with mock.patch(
                "na0s.layer16.detectors.embedding_drift._HAS_EMBEDDINGS", True
            ), mock.patch(
                "na0s.layer16.detectors.embedding_drift.SentenceTransformer",
                fake_st_class,
                create=True,
            ):
                errors: list = []

                def worker() -> None:
                    try:
                        barrier.wait(timeout=5)
                        EmbeddingDriftDetector._load_model()
                    except Exception as exc:
                        errors.append(exc)

                threads = [threading.Thread(target=worker) for _ in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=10)

                assert not errors, f"Thread errors: {errors}"
                # The model constructor should have been called exactly once
                assert fake_st_class.call_count == 1
                assert EmbeddingDriftDetector._model is fake_model
        finally:
            EmbeddingDriftDetector._model = original_model
