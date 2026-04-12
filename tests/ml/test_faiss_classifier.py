"""Tests for the FAISS KNN classifier (Layer 5).

Covers:
  - Index building with synthetic embeddings
  - Classification with known neighbors
  - Graceful degradation without faiss
  - Save/load round-trip
  - L2 normalization
  - Thread safety
  - Singleton pattern
  - Edge cases (empty index, single vector, k > n)
"""

from __future__ import annotations

import os
import pickle
import threading
import tempfile
from unittest import mock

import numpy as np
import pytest

try:
    import faiss  # noqa: F401
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

requires_faiss = pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_embeddings():
    """Generate synthetic 384-dim embeddings with known structure."""
    np.random.seed(42)
    dim = 384

    # Create two clusters: malicious (near [1,0,...]) and safe (near [0,1,...])
    n_malicious = 20
    n_safe = 20

    mal_center = np.zeros(dim, dtype=np.float32)
    mal_center[0] = 1.0

    safe_center = np.zeros(dim, dtype=np.float32)
    safe_center[1] = 1.0

    mal_embeddings = mal_center + np.random.randn(n_malicious, dim).astype(np.float32) * 0.1
    safe_embeddings = safe_center + np.random.randn(n_safe, dim).astype(np.float32) * 0.1

    embeddings = np.vstack([mal_embeddings, safe_embeddings])
    labels = np.array([1] * n_malicious + [0] * n_safe, dtype=np.int64)

    return embeddings, labels, mal_center, safe_center


@pytest.fixture
def malicious_only_embeddings(sample_embeddings):
    """Return only the malicious embeddings."""
    embeddings, labels, mal_center, _ = sample_embeddings
    mask = labels == 1
    return embeddings[mask], labels[mask], mal_center


@pytest.fixture
def tmp_index_path():
    """Provide a temporary path for index files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_faiss_index.bin")


# ---------------------------------------------------------------------------
# Test: L2 normalization
# ---------------------------------------------------------------------------

class TestL2Normalize:
    def test_normalizes_2d_array(self):
        from na0s.faiss_classifier import _l2_normalize

        vectors = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        normalized = _l2_normalize(vectors)

        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_normalizes_1d_array(self):
        from na0s.faiss_classifier import _l2_normalize

        vector = np.array([3.0, 4.0], dtype=np.float32)
        normalized = _l2_normalize(vector)

        assert normalized.ndim == 2
        assert normalized.shape == (1, 2)
        np.testing.assert_allclose(np.linalg.norm(normalized), 1.0, atol=1e-6)

    def test_zero_vector_unchanged(self):
        from na0s.faiss_classifier import _l2_normalize

        vector = np.zeros((1, 10), dtype=np.float32)
        normalized = _l2_normalize(vector)

        # Zero vector should remain zero (no NaN from division)
        assert not np.any(np.isnan(normalized))
        np.testing.assert_allclose(normalized, 0.0)

    def test_already_normalized_unchanged(self):
        from na0s.faiss_classifier import _l2_normalize

        vector = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        normalized = _l2_normalize(vector)

        np.testing.assert_allclose(normalized, vector, atol=1e-6)

    def test_preserves_direction(self):
        from na0s.faiss_classifier import _l2_normalize

        vector = np.array([[2.0, 2.0]], dtype=np.float32)
        normalized = _l2_normalize(vector)

        # Direction should be preserved (both components equal)
        assert abs(normalized[0, 0] - normalized[0, 1]) < 1e-6


# ---------------------------------------------------------------------------
# Test: Index building
# ---------------------------------------------------------------------------

@requires_faiss
class TestBuildIndex:
    def test_build_index_basic(self, malicious_only_embeddings):
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf = FAISSClassifier(k=5, threshold=0.6)
        clf.build_index(embeddings, labels)

        assert clf._index is not None
        assert clf._index.ntotal == len(embeddings)
        assert clf._loaded is True

    def test_build_index_sets_labels(self, malicious_only_embeddings):
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf = FAISSClassifier()
        clf.build_index(embeddings, labels)

        np.testing.assert_array_equal(clf._labels, labels)

    def test_build_index_normalizes_embeddings(self):
        """Verify that build_index normalizes input embeddings."""
        from na0s.faiss_classifier import FAISSClassifier

        # Create an unnormalized vector
        embeddings = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)

        clf = FAISSClassifier(k=1, threshold=0.0)
        clf.build_index(embeddings, labels)

        # Query with the same vector (will be normalized on query too)
        result = clf.classify(embeddings[0])

        # Inner product of two identical normalized vectors should be ~1.0
        assert result["max_similarity"] > 0.99

    def test_build_index_without_faiss_raises(self):
        """When faiss is not importable, build_index should raise."""
        with mock.patch.dict("sys.modules", {"faiss": None}):
            # We need to reload the module to pick up the mock
            import importlib
            import na0s.faiss_classifier as fc_mod

            original_flag = fc_mod._HAS_FAISS
            fc_mod._HAS_FAISS = False
            try:
                clf = fc_mod.FAISSClassifier()
                with pytest.raises(RuntimeError, match="faiss-cpu is required"):
                    clf.build_index(
                        np.ones((5, 10), dtype=np.float32),
                        np.ones(5, dtype=np.int64),
                    )
            finally:
                fc_mod._HAS_FAISS = original_flag


# ---------------------------------------------------------------------------
# Test: Classification
# ---------------------------------------------------------------------------

@requires_faiss
class TestClassify:
    def test_classify_near_malicious(self, malicious_only_embeddings):
        """A vector near malicious cluster should score high."""
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, mal_center = malicious_only_embeddings
        clf = FAISSClassifier(k=5, threshold=0.5)
        clf.build_index(embeddings, labels)

        result = clf.classify(mal_center)
        assert result["score"] > 0.0
        assert result["max_similarity"] > 0.5
        assert result["neighbors_within_threshold"] > 0

    def test_classify_far_from_malicious(self, malicious_only_embeddings):
        """A vector far from all malicious should score low."""
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf = FAISSClassifier(k=5, threshold=0.9)
        clf.build_index(embeddings, labels)

        # Create a vector orthogonal to the malicious cluster
        far_vector = np.zeros(384, dtype=np.float32)
        far_vector[200] = 1.0  # Different dimension

        result = clf.classify(far_vector)
        assert result["score"] == 0.0
        assert result["label"] == "SAFE"

    def test_classify_returns_correct_structure(self, malicious_only_embeddings):
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, mal_center = malicious_only_embeddings
        clf = FAISSClassifier(k=5, threshold=0.6)
        clf.build_index(embeddings, labels)

        result = clf.classify(mal_center)

        assert "score" in result
        assert "label" in result
        assert "k" in result
        assert "neighbors_within_threshold" in result
        assert "max_similarity" in result
        assert "mean_similarity" in result
        assert result["k"] == 5
        assert result["label"] in ("SAFE", "MALICIOUS")

    def test_classify_unloaded_returns_safe(self):
        """Unloaded classifier should gracefully return SAFE."""
        from na0s.faiss_classifier import FAISSClassifier

        clf = FAISSClassifier(index_path="/nonexistent/path.bin")
        result = clf.classify(np.ones(384, dtype=np.float32))

        assert result["score"] == 0.0
        assert result["label"] == "SAFE"

    def test_classify_k_larger_than_index(self):
        """When k > number of vectors, should clamp to index size."""
        from na0s.faiss_classifier import FAISSClassifier

        embeddings = np.random.randn(3, 10).astype(np.float32)
        labels = np.ones(3, dtype=np.int64)

        clf = FAISSClassifier(k=10, threshold=0.0)
        clf.build_index(embeddings, labels)

        result = clf.classify(embeddings[0])
        # Should not error, k is clamped
        assert result["neighbors_within_threshold"] <= 3

    def test_classify_single_vector_index(self):
        """Index with a single vector should work."""
        from na0s.faiss_classifier import FAISSClassifier

        embedding = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)

        clf = FAISSClassifier(k=5, threshold=0.5)
        clf.build_index(embedding, labels)

        result = clf.classify(embedding[0])
        assert result["max_similarity"] > 0.99
        assert result["neighbors_within_threshold"] == 1


# ---------------------------------------------------------------------------
# Test: Save/Load round-trip
# ---------------------------------------------------------------------------

@requires_faiss
class TestSaveLoad:
    def test_save_load_roundtrip(self, malicious_only_embeddings, tmp_index_path):
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, mal_center = malicious_only_embeddings

        # Build and save
        clf1 = FAISSClassifier(k=5, threshold=0.6)
        clf1.build_index(embeddings, labels)
        clf1.save(tmp_index_path)

        # Load into new instance
        clf2 = FAISSClassifier(k=5, threshold=0.6)
        clf2.load(tmp_index_path)

        # Classify should produce same results
        result1 = clf1.classify(mal_center)
        result2 = clf2.classify(mal_center)

        assert result1["score"] == result2["score"]
        assert result1["max_similarity"] == pytest.approx(result2["max_similarity"], abs=1e-6)
        assert result1["neighbors_within_threshold"] == result2["neighbors_within_threshold"]

    def test_save_creates_files(self, malicious_only_embeddings, tmp_index_path):
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf = FAISSClassifier()
        clf.build_index(embeddings, labels)
        clf.save(tmp_index_path)

        assert os.path.exists(tmp_index_path)
        assert os.path.exists(tmp_index_path + ".labels.pkl")

    def test_save_without_index_raises(self, tmp_index_path):
        from na0s.faiss_classifier import FAISSClassifier

        clf = FAISSClassifier()
        with pytest.raises(RuntimeError, match="No index to save"):
            clf.save(tmp_index_path)

    def test_load_labels_match(self, malicious_only_embeddings, tmp_index_path):
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf1 = FAISSClassifier()
        clf1.build_index(embeddings, labels)
        clf1.save(tmp_index_path)

        clf2 = FAISSClassifier()
        clf2.load(tmp_index_path)

        np.testing.assert_array_equal(clf2._labels, labels)


# ---------------------------------------------------------------------------
# Test: Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_is_available_false_without_faiss(self):
        import na0s.faiss_classifier as fc_mod

        original = fc_mod._HAS_FAISS
        fc_mod._HAS_FAISS = False
        try:
            assert fc_mod.FAISSClassifier.is_available() is False
        finally:
            fc_mod._HAS_FAISS = original

    def test_is_available_false_without_index_file(self):
        from na0s.faiss_classifier import FAISSClassifier

        assert FAISSClassifier.is_available("/nonexistent/path.bin") is False

    @requires_faiss
    def test_is_available_true_with_index(self, malicious_only_embeddings, tmp_index_path):
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf = FAISSClassifier()
        clf.build_index(embeddings, labels)
        clf.save(tmp_index_path)

        assert FAISSClassifier.is_available(tmp_index_path) is True

    def test_classify_returns_safe_when_unavailable(self):
        import na0s.faiss_classifier as fc_mod

        original = fc_mod._HAS_FAISS
        fc_mod._HAS_FAISS = False
        try:
            clf = fc_mod.FAISSClassifier(index_path="/nonexistent/path.bin")
            result = clf.classify(np.ones(10, dtype=np.float32))
            assert result["score"] == 0.0
            assert result["label"] == "SAFE"
        finally:
            fc_mod._HAS_FAISS = original


# ---------------------------------------------------------------------------
# Test: Thread safety
# ---------------------------------------------------------------------------

@requires_faiss
class TestThreadSafety:
    def test_concurrent_classify(self, malicious_only_embeddings):
        """Multiple threads classifying concurrently should not crash."""
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf = FAISSClassifier(k=5, threshold=0.5)
        clf.build_index(embeddings, labels)

        results = []
        errors = []

        def classify_worker(vec):
            try:
                r = clf.classify(vec)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            vec = np.random.randn(384).astype(np.float32)
            t = threading.Thread(target=classify_worker, args=(vec,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    def test_concurrent_lazy_load(self, malicious_only_embeddings, tmp_index_path):
        """Multiple threads triggering lazy load simultaneously should be safe."""
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings

        # Save an index so lazy load can find it
        clf_save = FAISSClassifier()
        clf_save.build_index(embeddings, labels)
        clf_save.save(tmp_index_path)

        # Create a fresh classifier that will lazy-load
        clf = FAISSClassifier(index_path=tmp_index_path, k=5, threshold=0.5)

        results = []
        errors = []

        def load_and_classify(vec):
            try:
                r = clf.classify(vec)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            vec = np.random.randn(384).astype(np.float32)
            t = threading.Thread(target=load_and_classify, args=(vec,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


# ---------------------------------------------------------------------------
# Test: Singleton pattern
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_faiss_classifier_returns_same_instance(self):
        from na0s.faiss_classifier import get_faiss_classifier, reset_singleton

        reset_singleton()
        clf1 = get_faiss_classifier()
        clf2 = get_faiss_classifier()
        assert clf1 is clf2
        reset_singleton()

    def test_reset_singleton_clears_instance(self):
        from na0s.faiss_classifier import get_faiss_classifier, reset_singleton

        reset_singleton()
        clf1 = get_faiss_classifier()
        reset_singleton()
        clf2 = get_faiss_classifier()
        assert clf1 is not clf2
        reset_singleton()


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

@requires_faiss
class TestEdgeCases:
    def test_classify_2d_input(self, malicious_only_embeddings):
        """classify() should accept both 1-D and 2-D input."""
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, mal_center = malicious_only_embeddings
        clf = FAISSClassifier(k=5, threshold=0.5)
        clf.build_index(embeddings, labels)

        # 1-D input
        result_1d = clf.classify(mal_center)
        # 2-D input
        result_2d = clf.classify(mal_center.reshape(1, -1))

        assert result_1d["score"] == result_2d["score"]
        assert result_1d["max_similarity"] == pytest.approx(
            result_2d["max_similarity"], abs=1e-6
        )

    def test_threshold_boundary(self):
        """Neighbors exactly at threshold should count."""
        from na0s.faiss_classifier import FAISSClassifier

        # Build index with a single normalized vector
        vec = np.array([[1.0, 0.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)

        clf = FAISSClassifier(k=1, threshold=1.0)
        clf.build_index(vec, labels)

        # Query with identical vector — similarity should be ~1.0
        result = clf.classify(vec[0])
        assert result["neighbors_within_threshold"] == 1

    def test_high_threshold_excludes_all(self, malicious_only_embeddings):
        """With threshold > 1.0, no neighbor should pass."""
        from na0s.faiss_classifier import FAISSClassifier

        embeddings, labels, mal_center = malicious_only_embeddings
        clf = FAISSClassifier(k=5, threshold=1.1)
        clf.build_index(embeddings, labels)

        result = clf.classify(mal_center)
        assert result["neighbors_within_threshold"] == 0
        assert result["score"] == 0.0
        assert result["label"] == "SAFE"
