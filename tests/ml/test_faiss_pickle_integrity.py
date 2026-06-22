"""Integrity-gate regression tests for the FAISS / stacking pickle sinks.

These cover the DEF-2 / DEF-10a hardening: every ``.labels.pkl`` (FAISS) and
stacking-model pickle must be loaded through ``na0s.integrity.safe_pickle``
(verified against an HMAC/SHA-256 sidecar) and must FAIL CLOSED — raise rather
than return an attacker-controlled unpickled object — on a tampered artifact.

A swapped ``.labels.pkl`` whose first opcodes invoke ``__reduce__`` is the
classic pickle RCE sink; before the fix ``FAISSClassifier.load`` did a raw
``pickle.load`` of that file. These tests assert it is now rejected.

faiss is mocked where the index binary is not actually needed, so the suite
runs without faiss-cpu installed and never touches the network.
"""

from __future__ import annotations

import os
import pickle
import tempfile

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
def tmp_index_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_faiss_index.bin")


@pytest.fixture
def malicious_only_embeddings():
    """A small block of malicious-only embeddings + int64 labels."""
    np.random.seed(7)
    dim = 384
    n = 12
    center = np.zeros(dim, dtype=np.float32)
    center[0] = 1.0
    embeddings = center + np.random.randn(n, dim).astype(np.float32) * 0.1
    labels = np.ones(n, dtype=np.int64)
    return embeddings, labels, center


# ---------------------------------------------------------------------------
# DEF-2: FAISS .labels.pkl integrity
# ---------------------------------------------------------------------------

class TestFaissLabelsIntegrity:
    @requires_faiss
    def test_safe_dump_load_roundtrip(self, malicious_only_embeddings, tmp_index_path):
        """save() writes a sidecar; load() verifies and round-trips labels."""
        from na0s.ml.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf1 = FAISSClassifier()
        clf1.build_index(embeddings, labels)
        clf1.save(tmp_index_path)

        labels_path = tmp_index_path + ".labels.pkl"
        # safe_dump writes the pickle plus an integrity sidecar (HMAC or SHA-256).
        assert os.path.exists(labels_path)
        assert os.path.exists(labels_path + ".hmac") or os.path.exists(
            labels_path + ".sha256"
        )

        clf2 = FAISSClassifier()
        clf2.load(tmp_index_path)
        np.testing.assert_array_equal(clf2._labels, labels)

    @requires_faiss
    def test_tampered_labels_rejected(self, malicious_only_embeddings, tmp_index_path):
        """A swapped .labels.pkl WITHOUT a matching sidecar is rejected.

        We overwrite the pickle bytes (simulating an attacker swapping in an
        RCE payload) but leave the original SHA-256 sidecar in place. The
        digest no longer matches, so safe_load must raise ValueError.
        """
        from na0s.ml.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf1 = FAISSClassifier()
        clf1.build_index(embeddings, labels)
        clf1.save(tmp_index_path)

        labels_path = tmp_index_path + ".labels.pkl"
        # Forge a different, valid-magic pickle whose digest won't match the sidecar.
        forged = pickle.dumps({"attacker": "object", "labels": [9, 9, 9]})
        with open(labels_path, "wb") as f:
            f.write(forged)

        clf2 = FAISSClassifier()
        with pytest.raises(ValueError, match="Integrity check failed"):
            clf2.load(tmp_index_path)

    @requires_faiss
    def test_missing_sidecar_rejected(self, malicious_only_embeddings, tmp_index_path):
        """A .labels.pkl with NO sidecar at all is refused (no raw fallback)."""
        from na0s.ml.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf1 = FAISSClassifier()
        clf1.build_index(embeddings, labels)
        clf1.save(tmp_index_path)

        labels_path = tmp_index_path + ".labels.pkl"
        for sidecar in (labels_path + ".hmac", labels_path + ".sha256"):
            if os.path.exists(sidecar):
                os.remove(sidecar)

        clf2 = FAISSClassifier()
        # No integrity source available -> FileNotFoundError from safe_load.
        with pytest.raises((FileNotFoundError, ValueError)):
            clf2.load(tmp_index_path)

    @requires_faiss
    def test_ensure_loaded_fails_closed_on_tamper(
        self, malicious_only_embeddings, tmp_index_path
    ):
        """The live lazy-load path (_ensure_loaded) refuses on integrity failure.

        This is the NA0S_FAISS_ENABLED=1 entry point: predict_embedding calls
        ``faiss_clf._ensure_loaded()`` and only queries when it returns True.
        On a tampered artifact it must return False (refuse) — never load the
        attacker object.
        """
        from na0s.ml.faiss_classifier import FAISSClassifier

        embeddings, labels, _ = malicious_only_embeddings
        clf1 = FAISSClassifier()
        clf1.build_index(embeddings, labels)
        clf1.save(tmp_index_path)

        labels_path = tmp_index_path + ".labels.pkl"
        with open(labels_path, "wb") as f:
            f.write(pickle.dumps({"attacker": "object"}))

        clf2 = FAISSClassifier(index_path=tmp_index_path)
        assert clf2._ensure_loaded() is False
        assert clf2._loaded is False
        assert clf2._init_failed is True
        # And a subsequent classify() degrades safe, never raising RCE-loaded data.
        result = clf2.classify(np.ones(384, dtype=np.float32))
        assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# DEF-10a: stacking-classifier pickle integrity
# ---------------------------------------------------------------------------

class TestStackingIntegrity:
    def _trained_learner(self):
        pytest.importorskip("sklearn")
        from na0s.ml.stacking_classifier import StackingMetaLearner

        learner = StackingMetaLearner()
        # 5 stage-2 features per the FEATURE_NAMES contract.
        X = np.random.RandomState(0).rand(40, 5)
        y = (X[:, 0] > 0.5).astype(int)
        learner.train(X, y)
        return learner, StackingMetaLearner

    def test_safe_roundtrip(self, tmp_index_path):
        learner, StackingMetaLearner = self._trained_learner()
        path = tmp_index_path + ".stack.pkl"
        learner.save(path)

        assert os.path.exists(path + ".hmac") or os.path.exists(path + ".sha256")

        loaded = StackingMetaLearner()
        loaded.load(path)
        assert loaded.is_available() is True

    def test_tampered_stacking_model_rejected(self, tmp_index_path):
        learner, StackingMetaLearner = self._trained_learner()
        path = tmp_index_path + ".stack.pkl"
        learner.save(path)

        # Swap in a forged pickle; sidecar digest no longer matches.
        with open(path, "wb") as f:
            f.write(pickle.dumps({"model": None, "trained": True}))

        loaded = StackingMetaLearner()
        with pytest.raises(ValueError, match="Integrity check failed"):
            loaded.load(path)


# ---------------------------------------------------------------------------
# No raw pickle.load survives in ml/ (defense-in-depth source assertion)
# ---------------------------------------------------------------------------

def test_no_raw_pickle_load_in_ml_sources():
    """Grep-equivalent guard: ml/ sources must not call a bare pickle.load."""
    import na0s.ml as ml_pkg

    ml_dir = os.path.dirname(ml_pkg.__file__)
    offenders = []
    for root, _dirs, files in os.walk(ml_dir):
        if "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            fpath = os.path.join(root, name)
            with open(fpath, "r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    if "pickle.load(" in line:
                        offenders.append("{}:{}".format(fpath, lineno))
    assert offenders == [], "raw pickle.load found in ml/: {}".format(offenders)
