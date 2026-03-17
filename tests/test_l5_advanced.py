"""Tests for Layer 5 P2 advanced items: contrastive learning, knowledge
distillation, adapter layer, and GCG adversarial suffix generation.

At least 20 tests covering:
  - Contrastive pair creation logic
  - Distillation with mock teacher
  - Adapter architecture
  - GCG suffix patterns
  - Sample generation
  - Graceful degradation for all modules
"""

import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# ============================================================================
# 1. Contrastive Learning Tests
# ============================================================================

class TestContrastivePairCreation(unittest.TestCase):
    """Test create_training_pairs from contrastive_finetune.py."""

    def _make_df(self, n_safe=20, n_mal=20):
        rows = []
        for i in range(n_safe):
            rows.append({"text": "safe text {0}".format(i), "label": 0})
        for i in range(n_mal):
            rows.append({"text": "malicious text {0}".format(i), "label": 1})
        return pd.DataFrame(rows)

    def test_pair_count(self):
        """Generated pairs should respect max_pairs limit."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from contrastive_finetune import create_training_pairs

        df = self._make_df()
        pairs = create_training_pairs(df, max_pairs=100)
        self.assertEqual(len(pairs), 100)

    def test_pair_structure_dict_fallback(self):
        """Without sentence_transformers, pairs should be dicts."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from contrastive_finetune import _make_pair

        pair = _make_pair(["hello", "world"], 1.0)
        # Either InputExample or dict
        if isinstance(pair, dict):
            self.assertIn("texts", pair)
            self.assertIn("label", pair)
            self.assertEqual(len(pair["texts"]), 2)
            self.assertEqual(pair["label"], 1.0)

    def test_pair_labels_are_binary(self):
        """All pair labels should be 0.0 or 1.0."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from contrastive_finetune import create_training_pairs

        df = self._make_df()
        pairs = create_training_pairs(df, max_pairs=50)
        for p in pairs:
            label = p["label"] if isinstance(p, dict) else p.label
            self.assertIn(label, [0.0, 1.0])

    def test_positive_negative_balance(self):
        """Roughly half positive, half negative pairs."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from contrastive_finetune import create_training_pairs

        df = self._make_df()
        pairs = create_training_pairs(df, max_pairs=100)
        labels = [p["label"] if isinstance(p, dict) else p.label for p in pairs]
        n_pos = sum(1 for l in labels if l == 1.0)
        n_neg = sum(1 for l in labels if l == 0.0)
        self.assertEqual(n_pos, 50)
        self.assertEqual(n_neg, 50)

    def test_empty_class_raises(self):
        """Should raise ValueError if one class is missing."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from contrastive_finetune import create_training_pairs

        df = pd.DataFrame({"text": ["a", "b"], "label": [0, 0]})
        with self.assertRaises(ValueError):
            create_training_pairs(df)

    def test_reproducibility(self):
        """Same seed should produce identical pairs."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from contrastive_finetune import create_training_pairs

        df = self._make_df()
        p1 = create_training_pairs(df, max_pairs=20, seed=99)
        p2 = create_training_pairs(df, max_pairs=20, seed=99)

        for a, b in zip(p1, p2):
            t_a = a["texts"] if isinstance(a, dict) else a.texts
            t_b = b["texts"] if isinstance(b, dict) else b.texts
            self.assertEqual(t_a, t_b)

    def test_cli_parser(self):
        """CLI parser should accept required arguments."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from contrastive_finetune import _parse_args

        args = _parse_args(["--model", "test-model", "--epochs", "5"])
        self.assertEqual(args.model, "test-model")
        self.assertEqual(args.epochs, 5)


# ============================================================================
# 2. Knowledge Distillation Tests
# ============================================================================

class TestKnowledgeDistillation(unittest.TestCase):
    """Test distill_model.py distillation logic."""

    def test_soften_probabilities(self):
        """Temperature softening should push probabilities toward 0.5."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from distill_model import _soften_probabilities

        probs = np.array([0.9, 0.1, 0.5])
        softened = _soften_probabilities(probs, temperature=5.0)
        # High temperature should push toward 0.5
        self.assertLess(softened[0], 0.9)
        self.assertGreater(softened[1], 0.1)
        self.assertAlmostEqual(softened[2], 0.5, places=2)

    def test_soften_temperature_1(self):
        """Temperature=1 should approximately preserve probabilities."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from distill_model import _soften_probabilities

        probs = np.array([0.8, 0.2, 0.5])
        softened = _soften_probabilities(probs, temperature=1.0)
        np.testing.assert_allclose(softened, probs, atol=1e-5)

    def test_distill_basic(self):
        """Distillation should produce a fitted LogisticRegression."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from distill_model import distill

        rng = np.random.RandomState(42)
        n, d = 200, 50
        X = rng.randn(n, d)
        teacher_preds = rng.uniform(0, 1, size=n)

        student = distill(teacher_preds, X, temperature=2.0)
        self.assertTrue(hasattr(student, "predict"))
        self.assertTrue(hasattr(student, "predict_proba"))

        preds = student.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_distill_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from distill_model import distill

        X = np.random.randn(100, 50)
        teacher = np.random.uniform(0, 1, size=50)  # wrong shape

        with self.assertRaises(ValueError):
            distill(teacher, X)

    def test_distill_2d_teacher_predictions(self):
        """Teacher predictions can be 2D (n_samples, 2)."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from distill_model import distill

        rng = np.random.RandomState(42)
        n, d = 200, 50
        X = rng.randn(n, d)
        p_mal = rng.uniform(0, 1, size=n)
        teacher_preds = np.column_stack([1 - p_mal, p_mal])

        student = distill(teacher_preds, X)
        self.assertTrue(hasattr(student, "predict"))

    def test_evaluate_distilled(self):
        """evaluate_distilled should return metrics dict."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from distill_model import distill, evaluate_distilled

        rng = np.random.RandomState(42)
        n, d = 200, 50
        X = rng.randn(n, d)
        y = rng.randint(0, 2, size=n)
        teacher_preds = y.astype(float) + rng.normal(0, 0.1, size=n)
        teacher_preds = np.clip(teacher_preds, 0, 1)

        student = distill(teacher_preds, X)
        metrics = evaluate_distilled(student, X, y)

        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("report", metrics)

    def test_cli_parser(self):
        """CLI parser should handle arguments."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from distill_model import _parse_args

        args = _parse_args(["--temperature", "3.0", "--eval"])
        self.assertEqual(args.temperature, 3.0)
        self.assertTrue(args.eval)


# ============================================================================
# 3. Adapter Layer Tests
# ============================================================================

class TestEmbeddingAdapter(unittest.TestCase):
    """Test embedding_adapter.py architecture and training."""

    def test_adapter_module_imports(self):
        """Module should import without error."""
        from na0s.embedding_adapter import (
            EmbeddingAdapter,
            AdapterClassifier,
            train_adapter,
            _HAS_TORCH,
        )

    def test_adapter_has_torch_flag(self):
        """Module should expose _HAS_TORCH flag."""
        from na0s.embedding_adapter import _HAS_TORCH
        self.assertIsInstance(_HAS_TORCH, bool)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch not installed",
    )
    def test_adapter_architecture(self):
        """EmbeddingAdapter should have expected layer structure."""
        import torch
        from na0s.embedding_adapter import EmbeddingAdapter

        adapter = EmbeddingAdapter(input_dim=384, hidden_dim=128, num_classes=2)
        # Check forward pass shape
        x = torch.randn(4, 384)
        out = adapter(x)
        self.assertEqual(out.shape, (4, 2))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch not installed",
    )
    def test_adapter_different_dims(self):
        """Adapter should work with various input dimensions."""
        import torch
        from na0s.embedding_adapter import EmbeddingAdapter

        for dim in [64, 256, 768]:
            adapter = EmbeddingAdapter(input_dim=dim, hidden_dim=64)
            x = torch.randn(2, dim)
            out = adapter(x)
            self.assertEqual(out.shape, (2, 2))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch not installed",
    )
    def test_train_adapter_basic(self):
        """train_adapter should train and return an adapter in eval mode."""
        from na0s.embedding_adapter import train_adapter

        rng = np.random.RandomState(42)
        n, d = 100, 64
        embeddings = rng.randn(n, d).astype(np.float32)
        labels = rng.randint(0, 2, size=n)

        adapter = train_adapter(
            embeddings, labels,
            input_dim=d, hidden_dim=32, epochs=2, batch_size=16,
            device="cpu",
        )
        self.assertFalse(adapter.training)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch not installed",
    )
    def test_train_adapter_auto_input_dim(self):
        """train_adapter should auto-detect input_dim from embeddings."""
        from na0s.embedding_adapter import train_adapter

        rng = np.random.RandomState(42)
        embeddings = rng.randn(50, 128).astype(np.float32)
        labels = rng.randint(0, 2, size=50)

        # Intentionally pass wrong input_dim — should auto-correct
        adapter = train_adapter(
            embeddings, labels,
            input_dim=999, hidden_dim=32, epochs=1,
            device="cpu",
        )
        self.assertIsNotNone(adapter)

    def test_adapter_graceful_without_torch(self):
        """Without torch, EmbeddingAdapter init should raise ImportError."""
        from na0s.embedding_adapter import _HAS_TORCH

        if _HAS_TORCH:
            self.skipTest("torch is installed — cannot test graceful degradation")

        from na0s.embedding_adapter import EmbeddingAdapter
        with self.assertRaises(ImportError):
            EmbeddingAdapter()

    def test_train_adapter_graceful_without_torch(self):
        """Without torch, train_adapter should raise ImportError."""
        from na0s.embedding_adapter import _HAS_TORCH

        if _HAS_TORCH:
            self.skipTest("torch is installed — cannot test graceful degradation")

        from na0s.embedding_adapter import train_adapter
        with self.assertRaises(ImportError):
            train_adapter(np.zeros((10, 64)), np.zeros(10))


# ============================================================================
# 4. GCG Adversarial Suffix Tests
# ============================================================================

class TestGCGSuffixPatterns(unittest.TestCase):
    """Test GCG suffix patterns and sample generation."""

    def test_suffix_patterns_non_empty(self):
        """GCG_SUFFIX_PATTERNS should contain patterns."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import GCG_SUFFIX_PATTERNS

        self.assertGreater(len(GCG_SUFFIX_PATTERNS), 15)

    def test_all_suffixes_are_strings(self):
        """Every suffix should be a non-empty string."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import GCG_SUFFIX_PATTERNS

        for suffix in GCG_SUFFIX_PATTERNS:
            self.assertIsInstance(suffix, str)
            self.assertTrue(len(suffix) > 0)

    def test_generate_samples_count(self):
        """generate_samples should produce exactly n samples."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import generate_samples

        df = generate_samples(
            base_prompts=["hello", "world"],
            n=100,
        )
        self.assertEqual(len(df), 100)

    def test_generate_samples_columns(self):
        """Output DataFrame should have text and label columns."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import generate_samples

        df = generate_samples(base_prompts=["test prompt"], n=10)
        self.assertIn("text", df.columns)
        self.assertIn("label", df.columns)

    def test_generate_samples_ratio(self):
        """Malicious/safe ratio should match malicious_ratio."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import generate_samples

        df = generate_samples(
            base_prompts=["test"],
            n=100,
            malicious_ratio=0.7,
        )
        n_mal = int((df["label"] == 1).sum())
        self.assertEqual(n_mal, 70)

    def test_malicious_samples_contain_suffix(self):
        """Malicious samples should be base + suffix (longer than base)."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import generate_samples, GCG_SUFFIX_PATTERNS

        base = ["short prompt"]
        df = generate_samples(base_prompts=base, n=50, malicious_ratio=1.0)

        for text in df["text"]:
            self.assertTrue(
                len(text) > len("short prompt"),
                "Malicious sample should be longer than base prompt",
            )

    def test_generate_empty_prompts_raises(self):
        """Empty base_prompts should raise ValueError."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import generate_samples

        with self.assertRaises(ValueError):
            generate_samples(base_prompts=[], n=10)

    def test_generate_reproducibility(self):
        """Same seed should produce identical samples."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import generate_samples

        df1 = generate_samples(base_prompts=["test"], n=50, seed=123)
        df2 = generate_samples(base_prompts=["test"], n=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_cli_parser(self):
        """CLI parser should accept arguments."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import _parse_args

        args = _parse_args(["--n", "500", "--seed", "99"])
        self.assertEqual(args.n, 500)
        self.assertEqual(args.seed, 99)

    def test_save_csv(self):
        """Generated samples should save to CSV correctly."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_gcg_samples import generate_samples

        df = generate_samples(base_prompts=["test"], n=20)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            df.to_csv(path, index=False)
            loaded = pd.read_csv(path)
            self.assertEqual(len(loaded), 20)
            self.assertIn("text", loaded.columns)
            self.assertIn("label", loaded.columns)
        finally:
            os.unlink(path)


# ============================================================================
# 5. Cross-module graceful degradation
# ============================================================================

class TestGracefulDegradation(unittest.TestCase):
    """Verify all modules degrade gracefully when optional deps are missing."""

    def test_contrastive_has_import_flag(self):
        """contrastive_finetune should expose _HAS_SENTENCE_TRANSFORMERS."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from contrastive_finetune import _HAS_SENTENCE_TRANSFORMERS
        self.assertIsInstance(_HAS_SENTENCE_TRANSFORMERS, bool)

    def test_distill_has_import_flag(self):
        """distill_model should expose _HAS_SKLEARN."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from distill_model import _HAS_SKLEARN
        self.assertIsInstance(_HAS_SKLEARN, bool)

    def test_adapter_has_import_flags(self):
        """embedding_adapter should expose _HAS_TORCH and _HAS_SENTENCE_TRANSFORMERS."""
        from na0s.embedding_adapter import _HAS_TORCH, _HAS_SENTENCE_TRANSFORMERS
        self.assertIsInstance(_HAS_TORCH, bool)
        self.assertIsInstance(_HAS_SENTENCE_TRANSFORMERS, bool)

    def test_contrastive_finetune_returns_none_without_deps(self):
        """finetune() should return None if sentence-transformers missing."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import contrastive_finetune

        if contrastive_finetune._HAS_SENTENCE_TRANSFORMERS:
            self.skipTest("sentence-transformers installed")

        result = contrastive_finetune.finetune(
            model_name="test",
            pairs=[{"texts": ["a", "b"], "label": 1.0}],
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
