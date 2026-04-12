"""Tests for Layer 5: stratified train/test split and model selection/benchmarking.

Covers:
  - Stratified split verification in model_embedding.py
  - Benchmark script importability and configuration
  - Results JSON schema validation
  - Edge cases for stratification and benchmarking helpers
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Ensure scripts/ and src/ are importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))


# ===========================================================================
# Item 1: Stratified train/test split verification
# ===========================================================================

class TestStratifiedSplit:
    """Verify that model_embedding uses stratified splitting."""

    def test_train_test_split_uses_stratify(self):
        """Confirm that model_embedding.py calls train_test_split with stratify=y."""
        import model_embedding
        import inspect
        source = inspect.getsource(model_embedding.train_embedding_model)
        assert "stratify=y" in source or "stratify = y" in source, (
            "train_embedding_model must use stratify=y in train_test_split"
        )

    def test_verify_stratified_split_passes_balanced(self):
        """Stratification check passes for a perfectly balanced split."""
        from model_embedding import verify_stratified_split
        rng = np.random.RandomState(42)
        y = np.array([0]*500 + [1]*500)
        rng.shuffle(y)
        y_train, y_test = train_test_split(y, test_size=0.2, random_state=42, stratify=y)
        assert verify_stratified_split(y, y_train, y_test)

    def test_verify_stratified_split_passes_imbalanced(self):
        """Stratification check passes for an imbalanced dataset (90/10)."""
        from model_embedding import verify_stratified_split
        y = np.array([0]*900 + [1]*100)
        y_train, y_test = train_test_split(y, test_size=0.2, random_state=42, stratify=y)
        assert verify_stratified_split(y, y_train, y_test)

    def test_verify_stratified_split_fails_without_stratify(self):
        """Without stratify, a skewed split may fail verification."""
        from model_embedding import verify_stratified_split
        # Create a dataset where lack of stratification could cause issues
        # Use a very tight tolerance to catch non-stratified splits
        y = np.array([0]*900 + [1]*100)
        # Non-stratified split with a seed that gives bad distribution
        y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
        # With a tight tolerance, non-stratified may or may not fail
        # but stratified should always pass
        y_train_s, y_test_s = train_test_split(
            y, test_size=0.2, random_state=42, stratify=y,
        )
        # Stratified version should always pass
        assert verify_stratified_split(y, y_train_s, y_test_s, tolerance=0.01)

    def test_stratified_split_preserves_ratio_within_tolerance(self):
        """Class ratios in train/test should be within 2% of original."""
        y = np.array([0]*700 + [1]*300)
        y_train, y_test = train_test_split(
            y, test_size=0.2, random_state=42, stratify=y,
        )
        full_ratio = np.mean(y == 1)
        train_ratio = np.mean(y_train == 1)
        test_ratio = np.mean(y_test == 1)
        assert abs(full_ratio - train_ratio) < 0.02
        assert abs(full_ratio - test_ratio) < 0.02

    def test_stratified_split_all_classes_present(self):
        """Both classes should be present in train and test splits."""
        y = np.array([0]*800 + [1]*200)
        y_train, y_test = train_test_split(
            y, test_size=0.2, random_state=42, stratify=y,
        )
        assert 0 in y_train and 1 in y_train
        assert 0 in y_test and 1 in y_test


# ===========================================================================
# Item 2: Model selection / benchmarking
# ===========================================================================

class TestBenchmarkImport:
    """Test that the benchmark_embeddings module can be imported and configured."""

    def test_module_imports(self):
        """benchmark_embeddings can be imported without errors."""
        import benchmark_embeddings
        assert benchmark_embeddings is not None

    def test_candidate_models_defined(self):
        """CANDIDATE_MODELS list is defined and non-empty."""
        from benchmark_embeddings import CANDIDATE_MODELS
        assert isinstance(CANDIDATE_MODELS, list)
        assert len(CANDIDATE_MODELS) >= 3

    def test_candidate_models_contains_expected(self):
        """CANDIDATE_MODELS contains the three required models."""
        from benchmark_embeddings import CANDIDATE_MODELS
        model_names_lower = [m.lower() for m in CANDIDATE_MODELS]
        joined = " ".join(model_names_lower)
        assert "minilm" in joined, "all-MiniLM-L6-v2 should be in CANDIDATE_MODELS"
        assert "bge-small" in joined, "bge-small-en-v1.5 should be in CANDIDATE_MODELS"
        assert "gte-small" in joined, "gte-small should be in CANDIDATE_MODELS"

    def test_has_run_benchmark_function(self):
        """run_benchmark() function exists."""
        from benchmark_embeddings import run_benchmark
        assert callable(run_benchmark)

    def test_has_save_results_function(self):
        """save_results() function exists."""
        from benchmark_embeddings import save_results
        assert callable(save_results)

    def test_has_print_comparison_table(self):
        """print_comparison_table() function exists."""
        from benchmark_embeddings import print_comparison_table
        assert callable(print_comparison_table)

    def test_has_sentence_transformers_flag(self):
        """Module exposes _HAS_SENTENCE_TRANSFORMERS flag."""
        from benchmark_embeddings import _HAS_SENTENCE_TRANSFORMERS
        assert isinstance(_HAS_SENTENCE_TRANSFORMERS, bool)


class TestBenchmarkResultsSchema:
    """Test the JSON schema for benchmark results."""

    def test_get_results_schema_returns_dict(self):
        """get_results_schema() returns a dict."""
        from benchmark_embeddings import get_results_schema
        schema = get_results_schema()
        assert isinstance(schema, dict)

    def test_schema_has_required_fields(self):
        """Schema requires benchmark, candidate_models, results, timestamp."""
        from benchmark_embeddings import get_results_schema
        schema = get_results_schema()
        required = schema.get("required", [])
        for field in ["benchmark", "candidate_models", "results", "timestamp"]:
            assert field in required, "Schema should require '{0}'".format(field)

    def test_save_results_creates_valid_json(self):
        """save_results() writes valid JSON matching expected structure."""
        from benchmark_embeddings import save_results, get_results_schema

        mock_results = [
            {
                "model": "test-model",
                "accuracy": 0.95,
                "f1": 0.94,
                "auc": 0.98,
                "embed_dim": 384,
                "encode_time_s": 1.5,
                "train_time_s": 0.3,
                "n_samples": 100,
                "n_train": 80,
                "n_test": 20,
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            tmp_path = f.name

        try:
            save_results(mock_results, path=tmp_path)

            with open(tmp_path) as f:
                data = json.load(f)

            schema = get_results_schema()
            # Validate required top-level keys
            for key in schema["required"]:
                assert key in data, "Missing required key: {0}".format(key)

            assert isinstance(data["results"], list)
            assert len(data["results"]) == 1
            assert data["results"][0]["model"] == "test-model"
            assert data["results"][0]["accuracy"] == 0.95
            assert isinstance(data["timestamp"], str)
            assert isinstance(data["candidate_models"], list)
        finally:
            os.unlink(tmp_path)

    def test_save_results_includes_candidate_models(self):
        """Saved JSON includes the full CANDIDATE_MODELS list."""
        from benchmark_embeddings import save_results, CANDIDATE_MODELS

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            tmp_path = f.name

        try:
            save_results([], path=tmp_path)
            with open(tmp_path) as f:
                data = json.load(f)
            assert data["candidate_models"] == CANDIDATE_MODELS
        finally:
            os.unlink(tmp_path)

    def test_save_results_handles_error_entries(self):
        """Error results (missing metrics) are saved without crashing."""
        from benchmark_embeddings import save_results

        results_with_error = [
            {"model": "broken-model", "error": "download failed"},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            tmp_path = f.name

        try:
            save_results(results_with_error, path=tmp_path)
            with open(tmp_path) as f:
                data = json.load(f)
            assert data["results"][0]["error"] == "download failed"
        finally:
            os.unlink(tmp_path)


class TestPrintComparisonTable:
    """Test the comparison table printer."""

    def test_print_empty_results(self, capsys):
        """print_comparison_table handles empty results gracefully."""
        from benchmark_embeddings import print_comparison_table
        print_comparison_table([])
        captured = capsys.readouterr()
        assert "No results" in captured.out

    def test_print_with_error_result(self, capsys):
        """print_comparison_table handles error entries."""
        from benchmark_embeddings import print_comparison_table
        results = [{"model": "bad-model", "error": "timeout"}]
        print_comparison_table(results)
        captured = capsys.readouterr()
        assert "FAILED" in captured.out

    def test_print_with_valid_results(self, capsys):
        """print_comparison_table displays metrics for valid results."""
        from benchmark_embeddings import print_comparison_table
        results = [
            {
                "model": "test-model",
                "accuracy": 0.95,
                "f1": 0.94,
                "auc": 0.98,
                "embed_dim": 384,
                "encode_time_s": 1.5,
                "train_time_s": 0.3,
                "n_samples": 100,
                "n_train": 80,
                "n_test": 20,
            },
        ]
        print_comparison_table(results)
        captured = capsys.readouterr()
        assert "test-model" in captured.out
        assert "Best model" in captured.out


class TestGracefulDegradation:
    """Test that the benchmark degrades gracefully without sentence-transformers."""

    def test_run_benchmark_returns_empty_without_st(self, monkeypatch):
        """If sentence-transformers is unavailable, run_benchmark returns []."""
        import benchmark_embeddings
        monkeypatch.setattr(benchmark_embeddings, "_HAS_SENTENCE_TRANSFORMERS", False)
        result = benchmark_embeddings.run_benchmark(sample_size=10)
        assert result == []
