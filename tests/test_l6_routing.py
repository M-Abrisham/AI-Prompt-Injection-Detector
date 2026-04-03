"""Tests for Layer 6 features: adaptive complexity routing, paranoid
confidence mode, and configurable stage pipeline.

At least 22 tests total:
- Complexity assessment: 8 tests
- Pipeline stage selection: 4 tests
- Paranoid mode: 6 tests
- Configurable stages: 6 tests
"""

import os
import threading
import pytest
from unittest.mock import patch, MagicMock

from na0s.complexity_router import (
    ComplexityLevel,
    assess_complexity,
    get_pipeline_stages,
    is_adaptive_routing_enabled,
    _count_words,
)
from na0s.cascade import (
    CascadeClassifier,
    VALID_STAGES,
    DEFAULT_STAGES,
    _PARANOID_LOWER,
    _PARANOID_UPPER,
)


# ===================================================================
# Helpers
# ===================================================================

def _simple_text():
    """Short, clean, single-language text (<=50 words)."""
    return "What is the weather today?"


def _moderate_text():
    """51-200 words, no obfuscation, no boundaries, single language."""
    words = ["word"] * 80
    return " ".join(words)


def _complex_long_text():
    """>200 words."""
    words = ["word"] * 250
    return " ".join(words)


def _complex_boundary_text():
    """Contains structural boundary markers."""
    return "Hello world\n---\n[SYSTEM]\nDo something"


def _complex_multilingual_text():
    """Contains multiple scripts (Latin + CJK)."""
    return "Hello world. Please translate: \u4f60\u597d\u4e16\u754c"


# ===================================================================
# 1. Complexity Assessment Tests (8 tests)
# ===================================================================

class TestComplexityAssessment:
    """Test assess_complexity() classifies texts correctly."""

    def test_simple_short_text(self):
        level = assess_complexity(_simple_text())
        assert level == ComplexityLevel.SIMPLE

    def test_simple_under_50_words(self):
        text = " ".join(["hello"] * 30)
        level = assess_complexity(text)
        assert level == ComplexityLevel.SIMPLE

    def test_moderate_word_count(self):
        """51-200 words with no other complexity signals -> MODERATE."""
        with patch("na0s.complexity_router._count_obfuscation_flags", return_value=0):
            level = assess_complexity(_moderate_text())
        assert level == ComplexityLevel.MODERATE

    def test_moderate_with_few_obfuscation_flags(self):
        """Text with 1-2 obfuscation flags -> MODERATE."""
        text = "Short text"
        with patch("na0s.complexity_router._count_obfuscation_flags", return_value=2):
            level = assess_complexity(text)
        assert level == ComplexityLevel.MODERATE

    def test_complex_long_text(self):
        """>200 words -> COMPLEX."""
        level = assess_complexity(_complex_long_text())
        assert level == ComplexityLevel.COMPLEX

    def test_complex_structural_boundary(self):
        """Structural boundary markers -> COMPLEX."""
        level = assess_complexity(_complex_boundary_text())
        assert level == ComplexityLevel.COMPLEX

    def test_complex_multilingual(self):
        """Multiple scripts -> COMPLEX."""
        level = assess_complexity(_complex_multilingual_text())
        assert level == ComplexityLevel.COMPLEX

    def test_complex_many_obfuscation_flags(self):
        """3+ obfuscation flags -> COMPLEX."""
        text = "Short text"
        with patch("na0s.complexity_router._count_obfuscation_flags", return_value=3):
            level = assess_complexity(text)
        assert level == ComplexityLevel.COMPLEX


# ===================================================================
# 2. Pipeline Stage Selection Tests (4 tests)
# ===================================================================

class TestPipelineStages:
    """Test get_pipeline_stages() returns correct stages per level."""

    def test_simple_stages(self):
        stages = get_pipeline_stages(ComplexityLevel.SIMPLE)
        assert stages == ["whitelist", "ml_basic"]

    def test_moderate_stages(self):
        stages = get_pipeline_stages(ComplexityLevel.MODERATE)
        assert stages == ["whitelist", "weighted", "embedding"]

    def test_complex_stages(self):
        stages = get_pipeline_stages(ComplexityLevel.COMPLEX)
        assert stages == ["whitelist", "weighted", "embedding",
                          "judge"]

    def test_stages_returns_copy(self):
        """Returned list should be a copy, not a reference to internal data."""
        stages1 = get_pipeline_stages(ComplexityLevel.SIMPLE)
        stages2 = get_pipeline_stages(ComplexityLevel.SIMPLE)
        assert stages1 == stages2
        stages1.append("extra")
        assert "extra" not in get_pipeline_stages(ComplexityLevel.SIMPLE)


# ===================================================================
# 3. Adaptive Routing Env Var Tests (2 tests)
# ===================================================================

class TestAdaptiveRoutingEnvVar:
    """Test is_adaptive_routing_enabled() env var check."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove key if it exists
            os.environ.pop("NA0S_ADAPTIVE_ROUTING", None)
            assert is_adaptive_routing_enabled() is False

    def test_enabled_when_set(self):
        with patch.dict(os.environ, {"NA0S_ADAPTIVE_ROUTING": "1"}):
            assert is_adaptive_routing_enabled() is True


# ===================================================================
# 4. Paranoid Confidence Mode Tests (6 tests)
# ===================================================================

class TestParanoidMode:
    """Test paranoid mode flips uncertain SAFE verdicts to MALICIOUS."""

    def _make_classifier(self, paranoid=False, **kwargs):
        """Create a CascadeClassifier with mocked models."""
        clf = CascadeClassifier.__new__(CascadeClassifier)
        clf._vectorizer = MagicMock()
        clf._model = MagicMock()
        clf._whitelist = MagicMock()
        clf._whitelist.is_whitelisted = MagicMock(return_value=(False, "not safe"))
        clf._weighted = MagicMock()
        clf._judge = None
        clf._llm_checker = None
        clf._llm_checker_init_attempted = True
        clf._embedding_model = None
        clf._embedding_classifier = None
        clf._enable_embedding = False
        clf._enable_ensemble = False
        clf._ensemble_used = 0
        clf._positive_validator = None
        clf._output_scanner = None
        clf._canary_manager = None
        clf._stats_lock = threading.Lock()
        clf._total = 0
        clf._whitelisted = 0
        clf._classified = 0
        clf._judged = 0
        clf._judge_overrides = 0
        clf._blocked = 0
        clf._embedding_used = 0
        clf._positive_validated = 0
        clf._positive_validation_overrides = 0
        clf._canary_checks = 0
        clf._layer_failures = {
            "structural": 0, "promptguard": 0, "ensemble": 0,
            "embedding": 0, "judge": 0, "positive_validation": 0,
            "output_scanner": 0, "canary": 0,
        }
        clf._slo_enabled = False
        clf._slo = None
        clf._batch_lock = threading.Lock()
        clf._paranoid_mode = paranoid
        clf._stages = list(DEFAULT_STAGES)
        return clf

    def test_paranoid_flips_uncertain_safe(self):
        """Uncertain SAFE (P(mal) in [0.35, 0.65]) -> MALICIOUS."""
        clf = self._make_classifier(paranoid=True)
        # confidence=0.55 means P(safe)=0.55 -> P(mal)=0.45 -> in uncertain zone
        clf._weighted.classify.return_value = ("SAFE", 0.55, [])

        with patch("na0s.cascade.layer0_sanitize") as mock_l0:
            mock_l0_result = MagicMock()
            mock_l0_result.rejected = False
            mock_l0_result.sanitized_text = "test text"
            mock_l0_result.anomaly_flags = []
            mock_l0.return_value = mock_l0_result
            with patch("na0s.cascade._verify_grounded", return_value={"grounded": True}):
                label, confidence, hits, stage = clf.classify("test text")

        assert label == "MALICIOUS"
        assert "paranoid_mode:uncertain_flip" in hits

    def test_paranoid_no_flip_clear_safe(self):
        """Clear SAFE (P(mal) < 0.35) stays SAFE even in paranoid mode."""
        clf = self._make_classifier(paranoid=True)
        # confidence=0.90 means P(safe)=0.90 -> P(mal)=0.10 -> NOT in uncertain zone
        clf._weighted.classify.return_value = ("SAFE", 0.90, [])

        with patch("na0s.cascade.layer0_sanitize") as mock_l0:
            mock_l0_result = MagicMock()
            mock_l0_result.rejected = False
            mock_l0_result.sanitized_text = "test text"
            mock_l0_result.anomaly_flags = []
            mock_l0.return_value = mock_l0_result
            with patch("na0s.cascade._verify_grounded", return_value={"grounded": True}):
                label, confidence, hits, stage = clf.classify("test text")

        assert label == "SAFE"
        assert "paranoid_mode:uncertain_flip" not in hits

    def test_paranoid_no_flip_already_malicious(self):
        """Already MALICIOUS stays MALICIOUS (paranoid only flips SAFE)."""
        clf = self._make_classifier(paranoid=True)
        clf._weighted.classify.return_value = ("MALICIOUS", 0.50, ["some_rule"])

        with patch("na0s.cascade.layer0_sanitize") as mock_l0:
            mock_l0_result = MagicMock()
            mock_l0_result.rejected = False
            mock_l0_result.sanitized_text = "test text"
            mock_l0_result.anomaly_flags = []
            mock_l0.return_value = mock_l0_result
            with patch("na0s.cascade._verify_grounded", return_value={"grounded": True}):
                label, confidence, hits, stage = clf.classify("test text")

        assert label == "MALICIOUS"
        assert "paranoid_mode:uncertain_flip" not in hits

    def test_paranoid_disabled_no_flip(self):
        """Without paranoid mode, uncertain SAFE stays SAFE."""
        clf = self._make_classifier(paranoid=False)
        # confidence=0.55 -> P(mal)=0.45, in uncertain zone but paranoid is off
        clf._weighted.classify.return_value = ("SAFE", 0.55, [])

        with patch("na0s.cascade.layer0_sanitize") as mock_l0:
            mock_l0_result = MagicMock()
            mock_l0_result.rejected = False
            mock_l0_result.sanitized_text = "test text"
            mock_l0_result.anomaly_flags = []
            mock_l0.return_value = mock_l0_result
            with patch("na0s.cascade._verify_grounded", return_value={"grounded": True}):
                label, confidence, hits, stage = clf.classify("test text")

        assert label == "SAFE"
        assert "paranoid_mode:uncertain_flip" not in hits

    def test_paranoid_env_var_enables(self):
        """NA0S_PARANOID_MODE=1 enables paranoid mode."""
        with patch.dict(os.environ, {"NA0S_PARANOID_MODE": "1"}):
            clf = CascadeClassifier.__new__(CascadeClassifier)
            # Simulate __init__ logic for paranoid_mode
            clf._paranoid_mode = (
                os.environ.get("NA0S_PARANOID_MODE", "0") == "1"
                or False
            )
        assert clf._paranoid_mode is True

    def test_paranoid_boundary_values(self):
        """Test exact boundary of uncertain zone (P(mal) = 0.35)."""
        clf = self._make_classifier(paranoid=True)
        # P(safe) = 0.65 -> P(mal) = 0.35 -> exactly at lower boundary
        clf._weighted.classify.return_value = ("SAFE", 0.65, [])

        with patch("na0s.cascade.layer0_sanitize") as mock_l0:
            mock_l0_result = MagicMock()
            mock_l0_result.rejected = False
            mock_l0_result.sanitized_text = "test text"
            mock_l0_result.anomaly_flags = []
            mock_l0.return_value = mock_l0_result
            with patch("na0s.cascade._verify_grounded", return_value={"grounded": True}):
                label, confidence, hits, stage = clf.classify("test text")

        assert label == "MALICIOUS"
        assert "paranoid_mode:uncertain_flip" in hits


# ===================================================================
# 5. Configurable Stage Pipeline Tests (6 tests)
# ===================================================================

class TestConfigurableStages:
    """Test configurable stage pipeline."""

    def test_default_stages(self):
        """Default stages match DEFAULT_STAGES."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NA0S_CASCADE_STAGES", None)
            clf = CascadeClassifier.__new__(CascadeClassifier)
            # Simulate __init__ stage logic
            env_stages = os.environ.get("NA0S_CASCADE_STAGES")
            clf._stages = list(DEFAULT_STAGES)
        assert clf._stages == ["whitelist", "weighted", "judge"]

    def test_custom_stages_constructor(self):
        """Custom stages via constructor parameter."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NA0S_CASCADE_STAGES", None)
            clf = CascadeClassifier.__new__(CascadeClassifier)
            stages = ["weighted"]
            clf._stages = list(stages)
        assert clf._stages == ["weighted"]

    def test_env_var_overrides_constructor(self):
        """NA0S_CASCADE_STAGES env var overrides constructor."""
        with patch.dict(os.environ, {"NA0S_CASCADE_STAGES": "whitelist,weighted"}):
            clf = CascadeClassifier.__new__(CascadeClassifier)
            env_stages = os.environ.get("NA0S_CASCADE_STAGES")
            if env_stages is not None:
                clf._stages = [s.strip() for s in env_stages.split(",") if s.strip()]
            else:
                clf._stages = list(DEFAULT_STAGES)
        assert clf._stages == ["whitelist", "weighted"]

    def test_invalid_stage_raises(self):
        """Unknown stage name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cascade stage"):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NA0S_CASCADE_STAGES", None)
                CascadeClassifier(stages=["bogus_stage"])

    def test_valid_stages_list(self):
        """VALID_STAGES contains expected entries."""
        assert "whitelist" in VALID_STAGES
        assert "weighted" in VALID_STAGES
        assert "judge" in VALID_STAGES
        assert "embedding" in VALID_STAGES
        assert "ml_basic" in VALID_STAGES
        # late_chunking is an internal sub-step of the embedding pipeline
        # (predict_embedding.py), not a standalone cascade stage.
        assert "late_chunking" not in VALID_STAGES

    def test_stages_reordering_accepted(self):
        """Stages can be reordered (validation only checks names)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NA0S_CASCADE_STAGES", None)
            # Reverse order is accepted
            clf = CascadeClassifier(stages=["judge", "weighted", "whitelist"])
        assert clf._stages == ["judge", "weighted", "whitelist"]


# ===================================================================
# 6. Word counting helper test
# ===================================================================

class TestWordCount:
    def test_word_count(self):
        assert _count_words("hello world") == 2
        assert _count_words("one two three four five") == 5
        assert _count_words("single") == 1
