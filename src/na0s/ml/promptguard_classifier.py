"""N5 -- mDeBERTa-based classifier (Prompt Guard 2) for prompt injection detection.

This module wraps Meta's Prompt-Guard-2-22M model (a fine-tuned mDeBERTa
22M-parameter sequence classifier) for high-accuracy injection/jailbreak
detection.  It can either replace or augment the TF-IDF L4 classifier.

Key features:
  - **Lazy model loading**: model and tokenizer are loaded on first call.
  - **Batch support**: ``classify_batch()`` for efficient multi-input scoring.
  - **LRU cache**: repeated identical inputs skip re-inference.
  - **Device selection**: auto-detects CUDA/MPS/CPU; configurable via
    ``NA0S_DEVICE`` env var.
  - **Graceful degradation**: if ``transformers`` is not installed,
    ``_HAS_TRANSFORMERS = False`` and all calls return None.
  - **Thread-safe**: model loading uses ``threading.Lock()``.
  - **Truncation**: auto-truncates to model's max_length (512 tokens).

Configuration (env vars):
  - ``NA0S_PROMPTGUARD_MODEL``: HuggingFace model name or local path.
    Default: ``meta-llama/Prompt-Guard-2-22M``.
  - ``NA0S_DEVICE``: PyTorch device string (``cpu``, ``cuda``, ``mps``).
    Default: auto-detect.
  - ``NA0S_ENABLE_PROMPTGUARD``: Set to ``1`` to enable PromptGuard in the
    pipeline (opt-in since it requires downloading a model).

Usage::

    from na0s.promptguard_classifier import PromptGuardClassifier

    clf = PromptGuardClassifier()
    if clf.is_available():
        result = clf.classify("Ignore all previous instructions")
        print(result.label)       # "INJECTION"
        print(result.confidence)  # 0.97
        print(result.raw_scores)  # {"BENIGN": 0.01, "INJECTION": 0.97, ...}
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import: transformers + torch are optional dependencies
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# Pure-python (no transformers dep): hardened from_pretrained kwargs
# (use_safetensors / trust_remote_code=False / pinned revision).
from na0s.integrity.hf_loading import (
    hf_from_pretrained_kwargs,
    hf_tokenizer_kwargs,
)

# ---------------------------------------------------------------------------
# Label mapping -- Prompt-Guard-2-22M uses integer label ids
# ---------------------------------------------------------------------------
_LABEL_MAP: Dict[int, str] = {0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"}

# Model maximum sequence length (mDeBERTa limit)
_MAX_TOKENS = 512

# Default model identifier on HuggingFace Hub
DEFAULT_MODEL_NAME = "meta-llama/Prompt-Guard-2-22M"

# Default LRU cache size (number of cached classify results)
_DEFAULT_CACHE_SIZE = 256

# ---------------------------------------------------------------------------
# Environment variable keys
# ---------------------------------------------------------------------------
_ENV_MODEL = "NA0S_PROMPTGUARD_MODEL"
_ENV_DEVICE = "NA0S_DEVICE"
_ENV_ENABLE = "NA0S_ENABLE_PROMPTGUARD"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptGuardResult:
    """Classification result from PromptGuard.

    Attributes
    ----------
    label : str
        Top predicted class: ``"INJECTION"``, ``"JAILBREAK"``, or ``"BENIGN"``.
    confidence : float
        Probability of the top class, in [0.0, 1.0].
    raw_scores : dict
        Mapping from each class name to its softmax probability.
    """
    label: str
    confidence: float
    raw_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Auto-detect the best available device: CUDA > MPS > CPU.

    Respects the ``NA0S_DEVICE`` env var if set.
    """
    env_device = os.environ.get(_ENV_DEVICE, "").strip()
    if env_device:
        return env_device

    if not _HAS_TRANSFORMERS:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# PromptGuardClassifier
# ---------------------------------------------------------------------------

class PromptGuardClassifier:
    """Wrapper around Meta Prompt-Guard-2-22M for injection/jailbreak detection.

    Parameters
    ----------
    model_name : str or None
        HuggingFace model identifier or local path.  If *None*, reads from
        ``NA0S_PROMPTGUARD_MODEL`` env var, falling back to the default.
    device : str or None
        PyTorch device string.  If *None*, auto-detects via ``_detect_device()``.
    cache_size : int
        Maximum number of cached classify results (LRU eviction).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ):
        if model_name is None:
            model_name = os.environ.get(_ENV_MODEL, DEFAULT_MODEL_NAME)
        if device is None:
            device = _detect_device()

        self._model_name = model_name
        self._device = device
        self._cache_size = cache_size

        # Lazy-loaded state
        self._tokenizer: Optional[object] = None
        self._model: Optional[object] = None
        self._lock = threading.Lock()
        self._init_failed = False

        # Build a per-instance LRU cache (wrapping a private method)
        self._classify_cached = lru_cache(maxsize=cache_size)(self._classify_impl)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the ``transformers`` library is installed."""
        return _HAS_TRANSFORMERS

    @staticmethod
    def is_enabled() -> bool:
        """Return True if PromptGuard is enabled via env var."""
        return os.environ.get(_ENV_ENABLE, "0").lower() in ("1", "true", "yes")

    def classify(self, text: str) -> Optional[PromptGuardResult]:
        """Classify *text* as BENIGN, INJECTION, or JAILBREAK.

        Parameters
        ----------
        text : str
            The input text to classify.

        Returns
        -------
        PromptGuardResult or None
            None if ``transformers`` is not installed.
        """
        if not _HAS_TRANSFORMERS:
            return None

        return self._classify_cached(text)

    def classify_batch(self, texts: List[str]) -> List[Optional[PromptGuardResult]]:
        """Classify multiple texts in a single batched forward pass.

        Parameters
        ----------
        texts : list[str]
            The input texts to classify.

        Returns
        -------
        list[PromptGuardResult or None]
            One result per input.  All None if transformers unavailable.
        """
        if not _HAS_TRANSFORMERS:
            return [None] * len(texts)

        if not texts:
            return []

        # Use the cached classify path for each input.
        # The LRU cache ensures repeated texts skip re-inference.
        # For truly new inputs, classify() delegates to _classify_impl
        # which loads the model lazily and runs single-input inference.
        #
        # NOTE: A future optimisation could batch-tokenize uncached inputs
        # and run a single forward pass via _classify_batch_impl.
        # For now, per-input classification with caching is correct and simple.
        results: List[Optional[PromptGuardResult]] = []
        for text in texts:
            results.append(self.classify(text))
        return results

    def get_injection_score(self, text: str) -> float:
        """Return P(INJECTION) + P(JAILBREAK) as a single float in [0, 1].

        Convenience method for pipeline integration.  Returns 0.0 if
        transformers is not available.
        """
        result = self.classify(text)
        if result is None:
            return 0.0
        injection_prob = result.raw_scores.get("INJECTION", 0.0)
        jailbreak_prob = result.raw_scores.get("JAILBREAK", 0.0)
        return min(injection_prob + jailbreak_prob, 1.0)

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    @property
    def device(self) -> str:
        """Return the configured device."""
        return self._device

    def cache_info(self):
        """Return LRU cache statistics."""
        return self._classify_cached.cache_info()

    def cache_clear(self):
        """Clear the LRU cache."""
        self._classify_cached.cache_clear()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _classify_impl(self, text: str) -> Optional[PromptGuardResult]:
        """Core classification logic (called by LRU cache wrapper).

        Parameters
        ----------
        text : str
            Input to classify.

        Returns
        -------
        PromptGuardResult or None
            None if model loading fails.
        """
        if not self._ensure_loaded():
            return None

        # Tokenize with truncation to model max length
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKENS,
            padding=False,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]

        raw_scores: Dict[str, float] = {}
        for idx, label_name in _LABEL_MAP.items():
            raw_scores[label_name] = round(float(probs[idx]), 6)

        top_idx = int(torch.argmax(probs))
        top_label = _LABEL_MAP.get(top_idx, "BENIGN")
        top_score = raw_scores[top_label]

        return PromptGuardResult(
            label=top_label,
            confidence=top_score,
            raw_scores=raw_scores,
        )

    def _classify_batch_impl(
        self, texts: List[str]
    ) -> List[Optional[PromptGuardResult]]:
        """Batched forward pass for multiple texts.

        Parameters
        ----------
        texts : list[str]
            Texts to classify.

        Returns
        -------
        list[PromptGuardResult or None]
        """
        if not self._ensure_loaded():
            return [None] * len(texts)

        # Tokenize batch with truncation and padding
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKENS,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.logits shape: (batch_size, num_classes)
        all_probs = torch.softmax(outputs.logits, dim=-1)

        results: List[Optional[PromptGuardResult]] = []
        for i in range(len(texts)):
            probs = all_probs[i]
            raw_scores: Dict[str, float] = {}
            for idx, label_name in _LABEL_MAP.items():
                raw_scores[label_name] = round(float(probs[idx]), 6)

            top_idx = int(torch.argmax(probs))
            top_label = _LABEL_MAP.get(top_idx, "BENIGN")
            top_score = raw_scores[top_label]

            results.append(PromptGuardResult(
                label=top_label,
                confidence=top_score,
                raw_scores=raw_scores,
            ))

        return results

    def _ensure_loaded(self) -> bool:
        """Lazy-load model and tokenizer.  Thread-safe via double-checked locking."""
        if self._model is not None:
            return True
        if self._init_failed:
            return False

        with self._lock:
            if self._model is not None:
                return True
            if self._init_failed:
                return False

            try:
                logger.info(
                    "Loading Prompt Guard model '%s' on device '%s'...",
                    self._model_name, self._device,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name,
                    **hf_tokenizer_kwargs(self._model_name),
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_name,
                    **hf_from_pretrained_kwargs(self._model_name),
                ).to(self._device)
                self._model.eval()
                logger.info("Prompt Guard model loaded successfully.")
                return True

            except Exception as exc:
                logger.warning(
                    "Failed to load Prompt Guard model '%s': %s",
                    self._model_name, exc,
                )
                self._init_failed = True
                return False


# ---------------------------------------------------------------------------
# Module-level singleton for pipeline integration
# ---------------------------------------------------------------------------
_singleton: Optional[PromptGuardClassifier] = None
_singleton_lock = threading.Lock()


def get_promptguard_classifier() -> Optional[PromptGuardClassifier]:
    """Return the module-level singleton PromptGuardClassifier.

    Returns None if:
      - ``transformers`` is not installed
      - ``NA0S_ENABLE_PROMPTGUARD`` is not set to a truthy value

    Thread-safe with double-checked locking.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    if not _HAS_TRANSFORMERS:
        return None
    if not PromptGuardClassifier.is_enabled():
        return None

    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        try:
            _singleton = PromptGuardClassifier()
            return _singleton
        except Exception as exc:
            logger.debug("PromptGuardClassifier singleton init failed: %s", exc)
            return None


def reset_singleton() -> None:
    """Reset the module-level singleton.  Used in tests only."""
    global _singleton
    with _singleton_lock:
        _singleton = None


def get_promptguard_score(text: str) -> float:
    """Return P(INJECTION) + P(JAILBREAK) from the singleton classifier.

    Convenience function for pipeline integration.  Returns 0.0 when
    PromptGuard is disabled, unavailable, or fails.
    """
    clf = get_promptguard_classifier()
    if clf is None:
        return 0.0
    try:
        return clf.get_injection_score(text)
    except Exception as exc:
        logger.debug("PromptGuard scoring failed: %s", exc)
        return 0.0
