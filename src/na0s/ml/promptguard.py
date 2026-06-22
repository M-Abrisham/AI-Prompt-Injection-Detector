"""Layer 4 (P2): Meta Prompt Guard 2 classifier for prompt injection detection.

This module wraps Meta's Prompt-Guard-2-22M model (a fine-tuned mDeBERTa
22M-parameter sequence classifier) as an additional injection/jailbreak
detection signal.  The model outputs three classes: BENIGN, INJECTION,
and JAILBREAK.

Key design decisions:
  - Optional dependency: ``transformers`` is NOT required.  If missing,
    ``is_available()`` returns False and ``classify()`` raises RuntimeError.
  - Lazy model loading: the HuggingFace model and tokenizer are loaded on
    the first ``classify()`` call, not at import time.
  - Thread-safe: a threading.Lock guards lazy initialisation so concurrent
    callers do not race on model loading.
  - Input truncation: text is truncated to 512 tokens (the model maximum).

Usage::

    from na0s.promptguard import PromptGuardClassifier

    if PromptGuardClassifier.is_available():
        clf = PromptGuardClassifier()
        result = clf.classify("Ignore all previous instructions")
        # result == {
        #     "label": "INJECTION",
        #     "score": 0.97,
        #     "probabilities": {"BENIGN": 0.01, "INJECTION": 0.97, "JAILBREAK": 0.02},
        # }
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import: transformers is an optional dependency
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
# Label mapping — Prompt-Guard-2-22M uses integer label ids
# ---------------------------------------------------------------------------
_LABEL_MAP = {0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"}

# Model maximum sequence length (mDeBERTa limit)
_MAX_TOKENS = 512

# Default model identifier on HuggingFace Hub
DEFAULT_MODEL_NAME = "meta-llama/Prompt-Guard-2-22M"


class PromptGuardClassifier:
    """Wrapper around Meta Prompt-Guard-2-22M for injection/jailbreak detection.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or local path.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, etc.).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._device = device

        # Lazy-loaded state
        self._tokenizer: Optional[object] = None
        self._model: Optional[object] = None
        self._lock = threading.Lock()
        self._init_failed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the ``transformers`` library is installed."""
        return _HAS_TRANSFORMERS

    def classify(self, text: str) -> Dict:
        """Classify *text* as BENIGN, INJECTION, or JAILBREAK.

        Parameters
        ----------
        text : str
            The input text to classify.

        Returns
        -------
        dict
            ``{"label": str, "score": float, "probabilities": dict}``
            where *label* is the top predicted class, *score* is its
            probability, and *probabilities* maps each class name to
            its softmax probability.

        Raises
        ------
        RuntimeError
            If ``transformers`` is not installed.
        """
        if not _HAS_TRANSFORMERS:
            raise RuntimeError(
                "PromptGuardClassifier requires the `transformers` package. "
                "Install it with: pip install transformers torch"
            )

        if not self._ensure_loaded():
            raise RuntimeError(
                "Failed to load Prompt Guard model '{}'. "
                "Check logs for details.".format(self._model_name)
            )

        # Tokenize with truncation to model max length
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKENS,
            padding=False,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass (no gradient computation needed)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Softmax over logits to get per-class probabilities
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Build probabilities dict
        probabilities = {}
        for idx, label_name in _LABEL_MAP.items():
            probabilities[label_name] = round(float(probs[idx]), 6)

        # Top prediction
        top_idx = int(torch.argmax(probs))
        top_label = _LABEL_MAP.get(top_idx, "BENIGN")
        top_score = probabilities[top_label]

        return {
            "label": top_label,
            "score": top_score,
            "probabilities": probabilities,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Lazy-load model and tokenizer.  Thread-safe with double-checked locking."""
        if self._model is not None:
            return True
        if self._init_failed:
            return False

        with self._lock:
            # Re-check after acquiring lock
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
