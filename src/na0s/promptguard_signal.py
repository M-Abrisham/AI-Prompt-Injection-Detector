"""Convenience wrapper: expose Prompt Guard as a single-float signal.

This module provides ``get_promptguard_score(text) -> float`` which returns
the injection probability in [0.0, 1.0].  It is designed to be called from
the cascade/ensemble pipeline as a lightweight scoring function.

Behaviour:
  - If ``NA0S_PROMPTGUARD_ENABLED`` is set to ``0``/``false``/``no`` → disabled.
  - If set to ``1``/``true``/``yes`` → enabled.
  - If **unset** → auto-detect (enabled when ``transformers`` is importable).
  - Returns 0.0 if ``transformers`` is not installed (graceful degradation).
  - Uses a module-level singleton to avoid reloading the model on every call.
  - The score is ``P(INJECTION) + P(JAILBREAK)``, i.e. the probability that
    the input is NOT benign according to Prompt Guard 2.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment toggle — auto-detects when unset
# ---------------------------------------------------------------------------
_ENV_KEY = "NA0S_PROMPTGUARD_ENABLED"


def _is_enabled() -> bool:
    """Return True if the Prompt Guard signal should be active.

    Priority:
      1. Env var explicitly set to a falsy value (0/false/no) → False
      2. Env var explicitly set to a truthy value (1/true/yes) → True
      3. Env var **unset** → auto-detect (True when ``transformers``
         is importable, False otherwise)
    """
    raw = os.environ.get(_ENV_KEY)
    if raw is None:
        # Auto-detect: enable when transformers is available
        return importlib.util.find_spec("transformers") is not None
    return raw.lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Singleton classifier instance
# ---------------------------------------------------------------------------
_instance: Optional[object] = None
_instance_lock = threading.Lock()


def _get_instance():
    """Return the singleton PromptGuardClassifier, or None if unavailable."""
    global _instance
    if _instance is not None:
        return _instance

    with _instance_lock:
        if _instance is not None:
            return _instance

        try:
            from .promptguard import PromptGuardClassifier

            if not PromptGuardClassifier.is_available():
                return None

            _instance = PromptGuardClassifier()
            return _instance

        except Exception as exc:
            logger.debug("PromptGuard singleton init failed: %s", exc)
            return None


def reset_singleton() -> None:
    """Reset the module-level singleton.  Used in tests only."""
    global _instance
    with _instance_lock:
        _instance = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_promptguard_score(text: str) -> float:
    """Return injection probability from Prompt Guard 2.

    Parameters
    ----------
    text : str
        The input text to score.

    Returns
    -------
    float
        A value in [0.0, 1.0] representing the probability that *text*
        is an injection or jailbreak attempt.  Returns 0.0 when Prompt
        Guard is disabled or unavailable.
    """
    if not _is_enabled():
        return 0.0

    clf = _get_instance()
    if clf is None:
        return 0.0

    try:
        result = clf.classify(text)
        probs = result.get("probabilities", {})
        # Score = P(INJECTION) + P(JAILBREAK), i.e. 1 - P(BENIGN)
        injection_prob = probs.get("INJECTION", 0.0)
        jailbreak_prob = probs.get("JAILBREAK", 0.0)
        return min(injection_prob + jailbreak_prob, 1.0)
    except Exception as exc:
        logger.debug("PromptGuard scoring failed: %s", exc)
        return 0.0
