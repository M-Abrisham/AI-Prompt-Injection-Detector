"""Convenience wrapper: expose Prompt Guard as a single-float signal.

This module provides ``get_promptguard_score(text) -> float`` which returns
the injection probability in [0.0, 1.0].  It is designed to be called from
the cascade/ensemble pipeline as a lightweight scoring function.

Behaviour:
  - Returns 0.0 if ``transformers`` is not installed (graceful degradation).
  - Returns 0.0 if disabled via ``NA0S_PROMPTGUARD_ENABLED`` env var
    (default: ``"0"`` — disabled, since transformers is an optional dep).
  - Uses a module-level singleton to avoid reloading the model on every call.
  - The score is ``P(INJECTION) + P(JAILBREAK)``, i.e. the probability that
    the input is NOT benign according to Prompt Guard 2.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment toggle — disabled by default (requires extra deps)
# ---------------------------------------------------------------------------
_ENV_KEY = "NA0S_PROMPTGUARD_ENABLED"


def _is_enabled() -> bool:
    """Return True if the Prompt Guard signal is enabled via env var."""
    return os.environ.get(_ENV_KEY, "0").lower() in ("1", "true", "yes")


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
