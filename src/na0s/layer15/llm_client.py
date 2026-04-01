"""Lightweight LLM client for Layer 15 intelligence features.

Follows the same pattern as Na0S's llm_judge.py: OpenAI-compatible API,
optional, graceful degradation when no API key is configured.

Environment variables:
  NAOS_L15_LLM_MODEL: Model name (default: gpt-4o-mini)
  NAOS_L15_LLM_BASE_URL: API base URL (default: https://api.openai.com/v1)
  NAOS_L15_LLM_API_KEY: API key (default: empty = LLM disabled)
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency — graceful degradation if openai is not installed
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]
    _HAS_OPENAI = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_MAX_RETRY_DELAY = 30  # seconds


class Layer15LLMClient:
    """OpenAI-compatible LLM client for Layer 15.

    Returns ``None`` from :meth:`chat` whenever the LLM is unavailable
    (missing API key, missing ``openai`` package, or API failure), so
    callers can fall back to non-LLM logic without exception handling.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model or os.getenv("NAOS_L15_LLM_MODEL", _DEFAULT_MODEL)
        self.base_url = base_url or os.getenv(
            "NAOS_L15_LLM_BASE_URL", _DEFAULT_BASE_URL
        )
        self._api_key = api_key or os.getenv("NAOS_L15_LLM_API_KEY", "")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Optional[OpenAI] = None  # type: ignore[assignment]

        if _HAS_OPENAI and self._api_key:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                timeout=self._timeout,
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the LLM is configured (package + API key)."""
        if not _HAS_OPENAI:
            return False
        key = os.getenv("NAOS_L15_LLM_API_KEY", "")
        return bool(key)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Send a chat completion request and return the assistant reply.

        Returns ``None`` when the LLM is unavailable or the call fails
        (after retries).
        """
        if self._client is None:
            logger.debug("Layer15LLMClient: no client configured, returning None")
            return None

        start = time.monotonic()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self._call_with_retry(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning(
                "Layer15LLMClient: chat failed after retries "
                "(model=%s, elapsed=%.0fms): %s",
                self.model,
                elapsed_ms,
                exc,
            )
            return None

        elapsed_ms = (time.monotonic() - start) * 1000
        content = response.choices[0].message.content or ""

        # Log usage stats
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = (
            getattr(usage, "completion_tokens", 0) if usage else 0
        )
        logger.info(
            "Layer15LLMClient: model=%s tokens_in=%d tokens_out=%d latency=%.0fms",
            self.model,
            prompt_tokens,
            completion_tokens,
            elapsed_ms,
        )

        return content

    # ------------------------------------------------------------------
    # Internal — retry with exponential backoff
    # ------------------------------------------------------------------

    def _call_with_retry(self, *, messages, temperature, max_tokens):
        """Call the API with exponential backoff on transient errors."""
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                if status_code is None:
                    resp = getattr(exc, "response", None)
                    if resp is not None:
                        status_code = getattr(resp, "status_code", None)

                is_retryable = status_code in (429, 500, 503)
                if not is_retryable or attempt >= self._max_retries:
                    raise

                delay = min(
                    (2**attempt) + random.uniform(0, 1),
                    _MAX_RETRY_DELAY,
                )
                logger.warning(
                    "Layer15LLMClient: retry %d/%d after HTTP %s (delay=%.1fs)",
                    attempt + 1,
                    self._max_retries,
                    status_code,
                    delay,
                )
                time.sleep(delay)

        raise last_exc  # pragma: no cover
