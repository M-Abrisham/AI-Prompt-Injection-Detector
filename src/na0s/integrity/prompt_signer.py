"""HMAC/JWT-style prompt integrity verification.

Signs system prompts with HMAC-SHA256 to detect mid-pipeline tampering.
Includes nonce + timestamp for replay protection.

Gated by ``NA0S_PROMPT_SIGNING=1`` env var (default: disabled).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
import warnings
from collections import deque


class PromptSigner:
    """Sign and verify prompts using HMAC-SHA256 with nonce and timestamp."""

    _NONCE_CACHE_LIMIT = 10_000

    def __init__(self, secret_key: str | None = None) -> None:
        if secret_key is not None:
            self._key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        else:
            env_key = os.environ.get("NA0S_PROMPT_SIGN_KEY")
            if env_key:
                self._key = env_key.encode()
            else:
                self._key = secrets.token_bytes(32)
                warnings.warn(
                    "No signing key provided — generated a random ephemeral key. "
                    "Set NA0S_PROMPT_SIGN_KEY for persistent signing.",
                    stacklevel=2,
                )
        self._used_nonces: deque[str] = deque(maxlen=self._NONCE_CACHE_LIMIT)
        self._used_nonce_set: set[str] = set()

    # ------------------------------------------------------------------
    @staticmethod
    def is_enabled() -> bool:
        """Return True when prompt signing is activated via env var."""
        return os.environ.get("NA0S_PROMPT_SIGNING", "0") == "1"

    # ------------------------------------------------------------------
    def sign(self, prompt: str) -> dict:
        """Sign *prompt* and return a dict with signature metadata."""
        nonce = secrets.token_hex(8)  # 16 hex chars
        timestamp = time.time()
        message = f"{nonce}:{timestamp}:{prompt}"
        digest = hmac.new(self._key, message.encode(), hashlib.sha256).hexdigest()
        return {
            "prompt": prompt,
            "signature": digest,
            "nonce": nonce,
            "timestamp": timestamp,
        }

    # ------------------------------------------------------------------
    def verify(self, signed_prompt: dict, max_age_seconds: int = 300) -> dict:
        """Verify a previously signed prompt dict.

        Returns ``{"valid": bool, "reason": str}``.
        """
        try:
            prompt = signed_prompt["prompt"]
            signature = signed_prompt["signature"]
            nonce = signed_prompt["nonce"]
            timestamp = signed_prompt["timestamp"]
        except KeyError as exc:
            return {"valid": False, "reason": f"missing field: {exc}"}

        # Replay check
        if nonce in self._used_nonce_set:
            return {"valid": False, "reason": "nonce already used — possible replay"}

        # Freshness check
        age = time.time() - timestamp
        if age > max_age_seconds:
            return {"valid": False, "reason": f"signature expired ({age:.1f}s > {max_age_seconds}s)"}
        if age < 0:
            return {"valid": False, "reason": "timestamp is in the future"}

        # HMAC verification (constant-time)
        message = f"{nonce}:{timestamp}:{prompt}"
        expected = hmac.new(self._key, message.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return {"valid": False, "reason": "signature mismatch — prompt tampered"}

        # Record nonce
        if len(self._used_nonce_set) >= self._NONCE_CACHE_LIMIT:
            evicted = self._used_nonces[0]  # oldest
            self._used_nonce_set.discard(evicted)
        self._used_nonces.append(nonce)
        self._used_nonce_set.add(nonce)

        return {"valid": True, "reason": ""}
