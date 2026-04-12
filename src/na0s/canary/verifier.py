"""Canary token verifier \u2014 embed and verify canary tokens for mid-pipeline tampering.

Embeds a canary at prompt creation time. At inference time, verifies the canary
is still present and unmodified. If the canary was stripped or modified, the
prompt was tampered with mid-pipeline.

Gated by ``NA0S_CANARY_VERIFY=1`` env var (default: disabled).
"""

from __future__ import annotations

import hmac
import os
import re
import secrets


class CanaryTokenVerifier:
    """Embed and verify integrity canary tokens in prompts."""

    _CANARY_PREFIX = "__NA0S_VERIFY_"
    _CANARY_SUFFIX = "__"
    _INTEGRITY_RE = re.compile(
        r"\[INTEGRITY_CHECK:\s*(" + re.escape(_CANARY_PREFIX) + r"[0-9a-f]+" + re.escape(_CANARY_SUFFIX) + r")\]"
    )

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    @staticmethod
    def is_enabled() -> bool:
        """Return True when canary verification is activated via env var."""
        return os.environ.get("NA0S_CANARY_VERIFY", "0") == "1"

    # ------------------------------------------------------------------
    def embed(self, prompt: str) -> tuple[str, str]:
        """Embed a verification canary into *prompt*.

        Returns ``(modified_prompt, canary_value)``.
        """
        token = secrets.token_hex(8)
        canary = f"{self._CANARY_PREFIX}{token}{self._CANARY_SUFFIX}"
        modified = f"{prompt}\n[INTEGRITY_CHECK: {canary}]"
        return modified, canary

    # ------------------------------------------------------------------
    def verify(self, prompt: str, expected_canary: str) -> dict:
        """Check whether *expected_canary* is still present in *prompt*.

        Uses ``hmac.compare_digest`` for timing-safe comparison to prevent
        an attacker from brute-forcing the canary value via timing
        side-channels.

        Returns ``{"intact": bool, "reason": str}``.
        """
        match = self._INTEGRITY_RE.search(prompt)
        if match is None:
            return {"intact": False, "reason": "canary stripped \u2014 prompt tampered"}
        found_canary = match.group(1)
        if hmac.compare_digest(found_canary, expected_canary):
            return {"intact": True, "reason": ""}
        return {"intact": False, "reason": "canary stripped \u2014 prompt tampered"}
