"""Canary token verifier \u2014 DEPRECATED, NOT a real tamper gate.

.. deprecated::
    Use :class:`na0s.integrity.prompt_signer.PromptSigner` instead. ``PromptSigner``
    is the canonical, content-and-key-bound tamper gate (HMAC-SHA256 over the whole
    prompt, with nonce + timestamp replay protection).

``CanaryTokenVerifier`` is a strictly-weaker, orphaned duplicate of
``PromptSigner``. It embeds the canary in **plaintext** and only checks that the
canary *string* is still present, which is NOT integrity verification:

* Any tamper **outside** the canary line is undetected \u2014 the body can be rewritten
  freely as long as the ``[INTEGRITY_CHECK: ...]`` marker survives.
* A **prepended forged** ``[INTEGRITY_CHECK: ...]`` marker bypasses the check: the
  first-match regex matches the attacker's marker, not the genuine trailing one.
* The canary value is **not a secret** and is not key-bound, so an attacker who
  sees the prompt can reproduce a "valid" marker at will.

It is **NOT wired into the pipeline** (imported nowhere in ``src/``). It is kept
importable for backward compatibility only; do not build new tamper detection on
it. Use :class:`~na0s.integrity.prompt_signer.PromptSigner`.

Historically gated by ``NA0S_CANARY_VERIFY=1`` (default: disabled).
"""

from __future__ import annotations

import hmac
import os
import re
import secrets
import warnings

_DEPRECATION_MSG = (
    "CanaryTokenVerifier is deprecated and is NOT a real tamper gate: it embeds "
    "the canary in plaintext and only checks the canary string is present, so any "
    "tamper outside the canary line \u2014 or a prepended forged [INTEGRITY_CHECK:] "
    "marker \u2014 is undetected. It is wired nowhere in the pipeline. Use "
    "na0s.integrity.prompt_signer.PromptSigner (content- and key-bound HMAC-SHA256) "
    "as the canonical tamper gate instead."
)


class CanaryTokenVerifier:
    """DEPRECATED \u2014 embed/verify plaintext canary markers; not a real tamper gate.

    .. deprecated::
        Use :class:`na0s.integrity.prompt_signer.PromptSigner`. This class only
        checks that a plaintext canary string is still present, which does not
        bind the prompt content to a key and is trivially bypassed (see module
        docstring). It is orphaned (not wired into the pipeline).
    """

    _CANARY_PREFIX = "__NA0S_VERIFY_"
    _CANARY_SUFFIX = "__"
    _INTEGRITY_RE = re.compile(
        r"\[INTEGRITY_CHECK:\s*(" + re.escape(_CANARY_PREFIX) + r"[0-9a-f]+" + re.escape(_CANARY_SUFFIX) + r")\]"
    )

    def __init__(self) -> None:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

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

        DEPRECATED and weak: this only confirms the plaintext canary *string*
        survived; it does NOT verify the rest of the prompt and is bypassable
        by a prepended forged marker (first-match regex). The comparison below
        uses ``hmac.compare_digest`` purely for an exact byte-string compare —
        the canary is not a secret, so there is no real timing side-channel to
        defend here. Use :class:`na0s.integrity.prompt_signer.PromptSigner` for
        actual content/key-bound tamper detection.

        Returns ``{"intact": bool, "reason": str}``.
        """
        match = self._INTEGRITY_RE.search(prompt)
        if match is None:
            return {"intact": False, "reason": "canary stripped \u2014 prompt tampered"}
        found_canary = match.group(1)
        if hmac.compare_digest(found_canary, expected_canary):
            return {"intact": True, "reason": ""}
        return {"intact": False, "reason": "canary stripped \u2014 prompt tampered"}
