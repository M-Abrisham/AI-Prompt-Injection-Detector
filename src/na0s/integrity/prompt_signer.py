"""HMAC/JWT-style prompt integrity verification.

Signs system prompts with HMAC-SHA256 to detect mid-pipeline tampering.
Includes nonce + timestamp for replay protection.

Gated by ``NA0S_PROMPT_SIGNING=1`` env var (default: disabled).

Security properties (S2 hardening)
----------------------------------
* **Fail-closed type validation** — a malformed ``signed_prompt`` (wrong field
  types) returns ``{valid: False, reason: 'malformed field'}`` instead of
  raising ``TypeError`` out of :meth:`verify`.
* **Transport-invariant canonicalization** — the MAC is computed over a
  length-prefixed byte framing of ``(nonce, timestamp_ms, prompt)`` where the
  timestamp is canonicalized to integer milliseconds. This survives a JSON
  round-trip (no float-repr drift) and is unambiguous regardless of what
  characters the prompt contains (no ``:`` delimiter to spoof).
* **Clock-skew leeway** — a future timestamp is only rejected when it is more
  than ``_CLOCK_SKEW_LEEWAY_SECONDS`` ahead of local time.
* **Surrogate-safe encoding** — both ``sign`` and ``verify`` encode with
  ``surrogatepass`` so a lone-surrogate prompt produces identical bytes and
  never crashes.
* **Nonce single-use with idempotent re-verify** — a nonce is recorded only
  after a *successful* signature check (an attacker cannot burn nonces with
  forged signatures). Re-presenting the same nonce with the same signature is
  treated as an idempotent re-verify (``valid: True``); a different
  signature/prompt for a used nonce is a true replay and is rejected.
* **Strict key management (opt-in)** — ``NA0S_PROMPT_SIGN_STRICT=1`` fails
  closed at construction when no persistent key is configured. A configured
  key shorter than 32 bytes warns.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import struct
import time
import warnings
from collections import deque
from typing import Any, Protocol, runtime_checkable

# Typical NTP synchronisation error between two well-behaved hosts is well
# under a second; 5s is a conservative bound that tolerates poll-interval slop
# without meaningfully widening the replay/forgery window.
_CLOCK_SKEW_LEEWAY_SECONDS = 5

# Minimum recommended key length (bytes). HMAC-SHA256's security degrades for
# keys shorter than the block/output size; 32 bytes == SHA-256 output size.
_MIN_KEY_BYTES = 32

# Domain-separation tag baked into the canonical MAC input so signatures from
# this signer cannot be confused with any other HMAC use of the same key.
_DOMAIN = b"na0s.prompt_signer.v1"


@runtime_checkable
class ReplayStore(Protocol):
    """Pluggable persisted replay store.

    A store records the *signature* last seen for a nonce so that an idempotent
    re-verify (same nonce + same signature) can be distinguished from a true
    replay (same nonce, different signature). Implementations must be local /
    keyless — no network or cloud dependency.
    """

    def get(self, nonce: str) -> str | None:
        """Return the signature recorded for *nonce*, or ``None`` if unused."""

    def add(self, nonce: str, signature: str) -> None:
        """Record *signature* against *nonce* (single-use marker)."""


class _InMemoryReplayStore:
    """Default bounded in-memory replay store (deque + dict).

    Preserves the original single-process behaviour: a FIFO eviction cache of
    at most ``maxlen`` nonces. Stores the signature per nonce to support
    idempotent re-verify.
    """

    def __init__(self, maxlen: int) -> None:
        self._maxlen = maxlen
        self._order: deque[str] = deque(maxlen=maxlen)
        self._sigs: dict[str, str] = {}

    def get(self, nonce: str) -> str | None:
        return self._sigs.get(nonce)

    def add(self, nonce: str, signature: str) -> None:
        if nonce in self._sigs:
            # Already recorded; keep the original signature for idempotency.
            return
        if len(self._sigs) >= self._maxlen and self._order:
            evicted = self._order[0]  # oldest
            self._sigs.pop(evicted, None)
        self._order.append(nonce)
        self._sigs[nonce] = signature


class PromptSigner:
    """Sign and verify prompts using HMAC-SHA256 with nonce and timestamp."""

    _NONCE_CACHE_LIMIT = 10_000

    def __init__(
        self,
        secret_key: str | bytes | None = None,
        *,
        replay_store: ReplayStore | None = None,
        key_id: str | None = None,
    ) -> None:
        strict = os.environ.get("NA0S_PROMPT_SIGN_STRICT", "0") == "1"
        self._key_id = key_id

        if secret_key is not None:
            self._key = secret_key.encode() if isinstance(secret_key, str) else secret_key
            if len(self._key) < _MIN_KEY_BYTES:
                warnings.warn(
                    f"Signing key is shorter than {_MIN_KEY_BYTES} bytes "
                    f"({len(self._key)}); use a longer key for full HMAC-SHA256 strength.",
                    stacklevel=2,
                )
        else:
            env_key = os.environ.get("NA0S_PROMPT_SIGN_KEY")
            if env_key:
                self._key = env_key.encode()
                if len(self._key) < _MIN_KEY_BYTES:
                    warnings.warn(
                        f"NA0S_PROMPT_SIGN_KEY is shorter than {_MIN_KEY_BYTES} bytes "
                        f"({len(self._key)}); use a longer key for full HMAC-SHA256 strength.",
                        stacklevel=2,
                    )
            elif strict:
                # Fail closed: no persistent key configured but strict mode is on.
                raise RuntimeError(
                    "NA0S_PROMPT_SIGN_STRICT=1 but no persistent signing key is "
                    "configured. Set NA0S_PROMPT_SIGN_KEY (or pass secret_key) to "
                    "a persistent >= 32-byte secret."
                )
            else:
                self._key = secrets.token_bytes(32)
                warnings.warn(
                    "No signing key provided — generated a random ephemeral key. "
                    "Set NA0S_PROMPT_SIGN_KEY for persistent signing.",
                    stacklevel=2,
                )

        self._store: ReplayStore = (
            replay_store if replay_store is not None
            else _InMemoryReplayStore(self._NONCE_CACHE_LIMIT)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def is_enabled() -> bool:
        """Return True when prompt signing is activated via env var."""
        return os.environ.get("NA0S_PROMPT_SIGNING", "0") == "1"

    # ------------------------------------------------------------------
    @staticmethod
    def _to_ms(timestamp: float) -> int:
        """Canonicalize a float-seconds timestamp to integer milliseconds.

        Integer ms survives a JSON round-trip exactly (no float-repr drift) and
        is the value bound into the MAC. ``round`` (not ``int``) so that the
        same wall-clock instant maps to the same ms regardless of tiny float
        noise.
        """
        return int(round(timestamp * 1000))

    @classmethod
    def _canonical_bytes(cls, nonce: str, timestamp_ms: int, prompt: str) -> bytes:
        """Build the unambiguous, transport-invariant MAC input.

        Framing: domain tag, then each variable field as a 4-byte big-endian
        length prefix followed by its ``surrogatepass``-encoded UTF-8 bytes,
        with the timestamp as a fixed-width signed 64-bit integer. Length
        prefixes mean no in-band delimiter can be spoofed by the prompt, and
        the integer ms means JSON round-tripping cannot perturb the bytes.

        ``sign`` and ``verify`` MUST call this identically.
        """
        nonce_b = nonce.encode("utf-8", "surrogatepass")
        prompt_b = prompt.encode("utf-8", "surrogatepass")
        return b"".join(
            (
                _DOMAIN,
                struct.pack(">I", len(nonce_b)),
                nonce_b,
                struct.pack(">q", timestamp_ms),
                struct.pack(">I", len(prompt_b)),
                prompt_b,
            )
        )

    def _compute_digest(self, nonce: str, timestamp_ms: int, prompt: str) -> str:
        message = self._canonical_bytes(nonce, timestamp_ms, prompt)
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    # ------------------------------------------------------------------
    def sign(self, prompt: str) -> dict:
        """Sign *prompt* and return a dict with signature metadata."""
        nonce = secrets.token_hex(8)  # 16 hex chars
        timestamp = time.time()
        timestamp_ms = self._to_ms(timestamp)
        digest = self._compute_digest(nonce, timestamp_ms, prompt)
        result = {
            "prompt": prompt,
            "signature": digest,
            "nonce": nonce,
            "timestamp": timestamp,
        }
        if self._key_id is not None:
            result["key_id"] = self._key_id
        return result

    # ------------------------------------------------------------------
    def verify(self, signed_prompt: dict, max_age_seconds: int = 300) -> dict:
        """Verify a previously signed prompt dict.

        Returns ``{"valid": bool, "reason": str}``. Fails closed on any
        malformed input.
        """
        try:
            prompt = signed_prompt["prompt"]
            signature = signed_prompt["signature"]
            nonce = signed_prompt["nonce"]
            timestamp = signed_prompt["timestamp"]
        except (KeyError, TypeError) as exc:
            # TypeError: signed_prompt is not a mapping at all.
            return {"valid": False, "reason": f"missing field: {exc}"}

        # Fail-closed type validation. ``bool`` is an ``int`` subclass, so a
        # boolean timestamp must be rejected explicitly.
        if (
            not isinstance(prompt, str)
            or not isinstance(signature, str)
            or not isinstance(nonce, str)
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
        ):
            return {"valid": False, "reason": "malformed field"}

        try:
            timestamp_ms = self._to_ms(timestamp)

            # Idempotent re-verify vs. true replay. We must verify the HMAC of a
            # used nonce against the *recorded* signature before deciding; a
            # forged signature for a used nonce is a true replay.
            recorded_sig = self._store.get(nonce)

            # Freshness check (uses the float timestamp from the dict).
            age = time.time() - timestamp
            if age > max_age_seconds:
                return {
                    "valid": False,
                    "reason": f"signature expired ({age:.1f}s > {max_age_seconds}s)",
                }
            if age < -_CLOCK_SKEW_LEEWAY_SECONDS:
                return {"valid": False, "reason": "timestamp is in the future"}

            # HMAC verification (constant-time) over the canonical bytes.
            expected = self._compute_digest(nonce, timestamp_ms, prompt)
            sig_ok = hmac.compare_digest(expected, signature)

            if recorded_sig is not None:
                # Nonce already used. Only an exact re-presentation of the same,
                # genuinely-valid signature is an idempotent re-verify.
                if sig_ok and hmac.compare_digest(recorded_sig, signature):
                    return {"valid": True, "reason": ""}
                return {"valid": False, "reason": "nonce already used — possible replay"}

            if not sig_ok:
                # Do NOT record the nonce: an attacker must not be able to burn
                # nonces with forged signatures.
                return {"valid": False, "reason": "signature mismatch — prompt tampered"}

            # First successful verification for this nonce — record it.
            self._store.add(nonce, signature)
            return {"valid": True, "reason": ""}
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return {"valid": False, "reason": f"malformed field: {exc}"}
