"""S2 hardening tests for the canonical PromptSigner HMAC tamper gate.

Imports the CANONICAL module (``na0s.integrity.prompt_signer``), never the
deprecated ``na0s.prompt_signer`` shim. Every test is a fail-before / pass-after
guard for one S2 hardening item.
"""

from __future__ import annotations

import json
import time

import pytest

from na0s.integrity.prompt_signer import (
    _CLOCK_SKEW_LEEWAY_SECONDS,
    _InMemoryReplayStore,
    PromptSigner,
)

# A 32-byte key so the < 32-byte warning never fires in the happy-path tests.
KEY = "k" * 32


# ----------------------------------------------------------------------
# 1. Fail-closed type validation
# ----------------------------------------------------------------------
class TestMalformedTypes:
    def test_verify_rejects_malformed_types(self):
        signer = PromptSigner(secret_key=KEY)
        signed = signer.sign("hello")

        # Non-float/int timestamp (a str) must NOT raise — fail closed.
        bad_ts = dict(signed)
        bad_ts["timestamp"] = "not-a-number"
        res = signer.verify(bad_ts)
        assert res["valid"] is False
        assert res["reason"] == "malformed field"

        # bool timestamp is an int subclass but must be rejected.
        bad_bool = dict(signed)
        bad_bool["timestamp"] = True
        res = signer.verify(bad_bool)
        assert res["valid"] is False
        assert res["reason"] == "malformed field"

        # Non-str signature.
        bad_sig = dict(signed)
        bad_sig["signature"] = 12345
        res = signer.verify(bad_sig)
        assert res["valid"] is False
        assert res["reason"] == "malformed field"

        # Non-str prompt / nonce.
        for field in ("prompt", "nonce"):
            bad = dict(signed)
            bad[field] = object()
            res = signer.verify(bad)
            assert res["valid"] is False, field
            assert res["reason"] == "malformed field", field

    def test_verify_non_mapping_fails_closed(self):
        signer = PromptSigner(secret_key=KEY)
        # A non-dict argument must not raise.
        res = signer.verify(["not", "a", "dict"])
        assert res["valid"] is False


# ----------------------------------------------------------------------
# 2. Transport-invariant canonicalization (JSON round-trip)
# ----------------------------------------------------------------------
class TestJsonRoundTrip:
    def test_verify_survives_json_roundtrip(self):
        signer = PromptSigner(secret_key=KEY)
        signed = signer.sign("payload that goes over the wire")
        # Simulate JSON transport (float-repr drift would break a naive f-string MAC).
        round_tripped = json.loads(json.dumps(signed))
        res = signer.verify(round_tripped)
        assert res["valid"] is True, res
        assert res["reason"] == ""

    def test_float_timestamp_drift_does_not_break_mac(self):
        signer = PromptSigner(secret_key=KEY)
        signed = signer.sign("p")
        # Perturb the float below the ms-canonicalization resolution; the MAC is
        # bound to integer ms, so a sub-ms change must still verify.
        signed2 = dict(signed)
        signed2["timestamp"] = signed["timestamp"] + 1e-7
        res = signer.verify(signed2)
        assert res["valid"] is True, res


# ----------------------------------------------------------------------
# 3. Clock-skew leeway
# ----------------------------------------------------------------------
class TestClockSkew:
    def test_verify_tolerates_small_future_skew(self):
        signer = PromptSigner(secret_key=KEY)
        prompt = "future-skew prompt"
        nonce = "a" * 16
        # A timestamp slightly in the future but within the leeway window.
        future_ts = time.time() + (_CLOCK_SKEW_LEEWAY_SECONDS - 1)
        ts_ms = signer._to_ms(future_ts)
        sig = signer._compute_digest(nonce, ts_ms, prompt)
        res = signer.verify(
            {"prompt": prompt, "signature": sig, "nonce": nonce, "timestamp": future_ts}
        )
        assert res["valid"] is True, res

    def test_verify_rejects_far_future(self):
        signer = PromptSigner(secret_key=KEY)
        prompt = "far-future prompt"
        nonce = "b" * 16
        far_ts = time.time() + (_CLOCK_SKEW_LEEWAY_SECONDS + 60)
        ts_ms = signer._to_ms(far_ts)
        sig = signer._compute_digest(nonce, ts_ms, prompt)
        res = signer.verify(
            {"prompt": prompt, "signature": sig, "nonce": nonce, "timestamp": far_ts}
        )
        assert res["valid"] is False
        assert "future" in res["reason"]


# ----------------------------------------------------------------------
# 4. Surrogate-safe encoding
# ----------------------------------------------------------------------
class TestSurrogateSafe:
    def test_sign_verify_handles_surrogate_prompt(self):
        signer = PromptSigner(secret_key=KEY)
        # Lone surrogate — crashes a plain .encode() without surrogatepass.
        prompt = "danger \ud800 lone-surrogate"
        signed = signer.sign(prompt)  # must not raise
        res = signer.verify(signed)  # identical bytes both directions
        assert res["valid"] is True, res

    def test_surrogate_prompt_tamper_still_detected(self):
        signer = PromptSigner(secret_key=KEY)
        signed = signer.sign("clean \ud801 prompt")
        signed["prompt"] = "tampered \ud801 prompt"
        res = signer.verify(signed)
        assert res["valid"] is False
        assert "tampered" in res["reason"]


# ----------------------------------------------------------------------
# 5. Nonce single-use / idempotency
# ----------------------------------------------------------------------
class TestNonceIdempotency:
    def test_idempotent_reverify_not_replay(self):
        signer = PromptSigner(secret_key=KEY)
        signed = signer.sign("idem prompt")
        first = signer.verify(signed)
        assert first["valid"] is True
        # Same nonce + same signature presented again == idempotent re-verify.
        second = signer.verify(signed)
        assert second["valid"] is True, second
        assert second["reason"] == ""

    def test_true_replay_rejected(self):
        signer = PromptSigner(secret_key=KEY)
        signed = signer.sign("replay prompt")
        assert signer.verify(signed)["valid"] is True
        # Same nonce, but a DIFFERENT (forged / mismatched) signature == true replay.
        forged = dict(signed)
        forged["signature"] = "0" * 64
        res = signer.verify(forged)
        assert res["valid"] is False
        assert "replay" in res["reason"]

    def test_forged_sig_does_not_burn_nonce(self):
        """An attacker must not consume a nonce by presenting a forged signature
        before the legitimate holder verifies."""
        signer = PromptSigner(secret_key=KEY)
        signed = signer.sign("genuine prompt")
        forged = dict(signed)
        forged["signature"] = "f" * 64
        # Attacker tries first with a bad signature.
        bad = signer.verify(forged)
        assert bad["valid"] is False
        assert "tampered" in bad["reason"]  # NOT recorded as replay
        # Legitimate verify still succeeds — nonce was not burned.
        good = signer.verify(signed)
        assert good["valid"] is True, good


# ----------------------------------------------------------------------
# 6. Pluggable persisted replay store
# ----------------------------------------------------------------------
class TestPluggableStore:
    def test_injected_store_records_nonce(self):
        store = _InMemoryReplayStore(maxlen=100)
        signer = PromptSigner(secret_key=KEY, replay_store=store)
        signed = signer.sign("stored prompt")
        assert signer.verify(signed)["valid"] is True
        # The injected store now knows this nonce.
        assert store.get(signed["nonce"]) == signed["signature"]

    def test_custom_store_shares_replay_state(self):
        store = _InMemoryReplayStore(maxlen=100)
        s1 = PromptSigner(secret_key=KEY, replay_store=store)
        s2 = PromptSigner(secret_key=KEY, replay_store=store)
        signed = s1.sign("cross-instance prompt")
        assert s1.verify(signed)["valid"] is True
        # Second signer shares the store: a forged-sig replay is caught there too.
        forged = dict(signed)
        forged["signature"] = "0" * 64
        res = s2.verify(forged)
        assert res["valid"] is False
        assert "replay" in res["reason"]


# ----------------------------------------------------------------------
# 7. Delimiter robustness (prompt contains the old ':' delimiter)
# ----------------------------------------------------------------------
class TestDelimiterRobustness:
    def test_prompt_containing_colon_signs_and_verifies(self):
        signer = PromptSigner(secret_key=KEY)
        # The old f"{nonce}:{ts}:{prompt}" MAC was ambiguous for colon-bearing
        # prompts; the length-prefixed canonicalization is not.
        prompt = "role: system\ntime: 12:00:00 :: do thing"
        signed = signer.sign(prompt)
        assert signer.verify(signed)["valid"] is True

    def test_colon_prompt_not_confusable_across_fields(self):
        """A prompt crafted to look like 'nonce:ts:prompt' must not be confused
        with a different (nonce, ts, prompt) split."""
        signer = PromptSigner(secret_key=KEY)
        signed = signer.sign("x")
        # Move part of the nonce into the prompt with a colon — must NOT verify.
        nonce = signed["nonce"]
        ts = signed["timestamp"]
        attack = {
            "prompt": "x",
            "signature": signed["signature"],
            "nonce": nonce[:-1],
            "timestamp": ts,
        }
        # Different nonce framing -> different MAC -> reject.
        res = signer.verify(attack)
        assert res["valid"] is False


# ----------------------------------------------------------------------
# 8. Strict key management
# ----------------------------------------------------------------------
class TestStrictKeyMgmt:
    def test_strict_mode_fails_closed_without_key(self, monkeypatch):
        monkeypatch.setenv("NA0S_PROMPT_SIGN_STRICT", "1")
        monkeypatch.delenv("NA0S_PROMPT_SIGN_KEY", raising=False)
        with pytest.raises(RuntimeError, match="no persistent signing key"):
            PromptSigner()

    def test_strict_mode_ok_with_env_key(self, monkeypatch):
        monkeypatch.setenv("NA0S_PROMPT_SIGN_STRICT", "1")
        monkeypatch.setenv("NA0S_PROMPT_SIGN_KEY", "k" * 40)
        signer = PromptSigner()  # must not raise
        signed = signer.sign("ok")
        assert signer.verify(signed)["valid"] is True

    def test_short_key_warns(self):
        with pytest.warns(UserWarning, match="shorter than 32 bytes"):
            PromptSigner(secret_key="short")

    def test_non_strict_ephemeral_still_warns(self, monkeypatch):
        monkeypatch.delenv("NA0S_PROMPT_SIGN_STRICT", raising=False)
        monkeypatch.delenv("NA0S_PROMPT_SIGN_KEY", raising=False)
        with pytest.warns(UserWarning, match="random ephemeral key"):
            PromptSigner()

    def test_key_id_round_trips(self):
        signer = PromptSigner(secret_key=KEY, key_id="kid-2026")
        signed = signer.sign("with kid")
        assert signed["key_id"] == "kid-2026"
        assert signer.verify(signed)["valid"] is True
