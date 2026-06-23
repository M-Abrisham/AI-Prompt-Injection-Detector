"""HMAC-SHA256 supply-chain integrity tests for safe_pickle.

Verifies that HMAC-SHA256 signing prevents replace-both-files attacks
on the model supply chain, while maintaining backward compatibility
with plain SHA-256 sidecar files.

Run with:
    SCAN_TIMEOUT_SEC=0 python3 -m unittest tests.test_safe_pickle -v
"""

import io
import os
import pickle
import tempfile
import threading
import unittest
import warnings
from unittest.mock import patch

# Disable scan timeout before any na0s imports
os.environ["SCAN_TIMEOUT_SEC"] = "0"

from na0s.safe_pickle import (
    _MIN_PICKLE_KEY_LEN,
    _WEAK_PICKLE_KEY_LEN,
    _atomic_write_binary,
    _format_sidecar,
    _get_signing_key,
    _hash_path,
    _hmac_path,
    _hmac_sha256,
    _resolve_expected_hash,
    _sha256,
    safe_dump,
    safe_load,
)
from na0s.integrity.safe_pickle import (
    _CACHE_MAXSIZE,
    _cache_get_or_compute,
    _cached_hmac_sha256,
    _cached_sha256,
    _file_identity,
    _hmac_cache,
    _reset_caches,
    _sha256_cache,
)


class TestHelpers(unittest.TestCase):
    """Unit tests for low-level helper functions."""

    def test_hash_path_extension(self):
        """_hash_path appends .sha256 to the pickle path."""
        self.assertEqual(_hash_path("model.pkl"), "model.pkl.sha256")

    def test_hmac_path_extension(self):
        """_hmac_path appends .hmac to the pickle path."""
        self.assertEqual(_hmac_path("model.pkl"), "model.pkl.hmac")

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_get_signing_key_none_without_env(self):
        """_get_signing_key returns None when NA0S_PICKLE_KEY is unset."""
        os.environ.pop("NA0S_PICKLE_KEY", None)
        self.assertIsNone(_get_signing_key())

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "my_secret"})
    def test_get_signing_key_bytes_with_env(self):
        """_get_signing_key returns bytes when NA0S_PICKLE_KEY is set."""
        key = _get_signing_key()
        self.assertIsInstance(key, bytes)
        self.assertEqual(key, b"my_secret")


class TestHMACRoundTrip(unittest.TestCase):
    """Round-trip tests for HMAC and SHA-256 dump/load cycles."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self.pkl_path = os.path.join(self.tmpdir, "test_model.pkl")
        self.test_obj = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}

    def tearDown(self):
        self._tmpdir.cleanup()

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "testsecret"})
    def test_hmac_dump_creates_hmac_sidecar(self):
        """With NA0S_PICKLE_KEY, safe_dump creates .hmac, NOT .sha256."""
        safe_dump(self.test_obj, self.pkl_path)
        self.assertTrue(os.path.exists(_hmac_path(self.pkl_path)))
        self.assertFalse(os.path.exists(_hash_path(self.pkl_path)))

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "testsecret"})
    def test_hmac_load_succeeds_with_correct_key(self):
        """Dump then load with the same key returns the same object."""
        safe_dump(self.test_obj, self.pkl_path)
        loaded = safe_load(self.pkl_path)
        self.assertEqual(loaded, self.test_obj)

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_sha256_dump_creates_sha256_sidecar(self):
        """Without NA0S_PICKLE_KEY, safe_dump creates .sha256, NOT .hmac."""
        os.environ.pop("NA0S_PICKLE_KEY", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            safe_dump(self.test_obj, self.pkl_path)
        self.assertTrue(os.path.exists(_hash_path(self.pkl_path)))
        self.assertFalse(os.path.exists(_hmac_path(self.pkl_path)))

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_sha256_load_succeeds_without_key(self):
        """Dump (no key) then load (no key) round-trips correctly."""
        os.environ.pop("NA0S_PICKLE_KEY", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            safe_dump(self.test_obj, self.pkl_path)
        loaded = safe_load(self.pkl_path)
        self.assertEqual(loaded, self.test_obj)

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_sha256_dump_emits_warning(self):
        """Without key, safe_dump emits a UserWarning about missing key."""
        os.environ.pop("NA0S_PICKLE_KEY", None)
        with self.assertWarns(UserWarning) as cm:
            safe_dump(self.test_obj, self.pkl_path)
        self.assertIn("NA0S_PICKLE_KEY is not set", str(cm.warning))


class TestTamperingDetection(unittest.TestCase):
    """Tests that tampering is detected for both HMAC and SHA-256 modes."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self.pkl_path = os.path.join(self.tmpdir, "test_model.pkl")
        self.test_obj = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}

    def tearDown(self):
        self._tmpdir.cleanup()

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "testsecret"})
    def test_tampered_pickle_detected_hmac(self):
        """Overwriting pkl content after HMAC dump causes ValueError on load."""
        safe_dump(self.test_obj, self.pkl_path)
        # Tamper with the pickle file
        with open(self.pkl_path, "wb") as f:
            pickle.dump({"malicious": True}, f)
        with self.assertRaises(ValueError) as ctx:
            safe_load(self.pkl_path)
        self.assertIn("Integrity check failed", str(ctx.exception))

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_tampered_pickle_detected_sha256(self):
        """Overwriting pkl after SHA-256 dump causes ValueError on load."""
        os.environ.pop("NA0S_PICKLE_KEY", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            safe_dump(self.test_obj, self.pkl_path)
        # Tamper with the pickle file
        with open(self.pkl_path, "wb") as f:
            pickle.dump({"malicious": True}, f)
        with self.assertRaises(ValueError) as ctx:
            safe_load(self.pkl_path)
        self.assertIn("Integrity check failed", str(ctx.exception))

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "testsecret"})
    def test_tampered_hmac_sidecar_detected(self):
        """Overwriting .hmac with a wrong value causes ValueError on load."""
        safe_dump(self.test_obj, self.pkl_path)
        # Tamper with the HMAC sidecar
        with open(_hmac_path(self.pkl_path), "w") as f:
            f.write("0" * 64)
        with self.assertRaises(ValueError) as ctx:
            safe_load(self.pkl_path)
        self.assertIn("Integrity check failed", str(ctx.exception))

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "testsecret"})
    def test_replace_both_attack_blocked(self):
        """Replace-both-files attack: new pkl + forged sha256, but HMAC sidecar
        exists so load uses HMAC verification which the attacker cannot forge."""
        safe_dump(self.test_obj, self.pkl_path)
        # Attacker replaces pickle with malicious payload
        malicious_obj = {"payload": "evil"}
        with open(self.pkl_path, "wb") as f:
            pickle.dump(malicious_obj, f)
        # Attacker writes a valid SHA-256 of the new pickle
        forged_sha = _sha256(self.pkl_path)
        sha_path = _hash_path(self.pkl_path)
        with open(sha_path, "w") as f:
            f.write(forged_sha)
        # Load should use the .hmac sidecar (preferred over .sha256),
        # and HMAC verification will fail because the attacker doesn't
        # know the secret key
        with self.assertRaises(ValueError) as ctx:
            safe_load(self.pkl_path)
        self.assertIn("Integrity check failed", str(ctx.exception))

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_tampered_sha256_sidecar_detected(self):
        """Overwriting .sha256 with a wrong value causes ValueError on load."""
        os.environ.pop("NA0S_PICKLE_KEY", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            safe_dump(self.test_obj, self.pkl_path)
        # Tamper with the SHA-256 sidecar
        with open(_hash_path(self.pkl_path), "w") as f:
            f.write("0" * 64)
        with self.assertRaises(ValueError) as ctx:
            safe_load(self.pkl_path)
        self.assertIn("Integrity check failed", str(ctx.exception))


class TestBackwardCompatibility(unittest.TestCase):
    """Tests backward compatibility and edge cases."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self.pkl_path = os.path.join(self.tmpdir, "test_model.pkl")
        self.test_obj = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_hmac_sidecar_without_key_raises_error(self):
        """Dump with key, then clear key -> load raises ValueError."""
        # Dump with key set
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": "testsecret"}):
            safe_dump(self.test_obj, self.pkl_path)
        # Load without key
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with self.assertRaises(ValueError) as ctx:
                safe_load(self.pkl_path)
            self.assertIn("NA0S_PICKLE_KEY is not set", str(ctx.exception))

    def test_missing_sidecar_raises_file_not_found(self):
        """Dump, delete both sidecars -> FileNotFoundError."""
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": "testsecret"}):
            safe_dump(self.test_obj, self.pkl_path)
        # Delete the HMAC sidecar
        os.remove(_hmac_path(self.pkl_path))
        # Also ensure no SHA-256 sidecar exists
        sha_path = _hash_path(self.pkl_path)
        if os.path.exists(sha_path):
            os.remove(sha_path)
        with self.assertRaises(FileNotFoundError):
            safe_load(self.pkl_path)

    def test_key_set_but_sha256_sidecar_warns(self):
        """Dump without key, then set key and load -> REFUSES the downgrade.

        Behavior change (item #07, downgrade fail-closed): a plain ``.sha256``
        sidecar is attacker-forgeable, so once the operator sets
        ``NA0S_PICKLE_KEY`` (opting into HMAC) it is a downgrade and must fail
        closed by default. Previously this warned-and-accepted; it now raises.
        The opt-out path is covered by ``test_key_set_sha256_downgrade_optout``.
        """
        # Dump without key (creates .sha256 sidecar)
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                safe_dump(self.test_obj, self.pkl_path)
        # Load with key set and no opt-out -> fail closed (refuse downgrade).
        key_env = dict(env_no_key, NA0S_PICKLE_KEY="newsecret")
        key_env.pop("NA0S_ALLOW_SHA256_DOWNGRADE", None)
        with patch.dict(os.environ, key_env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                safe_load(self.pkl_path)
            self.assertIn("refusing to downgrade", str(ctx.exception))

    def test_key_set_sha256_downgrade_optout(self):
        """With NA0S_ALLOW_SHA256_DOWNGRADE=1, the SHA-256 sidecar loads with
        an audit/logger warning (migration-window opt-out, item #07)."""
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                safe_dump(self.test_obj, self.pkl_path)
        opt_env = dict(env_no_key, NA0S_PICKLE_KEY="newsecret",
                       NA0S_ALLOW_SHA256_DOWNGRADE="1")
        with patch.dict(os.environ, opt_env, clear=True):
            with self.assertLogs("na0s.safe_pickle", level="WARNING") as cm:
                loaded = safe_load(self.pkl_path)
            self.assertEqual(loaded, self.test_obj)
            self.assertTrue(any("plain SHA-256 sidecar" in msg
                                for msg in cm.output))


class TestLargeObjectTruncation(unittest.TestCase):
    """Regression tests for the 'pickle data was truncated' bug.

    Root cause: ``_atomic_write_binary`` issued a single ``os.write`` and
    discarded its return value. ``write(2)`` is capped at INT_MAX (2 GiB - 1)
    per call and may short-write, so any pickle larger than ~2 GiB (e.g. the
    400k x 15029 sparse ``features.pkl``) landed on disk truncated, and
    ``safe_load`` later raised ``pickle data was truncated``. These tests
    pin the fix without requiring a multi-GiB object in CI.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self.pkl_path = os.path.join(self.tmpdir, "features.pkl")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_atomic_write_handles_short_writes(self):
        """_atomic_write_binary writes every byte even when os.write short-writes.

        Simulates the POSIX short-write that caused the production truncation:
        a real ``os.write`` that never transfers more than 1 MiB per call.
        The file on disk must still contain the complete buffer.
        """
        real_os_write = os.write
        chunk_cap = 1 << 20  # 1 MiB — force many partial writes

        def short_write(fd, data):
            return real_os_write(fd, bytes(data[:chunk_cap]))

        payload = os.urandom(5 * chunk_cap + 123)  # not a clean multiple
        out = os.path.join(self.tmpdir, "short.bin")
        with patch("na0s.integrity.safe_pickle.os.write", side_effect=short_write):
            _atomic_write_binary(out, payload)

        with open(out, "rb") as f:
            written = f.read()
        self.assertEqual(len(written), len(payload))
        self.assertEqual(written, payload)

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "testsecret"})
    def test_sparse_matrix_roundtrip_with_simulated_short_write(self):
        """A scipy sparse (X, y) tuple round-trips even under short writes.

        This is the exact shape of ``features.pkl``. With os.write capped at
        64 KiB per call (so the multi-hundred-KiB pickle needs many writes),
        safe_dump/safe_load must still produce an equal matrix and the HMAC
        sidecar must validate.
        """
        import numpy as np
        import scipy.sparse as sp

        X = sp.random(2000, 1500, density=0.05, format="csr", random_state=7)
        y = np.array([0, 1] * 1000, dtype=np.int64)

        real_os_write = os.write

        def short_write(fd, data):
            return real_os_write(fd, bytes(data[: 1 << 16]))  # 64 KiB cap

        with patch("na0s.integrity.safe_pickle.os.write", side_effect=short_write):
            safe_dump((X, y), self.pkl_path)

        # Sidecar is the HMAC variant and the file is whole.
        self.assertTrue(os.path.exists(_hmac_path(self.pkl_path)))

        X2, y2 = safe_load(self.pkl_path)
        self.assertEqual(X2.shape, X.shape)
        self.assertEqual(X2.nnz, X.nnz)
        self.assertTrue(np.array_equal(X2.indices, X.indices))
        self.assertTrue(np.allclose(X2.data, X.data))
        self.assertTrue(np.array_equal(y2, y))

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "testsecret"})
    def test_dump_uses_high_pickle_protocol(self):
        """safe_dump pins HIGHEST_PROTOCOL (>=4) for >4 GiB-capable framing."""
        safe_dump({"a": 1}, self.pkl_path)
        with open(self.pkl_path, "rb") as f:
            head = f.read(2)
        # Protocol 2+ files start with the PROTO opcode (0x80) + version byte.
        self.assertEqual(head[0], 0x80)
        self.assertGreaterEqual(head[1], 4)


class TestHardcodedHashPathScoping(unittest.TestCase):
    """The hardcoded KNOWN_HASHES pin must apply ONLY to the shipped package model.

    Regression for the auto-retrain GAP-03 blocker: a freshly-trained candidate
    (e.g. data/processed/model.pkl) shares the basename 'model.pkl' with the
    shipped artifact, so a basename-only KNOWN_HASHES lookup rejected it against
    the OLD shipped hash ("File may be tampered") and the model could never be
    retrained.  The candidate must instead verify via its own fresh sidecar.
    """

    def test_candidate_same_basename_uses_sidecar_not_shipped_pin(self):
        from na0s.safe_pickle import _resolve_expected_hash

        with tempfile.TemporaryDirectory() as d:
            cand = os.path.join(d, "model.pkl")  # basename collides with KNOWN_HASHES
            safe_dump({"candidate": True, "v": 999}, cand)  # writes a fresh sidecar
            _, source = _resolve_expected_hash(cand)
            self.assertIn(source, ("sidecar_hmac", "sidecar_sha256"))
            # Must NOT raise "File may be tampered" against the shipped pin.
            self.assertEqual(safe_load(cand), {"candidate": True, "v": 999})

    def test_shipped_model_still_uses_hardcoded_pin(self):
        from na0s.safe_pickle import _resolve_expected_hash
        from na0s.models import get_model_path

        _, source = _resolve_expected_hash(get_model_path("model.pkl"))
        self.assertEqual(source, "hardcoded")


class TestKeyAwareSidecarResolution(unittest.TestCase):
    """Item #07 — key-aware sidecar selection: keyless .hmac DoS fix (Threat B)
    and plain-SHA256 downgrade fail-closed (Threat A)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self.pkl_path = os.path.join(self.tmpdir, "ka_model.pkl")
        self.test_obj = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}

    def tearDown(self):
        self._tmpdir.cleanup()

    def _dump_keyless(self):
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                safe_dump(self.test_obj, self.pkl_path)

    # --- Threat B: keyless dropped .hmac must not brick a valid .sha256 ---

    def test_keyless_dropped_hmac_does_not_brick_valid_sha256(self):
        """HEADLINE: a stray attacker-dropped .hmac beside a valid .sha256 must
        NOT brick a keyless load; the .sha256 wins and verifies."""
        self._dump_keyless()
        # Attacker drops an arbitrary .hmac sidecar next to the valid .sha256.
        with open(_hmac_path(self.pkl_path), "w") as f:
            f.write(_format_sidecar("hmac-sha256", "0" * 64))
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            loaded = safe_load(self.pkl_path)
        self.assertEqual(loaded, self.test_obj)

    def test_keyless_dropped_hmac_plus_tampered_sha256_still_raises(self):
        """The .sha256 fallback is verified, not blindly accepted: tamper the
        pkl so the .sha256 no longer matches -> still raises."""
        self._dump_keyless()
        with open(_hmac_path(self.pkl_path), "w") as f:
            f.write(_format_sidecar("hmac-sha256", "0" * 64))
        # Tamper the pickle so the .sha256 no longer matches.
        with open(self.pkl_path, "wb") as f:
            pickle.dump({"malicious": True}, f)
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with self.assertRaises(ValueError) as ctx:
                safe_load(self.pkl_path)
        self.assertIn("Integrity check failed", str(ctx.exception))

    def test_keyless_only_hmac_no_sha256_fallback_raises(self):
        """Keyless, only .hmac, no .sha256 -> raises; message names the missing
        SHA-256 fallback (genuinely unverifiable without a key)."""
        # Dump with a key to get an .hmac sidecar, then delete the would-be sha.
        # Key lengthened (item #09): "k" (1 char) is now hard-rejected by the
        # _MIN_PICKLE_KEY_LEN floor; a >= 32-char key is accepted silently.
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": "k" * 32}):
            safe_dump(self.test_obj, self.pkl_path)
        sha = _hash_path(self.pkl_path)
        if os.path.exists(sha):
            os.remove(sha)
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with self.assertRaises(ValueError) as ctx:
                safe_load(self.pkl_path)
        msg = str(ctx.exception)
        self.assertIn("NA0S_PICKLE_KEY is not set", msg)
        self.assertIn(".sha256", msg)

    # --- Threat A: plain-SHA256 downgrade fail-closed when a key is set ---

    def test_key_set_forged_sha256_swap_still_refused(self):
        """The real attack: keyless-dump, then with a key set the attacker
        swaps the pkl and recomputes a VALID .sha256 of the malicious pickle and
        deletes any .hmac. Default (no opt-out) refuses before any compare."""
        self._dump_keyless()
        # Attacker swaps payload and forges a matching plain SHA-256.
        with open(self.pkl_path, "wb") as f:
            pickle.dump({"payload": "evil"}, f)
        with open(_hash_path(self.pkl_path), "w") as f:
            f.write(_format_sidecar("sha256", _sha256(self.pkl_path)))
        hmac_side = _hmac_path(self.pkl_path)
        if os.path.exists(hmac_side):
            os.remove(hmac_side)
        env = {k: v for k, v in os.environ.items()
               if k not in ("NA0S_PICKLE_KEY", "NA0S_ALLOW_SHA256_DOWNGRADE")}
        # Key lengthened (item #09): "secret" (6 chars) is now hard-rejected by
        # the _MIN_PICKLE_KEY_LEN floor; a >= 32-char key is accepted silently.
        env["NA0S_PICKLE_KEY"] = "secret" * 6  # 36 chars
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                safe_load(self.pkl_path)
        self.assertIn("refusing to downgrade", str(ctx.exception))

    def test_key_set_prefers_hmac_over_sha256(self):
        """Regression guard: both sidecars present + key set -> HMAC verified."""
        # Key lengthened (item #09): "secret" (6 chars) is now hard-rejected by
        # the _MIN_PICKLE_KEY_LEN floor; a >= 32-char key is accepted silently.
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": "secret" * 6}):
            safe_dump(self.test_obj, self.pkl_path)
            # Add a stray (irrelevant) .sha256 alongside the .hmac.
            with open(_hash_path(self.pkl_path), "w") as f:
                f.write(_format_sidecar("sha256", _sha256(self.pkl_path)))
            expected, source = _resolve_expected_hash(self.pkl_path)
            self.assertEqual(source, "sidecar_hmac")
            loaded = safe_load(self.pkl_path)
        self.assertEqual(loaded, self.test_obj)

    # --- Edge case 9: algorithm-tag / extension mismatch ---

    def test_typed_sidecar_algo_mismatch_refused(self):
        """A v1:hmac-sha256: payload smuggled into a .sha256-named file is
        refused (selection keys off the extension)."""
        self._dump_keyless()
        # Overwrite the .sha256 with a payload that DECLARES hmac-sha256.
        with open(_hash_path(self.pkl_path), "w") as f:
            f.write(_format_sidecar("hmac-sha256", _sha256(self.pkl_path)))
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with self.assertRaises(ValueError) as ctx:
                safe_load(self.pkl_path)
        self.assertIn("algorithm tag does not match", str(ctx.exception))

    def test_same_basename_outside_package_uses_sidecar_not_hardcoded(self):
        """Path-scoping (PR #457 GAP-03, composed with #07 key-aware resolution):
        a file that merely SHARES a KNOWN_HASHES basename but is NOT the bundled
        package artifact must NOT inherit the trusted hardcoded pin — it resolves
        by its own sidecar. (Previously this asserted 'hardcoded' for any matching
        basename; ``_is_packaged_model`` correctly tightened that so a same-named
        retrain candidate elsewhere on disk can't ride the shipped pin.) The
        genuinely bundled artifact still resolves 'hardcoded' — see
        test_shipped_model_still_uses_hardcoded_pin."""
        name = "model.pkl"  # shares a KNOWN_HASHES basename, but lives in tmpdir
        path = os.path.join(self.tmpdir, name)
        with open(_hash_path(path), "w") as f:
            f.write(_format_sidecar("sha256", "a" * 64))
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            expected, source = _resolve_expected_hash(path)
        self.assertNotEqual(source, "hardcoded")
        self.assertEqual(source, "sidecar_sha256")
        self.assertEqual(expected, "a" * 64)


class TestKeyStrength(unittest.TestCase):
    """Item #09 — ``_get_signing_key`` enforces an NA0S_PICKLE_KEY strength
    policy: reject empty/whitespace-only and ``< _MIN_PICKLE_KEY_LEN``, warn for
    the weak band ``[_MIN .. _WEAK)``, accept ``>= _WEAK`` silently. The keyless
    (unset) path is unchanged and must never raise.
    """

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_key_unset_returns_none(self):
        """K1: unset NA0S_PICKLE_KEY -> None (keyless SHA-256 fallback intact)."""
        os.environ.pop("NA0S_PICKLE_KEY", None)
        self.assertIsNone(_get_signing_key())

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "   "})
    def test_key_whitespace_only_raises(self):
        """K2: whitespace-only key -> ValueError (was 3 bytes of whitespace)."""
        with self.assertRaises(ValueError) as ctx:
            _get_signing_key()
        self.assertIn("empty/whitespace-only", str(ctx.exception))

    def test_key_empty_string_raises(self):
        """An explicitly empty string is rejected like whitespace-only."""
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": ""}):
            with self.assertRaises(ValueError) as ctx:
                _get_signing_key()
            self.assertIn("empty/whitespace-only", str(ctx.exception))

    def test_key_too_short_raises(self):
        """K3: a key shorter than the floor -> ValueError naming 'too weak'."""
        with patch.dict(os.environ,
                        {"NA0S_PICKLE_KEY": "x" * (_MIN_PICKLE_KEY_LEN - 1)}):
            with self.assertRaises(ValueError) as ctx:
                _get_signing_key()
            self.assertIn("too weak", str(ctx.exception))

    def test_key_min_boundary(self):
        """Boundary: exactly _MIN accepted, _MIN-1 rejected (pins the floor)."""
        with patch.dict(os.environ,
                        {"NA0S_PICKLE_KEY": "x" * _MIN_PICKLE_KEY_LEN}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.assertEqual(_get_signing_key(), b"x" * _MIN_PICKLE_KEY_LEN)
        with patch.dict(os.environ,
                        {"NA0S_PICKLE_KEY": "x" * (_MIN_PICKLE_KEY_LEN - 1)}):
            with self.assertRaises(ValueError):
                _get_signing_key()

    def test_key_weak_band_warns_but_accepts(self):
        """[_MIN .. _WEAK): accepted, returns bytes, emits a weakness warning."""
        weak_len = _MIN_PICKLE_KEY_LEN  # 8 -> in the warn band
        self.assertLess(weak_len, _WEAK_PICKLE_KEY_LEN)
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": "x" * weak_len}):
            with self.assertWarns(UserWarning) as cm:
                key = _get_signing_key()
        self.assertEqual(key, b"x" * weak_len)
        self.assertIn("recommended for full HMAC", str(cm.warning))

    def test_key_strong_accepted_silently(self):
        """K4: >= _WEAK accepted with NO warning, returns the encoded bytes."""
        strong = "x" * _WEAK_PICKLE_KEY_LEN
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": strong}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                key = _get_signing_key()
        self.assertEqual(key, strong.encode())
        self.assertEqual(
            [w for w in caught if issubclass(w.category, UserWarning)], []
        )

    def test_key_stripped_before_encode(self):
        """Surrounding whitespace is stripped before encoding (the trailing-
        newline-in-env footgun is not part of the secret)."""
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": "  " + "a" * 32 + "  "}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.assertEqual(_get_signing_key(), b"a" * 32)


class TestMalformedSidecarFailFast(unittest.TestCase):
    """Item #09 — end-to-end loader contract: a malformed sidecar fails fast at
    PARSE with an accurate 'malformed integrity sidecar' message, distinct from
    the valid-shape compare-mismatch ('Integrity check failed'). And a weak key
    is rejected at dump time, not just load time.
    """

    _STRONG_KEY = "strong-key-aaaaaaaaaaaaaaaaaaaaaa"  # 33 chars, >= _WEAK

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self.pkl_path = os.path.join(self.tmpdir, "fastfail_model.pkl")
        self.test_obj = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_malformed_hmac_sidecar_fails_fast(self):
        """Overwrite the .hmac sidecar with a malformed (non-64-hex) body; load
        raises at parse, naming the malformed sidecar — NOT 'Integrity check
        failed' (which would wrongly imply a tampered pickle)."""
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": self._STRONG_KEY}):
            safe_dump(self.test_obj, self.pkl_path)
            with open(_hmac_path(self.pkl_path), "w") as f:
                f.write("v1:hmac-sha256:zzzz")  # malformed digest body
            with self.assertRaises(ValueError) as ctx:
                safe_load(self.pkl_path)
        msg = str(ctx.exception)
        self.assertIn("malformed integrity sidecar", msg)
        self.assertNotIn("Integrity check failed", msg)

    def test_valid_shape_tamper_still_hits_compare_mismatch(self):
        """Contrast: a VALID-shape but wrong .hmac digest (64 zeros) reaches the
        compare and raises 'Integrity check failed' — the two failure points are
        distinct and both preserved."""
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": self._STRONG_KEY}):
            safe_dump(self.test_obj, self.pkl_path)
            with open(_hmac_path(self.pkl_path), "w") as f:
                f.write(_format_sidecar("hmac-sha256", "0" * 64))
            with self.assertRaises(ValueError) as ctx:
                safe_load(self.pkl_path)
        self.assertIn("Integrity check failed", str(ctx.exception))

    def test_load_legit_still_roundtrips(self):
        """Regression: dump+load with an adequate key returns the object."""
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": self._STRONG_KEY}):
            safe_dump(self.test_obj, self.pkl_path)
            loaded = safe_load(self.pkl_path)
        self.assertEqual(loaded, self.test_obj)

    def test_dump_with_weak_key_raises(self):
        """A too-short key is rejected at WRITE time (in _get_signing_key via
        safe_dump), not only at load."""
        env = {k: v for k, v in os.environ.items() if k != "NA0S_PICKLE_KEY"}
        env["NA0S_PICKLE_KEY"] = "x"  # 1 char -> below the floor
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                safe_dump(self.test_obj, self.pkl_path)
        self.assertIn("too weak", str(ctx.exception))

    def test_keyless_dump_still_works(self):
        """K1 regression: unset key -> SHA-256 sidecar written, no raise."""
        env = {k: v for k, v in os.environ.items() if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                safe_dump(self.test_obj, self.pkl_path)
            self.assertTrue(os.path.exists(_hash_path(self.pkl_path)))
            self.assertFalse(os.path.exists(_hmac_path(self.pkl_path)))


class TestDigestCacheBoundedLRU(unittest.TestCase):
    """Item #15 — the mtime hash cache is now a bounded LRU keyed on
    ``(st_mtime_ns, st_size, st_ino)`` under a lock.

    These tests exercise the cache directly via ``_cached_sha256`` /
    ``_cached_hmac_sha256`` plus two end-to-end ``safe_dump``/``safe_load``
    round-trips. They use real files in a ``TemporaryDirectory`` and reset the
    caches in ``setUp`` so each case starts cold. The cache sits AFTER the
    ``compare_digest`` trust decision, so none of this touches accept/reject.
    """

    def setUp(self):
        _reset_caches()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name

    def tearDown(self):
        _reset_caches()
        self._tmpdir.cleanup()

    def _write(self, name, data=b"payload"):
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    # --- Defect A: bounded / evicting ---

    def test_cache_bounded_to_maxsize(self):
        """Hashing _CACHE_MAXSIZE + 10 distinct paths caps the cache at exactly
        _CACHE_MAXSIZE (proves the bound — fails if eviction is dropped)."""
        n = _CACHE_MAXSIZE + 10
        for i in range(n):
            p = self._write("m{}.bin".format(i), data=b"x%d" % i)
            _cached_sha256(p)
        self.assertEqual(len(_sha256_cache), _CACHE_MAXSIZE)

    def test_lru_evicts_oldest_not_newest(self):
        """Fill to cap, re-touch the oldest entry (-> MRU), insert one more:
        the re-touched entry survives and the SECOND-oldest is evicted. Proves
        LRU recency, not plain FIFO."""
        paths = []
        for i in range(_CACHE_MAXSIZE):
            p = self._write("m{}.bin".format(i), data=b"x%d" % i)
            _cached_sha256(p)
            paths.append(p)
        # Re-access the oldest (paths[0]) -> moves it to MRU.
        _cached_sha256(paths[0])
        # Insert one more, forcing exactly one eviction.
        extra = self._write("extra.bin", data=b"extra")
        _cached_sha256(extra)
        self.assertEqual(len(_sha256_cache), _CACHE_MAXSIZE)
        # paths[0] survived (it was MRU); paths[1] (now oldest) was evicted.
        self.assertIn(paths[0], _sha256_cache)
        self.assertNotIn(paths[1], _sha256_cache)
        self.assertIn(extra, _sha256_cache)

    def test_evicted_entry_recomputes_correct_digest(self):
        """After eviction, re-accessing an evicted path returns the CORRECT
        digest (== _sha256 of the file) and re-inserts it — eviction is not data
        loss / not a stale value."""
        first = self._write("first.bin", data=b"the-first-file")
        _cached_sha256(first)
        # Evict it by flooding with cap more distinct paths.
        for i in range(_CACHE_MAXSIZE):
            _cached_sha256(self._write("flood{}.bin".format(i), data=b"f%d" % i))
        self.assertNotIn(first, _sha256_cache)
        # Re-access: must recompute and match the ground-truth digest.
        self.assertEqual(_cached_sha256(first), _sha256(first))
        self.assertIn(first, _sha256_cache)

    # --- Defect B: richer key catches stale content ---

    def test_same_mtime_different_size_is_cache_miss(self):
        """A same-mtime-SECONDS rewrite that changes the file SIZE is a MISS:
        the second call returns the NEW digest, not the stale cached one. This is
        the headline [test-update] case — fails if the key is mtime-seconds."""
        p = self._write("rw.bin", data=b"short")
        d1 = _cached_sha256(p)
        st = os.stat(p)
        # Rewrite with different-length content, then force the SAME
        # mtime-seconds via os.utime (mtime_ns may still differ, but size alone
        # must already discriminate).
        with open(p, "wb") as f:
            f.write(b"a-much-longer-payload-than-before")
        same_sec = float(int(st.st_mtime))
        os.utime(p, (same_sec, same_sec))
        d2 = _cached_sha256(p)
        self.assertNotEqual(d1, d2)
        self.assertEqual(d2, _sha256(p))

    def test_utime_reset_with_changed_content_is_cache_miss(self):
        """Change content then RESET mtime to the original via os.utime: still a
        MISS + correct new digest (st_size / st_ino discriminate even when the
        attacker/race restores the old mtime)."""
        p = self._write("reset.bin", data=b"original-content-here")
        st0 = os.stat(p)
        d1 = _cached_sha256(p)
        with open(p, "wb") as f:
            f.write(b"tampered!")  # different size, too
        # Restore the original mtime exactly (ns).
        os.utime(p, ns=(st0.st_atime_ns, st0.st_mtime_ns))
        d2 = _cached_sha256(p)
        self.assertNotEqual(d1, d2)
        self.assertEqual(d2, _sha256(p))

    def test_unchanged_file_is_cache_hit(self):
        """Two _cached_sha256 of an untouched file: the second is a HIT — the
        underlying _sha256 is invoked EXACTLY ONCE (the optimization still works,
        no spurious miss / no perf regression)."""
        p = self._write("hit.bin", data=b"stable-bytes")
        with patch(
            "na0s.integrity.safe_pickle._sha256",
            wraps=__import__("na0s.integrity.safe_pickle",
                             fromlist=["_sha256"])._sha256,
        ) as spy:
            _cached_sha256(p)
            _cached_sha256(p)
        self.assertEqual(spy.call_count, 1)

    # --- Defect C + isolation ---

    def test_sha256_and_hmac_caches_are_independent(self):
        """The SHA and HMAC caches do not conflate: the same path yields a
        SHA-256 digest in one and an HMAC digest in the other, and neither cache
        returns the other's value (regression guard for a future shared-helper
        refactor)."""
        p = self._write("iso.bin", data=b"isolation-test")
        key = b"k" * 32
        sha = _cached_sha256(p)
        mac = _cached_hmac_sha256(p, key)
        self.assertNotEqual(sha, mac)
        self.assertEqual(_sha256_cache[p][1], sha)
        self.assertEqual(_hmac_cache[p][1], mac)
        self.assertEqual(mac, _hmac_sha256(p, key))

    def test_hmac_cache_key_excludes_signing_key_documented(self):
        """DOCUMENTED LIMITATION: the HMAC cache key excludes the signing key, so
        a second call with a DIFFERENT key on an UNCHANGED file returns the STALE
        first-key digest. This pins the current single-key-per-process contract
        so item-7's key-aware work can't silently regress into a multi-key
        hazard. (A real key rotation must _reset_caches().)"""
        p = self._write("multikey.bin", data=b"same-bytes")
        key1 = b"k" * 32
        key2 = b"j" * 32
        d1 = _cached_hmac_sha256(p, key1)
        d2 = _cached_hmac_sha256(p, key2)  # unchanged file -> HIT, stale key1
        self.assertEqual(d1, d2)
        self.assertEqual(d1, _hmac_sha256(p, key1))
        self.assertNotEqual(d1, _hmac_sha256(p, key2))
        # After a reset, the new key is honoured.
        _reset_caches()
        self.assertEqual(_cached_hmac_sha256(p, key2), _hmac_sha256(p, key2))

    def test_concurrent_loads_around_cap_do_not_raise(self):
        """N threads each hashing one of _CACHE_MAXSIZE*2 distinct files: no
        exception, and the cache settles at exactly _CACHE_MAXSIZE (the lock +
        non-iterating popitem hold under contention)."""
        paths = [self._write("c{}.bin".format(i), data=b"c%d" % i)
                 for i in range(_CACHE_MAXSIZE * 2)]
        errors = []

        def worker(pth):
            try:
                _cached_sha256(pth)
            except Exception as exc:  # noqa: BLE001 - test records, re-asserts
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(p,)) for p in paths]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(_sha256_cache), _CACHE_MAXSIZE)

    def test_cache_get_or_compute_computes_outside_lock(self):
        """Perf invariant: the compute() callable runs while _cache_lock is NOT
        held (so concurrent loads don't serialise on I/O). Asserted by checking
        the lock is acquirable from inside compute()."""
        p = self._write("outside.bin", data=b"lock-check")
        observed = {}

        def compute():
            observed["acquired"] = _cache_lock_acquirable()
            return _sha256(p)

        def _cache_lock_acquirable():
            from na0s.integrity.safe_pickle import _cache_lock
            got = _cache_lock.acquire(blocking=False)
            if got:
                _cache_lock.release()
            return got

        _reset_caches()
        result = _cache_get_or_compute(_sha256_cache, p, compute)
        self.assertEqual(result, _sha256(p))
        self.assertTrue(observed["acquired"],
                        "compute() must run OUTSIDE the cache lock")

    # --- Use-case / behavior end-to-end ---

    def test_safe_load_roundtrip_unchanged_with_bounded_cache(self):
        """Full safe_dump -> safe_load (keyless SHA path) returns the original
        object, and a second safe_load is a cache HIT (_sha256 called once for
        the load-time digest)."""
        obj = {"weights": [1.0, 2.0], "bias": 0.25}
        path = os.path.join(self.tmpdir, "rt.pkl")
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                safe_dump(obj, path)
            _reset_caches()
            real_sha256 = __import__(
                "na0s.integrity.safe_pickle", fromlist=["_sha256"]
            )._sha256
            with patch(
                "na0s.integrity.safe_pickle._sha256", wraps=real_sha256
            ) as spy:
                self.assertEqual(safe_load(path), obj)
                self.assertEqual(safe_load(path), obj)
            # First load computes the digest; second load is a cache HIT.
            self.assertEqual(spy.call_count, 1)

    def test_steady_state_few_paths_never_evict(self):
        """Edge case 8: repeatedly loading a small fixed set of paths (the real
        SDK shape) keeps the cache well under the cap and never evicts — proves
        zero behavior/perf change for the production workload."""
        obj = {"a": 1}
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        paths = []
        with patch.dict(os.environ, env_no_key, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for i in range(4):  # ~ the real ≤6 model-path universe
                    p = os.path.join(self.tmpdir, "fixed{}.pkl".format(i))
                    safe_dump(obj, p)
                    paths.append(p)
            _reset_caches()
            for _ in range(5):  # many loads of the same fixed set
                for p in paths:
                    self.assertEqual(safe_load(p), obj)
        self.assertEqual(len(_sha256_cache), len(paths))
        self.assertLessEqual(len(_sha256_cache), _CACHE_MAXSIZE)


class TestReadOnceBufferTOCTOU(unittest.TestCase):
    """Item #04a — TOCTOU read-once buffer (CWE-367).

    ``safe_load`` must read the pickle bytes EXACTLY ONCE into an in-memory
    buffer and unpickle that buffer, so the bytes verified by the integrity
    digest are byte-for-byte the bytes executed by the unpickler. An attacker
    who swaps the on-disk file between the digest check and the load must NOT be
    able to execute UN-verified bytes (the classic verify-then-reopen window).

    Each test asserts an observable outcome (the exact bytes unpickled, a raised
    type+message, or the absence of a side effect) — never merely "no crash".
    """

    def setUp(self):
        import na0s.integrity.safe_pickle as sp
        self.sp = sp
        self.sp._reset_caches()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self.pkl_path = os.path.join(self.tmpdir, "toctou_model.pkl")
        self.benign = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}

    def tearDown(self):
        self.sp._reset_caches()
        self._tmpdir.cleanup()

    def _dump_keyless(self, obj, path):
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                safe_dump(obj, path)

    # --- HEADLINE: executed bytes == verified bytes; no second content read ---

    def test_pickle_load_unpickles_exactly_the_read_buffer(self):
        """The bytes fed to pickle.load are byte-for-byte the bytes returned by
        the single _read_file_bytes call — proving the content is read once and
        the executed bytes ARE the buffer (not a re-read of the file).

        Spies on _read_file_bytes (the one content read) and pickle.load, then
        asserts the BytesIO handed to pickle.load wraps exactly that buffer. If
        safe_load ever re-opened the path for the load, the two byte strings
        would not be guaranteed identical and this guard would have teeth.
        """
        self._dump_keyless(self.benign, self.pkl_path)
        captured = {}

        real_read = self.sp._read_file_bytes
        real_pickle_load = self.sp.pickle.load

        def spy_read(path):
            data = real_read(path)
            captured.setdefault("read_bufs", []).append(data)
            return data

        def spy_pickle_load(fileobj):
            # The buffer-based path passes an io.BytesIO wrapping the verified
            # bytes; capture what pickle.load actually executes.
            self.assertIsInstance(fileobj, io.BytesIO)
            captured["loaded_bytes"] = fileobj.getvalue()
            return real_pickle_load(fileobj)

        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with patch.object(self.sp, "_read_file_bytes", side_effect=spy_read):
                with patch.object(self.sp.pickle, "load",
                                  side_effect=spy_pickle_load):
                    loaded = safe_load(self.pkl_path)

        self.assertEqual(loaded, self.benign)
        # Exactly ONE content read produced the executed buffer.
        self.assertEqual(len(captured["read_bufs"]), 1)
        # The bytes unpickled are byte-for-byte the bytes that were read+hashed.
        self.assertEqual(captured["loaded_bytes"], captured["read_bufs"][0])

    def test_post_verify_file_swap_cannot_execute_unverified_bytes(self):
        """The core TOCTOU regression guard: swap the on-disk file to a malicious
        payload AFTER the buffer is read, by hooking verify_file_digest to mutate
        the file mid-load. With read-once, safe_load unpickles the ORIGINAL
        verified buffer — never the swapped-in malicious bytes.

        The malicious payload, if ever unpickled, writes a sentinel file via
        __reduce__. We assert the sentinel is NEVER created AND the returned
        object is the benign one (or the buffer guard raises) — both outcomes
        prove the un-verified bytes did not execute.

        Pre-fix (re-open at pickle.load time) this FAILS: the third open would
        read the swapped malicious file and run it, creating the sentinel.
        """
        self._dump_keyless(self.benign, self.pkl_path)
        sentinel = os.path.join(self.tmpdir, "PWNED")

        # Build a malicious pickle whose __reduce__ writes the sentinel on load,
        # plus a VALID sidecar over it (simulates an attacker who can rewrite
        # both the file and its sha256 between the buffer read and the hash).
        malicious_path = os.path.join(self.tmpdir, "_evil_src.pkl")
        with open(malicious_path, "wb") as f:
            f.write(pickle.dumps(_SentinelReduce(sentinel)))
        with open(malicious_path, "rb") as f:
            malicious_bytes = f.read()

        real_verify = self.sp.verify_file_digest

        def swapping_verify(path):
            # Attacker swaps the on-disk file (and a matching sha256 sidecar) to
            # the malicious payload AFTER safe_load already buffered the benign
            # bytes. verify_file_digest now hashes the malicious file.
            with open(path, "wb") as f:
                f.write(malicious_bytes)
            with open(_hash_path(path), "w") as f:
                f.write(_format_sidecar("sha256", self.sp._sha256(path)))
            return real_verify(path)

        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with patch.object(self.sp, "verify_file_digest",
                              side_effect=swapping_verify):
                # Either the benign object loads (executed buffer == verified
                # buffer) OR the buffer guard catches the swap and raises — never
                # the malicious bytes.
                try:
                    loaded = safe_load(self.pkl_path)
                    self.assertEqual(loaded, self.benign)
                except ValueError as exc:
                    self.assertIn("Integrity check failed", str(exc))

        self.assertFalse(
            os.path.exists(sentinel),
            "malicious swapped-in payload executed — TOCTOU window still open",
        )

    def test_buffer_guard_rejects_mismatch_against_resolved_digest(self):
        """The buffer guard (_verify_buffer_digest) raises when the in-memory
        bytes do not match the resolved expected digest — same message/event as
        an on-disk tamper. Directly drives the helper to prove it has teeth and
        is not a hollow pass-through.
        """
        self._dump_keyless(self.benign, self.pkl_path)
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            # Correct buffer verifies silently (returns None).
            good = self.sp._read_file_bytes(self.pkl_path)
            self.assertIsNone(self.sp._verify_buffer_digest(self.pkl_path, good))
            # A tampered buffer (one flipped byte) is rejected.
            tampered = bytearray(good)
            tampered[-1] ^= 0xFF
            with self.assertRaises(ValueError) as ctx:
                self.sp._verify_buffer_digest(self.pkl_path, bytes(tampered))
        self.assertIn("Integrity check failed", str(ctx.exception))

    def test_buffer_guard_hmac_tier_requires_key(self):
        """The buffer guard mirrors verify_file_digest's keyless-lone-.hmac
        refusal: a lone .hmac with no key and no .sha256 fallback raises the
        'NA0S_PICKLE_KEY is not set' error rather than silently accepting."""
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": "k" * 32}):
            safe_dump(self.benign, self.pkl_path)
        sha = _hash_path(self.pkl_path)
        if os.path.exists(sha):
            os.remove(sha)
        data = self.sp._read_file_bytes(self.pkl_path)
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with self.assertRaises(ValueError) as ctx:
                self.sp._verify_buffer_digest(self.pkl_path, data)
        self.assertIn("NA0S_PICKLE_KEY is not set", str(ctx.exception))

    def test_roundtrip_unchanged_hmac_tier(self):
        """Regression: read-once buffer leaves the HMAC tier round-trip intact."""
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": "k" * 32}):
            safe_dump(self.benign, self.pkl_path)
            self.assertEqual(safe_load(self.pkl_path), self.benign)

    def test_load_never_reopens_path_for_content_after_buffer(self):
        """No second CONTENT open of the pickle path occurs during a load: the
        only read of `path`'s bytes is the single _read_file_bytes call. Hooks
        _read_file_bytes (the sanctioned single read) and a real builtins.open
        spy that records every open of the pickle path in 'rb'/binary-read mode.

        verify_file_digest legitimately re-opens to STREAM-hash the file (that is
        the on-disk verification, not an execution read); the TOCTOU guarantee is
        that the EXECUTED bytes come from the buffer, asserted by
        test_pickle_load_unpickles_exactly_the_read_buffer. Here we assert the
        weaker-but-explicit property that pickle.load is NEVER handed a file
        object opened on `path` (it is handed an io.BytesIO).
        """
        self._dump_keyless(self.benign, self.pkl_path)
        real_pickle_load = self.sp.pickle.load
        loaded_from = {}

        def spy_pickle_load(fileobj):
            loaded_from["type"] = type(fileobj).__name__
            loaded_from["is_bytesio"] = isinstance(fileobj, io.BytesIO)
            return real_pickle_load(fileobj)

        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with patch.object(self.sp.pickle, "load",
                              side_effect=spy_pickle_load):
                self.assertEqual(safe_load(self.pkl_path), self.benign)
        # pickle.load consumed an in-memory buffer, not a freshly-opened file.
        self.assertTrue(loaded_from["is_bytesio"],
                        "pickle.load was handed {} — expected io.BytesIO "
                        "(a re-opened file is a TOCTOU window)"
                        .format(loaded_from.get("type")))


class _SentinelReduce:
    """A pickle whose __reduce__ writes a sentinel file when unpickled.

    Used only by TestReadOnceBufferTOCTOU to prove a swapped-in malicious
    payload is NEVER deserialized: if safe_load ever unpickled these bytes, the
    sentinel would be created and the test fails.
    """

    def __init__(self, sentinel_path):
        self.sentinel_path = sentinel_path

    def __reduce__(self):
        return (_write_sentinel, (self.sentinel_path,))


def _write_sentinel(path):  # pragma: no cover - must NEVER run in these tests
    with open(path, "w", encoding="utf-8") as f:
        f.write("pwned")
    return path


if __name__ == "__main__":
    unittest.main()
