"""HMAC-SHA256 supply-chain integrity tests for safe_pickle.

Verifies that HMAC-SHA256 signing prevents replace-both-files attacks
on the model supply chain, while maintaining backward compatibility
with plain SHA-256 sidecar files.

Run with:
    SCAN_TIMEOUT_SEC=0 python3 -m unittest tests.test_safe_pickle -v
"""

import os
import pickle
import tempfile
import unittest
import warnings
from unittest.mock import patch

# Disable scan timeout before any na0s imports
os.environ["SCAN_TIMEOUT_SEC"] = "0"

from na0s.safe_pickle import (
    _atomic_write_binary,
    _get_signing_key,
    _hash_path,
    _hmac_path,
    _hmac_sha256,
    _sha256,
    safe_dump,
    safe_load,
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
        """Dump without key, then set key and load -> logs warning but works."""
        # Dump without key (creates .sha256 sidecar)
        env_no_key = {k: v for k, v in os.environ.items()
                      if k != "NA0S_PICKLE_KEY"}
        with patch.dict(os.environ, env_no_key, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                safe_dump(self.test_obj, self.pkl_path)
        # Load with key set -> should log warning but still work
        with patch.dict(os.environ, {"NA0S_PICKLE_KEY": "newsecret"}):
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


if __name__ == "__main__":
    unittest.main()
