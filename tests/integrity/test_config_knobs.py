"""Tests for the externalized Layer-11 integrity knobs in na0s.config.

Item 16 (ROADMAP_V2.md:1177) moved two in-package integrity knobs out of
``na0s.integrity.safe_pickle`` and into ``na0s.config``:

* ``INTEGRITY_HASH_CHUNK_BYTES`` — the read-batch size for the incremental
  SHA-256 / HMAC hashing (a pure I/O batching choice; digest-irrelevant).
* ``PICKLE_SIGNING_KEY_ENV`` — the *name* of the env var that holds the HMAC
  signing key, de-duplicated from the getenv call and the operator messages.

These tests assert BEHAVIOUR, not just literal equality:

* the ``safe_int_env`` clamp on the chunk size actually rejects a 0/negative
  override (a 0-chunk would make ``iter(lambda: f.read(n), b"")`` spin forever)
  and falls back to the 64 KiB default;
* the resulting digest is independent of the chunk size (the core correctness
  claim that makes K1 behaviour-preserving) — same file, two chunk sizes, one
  digest, and a tampered file is still rejected under a non-default chunk;
* the K4 indirection still emits the operator warning containing the literal
  ``NA0S_PICKLE_KEY`` substring (guards the wording-drift regression that the
  interpolation could introduce — pinned by test_safe_pickle.py too).
"""

import importlib
import os
import tempfile
import unittest
import warnings
from unittest.mock import patch


def _reload_config():
    """Re-execute config.py's module-level env reads."""
    import na0s.config as cfg
    importlib.reload(cfg)
    return cfg


class TestIntegrityHashChunkBytes(unittest.TestCase):
    """``INTEGRITY_HASH_CHUNK_BYTES`` default / override / clamp."""

    def tearDown(self):
        # Restore the module to its env-free default for later tests.
        _reload_config()

    def test_default_is_64kib(self):
        cfg = _reload_config()
        self.assertEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 1 << 16)
        self.assertEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 65536)

    def test_env_override_takes_effect_after_reload(self):
        with patch.dict(os.environ, {"NA0S_INTEGRITY_HASH_CHUNK_BYTES": "131072"}):
            cfg = _reload_config()
            self.assertEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 131072)

    def test_env_garbage_falls_back_to_default(self):
        with patch.dict(os.environ, {"NA0S_INTEGRITY_HASH_CHUNK_BYTES": "abc"}):
            cfg = _reload_config()
            self.assertEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 65536)

    def test_env_zero_falls_back_not_zero(self):
        """A 0 chunk would make ``iter(lambda: f.read(0), b'')`` spin forever;
        the lo=4096 clamp MUST reject it and fall back to the default."""
        with patch.dict(os.environ, {"NA0S_INTEGRITY_HASH_CHUNK_BYTES": "0"}):
            cfg = _reload_config()
            self.assertNotEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 0)
            self.assertEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 65536)

    def test_env_negative_falls_back(self):
        with patch.dict(os.environ, {"NA0S_INTEGRITY_HASH_CHUNK_BYTES": "-1"}):
            cfg = _reload_config()
            self.assertEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 65536)

    def test_env_below_floor_falls_back(self):
        """A value under the 4096 floor is rejected (guardrail, not clamped-to)."""
        with patch.dict(os.environ, {"NA0S_INTEGRITY_HASH_CHUNK_BYTES": "100"}):
            cfg = _reload_config()
            self.assertEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 65536)

    def test_env_above_ceiling_falls_back(self):
        with patch.dict(os.environ, {"NA0S_INTEGRITY_HASH_CHUNK_BYTES": str(1 << 30)}):
            cfg = _reload_config()
            self.assertEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 65536)

    def test_env_at_floor_is_accepted(self):
        with patch.dict(os.environ, {"NA0S_INTEGRITY_HASH_CHUNK_BYTES": "4096"}):
            cfg = _reload_config()
            self.assertEqual(cfg.INTEGRITY_HASH_CHUNK_BYTES, 4096)


class TestPickleSigningKeyEnvName(unittest.TestCase):
    """``PICKLE_SIGNING_KEY_ENV`` is the literal env-var NAME (not overridable)."""

    def test_name_is_na0s_pickle_key(self):
        cfg = _reload_config()
        self.assertEqual(cfg.PICKLE_SIGNING_KEY_ENV, "NA0S_PICKLE_KEY")

    def test_name_is_not_env_overridable(self):
        """Renaming the var that names the key var is circular — it stays fixed.
        Downstream docs/tests assert the literal, so an override must NOT win."""
        with patch.dict(os.environ, {"PICKLE_SIGNING_KEY_ENV": "SOMETHING_ELSE"}):
            cfg = _reload_config()
            self.assertEqual(cfg.PICKLE_SIGNING_KEY_ENV, "NA0S_PICKLE_KEY")

    def test_safe_pickle_uses_the_config_name(self):
        """safe_pickle reads the env var whose name is the config constant.

        Does NOT reload safe_pickle — that would swap its module-level digest
        caches out from under other tests that imported them by reference. The
        constant is bound at import; comparing the live binding is sufficient.
        """
        import na0s.config as cfg
        import na0s.integrity.safe_pickle as sp
        self.assertEqual(sp.PICKLE_SIGNING_KEY_ENV, cfg.PICKLE_SIGNING_KEY_ENV)
        self.assertEqual(sp.PICKLE_SIGNING_KEY_ENV, "NA0S_PICKLE_KEY")


class TestChunkSizeIsDigestIrrelevant(unittest.TestCase):
    """The core K1 correctness claim: the digest does NOT depend on chunk size,
    so externalizing/overriding the chunk size can never change which files
    verify. This is the anti-hollow assertion — same file, two chunk sizes,
    one digest; and a tampered file is still rejected under a non-default chunk.
    """

    def setUp(self):
        # Do NOT reload safe_pickle: it would swap out the module-level digest
        # caches that test_safe_pickle.py imported by reference, breaking the
        # bounded-LRU tests under collection order. Mutate the chunk-size global
        # on the LIVE module and restore it in tearDown instead.
        import na0s.integrity.safe_pickle as sp
        self.sp = sp
        self.sp._reset_caches()
        self._orig_chunk = sp.INTEGRITY_HASH_CHUNK_BYTES
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self.sp.INTEGRITY_HASH_CHUNK_BYTES = self._orig_chunk
        self.sp._reset_caches()
        self._tmp.cleanup()

    def test_sha256_digest_independent_of_chunk(self):
        p = os.path.join(self.tmp, "blob.bin")
        with open(p, "wb") as f:
            f.write(os.urandom(200_000))  # spans many chunks at 4 KiB
        self.sp.INTEGRITY_HASH_CHUNK_BYTES = 4096
        small = self.sp._sha256(p)
        self.sp.INTEGRITY_HASH_CHUNK_BYTES = 1 << 20
        big = self.sp._sha256(p)
        self.assertEqual(small, big)

    def test_hmac_digest_independent_of_chunk(self):
        p = os.path.join(self.tmp, "blob.bin")
        with open(p, "wb") as f:
            f.write(os.urandom(200_000))
        key = b"a_signing_key_long_enough_32_byteslong"
        self.sp.INTEGRITY_HASH_CHUNK_BYTES = 4096
        small = self.sp._hmac_sha256(p, key)
        self.sp.INTEGRITY_HASH_CHUNK_BYTES = 1 << 20
        big = self.sp._hmac_sha256(p, key)
        self.assertEqual(small, big)

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_round_trip_loads_under_small_chunk(self):
        os.environ.pop("NA0S_PICKLE_KEY", None)
        self.sp.INTEGRITY_HASH_CHUNK_BYTES = 4096
        obj = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}
        p = os.path.join(self.tmp, "m.pkl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.sp.safe_dump(obj, p)
            loaded = self.sp.safe_load(p)
        self.assertEqual(loaded, obj)

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_tamper_rejected_under_small_chunk(self):
        os.environ.pop("NA0S_PICKLE_KEY", None)
        self.sp.INTEGRITY_HASH_CHUNK_BYTES = 4096
        obj = {"weights": [1.0, 2.0, 3.0]}
        p = os.path.join(self.tmp, "m.pkl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.sp.safe_dump(obj, p)
        self.sp._reset_caches()
        with open(p, "r+b") as f:
            f.seek(10)
            f.write(b"\xff")
        with self.assertRaises(ValueError) as cm:
            self.sp.safe_load(p)
        self.assertIn("Integrity check failed", str(cm.exception))


class TestK4WordingDriftGuard(unittest.TestCase):
    """The K4 interpolation must keep the operator-facing wording byte-identical
    where downstream tests assert a substring — specifically the keyless
    safe_dump warning must still contain the literal ``NA0S_PICKLE_KEY``.
    """

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_keyless_dump_warning_contains_literal_env_name(self):
        import na0s.integrity.safe_pickle as sp
        os.environ.pop("NA0S_PICKLE_KEY", None)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pkl")
            with self.assertWarns(UserWarning) as cm:
                sp.safe_dump({"a": 1}, p)
            self.assertIn("NA0S_PICKLE_KEY is not set", str(cm.warning))


if __name__ == "__main__":
    unittest.main()
