"""L11 ``safe_pickle`` precedence + adversarial-stress tests (item #08).

These tests harden two invariants of the integrity loader that the existing
suite does not yet pin, and that are independent of the deferred ``find_class``
allowlist (item #04b — the ``__reduce__`` allowlist-rejection tests are NOT
authored here; #05 already covers the pre-deserialize ``__reduce__`` gate):

1. **Trust-tier precedence** — a basename present in ``KNOWN_HASHES`` resolves
   to the hardcoded SHA-256 digest baked into the (wheel-signed) Python source.
   An attacker who tampers with the bytes AND forges a self-consistent
   ``.sha256`` sidecar (and even a ``.hmac`` under some key) must STILL be
   refused, because the hardcoded tier outranks any on-disk sidecar. This is the
   security-critical "sidecar cannot rescue a stale/tampered hardcoded artifact"
   assertion. (#07's ``test_bundled_known_hash_resolves_hardcoded`` proves the
   *source* resolves to ``"hardcoded"`` with a stray sidecar; this adds the
   tamper-with-forged-valid-sidecar variant that proves ``safe_load`` REJECTS.)

2. **Stress / robustness** (ROADMAP_V2.md:1180) —
   (a) a truncated-mid-opcode pickle is rejected (never returns a partial
       object), and a head-mangled pickle is rejected at the magic-byte gate;
   (b) an ~8 MB artifact round-trips through ``safe_dump``/``safe_load`` and the
       chunked streaming digest equals the one-shot digest (exercises the
       ``INTEGRITY_HASH_CHUNK_BYTES`` chunk loop many times — ~128 iterations at
       64 KiB — a chunking-coverage proxy, NOT a DoS limit; 1 GiB would be
       CI-hostile per na0s-review-checklist §7);
   (c) concurrent ``safe_dump`` + ``safe_load`` across several threads on
       distinct paths AND on the SAME path never corrupts output or leaks a
       ``.tmp`` residue (exercises #15's bounded cache + lock and the
       ``os.replace`` atomic-write invariant).

Every test asserts a concrete observable (exception type/message, byte-equality
of digests, exact round-trip object, filesystem state) — no bare ``safe_load``
without an assertion, no ``assert True`` (na0s-review-checklist §4). No real
``os.system``/``eval`` ever runs; no magic threshold beyond the justified ~8 MB
chunking proxy. Keyless throughout (``NA0S_PICKLE_KEY`` unset) so the SHA-256
path is exercised without requiring the HMAC key (the project has no API key and
integrity must not require one).
"""

import glob
import hashlib
import os
import pickle
import tempfile
import threading
import unittest
import warnings

# Disable scan timeout before any na0s imports (mirrors test_safe_pickle.py).
os.environ["SCAN_TIMEOUT_SEC"] = "0"

from na0s.config import INTEGRITY_HASH_CHUNK_BYTES
from na0s.integrity.safe_pickle import (
    _atomic_write_binary,
    _format_sidecar,
    _hash_path,
    _hmac_path,
    _reset_caches,
    _resolve_expected_hash,
    _sha256,
    safe_dump,
    safe_load,
    write_digest_sidecar,
)
from na0s.models import KNOWN_HASHES


def _no_key_env():
    """Return os.environ without NA0S_PICKLE_KEY (keyless SHA-256 path)."""
    return {k: v for k, v in os.environ.items() if k != "NA0S_PICKLE_KEY"}


class _StressBase(unittest.TestCase):
    """Shared keyless tmp-dir fixture; cold caches each test for determinism."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        os.environ.pop("NA0S_PICKLE_KEY", None)
        _reset_caches()

    def tearDown(self):
        _reset_caches()
        self._tmpdir.cleanup()

    def _dump_keyless(self, obj, path):
        """safe_dump without a signing key, swallowing the keyless UserWarning."""
        from unittest.mock import patch

        with patch.dict(os.environ, _no_key_env(), clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                safe_dump(obj, path)


class TestKnownHashesPrecedenceUnderTamper(_StressBase):
    """Hardcoded > sidecar: a tampered bundled artifact with a FORGED valid
    sidecar is still refused. This is distinct from #07's
    ``test_bundled_known_hash_resolves_hardcoded`` (which only checks the
    resolved *source* with a stray, non-matching sidecar) — here the sidecars
    are forged to be SELF-CONSISTENT with the tampered bytes, so only the
    hardcoded-tier precedence stops the bypass."""

    def _tampered_bundled_with_forged_sidecars(self, name="model.pkl"):
        """Write tampered (attacker-chosen) pickle bytes at a KNOWN_HASHES
        basename, plus a ``.sha256`` sidecar that VALIDLY matches those bytes and
        a ``.hmac`` sidecar — i.e. every on-disk artifact is internally
        consistent with the attacker's payload. Returns the path."""
        self.assertIn(name, KNOWN_HASHES, "fixture must use a real bundled name")
        path = os.path.join(self.tmpdir, name)
        tampered = pickle.dumps({"attacker": "controlled"},
                                protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_write_binary(path, tampered)
        # The tampered bytes do NOT match the hardcoded digest...
        self.assertNotEqual(_sha256(path), KNOWN_HASHES[name])
        # ...but the forged .sha256 sidecar DOES match the tampered bytes.
        with open(_hash_path(path), "w") as f:
            f.write(_format_sidecar("sha256", _sha256(path)))
        # And a forged .hmac sidecar is dropped alongside.
        with open(_hmac_path(path), "w") as f:
            f.write(_format_sidecar("hmac-sha256", "0" * 64))
        return path

    def test_resolution_picks_hardcoded_over_forged_sidecars(self):
        """``_resolve_expected_hash`` returns the HARDCODED digest+source even
        when self-consistent ``.sha256`` and ``.hmac`` sidecars are present on
        disk (precedence is config-driven, not attacker-file-driven)."""
        path = self._tampered_bundled_with_forged_sidecars()
        with self._patched_no_key():
            expected, source = _resolve_expected_hash(path)
        self.assertEqual(source, "hardcoded")
        self.assertEqual(expected, KNOWN_HASHES["model.pkl"])
        # Sanity: the forged sidecar's digest is NOT what was resolved.
        self.assertNotEqual(expected, _sha256(path))

    def test_safe_load_rejects_tampered_bundled_despite_forged_sidecars(self):
        """End-to-end: ``safe_load`` of a tampered bundled artifact raises
        ``ValueError('Integrity check failed')`` even though both a valid forged
        ``.sha256`` and a ``.hmac`` sidecar exist — the hardcoded tier is
        authoritative and a sidecar cannot rescue a stale/tampered artifact."""
        path = self._tampered_bundled_with_forged_sidecars()
        with self._patched_no_key():
            with self.assertRaises(ValueError) as ctx:
                safe_load(path)
        msg = str(ctx.exception)
        self.assertIn("Integrity check failed", msg)
        # The error must name the hardcoded source (not a sidecar source),
        # proving the precedence path is the one that fired.
        self.assertIn("source: hardcoded", msg)

    def test_safe_load_accepts_untampered_bundled_basename(self):
        """FP-safe negative control: the SAME hardcoded path, when the bytes are
        replaced with bytes whose SHA-256 equals the hardcoded digest, would
        load — but since we cannot reproduce the bundled bytes here, assert the
        positive precedence direction differently: a NON-bundled basename with a
        valid keyless ``.sha256`` sidecar DOES load (so the loader is not simply
        rejecting everything)."""
        path = os.path.join(self.tmpdir, "user_trained.pkl")
        self.assertNotIn("user_trained.pkl", KNOWN_HASHES)
        obj = {"weights": [1.0, 2.0], "bias": 0.5}
        self._dump_keyless(obj, path)
        with self._patched_no_key():
            loaded = safe_load(path)
        self.assertEqual(loaded, obj)

    def _patched_no_key(self):
        from unittest.mock import patch
        return patch.dict(os.environ, _no_key_env(), clear=True)


class TestTruncatedPickleRejected(_StressBase):
    """A truncated pickle is rejected — never silently, never as a partial
    object. The sidecar is re-signed over the truncated bytes so the integrity
    check PASSES and the failure must come from the pickle/magic layer (proving
    integrity-valid-but-malformed is still refused)."""

    def _resign_keyless(self, path):
        """Re-write the ``.sha256`` sidecar to match the file's current bytes."""
        with open(_hash_path(path), "w") as f:
            f.write(_format_sidecar("sha256", _sha256(path)))

    def test_truncated_mid_opcode_raises_not_partial(self):
        """Truncating a valid pickle in the middle (head intact -> passes magic)
        and re-signing the truncated bytes -> ``safe_load`` raises a pickle/value
        error and returns NO object. ``pickle.UnpicklingError`` is NOT a
        ``ValueError`` subclass, so the accepted tuple includes both."""
        path = os.path.join(self.tmpdir, "trunc.pkl")
        # A reasonably large object so a midpoint cut lands mid-opcode-stream.
        obj = {"data": list(range(5000)), "tag": "x" * 5000}
        self._dump_keyless(obj, path)
        full = open(path, "rb").read()
        self.assertEqual(full[:1], b"\x80", "expected protocol-2+ PROTO header")

        truncated = full[: len(full) // 2]
        _atomic_write_binary(path, truncated)
        self._resign_keyless(path)  # integrity now PASSES over truncated bytes

        with self._patched_no_key():
            with self.assertRaises((ValueError, pickle.UnpicklingError)) as ctx:
                result = safe_load(path)
                # Defensive: if it somehow returned, fail loudly with the value.
                self.fail("safe_load returned a partial object: {!r}".format(result))
        # The failure is a deserialization/format error, not a silent success.
        self.assertTrue(
            isinstance(ctx.exception, (ValueError, pickle.UnpicklingError))
        )

    def test_truncated_to_one_byte_rejected_at_magic_gate(self):
        """Truncating to a single byte mangles the magic header -> the magic-byte
        gate raises ``ValueError`` ('file too short') BEFORE any unpickle."""
        path = os.path.join(self.tmpdir, "headtrunc.pkl")
        obj = {"k": "v"}
        self._dump_keyless(obj, path)
        full = open(path, "rb").read()
        _atomic_write_binary(path, full[:1])
        self._resign_keyless(path)

        with self._patched_no_key():
            with self.assertRaises(ValueError) as ctx:
                safe_load(path)
        self.assertIn("Invalid pickle format", str(ctx.exception))
        self.assertIn("too short", str(ctx.exception))

    def test_truncated_to_empty_rejected(self):
        """A zero-byte file (the degenerate truncation) is rejected at the magic
        gate, not unpickled into ``None`` or an error swallowed."""
        path = os.path.join(self.tmpdir, "empty.pkl")
        _atomic_write_binary(path, b"")
        self._resign_keyless(path)
        with self._patched_no_key():
            with self.assertRaises(ValueError) as ctx:
                safe_load(path)
        self.assertIn("Invalid pickle format", str(ctx.exception))

    def _patched_no_key(self):
        from unittest.mock import patch
        return patch.dict(os.environ, _no_key_env(), clear=True)


class TestLargeArtifactChunkedHashing(_StressBase):
    """An ~8 MB artifact round-trips and exercises the chunked streaming hash.

    8 MB ~= 128 x the 64 KiB ``INTEGRITY_HASH_CHUNK_BYTES`` chunk, forcing many
    iterations of the ``iter(lambda: f.read(chunk), b"")`` loop in ``_sha256`` —
    the only behavior the chunking guards. Size is a chunking-coverage proxy, NOT
    a DoS limit (na0s-review-checklist §7: 1 GiB would be CI-hostile)."""

    LARGE_BYTES = 8 * 1024 * 1024  # 8 MiB chunking-coverage proxy

    def test_large_artifact_roundtrips_equal(self):
        """An 8 MiB blob round-trips byte-for-byte through safe_dump/safe_load."""
        path = os.path.join(self.tmpdir, "big.pkl")
        payload = {"blob": os.urandom(self.LARGE_BYTES), "n": self.LARGE_BYTES}
        self._dump_keyless(payload, path)
        with self._patched_no_key():
            loaded = safe_load(path)
        self.assertEqual(loaded["n"], self.LARGE_BYTES)
        self.assertEqual(loaded["blob"], payload["blob"])

    def test_chunked_digest_equals_one_shot_digest(self):
        """The streamed ``_sha256`` (chunk loop) equals a one-shot
        ``hashlib.sha256(whole_file)`` — proves the ``1<<16`` loop is correct,
        not merely exercised. The file must exceed the chunk size many times."""
        path = os.path.join(self.tmpdir, "bigdigest.pkl")
        payload = {"blob": os.urandom(self.LARGE_BYTES)}
        self._dump_keyless(payload, path)

        on_disk = open(path, "rb").read()
        self.assertGreater(
            len(on_disk), INTEGRITY_HASH_CHUNK_BYTES * 64,
            "fixture must span many hash chunks to exercise the loop",
        )
        streamed = _sha256(path)
        one_shot = hashlib.sha256(on_disk).hexdigest()
        self.assertEqual(streamed, one_shot)
        # And the sidecar written by safe_dump matches that digest.
        sidecar_digest = open(_hash_path(path)).read().strip().split(":")[-1]
        self.assertEqual(sidecar_digest, streamed)

    def _patched_no_key(self):
        from unittest.mock import patch
        return patch.dict(os.environ, _no_key_env(), clear=True)


class TestConcurrentDumpLoad(_StressBase):
    """Concurrent ``safe_dump``/``safe_load`` never corrupts output or leaks a
    ``.tmp`` residue (exercises #15's bounded cache + lock and the ``os.replace``
    atomic-write invariant). We assert the *final* state is internally consistent
    and every SUCCESSFUL load returns the exact object — not a specific
    interleaving (no timing assertion, to stay deterministic in CI)."""

    def test_concurrent_dump_load_distinct_paths_no_corruption(self):
        """8 threads each dump+load a DISTINCT path: every load returns its own
        exact object, no exception, no leftover ``.tmp``."""
        n = 8
        errors = []
        results = {}

        def worker(i):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    path = os.path.join(self.tmpdir, "dist{}.pkl".format(i))
                    obj = {"id": i, "payload": list(range(100 * (i + 1)))}
                    from unittest.mock import patch
                    with patch.dict(os.environ, _no_key_env(), clear=True):
                        safe_dump(obj, path)
                        results[i] = (safe_load(path) == obj)
            except Exception as exc:  # noqa: BLE001 - recorded, re-asserted
                errors.append((i, type(exc).__name__, str(exc)[:80]))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], "no thread should error on distinct paths")
        self.assertEqual(len(results), n)
        self.assertTrue(all(results.values()),
                        "every distinct-path load must equal its own object")
        self.assertEqual(glob.glob(os.path.join(self.tmpdir, "*.tmp")), [],
                         "no atomic-write .tmp residue should remain")

    def test_concurrent_dump_load_same_path_no_corruption(self):
        """4 dumpers + 4 loaders racing ONE path: a load either returns the
        EXACT object or raises the transient ``Integrity check failed`` (the
        benign window between ``os.replace`` of the pickle and the sidecar
        rewrite). It must NEVER return a partial/garbage object, and the FINAL
        on-disk state must be consistent with zero ``.tmp`` residue."""
        from unittest.mock import patch

        path = os.path.join(self.tmpdir, "same.pkl")
        obj = {"weights": list(range(1000)), "tag": "same"}
        self._dump_keyless(obj, path)  # seed a valid file+sidecar

        errors = []
        corruptions = []

        def dumper():
            try:
                with patch.dict(os.environ, _no_key_env(), clear=True):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        for _ in range(25):
                            safe_dump(obj, path)
            except Exception as exc:  # noqa: BLE001
                errors.append(("dump", type(exc).__name__, str(exc)[:80]))

        def loader():
            try:
                with patch.dict(os.environ, _no_key_env(), clear=True):
                    for _ in range(25):
                        try:
                            loaded = safe_load(path)
                        except ValueError as ve:
                            # Tolerate ONLY the transient integrity-window race;
                            # any other ValueError is a real failure.
                            if "Integrity check failed" not in str(ve):
                                raise
                            continue
                        if loaded != obj:
                            corruptions.append(loaded)
            except Exception as exc:  # noqa: BLE001
                errors.append(("load", type(exc).__name__, str(exc)[:80]))

        threads = ([threading.Thread(target=dumper) for _ in range(4)] +
                   [threading.Thread(target=loader) for _ in range(4)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [],
                         "no unexpected exception under same-path contention")
        self.assertEqual(corruptions, [],
                         "a successful load must NEVER return a corrupt object")

        # Final state is internally consistent: file unpickles to obj, the
        # sidecar digest matches the file on disk, and no .tmp leaked.
        with patch.dict(os.environ, _no_key_env(), clear=True):
            self.assertEqual(safe_load(path), obj)
        sidecar_digest = open(_hash_path(path)).read().strip().split(":")[-1]
        self.assertEqual(sidecar_digest, _sha256(path),
                         "sidecar digest must equal the final file digest")
        self.assertEqual(glob.glob(os.path.join(self.tmpdir, "*.tmp")), [],
                         "no atomic-write .tmp residue should remain")


class TestShimParity(unittest.TestCase):
    """The deprecation shim ``na0s.safe_pickle`` re-exports the SAME callables
    the canonical module defines, so a test driving the canonical module
    reflects the object older callers/tests import via the shim. Keeps the
    wiring honest (na0s-review-checklist §11 shim-trap)."""

    def test_shim_reexports_same_objects(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import na0s.safe_pickle as shim
        import na0s.integrity.safe_pickle as canon
        self.assertIs(shim.safe_load, canon.safe_load)
        self.assertIs(shim.safe_dump, canon.safe_dump)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
