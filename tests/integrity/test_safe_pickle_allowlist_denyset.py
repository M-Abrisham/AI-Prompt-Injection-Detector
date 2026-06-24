"""Tests for the restricted-unpickler allowlist + gadget deny-set (item #04b / R5).

``_SafeUnpickler`` replaces the bare ``pickle.load`` in ``safe_load`` so that
``find_class`` is constrained to a MEASURED numpy/sklearn allowlist intersected
with a fickling-catalogued gadget deny-set. These tests assert concrete,
observable outcomes (a return value of the expected type, a raised
``UnpicklingError`` naming the blocked global, an audit record, a sentinel file
that is NEVER created), never merely "no crash" — so each test FAILS if the
deny/allow policy is broken:

  * Revert line 964 to ``pickle.load(io.BytesIO(data))`` -> every Section A
    rejection test fails (the gadget executes; the sentinel appears).
  * Drop a numpy gadget prefix from ``_DENY_PREFIXES`` -> the §A3 numpy-gadget
    test fails (it rides in on the ``numpy.`` allow-prefix).
  * Over-tighten the allow-set (drop a measured ``numpy._core``/``sklearn`` pair)
    -> the §B bundled-artifact loads false-reject and fail.

Reuses the env-key fixture pattern from ``test_safe_pickle.py``.
"""

import io
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import pytest

from na0s.integrity.safe_pickle import (
    _GADGET_DENY,
    _PICKLE_ALLOW_EXACT,
    _SafeUnpickler,
    safe_dump,
    safe_load,
)
from na0s.models import get_model_path


# ---------------------------------------------------------------------------
# Adversarial payloads (module-level so they pickle by reference)
# ---------------------------------------------------------------------------
#
# Each __reduce__ targets a known RCE/abuse gadget. The reduce CALLABLE writes a
# sentinel file; if the unpickler ever resolved + invoked it, the sentinel would
# exist. The deny-set must raise during find_class (opcode interpretation),
# BEFORE the callable is invoked, so the sentinel is never created.

_SENTINEL = {"path": None}


def _touch_sentinel():  # pragma: no cover - must NEVER run in these tests
    with open(_SENTINEL["path"], "w", encoding="utf-8") as f:
        f.write("pwned")


class _EvilOsSystem:
    def __reduce__(self):
        # os.system pickles as posix.system on macOS/Linux, nt.system on Windows
        # — both are in _DENY_PREFIXES, so the global is blocked before exec.
        return (os.system, ("touch " + _SENTINEL["path"],))


class _EvilEval:
    def __reduce__(self):
        return (eval, ("open(%r,'w').write('pwned')" % _SENTINEL["path"],))


class _EvilExec:
    def __reduce__(self):
        return (exec, ("open(%r,'w').write('pwned')" % _SENTINEL["path"],))


class _EvilSubprocess:
    def __reduce__(self):
        import subprocess
        return (subprocess.Popen, (["touch", _SENTINEL["path"]],))


class _EvilNumpyGadget:
    """Targets a fickling-catalogued numpy gadget that lives UNDER the allowed
    ``numpy.`` prefix — the R5-specific proof that prefix-allow is intersected
    with the deny-set."""

    def __reduce__(self):
        import numpy.testing._private.utils as u  # noqa: WPS433
        return (u.runstring, ("open(%r,'w').write('x')" % _SENTINEL["path"], {}))


# ---------------------------------------------------------------------------
# Section A — deny-set rejects gadgets BEFORE execution
# ---------------------------------------------------------------------------

class TestGadgetRejection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        _SENTINEL["path"] = os.path.join(self._tmp.name, "PWNED")

    def tearDown(self):
        self.assertFalse(
            os.path.exists(_SENTINEL["path"]),
            "sentinel exists -> a gadget EXECUTED (deny-set failed)",
        )
        self._tmp.cleanup()

    def _assert_blocked(self, evil):
        raw = pickle.dumps(evil)
        with self.assertRaises(pickle.UnpicklingError):
            _SafeUnpickler(io.BytesIO(raw)).load()
        self.assertFalse(
            os.path.exists(_SENTINEL["path"]),
            "rejection was NOT pre-execution (sentinel created)",
        )

    def test_reduce_os_system_blocked(self):
        self._assert_blocked(_EvilOsSystem())

    def test_reduce_builtins_eval_blocked(self):
        self._assert_blocked(_EvilEval())

    def test_reduce_builtins_exec_blocked(self):
        self._assert_blocked(_EvilExec())

    def test_reduce_subprocess_popen_blocked(self):
        self._assert_blocked(_EvilSubprocess())

    def test_numpy_gadget_blocked_despite_allow_prefix(self):
        """A numpy gadget under the allowed ``numpy.`` prefix is STILL blocked.

        This is the R5 hardening of 04...md:128-130's blanket-numpy trust: the
        prefix-allow must be intersected with the gadget deny-set.
        """
        try:
            import numpy.testing._private.utils  # noqa: F401
        except Exception:
            pytest.skip("numpy.testing._private.utils.runstring not importable here")
        raw = pickle.dumps(_EvilNumpyGadget())
        with self.assertRaises(pickle.UnpicklingError) as ctx:
            _SafeUnpickler(io.BytesIO(raw)).load()
        self.assertIn("numpy.testing", str(ctx.exception))
        self.assertFalse(os.path.exists(_SENTINEL["path"]))

    def test_unpickling_error_message_names_blocked_global(self):
        raw = pickle.dumps(_EvilEval())
        with self.assertRaises(pickle.UnpicklingError) as ctx:
            _SafeUnpickler(io.BytesIO(raw)).load()
        # Names the specific blocked global, not a generic message.
        self.assertIn("eval", str(ctx.exception))

    def test_blocked_event_audited(self):
        raw = pickle.dumps(_EvilEval())
        with self.assertLogs("na0s.integrity_audit", level="ERROR") as cm:
            with self.assertRaises(pickle.UnpicklingError):
                _SafeUnpickler(io.BytesIO(raw)).load()
        joined = "\n".join(cm.output)
        self.assertIn("find_class_blocked", joined)
        self.assertIn("eval", joined)


class TestSafeLoadValidSidecarGadget(unittest.TestCase):
    """The teeth that the existing digest gate does NOT provide: a malicious
    pickle carrying a VALID sidecar (a sidecar-rewrite / KNOWN_HASHES-poison
    adversary) passes digest verification, yet ``_SafeUnpickler`` still rejects
    it at find_class BEFORE the payload runs."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        _SENTINEL["path"] = os.path.join(self._tmp.name, "PWNED_SAFELOAD")
        self._evil_path = os.path.join(self._tmp.name, "evil.pkl")

    def tearDown(self):
        self._tmp.cleanup()

    @patch.dict(os.environ, {k: v for k, v in os.environ.items()
                             if k != "NA0S_PICKLE_KEY"})
    def test_valid_sidecar_malicious_pickle_rejected_pre_execution(self):
        os.environ.pop("NA0S_PICKLE_KEY", None)
        os.environ.pop("NA0S_ALLOW_SHA256_DOWNGRADE", None)
        # safe_dump writes a VALID .sha256 over the malicious bytes -> the digest
        # gate passes; only the allowlist can stop it.
        safe_dump(_EvilOsSystem(), self._evil_path)
        with self.assertRaises(pickle.UnpicklingError):
            safe_load(self._evil_path)
        self.assertFalse(
            os.path.exists(_SENTINEL["path"]),
            "safe_load executed the payload despite a valid sidecar",
        )


# ---------------------------------------------------------------------------
# Section B — NO false-rejects (the bundled models MUST still load)
# ---------------------------------------------------------------------------

class TestBundledArtifactsStillLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pytest.importorskip("sklearn")
        pytest.importorskip("numpy")

    def _load(self, basename, expected_type_name):
        try:
            path = get_model_path(basename)
        except Exception as exc:  # pragma: no cover
            self.skipTest("cannot resolve %s: %s" % (basename, exc))
        if not os.path.exists(path):
            self.skipTest("%s not present on disk" % basename)
        obj = safe_load(path)
        self.assertIsNotNone(obj)
        self.assertEqual(type(obj).__name__, expected_type_name)
        return obj

    def test_model_pkl_loads(self):
        self._load("model.pkl", "CalibratedClassifierCV")

    def test_tfidf_vectorizer_loads(self):
        self._load("tfidf_vectorizer.pkl", "TfidfVectorizer")

    def test_structural_scaler_loads(self):
        self._load("structural_scaler.pkl", "StandardScaler")

    def test_model_embedding_loads(self):
        self._load("model_embedding.pkl", "CalibratedClassifierCV")


class TestBenignRoundTrip(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = os.path.join(self._tmp.name, "benign.pkl")

    def tearDown(self):
        self._tmp.cleanup()

    @patch.dict(os.environ, {"NA0S_PICKLE_KEY": "benign-round-trip-key-32-bytes!!"})
    def test_plain_dict_round_trips(self):
        obj = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        safe_dump(obj, self._path)
        loaded = safe_load(self._path)
        self.assertEqual(loaded, obj)


class TestNumpyCoreRemapBothSpellings(unittest.TestCase):
    """Both the numpy ``_core`` (2.x) and legacy ``core`` spellings of the
    measured reconstruction primitives must resolve through the policy — they
    are emitted by the SAME bundled artifacts (§1)."""

    def setUp(self):
        pytest.importorskip("numpy")

    def _resolve(self, module, name):
        # find_class must NOT raise for a measured-allowed numpy reconstruction
        # primitive in either spelling.
        u = _SafeUnpickler(io.BytesIO(b""))
        return u.find_class(module, name)

    def test_core_multiarray_scalar_allowed(self):
        self.assertIsNotNone(self._resolve("numpy.core.multiarray", "scalar"))

    def test__core_multiarray_scalar_allowed(self):
        self.assertIsNotNone(self._resolve("numpy._core.multiarray", "scalar"))

    def test_core_multiarray_reconstruct_allowed(self):
        self.assertIsNotNone(self._resolve("numpy.core.multiarray", "_reconstruct"))

    def test_numpy_core_exceptions_denied_both_spellings(self):
        # numpy.core._exceptions / numpy._core._exceptions are a DENY prefix —
        # the remap must make the deny spelling-agnostic so neither rides in on
        # the numpy. allow-prefix.
        for module in ("numpy.core._exceptions", "numpy._core._exceptions"):
            with self.assertRaises(pickle.UnpicklingError):
                _SafeUnpickler(io.BytesIO(b"")).find_class(module, "_ArrayMemoryError")


# ---------------------------------------------------------------------------
# Section C — policy-constant sanity (the allow-set is the MEASURED set)
# ---------------------------------------------------------------------------

class TestPolicyConstants(unittest.TestCase):
    def test_measured_sklearn_estimators_in_allow_exact(self):
        # The four bundled artifacts deserialize into these — if any were
        # dropped from _PICKLE_ALLOW_EXACT the §B loads would false-reject.
        for pair in [
            ("sklearn.calibration", "CalibratedClassifierCV"),
            ("sklearn.feature_extraction.text", "TfidfVectorizer"),
            ("sklearn.preprocessing._data", "StandardScaler"),
        ]:
            self.assertIn(pair, _PICKLE_ALLOW_EXACT)

    def test_universal_rce_primitives_in_deny(self):
        # builtins.eval/exec must be in the exact-gadget deny-set.
        self.assertIn(("builtins", "eval"), _GADGET_DENY)
        self.assertIn(("builtins", "exec"), _GADGET_DENY)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
