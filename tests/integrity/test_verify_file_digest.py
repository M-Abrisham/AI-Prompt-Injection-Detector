"""Tests for the format-agnostic integrity helpers added for item #05.

``verify_file_digest`` / ``write_digest_sidecar`` are extracted from
``safe_load`` / ``safe_dump`` so that *non-pickle* artifacts (a torch
``.pt``/``.pth`` zip) can be gated by the same digest trust hierarchy without
``safe_load``'s pickle-magic validation rejecting the zip header.

These tests assert concrete outcomes (a passing verify returns ``None``, a
tampered file raises ``ValueError``, a sidecar-less file raises
``FileNotFoundError``, a malicious payload is refused *before* any unpickling),
never merely "no crash".
"""

import os
import pickle
import tempfile

import pytest

from na0s.integrity.safe_pickle import (
    safe_load,
    verify_file_digest,
    write_digest_sidecar,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_artifact():
    """A throwaway non-pickle artifact path (basename NOT in KNOWN_HASHES)."""
    d = tempfile.TemporaryDirectory()
    path = os.path.join(d.name, "adapter_state.pt")
    with open(path, "wb") as f:
        f.write(b"PK\x03\x04 not-a-pickle torch-zip-like bytes \x00\x01\x02")
    try:
        yield path
    finally:
        d.cleanup()


@pytest.fixture
def keyless_env(monkeypatch):
    """Ensure a keyless deployment (no NA0S_PICKLE_KEY, no downgrade opt-out)."""
    monkeypatch.delenv("NA0S_PICKLE_KEY", raising=False)
    monkeypatch.delenv("NA0S_ALLOW_SHA256_DOWNGRADE", raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Code-level: write_digest_sidecar + verify_file_digest round-trip
# ---------------------------------------------------------------------------

class TestVerifyFileDigestKeyless:
    def test_roundtrip_passes(self, tmp_artifact, keyless_env):
        """A freshly side-cared file verifies (returns None, no raise)."""
        sidecar = write_digest_sidecar(tmp_artifact)
        assert sidecar == tmp_artifact + ".sha256"
        assert os.path.exists(sidecar)
        # verify returns None on success.
        assert verify_file_digest(tmp_artifact) is None

    def test_sidecar_content_matches_recomputed_digest(self, tmp_artifact, keyless_env):
        import hashlib

        write_digest_sidecar(tmp_artifact)
        with open(tmp_artifact, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        with open(tmp_artifact + ".sha256", "r", encoding="utf-8") as f:
            raw = f.read().strip()
        # Versioned sidecar: v1:sha256:<digest>
        assert raw == "v1:sha256:{}".format(expected)

    def test_tamper_raises_value_error(self, tmp_artifact, keyless_env):
        write_digest_sidecar(tmp_artifact)
        # Flip one byte AFTER writing the sidecar.
        with open(tmp_artifact, "r+b") as f:
            data = bytearray(f.read())
            data[0] ^= 0xFF
            f.seek(0)
            f.write(data)
        with pytest.raises(ValueError, match="Integrity check failed"):
            verify_file_digest(tmp_artifact)

    def test_missing_sidecar_raises_file_not_found(self, tmp_artifact, keyless_env):
        # No sidecar written and basename not in KNOWN_HASHES.
        with pytest.raises(FileNotFoundError):
            verify_file_digest(tmp_artifact)


class TestVerifyFileDigestHMAC:
    def test_hmac_roundtrip(self, tmp_artifact, monkeypatch):
        monkeypatch.setenv("NA0S_PICKLE_KEY", "verify-helper-key")
        sidecar = write_digest_sidecar(tmp_artifact)
        assert sidecar == tmp_artifact + ".hmac"
        assert verify_file_digest(tmp_artifact) is None

    def test_hmac_tamper_raises(self, tmp_artifact, monkeypatch):
        monkeypatch.setenv("NA0S_PICKLE_KEY", "verify-helper-key")
        write_digest_sidecar(tmp_artifact)
        with open(tmp_artifact, "ab") as f:
            f.write(b"appended-tamper")
        with pytest.raises(ValueError, match="Integrity check failed"):
            verify_file_digest(tmp_artifact)

    def test_hmac_sidecar_without_key_refused(self, tmp_artifact, monkeypatch):
        # Write an .hmac sidecar, then drop the key. With no .sha256 fallback,
        # a lone .hmac is unverifiable and must be refused (not silently passed).
        monkeypatch.setenv("NA0S_PICKLE_KEY", "verify-helper-key")
        write_digest_sidecar(tmp_artifact)
        monkeypatch.delenv("NA0S_PICKLE_KEY", raising=False)
        monkeypatch.delenv("NA0S_ALLOW_SHA256_DOWNGRADE", raising=False)
        with pytest.raises(ValueError, match="NA0S_PICKLE_KEY is not set"):
            verify_file_digest(tmp_artifact)


# ---------------------------------------------------------------------------
# Security: the gate fires BEFORE deserialization (no code execution)
# ---------------------------------------------------------------------------

_SENTINEL_HOLDER = {}


class _ExplodingPayload:
    """A pickle whose __reduce__ would write a sentinel file on unpickling.

    If verify_file_digest / safe_load ever unpickled this, the sentinel path
    would be created. The gate must raise FIRST so it never is.
    """

    def __reduce__(self):
        sentinel = _SENTINEL_HOLDER["path"]
        # os.rename of a path to itself is a side effect we can detect: instead
        # we use a builtin that, when executed during unpickling, creates the
        # sentinel file. We pickle a call to ``open`` + write via a helper.
        return (_write_sentinel, (sentinel,))


def _write_sentinel(path):  # pragma: no cover - must NEVER run in these tests
    with open(path, "w", encoding="utf-8") as f:
        f.write("pwned")
    return path


class TestGateFiresPreDeserialize:
    def test_malicious_pickle_no_sidecar_refused_before_unpickle(self, tmp_path, keyless_env):
        sentinel = str(tmp_path / "PWNED")
        _SENTINEL_HOLDER["path"] = sentinel
        evil_path = str(tmp_path / "evil.pkl")
        with open(evil_path, "wb") as f:
            pickle.dump(_ExplodingPayload(), f)

        # No sidecar exists -> FileNotFoundError, and crucially verify never
        # deserializes, so the sentinel is never created.
        with pytest.raises(FileNotFoundError):
            verify_file_digest(evil_path)
        assert not os.path.exists(sentinel), "payload executed despite missing sidecar"

        # safe_load on the same malicious file must also refuse pre-unpickle.
        with pytest.raises(FileNotFoundError):
            safe_load(evil_path)
        assert not os.path.exists(sentinel), "safe_load executed the payload"

    def test_malicious_pickle_tampered_sidecar_refused(self, tmp_path, keyless_env):
        sentinel = str(tmp_path / "PWNED2")
        _SENTINEL_HOLDER["path"] = sentinel
        evil_path = str(tmp_path / "evil2.pkl")
        with open(evil_path, "wb") as f:
            pickle.dump(_ExplodingPayload(), f)
        # Write a sidecar for the CURRENT bytes, then tamper the file so the
        # digest no longer matches -> ValueError before unpickling.
        write_digest_sidecar(evil_path)
        with open(evil_path, "ab") as f:
            f.write(b"\x00")
        with pytest.raises(ValueError, match="Integrity check failed"):
            safe_load(evil_path)
        assert not os.path.exists(sentinel), "tampered payload executed"


# ---------------------------------------------------------------------------
# safe_load still works end-to-end after the refactor (delegates to helper)
# ---------------------------------------------------------------------------

class TestSafeLoadDelegation:
    def test_safe_dump_safe_load_roundtrip_keyless(self, tmp_path, keyless_env):
        obj = {"a": [1, 2, 3], "b": "hello"}
        path = str(tmp_path / "obj.pkl")
        from na0s.integrity.safe_pickle import safe_dump

        safe_dump(obj, path)
        assert safe_load(path) == obj

    def test_safe_load_tamper_raises(self, tmp_path, keyless_env):
        from na0s.integrity.safe_pickle import safe_dump

        obj = {"a": 1}
        path = str(tmp_path / "obj2.pkl")
        safe_dump(obj, path)
        with open(path, "ab") as f:
            f.write(b"x")
        with pytest.raises(ValueError, match="Integrity check failed"):
            safe_load(path)


# ---------------------------------------------------------------------------
# torch adapter integration (skipped when torch absent)
# ---------------------------------------------------------------------------

class TestAdapterIntegration:
    def test_adapter_roundtrip_and_tamper(self, tmp_path, keyless_env):
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")
        import torch

        from na0s.ml.embedding_adapter import AdapterClassifier, EmbeddingAdapter

        clf = AdapterClassifier.__new__(AdapterClassifier)
        clf._hidden_dim = 16
        clf._dropout = 0.3
        clf._input_dim = 8
        clf._adapter = EmbeddingAdapter(input_dim=8, hidden_dim=16, dropout=0.3)
        clf._adapter.eval()

        path = str(tmp_path / "adapter.pt")
        clf.save(path)
        assert os.path.exists(path + ".sha256")

        # Reference forward output before reload.
        x = torch.randn(2, 8)
        with torch.no_grad():
            before = clf._adapter(x)

        clf2 = AdapterClassifier.__new__(AdapterClassifier)
        clf2._hidden_dim = 16
        clf2._dropout = 0.3
        clf2.load(path, input_dim=8)
        with torch.no_grad():
            after = clf2._adapter(x)
        assert torch.allclose(before, after, atol=1e-6)

        # Tamper -> load refuses before torch.load.
        with open(path, "ab") as f:
            f.write(b"tamper")
        clf3 = AdapterClassifier.__new__(AdapterClassifier)
        clf3._hidden_dim = 16
        clf3._dropout = 0.3
        with pytest.raises(ValueError, match="Integrity check failed"):
            clf3.load(path, input_dim=8)

    def test_adapter_load_missing_sidecar_refused(self, tmp_path, keyless_env):
        pytest.importorskip("torch")
        import torch

        from na0s.ml.embedding_adapter import AdapterClassifier, EmbeddingAdapter

        adapter = EmbeddingAdapter(input_dim=8, hidden_dim=16, dropout=0.3)
        path = str(tmp_path / "legacy_adapter.pt")
        torch.save(adapter.state_dict(), path)  # no sidecar (legacy artifact)

        clf = AdapterClassifier.__new__(AdapterClassifier)
        clf._hidden_dim = 16
        clf._dropout = 0.3
        with pytest.raises(FileNotFoundError):
            clf.load(path, input_dim=8)
