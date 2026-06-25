"""End-to-end supply-chain *use-case* scenarios for the load-time integrity gate.

Why a separate file (and why NOT the F14 ``scan(text)`` corpus)
---------------------------------------------------------------
The Na0S taxonomy already defines the load-time supply-chain attack classes —
``S1.1`` Model-file-tampering, ``S1.2`` Pickle-RCE, ``AD2.2`` Framework
deserialization attack, ``IM5.4`` Rug-pull (model swap after trust), and
``IM5.5`` Supply-chain model poisoning — but **zero F14 scenarios exercise
them**.  That gap is correct, not a bug: these are *load-time* gates.  The threat
is a tampered ``.pkl`` / forged sidecar on disk, not a malicious string flowing
through ``predict.scan(text)``.  A ``scan(text)`` scenario can never reach the
``safe_load`` / ``ModelRollback.restore`` code path, so the right home for these
use-cases is ``tests/integrity/`` (here), exercising the gate the way a real
deployment hits it.

Each test names its taxonomy code in a comment so the coverage is auditable.
Everything is local / keyless: ``NA0S_PICKLE_KEY`` is set per-test only where the
HMAC tier is under test, and no network or API key is required.

Scenario map
------------
* ``test_tampered_pickle_rejected_by_safe_load``     -> S1.2 / S1.1
* ``test_sha256_downgrade_refused_when_key_set``     -> S1.1  (downgrade gate)
* ``test_gadget_pickle_blocked``  (xfail, PR #478)   -> S1.2 / AD2.2
* ``test_rollback_restore_rejects_tampered_backup``  -> IM5.5 (cross-ref S1.7)

The gadget-pickle test is marked ``xfail(strict=True)`` rather than omitted: the
``_SafeUnpickler.find_class`` allowlist that would block a *valid-digest* gadget
pickle lands in PR #478 and is NOT on this branch.  Marking it strict-xfail keeps
the AD2.2 / S1.2 ``find_class`` gap TRACKED (it flips to a hard failure the moment
the allowlist arrives) instead of silently hidden.
"""

from __future__ import annotations

import os
import pickle

import pytest

from na0s.integrity import safe_pickle
from na0s.integrity.safe_pickle import (
    _reset_caches,
    safe_dump,
    safe_load,
    verify_file_digest,
)


@pytest.fixture(autouse=True)
def _cold_digest_cache():
    """Start every test from a cold digest cache.

    ``safe_pickle`` memoises file digests keyed on (path, mtime_ns, size, inode).
    The tmp_path files below are written, then deliberately tampered; a cold
    cache removes any chance of a stale digest masking a tamper across tests.
    """
    _reset_caches()
    yield
    _reset_caches()


# ===================================================================
# (a) S1.2 / S1.1 — a tampered .pkl is rejected by safe_load (digest mismatch)
# ===================================================================

class TestTamperedPickleRejected:
    """A model file mutated after signing must fail the load-time integrity gate.

    Taxonomy: S1.2 (Pickle-RCE — the unverified bytes would be the RCE vector) /
    S1.1 (Model-file-tampering — the on-disk artifact was swapped).
    """

    def test_clean_pickle_round_trips(self, tmp_path):
        """Baseline: a safe_dump'd model loads back identically (keyless SHA-256)."""
        model = tmp_path / "model.pkl"
        obj = {"weights": [1, 2, 3], "framework": "scikit-learn"}
        safe_dump(obj, str(model))
        # Sidecar written and the load verifies + returns the object.
        assert (tmp_path / "model.pkl.sha256").exists()
        assert safe_load(str(model)) == obj

    def test_tampered_pickle_rejected_by_safe_load(self, tmp_path):
        """S1.2/S1.1: mutating the pickle after signing fails safe_load closed."""
        model = tmp_path / "model.pkl"
        safe_dump({"weights": [1, 2, 3]}, str(model))
        _reset_caches()  # drop the digest cached by safe_dump's sidecar write

        # Attacker swaps the payload for a different (still well-formed) pickle but
        # cannot update the signed .sha256 digest -> constant-time compare fails.
        model.write_bytes(pickle.dumps({"weights": "evil"}))
        _reset_caches()

        with pytest.raises(ValueError, match="Integrity check failed|tampered"):
            safe_load(str(model))

    def test_verify_file_digest_detects_tamper_without_unpickling(self, tmp_path):
        """S1.1: the format-agnostic gate flags tamper before any deserialization.

        ``verify_file_digest`` never unpickles, so it is the safe pre-load probe:
        a tampered artifact is rejected without the attacker bytes ever reaching
        ``pickle.load``.
        """
        model = tmp_path / "model.pkl"
        safe_dump({"ok": True}, str(model))
        _reset_caches()
        model.write_bytes(pickle.dumps({"ok": False, "tamper": 1}))
        _reset_caches()
        with pytest.raises(ValueError, match="Integrity check failed|tampered"):
            verify_file_digest(str(model))


# ===================================================================
# (b) S1.1 — key-present + sha256-only sidecar is a refused downgrade
# ===================================================================

class TestSha256DowngradeRefused:
    """When NA0S_PICKLE_KEY is set, a plain-SHA256-only sidecar is a downgrade.

    Taxonomy: S1.1 (Model-file-tampering). An attacker who can drop a file on disk
    could strip the unforgeable ``.hmac`` and leave only an attacker-recomputable
    ``.sha256``. Operator opted into HMAC by setting the key, so the gate must
    FAIL CLOSED rather than silently accept the weaker, forgeable sidecar.
    """

    def test_sha256_only_sidecar_refused_when_key_set(self, tmp_path, monkeypatch):
        """S1.1: key set + only a .sha256 sidecar -> safe_load refuses (downgrade)."""
        model = tmp_path / "model.pkl"
        # Write the pickle + a plain SHA-256 sidecar in the KEYLESS regime.
        safe_dump({"weights": [1, 2, 3]}, str(model))
        assert (tmp_path / "model.pkl.sha256").exists()
        assert not (tmp_path / "model.pkl.hmac").exists()

        # Operator now opts into HMAC. The lone .sha256 is a downgrade from the
        # configured tier and must be refused by default (fail closed).
        monkeypatch.setenv("NA0S_PICKLE_KEY", os.urandom(32).hex())
        _reset_caches()
        with pytest.raises(ValueError, match="refusing to downgrade|downgrade"):
            safe_load(str(model))

    def test_downgrade_permitted_only_with_explicit_optout(self, tmp_path, monkeypatch):
        """The downgrade is allowed ONLY behind the explicit migration opt-out.

        This pins that the refusal above is a real gate (not an unconditional
        block): with ``NA0S_ALLOW_SHA256_DOWNGRADE=1`` the same .sha256 verifies,
        so the default-closed behavior is a deliberate policy, not a side effect.
        """
        model = tmp_path / "model.pkl"
        obj = {"weights": [1, 2, 3]}
        safe_dump(obj, str(model))

        monkeypatch.setenv("NA0S_PICKLE_KEY", os.urandom(32).hex())
        monkeypatch.setenv("NA0S_ALLOW_SHA256_DOWNGRADE", "1")
        _reset_caches()
        # Explicit opt-out -> the (untampered) .sha256 verifies and the load works.
        assert safe_load(str(model)) == obj

    def test_forged_sha256_under_downgrade_still_caught_on_tamper(self, tmp_path, monkeypatch):
        """Even under the opt-out, a TAMPERED payload still fails the digest check.

        The downgrade opt-out only relaxes *which tier* is accepted; it never
        disables integrity. A payload swap is still caught by the SHA-256 compare.
        """
        model = tmp_path / "model.pkl"
        safe_dump({"weights": [1, 2, 3]}, str(model))
        monkeypatch.setenv("NA0S_PICKLE_KEY", os.urandom(32).hex())
        monkeypatch.setenv("NA0S_ALLOW_SHA256_DOWNGRADE", "1")
        _reset_caches()
        model.write_bytes(pickle.dumps({"weights": "evil"}))
        _reset_caches()
        with pytest.raises(ValueError, match="Integrity check failed|tampered"):
            safe_load(str(model))


# ===================================================================
# (c) S1.2 / AD2.2 — gadget pickle with a VALID digest (PR #478, xfail)
# ===================================================================

class _Gadget:
    """A reduce-based gadget: ``__reduce__`` makes unpickling call os.system.

    Standing in for the deserialization-RCE primitive. With a valid digest, the
    digest gate (which only authenticates *bytes*, not *opcodes*) passes — only a
    ``find_class`` allowlist (PR #478) can refuse the dangerous global. Kept inert
    here: it would call ``echo`` (harmless) if ever executed, but the strict-xfail
    asserts it is NOT executed once the allowlist lands.
    """

    def __reduce__(self):  # pragma: no cover - exercised only post-#478
        return (os.system, ("echo na0s-gadget-pickle-canary",))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "S1.2/AD2.2: the _SafeUnpickler.find_class allowlist that blocks a "
        "valid-digest gadget pickle is NOT on this branch — it lands in PR #478. "
        "Strict-xfail keeps the find_class gap TRACKED: this flips to a hard "
        "failure (XPASS) the moment the allowlist arrives, forcing the assertion "
        "below to be tightened. Until then, a correctly-signed gadget pickle is "
        "intentionally accepted by the digest-only gate."
    ),
)
def test_gadget_pickle_blocked(tmp_path):
    """S1.2/AD2.2: a digest-VALID gadget pickle must be refused by find_class.

    The digest gate authenticates bytes, so a gadget that is itself the signed
    artifact passes ``verify_file_digest``. Blocking it requires opcode/global
    restriction (``_SafeUnpickler.find_class``), which is PR #478. This test
    therefore EXPECTS to fail on this branch (xfail) and will XPASS->fail once
    the allowlist is merged, at which point this assertion should hold for real.
    """
    model = tmp_path / "gadget.pkl"
    # safe_dump signs whatever bytes it writes, so the gadget gets a VALID sidecar.
    safe_dump(_Gadget(), str(model))
    _reset_caches()
    # On a branch WITH the allowlist, safe_load must refuse the os.system global.
    with pytest.raises(Exception):
        safe_load(str(model))


# ===================================================================
# (d) IM5.5 — ModelRollback.restore rejects a tampered backup (cross-ref S1.7)
# ===================================================================

class TestRollbackRejectsTamperedBackup:
    """Restoring a tampered backup must fail closed (cross-ref S1.7 rollback tests).

    Taxonomy: IM5.5 (Supply-chain model poisoning) — the backup directory is an
    untrusted supply-chain stage; an attacker who writes there could plant a
    poisoned model to be promoted on the next rollback. ``restore`` re-verifies
    the installed target via ``verify_file_digest`` and fails closed.

    These mirror ``TestModelRollbackRestoreVerify`` in
    ``test_l11_encryption_rollback.py`` (the canonical S1.7 suite) but frame the
    same gate as the IM5.5 *supply-chain* use-case for taxonomy coverage. They are
    intentionally NOT a duplicate of the HMAC both-files-replace case there.
    """

    def test_rollback_restore_rejects_tampered_backup(self, tmp_path):
        """IM5.5: a poisoned backup pickle is caught on restore; nothing installed."""
        from na0s.integrity.model_rollback import ModelRollback

        model = tmp_path / "model.pkl"
        safe_dump({"weights": [1, 2, 3]}, str(model))
        rb = ModelRollback(backup_dir=tmp_path / "backups")
        bpath = rb.backup(model)

        # Poison the BACKUP payload; its signed sidecar still holds the clean
        # digest, so re-verify on restore detects the swap.
        bpath.write_bytes(pickle.dumps({"weights": "poisoned"}))
        _reset_caches()

        target = tmp_path / "restored" / "model.pkl"
        with pytest.raises(ValueError, match="integrity verification"):
            rb.restore(bpath, target)
        # Fail closed: the poisoned model is never installed at the live path.
        assert not target.exists()
        assert not (target.parent / (target.name + ".sha256")).exists()

    def test_rollback_restore_accepts_clean_backup(self, tmp_path):
        """IM5.5 baseline: a properly-signed backup round-trips and verifies clean."""
        from na0s.integrity.model_rollback import ModelRollback

        model = tmp_path / "model.pkl"
        safe_dump({"weights": [1, 2, 3]}, str(model))
        rb = ModelRollback(backup_dir=tmp_path / "backups")
        bpath = rb.backup(model)

        target = tmp_path / "restored" / "model.pkl"
        out = rb.restore(bpath, target)
        assert out == target
        assert target.exists()
        assert target.read_bytes() == model.read_bytes()
        # The installed target verifies clean (no raise).
        verify_file_digest(str(target))
