"""Safe pickle wrapper — verifies integrity before unpickling.

A tampered .pkl file can execute arbitrary code via ``pickle.loads``.
This module verifies every file against an integrity digest **before**
it is unpickled.

Trust hierarchy. The tier is chosen by **operator configuration** — whether
``NA0S_PICKLE_KEY`` is set — not by which sidecar file an attacker chose to
drop on disk:

1. **Hardcoded hashes** — ``KNOWN_HASHES`` in ``models/__init__.py``.
   These live inside the Python source, which is signed by pip's wheel
   signature.  An attacker who tampers with a ``.pkl`` cannot update the
   expected hash without also patching the installed Python code.

2. **HMAC-SHA256 sidecar** — ``<file>.hmac`` on disk, keyed by the
   ``NA0S_PICKLE_KEY`` environment variable.  An attacker who replaces
   the pickle cannot forge the HMAC without the secret key.

3. **Plain SHA-256 sidecar** — ``<file>.sha256`` on disk (legacy /
   user-trained models).  It is attacker-recomputable (anyone with write
   access can rewrite both the pickle and this sidecar), so it is only the
   verifiable artifact in **keyless** deployments.

Key-aware selection rules:

* **Key set** → require an HMAC-authenticated sidecar. A plain ``.sha256`` is
  a *downgrade* and **fails closed** by default; set
  ``NA0S_ALLOW_SHA256_DOWNGRADE=1`` to permit it during a SHA→HMAC migration.
* **Keyless** → the ``.sha256`` is the verifiable artifact. A present-but-
  unverifiable ``.hmac`` is ignored and must never veto a valid ``.sha256``
  (avoids a dropped-file denial-of-service). A lone ``.hmac`` with no
  ``.sha256`` fallback is refused (genuinely unverifiable without a key).

All comparisons use ``hmac.compare_digest()`` for constant-time
equality, preventing timing side-channels.
"""

import hashlib
import hmac
import json
import logging
import os
import pickle
import stat
import tempfile

import warnings

from na0s.models import KNOWN_HASHES

_logger = logging.getLogger("na0s.safe_pickle")
_audit = logging.getLogger("na0s.integrity_audit")

# Valid pickle protocol 0 start bytes
_PROTO0_OPCODES = frozenset(b"(]})\x89\x88IX\x80")

# mtime-gated hash cache: avoids re-hashing unchanged files on every load.
# Maps path -> (mtime, digest) for both SHA-256 and HMAC-SHA256.
_sha256_cache: dict = {}  # path -> (mtime, hex_digest)
_hmac_cache: dict = {}    # path -> (mtime, hex_digest)



def _hash_path(pkl_path):
    return pkl_path + ".sha256"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_sha256(path):
    """Return SHA-256 hex digest, using mtime-gated cache."""
    mtime = os.path.getmtime(path)
    cached = _sha256_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    digest = _sha256(path)
    _sha256_cache[path] = (mtime, digest)
    return digest



def _hmac_path(pkl_path):
    """Path for HMAC-SHA256 sidecar."""
    return pkl_path + ".hmac"


def _get_signing_key():
    """Return the HMAC signing key from NA0S_PICKLE_KEY env var, or None."""
    key_str = os.getenv("NA0S_PICKLE_KEY", "")
    return key_str.encode() if key_str else None


def _allow_sha256_downgrade():
    """True if the operator explicitly permits a plain-SHA256 sidecar despite
    a signing key being set.

    Read at call time (not import time) so tests can ``monkeypatch.setenv``
    without re-importing the module. Default is **refuse** (fail closed): when
    ``NA0S_PICKLE_KEY`` is set the operator has opted into forgery-resistant
    HMAC integrity, so silently accepting an attacker-forgeable plain SHA-256
    sidecar would be a downgrade. The opt-out exists only for a migration
    window (a fleet mid-upgrade from SHA-256 to HMAC sidecars).
    """
    return os.getenv("NA0S_ALLOW_SHA256_DOWNGRADE", "0") not in (
        "0", "false", "False", "",
    )


def _hmac_sha256(path, key):
    """Compute HMAC-SHA256 of file at *path* using *key*."""
    h = hmac.new(key, digestmod=hashlib.sha256)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_hmac_sha256(path, key):
    """Return HMAC-SHA256 hex digest, using mtime-gated cache."""
    mtime = os.path.getmtime(path)
    cached = _hmac_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    digest = _hmac_sha256(path, key)
    _hmac_cache[path] = (mtime, digest)
    return digest


def _format_sidecar(algorithm, digest):
    """Format a versioned sidecar string: ``v1:<algorithm>:<digest>``."""
    return "v1:{}:{}".format(algorithm, digest)


def _parse_sidecar(raw):
    """Parse a sidecar value, returning the bare hex digest.

    Accepts both versioned (``v1:algo:digest``) and legacy bare hex formats
    for backward compatibility.
    """
    raw = raw.strip()
    if raw.startswith("v1:"):
        parts = raw.split(":", 2)
        if len(parts) == 3:
            return parts[2]
    # Legacy bare hex digest
    return raw


def _parse_sidecar_typed(raw):
    """Parse a sidecar value, returning ``(algorithm_or_None, hex_digest)``.

    Versioned sidecars (``v1:<algo>:<digest>``) yield their declared algorithm
    tag so the caller can cross-check it against the sidecar's filename
    extension (defends against an attacker smuggling, e.g., an HMAC payload in
    a ``.sha256``-named file). Legacy bare-hex sidecars have no tag, so the
    algorithm is ``None`` and the caller falls back to trusting the extension
    (backward compatibility — see ``test_l11_safe_pickle_fixes.py``).
    """
    raw = raw.strip()
    if raw.startswith("v1:"):
        parts = raw.split(":", 2)
        if len(parts) == 3:
            return parts[1], parts[2]
    # Legacy bare hex digest — no declared algorithm.
    return None, raw


def _validate_pickle_magic(path):
    """Validate that *path* starts with valid pickle opcodes.

    Raises ``ValueError`` if the file does not look like a valid pickle.
    Must be called BEFORE computing any hash (fail fast).
    """
    with open(path, "rb") as f:
        header = f.read(2)

    if len(header) < 2:
        raise ValueError(
            "Invalid pickle format: file too short ({} bytes)".format(len(header))
        )

    # Protocol 2+ starts with PROTO opcode (0x80) followed by version 2-5
    if header[0] == 0x80:
        if header[1] not in (2, 3, 4, 5):
            raise ValueError(
                "Invalid pickle format: PROTO opcode with unsupported "
                "version {}".format(header[1])
            )
        return

    # Protocol 0/1: first byte should be a valid opcode
    if header[0] in _PROTO0_OPCODES:
        return

    raise ValueError(
        "Invalid pickle format: unexpected first byte 0x{:02x}".format(header[0])
    )


def _atomic_write_binary(path, data):
    """Write *data* to *path* atomically via temp-file + os.replace().

    ``os.write`` wraps the POSIX ``write(2)`` syscall, which is **not**
    guaranteed to consume the whole buffer in one call: a single transfer is
    capped at ``INT_MAX`` (2 GiB - 1) and may also short-write under memory
    pressure. The previous implementation issued one ``os.write`` and ignored
    its return value, so any object whose pickle exceeded ~2 GiB (e.g. the
    400k x 15029 sparse ``features.pkl``) was written truncated. The integrity
    sidecar was then computed over the truncated file, so corruption only
    surfaced later as ``pickle data was truncated`` in ``safe_load``.

    We therefore loop until every byte is written, then ``fsync`` before
    closing so the sidecar hash (computed afterwards on the same bytes) and the
    on-disk file are guaranteed identical and durable.
    """
    dir_name = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    closed = False
    try:
        mv = memoryview(data)
        total = len(mv)
        written = 0
        while written < total:
            n = os.write(fd, mv[written:])
            if n == 0:  # pragma: no cover - should not happen for a regular file
                raise OSError(
                    "os.write wrote 0 bytes ({} of {} written) for {}".format(
                        written, total, tmp
                    )
                )
            written += n
        os.fsync(fd)
        os.close(fd)
        closed = True
        os.replace(tmp, path)
    except BaseException:
        if not closed:
            os.close(fd)
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_text(path, text):
    """Write *text* to *path* atomically via temp-file + os.replace()."""
    _atomic_write_binary(path, text.encode("utf-8"))


def _check_permissions(path, label="file"):
    """Warn via audit log if *path* has overly permissive Unix permissions."""
    if os.name != "posix":
        return
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return
    if mode & stat.S_IROTH:  # world-readable (0o004)
        _audit.warning(
            json.dumps({
                "event": "permission_warning",
                "path": path,
                "label": label,
                "issue": "world-readable",
                "mode": oct(mode),
            })
        )
    if mode & stat.S_IWGRP:  # group-writable (0o020)
        _audit.warning(
            json.dumps({
                "event": "permission_warning",
                "path": path,
                "label": label,
                "issue": "group-writable",
                "mode": oct(mode),
            })
        )



# Expected ``v1:<algo>:`` tag for each sidecar extension. A typed sidecar
# whose declared algorithm contradicts its filename extension is rejected:
# selection keys off the extension (it drives which verifier runs), so a
# mismatching tag is an attempt to smuggle the wrong algorithm.
_SIDECAR_EXPECTED_ALGO = {
    "sidecar_hmac": "hmac-sha256",
    "sidecar_sha256": "sha256",
}


def _read_sidecar(sidecar_path, source):
    """Read *sidecar_path*, returning the bare hex digest for *source*.

    Cross-checks any declared ``v1:<algo>:`` tag against the algorithm implied
    by *source* (the filename extension). A mismatch raises ``ValueError``.
    Legacy bare-hex sidecars (no tag) are accepted and trust the extension.
    """
    with open(sidecar_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    declared_algo, digest = _parse_sidecar_typed(raw)
    expected_algo = _SIDECAR_EXPECTED_ALGO[source]
    if declared_algo is not None and declared_algo != expected_algo:
        raise ValueError(
            "Sidecar algorithm tag does not match its filename for {}: "
            "file declares 'v1:{}:' but the {} extension expects '{}'. "
            "Refusing to verify a smuggled algorithm.".format(
                sidecar_path, declared_algo, os.path.splitext(sidecar_path)[1],
                expected_algo,
            )
        )
    return digest


def _resolve_expected_hash(path):
    """Return ``(expected_hex_digest, source)`` for *path*.

    *source* is ``"hardcoded"``, ``"sidecar_hmac"``, or ``"sidecar_sha256"``.

    Selection is **key-aware**: the trust tier is chosen by operator
    configuration (whether ``NA0S_PICKLE_KEY`` is set), NOT by which sidecar
    file happens to be present on disk. This closes two flaws:

    * **Downgrade (key set, only ``.sha256``)** — a plain SHA-256 sidecar is
      attacker-forgeable, so when a key is set it is refused by default (fail
      closed). ``NA0S_ALLOW_SHA256_DOWNGRADE=1`` re-enables it for a migration
      window. The ``.hmac`` sidecar is still preferred when present.
    * **Keyless ``.hmac`` DoS** — in keyless mode a present-but-unverifiable
      ``.hmac`` must NOT veto a present, verifiable ``.sha256``; the ``.sha256``
      wins and is verified. Only when no ``.sha256`` fallback exists does a
      lone ``.hmac`` raise (it is genuinely unverifiable without a key).

    Raises ``FileNotFoundError`` when no source is available, or ``ValueError``
    for a refused downgrade / an algorithm-tag mismatch.
    """
    basename = os.path.basename(path)
    if basename in KNOWN_HASHES:
        return KNOWN_HASHES[basename], "hardcoded"

    key = _get_signing_key()
    hmac_file = _hmac_path(path)
    hash_file = _hash_path(path)
    has_hmac = os.path.exists(hmac_file)
    has_sha256 = os.path.exists(hash_file)

    if key:
        # Operator opted into HMAC. Prefer the .hmac sidecar (strongest).
        if has_hmac:
            return _read_sidecar(hmac_file, "sidecar_hmac"), "sidecar_hmac"
        # Only a plain SHA-256 sidecar: this is a DOWNGRADE from the HMAC tier
        # the operator configured. Refuse by default (fail closed); the SHA is
        # attacker-recomputable so accepting it silently defeats the key.
        if has_sha256:
            if _allow_sha256_downgrade():
                return _read_sidecar(hash_file, "sidecar_sha256"), "sidecar_sha256"
            raise ValueError(
                "NA0S_PICKLE_KEY is set but only a plain SHA-256 sidecar "
                "exists for {}; refusing to downgrade. Re-run safe_dump() to "
                "write an HMAC sidecar, or set NA0S_ALLOW_SHA256_DOWNGRADE=1 "
                "for a migration window.".format(path)
            )
    else:
        # Keyless. The verifiable artifact is the .sha256. A present .hmac
        # cannot be key-checked, so it must NOT remove our ability to verify a
        # valid .sha256 (Threat B / DoS fix): the .sha256 wins and is verified.
        if has_sha256:
            return _read_sidecar(hash_file, "sidecar_sha256"), "sidecar_sha256"
        # Only a lone .hmac and no .sha256 fallback: genuinely unverifiable
        # without a key. safe_load raises a key-specific error; surface a
        # FileNotFoundError-style "no fallback" hint via that path by returning
        # the hmac source (safe_load's no-key guard fires).
        if has_hmac:
            return _read_sidecar(hmac_file, "sidecar_hmac"), "sidecar_hmac"

    raise FileNotFoundError(
        "No integrity hash available for {}.  "
        "Not found in KNOWN_HASHES and sidecar missing: {} and {}.  "
        "Re-run training to generate a sidecar, or add the hash to "
        "models/__init__.py KNOWN_HASHES.".format(path, hmac_file, hash_file)
    )


def write_digest_sidecar(path, _event="write_digest_sidecar", _artifact_label="artifact"):
    """Write an integrity sidecar for the *already-written* file at *path*.

    Format-agnostic: hashes the raw bytes on disk and writes a versioned
    sidecar next to them. It does NOT (re-)serialize the payload, so it works
    for any artifact format — pickle, joblib, or a torch ``.pt``/``.pth`` zip
    (the torch loader path in ``embedding_adapter`` writes the file with
    ``torch.save`` and then calls this to attach a verifiable digest).

    Trust-tier selection mirrors :func:`safe_dump`: an HMAC-SHA256 ``.hmac``
    sidecar when ``NA0S_PICKLE_KEY`` is set, otherwise a plain SHA-256
    ``.sha256`` sidecar with a warning. Written atomically.

    Returns the sidecar path written. Keeping this logic here (rather than
    duplicating it into ``embedding_adapter``) means the ``v1:<algo>:<digest>``
    format and the HMAC-vs-SHA256 decision live in exactly one place.

    ``_event`` / ``_artifact_label`` are internal hooks so :func:`safe_dump`
    can keep emitting its historical ``"safe_dump"`` audit event and
    ``"pickle"`` permission label unchanged.
    """
    key = _get_signing_key()
    if key:
        digest = _hmac_sha256(path, key)
        sidecar_content = _format_sidecar("hmac-sha256", digest)
        sidecar_path = _hmac_path(path)
        _atomic_write_text(sidecar_path, sidecar_content)
        algorithm = "hmac-sha256"
    else:
        warnings.warn(
            "NA0S_PICKLE_KEY is not set. Writing plain SHA-256 sidecar. "
            "Set NA0S_PICKLE_KEY for HMAC-SHA256 signing.",
            UserWarning,
            stacklevel=2,
        )
        digest = _sha256(path)
        sidecar_content = _format_sidecar("sha256", digest)
        sidecar_path = _hash_path(path)
        _atomic_write_text(sidecar_path, sidecar_content)
        algorithm = "sha256"

    _audit.info(
        json.dumps({
            "event": _event,
            "path": path,
            "algorithm": algorithm,
            "digest_prefix": digest[:16],
        })
    )

    # BUG-L11-5: Check file permissions after writing
    _check_permissions(path, label=_artifact_label)
    _check_permissions(sidecar_path, label="sidecar")
    return sidecar_path


def verify_file_digest(path):
    """Verify the integrity digest of *path* **without** deserializing it.

    Format-agnostic counterpart to :func:`write_digest_sidecar`: resolves the
    expected digest via the same key-aware trust hierarchy as :func:`safe_load`
    (KNOWN_HASHES → ``.hmac`` → ``.sha256``) and constant-time-compares it
    against the file's recomputed digest. It does NOT read or unpickle the
    payload, so it is safe to use as a pre-load gate for non-pickle formats
    (e.g. a torch ``.pt``/``.pth`` zip, where ``safe_load`` cannot be used
    because ``_validate_pickle_magic`` would reject the zip header).

    Raises
    ------
    FileNotFoundError
        When no integrity source (KNOWN_HASHES entry or sidecar) is available.
    ValueError
        On a digest mismatch (tampered file), a refused SHA-256 downgrade, or a
        sidecar algorithm-tag/extension mismatch.
    """
    expected, source = _resolve_expected_hash(path)
    key = _get_signing_key()

    if source == "hardcoded":
        actual = _cached_sha256(path)
    elif source == "sidecar_hmac":
        # Mirror safe_load's defensive invariant: key-aware resolution returns
        # "sidecar_hmac" with no key ONLY for the keyless lone-.hmac case (no
        # verifiable .sha256 fallback). That artifact is unverifiable without
        # the signing key, so refuse rather than pretend it verified.
        if not key:
            raise ValueError(
                "HMAC sidecar exists for {} but NA0S_PICKLE_KEY is not set "
                "and no plain SHA-256 (.sha256) fallback is present. "
                "Cannot verify without the signing key.".format(path)
            )
        actual = _cached_hmac_sha256(path, key)
    else:  # sidecar_sha256
        if key:
            # Reached only via the explicit NA0S_ALLOW_SHA256_DOWNGRADE opt-out.
            _audit.warning(
                json.dumps({
                    "event": "verify_file_digest_fallback",
                    "path": path,
                    "issue": "sha256_sidecar_with_key_set",
                })
            )
            _logger.warning(
                "NA0S_PICKLE_KEY is set but %s uses a plain SHA-256 sidecar. "
                "Re-run write_digest_sidecar() to upgrade to HMAC protection.",
                path,
            )
        actual = _cached_sha256(path)

    if not hmac.compare_digest(actual, expected):
        _audit.error(
            json.dumps({
                "event": "integrity_failure",
                "path": path,
                "source": source,
                "expected_prefix": expected[:16],
                "actual_prefix": actual[:16],
            })
        )
        raise ValueError(
            "Integrity check failed for {} (source: {}). "
            "Expected {}, got {}. File may be tampered.".format(
                path, source, expected, actual
            )
        )

    _audit.info(
        json.dumps({
            "event": "verify_file_digest",
            "path": path,
            "source": source,
            "result": "ok",
        })
    )


def safe_dump(obj, path):
    """Pickle *obj* to *path* and write an integrity sidecar.

    Uses HMAC-SHA256 when ``NA0S_PICKLE_KEY`` is set, otherwise falls
    back to plain SHA-256 with a warning.

    Both the pickle file and the sidecar are written atomically via
    temp-file + ``os.replace()`` to prevent corruption on crash.
    """
    # Pin the highest protocol explicitly: protocol >= 4 enables the framed
    # format required to serialise objects larger than 4 GiB, and keeps the
    # written bytes independent of the interpreter's default protocol.
    pkl_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_binary(path, pkl_data)

    # Share the sidecar-write + permission-check logic with the torch path,
    # but preserve the historical "safe_dump" audit event / "pickle" label.
    write_digest_sidecar(path, _event="safe_dump", _artifact_label="pickle")



def safe_load(path):
    """Load a pickle after verifying its integrity digest.

    Trust tier is chosen by ``_resolve_expected_hash`` from operator
    configuration (``NA0S_PICKLE_KEY`` presence), not from which sidecar file
    is on disk:

    1. Hardcoded hash in KNOWN_HASHES (most trusted, plain SHA-256).
    2. HMAC-SHA256 sidecar (selected only when NA0S_PICKLE_KEY is set, except
       the keyless lone-``.hmac`` case which has no verifiable fallback and is
       refused below).
    3. Plain SHA-256 sidecar (keyless legacy path; or, with a key set, only the
       explicit ``NA0S_ALLOW_SHA256_DOWNGRADE=1`` migration opt-out).
    """
    # BUG-L11-6: Validate pickle magic bytes before any hash computation.
    # (Pickle-specific; verify_file_digest is format-agnostic so this guard
    # stays here and is NOT moved into the shared helper.)
    _validate_pickle_magic(path)

    # Shared, format-agnostic digest verification: resolves the trust tier and
    # constant-time-compares, raising ValueError on tamper / FileNotFoundError
    # when no source exists. One implementation for both pickle and torch.
    verify_file_digest(path)

    _audit.info(
        json.dumps({
            "event": "safe_load",
            "path": path,
            "result": "ok",
        })
    )

    # Suppress sklearn's per-load InconsistentVersionWarning. The bundled
    # models were trained on a specific sklearn version (see
    # ``docs/MODEL_PROVENANCE.md``); when the runtime sklearn differs, the
    # warning would fire on every cold start. We swallow only that single
    # warning class — all other warnings still propagate normally — and
    # surface the version mismatch once per process via ``predict.py``.
    with warnings.catch_warnings():
        try:
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.simplefilter("ignore", InconsistentVersionWarning)
        except ImportError:
            # sklearn not installed — nothing to suppress.
            pass
        with open(path, "rb") as f:
            return pickle.load(f)
