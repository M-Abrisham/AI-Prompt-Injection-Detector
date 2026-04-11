"""Safe pickle wrapper — verifies integrity before unpickling.

A tampered .pkl file can execute arbitrary code via ``pickle.loads``.
This module verifies every file against an integrity digest **before**
it is unpickled.

Trust hierarchy (checked in order):

1. **Hardcoded hashes** — ``KNOWN_HASHES`` in ``models/__init__.py``.
   These live inside the Python source, which is signed by pip's wheel
   signature.  An attacker who tampers with a ``.pkl`` cannot update the
   expected hash without also patching the installed Python code.

2. **HMAC-SHA256 sidecar** — ``<file>.hmac`` on disk, keyed by the
   ``NA0S_PICKLE_KEY`` environment variable.  An attacker who replaces
   the pickle cannot forge the HMAC without the secret key.

3. **Plain SHA-256 sidecar** — ``<file>.sha256`` on disk (legacy /
   user-trained models).  Accepted as a backward-compatible fallback,
   but weaker because an attacker with write access can rewrite the
   sidecar.

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

from .models import KNOWN_HASHES

_logger = logging.getLogger(__name__)
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
    """Write *data* to *path* atomically via temp-file + os.replace()."""
    dir_name = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    closed = False
    try:
        os.write(fd, data)
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



def _resolve_expected_hash(path):
    """Return ``(expected_hex_digest, source)`` for *path*.

    *source* is ``"hardcoded"``, ``"sidecar_hmac"``, or ``"sidecar_sha256"``.
    Raises ``FileNotFoundError`` when no source is available.
    """
    basename = os.path.basename(path)
    if basename in KNOWN_HASHES:
        return KNOWN_HASHES[basename], "hardcoded"

    # Prefer HMAC sidecar over SHA-256 sidecar
    hmac_file = _hmac_path(path)
    if os.path.exists(hmac_file):
        with open(hmac_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return _parse_sidecar(raw), "sidecar_hmac"


    hash_file = _hash_path(path)
    if os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return _parse_sidecar(raw), "sidecar_sha256"


    raise FileNotFoundError(
        "No integrity hash available for {}.  "
        "Not found in KNOWN_HASHES and sidecar missing: {} and {}.  "
        "Re-run training to generate a sidecar, or add the hash to "
        "models/__init__.py KNOWN_HASHES.".format(path, hmac_file, hash_file)
    )


def safe_dump(obj, path):
    """Pickle *obj* to *path* and write an integrity sidecar.

    Uses HMAC-SHA256 when ``NA0S_PICKLE_KEY`` is set, otherwise falls
    back to plain SHA-256 with a warning.

    Both the pickle file and the sidecar are written atomically via
    temp-file + ``os.replace()`` to prevent corruption on crash.
    """
    pkl_data = pickle.dumps(obj)
    _atomic_write_binary(path, pkl_data)

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
            "event": "safe_dump",
            "path": path,
            "algorithm": algorithm,
            "digest_prefix": digest[:16],
        })
    )

    # BUG-L11-5: Check file permissions after writing
    _check_permissions(path, label="pickle")
    _check_permissions(sidecar_path, label="sidecar")



def safe_load(path):
    """Load a pickle after verifying its integrity digest.

    Trust hierarchy:
    1. Hardcoded hash in KNOWN_HASHES (most trusted, plain SHA-256).
    2. HMAC-SHA256 sidecar (trusted when NA0S_PICKLE_KEY is set).
    3. Plain SHA-256 sidecar (legacy, backward compatible).
    """
    # BUG-L11-6: Validate pickle magic bytes before any hash computation
    _validate_pickle_magic(path)

    expected, source = _resolve_expected_hash(path)
    key = _get_signing_key()

    if source == "hardcoded":
        actual = _cached_sha256(path)

    elif source == "sidecar_hmac":
        if not key:
            raise ValueError(
                "HMAC sidecar exists for {} but NA0S_PICKLE_KEY is not set. "
                "Cannot verify without the signing key.".format(path)
            )
        actual = _cached_hmac_sha256(path, key)
    else:  # sidecar_sha256
        if key:
            _audit.warning(
                json.dumps({
                    "event": "safe_load_fallback",
                    "path": path,
                    "issue": "sha256_sidecar_with_key_set",
                })
            )

            _logger.warning(
                "NA0S_PICKLE_KEY is set but %s uses a plain SHA-256 sidecar. "
                "Re-run safe_dump() to upgrade to HMAC protection.", path
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
            "event": "safe_load",
            "path": path,
            "source": source,
            "result": "ok",
        })
    )

    with open(path, "rb") as f:
        return pickle.load(f)
