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
import logging
import os
import pickle
import warnings

from .models import KNOWN_HASHES

_logger = logging.getLogger(__name__)

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
        with open(hmac_file, "r") as f:
            return f.read().strip(), "sidecar_hmac"

    hash_file = _hash_path(path)
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            return f.read().strip(), "sidecar_sha256"

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
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    key = _get_signing_key()
    if key:
        digest = _hmac_sha256(path, key)
        with open(_hmac_path(path), "w") as f:
            f.write(digest)
    else:
        warnings.warn(
            "NA0S_PICKLE_KEY is not set. Writing plain SHA-256 sidecar. "
            "Set NA0S_PICKLE_KEY for HMAC-SHA256 signing.",
            UserWarning,
            stacklevel=2,
        )
        digest = _sha256(path)
        with open(_hash_path(path), "w") as f:
            f.write(digest)


def safe_load(path):
    """Load a pickle after verifying its integrity digest.

    Trust hierarchy:
    1. Hardcoded hash in KNOWN_HASHES (most trusted, plain SHA-256).
    2. HMAC-SHA256 sidecar (trusted when NA0S_PICKLE_KEY is set).
    3. Plain SHA-256 sidecar (legacy, backward compatible).
    """
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
            _logger.warning(
                "NA0S_PICKLE_KEY is set but %s uses a plain SHA-256 sidecar. "
                "Re-run safe_dump() to upgrade to HMAC protection.", path
            )
        actual = _cached_sha256(path)

    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            "Integrity check failed for {} (source: {}). "
            "Expected {}, got {}. File may be tampered.".format(
                path, source, expected, actual
            )
        )
    with open(path, "rb") as f:
        return pickle.load(f)
