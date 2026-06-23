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
import re
import stat
import tempfile
import threading

import warnings
from collections import OrderedDict

from na0s.config import INTEGRITY_HASH_CHUNK_BYTES, PICKLE_SIGNING_KEY_ENV
from na0s.models import KNOWN_HASHES

_logger = logging.getLogger("na0s.safe_pickle")
_audit = logging.getLogger("na0s.integrity_audit")

# Valid pickle protocol 0 start bytes
_PROTO0_OPCODES = frozenset(b"(]})\x89\x88IX\x80")

# Canonical shape of an expected integrity digest. SHA-256 and HMAC-SHA256 both
# emit ``hexdigest()`` = exactly 64 lowercase-hex chars (verified empirically:
# ``len(hashlib.sha256(b'x').hexdigest()) == 64`` and the HMAC equivalent). This
# is an invariant of the hash construction, NOT a tunable threshold. Accept
# A-F so externally-generated uppercase digests parse; the parser normalises to
# lowercase on return because ``hmac.compare_digest`` is byte-exact against the
# lowercase ``hexdigest()`` output.
_HEX64_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")

# NA0S_PICKLE_KEY strength policy (bytes of the stripped key):
#   < _MIN_PICKLE_KEY_LEN        -> hard REJECT (ValueError); a key this short
#                                   gives an HMAC-SHA256 secret too little
#                                   entropy to resist offline brute force.
#   [_MIN .. _WEAK_PICKLE_KEY_LEN) -> ACCEPT but WARN (operationally weak).
#   >= _WEAK_PICKLE_KEY_LEN      -> ACCEPT silently (>= the 256-bit HMAC-SHA256
#                                   output/recommended key size).
# 8 bytes ~= 48 bits for a random alphanumeric secret: rejects only degenerate
# 1-7 char / whitespace-only keys while passing every legitimate operator key.
# 32 bytes = 256 bits = the HMAC-SHA256 output (digest) size, the recommended
# key length past which a random key is full-entropy for HMAC-SHA256. (Keys are
# only hashed down above the 64-byte SHA-256 *block* size — a separate bound.)
# Both are named,
# justified constants (na0s-review-checklist: no magic thresholds), not numbers
# chosen to keep tests green.
# TODO(P3): externalize via config.py (ROADMAP_V2.md:1177 — integrity knobs).
_MIN_PICKLE_KEY_LEN = 8
_WEAK_PICKLE_KEY_LEN = 32

# File-identity-gated digest cache: avoids re-hashing unchanged files on every
# load. This is a transparent OPTIMIZATION that sits *after* the trust decision
# (``hmac.compare_digest(actual, expected)`` in ``verify_file_digest``): it only
# memoises a recomputed digest of the *same* file, so it cannot change which hash
# is expected or whether a load is accepted. It is NOT an integrity boundary.
#
# Each cache is a bounded LRU keyed on the path string, with the value
# ``(file_identity, hex_digest)``. ``file_identity`` is ``(st_mtime_ns, st_size,
# st_ino)`` from one ``os.stat`` (see ``_file_identity``) — strictly finer than
# the old ``getmtime()``-seconds gate, so a same-mtime-tick rewrite (different
# size/inode) is a cache MISS and is re-hashed rather than served stale.
#
# ``_CACHE_MAXSIZE`` bounds memory at O(cap), not O(distinct paths seen): the
# real production universe is the ~6 fixed model pkls (four in
# ``predict.py``'s loaders, two in ``predict_embedding.py``), each loaded once;
# the bound only ever engages for pathological batch/rollback/test workloads
# that ``safe_load``/``safe_dump`` over many distinct/temp paths. 64 is ~10x the
# real path count, so the steady-state SDK never evicts. The eviction idiom
# (``OrderedDict`` + ``move_to_end`` + ``popitem(last=False)``) mirrors the
# bounded LRU in ``judge/llm_judge.py`` (``cache_size`` default 128,
# ``popitem(last=False)`` eviction). Mutations are serialised by ``_cache_lock``;
# the (slow, I/O-bound) hash itself is computed OUTSIDE the lock.
_CACHE_MAXSIZE = 64
_sha256_cache: "OrderedDict[str, tuple]" = OrderedDict()  # path -> (file_id, hex)
_hmac_cache: "OrderedDict[str, tuple]" = OrderedDict()    # path -> (file_id, hex)
_cache_lock = threading.Lock()


def _file_identity(path):
    """Cheap change-detection key for the digest cache.

    Returns ``(st_mtime_ns, st_size, st_ino)`` from a single ``os.stat`` — a
    drop-in replacement for the old ``os.path.getmtime`` (itself a ``stat`` under
    the hood), so this is net-zero syscall cost.

    * ``st_mtime_ns`` (nanosecond mtime) is finer than ``getmtime()``'s float
      seconds, catching most rapid-rewrite cases.
    * ``st_size`` catches a same-mtime-tick rewrite that changes length.
    * ``st_ino`` catches an atomic ``os.replace`` (``safe_dump``) that changes
      the inode even when size+mtime coincide. On some Windows filesystems
      ``st_ino`` may be 0; that simply degrades the key to ``(mtime_ns, size)``,
      still strictly better than the old seconds-only gate. We do not branch on
      platform.

    An ``OSError`` from ``os.stat`` (e.g. the file vanished) propagates exactly
    as ``os.path.getmtime`` did before — no new error semantics.
    """
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _cache_get_or_compute(cache, path, compute):
    """Look ``path`` up in *cache*, recomputing via *compute* on a miss.

    The single shared core for ``_cached_sha256`` / ``_cached_hmac_sha256`` so
    the two paths cannot drift. Returns the cached digest on a HIT (and promotes
    it to most-recently-used); on a MISS computes the digest **outside** the
    lock (the hash is the slow, I/O-bound part — holding the lock across 64 KiB-
    chunked reads would serialise every concurrent load), then stores it and
    evicts the least-recently-used entries past ``_CACHE_MAXSIZE``.

    Two threads racing the same cold path may both ``compute()`` once — a benign,
    idempotent double-hash (same value, last write wins), never a correctness
    bug. ``popitem(last=False)`` does not iterate, so the locked mutation cannot
    raise ``dictionary changed size during iteration``.
    """
    file_id = _file_identity(path)
    with _cache_lock:
        cached = cache.get(path)
        if cached is not None and cached[0] == file_id:
            cache.move_to_end(path)  # mark most-recently-used
            return cached[1]
    digest = compute()  # slow I/O — computed OUTSIDE the lock
    with _cache_lock:
        cache[path] = (file_id, digest)
        cache.move_to_end(path)
        while len(cache) > _CACHE_MAXSIZE:
            cache.popitem(last=False)  # evict least-recently-used
    return digest


def _reset_caches():
    """Clear both digest caches under the lock (test seam).

    Underscore-prefixed and not exported; tests call this in ``setUp`` so each
    case starts from a cold, deterministic cache without reaching into the
    ``OrderedDict`` internals byte-by-byte.
    """
    with _cache_lock:
        _sha256_cache.clear()
        _hmac_cache.clear()



def _hash_path(pkl_path):
    return pkl_path + ".sha256"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(INTEGRITY_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_sha256(path):
    """Return the SHA-256 hex digest of *path*, via the bounded LRU cache.

    Signature unchanged; the bound/eviction/richer-key/lock all live behind
    :func:`_cache_get_or_compute`, so the ``safe_load`` call sites are untouched.
    """
    return _cache_get_or_compute(_sha256_cache, path, lambda: _sha256(path))



def _hmac_path(pkl_path):
    """Path for HMAC-SHA256 sidecar."""
    return pkl_path + ".hmac"


def _get_signing_key():
    """Return the HMAC signing key from NA0S_PICKLE_KEY, or None if unset.

    Enforces a key-strength policy at the trust boundary (the key's whole job is
    to be unforgeable, so a low-entropy key silently defeats the HMAC tier):

    * unset                          -> ``None`` (keyless SHA-256 fallback; the
      documented backward-compatible path, never raises).
    * empty / whitespace-only        -> ``ValueError`` (was a 0-after-strip
      "key"; previously ``.encode()``d as bytes of whitespace).
    * ``< _MIN_PICKLE_KEY_LEN``      -> ``ValueError`` (too weak for HMAC).
    * ``[_MIN .. _WEAK_PICKLE_KEY_LEN)`` -> accepted, but ``warnings.warn`` that
      the key is operationally weak.
    * ``>= _WEAK_PICKLE_KEY_LEN``    -> accepted silently.

    The key is **stripped before encoding** so a trailing newline in an env var
    (the common ``export NA0S_PICKLE_KEY=$(cat keyfile)`` footgun) is not part of
    the secret. Length checks are on the stripped value.
    """
    key_str = os.getenv(PICKLE_SIGNING_KEY_ENV)
    if key_str is None:
        return None
    stripped = key_str.strip()
    if not stripped:
        raise ValueError(
            "{} is set but empty/whitespace-only. Unset it to use "
            "the keyless SHA-256 fallback, or set a key of at least {} chars.".format(
                PICKLE_SIGNING_KEY_ENV, _MIN_PICKLE_KEY_LEN
            )
        )
    if len(stripped) < _MIN_PICKLE_KEY_LEN:
        raise ValueError(
            "{} too weak ({} chars); require >= {} chars for "
            "HMAC-SHA256 strength. A short key gives an attacker a tractable "
            "offline brute force against the integrity signature.".format(
                PICKLE_SIGNING_KEY_ENV, len(stripped), _MIN_PICKLE_KEY_LEN
            )
        )
    if len(stripped) < _WEAK_PICKLE_KEY_LEN:
        warnings.warn(
            "{} is {} chars; >= {} chars (a full 256-bit HMAC-SHA256 key) "
            "is recommended for full HMAC-SHA256 key entropy.".format(
                PICKLE_SIGNING_KEY_ENV, len(stripped), _WEAK_PICKLE_KEY_LEN
            ),
            UserWarning,
            stacklevel=2,
        )
    return stripped.encode()


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
        for chunk in iter(lambda: f.read(INTEGRITY_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_hmac_sha256(path, key):
    """Return the HMAC-SHA256 hex digest of *path*, via the bounded LRU cache.

    DOCUMENTED LIMITATION — the cache key is the *path* and the file identity
    only; the signing *key* is captured by the ``compute`` closure and is NOT
    part of the cache key. ``_hmac_cache`` is consulted only on the
    ``sidecar_hmac`` branch, where the process has a single fixed
    ``NA0S_PICKLE_KEY``, so this is safe today. A future multi-key caller MUST
    ``_reset_caches()`` (or extend the key) on a key rotation, otherwise a stale
    digest computed under the previous key would be returned for an unchanged
    file. Pinned by ``test_safe_pickle.py`` so item-7's key-aware selection
    cannot silently introduce a multi-key hazard.
    """
    return _cache_get_or_compute(
        _hmac_cache, path, lambda: _hmac_sha256(path, key)
    )


def _format_sidecar(algorithm, digest):
    """Format a versioned sidecar string: ``v1:<algorithm>:<digest>``."""
    return "v1:{}:{}".format(algorithm, digest)


def _extract_and_validate_digest(raw, where):
    """Extract and shape-validate the digest from a sidecar value.

    Returns ``(declared_algo_or_None, lowercase_hex_digest)``. Accepts versioned
    (``v1:<algo>:<digest>``) and legacy bare-hex formats. Raises ``ValueError``
    (with *where* naming the source) when:

    * a ``v1:`` header has fewer than three colon-parts (e.g. ``"v1:sha256"`` —
      previously this fell through and returned the prefix itself as a
      "digest"), or
    * the extracted candidate is not a 64-char hex SHA-256 / HMAC-SHA256 digest.

    The 64-hex shape is the exact, invariant output length of
    ``hashlib.sha256(...).hexdigest()`` / the HMAC equivalent (see
    ``_HEX64_RE``), so a legitimate expected digest *always* matches — this
    validation has zero false-positive risk for genuine sidecars. Validating
    here makes a corrupt sidecar surface immediately and accurately, instead of
    as a deferred, misleading ``compare_digest`` mismatch ("File may be
    tampered") in :func:`verify_file_digest`.
    """
    raw = raw.strip()
    if raw.startswith("v1:"):
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "malformed integrity sidecar for {}: 'v1:' header without "
                "algo:digest body ({!r})".format(where, raw[:80])
            )
        declared_algo, candidate = parts[1], parts[2]
    else:
        declared_algo, candidate = None, raw  # legacy bare hex digest, no tag
    if not _HEX64_RE.match(candidate):
        raise ValueError(
            "malformed integrity sidecar for {}: expected a 64-char hex "
            "digest, got {!r} (len {})".format(where, candidate[:80], len(candidate))
        )
    return declared_algo, candidate.lower()


def _parse_sidecar(raw):
    """Parse a sidecar value, returning the validated lowercase hex digest.

    Accepts both versioned (``v1:algo:digest``) and legacy bare hex formats for
    backward compatibility. Raises ``ValueError`` if the extracted value is not
    a 64-char hex SHA-256/HMAC-SHA256 digest (including a ``v1:`` header with
    fewer than three colon-parts, which previously fell through and returned the
    ``"v1:algo"`` prefix verbatim as the "digest").
    """
    _algo, digest = _extract_and_validate_digest(raw, "<value>")
    return digest


def _parse_sidecar_typed(raw):
    """Parse a sidecar value, returning ``(algorithm_or_None, hex_digest)``.

    Versioned sidecars (``v1:<algo>:<digest>``) yield their declared algorithm
    tag so the caller can cross-check it against the sidecar's filename
    extension (defends against an attacker smuggling, e.g., an HMAC payload in
    a ``.sha256``-named file). Legacy bare-hex sidecars have no tag, so the
    algorithm is ``None`` and the caller falls back to trusting the extension
    (backward compatibility — see ``test_l11_safe_pickle_fixes.py``).

    The returned digest is shape-validated against the 64-char hex invariant on
    BOTH paths (versioned and legacy bare), raising ``ValueError`` on a
    malformed sidecar so the live load path (:func:`_read_sidecar` ->
    :func:`_resolve_expected_hash`) fails fast at parse instead of as a
    deferred, misleading compare-mismatch.
    """
    return _extract_and_validate_digest(raw, "<value>")


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
        # Bound the read: a legitimate sidecar is "v1:hmac-sha256:" + 64 hex
        # (< 80 bytes). 256 bytes is comfortable headroom while preventing an
        # attacker-supplied multi-megabyte sidecar from being slurped into
        # memory before the shape check rejects it.
        raw = f.read(256).strip()
    # Validate with the concrete sidecar path so a corrupt/truncated sidecar
    # raises "malformed integrity sidecar for <real path>" — naming the file at
    # fault — rather than a generic value or a deferred compare-mismatch.
    declared_algo, digest = _extract_and_validate_digest(raw, sidecar_path)
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
                "{} is set but only a plain SHA-256 sidecar "
                "exists for {}; refusing to downgrade. Re-run safe_dump() to "
                "write an HMAC sidecar, or set NA0S_ALLOW_SHA256_DOWNGRADE=1 "
                "for a migration window.".format(PICKLE_SIGNING_KEY_ENV, path)
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
            "{0} is not set. Writing plain SHA-256 sidecar. "
            "Set {0} for HMAC-SHA256 signing.".format(PICKLE_SIGNING_KEY_ENV),
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
                "HMAC sidecar exists for {} but {} is not set "
                "and no plain SHA-256 (.sha256) fallback is present. "
                "Cannot verify without the signing key.".format(
                    path, PICKLE_SIGNING_KEY_ENV
                )
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
                "%s is set but %s uses a plain SHA-256 sidecar. "
                "Re-run write_digest_sidecar() to upgrade to HMAC protection.",
                PICKLE_SIGNING_KEY_ENV,
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
