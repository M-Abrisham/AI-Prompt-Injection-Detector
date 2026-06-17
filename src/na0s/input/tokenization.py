import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time

# tiktoken is optional — if not installed, tokenization checks degrade
# gracefully (return empty flags) rather than crashing all of Layer 0.
try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    tiktoken = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False

logger = logging.getLogger(__name__)

_ENCODER = None
_ENCODER_LOCK = threading.Lock()


def _get_encoder():
    """Lazy-load the tiktoken encoder on first use.

    tiktoken.get_encoding() downloads the BPE file from the network
    on first call if it's not cached locally.  Doing this at import
    time breaks the package in offline environments (Docker, CI,
    air-gapped servers).  Lazy loading defers the network call to
    first actual use and allows graceful fallback.

    Returns ``None`` when tiktoken is not installed or the encoder
    cannot be obtained (network error, etc.).
    """
    global _ENCODER
    if not _HAS_TIKTOKEN:
        return None
    if _ENCODER is not None:
        return _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is not None:
            return _ENCODER
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning(
                "tiktoken is installed but encoder could not be loaded; "
                "tokenization checks will be skipped"
            )
            return None
    return _ENCODER

# ---------------------------------------------------------------------------
# Configurable thresholds (named constants, env-overridable)
# ---------------------------------------------------------------------------

def _safe_float_env(name, default, lo=0.0, hi=1.0):
    """Read a float from env, clamping to [lo, hi]. Falls back to *default*."""
    import math
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(val):
        return default
    if val < lo or val > hi:
        return default
    return val


def _safe_int_env(name, default, lo=0, hi=None):
    """Read an int from env, clamping to [lo, hi]. Falls back to *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return default
    if val < lo:
        return default
    if hi is not None and val > hi:
        return default
    return val


# --- Ratio thresholds ---
# Start permissive — tighten based on real attacks, not guesses.
# Normal English: ~0.25-0.35 | CJK/emoji: ~0.50-0.60 | Adversarial: ~0.80+
# Override: L0_GLOBAL_RATIO_THRESHOLD (float, 0-1)
GLOBAL_RATIO_THRESHOLD = _safe_float_env(
    "L0_GLOBAL_RATIO_THRESHOLD", 0.75, lo=0.0, hi=1.0
)

# Sliding-window size (in characters) for localized anomaly detection.
# Override: L0_WINDOW_SIZE (int, >= 10)
WINDOW_SIZE = _safe_int_env("L0_WINDOW_SIZE", 50, lo=10)

# Per-window ratio threshold for flagging localized adversarial suffixes.
# Override: L0_WINDOW_RATIO_THRESHOLD (float, 0-1)
WINDOW_RATIO_THRESHOLD = _safe_float_env(
    "L0_WINDOW_RATIO_THRESHOLD", 0.85, lo=0.0, hi=1.0
)

# Fraction of characters that must be CJK/emoji for the text to be
# considered "high-token-script" (and exempt from tokenization_spike).
# Default 0.3 (30%).  Override: L0_CJK_FRACTION_THRESHOLD
_CJK_FRACTION_THRESHOLD = _safe_float_env(
    "L0_CJK_FRACTION_THRESHOLD", 0.3, lo=0.0, hi=1.0
)

# Minimum text length (in characters) for tokenization analysis.
# Texts shorter than this are skipped (too little signal).
# Default 10.  Override: L0_MIN_TEXT_LENGTH_FOR_TOKENIZATION
_MIN_TEXT_LENGTH_FOR_TOKENIZATION = _safe_int_env(
    "L0_MIN_TEXT_LENGTH_FOR_TOKENIZATION", 10, lo=1
)

# CJK/emoji Unicode ranges — these scripts naturally have high token ratios
_CJK_RANGES = (
    (0x2E80, 0x9FFF),   # CJK Radicals through Unified Ideographs
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x1F300, 0x1FAFF), # Emoji
    (0x20000, 0x2FA1F), # CJK Extension B+
)


def _is_high_token_script(text):
    """Check if text is predominantly CJK/emoji — these have inherently
    high token ratios and should not trigger tokenization_spike."""
    if not text:
        return False
    high_count = 0
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                high_count += 1
                break
    return high_count / len(text) > _CJK_FRACTION_THRESHOLD

# Strip everything except lowercase alphanumeric + spaces for normalized hash
_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]")


def _compute_fingerprint(text):
    """Build a multi-layer structural fingerprint.

    Produces four hashes:
        content_hash:    SHA-256 of exact text (exact match)
        normalized_hash: SHA-256 of lowercased, punctuation-stripped text
                         (catches case/whitespace/punctuation mutations)
        token_hash:      SHA-256 of the BPE token-id sequence
                         (catches identical token patterns)
    Plus numeric features: token_count, char_count, ratio.
    """
    encoder = _get_encoder()

    # Normalized form: lowercase, strip punctuation, collapse whitespace
    norm = _NORMALIZE_RE.sub("", text.lower())
    norm = " ".join(norm.split())

    if encoder is None:
        # Offline fallback: word count as rough proxy for token count.
        # Loses token_hash but keeps the system functional.
        token_count = len(text.split())
        return {
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "normalized_hash": hashlib.sha256(norm.encode("utf-8")).hexdigest(),
            "token_hash": "",
            "token_count": token_count,
            "char_count": len(text),
            "ratio": round(token_count / max(len(text), 1), 4),
        }

    tokens = encoder.encode(text, disallowed_special=())
    token_bytes = ",".join(str(t) for t in tokens).encode("utf-8")

    return {
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_hash": hashlib.sha256(norm.encode("utf-8")).hexdigest(),
        "token_hash": hashlib.sha256(token_bytes).hexdigest(),
        "token_count": len(tokens),
        "char_count": len(text),
        "ratio": round(len(tokens) / max(len(text), 1), 4),
    }


class FingerprintStore:
    """Persistent store of known-malicious input fingerprints.

    Uses SQLite with WAL mode for ACID-safe concurrent access.
    Stores three hash columns (content, normalized, token) in a single
    indexed table.  Auto-prunes entries older than TTL_DAYS and enforces
    a MAX_ENTRIES cap using LRU eviction.

    Each entry tracks hit_count and last_seen for monitoring.
    """

    @staticmethod
    def _resolve_default_path():
        """Return a user-writable path for the fingerprint database.

        Resolution order:
        1. ``$L0_FINGERPRINT_STORE`` — explicit override (existing contract)
        2. ``$NA0S_DATA_DIR/fingerprints.db`` — project-level override
        3. Platform user-data directory:
           - Linux/macOS: ``~/.local/share/na0s/fingerprints.db``
           - Windows: ``%LOCALAPPDATA%/na0s/fingerprints.db``

        The old ``__file__``-relative path only worked in editable installs
        and broke after ``pip install`` from a wheel because ``src/data/``
        does not exist inside ``site-packages``.
        """
        env_path = os.getenv("L0_FINGERPRINT_STORE")
        if env_path:
            return env_path

        data_dir = os.getenv("NA0S_DATA_DIR")
        if data_dir:
            return os.path.join(data_dir, "fingerprints.db")

        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        else:
            base = os.environ.get(
                "XDG_DATA_HOME",
                os.path.join(os.path.expanduser("~"), ".local", "share"),
            )
        return os.path.join(base, "na0s", "fingerprints.db")

    TTL_DAYS = 90
    MAX_ENTRIES = 50_000

    def __init__(self, store_path=None):
        self._path = os.path.normpath(
            store_path or self._resolve_default_path()
        )
        self._lock = threading.Lock()
        self._init_db()
        self._migrate_json()
        self._prune()

    def _init_db(self):
        is_memory = (self._path == ":memory:")
        if not is_memory:
            dirname = os.path.dirname(self._path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        if not is_memory:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                content_hash    TEXT PRIMARY KEY,
                normalized_hash TEXT,
                token_hash      TEXT,
                label           TEXT,
                ratio           REAL,
                preview         TEXT,
                hit_count       INTEGER DEFAULT 0,
                first_seen      REAL,
                last_seen       REAL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_normalized
            ON fingerprints(normalized_hash)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token
            ON fingerprints(token_hash)
        """)
        self._conn.commit()

    def _migrate_json(self):
        """One-time migration: import entries from the old JSON store."""
        json_path = self._path.replace(".db", ".json")
        if not os.path.exists(json_path):
            return
        # Only migrate if the DB is empty
        row = self._conn.execute(
            "SELECT COUNT(*) FROM fingerprints"
        ).fetchone()
        if row[0] > 0:
            return
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        content = data.get("content_hashes", {})
        normalized = data.get("normalized_hashes", {})
        token = data.get("token_hashes", {})
        # Build a map: preview → (content_hash, normalized_hash, token_hash, entry)
        # Match entries across the three dicts by preview text
        entries = {}
        for h, e in content.items():
            key = e.get("preview", "")
            entries[key] = {"content_hash": h, **e}
        for h, e in normalized.items():
            key = e.get("preview", "")
            if key in entries:
                entries[key]["normalized_hash"] = h
        for h, e in token.items():
            key = e.get("preview", "")
            if key in entries:
                entries[key]["token_hash"] = h
        for e in entries.values():
            self._conn.execute(
                """INSERT OR IGNORE INTO fingerprints
                   (content_hash, normalized_hash, token_hash,
                    label, ratio, preview, hit_count, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    e.get("content_hash", ""),
                    e.get("normalized_hash", ""),
                    e.get("token_hash", ""),
                    e.get("label", "malicious"),
                    e.get("ratio", 0.0),
                    e.get("preview", ""),
                    e.get("hit_count", 0),
                    e.get("first_seen", 0.0),
                    e.get("last_seen", 0.0),
                ),
            )
        self._conn.commit()

    def _prune(self):
        """Remove stale entries (TTL) and enforce max-size cap (LRU)."""
        with self._lock:
            cutoff = time.time() - (self.TTL_DAYS * 86400)
            self._conn.execute(
                "DELETE FROM fingerprints WHERE last_seen < ?", (cutoff,)
            )
            count = self._conn.execute(
                "SELECT COUNT(*) FROM fingerprints"
            ).fetchone()[0]
            if count > self.MAX_ENTRIES:
                self._conn.execute(
                    """DELETE FROM fingerprints
                       WHERE content_hash NOT IN (
                           SELECT content_hash FROM fingerprints
                           ORDER BY hit_count DESC, last_seen DESC
                           LIMIT ?
                       )""",
                    (self.MAX_ENTRIES,),
                )
            self._conn.commit()

    def register(self, text, label="malicious"):
        """Add a confirmed-malicious input to the store.

        Stores all three hashes so future lookups can match by
        exact text, normalized form, or token sequence.
        Returns the fingerprint dict.
        """
        fp = _compute_fingerprint(text)
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO fingerprints
                   (content_hash, normalized_hash, token_hash,
                    label, ratio, preview, hit_count, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                   ON CONFLICT(content_hash) DO NOTHING""",
                (
                    fp["content_hash"],
                    fp["normalized_hash"],
                    fp["token_hash"],
                    label,
                    fp["ratio"],
                    text[:80],
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return fp

    def check(self, fingerprint):
        """Look up a fingerprint against all three indices.

        Returns a list of match flags. Increments hit_count and
        updates last_seen on every match.
        """
        with self._lock:
            flags = []
            now = time.time()

            checks = [
                ("content_hash", "content_hash", "known_malicious_exact"),
                ("normalized_hash", "normalized_hash", "known_malicious_normalized"),
                ("token_hash", "token_hash", "known_malicious_token_pattern"),
            ]

            _VALID_COLUMNS = {"content_hash", "normalized_hash", "token_hash"}

            for fp_key, col, flag_name in checks:
                assert col in _VALID_COLUMNS, "Invalid column: {}".format(col)
                hash_val = fingerprint.get(fp_key, "")
                if not hash_val:
                    continue
                row = self._conn.execute(
                    "SELECT 1 FROM fingerprints WHERE {} = ? LIMIT 1".format(col),
                    (hash_val,),
                ).fetchone()
                if row:
                    flags.append(flag_name)
                    self._conn.execute(
                        """UPDATE fingerprints
                           SET hit_count = hit_count + 1, last_seen = ?
                           WHERE {} = ?""".format(col),
                        (now, hash_val),
                    )

            if flags:
                self._conn.commit()

            return flags

    def stats(self):
        """Return summary stats for monitoring."""
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(hit_count), 0)
                   FROM fingerprints"""
            ).fetchone()
            return {
                "entries": row[0],
                "total_hits": row[1],
            }


# --- Module-level instance (used by sanitizer) ---
_default_store = None
_default_store_lock = threading.Lock()


def _get_default_store():
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            # Double-checked locking: re-check after acquiring lock
            if _default_store is None:
                _default_store = FingerprintStore()
    return _default_store


def register_malicious(text, label="malicious"):
    """Module-level convenience — registers into the default store."""
    return _get_default_store().register(text, label)


# --- Main detection entry point ---

def check_tokenization_anomaly(text):
    """Fingerprint the input and check for anomalies.

    Four checks:
        1. Known-malicious store lookup (exact + normalized + token)
        2. Global token:char ratio spike
        3. Sliding window for localized adversarial suffixes

    Returns (anomaly_flags, token_char_ratio, fingerprint_dict).
    """
    flags = []

    if len(text) < _MIN_TEXT_LENGTH_FOR_TOKENIZATION:
        return flags, 0.0, {}

    fp = _compute_fingerprint(text)
    store = _get_default_store()

    # Check 1: known-malicious store lookup
    store_flags = store.check(fp)
    flags.extend(store_flags)

    # Checks 2 and 3 require the tokenizer — skip if offline
    encoder = _get_encoder()
    if encoder is None:
        return flags, fp["ratio"], fp

    # Check 2: global ratio (skip for CJK/emoji — they naturally tokenize high)
    if fp["ratio"] >= GLOBAL_RATIO_THRESHOLD and not _is_high_token_script(text):
        flags.append("tokenization_spike")

    # Check 3: sliding window for localized anomalies
    char_count = fp["char_count"]
    if char_count > WINDOW_SIZE and not _is_high_token_script(text):
        for start in range(0, char_count - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = text[start : start + WINDOW_SIZE]
            w_tokens = len(encoder.encode(window, disallowed_special=()))
            w_ratio = w_tokens / len(window)
            if w_ratio >= WINDOW_RATIO_THRESHOLD:
                flags.append("tokenization_spike_local")
                break

    return flags, fp["ratio"], fp
