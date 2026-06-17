"""Dedicated unit tests for src/na0s/layer0/tokenization.py

Covers:
- check_tokenization_anomaly(): normal text, anomalous text, short text,
  CJK exemption
- _compute_fingerprint(): dict keys, deterministic hashes, ratio calculation
- FingerprintStore: register/check round-trip, TTL, LRU eviction, WAL mode
- Concurrent FingerprintStore access from multiple threads
- _get_encoder(): returns encoder or None
- Edge cases: empty text, very long text, unicode, adversarial input
"""

import concurrent.futures
import hashlib
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

# Disable timeout signals before importing na0s modules (thread compat)
os.environ.setdefault("SCAN_TIMEOUT_SEC", "0")

from na0s.layer0.tokenization import (
    FingerprintStore,
    GLOBAL_RATIO_THRESHOLD,
    WINDOW_RATIO_THRESHOLD,
    WINDOW_SIZE,
    _compute_fingerprint,
    _get_encoder,
    _is_high_token_script,
    check_tokenization_anomaly,
)


# ---------------------------------------------------------------------------
# TestGetEncoder
# ---------------------------------------------------------------------------
class TestGetEncoder(unittest.TestCase):
    """Tests for _get_encoder() — lazy-loading tiktoken encoder."""

    def test_returns_encoder_object(self):
        enc = _get_encoder()
        # tiktoken is installed in this env, so encoder should be non-None
        self.assertIsNotNone(enc)

    def test_encoder_can_encode_text(self):
        enc = _get_encoder()
        if enc is None:
            self.skipTest("tiktoken not available")
        tokens = enc.encode("hello world")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_encoder_is_cached(self):
        """Calling _get_encoder() twice returns the same object."""
        enc1 = _get_encoder()
        enc2 = _get_encoder()
        self.assertIs(enc1, enc2)


# ---------------------------------------------------------------------------
# TestComputeFingerprint
# ---------------------------------------------------------------------------
class TestComputeFingerprint(unittest.TestCase):
    """Tests for _compute_fingerprint()."""

    EXPECTED_KEYS = {
        "content_hash",
        "normalized_hash",
        "token_hash",
        "token_count",
        "char_count",
        "ratio",
    }

    def test_all_expected_keys_present(self):
        fp = _compute_fingerprint("This is a normal sentence.")
        self.assertEqual(set(fp.keys()), self.EXPECTED_KEYS)

    def test_content_hash_is_deterministic(self):
        text = "Deterministic hash check 12345"
        fp1 = _compute_fingerprint(text)
        fp2 = _compute_fingerprint(text)
        self.assertEqual(fp1["content_hash"], fp2["content_hash"])

    def test_normalized_hash_is_deterministic(self):
        text = "Deterministic hash check 12345"
        fp1 = _compute_fingerprint(text)
        fp2 = _compute_fingerprint(text)
        self.assertEqual(fp1["normalized_hash"], fp2["normalized_hash"])

    def test_content_hash_matches_sha256(self):
        text = "sha256 verification text"
        fp = _compute_fingerprint(text)
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(fp["content_hash"], expected)

    def test_normalized_hash_case_insensitive(self):
        fp_lower = _compute_fingerprint("hello world")
        fp_upper = _compute_fingerprint("HELLO WORLD")
        self.assertEqual(
            fp_lower["normalized_hash"], fp_upper["normalized_hash"]
        )

    def test_normalized_hash_ignores_punctuation(self):
        fp1 = _compute_fingerprint("hello, world!")
        fp2 = _compute_fingerprint("hello world")
        self.assertEqual(fp1["normalized_hash"], fp2["normalized_hash"])

    def test_char_count_correct(self):
        text = "exactly twenty chars"
        fp = _compute_fingerprint(text)
        self.assertEqual(fp["char_count"], len(text))

    def test_token_count_positive_for_real_text(self):
        fp = _compute_fingerprint("The quick brown fox jumps over the lazy dog.")
        self.assertGreater(fp["token_count"], 0)

    def test_ratio_calculation(self):
        """ratio should equal token_count / char_count (rounded)."""
        fp = _compute_fingerprint("A moderately long piece of text for ratio calc.")
        expected_ratio = round(fp["token_count"] / max(fp["char_count"], 1), 4)
        self.assertAlmostEqual(fp["ratio"], expected_ratio, places=4)

    def test_token_hash_nonempty_with_tiktoken(self):
        enc = _get_encoder()
        if enc is None:
            self.skipTest("tiktoken not available")
        fp = _compute_fingerprint("Token hash check")
        self.assertGreater(len(fp["token_hash"]), 0)

    def test_different_texts_produce_different_content_hashes(self):
        fp1 = _compute_fingerprint("text one")
        fp2 = _compute_fingerprint("text two")
        self.assertNotEqual(fp1["content_hash"], fp2["content_hash"])

    def test_single_char_text(self):
        fp = _compute_fingerprint("a")
        self.assertEqual(fp["char_count"], 1)
        self.assertGreater(fp["token_count"], 0)

    def test_unicode_text(self):
        fp = _compute_fingerprint("Caf\u00e9 na\u00efvet\u00e9 r\u00e9sum\u00e9")
        self.assertEqual(set(fp.keys()), self.EXPECTED_KEYS)
        self.assertGreater(fp["char_count"], 0)


# ---------------------------------------------------------------------------
# TestIsHighTokenScript
# ---------------------------------------------------------------------------
class TestIsHighTokenScript(unittest.TestCase):
    """Tests for the CJK/emoji detection helper."""

    def test_english_text_is_not_high_token(self):
        self.assertFalse(_is_high_token_script("Hello world, how are you?"))

    def test_cjk_text_is_high_token(self):
        # Chinese characters
        self.assertTrue(_is_high_token_script("\u4f60\u597d\u4e16\u754c\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5"))

    def test_korean_text_is_high_token(self):
        # Hangul
        self.assertTrue(_is_high_token_script("\uc548\ub155\ud558\uc138\uc694\uc138\uacc4"))

    def test_emoji_heavy_text_is_high_token(self):
        self.assertTrue(_is_high_token_script("\U0001F600\U0001F601\U0001F602\U0001F603\U0001F604"))

    def test_empty_text_returns_false(self):
        self.assertFalse(_is_high_token_script(""))

    def test_mixed_text_below_threshold(self):
        # Mostly English with a few CJK chars — should be below 30%
        text = "This is a mostly English sentence with one \u4f60 char."
        self.assertFalse(_is_high_token_script(text))


# ---------------------------------------------------------------------------
# TestCheckTokenizationAnomaly
# ---------------------------------------------------------------------------
class TestCheckTokenizationAnomaly(unittest.TestCase):
    """Tests for check_tokenization_anomaly()."""

    def test_normal_text_no_flags(self):
        """Normal English text should not trigger any flags."""
        text = "The quick brown fox jumps over the lazy dog. This is a normal sentence."
        flags, ratio, fp = check_tokenization_anomaly(text)
        self.assertNotIn("tokenization_spike", flags)
        self.assertNotIn("tokenization_spike_local", flags)

    def test_short_text_early_return(self):
        """Text shorter than 10 chars returns empty flags, 0.0 ratio, empty fp."""
        flags, ratio, fp = check_tokenization_anomaly("short")
        self.assertEqual(flags, [])
        self.assertEqual(ratio, 0.0)
        self.assertEqual(fp, {})

    def test_exactly_10_chars_not_skipped(self):
        """Text of exactly 10 chars should NOT be skipped."""
        text = "0123456789"
        flags, ratio, fp = check_tokenization_anomaly(text)
        # Should have a non-empty fingerprint
        self.assertIsInstance(fp, dict)
        self.assertIn("content_hash", fp)

    def test_9_chars_is_skipped(self):
        """Text of 9 chars should be skipped."""
        flags, ratio, fp = check_tokenization_anomaly("123456789")
        self.assertEqual(flags, [])
        self.assertEqual(fp, {})

    def test_returns_three_tuple(self):
        """Return type should be (list, float, dict)."""
        result = check_tokenization_anomaly("This is a test sentence for return type.")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        flags, ratio, fp = result
        self.assertIsInstance(flags, list)
        self.assertIsInstance(ratio, float)
        self.assertIsInstance(fp, dict)

    def test_cjk_text_no_spike_flag(self):
        """CJK text should be exempt from tokenization_spike due to naturally high ratio."""
        enc = _get_encoder()
        if enc is None:
            self.skipTest("tiktoken not available")
        # Long CJK text — will have high token ratio but should be exempted
        text = "\u4f60\u597d\u4e16\u754c" * 20  # 80 CJK chars
        flags, ratio, fp = check_tokenization_anomaly(text)
        self.assertNotIn("tokenization_spike", flags)
        self.assertNotIn("tokenization_spike_local", flags)

    def test_ratio_is_float_in_range(self):
        text = "A reasonable piece of text that is at least ten characters long."
        flags, ratio, fp = check_tokenization_anomaly(text)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 2.0)  # generous upper bound


# ---------------------------------------------------------------------------
# TestFingerprintStore
# ---------------------------------------------------------------------------
class TestFingerprintStore(unittest.TestCase):
    """Tests for FingerprintStore — SQLite-based fingerprint storage."""

    def _make_store(self):
        """Create an in-memory store for testing."""
        return FingerprintStore(":memory:")

    def test_register_returns_fingerprint_dict(self):
        store = self._make_store()
        fp = store.register("This is a test malicious input.")
        self.assertIsInstance(fp, dict)
        self.assertIn("content_hash", fp)
        self.assertIn("normalized_hash", fp)

    def test_register_and_check_exact_match(self):
        store = self._make_store()
        text = "known malicious payload for exact match"
        store.register(text)
        fp = _compute_fingerprint(text)
        flags = store.check(fp)
        self.assertIn("known_malicious_exact", flags)

    def test_register_and_check_normalized_match(self):
        store = self._make_store()
        store.register("Known Malicious Payload!")
        # Check with lowercased, no-punctuation variant
        fp = _compute_fingerprint("known malicious payload")
        flags = store.check(fp)
        self.assertIn("known_malicious_normalized", flags)

    def test_check_unregistered_returns_empty(self):
        store = self._make_store()
        fp = _compute_fingerprint("this text was never registered")
        flags = store.check(fp)
        self.assertEqual(flags, [])

    def test_register_idempotent(self):
        """Registering the same text twice does not crash (ON CONFLICT DO NOTHING)."""
        store = self._make_store()
        text = "duplicate registration test"
        store.register(text)
        store.register(text)  # should not raise
        stats = store.stats()
        self.assertEqual(stats["entries"], 1)

    def test_stats_empty_store(self):
        store = self._make_store()
        stats = store.stats()
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["total_hits"], 0)

    def test_stats_after_register(self):
        store = self._make_store()
        store.register("test payload one")
        store.register("test payload two")
        stats = store.stats()
        self.assertEqual(stats["entries"], 2)

    def test_hit_count_increments_on_check(self):
        store = self._make_store()
        text = "hit counter test text for verification"
        store.register(text)
        fp = _compute_fingerprint(text)
        store.check(fp)
        store.check(fp)
        # After 2 checks, hit_count should be incremented
        row = store._conn.execute(
            "SELECT hit_count FROM fingerprints WHERE content_hash = ?",
            (fp["content_hash"],),
        ).fetchone()
        # Each check increments for each matching column (content, normalized, token)
        # So hit_count should be >= 2
        self.assertGreaterEqual(row[0], 2)

    def test_lru_eviction(self):
        """When entries exceed MAX_ENTRIES, oldest (least hit, least recent) are evicted."""
        store = self._make_store()
        # Temporarily lower MAX_ENTRIES
        original_max = FingerprintStore.MAX_ENTRIES
        try:
            FingerprintStore.MAX_ENTRIES = 5
            for i in range(10):
                store.register(f"eviction test payload number {i} with enough chars")
            store._prune()
            stats = store.stats()
            self.assertLessEqual(stats["entries"], 5)
        finally:
            FingerprintStore.MAX_ENTRIES = original_max

    def test_ttl_expiration(self):
        """Entries with last_seen older than TTL_DAYS are pruned."""
        store = self._make_store()
        text = "TTL expiration test payload with enough chars"
        store.register(text)
        # Manually backdate last_seen to exceed TTL
        old_time = time.time() - (FingerprintStore.TTL_DAYS * 86400 + 1)
        store._conn.execute(
            "UPDATE fingerprints SET last_seen = ?", (old_time,)
        )
        store._conn.commit()
        store._prune()
        stats = store.stats()
        self.assertEqual(stats["entries"], 0)

    def test_ttl_non_expired_entries_kept(self):
        """Recent entries are NOT pruned by TTL."""
        store = self._make_store()
        store.register("fresh entry that should survive TTL pruning")
        store._prune()
        stats = store.stats()
        self.assertEqual(stats["entries"], 1)

    def test_wal_mode_for_file_store(self):
        """File-based stores should use WAL journal mode."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_wal.db")
            store = FingerprintStore(db_path)
            row = store._conn.execute("PRAGMA journal_mode").fetchone()
            self.assertEqual(row[0].lower(), "wal")

    def test_memory_store_no_wal(self):
        """In-memory stores skip WAL (not applicable to :memory:)."""
        store = self._make_store()
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        # :memory: defaults to 'memory' journal mode
        self.assertEqual(row[0].lower(), "memory")

    def test_table_schema_has_expected_columns(self):
        store = self._make_store()
        cursor = store._conn.execute("PRAGMA table_info(fingerprints)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "content_hash", "normalized_hash", "token_hash",
            "label", "ratio", "preview", "hit_count",
            "first_seen", "last_seen",
        }
        self.assertTrue(expected.issubset(columns))

    def test_preview_truncated_to_80_chars(self):
        store = self._make_store()
        long_text = "A" * 200
        store.register(long_text)
        fp = _compute_fingerprint(long_text)
        row = store._conn.execute(
            "SELECT preview FROM fingerprints WHERE content_hash = ?",
            (fp["content_hash"],),
        ).fetchone()
        self.assertLessEqual(len(row[0]), 80)

    def test_label_stored_correctly(self):
        store = self._make_store()
        store.register("labeled payload test for custom label", label="suspicious")
        row = store._conn.execute(
            "SELECT label FROM fingerprints LIMIT 1"
        ).fetchone()
        self.assertEqual(row[0], "suspicious")


# ---------------------------------------------------------------------------
# TestConcurrentFingerprintStore
# ---------------------------------------------------------------------------
class TestConcurrentFingerprintStore(unittest.TestCase):
    """Stress tests for concurrent access to FingerprintStore from multiple threads.

    Uses file-based stores (with WAL mode) and per-thread FingerprintStore
    instances, which is the realistic production concurrency pattern.
    Each thread opens its own connection to the same database file;
    SQLite WAL mode allows concurrent readers and serialized writers.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "concurrent_test.db")
        # Initialize the DB schema by creating one store
        FingerprintStore(self._db_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_concurrent_register(self):
        """10 threads each register 10 entries — no crashes, no lost writes."""
        errors = []

        def worker(thread_id):
            try:
                store = FingerprintStore(self._db_path)
                for i in range(10):
                    text = f"concurrent register thread {thread_id} item {i} padding text"
                    store.register(text)
            except Exception as e:
                errors.append((thread_id, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, t) for t in range(10)]
            concurrent.futures.wait(futures, timeout=60)

        self.assertEqual(len(errors), 0, f"Concurrent register errors: {errors}")
        verify_store = FingerprintStore(self._db_path)
        stats = verify_store.stats()
        self.assertEqual(stats["entries"], 100)

    def test_concurrent_check(self):
        """Pre-populate store, then 10 threads check simultaneously."""
        setup_store = FingerprintStore(self._db_path)
        texts = [f"pre-populated entry number {i} for concurrent check" for i in range(20)]
        for t in texts:
            setup_store.register(t)

        errors = []

        def worker(thread_id):
            try:
                store = FingerprintStore(self._db_path)
                for t in texts:
                    fp = _compute_fingerprint(t)
                    flags = store.check(fp)
                    if "known_malicious_exact" not in flags:
                        errors.append(
                            (thread_id, f"Missing exact match for: {t[:40]}")
                        )
            except Exception as e:
                errors.append((thread_id, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, t) for t in range(10)]
            concurrent.futures.wait(futures, timeout=60)

        self.assertEqual(len(errors), 0, f"Concurrent check errors: {errors}")

    def test_concurrent_mixed_register_and_check(self):
        """Half threads register, half check — no crashes or data corruption."""
        setup_store = FingerprintStore(self._db_path)
        seed_texts = [f"seed entry {i} with enough padding text" for i in range(10)]
        for t in seed_texts:
            setup_store.register(t)

        errors = []

        def register_worker(thread_id):
            try:
                store = FingerprintStore(self._db_path)
                for i in range(20):
                    text = f"new entry thread {thread_id} item {i} long enough text here"
                    store.register(text)
            except Exception as e:
                errors.append(("register", thread_id, e))

        def check_worker(thread_id):
            try:
                store = FingerprintStore(self._db_path)
                for t in seed_texts:
                    fp = _compute_fingerprint(t)
                    store.check(fp)
            except Exception as e:
                errors.append(("check", thread_id, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            for t in range(6):
                futures.append(executor.submit(register_worker, t))
            for t in range(6):
                futures.append(executor.submit(check_worker, t))
            concurrent.futures.wait(futures, timeout=60)

        self.assertEqual(len(errors), 0, f"Concurrent mixed errors: {errors}")

    def test_concurrent_access_100_operations(self):
        """100 operations from 10 threads — the reference stress test."""
        errors = []

        def worker(i):
            try:
                store = FingerprintStore(self._db_path)
                text = f"test input {i} " * 20
                store.register(text)
                fp = _compute_fingerprint(text)
                result = store.check(fp)
                if "known_malicious_exact" not in result:
                    errors.append((i, "Missing exact match after register"))
            except Exception as e:
                errors.append((i, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(100)]
            concurrent.futures.wait(futures, timeout=60)

        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases(unittest.TestCase):
    """Edge case coverage for tokenization functions."""

    def test_empty_text_fingerprint(self):
        fp = _compute_fingerprint("")
        self.assertEqual(fp["char_count"], 0)
        self.assertEqual(fp["ratio"], 0.0)

    def test_empty_text_anomaly_check(self):
        """Empty text is < 10 chars, should early-return."""
        flags, ratio, fp = check_tokenization_anomaly("")
        self.assertEqual(flags, [])
        self.assertEqual(ratio, 0.0)
        self.assertEqual(fp, {})

    def test_whitespace_only_text(self):
        fp = _compute_fingerprint("          ")  # 10 spaces
        self.assertEqual(fp["char_count"], 10)

    def test_very_long_text(self):
        """100K character text should not crash or timeout."""
        text = "The quick brown fox. " * 5000
        fp = _compute_fingerprint(text)
        self.assertGreater(fp["token_count"], 0)
        self.assertGreater(fp["char_count"], 50000)

    def test_unicode_emoji_text(self):
        text = "\U0001F600\U0001F601\U0001F602\U0001F603\U0001F604\U0001F605\U0001F606\U0001F607\U0001F608\U0001F609\U0001F60A"
        fp = _compute_fingerprint(text)
        self.assertEqual(fp["char_count"], 11)

    def test_mixed_script_text(self):
        """Text mixing Latin, CJK, Arabic, and emoji."""
        text = "Hello \u4f60\u597d \u0645\u0631\u062d\u0628\u0627 \U0001F44B this is mixed"
        fp = _compute_fingerprint(text)
        self.assertIn("content_hash", fp)
        self.assertGreater(fp["char_count"], 0)

    def test_newlines_and_tabs(self):
        text = "line1\nline2\tcolumn\r\nline3" + " pad" * 5
        fp = _compute_fingerprint(text)
        self.assertGreater(fp["token_count"], 0)

    def test_null_bytes_in_text(self):
        text = "some text\x00with null\x00bytes padded out"
        fp = _compute_fingerprint(text)
        self.assertIn("content_hash", fp)

    def test_repetitive_adversarial_pattern(self):
        """Highly repetitive text used in adversarial suffixes."""
        text = "AAAA" * 100  # 400 chars of same character
        fp = _compute_fingerprint(text)
        self.assertIn("content_hash", fp)
        # Ratio should still be calculable
        self.assertGreater(fp["ratio"], 0.0)

    def test_store_register_empty_text(self):
        """Register empty string — should not crash."""
        store = FingerprintStore(":memory:")
        fp = store.register("")
        self.assertIsInstance(fp, dict)

    def test_check_with_empty_hashes(self):
        """Check with a fingerprint containing empty hash values."""
        store = FingerprintStore(":memory:")
        empty_fp = {
            "content_hash": "",
            "normalized_hash": "",
            "token_hash": "",
        }
        flags = store.check(empty_fp)
        self.assertEqual(flags, [])


if __name__ == "__main__":
    unittest.main()
