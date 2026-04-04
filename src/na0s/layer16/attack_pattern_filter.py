"""Cross-session attack pattern Bloom filter (T3.4).

Privacy-preserving O(1) lookup for previously seen attack patterns.
Uses a Counting Bloom Filter backed by hashlib (not Python's hash()).
No raw text is stored -- only hash-derived counter positions.

Memory: ~48 KB for 10,000 patterns at 1% FPR.
"""

from __future__ import annotations

import hashlib
import math
import threading
from typing import List

from na0s.layer16 import config as layer16_config


class CountingBloomFilter:
    """Counting Bloom filter for attack pattern detection.

    Uses k hash functions over a fixed-size array of counters.
    Supports both insert and approximate count queries.
    ~48 KB memory for 10,000 patterns with 1% FPR.
    """

    # Counter ceiling to prevent overflow in very long-running processes.
    _MAX_COUNTER = 255

    def __init__(self, capacity: int = 10_000, fp_rate: float = 0.01) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0.0 < fp_rate < 1.0):
            raise ValueError("fp_rate must be in (0, 1)")

        self._capacity = capacity
        self._fp_rate = fp_rate

        # Optimal filter size: m = -n * ln(p) / (ln2)^2
        ln2_sq = math.log(2) ** 2
        self._size = max(1, int(-capacity * math.log(fp_rate) / ln2_sq))

        # Optimal hash count: k = (m/n) * ln(2)
        self._num_hashes = max(1, int((self._size / capacity) * math.log(2)))

        # Counter array (list of ints, each bounded by _MAX_COUNTER).
        self._counters: List[int] = [0] * self._size
        self._item_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of counter slots in the filter."""
        return self._size

    @property
    def num_hashes(self) -> int:
        """Number of hash functions used."""
        return self._num_hashes

    @property
    def item_count(self) -> int:
        """Number of items added (approximate -- does not account for dupes)."""
        return self._item_count

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def _hashes(self, item: str) -> List[int]:
        """Generate k hash positions using double-hashing with hashlib.

        Double hashing: h(i) = (h1 + i * h2) % m
        h1 and h2 are derived from SHA-256.
        """
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        # First 8 bytes -> h1, next 8 bytes -> h2
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big")
        # Ensure h2 is odd so it's coprime with m (better distribution)
        if h2 % 2 == 0:
            h2 += 1
        return [(h1 + i * h2) % self._size for i in range(self._num_hashes)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, item: str) -> None:
        """Increment counters for item."""
        for pos in self._hashes(item):
            if self._counters[pos] < self._MAX_COUNTER:
                self._counters[pos] += 1
        self._item_count += 1

    def query(self, item: str) -> int:
        """Return minimum counter value (approximate count)."""
        positions = self._hashes(item)
        return min(self._counters[pos] for pos in positions)

    def contains(self, item: str) -> bool:
        """Check if item was likely added."""
        return self.query(item) > 0

    def remove(self, item: str) -> bool:
        """Decrement counters for item. Returns False if item not present."""
        positions = self._hashes(item)
        if any(self._counters[pos] == 0 for pos in positions):
            return False
        for pos in positions:
            self._counters[pos] -= 1
        self._item_count = max(0, self._item_count - 1)
        return True


class AttackPatternStore:
    """Thread-safe wrapper around a CountingBloomFilter for attack n-grams.

    Records character n-grams from flagged text and checks new text for
    matches against previously seen patterns. No raw text is stored.
    """

    def __init__(
        self,
        capacity: int | None = None,
        fp_rate: float | None = None,
        ngram_size: int | None = None,
    ) -> None:
        cap = capacity if capacity is not None else layer16_config.BLOOM_FILTER_CAPACITY
        fpr = fp_rate if fp_rate is not None else layer16_config.BLOOM_FILTER_FP_RATE
        self._ngram_size = (
            ngram_size if ngram_size is not None
            else layer16_config.PATTERN_RECALL_NGRAM_SIZE
        )
        self._filter = CountingBloomFilter(capacity=cap, fp_rate=fpr)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # N-gram extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ngrams(text: str, n: int = 3) -> List[str]:
        """Extract character n-grams from text.

        Lowercases and strips whitespace runs to normalize.
        """
        if not text or n <= 0:
            return []
        # Normalize: lowercase, collapse whitespace
        normalized = " ".join(text.lower().split())
        if len(normalized) < n:
            return [normalized] if normalized else []
        return [normalized[i : i + n] for i in range(len(normalized) - n + 1)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_attack_ngrams(self, text: str, n: int | None = None) -> int:
        """Extract and record character n-grams from flagged text.

        Returns the number of n-grams added.
        """
        ngram_size = n if n is not None else self._ngram_size
        ngrams = self._extract_ngrams(text, ngram_size)
        with self._lock:
            for ng in ngrams:
                self._filter.add(ng)
        return len(ngrams)

    def check_pattern_match(self, text: str, n: int | None = None) -> float:
        """Return fraction of n-grams that hit in the filter (0.0 to 1.0)."""
        ngram_size = n if n is not None else self._ngram_size
        ngrams = self._extract_ngrams(text, ngram_size)
        if not ngrams:
            return 0.0
        with self._lock:
            hits = sum(1 for ng in ngrams if self._filter.contains(ng))
        return hits / len(ngrams)

    def get_match_score(self, text: str) -> float:
        """Normalized match score for text against known attack patterns.

        Returns a value in [0.0, 1.0] representing how much of the
        text matches previously recorded attack patterns.
        """
        return self.check_pattern_match(text)

    @property
    def filter(self) -> CountingBloomFilter:
        """Access the underlying bloom filter (for testing)."""
        return self._filter
