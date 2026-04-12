"""Tests for L7 judge features: response caching, token counting, exponential backoff.

All external API calls are mocked -- no network access or API keys required.
"""

import hashlib
import importlib
import json
import os
import sys
import threading
import time
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

# Check groq availability (same pattern as test_llm_judge_hardening.py)
try:
    importlib.import_module("groq")
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# Disable scan timeout for tests
os.environ["SCAN_TIMEOUT_SEC"] = "0"

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if _GROQ_AVAILABLE:
    from na0s.llm_judge import (
        JudgeVerdict,
        LLMJudge,
        MAX_CONTEXT_TOKENS,
        MAX_RETRY_DELAY,
        _safe_error,
    )


def _make_api_response(verdict="SAFE", confidence=0.95, reasoning="Test", nonce=""):
    """Build a mock API response object."""
    resp_json = {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "nonce": nonce,
    }
    msg = MagicMock()
    msg.content = json.dumps(resp_json)
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_judge(cache_size=128, max_context_tokens=None, use_few_shot=True):
    """Create an LLMJudge with a mocked Groq client."""
    with patch("na0s.llm_judge.Groq"):
        judge = LLMJudge(
            backend="groq",
            api_key="test-key",
            cache_size=cache_size,
            max_context_tokens=max_context_tokens,
            use_few_shot=use_few_shot,
        )
    # Replace the client with a fresh mock so we can control side_effect
    judge._client = MagicMock()
    return judge


def _setup_mock_create(judge):
    """Wire up mock to return SAFE verdict with matching nonce."""
    def _mock_create(**kwargs):
        system_msg = kwargs["messages"][0]["content"]
        nonce = ""
        if system_msg.startswith("NONCE: "):
            nonce = system_msg.split("\n")[0].replace("NONCE: ", "")
        return _make_api_response(nonce=nonce)
    judge._client.chat.completions.create.side_effect = _mock_create


# =========================================================================
# Cache tests
# =========================================================================

@unittest.skipUnless(_GROQ_AVAILABLE, "groq package not installed")
class TestResponseCache(unittest.TestCase):
    """Tests for LRU response caching."""

    def setUp(self):
        self.judge = _make_judge(cache_size=4)
        _setup_mock_create(self.judge)

    def test_cache_miss_then_hit(self):
        """First call is a miss, second call with same input is a hit."""
        v1 = self.judge.classify("hello world")
        v2 = self.judge.classify("hello world")
        self.assertEqual(v1.verdict, v2.verdict)
        stats = self.judge.cache_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["size"], 1)
        # API should only be called once
        self.assertEqual(self.judge._client.chat.completions.create.call_count, 1)

    def test_different_inputs_are_separate_entries(self):
        """Different inputs get different cache entries."""
        self.judge.classify("input A")
        self.judge.classify("input B")
        stats = self.judge.cache_stats()
        self.assertEqual(stats["size"], 2)
        self.assertEqual(stats["misses"], 2)

    def test_cache_clear(self):
        """clear_cache() empties the cache and resets counters."""
        self.judge.classify("test input")
        self.judge.classify("test input")  # hit
        self.judge.clear_cache()
        stats = self.judge.cache_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["size"], 0)

    def test_cache_stats_initial(self):
        """Initial cache stats are all zeros."""
        stats = self.judge.cache_stats()
        self.assertEqual(stats, {"hits": 0, "misses": 0, "size": 0})

    def test_lru_eviction(self):
        """When cache exceeds max size, oldest entries are evicted."""
        for i in range(6):  # cache_size=4, so 2 should be evicted
            self.judge.classify("input_{}".format(i))
        stats = self.judge.cache_stats()
        self.assertEqual(stats["size"], 4)

    def test_cache_thread_safety(self):
        """Concurrent access to cache does not corrupt state."""
        errors = []

        def worker(text):
            try:
                for _ in range(10):
                    self.judge.classify(text)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("thread_input_{}".format(i),))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_consistency_bypasses_cache(self):
        """classify_with_consistency should NOT use cache."""
        # First, populate cache via classify
        self.judge.classify("borderline input")
        initial_api_calls = self.judge._client.chat.completions.create.call_count

        # Now call consistency -- it should make fresh API calls
        self.judge.classify_with_consistency("borderline input", n=3)
        new_api_calls = self.judge._client.chat.completions.create.call_count
        # Should have made 3 new calls (not cached)
        self.assertEqual(new_api_calls - initial_api_calls, 3)

    def test_error_verdicts_not_cached(self):
        """Verdicts with errors should not be stored in cache."""
        # Make the API raise an exception
        self.judge._client.chat.completions.create.side_effect = RuntimeError("fail")
        v1 = self.judge.classify("error input")
        self.assertEqual(v1.verdict, "UNKNOWN")
        self.assertIsNotNone(v1.error)
        stats = self.judge.cache_stats()
        self.assertEqual(stats["size"], 0)  # not cached

    def test_cache_key_deterministic(self):
        """Same input always produces the same cache key."""
        k1 = self.judge._cache_key("test")
        k2 = self.judge._cache_key("test")
        self.assertEqual(k1, k2)
        # Different input -> different key
        k3 = self.judge._cache_key("other")
        self.assertNotEqual(k1, k3)


# =========================================================================
# Token counting tests
# =========================================================================

@unittest.skipUnless(_GROQ_AVAILABLE, "groq package not installed")
class TestTokenCounting(unittest.TestCase):
    """Tests for token counting and context truncation."""

    def setUp(self):
        self.judge = _make_judge(max_context_tokens=8000, use_few_shot=False)
        _setup_mock_create(self.judge)

    def test_count_tokens_fallback(self):
        """Without tiktoken, _count_tokens uses len//4 estimate."""
        with patch("na0s.llm_judge.HAS_TIKTOKEN", False):
            count = self.judge._count_tokens("a" * 100)
            self.assertEqual(count, 25)

    def test_count_tokens_with_tiktoken(self):
        """With tiktoken available, uses actual tokenizer."""
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            self.skipTest("tiktoken not installed")

        with patch("na0s.llm_judge.HAS_TIKTOKEN", True):
            count = self.judge._count_tokens("Hello, world!")
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

    def test_short_input_not_truncated(self):
        """Short input should pass through _truncate_for_context unchanged."""
        original = "short input"
        result = self.judge._truncate_for_context(original)
        self.assertEqual(result, original)

    def test_long_input_truncated(self):
        """Input exceeding context limit should be truncated."""
        # Use a very small context limit to force truncation
        judge = _make_judge(max_context_tokens=50, use_few_shot=False)
        long_input = "word " * 1000  # ~5000 chars = ~1250 tokens
        result = judge._truncate_for_context(long_input)
        self.assertLess(len(result), len(long_input))

    def test_truncation_logs_warning(self):
        """Truncation should log a warning."""
        judge = _make_judge(max_context_tokens=50, use_few_shot=False)
        long_input = "word " * 1000
        with patch("na0s.llm_judge.logger") as mock_logger:
            judge._truncate_for_context(long_input)
            mock_logger.warning.assert_called_once()
            args = mock_logger.warning.call_args[0]
            self.assertIn("truncated", args[0].lower())

    def test_estimate_message_tokens(self):
        """_estimate_message_tokens sums content tokens + overhead."""
        messages = [
            {"role": "system", "content": "You are a classifier."},
            {"role": "user", "content": "Hello"},
        ]
        total = self.judge._estimate_message_tokens(messages)
        self.assertGreater(total, 0)
        # Should include per-message overhead (4 tokens each)
        self.assertGreaterEqual(total, 8)  # at least 2 * 4 overhead

    def test_truncation_preserves_some_content(self):
        """Even after truncation, some content remains."""
        judge = _make_judge(max_context_tokens=50, use_few_shot=False)
        long_input = "important " * 500
        result = judge._truncate_for_context(long_input)
        self.assertGreater(len(result), 0)
        self.assertTrue(result.startswith("important"))

    def test_count_tokens_fallback_empty(self):
        """Empty string should return 0 tokens."""
        with patch("na0s.llm_judge.HAS_TIKTOKEN", False):
            count = self.judge._count_tokens("")
            self.assertEqual(count, 0)


# =========================================================================
# Exponential backoff tests
# =========================================================================

class _RateLimitError(Exception):
    """Simulated HTTP 429 error."""
    def __init__(self):
        super().__init__("Rate limit exceeded")
        self.status_code = 429


class _ServiceUnavailableError(Exception):
    """Simulated HTTP 503 error."""
    def __init__(self):
        super().__init__("Service unavailable")
        self.status_code = 503


class _BadRequestError(Exception):
    """Simulated HTTP 400 error (non-retryable)."""
    def __init__(self):
        super().__init__("Bad request")
        self.status_code = 400


@unittest.skipUnless(_GROQ_AVAILABLE, "groq package not installed")
class TestExponentialBackoff(unittest.TestCase):
    """Tests for retry with exponential backoff."""

    def setUp(self):
        self.judge = _make_judge()

    @patch("na0s.llm_judge.time.sleep")
    def test_retry_on_429(self, mock_sleep):
        """Should retry on HTTP 429 and eventually succeed."""
        call_count = [0]

        def _side_effect():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise _RateLimitError()
            return "success"

        result = self.judge._call_with_retry(_side_effect, max_retries=3, base_delay=1.0)
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("na0s.llm_judge.time.sleep")
    def test_retry_on_503(self, mock_sleep):
        """Should retry on HTTP 503 and eventually succeed."""
        call_count = [0]

        def _side_effect():
            call_count[0] += 1
            if call_count[0] <= 1:
                raise _ServiceUnavailableError()
            return "success"

        result = self.judge._call_with_retry(_side_effect, max_retries=3, base_delay=1.0)
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 2)

    @patch("na0s.llm_judge.time.sleep")
    def test_max_retries_exceeded(self, mock_sleep):
        """Should raise after exhausting all retries."""
        def _always_fail():
            raise _RateLimitError()

        with self.assertRaises(_RateLimitError):
            self.judge._call_with_retry(_always_fail, max_retries=2, base_delay=0.1)
        # 1 initial + 2 retries = 3 calls, 2 sleeps
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("na0s.llm_judge.time.sleep")
    def test_non_retryable_error_not_retried(self, mock_sleep):
        """Non-retryable errors (e.g., 400) should raise immediately."""
        def _bad_request():
            raise _BadRequestError()

        with self.assertRaises(_BadRequestError):
            self.judge._call_with_retry(_bad_request, max_retries=3)
        mock_sleep.assert_not_called()

    @patch("na0s.llm_judge.random.uniform", return_value=0.5)
    @patch("na0s.llm_judge.time.sleep")
    def test_exponential_delay_with_jitter(self, mock_sleep, mock_uniform):
        """Delays should follow exponential backoff pattern with jitter."""
        call_count = [0]

        def _side_effect():
            call_count[0] += 1
            if call_count[0] <= 3:
                raise _RateLimitError()
            return "success"

        self.judge._call_with_retry(_side_effect, max_retries=3, base_delay=1.0)
        delays = [c[0][0] for c in mock_sleep.call_args_list]
        # delay = base_delay * 2^attempt + 0.5 (jitter fixed)
        # attempt 0: 1.0 * 1 + 0.5 = 1.5
        # attempt 1: 1.0 * 2 + 0.5 = 2.5
        # attempt 2: 1.0 * 4 + 0.5 = 4.5
        self.assertAlmostEqual(delays[0], 1.5, places=1)
        self.assertAlmostEqual(delays[1], 2.5, places=1)
        self.assertAlmostEqual(delays[2], 4.5, places=1)

    @patch("na0s.llm_judge.time.sleep")
    def test_delay_capped_at_max(self, mock_sleep):
        """Delay should never exceed MAX_RETRY_DELAY."""
        call_count = [0]

        def _side_effect():
            call_count[0] += 1
            if call_count[0] <= 1:
                raise _RateLimitError()
            return "ok"

        # Use a huge base_delay to test capping
        self.judge._call_with_retry(_side_effect, max_retries=3, base_delay=100.0)
        delay = mock_sleep.call_args[0][0]
        self.assertLessEqual(delay, MAX_RETRY_DELAY + 1)  # +1 for jitter

    @patch("na0s.llm_judge.time.sleep")
    def test_retry_logs_warning(self, mock_sleep):
        """Each retry attempt should log a warning."""
        call_count = [0]

        def _side_effect():
            call_count[0] += 1
            if call_count[0] <= 1:
                raise _RateLimitError()
            return "ok"

        with patch("na0s.llm_judge.logger") as mock_logger:
            self.judge._call_with_retry(_side_effect, max_retries=3, base_delay=0.01)
            mock_logger.warning.assert_called()
            args = mock_logger.warning.call_args[0]
            self.assertIn("Retry", args[0])

    @patch("na0s.llm_judge.time.sleep")
    def test_retry_wired_into_classify(self, mock_sleep):
        """classify() should retry on 429 errors."""
        call_count = [0]

        def _mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                raise _RateLimitError()
            system_msg = kwargs["messages"][0]["content"]
            nonce = ""
            if system_msg.startswith("NONCE: "):
                nonce = system_msg.split("\n")[0].replace("NONCE: ", "")
            return _make_api_response(nonce=nonce)

        self.judge._client.chat.completions.create.side_effect = _mock_create
        verdict = self.judge.classify("test input")
        self.assertEqual(verdict.verdict, "SAFE")
        self.assertEqual(call_count[0], 2)

    @patch("na0s.llm_judge.time.sleep")
    def test_status_code_from_response_attr(self, mock_sleep):
        """Should detect status_code from exc.response.status_code."""
        class _ExcWithResponse(Exception):
            def __init__(self):
                super().__init__("rate limited")
                self.response = MagicMock()
                self.response.status_code = 429

        call_count = [0]

        def _side_effect():
            call_count[0] += 1
            if call_count[0] <= 1:
                raise _ExcWithResponse()
            return "ok"

        result = self.judge._call_with_retry(_side_effect, max_retries=2)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count[0], 2)

    def test_no_retry_on_generic_exception(self):
        """Generic exceptions without status_code should not be retried."""
        def _fail():
            raise ValueError("some error")

        with self.assertRaises(ValueError):
            self.judge._call_with_retry(_fail, max_retries=3)


if __name__ == "__main__":
    unittest.main()
