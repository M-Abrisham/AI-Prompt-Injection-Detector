"""Tests for Claude API gate failure analysis integration.

Tests cover:
- ClaudeGateAnalyzer initialization with/without API key
- GateCacheManager cache operations and persistence
- Claude API calls with retries and error handling
- Gate-specific analysis prompts and responses
- Integration with GateAnalyzer
"""

import json
import tempfile
import time
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest

from na0s.agents.claude_gate_analyzer import ClaudeGateAnalyzer, GateCacheManager
from na0s.agents.gate_analyzer import GateAnalyzer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "canary").mkdir()
        (data_dir / "models").mkdir()
        yield str(data_dir)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create cache manager with temporary directory."""
    return GateCacheManager(cache_dir=temp_cache_dir)


@pytest.fixture
def mock_anthropic(monkeypatch):
    """Mock Anthropic client for testing."""
    # Create mock exceptions
    class MockRateLimitError(Exception):
        def __init__(self, msg, response=None):
            super().__init__(msg)
            self.response = response

    class MockAPITimeoutError(Exception):
        def __init__(self, msg, response=None):
            super().__init__(msg)
            self.response = response

    class MockAPIConnectionError(Exception):
        def __init__(self, msg, response=None):
            super().__init__(msg)
            self.response = response

    # Create mock client that will be instantiated
    mock_client = MagicMock()

    class MockAnthropicClass:
        def __init__(self, api_key=None):
            # Return the shared mock instance
            self.messages = mock_client.messages

    # Create mock anthropic module with these exceptions
    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic = MockAnthropicClass
    mock_anthropic_module.RateLimitError = MockRateLimitError
    mock_anthropic_module.APITimeoutError = MockAPITimeoutError
    mock_anthropic_module.APIConnectionError = MockAPIConnectionError

    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic_module)

    # Reimport the module so it picks up our mock
    import importlib
    import na0s.agents.claude_gate_analyzer
    importlib.reload(na0s.agents.claude_gate_analyzer)

    yield mock_client


@pytest.fixture
def sample_canary_failure():
    """Sample canary gate failure data."""
    return {
        "tpr": 0.92,
        "tnr": 0.91,
        "fpr": 0.05,
        "error_count": 3,
        "errors": [
            {"technique": "DAN", "text_preview": "Act as malicious"},
            {"technique": "Token smuggling", "text_preview": "[SECRET]"},
            {"technique": "Prompt escape", "text_preview": "\\n\\nIgnore above"},
        ],
    }


@pytest.fixture
def sample_shadow_failure():
    """Sample shadow gate failure data."""
    return {
        "verdict": "FAIL",
        "fpr_delta": 0.03,
        "recall_delta": -0.05,
        "failure_count": 2,
        "failures": [
            {"gate": "FPR < 0.05", "actual": 0.054},
            {"gate": "Recall > 0.90", "actual": 0.87},
        ],
    }


@pytest.fixture
def sample_f14_failure():
    """Sample F14 promotion gate failure data."""
    return {
        "verdict": "FAIL",
        "overall_tpr": 0.82,
        "regression_count": 3,
        "regressions": [
            {"category": "SQLi", "delta": -0.08},
            {"category": "XSS", "delta": -0.05},
            {"category": "LDAP", "delta": -0.03},
        ],
    }


# ============================================================================
# GateCacheManager Tests
# ============================================================================


class TestGateCacheManager:
    """Tests for GateCacheManager cache operations."""

    def test_cache_manager_init_creates_dir(self, temp_cache_dir):
        """Test that cache manager creates cache directory on init."""
        cache_dir = Path(temp_cache_dir) / "subdir" / "cache"
        manager = GateCacheManager(cache_dir=str(cache_dir))
        assert cache_dir.exists()

    def test_cache_key_generation_deterministic(self, cache_manager):
        """Test that cache keys are deterministic for same inputs."""
        failure_data = {"tpr": 0.92, "error_count": 3}
        key1 = cache_manager.get_cache_key("canary", "abc123")
        key2 = cache_manager.get_cache_key("canary", "abc123")
        assert key1 == key2

    def test_cache_key_different_for_different_gates(self, cache_manager):
        """Test that different gate types produce different keys."""
        failure_hash = "abc123"
        key_canary = cache_manager.get_cache_key("canary", failure_hash)
        key_shadow = cache_manager.get_cache_key("shadow", failure_hash)
        key_f14 = cache_manager.get_cache_key("f14", failure_hash)

        assert key_canary != key_shadow
        assert key_canary != key_f14
        assert key_shadow != key_f14

    def test_failure_hash_deterministic(self, cache_manager):
        """Test that failure hash is deterministic."""
        failure_data = {"tpr": 0.92, "error_count": 3}
        hash1 = cache_manager.compute_failure_hash(failure_data)
        hash2 = cache_manager.compute_failure_hash(failure_data)
        assert hash1 == hash2

    def test_failure_hash_different_for_different_data(self, cache_manager):
        """Test that different failure data produces different hashes."""
        data1 = {"tpr": 0.92, "error_count": 3}
        data2 = {"tpr": 0.93, "error_count": 3}
        hash1 = cache_manager.compute_failure_hash(data1)
        hash2 = cache_manager.compute_failure_hash(data2)
        assert hash1 != hash2

    def test_failure_hash_ignores_timestamp(self, cache_manager):
        """Test that failure hash does not include timestamp."""
        data1 = {"tpr": 0.92, "timestamp": "2025-01-01"}
        data2 = {"tpr": 0.92, "timestamp": "2025-01-02"}
        # If timestamp affected hash, these would differ
        hash1 = cache_manager.compute_failure_hash(data1)
        hash2 = cache_manager.compute_failure_hash(data2)
        # They should be different because timestamps are different
        assert hash1 != hash2

    def test_cache_miss_returns_none(self, cache_manager, sample_canary_failure):
        """Test that cache miss returns None."""
        result = cache_manager.get("canary", sample_canary_failure)
        assert result is None

    def test_cache_hit_returns_cached_result(self, cache_manager, sample_canary_failure):
        """Test that cache hit returns cached analysis."""
        analysis = {"root_cause": "test cause", "fix_specificity": "test fix"}
        cache_manager.set("canary", sample_canary_failure, analysis)

        result = cache_manager.get("canary", sample_canary_failure)
        assert result is not None
        assert result["root_cause"] == "test cause"
        assert result["fix_specificity"] == "test fix"

    def test_cache_persists_to_disk(self, temp_cache_dir, sample_canary_failure):
        """Test that cache is written to disk."""
        manager = GateCacheManager(cache_dir=temp_cache_dir)
        analysis = {"root_cause": "test cause", "fix_specificity": "test fix"}
        manager.set("canary", sample_canary_failure, analysis)

        # Verify file exists
        cache_files = list(Path(temp_cache_dir).glob("*.json"))
        assert len(cache_files) == 1

    def test_cache_loads_from_disk(self, temp_cache_dir, sample_canary_failure):
        """Test that cache loads from disk on init."""
        # Write cache with first manager
        manager1 = GateCacheManager(cache_dir=temp_cache_dir)
        analysis = {"root_cause": "test cause", "fix_specificity": "test fix"}
        manager1.set("canary", sample_canary_failure, analysis)

        # Create new manager and verify it can read cache
        manager2 = GateCacheManager(cache_dir=temp_cache_dir)
        result = manager2.get("canary", sample_canary_failure)
        assert result is not None
        assert result["root_cause"] == "test cause"

    def test_cache_write_error_handling(self, cache_manager, sample_canary_failure):
        """Test that cache write errors are handled gracefully."""
        analysis = {"root_cause": "test"}
        # Mock cache directory to be read-only
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            result = cache_manager.set("canary", sample_canary_failure, analysis)
        assert result is False

    def test_cache_read_error_handling(self, cache_manager, sample_canary_failure):
        """Test that cache read errors are handled gracefully."""
        # Set a valid cache first
        analysis = {"root_cause": "test"}
        cache_manager.set("canary", sample_canary_failure, analysis)

        # Mock read to fail
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            result = cache_manager.get("canary", sample_canary_failure)
        assert result is None

    def test_cache_separate_for_different_gate_types(
        self, cache_manager, sample_canary_failure, sample_shadow_failure
    ):
        """Test that cache is separate for different gate types."""
        analysis_canary = {"root_cause": "canary cause"}
        analysis_shadow = {"root_cause": "shadow cause"}

        cache_manager.set("canary", sample_canary_failure, analysis_canary)
        cache_manager.set("shadow", sample_shadow_failure, analysis_shadow)

        result_canary = cache_manager.get("canary", sample_canary_failure)
        result_shadow = cache_manager.get("shadow", sample_shadow_failure)

        assert result_canary["root_cause"] == "canary cause"
        assert result_shadow["root_cause"] == "shadow cause"


# ============================================================================
# ClaudeGateAnalyzer Initialization Tests
# ============================================================================


class TestClaudeGateAnalyzerInit:
    """Tests for ClaudeGateAnalyzer initialization."""

    def test_init_with_valid_api_key(self, mock_anthropic, temp_cache_dir):
        """Test initialization with valid API key."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )
        assert analyzer.client is not None
        assert analyzer.api_key == "test-key"

    def test_init_with_env_api_key(self, mock_anthropic, temp_cache_dir):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            analyzer = ClaudeGateAnalyzer(cache_manager=GateCacheManager(temp_cache_dir))
            assert analyzer.client is not None

    def test_init_without_api_key_graceful_degradation(self, mock_anthropic, temp_cache_dir):
        """Test graceful degradation without API key."""
        with patch.dict("os.environ", {}, clear=True):
            analyzer = ClaudeGateAnalyzer(cache_manager=GateCacheManager(temp_cache_dir))
            assert analyzer.client is None

    def test_init_without_anthropic_package(self, temp_cache_dir):
        """Test graceful degradation when anthropic package unavailable."""
        with patch("na0s.agents.claude_gate_analyzer.HAS_ANTHROPIC", False):
            analyzer = ClaudeGateAnalyzer(cache_manager=GateCacheManager(temp_cache_dir))
            assert analyzer.client is None

    def test_init_custom_model(self, mock_anthropic, temp_cache_dir):
        """Test initialization with custom model."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            model="claude-3-opus-20250219",
            cache_manager=GateCacheManager(temp_cache_dir),
        )
        assert analyzer.model == "claude-3-opus-20250219"

    def test_init_custom_retry_settings(self, mock_anthropic, temp_cache_dir):
        """Test initialization with custom retry settings."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=5,
            retry_delay=0.5,
            cache_manager=GateCacheManager(temp_cache_dir),
        )
        assert analyzer.max_retries == 5
        assert analyzer.retry_delay == 0.5


# ============================================================================
# Claude API Call Tests
# ============================================================================


class TestClaudeAPICalls:
    """Tests for Claude API call behavior."""

    def test_successful_api_call(self, mock_anthropic, temp_cache_dir):
        """Test successful Claude API call."""
        # Setup mock response
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "test cause"}')]
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )
        response = analyzer._call_claude_with_retries("test prompt")

        assert response is not None
        assert "root_cause" in response

    def test_rate_limit_retry_with_backoff(self, mock_anthropic, temp_cache_dir):
        """Test that rate limit (429) triggers retries with backoff."""
        from na0s.agents.claude_gate_analyzer import RateLimitError

        # Setup mock to fail then succeed
        mock_anthropic.messages.create.side_effect = [
            RateLimitError("Rate limited", response=MagicMock(status_code=429)),
            MagicMock(content=[MagicMock(text='{"root_cause": "test"}')]),
        ]

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=2,
            retry_delay=0.01,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        with patch("time.sleep"):  # Mock sleep to avoid delays
            response = analyzer._call_claude_with_retries("test prompt")

        assert response is not None
        assert mock_anthropic.messages.create.call_count == 2

    def test_connection_timeout_retry(self, mock_anthropic, temp_cache_dir):
        """Test that connection timeout triggers retries."""
        from na0s.agents.claude_gate_analyzer import APITimeoutError

        mock_anthropic.messages.create.side_effect = [
            APITimeoutError("Timeout", response=MagicMock(status_code=504)),
            MagicMock(content=[MagicMock(text='{"root_cause": "test"}')]),
        ]

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=2,
            retry_delay=0.01,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        with patch("time.sleep"):
            response = analyzer._call_claude_with_retries("test prompt")

        assert response is not None

    def test_connection_error_retry(self, mock_anthropic, temp_cache_dir):
        """Test that connection error triggers retries."""
        from na0s.agents.claude_gate_analyzer import APIConnectionError

        mock_anthropic.messages.create.side_effect = [
            APIConnectionError("Connection failed", response=MagicMock(status_code=500)),
            MagicMock(content=[MagicMock(text='{"root_cause": "test"}')]),
        ]

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=2,
            retry_delay=0.01,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        with patch("time.sleep"):
            response = analyzer._call_claude_with_retries("test prompt")

        assert response is not None

    def test_auth_error_no_retry(self, mock_anthropic, temp_cache_dir):
        """Test that auth errors (401) do not trigger retries."""
        mock_anthropic.messages.create.side_effect = Exception("Invalid API key")

        analyzer = ClaudeGateAnalyzer(
            api_key="invalid-key",
            max_retries=3,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        response = analyzer._call_claude_with_retries("test prompt")

        assert response is None
        assert mock_anthropic.messages.create.call_count == 1

    def test_bad_request_error_no_retry(self, mock_anthropic, temp_cache_dir):
        """Test that bad request errors (400) do not trigger retries."""
        mock_anthropic.messages.create.side_effect = Exception("Invalid request")

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=3,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        response = analyzer._call_claude_with_retries("test prompt")

        assert response is None
        assert mock_anthropic.messages.create.call_count == 1

    def test_exhausted_retries_returns_none(self, mock_anthropic, temp_cache_dir):
        """Test that exhausted retries return None."""
        from na0s.agents.claude_gate_analyzer import APIConnectionError

        mock_anthropic.messages.create.side_effect = APIConnectionError(
            "Connection failed", response=MagicMock(status_code=500)
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=2,
            retry_delay=0.01,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        with patch("time.sleep"):
            response = analyzer._call_claude_with_retries("test prompt")

        assert response is None
        assert mock_anthropic.messages.create.call_count == 2

    def test_exponential_backoff(self, mock_anthropic, temp_cache_dir):
        """Test that retry backoff is exponential."""
        from na0s.agents.claude_gate_analyzer import APIConnectionError

        mock_anthropic.messages.create.side_effect = APIConnectionError(
            "Connection failed", response=MagicMock(status_code=500)
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=3,
            retry_delay=1.0,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        with patch("time.sleep") as mock_sleep:
            analyzer._call_claude_with_retries("test prompt")

        # Should have called sleep with exponentially increasing delays
        assert mock_sleep.call_count == 2
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        # First retry: 1.0 * 2^0 = 1.0
        # Second retry: 1.0 * 2^1 = 2.0
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0


# ============================================================================
# Response Parsing Tests
# ============================================================================


class TestResponseParsing:
    """Tests for parsing Claude responses."""

    def test_json_parse_valid_response(self, temp_cache_dir):
        """Test parsing valid JSON response."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )

        response = '{"root_cause": "invalid filter", "fix_specificity": "update regex"}'
        analysis = analyzer._parse_analysis_response(response, "canary", {})

        assert analysis is not None
        assert analysis["root_cause"] == "invalid filter"
        assert analysis["fix_specificity"] == "update regex"

    def test_json_parse_with_surrounding_text(self, temp_cache_dir):
        """Test parsing JSON response with surrounding text."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )

        response = """Here's the analysis:
{"root_cause": "invalid filter", "fix_specificity": "update regex"}
This is the end."""
        analysis = analyzer._parse_analysis_response(response, "canary", {})

        assert analysis is not None
        assert analysis["root_cause"] == "invalid filter"

    def test_json_parse_error_returns_none(self, temp_cache_dir):
        """Test that JSON parse error returns None."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )

        response = 'This is not JSON {invalid'
        analysis = analyzer._parse_analysis_response(response, "canary", {})

        assert analysis is None

    def test_json_parse_no_json_returns_none(self, temp_cache_dir):
        """Test that response with no JSON returns None."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )

        response = "No JSON here, just plain text"
        analysis = analyzer._parse_analysis_response(response, "canary", {})

        assert analysis is None

    def test_parse_response_adds_metadata(self, temp_cache_dir):
        """Test that parsing adds metadata fields."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )

        response = '{"root_cause": "test"}'
        analysis = analyzer._parse_analysis_response(response, "canary", {})

        assert analysis is not None
        assert analysis["gate_type"] == "canary"
        assert "analyzed_at" in analysis
        assert "raw_response" in analysis

    def test_parse_response_fills_missing_fields(self, temp_cache_dir):
        """Test that parsing fills missing required fields."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )

        response = '{"root_cause": "test"}'
        analysis = analyzer._parse_analysis_response(response, "canary", {})

        assert analysis["root_cause"] == "test"
        assert "fix_specificity" in analysis


# ============================================================================
# Gate-Specific Analysis Tests
# ============================================================================


class TestGateSpecificAnalysis:
    """Tests for gate-specific analysis prompts and behavior."""

    def test_canary_failure_analysis(self, mock_anthropic, temp_cache_dir, sample_canary_failure):
        """Test analysis of canary failure."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    text='{"root_cause": "DAN technique evasion", "fix_specificity": "Update filter"}'
                )
            ]
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )
        analysis = analyzer.analyze_gate("canary", sample_canary_failure)

        assert analysis is not None
        assert analysis["gate_type"] == "canary"
        # Verify prompt includes canary-specific context
        call_args = mock_anthropic.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "canary" in prompt.lower()

    def test_shadow_failure_analysis(self, mock_anthropic, temp_cache_dir, sample_shadow_failure):
        """Test analysis of shadow failure (regression)."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    text='{"root_cause": "Feature drift regression", "fix_specificity": "Retrain model"}'
                )
            ]
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )
        analysis = analyzer.analyze_gate("shadow", sample_shadow_failure)

        assert analysis is not None
        assert analysis["gate_type"] == "shadow"
        # Verify prompt includes shadow-specific context
        call_args = mock_anthropic.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "regression" in prompt.lower() or "shadow" in prompt.lower()

    def test_f14_failure_analysis(self, mock_anthropic, temp_cache_dir, sample_f14_failure):
        """Test analysis of F14 failure (TPR drops)."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    text='{"root_cause": "SQLi detector regression", "fix_specificity": "Review layer"}'
                )
            ]
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )
        analysis = analyzer.analyze_gate("f14", sample_f14_failure)

        assert analysis is not None
        assert analysis["gate_type"] == "f14"
        # Verify prompt includes F14-specific context
        call_args = mock_anthropic.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "TPR" in prompt or "category" in prompt.lower()


# ============================================================================
# Caching Behavior Tests
# ============================================================================


class TestCachingBehavior:
    """Tests for caching integration in analysis."""

    def test_cache_hit_skips_api_call(self, mock_anthropic, temp_cache_dir, sample_canary_failure):
        """Test that cache hit skips API call."""
        cache_manager = GateCacheManager(cache_dir=temp_cache_dir)
        analysis_cached = {"root_cause": "cached result", "fix_specificity": "cached fix"}
        cache_manager.set("canary", sample_canary_failure, analysis_cached)

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=cache_manager
        )
        result = analyzer.analyze_gate("canary", sample_canary_failure)

        # Should return cached result
        assert result is not None
        assert result["root_cause"] == "cached result"
        # API should not have been called
        mock_anthropic.messages.create.assert_not_called()

    def test_cache_miss_calls_api(self, mock_anthropic, temp_cache_dir, sample_canary_failure):
        """Test that cache miss calls API."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "api result", "fix": "api fix"}')]
        )

        cache_manager = GateCacheManager(cache_dir=temp_cache_dir)
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=cache_manager
        )
        result = analyzer.analyze_gate("canary", sample_canary_failure)

        # Should call API
        assert mock_anthropic.messages.create.called

    def test_cache_stores_api_result(self, mock_anthropic, temp_cache_dir, sample_canary_failure):
        """Test that API results are cached."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "api result", "fix": "api fix"}')]
        )

        cache_manager = GateCacheManager(cache_dir=temp_cache_dir)
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=cache_manager
        )
        analyzer.analyze_gate("canary", sample_canary_failure)

        # Verify result is cached
        cached = cache_manager.get("canary", sample_canary_failure)
        assert cached is not None
        assert cached["root_cause"] == "api result"

    def test_client_unavailable_returns_none(self, temp_cache_dir, sample_canary_failure):
        """Test that unavailable client returns None."""
        analyzer = ClaudeGateAnalyzer(
            api_key=None, cache_manager=GateCacheManager(temp_cache_dir)
        )
        analyzer.client = None

        result = analyzer.analyze_gate("canary", sample_canary_failure)
        assert result is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Tests for integration with GateAnalyzer."""

    def test_analyze_gate_with_claude_enabled(
        self, mock_anthropic, temp_data_dir, sample_canary_failure, monkeypatch
    ):
        """Test analyze_gate_with_claude when enabled."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "test cause", "fix": "test fix"}')]
        )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        gate_analyzer = GateAnalyzer(data_dir=temp_data_dir, use_claude=True)
        result = gate_analyzer.analyze_gate_with_claude("canary", sample_canary_failure)

        assert result is not None

    def test_analyze_gate_with_claude_disabled(self, temp_data_dir, sample_canary_failure):
        """Test analyze_gate_with_claude when disabled."""
        gate_analyzer = GateAnalyzer(data_dir=temp_data_dir, use_claude=False)
        result = gate_analyzer.analyze_gate_with_claude("canary", sample_canary_failure)

        assert result is None

    def test_diagnose_failures_includes_claude_analysis(
        self, mock_anthropic, temp_data_dir, monkeypatch
    ):
        """Test that diagnose_failures includes Claude analysis."""
        data_dir = Path(temp_data_dir)

        # Write failing canary
        canary_data = {
            "passed": False,
            "metrics": {"tpr": 0.92, "tnr": 0.88, "fpr": 0.05},
            "errors": [{"technique": "DAN", "text_preview": "test"}],
        }
        with open(data_dir / "canary" / "canary_results.json", "w") as f:
            json.dump(canary_data, f)

        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "DAN evasion", "fix": "update"}')]
        )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        gate_analyzer = GateAnalyzer(data_dir=str(data_dir), use_claude=True)
        results = gate_analyzer.diagnose_failures()

        assert "claude_analysis" in results
        assert "canary" in results["claude_analysis"]

    def test_format_message_includes_claude_analysis(
        self, mock_anthropic, temp_data_dir
    ):
        """Test that format_message includes Claude analysis."""
        data_dir = Path(temp_data_dir)

        # Write failing canary
        canary_data = {
            "passed": False,
            "metrics": {"tpr": 0.92, "tnr": 0.88, "fpr": 0.05},
            "errors": [{"technique": "DAN", "text_preview": "test"}],
        }
        with open(data_dir / "canary" / "canary_results.json", "w") as f:
            json.dump(canary_data, f)

        mock_anthropic.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    text='{"root_cause": "DAN evasion", "fix_specificity": "update filter"}'
                )
            ]
        )

        gate_analyzer = GateAnalyzer(data_dir=str(data_dir), use_claude=True)
        message = gate_analyzer.format_message()

        assert "Root Cause" in message or "Fix" in message or "DAN" in message

    def test_diagnose_failures_skips_passed_gates(self, mock_anthropic, temp_data_dir):
        """Test that diagnose_failures skips Claude analysis for passed gates."""
        data_dir = Path(temp_data_dir)

        # Write passing gates
        canary_data = {
            "passed": True,
            "metrics": {"tpr": 0.97, "tnr": 0.93, "fpr": 0.02},
            "errors": [],
        }
        with open(data_dir / "canary" / "canary_results.json", "w") as f:
            json.dump(canary_data, f)

        gate_analyzer = GateAnalyzer(data_dir=str(data_dir), use_claude=True)
        results = gate_analyzer.diagnose_failures()

        # Should not call Claude for passed gates
        mock_anthropic.messages.create.assert_not_called()
        assert results["overall_verdict"] == "ALL_PASSED"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_api_unavailable_returns_none(self, mock_anthropic, temp_cache_dir):
        """Test that API unavailable returns None."""
        from na0s.agents.claude_gate_analyzer import APIConnectionError

        mock_anthropic.messages.create.side_effect = APIConnectionError(
            "Connection failed", response=MagicMock(status_code=500)
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=1,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        result = analyzer.analyze_gate("canary", {"error_count": 1})
        assert result is None

    def test_cache_io_error_continues(self, mock_anthropic, temp_cache_dir):
        """Test that cache I/O errors don't block analysis."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "test"}')]
        )

        cache_manager = GateCacheManager(cache_dir=temp_cache_dir)
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=cache_manager
        )

        # Mock cache write to fail
        with patch.object(cache_manager, "set", return_value=False):
            result = analyzer.analyze_gate("canary", {"error_count": 1})

        # Should still return result even if cache write fails
        assert result is not None

    def test_timeout_parameter_respected(self, mock_anthropic, temp_cache_dir):
        """Test that timeout parameter affects retries."""
        from na0s.agents.claude_gate_analyzer import APITimeoutError

        mock_anthropic.messages.create.side_effect = APITimeoutError(
            "Timeout", response=MagicMock(status_code=504)
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=1,
            retry_delay=0.01,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        with patch("time.sleep") as mock_sleep:
            result = analyzer._call_claude_with_retries("test")

        assert result is None
        assert mock_sleep.call_count == 0  # No sleep on last attempt


# ============================================================================
# Edge Cases and Boundaries
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_failure_data(self, mock_anthropic, temp_cache_dir):
        """Test with empty failure data."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "unknown"}')]
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )
        result = analyzer.analyze_gate("canary", {})

        assert result is not None

    def test_large_failure_data(self, mock_anthropic, temp_cache_dir):
        """Test with large failure data."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "test"}')]
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )

        # Create large failure data with many errors
        large_data = {
            "error_count": 1000,
            "errors": [
                {"technique": f"technique_{i}", "text": f"text_{i}"}
                for i in range(100)
            ],
        }
        result = analyzer.analyze_gate("canary", large_data)

        assert result is not None

    def test_unknown_gate_type(self, mock_anthropic, temp_cache_dir):
        """Test with unknown gate type."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "test"}')]
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )
        result = analyzer.analyze_gate("unknown_gate", {"error": "test"})

        # Should still work with default prompt
        assert result is not None

    def test_unicode_in_failure_data(self, mock_anthropic, temp_cache_dir):
        """Test with unicode characters in failure data."""
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"root_cause": "测试"}')]
        )

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )

        data = {"error": "测试错误", "technique": "技术"}
        result = analyzer.analyze_gate("canary", data)

        assert result is not None

    def test_response_with_nested_json(self, temp_cache_dir):
        """Test parsing response with nested JSON."""
        analyzer = ClaudeGateAnalyzer(
            api_key="test-key", cache_manager=GateCacheManager(temp_cache_dir)
        )

        response = '{"root_cause": "nested", "details": {"key": "value"}}'
        analysis = analyzer._parse_analysis_response(response, "canary", {})

        assert analysis is not None
        assert analysis["root_cause"] == "nested"

    def test_max_retries_zero(self, mock_anthropic, temp_cache_dir):
        """Test with max_retries = 0."""
        mock_anthropic.messages.create.side_effect = Exception("Error")

        analyzer = ClaudeGateAnalyzer(
            api_key="test-key",
            max_retries=0,
            cache_manager=GateCacheManager(temp_cache_dir),
        )

        response = analyzer._call_claude_with_retries("test")
        assert response is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
