"""Tests for Layer 15 HTTP utilities (fetch_json, fetch_text, github_headers).

Covers the shared infrastructure used by all sync modules:
- fetch_json retry logic, rate limit detection, error handling
- fetch_text error handling
- github_headers with/without token
"""

import json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from na0s.layer15.base import RateLimitError, SourceUnavailableError
from na0s.layer15.http_utils import fetch_json, fetch_text, github_headers


# ---------------------------------------------------------------------------
# github_headers
# ---------------------------------------------------------------------------


class TestGithubHeaders:

    def test_includes_accept_and_user_agent(self):
        h = github_headers()
        assert "Accept" in h
        assert "User-Agent" in h

    def test_with_explicit_token(self):
        h = github_headers(token="ghp_test123")
        assert h["Authorization"] == "token ghp_test123"

    def test_without_token(self):
        with patch.dict("os.environ", {}, clear=True):
            h = github_headers()
            assert "Authorization" not in h

    def test_env_var_token(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_env"}):
            h = github_headers()
            assert h["Authorization"] == "token ghp_env"

    def test_explicit_token_overrides_env(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_env"}):
            h = github_headers(token="ghp_explicit")
            assert h["Authorization"] == "token ghp_explicit"


# ---------------------------------------------------------------------------
# fetch_json
# ---------------------------------------------------------------------------


def _mock_response(status=200, body=b'{"ok": true}', headers=None):
    """Create a mock urllib response."""
    mock = MagicMock()
    mock.status = status
    mock.read.return_value = body
    mock.getheaders.return_value = list((headers or {}).items())
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


class TestFetchJson:

    @patch("na0s.layer15.http_utils.urlopen")
    def test_happy_path(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            body=b'{"key": "value"}'
        )
        data, headers = fetch_json("https://example.com/api")
        assert data == {"key": "value"}

    @patch("na0s.layer15.http_utils.urlopen")
    def test_returns_response_headers(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            headers={"X-RateLimit-Remaining": "4999"}
        )
        _, headers = fetch_json("https://example.com/api")
        assert headers["x-ratelimit-remaining"] == "4999"

    @patch("na0s.layer15.http_utils.urlopen")
    def test_404_raises_source_unavailable(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "https://example.com", 404, "Not Found", {}, None
        )
        with pytest.raises(SourceUnavailableError, match="Not found"):
            fetch_json("https://example.com/api")

    @patch("na0s.layer15.http_utils.urlopen")
    def test_rate_limit_403_raises(self, mock_urlopen):
        err = HTTPError(
            "https://example.com", 403, "Forbidden", {}, None
        )
        err.headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1234567890"}
        mock_urlopen.side_effect = err
        with pytest.raises(RateLimitError, match="rate limit"):
            fetch_json("https://example.com/api")

    @patch("na0s.layer15.http_utils.urlopen")
    @patch("na0s.layer15.http_utils.time.sleep")
    def test_retries_on_500(self, mock_sleep, mock_urlopen):
        """500 errors trigger retry with backoff."""
        err = HTTPError("https://example.com", 500, "Server Error", {}, None)
        # Fail twice, then succeed
        mock_urlopen.side_effect = [
            err, err,
            _mock_response(body=b'{"ok": true}'),
        ]
        data, _ = fetch_json("https://example.com/api")
        assert data == {"ok": True}
        assert mock_sleep.call_count == 2

    @patch("na0s.layer15.http_utils.urlopen")
    @patch("na0s.layer15.http_utils.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep, mock_urlopen):
        """All retries exhausted → SourceUnavailableError."""
        err = HTTPError("https://example.com", 502, "Bad Gateway", {}, None)
        mock_urlopen.side_effect = err
        with pytest.raises(SourceUnavailableError, match="after 3 attempts"):
            fetch_json("https://example.com/api")

    @patch("na0s.layer15.http_utils.urlopen")
    @patch("na0s.layer15.http_utils.time.sleep")
    def test_network_error_retries(self, mock_sleep, mock_urlopen):
        """URLError (network) triggers retry."""
        mock_urlopen.side_effect = [
            URLError("Connection refused"),
            _mock_response(body=b'{"ok": true}'),
        ]
        data, _ = fetch_json("https://example.com/api")
        assert data == {"ok": True}


# ---------------------------------------------------------------------------
# fetch_text
# ---------------------------------------------------------------------------


class TestFetchText:

    @patch("na0s.layer15.http_utils.urlopen")
    def test_happy_path(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(body=b"hello world")
        text = fetch_text("https://example.com/file.txt")
        assert text == "hello world"

    @patch("na0s.layer15.http_utils.urlopen")
    def test_network_error_raises(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("Connection refused")
        with pytest.raises(SourceUnavailableError, match="Connection refused"):
            fetch_text("https://example.com/file.txt")

    @patch("na0s.layer15.http_utils.urlopen")
    def test_http_error_raises(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "https://example.com", 404, "Not Found", {}, None
        )
        with pytest.raises(SourceUnavailableError):
            fetch_text("https://example.com/file.txt")
