"""Tests for Layer 15 endpoint health checker.

All HTTP calls are mocked -- no real API hits.

Covers:
- EndpointStatus statuses: ok, slow, unreachable, rate_limited
- HealthReport terminal formatting
- HealthReport serialization (to_dict)
- HealthReport.all_ok logic
- EndpointHealthChecker.check_one for each status
- EndpointHealthChecker.check_all aggregation
"""

import json
import time
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from na0s.layer15.endpoint_health import (
    SLOW_THRESHOLD_MS,
    EndpointHealthChecker,
    EndpointStatus,
    HealthReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status=200, body=b"", headers=None):
    """Build a mock HTTP response object."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.headers = MagicMock()
    resp.headers.get = lambda key, default="": (headers or {}).get(key, default)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Tests: EndpointStatus
# ---------------------------------------------------------------------------

class TestEndpointStatus:

    def test_ok_status(self):
        ep = EndpointStatus(
            name="Test", url="https://example.com",
            status="ok", response_time_ms=150, status_code=200,
        )
        assert ep.status == "ok"
        assert ep.error == ""

    def test_slow_status(self):
        ep = EndpointStatus(
            name="Test", url="https://example.com",
            status="slow", response_time_ms=3000, status_code=200,
        )
        assert ep.status == "slow"

    def test_unreachable_status(self):
        ep = EndpointStatus(
            name="Test", url="https://example.com",
            status="unreachable", response_time_ms=0, status_code=0,
            error="Connection refused",
        )
        assert ep.status == "unreachable"
        assert ep.error == "Connection refused"

    def test_rate_limited_status(self):
        ep = EndpointStatus(
            name="Test", url="https://example.com",
            status="rate_limited", response_time_ms=100, status_code=429,
        )
        assert ep.status == "rate_limited"


# ---------------------------------------------------------------------------
# Tests: HealthReport
# ---------------------------------------------------------------------------

class TestHealthReport:

    @pytest.fixture
    def mixed_report(self):
        ts = datetime(2026, 3, 24, 12, 0, 0, tzinfo=timezone.utc)
        return HealthReport(
            endpoints=[
                EndpointStatus("ATLAS GitHub API", "https://api.github.com/repos/mitre-atlas/atlas-data",
                               "ok", 283, 200),
                EndpointStatus("Garak GitHub API", "https://api.github.com/repos/leondz/garak",
                               "ok", 194, 200),
                EndpointStatus("AIID GraphQL", "https://incidentdatabase.ai/api/graphql",
                               "slow", 2847, 200),
                EndpointStatus("Dead API", "https://example.com/gone",
                               "unreachable", 0, 404, error="HTTP 404"),
            ],
            timestamp=ts,
        )

    def test_format_terminal_contains_all_names(self, mixed_report):
        output = mixed_report.format_terminal()
        assert "ATLAS GitHub API" in output
        assert "Garak GitHub API" in output
        assert "AIID GraphQL" in output
        assert "Dead API" in output

    def test_format_terminal_ok_shows_time(self, mixed_report):
        output = mixed_report.format_terminal()
        assert "OK (283ms)" in output
        assert "OK (194ms)" in output

    def test_format_terminal_slow_shows_time(self, mixed_report):
        output = mixed_report.format_terminal()
        assert "SLOW (2847ms)" in output

    def test_format_terminal_unreachable_shows_error(self, mixed_report):
        output = mixed_report.format_terminal()
        assert "UNREACHABLE" in output
        assert "HTTP 404" in output

    def test_format_terminal_has_timestamp(self, mixed_report):
        output = mixed_report.format_terminal()
        assert "2026-03-24 12:00:00 UTC" in output

    def test_all_ok_false_when_unreachable(self, mixed_report):
        assert mixed_report.all_ok() is False

    def test_all_ok_true_when_all_reachable(self):
        report = HealthReport(
            endpoints=[
                EndpointStatus("A", "https://a.com", "ok", 100, 200),
                EndpointStatus("B", "https://b.com", "slow", 3000, 200),
            ],
            timestamp=datetime.now(timezone.utc),
        )
        assert report.all_ok() is True

    def test_all_ok_false_when_rate_limited(self):
        report = HealthReport(
            endpoints=[
                EndpointStatus("A", "https://a.com", "ok", 100, 200),
                EndpointStatus("B", "https://b.com", "rate_limited", 50, 429),
            ],
            timestamp=datetime.now(timezone.utc),
        )
        assert report.all_ok() is False

    def test_to_dict_structure(self, mixed_report):
        d = mixed_report.to_dict()
        assert d["timestamp"] == "2026-03-24T12:00:00+00:00"
        assert d["all_ok"] is False
        assert len(d["endpoints"]) == 4
        first = d["endpoints"][0]
        assert first["name"] == "ATLAS GitHub API"
        assert first["status"] == "ok"
        assert first["response_time_ms"] == 283
        assert first["status_code"] == 200
        assert first["error"] == ""

    def test_to_dict_is_json_serializable(self, mixed_report):
        d = mixed_report.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        roundtrip = json.loads(serialized)
        assert roundtrip["all_ok"] is False

    def test_format_terminal_rate_limited(self):
        report = HealthReport(
            endpoints=[
                EndpointStatus("GitHub", "https://api.github.com",
                               "rate_limited", 80, 429),
            ],
            timestamp=datetime(2026, 3, 24, 12, 0, 0, tzinfo=timezone.utc),
        )
        output = report.format_terminal()
        assert "RATE LIMITED" in output
        assert "429" in output


# ---------------------------------------------------------------------------
# Tests: EndpointHealthChecker.check_one
# ---------------------------------------------------------------------------

class TestCheckOne:

    def setup_method(self):
        self.checker = EndpointHealthChecker()

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_ok_response(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(200, b'{"ok": true}')
        result = self.checker.check_one("Test", "https://example.com", timeout=5)
        assert result.status == "ok"
        assert result.status_code == 200
        assert result.response_time_ms >= 0

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_slow_response(self, mock_urlopen):
        """Response that takes longer than SLOW_THRESHOLD_MS is marked slow."""
        def slow_urlopen(req, timeout=None):
            time.sleep(0.01)  # Small sleep to ensure non-zero time
            return _make_response(200, b'{}')
        mock_urlopen.side_effect = slow_urlopen

        # Patch time.monotonic to simulate slow response
        real_monotonic = time.monotonic
        call_count = {"n": 0}
        def mock_monotonic():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return 0.0  # start time
            return 3.0  # 3000ms elapsed
        with patch("na0s.layer15.endpoint_health.time.monotonic", side_effect=mock_monotonic):
            result = self.checker.check_one("Slow API", "https://example.com/slow", timeout=10)

        assert result.status == "slow"
        assert result.response_time_ms >= SLOW_THRESHOLD_MS

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_unreachable_connection_error(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        result = self.checker.check_one("Dead", "https://dead.example.com", timeout=5)
        assert result.status == "unreachable"
        assert result.status_code == 0
        assert "Connection refused" in result.error

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_unreachable_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("timed out")
        result = self.checker.check_one("Timeout", "https://slow.example.com", timeout=1)
        assert result.status == "unreachable"
        assert "timed out" in result.error

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_unreachable_http_error(self, mock_urlopen):
        from urllib.error import HTTPError
        err = HTTPError("https://example.com", 500, "Server Error", {}, BytesIO(b""))
        mock_urlopen.side_effect = err
        result = self.checker.check_one("500", "https://example.com", timeout=5)
        assert result.status == "unreachable"
        assert result.status_code == 500
        assert "HTTP 500" in result.error

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_rate_limited_429(self, mock_urlopen):
        from urllib.error import HTTPError
        headers = MagicMock()
        headers.get = lambda key, default="": {"X-RateLimit-Remaining": "0"}.get(key, default)
        err = HTTPError("https://api.github.com", 429, "Too Many Requests", headers, BytesIO(b""))
        err.headers = headers
        mock_urlopen.side_effect = err
        result = self.checker.check_one("GitHub", "https://api.github.com", timeout=5)
        assert result.status == "rate_limited"
        assert result.status_code == 429

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_rate_limited_403_with_zero_remaining(self, mock_urlopen):
        from urllib.error import HTTPError
        headers = MagicMock()
        headers.get = lambda key, default="": {"X-RateLimit-Remaining": "0"}.get(key, default)
        err = HTTPError("https://api.github.com", 403, "Forbidden", headers, BytesIO(b""))
        err.headers = headers
        mock_urlopen.side_effect = err
        result = self.checker.check_one("GitHub", "https://api.github.com", timeout=5)
        assert result.status == "rate_limited"
        assert result.status_code == 403

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_rate_limited_on_200_with_zero_remaining(self, mock_urlopen):
        """Even a 200 with X-RateLimit-Remaining: 0 flags rate_limited."""
        mock_urlopen.return_value = _make_response(
            200, b'{}', headers={"X-RateLimit-Remaining": "0"}
        )
        result = self.checker.check_one("GitHub", "https://api.github.com", timeout=5)
        assert result.status == "rate_limited"

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_post_method_used_for_graphql(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(200, b'{"data":{"__typename":"Query"}}')
        body = json.dumps({"query": "{ __typename }"}).encode("utf-8")
        result = self.checker.check_one(
            "AIID", "https://incidentdatabase.ai/api/graphql", timeout=5,
            method="POST",
            headers={"Content-Type": "application/json", "Origin": "https://incidentdatabase.ai"},
            body=body,
        )
        assert result.status == "ok"
        # Verify the request was made with POST
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        assert req.data == body


# ---------------------------------------------------------------------------
# Tests: EndpointHealthChecker.check_all
# ---------------------------------------------------------------------------

class TestCheckAll:

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_check_all_returns_report_for_each_endpoint(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(200, b'{}')
        checker = EndpointHealthChecker()
        report = checker.check_all(timeout=5)
        assert isinstance(report, HealthReport)
        assert len(report.endpoints) == len(checker.ENDPOINTS)
        assert report.timestamp is not None

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_check_all_names_match_endpoints(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(200, b'{}')
        checker = EndpointHealthChecker()
        report = checker.check_all(timeout=5)
        expected_names = {name for name, *_ in checker.ENDPOINTS}
        actual_names = {ep.name for ep in report.endpoints}
        assert actual_names == expected_names

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_check_all_mixed_statuses(self, mock_urlopen):
        """Simulate one endpoint failing while others succeed."""
        from urllib.error import URLError

        call_count = {"n": 0}
        def side_effect(req, timeout=None):
            call_count["n"] += 1
            # Fail the third endpoint (AIID)
            if call_count["n"] == 3:
                raise URLError("Connection refused")
            return _make_response(200, b'{}')

        mock_urlopen.side_effect = side_effect
        checker = EndpointHealthChecker()
        report = checker.check_all(timeout=5)
        statuses = [ep.status for ep in report.endpoints]
        assert "unreachable" in statuses
        assert statuses.count("ok") == len(checker.ENDPOINTS) - 1

    @patch("na0s.layer15.endpoint_health.urlopen")
    def test_check_all_all_ok(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(200, b'{}')
        checker = EndpointHealthChecker()
        report = checker.check_all(timeout=5)
        assert report.all_ok() is True
