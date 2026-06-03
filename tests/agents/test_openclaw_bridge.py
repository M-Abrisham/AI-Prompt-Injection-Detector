"""Tests for OpenClaw iMessage bridge.

Tests mode detection, real/mock OpenClaw switching, graceful degradation,
and fallback behavior.
"""

import json
import pytest
import sys
import socket
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from threading import Thread
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from na0s.agents.openclaw_bridge import OpenClawBridge
from na0s import config


class MockOpenClawHandler(BaseHTTPRequestHandler):
    """Mock OpenClaw HTTP server for testing."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        elif self.path == "/replies":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"reply": "approve"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/send":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"success": True, "message_id": "test-123"}).encode()
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress logging."""
        pass


def find_free_port():
    """Find a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def mock_server():
    """Start mock OpenClaw server for testing."""
    port = find_free_port()
    server = HTTPServer(("127.0.0.1", port), MockOpenClawHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # Give server time to start
    yield server, port
    server.shutdown()


class TestModeDetection:
    """Test OpenClaw mode detection."""

    def test_mock_mode_explicit(self):
        """Test explicit mock mode."""
        bridge = OpenClawBridge(mode="mock")
        assert bridge.active_mode == "mock"

    def test_real_mode_requires_openclaw(self):
        """Test that real mode fails if OpenClaw not available."""
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=False
        ):
            with pytest.raises(RuntimeError, match="real OpenClaw not available"):
                OpenClawBridge(mode="real")

    def test_real_mode_with_available_openclaw(self):
        """Test real mode when OpenClaw is available."""
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=True
        ):
            bridge = OpenClawBridge(mode="real")
            assert bridge.active_mode == "real"

    def test_auto_mode_real_available(self):
        """Test auto mode with real OpenClaw available."""
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=True
        ):
            bridge = OpenClawBridge(mode="auto")
            assert bridge.active_mode == "real"

    def test_auto_mode_fallback_to_mock(self):
        """Test auto mode falls back to mock when real unavailable."""
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=False
        ):
            bridge = OpenClawBridge(mode="auto")
            assert bridge.active_mode == "mock"


class TestRealOpenClawDetection:
    """Test detection of real OpenClaw availability."""

    def test_detect_via_health_check(self, mock_server):
        """Test detection via HTTP health check."""
        server, port = mock_server
        bridge = OpenClawBridge(base_url=f"http://127.0.0.1:{port}", mode="mock")
        result = bridge._check_real_openclaw()
        assert result is True

    def test_detect_via_health_check_unavailable(self):
        """Test detection when health check fails."""
        bridge = OpenClawBridge(
            base_url="http://127.0.0.1:9999", mode="mock"
        )  # Non-existent port
        result = bridge._check_real_openclaw()
        assert result is False

    @patch("pathlib.Path.exists")
    def test_detect_via_app_installed(self, mock_exists):
        """Test detection via app installation."""
        mock_exists.return_value = True
        bridge = OpenClawBridge(mode="mock")
        result = bridge._check_real_openclaw()
        assert result is True

    @patch("pathlib.Path.exists")
    def test_app_not_installed(self, mock_exists):
        """Test when app is not installed."""
        mock_exists.return_value = False
        with patch("requests.get", side_effect=Exception("Connection refused")):
            bridge = OpenClawBridge(mode="mock")
            result = bridge._check_real_openclaw()
            assert result is False


class TestMockMode:
    """Test mock mode message sending and polling."""

    def test_send_message_mock(self, mock_server):
        """Test sending message in mock mode."""
        server, port = mock_server
        bridge = OpenClawBridge(base_url=f"http://127.0.0.1:{port}", mode="mock")
        result = bridge.send_message("Test message")
        assert result is True

    def test_send_message_mock_failure(self):
        """Test send failure with unavailable mock."""
        bridge = OpenClawBridge(base_url="http://127.0.0.1:9999", mode="mock")
        result = bridge.send_message("Test message")
        assert result is False

    def test_poll_replies_mock(self, mock_server):
        """Test polling replies in mock mode."""
        server, port = mock_server
        bridge = OpenClawBridge(base_url=f"http://127.0.0.1:{port}", mode="mock")
        reply = bridge.poll_replies(timeout=1)
        assert reply == "approve"

    def test_poll_replies_mock_timeout(self):
        """Test poll timeout with unavailable mock."""
        bridge = OpenClawBridge(base_url="http://127.0.0.1:9999", mode="mock")
        reply = bridge.poll_replies(timeout=1)
        assert reply is None


class TestAutoModeFallback:
    """Test auto mode fallback behavior."""

    def test_auto_mode_switches_to_mock_on_init(self):
        """Test auto mode initializes to mock when real unavailable."""
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=False
        ):
            bridge = OpenClawBridge(mode="auto")
            assert bridge.active_mode == "mock"
            assert bridge.mode == "auto"


class TestConfigIntegration:
    """Test configuration integration."""

    def test_explicit_params_override_config(self):
        """Test explicit parameters override config."""
        bridge = OpenClawBridge(
            base_url="http://explicit:3000", timeout=60, mode="mock"
        )
        assert bridge.base_url == "http://explicit:3000"
        assert bridge.timeout == 60
        assert bridge.mode == "mock"


class TestMessageWithApproval:
    """Test message sending with approval flow."""

    def test_send_with_approval_timeout(self):
        """Test approval flow timeout."""
        bridge = OpenClawBridge(base_url="http://127.0.0.1:9999", mode="mock")
        reply = bridge.send_message_with_approval(
            "Deploy approved?", timeout=1, expected_replies=["approve"]
        )
        assert reply is None


class TestSkillRegistration:
    """Test slash command skill registration."""

    def test_register_skill_failure(self):
        """Test skill registration failure with unavailable server."""
        bridge = OpenClawBridge(base_url="http://127.0.0.1:9999", mode="mock")

        def handler(args: str) -> str:
            return "executed"

        result = bridge.register_skill("test_cmd", "Test skill", handler)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRealOpenClawIntegration:
    """Test real OpenClaw SDK integration."""

    def test_send_real_via_http_fallback(self, mock_server):
        """Test that _send_message_real works via HTTP when SDK unavailable."""
        server, port = mock_server
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=True
        ):
            bridge = OpenClawBridge(base_url=f"http://127.0.0.1:{port}", mode="real")

            # SDK import fails, should fallback to HTTP
            result = bridge._send_message_real("Test message")
            assert result is True

    def test_real_mode_initializes_correctly(self):
        """Test that real mode initializes when OpenClaw available."""
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=True
        ):
            bridge = OpenClawBridge(mode="real")
            assert bridge.active_mode == "real"
            assert bridge.mode == "real"


class TestGracefulDegradation:
    """Test graceful degradation when both real and mock unavailable."""

    def test_auto_mode_handles_both_unavailable(self):
        """Test auto mode doesn't crash when both real and mock unavailable."""
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=False
        ):
            # Should not crash, should init to mock
            bridge = OpenClawBridge(base_url="http://127.0.0.1:9999", mode="auto")
            assert bridge.active_mode == "mock"
            
            # Send should fail gracefully (no exception)
            result = bridge.send_message("Test")
            assert result is False
            
            # Poll should timeout gracefully
            reply = bridge.poll_replies(timeout=1)
            assert reply is None

    def test_real_mode_fails_explicitly(self):
        """Test real mode fails explicitly if OpenClaw unavailable."""
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=False
        ):
            with pytest.raises(RuntimeError):
                OpenClawBridge(mode="real")


class TestModeConsistency:
    """Test mode consistency across operations."""

    def test_all_operations_use_active_mode(self):
        """Test that all operations respect active_mode."""
        with patch.object(
            OpenClawBridge, "_check_real_openclaw", return_value=False
        ):
            bridge = OpenClawBridge(mode="auto")
            
            # All operations should use active_mode
            assert bridge.active_mode == "mock"
            
            # send_message routes to _send_message_mock
            with patch.object(bridge, "_send_message_mock", return_value=True) as mock_send:
                bridge.send_message("test")
                mock_send.assert_called_once()
            
            # poll_replies routes to _poll_replies_mock
            with patch.object(bridge, "_poll_replies_mock", return_value="ok") as mock_poll:
                bridge.poll_replies(timeout=1)
                mock_poll.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
