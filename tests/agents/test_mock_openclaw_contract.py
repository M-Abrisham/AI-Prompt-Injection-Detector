"""Contract test: OpenClawBridge against the real mock_openclaw HTTP server.

The audit found the bridge polled GET /replies while the mock only served
POST /poll_replies, so every local poll 404'd and the "works in mock" story
was false. This test spins up the actual mock server and drives the real
bridge against it, proving the send -> poll -> reply loop closes.
"""

import importlib.util
import socket
import threading
from pathlib import Path

import pytest

from na0s.agents.openclaw_bridge import OpenClawBridge, Reply

# Load scripts/mock_openclaw.py by path (scripts/ is not a package).
_MOCK_PATH = Path(__file__).parent.parent.parent / "scripts" / "mock_openclaw.py"
_spec = importlib.util.spec_from_file_location("mock_openclaw", _MOCK_PATH)
mock_openclaw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mock_openclaw)

from http.server import HTTPServer  # noqa: E402


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def mock_server():
    """Start the real MockOpenClawHandler on a free port; yield base_url."""
    # Reset shared class state between tests.
    mock_openclaw.MockOpenClawHandler.reply_queue = []
    mock_openclaw.MockOpenClawHandler.auto_reply = None
    mock_openclaw.MockOpenClawHandler.auto_reply_sender = None

    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), mock_openclaw.MockOpenClawHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _bridge(base_url: str) -> OpenClawBridge:
    return OpenClawBridge(base_url=base_url, timeout=5, mode="mock")


class TestMockContract:
    def test_health_detected(self, mock_server):
        bridge = _bridge(mock_server)
        # mock mode skips real detection, but the health endpoint must work.
        assert bridge._check_real_openclaw() is True

    def test_send_message_succeeds(self, mock_server):
        bridge = _bridge(mock_server)
        assert bridge.send_message("hello from test") is True

    def test_poll_returns_seeded_reply(self, mock_server):
        bridge = _bridge(mock_server)
        # Seed a reply the way a user's iMessage would arrive.
        mock_openclaw.MockOpenClawHandler.reply_queue.append("approve")
        assert bridge.poll_replies(timeout=5) == "approve"

    def test_poll_consumes_reply(self, mock_server):
        bridge = _bridge(mock_server)
        mock_openclaw.MockOpenClawHandler.reply_queue.append("reject")
        assert bridge.poll_replies(timeout=5) == "reject"
        # Queue is now empty -> next poll times out (no reply).
        assert bridge.poll_replies(timeout=1) is None

    def test_poll_times_out_when_no_reply(self, mock_server):
        bridge = _bridge(mock_server)
        assert bridge.poll_replies(timeout=1) is None

    def test_auto_reply_mode(self, mock_server):
        mock_openclaw.MockOpenClawHandler.auto_reply = "approve"
        bridge = _bridge(mock_server)
        assert bridge.poll_replies(timeout=5) == "approve"
        # auto-reply does not deplete: a second poll still returns it.
        assert bridge.poll_replies(timeout=5) == "approve"

    def test_full_loop_send_then_poll(self, mock_server):
        """End-to-end: send a message, user 'replies', bridge reads it."""
        bridge = _bridge(mock_server)
        assert bridge.send_message("Deploy now? approve | reject") is True
        mock_openclaw.MockOpenClawHandler.reply_queue.append("approve")
        assert bridge.poll_replies(timeout=5) == "approve"


class TestSenderAwareContract:
    """poll_replies_with_sender preserves the iMessage sender (defense-in-depth).

    The mock now lets a queued reply optionally carry a sender (dict shape); the
    sender-aware poll must surface it, while bare strings stay sender-less.
    """

    def _post_reply(self, base_url: str, body: dict) -> None:
        """Seed a reply via the real POST /replies contract (not direct queue)."""
        import requests

        resp = requests.post(f"{base_url}/replies", json=body, timeout=5)
        resp.raise_for_status()

    def test_dict_reply_surfaces_sender(self, mock_server):
        """A reply seeded with {"reply","sender"} -> Reply(text, sender)."""
        bridge = _bridge(mock_server)
        self._post_reply(mock_server, {"reply": "approve", "sender": "+15551234567"})

        reply = bridge.poll_replies_with_sender(timeout=5)
        assert isinstance(reply, Reply)
        assert reply.text == "approve"
        assert reply.sender == "+15551234567"

    def test_bare_string_reply_has_no_sender(self, mock_server):
        """Back-compat: a bare-string reply yields sender None."""
        bridge = _bridge(mock_server)
        mock_openclaw.MockOpenClawHandler.reply_queue.append("approve")

        reply = bridge.poll_replies_with_sender(timeout=5)
        assert isinstance(reply, Reply)
        assert reply.text == "approve"
        assert reply.sender is None

    def test_timeout_returns_none(self, mock_server):
        bridge = _bridge(mock_server)
        assert bridge.poll_replies_with_sender(timeout=1) is None

    def test_auto_reply_with_sender_flag(self, mock_server):
        """The --auto-reply-sender server flag attaches a sender to auto replies."""
        mock_openclaw.MockOpenClawHandler.auto_reply = "approve"
        mock_openclaw.MockOpenClawHandler.auto_reply_sender = "user@icloud.com"
        bridge = _bridge(mock_server)

        reply = bridge.poll_replies_with_sender(timeout=5)
        assert reply.text == "approve"
        assert reply.sender == "user@icloud.com"
