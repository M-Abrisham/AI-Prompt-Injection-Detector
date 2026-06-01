#!/usr/bin/env python3
"""Mock OpenClaw HTTP API for local testing.

Implements the same HTTP contract the OpenClawBridge speaks, so a local
dry-run exercises the real send -> poll -> reply loop:

    GET  /health           -> {"status": "ok"}
    POST /send             -> prints the iMessage, returns success
    GET  /replies          -> {"replies": [...]} and CONSUMES the queue
    POST /replies          -> seed a reply: body {"reply": "approve"}
                              or {"replies": ["approve", ...]}

The bridge polls GET /replies expecting {"replies": [...]}; earlier versions
of this mock only served POST /poll_replies, so every poll 404'd. Both are
supported now. Replies are served from an in-memory queue instead of a
hardcoded "approve", so reject/timeout paths can be exercised too.

  # always auto-approve (handy for a quick end-to-end smoke test)
  python scripts/mock_openclaw.py --auto-reply approve

  # start empty (simulates "no reply yet"); seed a reply over HTTP:
  python scripts/mock_openclaw.py
  curl -XPOST localhost:3000/replies -d '{"reply":"reject"}'
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MockOpenClawHandler(BaseHTTPRequestHandler):
    """Mock OpenClaw API endpoints.

    Shared state lives on the class so it persists across the per-request
    handler instances that HTTPServer creates:
        reply_queue: replies waiting to be served on GET /replies
        auto_reply:  if set, GET /replies always yields this reply
    """

    reply_queue: list = []
    auto_reply: str | None = None

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            return {}

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/replies":
            # Serve and consume queued replies (the bridge reads replies[0]).
            if self.auto_reply is not None:
                self._json(200, {"replies": [self.auto_reply]})
            elif MockOpenClawHandler.reply_queue:
                reply = MockOpenClawHandler.reply_queue.pop(0)
                self._json(200, {"replies": [reply]})
            else:
                self._json(200, {"replies": []})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests (iMessage send / reply seeding)."""
        if self.path in ("/send", "/send_message"):
            data = self._read_body()
            print(f"\n📱 iMessage: {data.get('text', 'no text')}\n")
            self._json(200, {"success": True, "message_id": "mock-123"})

        elif self.path == "/replies":
            # Seed a reply (or replies) for a subsequent GET /replies.
            data = self._read_body()
            if "replies" in data and isinstance(data["replies"], list):
                MockOpenClawHandler.reply_queue.extend(data["replies"])
            elif "reply" in data:
                MockOpenClawHandler.reply_queue.append(data["reply"])
            self._json(200, {"success": True, "queued": len(MockOpenClawHandler.reply_queue)})

        elif self.path == "/poll_replies":
            # Backward-compat: single-reply shape the old mock used.
            reply = self.auto_reply or (
                MockOpenClawHandler.reply_queue.pop(0)
                if MockOpenClawHandler.reply_queue
                else None
            )
            self._json(200, {"reply": reply})

        elif self.path == "/skills":
            # The bridge may register slash commands; accept and ignore.
            self._json(200, {"success": True})

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run_mock_openclaw(port: int = 3000, auto_reply: str | None = None) -> None:
    """Run mock OpenClaw server."""
    MockOpenClawHandler.auto_reply = auto_reply
    MockOpenClawHandler.reply_queue = []
    server = HTTPServer(("127.0.0.1", port), MockOpenClawHandler)
    print(f"🔌 Mock OpenClaw running at http://127.0.0.1:{port}")
    if auto_reply:
        print(f"   auto-reply mode: every poll returns '{auto_reply}'")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✋ Mock OpenClaw stopped")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock OpenClaw HTTP API")
    parser.add_argument("port", nargs="?", type=int, default=3000, help="Port (default 3000)")
    parser.add_argument(
        "--auto-reply",
        default=None,
        help="If set, every poll returns this reply (e.g. 'approve' or 'reject')",
    )
    args = parser.parse_args()
    run_mock_openclaw(args.port, args.auto_reply)
