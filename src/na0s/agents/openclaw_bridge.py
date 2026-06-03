"""OpenClaw iMessage bridge for agent communication.

Wraps OpenClaw's local HTTP API to send/receive iMessages and register
slash commands for user interaction from phone/Mac.

Supports three modes:
- "auto": Try real OpenClaw first, fallback to mock if unavailable
- "mock": Always use mock (for testing)
- "real": Use only real OpenClaw (error if unavailable)
"""

import json
import time
from typing import Optional, Callable, Dict, Any, Literal
from datetime import datetime
import requests
import logging
import os
from pathlib import Path
import subprocess

from na0s import config

logger = logging.getLogger(__name__)


class OpenClawBridge:
    """Interface to OpenClaw for iMessage-based agent control."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        mode: Optional[Literal["auto", "mock", "real"]] = None,
    ):
        """Initialize OpenClaw bridge.

        Args:
            base_url: OpenClaw local API endpoint (default from config)
            timeout: HTTP request timeout in seconds (default from config)
            mode: Operation mode ("auto", "mock", "real")
                - "auto": Try real OpenClaw first, fallback to mock
                - "mock": Always use mock (for testing)
                - "real": Use only real OpenClaw (error if unavailable)
                (default from config: OPENCLAW_MODE)
        """
        self.base_url = base_url or config.OPENCLAW_BASE_URL
        self.timeout = timeout if timeout is not None else config.OPENCLAW_TIMEOUT
        self.mode = mode or config.OPENCLAW_MODE
        self.active_mode = None
        self._detect_openclaw_mode()

    def _detect_openclaw_mode(self) -> None:
        """Detect available OpenClaw mode and log which is active.

        Updates self.active_mode to either "real" or "mock".
        """
        if self.mode == "mock":
            self.active_mode = "mock"
            logger.info("OpenClaw bridge in MOCK mode (testing)")
            return

        # Try to detect real OpenClaw
        real_available = self._check_real_openclaw()

        if self.mode == "real":
            if real_available:
                self.active_mode = "real"
                logger.info("OpenClaw bridge in REAL mode")
            else:
                raise RuntimeError(
                    "OpenClaw mode set to 'real' but real OpenClaw not available. "
                    "Check if OpenClaw is installed and running."
                )
        elif self.mode == "auto":
            if real_available:
                self.active_mode = "real"
                logger.info("OpenClaw bridge in AUTO mode → using REAL OpenClaw")
            else:
                self.active_mode = "mock"
                logger.info("OpenClaw bridge in AUTO mode → fallback to MOCK OpenClaw")

    def _check_real_openclaw(self) -> bool:
        """Check if real OpenClaw is available.

        Returns True if:
        1. OpenClaw HTTP API is responding to /health, OR
        2. OpenClaw app is installed (macOS)
        """
        # Try HTTP health check
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.debug("Real OpenClaw detected via HTTP health check")
                return True
        except Exception:
            pass

        # Check if OpenClaw app is installed (macOS)
        try:
            app_path = Path("/Applications/OpenClaw.app")
            if app_path.exists():
                logger.debug("Real OpenClaw app detected at /Applications/OpenClaw.app")
                return True
        except Exception:
            pass

        return False

    def _verify_connection(self) -> None:
        """Verify OpenClaw API is reachable (for backward compat)."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            logger.debug("OpenClaw bridge connected")
        except Exception as e:
            logger.debug(f"OpenClaw connection check: {e}")

    def send_message(self, text: str) -> bool:
        """Send iMessage via OpenClaw.

        Args:
            text: Message body (plain text or markdown)

        Returns:
            True if sent successfully
        """
        if self.active_mode == "real":
            return self._send_message_real(text)
        else:
            return self._send_message_mock(text)

    def _send_message_real(self, text: str) -> bool:
        """Send iMessage via real OpenClaw service.

        Args:
            text: Message body

        Returns:
            True if sent successfully
        """
        try:
            # Try using OpenClaw SDK first if available
            try:
                from openclaw import OpenClawClient

                client = OpenClawClient()
                client.send_message(text, channel="imessage")
                logger.info(f"Sent iMessage via OpenClaw SDK: {text[:50]}...")
                return True
            except (ImportError, AttributeError):
                # Fallback to HTTP API
                pass

            # Use HTTP API for real OpenClaw
            payload = {
                "type": "message",
                "channel": "imessage",
                "text": text,
                "timestamp": datetime.now().isoformat(),
            }
            resp = requests.post(
                f"{self.base_url}/send",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.info(f"Sent iMessage via real OpenClaw: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send iMessage via real OpenClaw: {e}")
            # In auto mode, try fallback to mock
            if self.mode == "auto":
                logger.info("Falling back to mock OpenClaw")
                return self._send_message_mock(text)
            return False

    def _send_message_mock(self, text: str) -> bool:
        """Send iMessage via mock OpenClaw (for testing).

        Args:
            text: Message body

        Returns:
            True if sent successfully
        """
        try:
            payload = {
                "type": "message",
                "channel": "imessage",
                "text": text,
                "timestamp": datetime.now().isoformat(),
            }
            resp = requests.post(
                f"{self.base_url}/send",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.info(f"Sent iMessage via mock OpenClaw: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send iMessage via mock OpenClaw: {e}")
            return False

    def poll_replies(self, timeout: int = 300) -> Optional[str]:
        """Poll for user reply to iMessage.

        Blocks until user replies or timeout expires.

        Args:
            timeout: Wait time in seconds (default 5 min)

        Returns:
            User's reply text, or None if timeout
        """
        if self.active_mode == "real":
            return self._poll_replies_real(timeout)
        else:
            return self._poll_replies_mock(timeout)

    def _poll_replies_real(self, timeout: int = 300) -> Optional[str]:
        """Poll for user reply via real OpenClaw service.

        Args:
            timeout: Wait time in seconds

        Returns:
            User's reply text, or None if timeout
        """
        deadline = time.time() + timeout
        poll_interval = 2

        try:
            # Try OpenClaw SDK first if available
            try:
                from openclaw import OpenClawClient

                client = OpenClawClient()
                reply = client.poll_replies(timeout=timeout)
                if reply:
                    logger.info(f"User replied via OpenClaw SDK: {reply}")
                    return reply
            except (ImportError, AttributeError):
                # Fallback to HTTP API
                pass

            # Use HTTP API for real OpenClaw
            while time.time() < deadline:
                try:
                    resp = requests.get(
                        f"{self.base_url}/replies",
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    if data.get("replies"):
                        reply = data["replies"][0]
                        logger.info(f"User replied via real OpenClaw: {reply}")
                        return reply

                    time.sleep(poll_interval)
                except Exception as e:
                    logger.error(f"Error polling replies from real OpenClaw: {e}")
                    time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Error in real OpenClaw poll: {e}")
            # In auto mode, try fallback to mock
            if self.mode == "auto":
                logger.info("Falling back to mock OpenClaw for poll")
                return self._poll_replies_mock(int(timeout - (time.time() - (deadline - timeout))))

        logger.warning(f"No reply within {timeout}s")
        return None

    def _poll_replies_mock(self, timeout: int = 300) -> Optional[str]:
        """Poll for user reply via mock OpenClaw (for testing).

        Args:
            timeout: Wait time in seconds

        Returns:
            User's reply text, or None if timeout
        """
        deadline = time.time() + timeout
        poll_interval = 2

        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{self.base_url}/replies",
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("replies"):
                    reply = data["replies"][0]
                    logger.info(f"User replied via mock OpenClaw: {reply}")
                    return reply
                elif data.get("reply"):
                    # Handle single reply format
                    reply = data["reply"]
                    logger.info(f"User replied via mock OpenClaw: {reply}")
                    return reply

                time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"Error polling replies from mock OpenClaw: {e}")
                time.sleep(poll_interval)

        logger.warning(f"No reply within {timeout}s")
        return None

    def register_skill(
        self,
        name: str,
        description: str,
        handler: Callable[[str], str],
    ) -> bool:
        """Register a slash command skill with OpenClaw.

        When user types /name in iMessage, handler is invoked.

        Args:
            name: Slash command name (e.g., "approve")
            description: Help text shown in OpenClaw UI
            handler: Function that takes (args: str) -> response: str

        Returns:
            True if registered successfully
        """
        try:
            payload = {
                "type": "skill",
                "name": name,
                "description": description,
                "handler_key": f"agent.{name}",  # Key for local handler routing
            }
            resp = requests.post(
                f"{self.base_url}/skills",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.info(f"Registered skill: /{name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register skill {name}: {e}")
            return False

    def send_message_with_approval(
        self,
        message: str,
        timeout: int = 600,
        expected_replies: Optional[list] = None,
    ) -> Optional[str]:
        """Send message and wait for specific approval.

        Args:
            message: iMessage to send
            timeout: How long to wait for reply (seconds)
            expected_replies: List of acceptable replies (e.g., ["approve", "reject"])

        Returns:
            User's reply if matched, None if timeout or unexpected reply
        """
        if not self.send_message(message):
            return None

        while True:
            reply = self.poll_replies(timeout=min(timeout, 300))
            if reply is None:
                return None

            reply_lower = reply.lower().strip()
            if expected_replies is None or reply_lower in expected_replies:
                return reply_lower

            logger.info(f"Unexpected reply: {reply}. Waiting for valid response...")
            self.send_message(
                f"I didn't understand that. Please reply with one of: {', '.join(expected_replies or ['yes', 'no'])}"
            )
