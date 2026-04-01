"""Shared fixtures for Layer 16 multi-turn tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from na0s.layer16.models import SessionConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def session_config():
    return SessionConfig(window_size=10, ttl_seconds=60)


@pytest.fixture
def monitor(session_config):
    try:
        from na0s.layer16.conversation_monitor import ConversationSecurityMonitor

        return ConversationSecurityMonitor(config=session_config)
    except Exception:
        pytest.skip("ConversationSecurityMonitor not available")


@pytest.fixture
def harness(monitor):
    from na0s.layer16.testing.conversation_harness import ConversationTestHarness

    return ConversationTestHarness(monitor)
