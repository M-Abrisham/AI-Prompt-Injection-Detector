"""Tests for Layer 16 public API exports.

Verifies that all key classes are importable from na0s.layer16 directly,
matching the pattern established by Layer 15.
"""

import pytest


class TestPublicImports:
    """Every symbol in __all__ must be importable from na0s.layer16."""

    def test_import_conversation_security_monitor(self):
        from na0s.layer16 import ConversationSecurityMonitor
        assert ConversationSecurityMonitor is not None

    def test_import_models(self):
        from na0s.layer16 import (
            Alert,
            ConversationState,
            ConversationTurn,
            MultiTurnAnalysis,
            SessionConfig,
        )
        assert all(cls is not None for cls in [
            Alert, ConversationState, ConversationTurn,
            MultiTurnAnalysis, SessionConfig,
        ])

    def test_import_session_manager(self):
        from na0s.layer16 import SessionManager
        assert SessionManager is not None

    def test_import_sliding_window(self):
        from na0s.layer16 import SlidingWindow
        assert SlidingWindow is not None

    def test_import_exceptions(self):
        from na0s.layer16 import (
            MaxSessionsReachedError,
            SessionExpiredError,
            SessionNotFoundError,
        )
        assert all(issubclass(exc, Exception) for exc in [
            MaxSessionsReachedError, SessionExpiredError, SessionNotFoundError,
        ])

    def test_all_list_matches_exports(self):
        import na0s.layer16 as l16
        for name in l16.__all__:
            assert hasattr(l16, name), f"{name} in __all__ but not importable"

    def test_convenience_workflow(self):
        """End-to-end: import from package, create monitor, process a turn."""
        from na0s.layer16 import ConversationSecurityMonitor, SessionConfig

        monitor = ConversationSecurityMonitor(config=SessionConfig(window_size=5))
        sid = monitor.create_session()
        analysis = monitor.process_turn("Hello", session_id=sid, risk_score=0.1)

        assert analysis.session_id == sid
        assert analysis.turn_count == 1
        assert not analysis.has_alerts
