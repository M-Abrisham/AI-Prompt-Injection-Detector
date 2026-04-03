"""Multi-turn conversation test harness for Layer 16.

Simulates multi-turn conversations without the full Na0S pipeline.
Feeds mocked single-turn results into ConversationSecurityMonitor so
tests stay fast and isolated.
"""

from __future__ import annotations

from typing import List, Optional

from na0s.layer16.conversation_monitor import ConversationSecurityMonitor
from na0s.layer16.models import Alert, MultiTurnAnalysis, SessionConfig


class ConversationTestHarness:
    """Simulates multi-turn conversations for testing Layer 16.

    Does NOT require the full Na0S pipeline. Works with mocked
    single-turn results so tests are fast and isolated.
    """

    def __init__(self, monitor: Optional[ConversationSecurityMonitor] = None) -> None:
        if monitor is not None:
            self._monitor = monitor
        else:
            self._monitor = ConversationSecurityMonitor(config=SessionConfig())
        self._session_id = self._monitor.create_session()
        self._turn_count = 0
        self._results: List[MultiTurnAnalysis] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def send(
        self,
        text: str,
        risk_score: float = 0.1,
        label: str = "safe",
    ) -> MultiTurnAnalysis:
        """Send a single turn through the monitor and return the analysis."""
        result = self._monitor.process_turn(
            text=text,
            session_id=self._session_id,
            risk_score=risk_score,
            label=label,
        )
        self._turn_count += 1
        self._results.append(result)
        return result

    def send_sequence(self, turns: list) -> List[MultiTurnAnalysis]:
        """Send a sequence of turns and return all analyses.

        Each turn is a dict with keys: ``text``, ``risk_score``, ``label``.
        Missing keys use defaults (risk_score=0.1, label="safe").
        """
        results = []
        for turn in turns:
            result = self.send(
                text=turn["text"],
                risk_score=turn.get("risk_score", 0.1),
                label=turn.get("label", "safe"),
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """The current session identifier."""
        return self._session_id

    @property
    def turn_count(self) -> int:
        """Number of turns sent so far."""
        return self._turn_count

    # ------------------------------------------------------------------
    # Alert inspection
    # ------------------------------------------------------------------

    def alerts_triggered(self) -> bool:
        """Whether any alert was triggered during the conversation."""
        return any(r.has_alerts for r in self._results)

    def latest_alert(self) -> Optional[Alert]:
        """Return the most recent alert, or None if no alerts."""
        for result in reversed(self._results):
            if result.alerts:
                return result.alerts[-1]
        return None

    def all_alerts(self) -> List[Alert]:
        """Return all alerts across all turns."""
        alerts: List[Alert] = []
        for result in self._results:
            alerts.extend(result.alerts)
        return alerts

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Discard the current session and start a new one."""
        self._session_id = self._monitor.create_session()
        self._turn_count = 0
        self._results = []

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def assert_no_alerts(self) -> None:
        """Assert that no alerts were triggered."""
        assert not self.alerts_triggered(), (
            f"Expected no alerts but got: {self.all_alerts()}"
        )

    def assert_alert(
        self, alert_type: str, min_confidence: float = 0.0
    ) -> Alert:
        """Assert that an alert of the given type exists.

        Args:
            alert_type: The expected alert_type string.
            min_confidence: Minimum confidence threshold.

        Returns:
            The first matching Alert.
        """
        matching = [
            a
            for a in self.all_alerts()
            if a.alert_type == alert_type and a.confidence >= min_confidence
        ]
        assert matching, (
            f"No {alert_type!r} alert with confidence >= {min_confidence}. "
            f"Got: {self.all_alerts()}"
        )
        return matching[0]
