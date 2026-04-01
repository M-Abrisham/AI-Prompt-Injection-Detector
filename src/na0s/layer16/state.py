"""Layer 16 ConversationState — extended with mutation and query methods.

Wraps the ConversationState dataclass from models.py with methods for
adding turns, computing risk trends, and serialization.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from na0s.layer16.models import Alert, ConversationState, ConversationTurn


def add_turn(
    state: ConversationState,
    text: str,
    risk_score: float = 0.0,
    label: str = "safe",
    flags: Optional[List[str]] = None,
) -> ConversationTurn:
    """Add a turn to the conversation state (mutates in place).

    Args:
        state: The conversation state to mutate.
        text: The raw text of the turn.
        risk_score: Risk score from single-turn analysis.
        label: Classification label from single-turn analysis.
        flags: Optional list of flag strings.

    Returns:
        The newly created ConversationTurn.
    """
    turn = ConversationTurn(
        turn_id=str(uuid.uuid4()),
        text=text,
        timestamp=datetime.now(timezone.utc),
        risk_score=risk_score,
        label=label,
        flags=flags if flags is not None else [],
    )
    state.turns.append(turn)
    state.last_activity = datetime.now(timezone.utc)
    return turn


def get_window(
    state: ConversationState, n: Optional[int] = None
) -> List[ConversationTurn]:
    """Return the last N turns from the conversation.

    Args:
        state: The conversation state.
        n: Number of recent turns to return. None means all turns.

    Returns:
        List of the most recent N ConversationTurn objects.
    """
    if n is None:
        return list(state.turns)
    return list(state.turns[-n:])


def get_risk_trend(state: ConversationState) -> List[float]:
    """Return the risk score for each turn in order.

    Args:
        state: The conversation state.

    Returns:
        List of risk_score floats, one per turn.
    """
    return [t.risk_score for t in state.turns]


def is_escalating(state: ConversationState, threshold: float = 0.15) -> bool:
    """Check if the risk trend slope exceeds the threshold.

    Uses simple linear regression (least-squares) over the risk scores.
    Returns False if fewer than 2 turns exist.

    Args:
        state: The conversation state.
        threshold: Minimum slope to be considered escalating.

    Returns:
        True if the risk trend slope exceeds the threshold.
    """
    scores = get_risk_trend(state)
    n = len(scores)
    if n < 2:
        return False

    # Simple linear regression: slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    for i, y in enumerate(scores):
        x = float(i)
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return False

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope > threshold


def get_combined_text(
    state: ConversationState, last_n: Optional[int] = None
) -> str:
    """Concatenate the text of the last N turns.

    Args:
        state: The conversation state.
        last_n: Number of recent turns to include. None means all.

    Returns:
        Single string with turn texts joined by newlines.
    """
    turns = get_window(state, last_n)
    return "\n".join(t.text for t in turns)


def _turn_to_dict(turn: ConversationTurn) -> Dict[str, Any]:
    """Serialize a ConversationTurn to a plain dict."""
    return {
        "turn_id": turn.turn_id,
        "text": turn.text,
        "timestamp": turn.timestamp.isoformat(),
        "risk_score": turn.risk_score,
        "label": turn.label,
        "flags": list(turn.flags),
        "metadata": dict(turn.metadata),
    }


def _turn_from_dict(d: Dict[str, Any]) -> ConversationTurn:
    """Deserialize a ConversationTurn from a plain dict."""
    return ConversationTurn(
        turn_id=d["turn_id"],
        text=d["text"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        risk_score=d["risk_score"],
        label=d["label"],
        flags=d.get("flags", []),
        metadata=d.get("metadata", {}),
    )


def _alert_to_dict(alert: Alert) -> Dict[str, Any]:
    """Serialize an Alert to a plain dict."""
    return {
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "confidence": alert.confidence,
        "description": alert.description,
        "turn_range": list(alert.turn_range),
        "evidence": list(alert.evidence),
    }


def _alert_from_dict(d: Dict[str, Any]) -> Alert:
    """Deserialize an Alert from a plain dict."""
    return Alert(
        alert_type=d["alert_type"],
        severity=d["severity"],
        confidence=d["confidence"],
        description=d["description"],
        turn_range=tuple(d.get("turn_range", (0, 0))),
        evidence=d.get("evidence", []),
    )


def to_dict(state: ConversationState) -> Dict[str, Any]:
    """Serialize a ConversationState to a JSON-compatible dict.

    Args:
        state: The conversation state.

    Returns:
        Plain dict suitable for json.dumps().
    """
    return {
        "session_id": state.session_id,
        "turns": [_turn_to_dict(t) for t in state.turns],
        "cumulative_risk": state.cumulative_risk,
        "active_alerts": [_alert_to_dict(a) for a in state.active_alerts],
        "created_at": state.created_at.isoformat(),
        "last_activity": state.last_activity.isoformat(),
        "metadata": dict(state.metadata),
    }


def from_dict(d: Dict[str, Any]) -> ConversationState:
    """Deserialize a ConversationState from a plain dict.

    Args:
        d: Dict produced by to_dict().

    Returns:
        Reconstructed ConversationState.
    """
    return ConversationState(
        session_id=d["session_id"],
        turns=[_turn_from_dict(t) for t in d.get("turns", [])],
        cumulative_risk=d.get("cumulative_risk", 0.0),
        active_alerts=[_alert_from_dict(a) for a in d.get("active_alerts", [])],
        created_at=datetime.fromisoformat(d["created_at"]),
        last_activity=datetime.fromisoformat(d["last_activity"]),
        metadata=d.get("metadata", {}),
    )
