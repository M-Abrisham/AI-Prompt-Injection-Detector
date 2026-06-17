"""Layer 16 ConversationState — extended with mutation and query methods.

Wraps the ConversationState dataclass from models.py with methods for
adding turns, computing risk trends, and serialization.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

MAX_TURNS_PER_SESSION = 500
MAX_TURN_TEXT_LENGTH = 50_000

# Always retain the most recent N turns on eviction — covers the escalation /
# payload-assembly windows so in-flight multi-turn context is never dropped.
_RECENT_ALWAYS_RETAIN = 50

from na0s.layer16.models import Alert, ConversationState, ConversationTurn


def _evict_to_cap(turns: List[ConversationTurn], cap: int) -> List[ConversationTurn]:
    """Risk-weighted eviction down to *cap* turns (G10).

    Plain FIFO tail-slicing lets an attacker flood a session with benign turns
    to push an earlier SUSPICIOUS turn out of the window.  Instead we always
    keep the most recent ``_RECENT_ALWAYS_RETAIN`` turns and fill the remaining
    slots with the HIGHEST-RISK older turns, preserving chronological order so
    escalation-slope and contiguous-assembly detectors still see a valid trend.
    """
    if len(turns) <= cap:
        return turns
    recent = turns[-_RECENT_ALWAYS_RETAIN:]
    older = turns[:-_RECENT_ALWAYS_RETAIN]
    slots = cap - len(recent)
    if slots <= 0:
        return turns[-cap:]
    # Pick the highest-risk older turns, then restore chronological order.
    keep_idx = sorted(
        sorted(range(len(older)), key=lambda i: older[i].risk_score, reverse=True)[:slots]
    )
    return [older[i] for i in keep_idx] + recent


def add_turn(
    state: ConversationState,
    text: str,
    risk_score: float = 0.0,
    label: str = "safe",
    flags: Optional[List[str]] = None,
    role: str = "user",
) -> ConversationTurn:
    """Add a turn to the conversation state (mutates in place).

    Args:
        state: The conversation state to mutate.
        text: The raw text of the turn.
        risk_score: Risk score from single-turn analysis.
        label: Classification label from single-turn analysis.
        flags: Optional list of flag strings.
        role: Role of the speaker ("user", "assistant", or "system").

    Returns:
        The newly created ConversationTurn.
    """
    # --- Input validation (T1.6) ---
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    text = text[:MAX_TURN_TEXT_LENGTH]
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    if not isinstance(risk_score, (int, float)) or math.isnan(risk_score) or math.isinf(risk_score) or risk_score < 0.0 or risk_score > 1.0:
        raise ValueError("risk_score must be between 0.0 and 1.0")
    if role not in ("user", "assistant", "system"):
        raise ValueError("role must be one of: 'user', 'assistant', 'system'")

    turn = ConversationTurn(
        turn_id=str(uuid.uuid4()),
        text=text,
        role=role,
        timestamp=datetime.now(timezone.utc),
        risk_score=risk_score,
        label=label,
        flags=flags if flags is not None else [],
    )
    state.turns.append(turn)
    if len(state.turns) > MAX_TURNS_PER_SESSION:
        state.turns = _evict_to_cap(state.turns, MAX_TURNS_PER_SESSION)
    state.last_activity = datetime.now(timezone.utc)

    # Update cumulative risk: EMA with decay, capped at [0.0, 1.0].
    # Older risk decays by (1 - alpha), new turn risk added with weight alpha.
    update_cumulative_risk(state, risk_score)

    # Update peak risk tracker (T2.1).
    state.peak_risk = max(state.peak_risk, risk_score)

    # Update CUSUM accumulator (T2.2).
    update_cusum(state, risk_score)

    return turn


def update_cumulative_risk(
    state: ConversationState,
    turn_risk: float,
    decay: float = 0.85,
    alpha: float = 0.3,
) -> float:
    """Update the cumulative risk on *state* after a new turn.

    Uses an exponential moving average approach:
        new_cumulative = decay * old_cumulative + alpha * turn_risk

    The result is clamped to [0.0, 1.0] so it never grows unbounded.

    Args:
        state: Conversation state to mutate.
        turn_risk: Risk score of the newly added turn (0.0-1.0).
        decay: Retention factor for prior cumulative risk (0 < decay < 1).
        alpha: Weight given to the new turn's risk score.

    Returns:
        The updated cumulative_risk value.
    """
    # --- Input validation (T1.6) ---
    if not isinstance(turn_risk, (int, float)) or math.isnan(turn_risk) or math.isinf(turn_risk) or turn_risk < 0.0 or turn_risk > 1.0:
        raise ValueError("turn_risk must be between 0.0 and 1.0")
    if not isinstance(decay, (int, float)) or decay < 0.0 or decay > 1.0:
        raise ValueError("decay must be between 0.0 and 1.0")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("alpha must be > 0")

    raw = decay * state.cumulative_risk + alpha * turn_risk
    state.cumulative_risk = max(0.0, min(1.0, raw))
    return state.cumulative_risk


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


def compute_peak_accumulation(
    state: ConversationState,
    risk_threshold: float = 0.3,
) -> float:
    """Peak + Accumulation conversation-level risk score.

    Combines three signals:
    1. Peak score -- max single-turn risk in the conversation
    2. Persistence ratio -- fraction of turns exceeding risk_threshold
    3. Category diversity -- number of distinct labels (non-"safe") seen

    Formula: 0.4 * peak + 0.35 * persistence + 0.25 * diversity_normalized
    Returns value in [0.0, 1.0].
    """
    if not state.turns:
        return 0.0

    peak = state.peak_risk

    # Persistence: fraction of turns above the risk threshold.
    above = sum(1 for t in state.turns if t.risk_score >= risk_threshold)
    persistence = above / len(state.turns)

    # Category diversity: distinct non-"safe" labels, normalized by 5.
    distinct_labels = {t.label for t in state.turns if t.label != "safe"}
    diversity_normalized = min(1.0, len(distinct_labels) / 5.0)

    raw = 0.4 * peak + 0.35 * persistence + 0.25 * diversity_normalized
    return max(0.0, min(1.0, raw))


def update_cusum(
    state: ConversationState,
    turn_risk: float,
    baseline_mean: float = 0.1,
    allowance: float = 0.05,
) -> float:
    """Update CUSUM score for change-point detection.

    Accumulates evidence of a sustained shift above baseline.
    Uses the standard one-sided CUSUM formula:
        S_n = max(0, S_{n-1} + (x_n - mu_0 - k))

    The score is capped at 10.0 as a practical bound to prevent
    unbounded growth in very long conversations.

    Returns the updated cusum_score.
    """
    raw = state.cusum_score + (turn_risk - baseline_mean - allowance)
    state.cusum_score = min(10.0, max(0.0, raw))
    return state.cusum_score


def is_cusum_alert(state: ConversationState, threshold: float = 1.0) -> bool:
    """Return True if CUSUM score exceeds the change-detection threshold."""
    return state.cusum_score > threshold


def _turn_to_dict(turn: ConversationTurn) -> Dict[str, Any]:
    """Serialize a ConversationTurn to a plain dict."""
    return {
        "turn_id": turn.turn_id,
        "text": turn.text,
        "role": turn.role,
        "timestamp": turn.timestamp.isoformat(),
        "risk_score": turn.risk_score,
        "label": turn.label,
        "flags": list(turn.flags),
        "metadata": dict(turn.metadata),
    }


def _turn_from_dict(d: Dict[str, Any]) -> ConversationTurn:
    """Deserialize a ConversationTurn from a plain dict."""
    if not isinstance(d.get("risk_score"), (int, float)):
        raise ValueError("risk_score must be a number")
    if not isinstance(d.get("text"), str):
        raise ValueError("text must be a string")
    if not isinstance(d.get("label"), str):
        raise ValueError("label must be a string")
    return ConversationTurn(
        turn_id=d["turn_id"],
        text=d["text"],
        role=d.get("role", "user"),
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
        "peak_risk": state.peak_risk,
        "cusum_score": state.cusum_score,
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
    for field in ("cumulative_risk", "peak_risk", "cusum_score"):
        val = d.get(field, 0.0)
        if not isinstance(val, (int, float)):
            raise ValueError(f"{field} must be a number")
    return ConversationState(
        session_id=d["session_id"],
        turns=[_turn_from_dict(t) for t in d.get("turns", [])],
        cumulative_risk=d.get("cumulative_risk", 0.0),
        peak_risk=d.get("peak_risk", 0.0),
        cusum_score=d.get("cusum_score", 0.0),
        active_alerts=[_alert_from_dict(a) for a in d.get("active_alerts", [])],
        created_at=datetime.fromisoformat(d["created_at"]),
        last_activity=datetime.fromisoformat(d["last_activity"]),
        metadata=d.get("metadata", {}),
    )
