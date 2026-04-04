"""Layer 16 Graduated Response — threat level computation.

Maps accumulated risk signals to discrete threat levels with
corresponding response actions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from na0s.layer16 import config as layer16_config
from na0s.layer16.models import (
    Alert,
    ConversationState,
    ThreatLevel,
    UserRiskProfile,
)


# Response actions per threat level
_RESPONSE_ACTIONS: Dict[ThreatLevel, Dict[str, bool]] = {
    ThreatLevel.NORMAL: {
        "allow": True,
        "log": False,
        "rate_limit": False,
        "human_review": False,
    },
    ThreatLevel.WATCH: {
        "allow": True,
        "log": True,
        "rate_limit": False,
        "human_review": False,
    },
    ThreatLevel.SUSPECT: {
        "allow": True,
        "log": True,
        "rate_limit": True,
        "human_review": False,
    },
    ThreatLevel.FLAGGED: {
        "allow": False,
        "log": True,
        "rate_limit": True,
        "human_review": True,
    },
    ThreatLevel.BLOCKED: {
        "allow": False,
        "log": True,
        "rate_limit": True,
        "human_review": True,
    },
}


def compute_threat_level(
    state: ConversationState,
    alerts: List[Alert],
    user_profile: Optional[UserRiskProfile] = None,
) -> ThreatLevel:
    """Determine the graduated threat level from conversation state and alerts.

    The highest matching level wins. Checks are ordered from most severe
    (BLOCKED) to least (WATCH), returning NORMAL as the default.

    Args:
        state: Current conversation state with accumulated metrics.
        alerts: Alerts from the current analysis pass.
        user_profile: Optional cross-session risk profile.

    Returns:
        The computed ThreatLevel.
    """
    severities = {a.severity for a in alerts} if alerts else set()

    blocked_risk = layer16_config.THREAT_LEVEL_BLOCKED_RISK
    flagged_risk = layer16_config.THREAT_LEVEL_FLAGGED_RISK
    suspect_risk = layer16_config.THREAT_LEVEL_SUSPECT_RISK
    watch_risk = layer16_config.THREAT_LEVEL_WATCH_RISK

    # Apply user risk multiplier to cumulative risk for threshold checks
    effective_risk = state.cumulative_risk
    if user_profile is not None:
        from na0s.layer16.user_risk_profile import RISK_MULTIPLIERS
        multiplier = RISK_MULTIPLIERS.get(user_profile.flag_level, 1.0)
        if multiplier == float("inf"):
            return ThreatLevel.BLOCKED
        effective_risk = min(1.0, effective_risk * multiplier)

    # BLOCKED
    if "critical" in severities:
        return ThreatLevel.BLOCKED
    if effective_risk >= blocked_risk:
        return ThreatLevel.BLOCKED
    if state.cusum_score >= 5.0:
        return ThreatLevel.BLOCKED

    # FLAGGED
    if "high" in severities:
        return ThreatLevel.FLAGGED
    if effective_risk >= flagged_risk:
        return ThreatLevel.FLAGGED
    if state.peak_risk >= 0.85:
        return ThreatLevel.FLAGGED

    # SUSPECT
    if "medium" in severities:
        return ThreatLevel.SUSPECT
    if effective_risk >= suspect_risk:
        return ThreatLevel.SUSPECT
    # Check for escalation via alerts
    if any(a.alert_type == "escalation" for a in alerts):
        return ThreatLevel.SUSPECT

    # WATCH
    if "low" in severities:
        return ThreatLevel.WATCH
    if effective_risk >= watch_risk:
        return ThreatLevel.WATCH

    return ThreatLevel.NORMAL


def get_response_action(level: ThreatLevel) -> Dict[str, bool]:
    """Return response actions appropriate for the given threat level.

    Args:
        level: The threat level.

    Returns:
        Dict with keys: allow, log, rate_limit, human_review.
    """
    return dict(_RESPONSE_ACTIONS.get(level, _RESPONSE_ACTIONS[ThreatLevel.NORMAL]))
