"""Graduated response helpers for Layer 16 threat handling."""

from __future__ import annotations

from typing import Dict, Iterable

from na0s.layer16 import config as layer16_config
from na0s.layer16.models import Alert, ConversationState, ThreatLevel, UserRiskProfile

_LEVEL_RANK = {
    ThreatLevel.NORMAL: 0,
    ThreatLevel.WATCH: 1,
    ThreatLevel.SUSPECT: 2,
    ThreatLevel.FLAGGED: 3,
    ThreatLevel.BLOCKED: 4,
}

_PROFILE_TO_LEVEL = {
    "normal": ThreatLevel.NORMAL,
    "watch": ThreatLevel.WATCH,
    "suspect": ThreatLevel.SUSPECT,
    "flagged": ThreatLevel.FLAGGED,
    "blocked": ThreatLevel.BLOCKED,
}


def _max_level(levels: Iterable[ThreatLevel]) -> ThreatLevel:
    best = ThreatLevel.NORMAL
    for level in levels:
        if _LEVEL_RANK[level] > _LEVEL_RANK[best]:
            best = level
    return best


def _severity_level(alerts: Iterable[Alert]) -> ThreatLevel:
    highest = ThreatLevel.NORMAL
    for alert in alerts:
        sev = (alert.severity or "").lower()
        if sev == "critical":
            return ThreatLevel.BLOCKED
        if sev == "high":
            highest = _max_level([highest, ThreatLevel.FLAGGED])
        elif sev == "medium":
            highest = _max_level([highest, ThreatLevel.SUSPECT])
        elif sev == "low":
            highest = _max_level([highest, ThreatLevel.WATCH])
    return highest


def _risk_level(state: ConversationState) -> ThreatLevel:
    # Conservative aggregate of session risk metrics.
    aggregate = max(
        float(getattr(state, "cumulative_risk", 0.0)),
        float(getattr(state, "peak_risk", 0.0)),
    )
    if aggregate >= layer16_config.THREAT_LEVEL_BLOCKED_RISK:
        return ThreatLevel.BLOCKED
    if aggregate >= layer16_config.THREAT_LEVEL_FLAGGED_RISK:
        return ThreatLevel.FLAGGED
    if aggregate >= layer16_config.THREAT_LEVEL_SUSPECT_RISK:
        return ThreatLevel.SUSPECT
    if aggregate >= layer16_config.THREAT_LEVEL_WATCH_RISK:
        return ThreatLevel.WATCH
    return ThreatLevel.NORMAL


def _profile_level(user_profile: UserRiskProfile | None) -> ThreatLevel:
    if user_profile is None:
        return ThreatLevel.NORMAL
    return _PROFILE_TO_LEVEL.get((user_profile.flag_level or "").lower(), ThreatLevel.NORMAL)


def compute_threat_level(
    state: ConversationState,
    alerts: list[Alert],
    user_profile: UserRiskProfile | None = None,
) -> ThreatLevel:
    """Compute final threat level from alerts + session risk + user profile."""
    return _max_level(
        [
            _severity_level(alerts or []),
            _risk_level(state),
            _profile_level(user_profile),
        ]
    )


def get_response_action(threat_level: ThreatLevel) -> Dict[str, bool]:
    """Map threat level to execution policy flags."""
    if threat_level == ThreatLevel.BLOCKED:
        return {
            "allow_response": False,
            "require_human_review": True,
            "mask_sensitive_context": True,
            "log_security_event": True,
        }
    if threat_level == ThreatLevel.FLAGGED:
        return {
            "allow_response": True,
            "require_human_review": True,
            "mask_sensitive_context": True,
            "log_security_event": True,
        }
    if threat_level == ThreatLevel.SUSPECT:
        return {
            "allow_response": True,
            "require_human_review": False,
            "mask_sensitive_context": True,
            "log_security_event": True,
        }
    if threat_level == ThreatLevel.WATCH:
        return {
            "allow_response": True,
            "require_human_review": False,
            "mask_sensitive_context": False,
            "log_security_event": True,
        }
    return {
        "allow_response": True,
        "require_human_review": False,
        "mask_sensitive_context": False,
        "log_security_event": False,
    }

