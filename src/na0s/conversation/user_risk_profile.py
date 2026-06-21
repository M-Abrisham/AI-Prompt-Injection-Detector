"""Layer 16 UserRiskProfileStore — cross-session risk accumulator.

Tracks per-user risk across sessions so that an attacker spreading
low-intensity probes across many sessions can be detected. All user
identifiers are stored as opaque hashes — never in plain text.
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from na0s.conversation import config as layer16_config
from na0s.conversation.models import UserRiskProfile


# Risk multipliers by flag level
RISK_MULTIPLIERS: Dict[str, float] = {
    "normal": 1.0,
    "watch": 1.2,
    "suspect": 1.5,
    "flagged": 2.0,
    "blocked": float("inf"),
}

# Thresholds for automatic flag level promotion
_FLAG_LEVEL_THRESHOLDS = [
    (0.8, "blocked"),
    (0.6, "flagged"),
    (0.4, "suspect"),
    (0.2, "watch"),
]


def _validate_user_hash(user_hash: str) -> None:
    """Validate that user_hash is a non-empty string."""
    if not isinstance(user_hash, str) or not user_hash.strip():
        raise ValueError("user_hash must be a non-empty string")


class UserRiskProfileStore:
    """Thread-safe in-memory store for cross-session user risk profiles.

    Caps at ``MAX_USER_PROFILES`` entries. When full, the oldest profile
    (by ``last_seen``) is evicted to make room.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profiles: Dict[str, UserRiskProfile] = {}

    @property
    def profile_count(self) -> int:
        with self._lock:
            return len(self._profiles)

    def get_or_create(self, user_hash: str) -> UserRiskProfile:
        """Return existing profile or create a new one.

        Args:
            user_hash: Opaque hash of user identifier.

        Returns:
            The UserRiskProfile for this user.
        """
        _validate_user_hash(user_hash)
        with self._lock:
            profile = self._profiles.get(user_hash)
            if profile is not None:
                return profile
            self._evict_if_full()
            profile = UserRiskProfile(user_hash=user_hash)
            self._profiles[user_hash] = profile
            return profile

    def get_profile(self, user_hash: str) -> Optional[UserRiskProfile]:
        """Return existing profile or None."""
        _validate_user_hash(user_hash)
        with self._lock:
            return self._profiles.get(user_hash)

    def update_from_session(
        self,
        user_hash: str,
        session_risk: float,
        technique_tags: Optional[List[str]] = None,
        was_flagged: bool = False,
    ) -> UserRiskProfile:
        """Update a user's risk profile after a session ends.

        Args:
            user_hash: Opaque hash of user identifier.
            session_risk: Final cumulative risk from the session [0.0, 1.0].
            technique_tags: Technique tags observed in the session.
            was_flagged: Whether the session triggered a flag/block.

        Returns:
            The updated UserRiskProfile.
        """
        _validate_user_hash(user_hash)

        # Sanitise session_risk
        if not isinstance(session_risk, (int, float)):
            raise ValueError("session_risk must be a number")
        if math.isnan(session_risk) or math.isinf(session_risk):
            raise ValueError("session_risk must be finite")
        session_risk = max(0.0, min(1.0, float(session_risk)))

        decay = layer16_config.USER_RISK_PROFILE_DECAY

        with self._lock:
            profile = self._profiles.get(user_hash)
            if profile is None:
                self._evict_if_full()
                profile = UserRiskProfile(user_hash=user_hash)
                self._profiles[user_hash] = profile

            # EMA update
            old_risk = profile.cumulative_risk
            new_risk = decay * old_risk + (1.0 - decay) * session_risk
            profile.cumulative_risk = max(0.0, min(1.0, new_risk))

            profile.session_count += 1
            if was_flagged:
                profile.flagged_session_count += 1

            # Merge technique fingerprints
            if technique_tags:
                max_fp = layer16_config.MAX_TECHNIQUE_FINGERPRINTS
                for tag in technique_tags:
                    if not isinstance(tag, str) or not tag:
                        continue
                    if tag in profile.technique_fingerprints:
                        profile.technique_fingerprints[tag] += 1
                    elif len(profile.technique_fingerprints) < max_fp:
                        profile.technique_fingerprints[tag] = 1

            # Update timestamps
            profile.last_seen = datetime.now(timezone.utc)

            # Auto-promote flag level based on cumulative risk
            profile.flag_level = self._compute_flag_level(profile)

            return profile

    def get_risk_multiplier(self, user_hash: str) -> float:
        """Return the risk multiplier for a user based on their flag level.

        Returns 1.0 for unknown users (no profile).
        """
        _validate_user_hash(user_hash)
        with self._lock:
            profile = self._profiles.get(user_hash)
            if profile is None:
                return 1.0
            return RISK_MULTIPLIERS.get(profile.flag_level, 1.0)

    @staticmethod
    def _compute_flag_level(profile: UserRiskProfile) -> str:
        """Determine flag level from cumulative risk and session history."""
        risk = profile.cumulative_risk
        for threshold, level in _FLAG_LEVEL_THRESHOLDS:
            if risk >= threshold:
                return level
        return "normal"

    def _evict_if_full(self) -> None:
        """Evict the oldest profile (by last_seen) if at capacity.

        Must be called while holding ``self._lock``.
        """
        max_profiles = layer16_config.MAX_USER_PROFILES
        if len(self._profiles) < max_profiles:
            return
        # Find the profile with the oldest last_seen
        oldest_hash = min(
            self._profiles,
            key=lambda h: self._profiles[h].last_seen,
        )
        del self._profiles[oldest_hash]

    def clear(self) -> None:
        """Remove all profiles."""
        with self._lock:
            self._profiles.clear()
