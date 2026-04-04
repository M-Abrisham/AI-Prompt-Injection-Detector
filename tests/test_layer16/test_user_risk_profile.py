"""Tests for Layer 16 T3.1 — UserRiskProfile cross-session risk accumulator."""

from __future__ import annotations

import threading
from datetime import datetime, timezone, timedelta
from unittest import mock

import pytest

from na0s.layer16.models import UserRiskProfile
from na0s.layer16.user_risk_profile import UserRiskProfileStore


# -----------------------------------------------------------------------
# Profile creation and retrieval
# -----------------------------------------------------------------------


class TestProfileCreation:
    def test_get_or_create_new_profile(self):
        store = UserRiskProfileStore()
        profile = store.get_or_create("abc123")
        assert profile.user_hash == "abc123"
        assert profile.cumulative_risk == 0.0
        assert profile.session_count == 0
        assert profile.flag_level == "normal"

    def test_get_or_create_returns_existing(self):
        store = UserRiskProfileStore()
        p1 = store.get_or_create("abc123")
        p1.session_count = 5
        p2 = store.get_or_create("abc123")
        assert p2.session_count == 5
        assert p1 is p2

    def test_get_profile_missing_returns_none(self):
        store = UserRiskProfileStore()
        assert store.get_profile("nonexistent") is None

    def test_get_profile_existing(self):
        store = UserRiskProfileStore()
        store.get_or_create("user1")
        assert store.get_profile("user1") is not None

    def test_invalid_user_hash_empty(self):
        store = UserRiskProfileStore()
        with pytest.raises(ValueError, match="non-empty"):
            store.get_or_create("")

    def test_invalid_user_hash_whitespace(self):
        store = UserRiskProfileStore()
        with pytest.raises(ValueError, match="non-empty"):
            store.get_or_create("   ")

    def test_invalid_user_hash_not_string(self):
        store = UserRiskProfileStore()
        with pytest.raises(ValueError):
            store.get_or_create(12345)  # type: ignore[arg-type]


# -----------------------------------------------------------------------
# EMA update across sessions
# -----------------------------------------------------------------------


class TestEMAUpdate:
    def test_single_session_update(self):
        store = UserRiskProfileStore()
        profile = store.update_from_session("user1", session_risk=0.5)
        # EMA: 0.7 * 0.0 + 0.3 * 0.5 = 0.15
        assert abs(profile.cumulative_risk - 0.15) < 1e-9
        assert profile.session_count == 1

    def test_multiple_session_ema(self):
        store = UserRiskProfileStore()
        store.update_from_session("user1", session_risk=0.5)
        profile = store.update_from_session("user1", session_risk=0.8)
        # Step 1: 0.7*0.0 + 0.3*0.5 = 0.15
        # Step 2: 0.7*0.15 + 0.3*0.8 = 0.105 + 0.24 = 0.345
        assert abs(profile.cumulative_risk - 0.345) < 1e-9
        assert profile.session_count == 2

    def test_ema_clamped_to_unit_interval(self):
        store = UserRiskProfileStore()
        # Even with repeated high-risk sessions, risk stays <= 1.0
        for _ in range(100):
            profile = store.update_from_session("user1", session_risk=1.0)
        assert profile.cumulative_risk <= 1.0
        assert profile.cumulative_risk >= 0.0

    def test_ema_with_zero_risk(self):
        store = UserRiskProfileStore()
        store.update_from_session("user1", session_risk=0.8)
        # Decay towards zero with zero-risk sessions
        for _ in range(20):
            profile = store.update_from_session("user1", session_risk=0.0)
        assert profile.cumulative_risk < 0.05

    def test_session_risk_nan_rejected(self):
        store = UserRiskProfileStore()
        with pytest.raises(ValueError, match="finite"):
            store.update_from_session("user1", session_risk=float("nan"))

    def test_session_risk_inf_rejected(self):
        store = UserRiskProfileStore()
        with pytest.raises(ValueError, match="finite"):
            store.update_from_session("user1", session_risk=float("inf"))

    def test_session_risk_clamped_high(self):
        store = UserRiskProfileStore()
        profile = store.update_from_session("user1", session_risk=5.0)
        # Clamped to 1.0 before EMA: 0.7*0 + 0.3*1.0 = 0.3
        assert abs(profile.cumulative_risk - 0.3) < 1e-9

    def test_session_risk_clamped_low(self):
        store = UserRiskProfileStore()
        profile = store.update_from_session("user1", session_risk=-2.0)
        # Clamped to 0.0: 0.7*0 + 0.3*0 = 0.0
        assert profile.cumulative_risk == 0.0

    def test_flagged_session_counter(self):
        store = UserRiskProfileStore()
        store.update_from_session("user1", session_risk=0.1, was_flagged=False)
        store.update_from_session("user1", session_risk=0.5, was_flagged=True)
        store.update_from_session("user1", session_risk=0.3, was_flagged=True)
        profile = store.get_profile("user1")
        assert profile.flagged_session_count == 2
        assert profile.session_count == 3


# -----------------------------------------------------------------------
# Technique fingerprint accumulation
# -----------------------------------------------------------------------


class TestTechniqueFingerprints:
    def test_fingerprints_accumulated(self):
        store = UserRiskProfileStore()
        store.update_from_session("u1", session_risk=0.3, technique_tags=["jailbreak", "obfuscation"])
        store.update_from_session("u1", session_risk=0.4, technique_tags=["jailbreak", "prompt_leak"])
        profile = store.get_profile("u1")
        assert profile.technique_fingerprints["jailbreak"] == 2
        assert profile.technique_fingerprints["obfuscation"] == 1
        assert profile.technique_fingerprints["prompt_leak"] == 1

    def test_fingerprints_cap(self):
        store = UserRiskProfileStore()
        # Create MAX_TECHNIQUE_FINGERPRINTS unique tags
        with mock.patch("na0s.layer16.config.MAX_TECHNIQUE_FINGERPRINTS", 5):
            tags = [f"tag_{i}" for i in range(10)]
            store.update_from_session("u1", session_risk=0.1, technique_tags=tags)
            profile = store.get_profile("u1")
            assert len(profile.technique_fingerprints) == 5

    def test_empty_tags_ignored(self):
        store = UserRiskProfileStore()
        store.update_from_session("u1", session_risk=0.1, technique_tags=["", None, "valid"])  # type: ignore
        profile = store.get_profile("u1")
        assert "valid" in profile.technique_fingerprints
        assert "" not in profile.technique_fingerprints

    def test_none_tags(self):
        store = UserRiskProfileStore()
        profile = store.update_from_session("u1", session_risk=0.1, technique_tags=None)
        assert profile.technique_fingerprints == {}


# -----------------------------------------------------------------------
# Risk multiplier by flag level
# -----------------------------------------------------------------------


class TestRiskMultiplier:
    def test_normal_multiplier(self):
        store = UserRiskProfileStore()
        store.get_or_create("u1")
        assert store.get_risk_multiplier("u1") == 1.0

    def test_unknown_user_multiplier(self):
        store = UserRiskProfileStore()
        assert store.get_risk_multiplier("unknown") == 1.0

    def test_watch_multiplier(self):
        store = UserRiskProfileStore()
        profile = store.get_or_create("u1")
        profile.flag_level = "watch"
        assert store.get_risk_multiplier("u1") == 1.2

    def test_suspect_multiplier(self):
        store = UserRiskProfileStore()
        profile = store.get_or_create("u1")
        profile.flag_level = "suspect"
        assert store.get_risk_multiplier("u1") == 1.5

    def test_flagged_multiplier(self):
        store = UserRiskProfileStore()
        profile = store.get_or_create("u1")
        profile.flag_level = "flagged"
        assert store.get_risk_multiplier("u1") == 2.0

    def test_blocked_multiplier(self):
        store = UserRiskProfileStore()
        profile = store.get_or_create("u1")
        profile.flag_level = "blocked"
        assert store.get_risk_multiplier("u1") == float("inf")

    def test_auto_promotion_to_watch(self):
        store = UserRiskProfileStore()
        # Push cumulative risk above 0.2 threshold
        # Need repeated high-risk sessions
        for _ in range(10):
            store.update_from_session("u1", session_risk=0.8)
        profile = store.get_profile("u1")
        assert profile.flag_level != "normal"
        assert store.get_risk_multiplier("u1") > 1.0


# -----------------------------------------------------------------------
# Max profiles cap (eviction)
# -----------------------------------------------------------------------


class TestEviction:
    def test_eviction_when_full(self):
        with mock.patch("na0s.layer16.config.MAX_USER_PROFILES", 3):
            store = UserRiskProfileStore()
            # Create 3 profiles with staggered times
            p1 = store.get_or_create("oldest")
            p1.last_seen = datetime(2020, 1, 1, tzinfo=timezone.utc)

            p2 = store.get_or_create("middle")
            p2.last_seen = datetime(2021, 1, 1, tzinfo=timezone.utc)

            p3 = store.get_or_create("newest")
            p3.last_seen = datetime(2022, 1, 1, tzinfo=timezone.utc)

            assert store.profile_count == 3

            # Adding a 4th should evict "oldest"
            store.get_or_create("newcomer")
            assert store.profile_count == 3
            assert store.get_profile("oldest") is None
            assert store.get_profile("newcomer") is not None

    def test_eviction_via_update(self):
        with mock.patch("na0s.layer16.config.MAX_USER_PROFILES", 2):
            store = UserRiskProfileStore()
            p1 = store.get_or_create("old")
            p1.last_seen = datetime(2020, 1, 1, tzinfo=timezone.utc)

            store.get_or_create("new")
            # Updating a non-existent user triggers creation+eviction
            store.update_from_session("brand_new", session_risk=0.1)
            assert store.profile_count == 2
            assert store.get_profile("old") is None


# -----------------------------------------------------------------------
# Thread safety
# -----------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_updates(self):
        store = UserRiskProfileStore()
        errors = []
        n_threads = 10
        n_updates = 50

        def worker(thread_id):
            try:
                for i in range(n_updates):
                    user = f"user_{thread_id}"
                    store.update_from_session(
                        user,
                        session_risk=0.5,
                        technique_tags=[f"tag_{i}"],
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Each thread creates one user profile
        assert store.profile_count == n_threads
        for i in range(n_threads):
            profile = store.get_profile(f"user_{i}")
            assert profile is not None
            assert profile.session_count == n_updates

    def test_concurrent_get_or_create(self):
        store = UserRiskProfileStore()
        results = []

        def worker():
            p = store.get_or_create("shared_user")
            results.append(id(p))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same profile object
        assert len(set(results)) == 1


# -----------------------------------------------------------------------
# Clear
# -----------------------------------------------------------------------


class TestClear:
    def test_clear_removes_all(self):
        store = UserRiskProfileStore()
        for i in range(5):
            store.get_or_create(f"user_{i}")
        assert store.profile_count == 5
        store.clear()
        assert store.profile_count == 0
