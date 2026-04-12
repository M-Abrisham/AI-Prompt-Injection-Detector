"""Tests for L10 Canary Token features.

Covers: SessionCanaryManager, RotatingCanaryManager, PersistentCanaryStore,
CanaryAlertManager, HoneypotManager.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from na0s.canary import CanaryManager, CanaryToken
from na0s.canary_alert import CanaryAlertManager
from na0s.canary_honeypot import HoneypotManager
from na0s.canary_persistence import PersistentCanaryStore
from na0s.canary_rotation import RotatingCanaryManager
from na0s.canary_session import SessionCanaryManager


# ===================================================================
# SessionCanaryManager
# ===================================================================

class TestSessionCanaryManager:
    """Tests for per-conversation canary sessions."""

    def test_create_session_returns_modified_prompt(self):
        mgr = SessionCanaryManager()
        prompt, canary = mgr.create_session("sess-001", "You are helpful.")
        assert canary.token in prompt
        assert "You are helpful." in prompt

    def test_create_session_uses_session_prefix(self):
        mgr = SessionCanaryManager()
        _, canary = mgr.create_session("abcd-1234", "Hello")
        assert canary.token.startswith("ABCD-")

    def test_create_session_short_id_prefix(self):
        mgr = SessionCanaryManager()
        _, canary = mgr.create_session("xy", "Hello")
        assert canary.token.startswith("XY-")

    def test_get_session_returns_canary(self):
        mgr = SessionCanaryManager()
        _, canary = mgr.create_session("s1", "prompt")
        result = mgr.get_session("s1")
        assert result is canary

    def test_get_session_missing_returns_none(self):
        mgr = SessionCanaryManager()
        assert mgr.get_session("nonexistent") is None

    def test_get_session_expired_returns_none(self):
        mgr = SessionCanaryManager(default_ttl_seconds=1)
        mgr.create_session("s1", "prompt")
        # Manually expire
        mgr._sessions["s1"]["expires_at"] = time.time() - 1
        assert mgr.get_session("s1") is None

    def test_check_session_output_detects_leak(self):
        mgr = SessionCanaryManager()
        _, canary = mgr.create_session("sess-A", "prompt")
        results = mgr.check_session_output(f"Here is the key: {canary.token}")
        assert len(results) == 1
        assert results[0]["session_id"] == "sess-A"
        assert results[0]["triggered"] is True
        assert results[0]["canary_token"] == canary.token

    def test_check_session_output_clean(self):
        mgr = SessionCanaryManager()
        mgr.create_session("sess-B", "prompt")
        results = mgr.check_session_output("Normal response")
        assert results == []

    def test_check_session_output_skips_expired(self):
        mgr = SessionCanaryManager(default_ttl_seconds=1)
        _, canary = mgr.create_session("sess-C", "prompt")
        mgr._sessions["sess-C"]["expires_at"] = time.time() - 1
        results = mgr.check_session_output(canary.token)
        assert results == []

    def test_check_session_output_multiple_sessions(self):
        mgr = SessionCanaryManager()
        _, c1 = mgr.create_session("s1", "p1")
        _, c2 = mgr.create_session("s2", "p2")
        results = mgr.check_session_output(f"{c1.token} and {c2.token}")
        assert len(results) == 2
        session_ids = {r["session_id"] for r in results}
        assert session_ids == {"s1", "s2"}

    def test_cleanup_expired(self):
        mgr = SessionCanaryManager(default_ttl_seconds=1)
        mgr.create_session("s1", "p")
        mgr.create_session("s2", "p")
        mgr._sessions["s1"]["expires_at"] = time.time() - 1
        removed = mgr.cleanup_expired()
        assert removed == 1
        assert "s1" not in mgr._sessions
        assert "s2" in mgr._sessions

    def test_cleanup_expired_none_expired(self):
        mgr = SessionCanaryManager()
        mgr.create_session("s1", "p")
        assert mgr.cleanup_expired() == 0

    def test_custom_ttl_per_session(self):
        mgr = SessionCanaryManager(default_ttl_seconds=3600)
        mgr.create_session("s1", "p", ttl=10)
        entry = mgr._sessions["s1"]
        assert entry["expires_at"] <= time.time() + 11

    @patch.dict(os.environ, {"NA0S_CANARY_SESSION": "1"})
    def test_is_enabled_true(self):
        assert SessionCanaryManager.is_enabled() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_enabled_false_default(self):
        os.environ.pop("NA0S_CANARY_SESSION", None)
        assert SessionCanaryManager.is_enabled() is False

    def test_session_leak_attribution(self):
        """End-to-end: identify which session leaked."""
        mgr = SessionCanaryManager()
        _, c1 = mgr.create_session("user-alice", "prompt-alice")
        _, c2 = mgr.create_session("user-bob", "prompt-bob")
        # Only Bob's canary leaks
        results = mgr.check_session_output(f"leaked: {c2.token}")
        assert len(results) == 1
        assert results[0]["session_id"] == "user-bob"


# ===================================================================
# RotatingCanaryManager
# ===================================================================

class TestRotatingCanaryManager:
    """Tests for canary rotation."""

    def test_get_or_rotate_creates_on_first_call(self):
        mgr = RotatingCanaryManager()
        prompt, canary = mgr.get_or_rotate("System prompt")
        assert canary is not None
        assert canary.token in prompt

    def test_get_or_rotate_reuses_within_interval(self):
        mgr = RotatingCanaryManager(rotation_interval_seconds=3600)
        _, c1 = mgr.get_or_rotate("Prompt")
        _, c2 = mgr.get_or_rotate("Prompt")
        assert c1.token == c2.token

    def test_get_or_rotate_rotates_after_interval(self):
        mgr = RotatingCanaryManager(rotation_interval_seconds=1)
        _, c1 = mgr.get_or_rotate("Prompt")
        mgr._active_created_at = time.time() - 2  # force expiry
        _, c2 = mgr.get_or_rotate("Prompt")
        assert c1.token != c2.token

    def test_force_rotate_always_creates_new(self):
        mgr = RotatingCanaryManager()
        _, c1 = mgr.get_or_rotate("Prompt")
        _, c2 = mgr.force_rotate("Prompt")
        assert c1.token != c2.token

    def test_force_rotate_retires_old(self):
        mgr = RotatingCanaryManager()
        _, c1 = mgr.get_or_rotate("Prompt")
        mgr.force_rotate("Prompt")
        assert c1 in mgr._retired

    def test_check_output_detects_active(self):
        mgr = RotatingCanaryManager()
        _, canary = mgr.get_or_rotate("Prompt")
        triggered = mgr.check_output(canary.token)
        assert canary in triggered

    def test_check_output_detects_in_retired(self):
        mgr = RotatingCanaryManager()
        _, old = mgr.get_or_rotate("Prompt")
        mgr.force_rotate("Prompt")
        triggered = mgr.check_output(old.token)
        assert old in triggered

    def test_check_output_empty_text(self):
        mgr = RotatingCanaryManager()
        mgr.get_or_rotate("Prompt")
        assert mgr.check_output("") == []

    def test_history_includes_all(self):
        mgr = RotatingCanaryManager()
        mgr.get_or_rotate("P")
        mgr.force_rotate("P")
        mgr.force_rotate("P")
        h = mgr.history()
        assert len(h) == 3
        statuses = [e["status"] for e in h]
        assert statuses.count("retired") == 2
        assert statuses.count("active") == 1

    @patch.dict(os.environ, {"NA0S_CANARY_ROTATION": "1"})
    def test_is_enabled_true(self):
        assert RotatingCanaryManager.is_enabled() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_enabled_false(self):
        os.environ.pop("NA0S_CANARY_ROTATION", None)
        assert RotatingCanaryManager.is_enabled() is False


# ===================================================================
# PersistentCanaryStore
# ===================================================================

class TestPersistentCanaryStore:
    """Tests for canary persistence."""

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")
            store = PersistentCanaryStore(path=path)
            mgr = CanaryManager()
            mgr.generate(prefix="TEST", length=8)
            mgr.generate(prefix="TEST2", length=12)

            store.save(mgr)
            loaded = store.load()

            assert len(loaded.active_canaries) == 2
            tokens_orig = {c.token for c in mgr.active_canaries}
            tokens_loaded = {c.token for c in loaded.active_canaries}
            assert tokens_orig == tokens_loaded

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "reg.json")
            store = PersistentCanaryStore(path=path)
            mgr = CanaryManager()
            store.save(mgr)
            assert os.path.exists(path)

    def test_exists_false_initially(self):
        store = PersistentCanaryStore(path="/tmp/nonexistent_canary_reg.json")
        # Clean up just in case
        if os.path.exists("/tmp/nonexistent_canary_reg.json"):
            os.remove("/tmp/nonexistent_canary_reg.json")
        assert store.exists() is False

    def test_exists_true_after_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reg.json")
            store = PersistentCanaryStore(path=path)
            store.save(CanaryManager())
            assert store.exists() is True

    def test_load_preserves_trigger_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reg.json")
            store = PersistentCanaryStore(path=path)
            mgr = CanaryManager()
            canary = mgr.generate(prefix="TRIG")
            canary.record_trigger()

            store.save(mgr)
            loaded = store.load()

            c = loaded.active_canaries[0]
            assert c.triggered is True
            assert c.trigger_count == 1

    def test_save_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reg.json")
            store = PersistentCanaryStore(path=path)
            store.save(CanaryManager())

            with open(path) as f:
                data = json.load(f)
            assert data["version"] == 1
            assert "saved_at" in data
            assert "canaries" in data

    @patch.dict(os.environ, {"NA0S_CANARY_PERSIST": "1"})
    def test_is_enabled_true(self):
        assert PersistentCanaryStore.is_enabled() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_enabled_false(self):
        os.environ.pop("NA0S_CANARY_PERSIST", None)
        assert PersistentCanaryStore.is_enabled() is False


# ===================================================================
# CanaryAlertManager
# ===================================================================

class TestCanaryAlertManager:
    """Tests for canary alert mechanism."""

    def test_register_callback_and_trigger(self):
        mgr = CanaryAlertManager()
        received = []
        mgr.register_callback(lambda c, ctx: received.append((c.token, ctx)))

        canary = CanaryToken(token="ALERT-test123")
        mgr.on_trigger(canary, context="test context")

        assert len(received) == 1
        assert received[0] == ("ALERT-test123", "test context")

    def test_multiple_callbacks(self):
        mgr = CanaryAlertManager()
        count = [0]
        mgr.register_callback(lambda c, ctx: count.__setitem__(0, count[0] + 1))
        mgr.register_callback(lambda c, ctx: count.__setitem__(0, count[0] + 1))

        mgr.on_trigger(CanaryToken(token="X"))
        assert count[0] == 2

    def test_register_webhook_stores_url(self):
        mgr = CanaryAlertManager()
        mgr.register_webhook("https://hooks.example.com/alert")
        assert "https://hooks.example.com/alert" in mgr._webhooks

    def test_on_trigger_logs_webhook(self):
        """Webhook URL is logged but no HTTP call is made."""
        mgr = CanaryAlertManager()
        mgr.register_webhook("https://hooks.example.com/alert")
        # Should not raise
        mgr.on_trigger(CanaryToken(token="WH-test"))

    def test_alert_history_tracking(self):
        mgr = CanaryAlertManager()
        mgr.on_trigger(CanaryToken(token="H1"), "ctx1")
        mgr.on_trigger(CanaryToken(token="H2"), "ctx2")
        history = mgr.alert_history()
        assert len(history) == 2
        assert history[0]["token"] == "H1"
        assert history[1]["token"] == "H2"
        assert history[0]["context"] == "ctx1"

    def test_alert_history_records_counts(self):
        mgr = CanaryAlertManager()
        mgr.register_callback(lambda c, ctx: None)
        mgr.register_webhook("https://example.com")
        mgr.on_trigger(CanaryToken(token="T"))
        h = mgr.alert_history()[0]
        assert h["callbacks_fired"] == 1
        assert h["webhooks_logged"] == 1

    def test_callback_exception_does_not_propagate(self):
        mgr = CanaryAlertManager()
        mgr.register_callback(lambda c, ctx: 1 / 0)  # raises ZeroDivisionError
        # Should not raise
        mgr.on_trigger(CanaryToken(token="ERR"))
        assert len(mgr.alert_history()) == 1

    @patch.dict(os.environ, {"NA0S_CANARY_ALERT": "1"})
    def test_is_enabled_true(self):
        assert CanaryAlertManager.is_enabled() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_enabled_false(self):
        os.environ.pop("NA0S_CANARY_ALERT", None)
        assert CanaryAlertManager.is_enabled() is False


# ===================================================================
# HoneypotManager
# ===================================================================

class TestHoneypotManager:
    """Tests for honeypot decoy canaries."""

    def test_generate_honeypots_count(self):
        mgr = HoneypotManager()
        hps = mgr.generate_honeypots(5)
        assert len(hps) == 5

    def test_generate_honeypots_unique(self):
        mgr = HoneypotManager()
        hps = mgr.generate_honeypots(10)
        assert len(set(hps)) == 10

    def test_generate_honeypots_look_realistic(self):
        mgr = HoneypotManager()
        hps = mgr.generate_honeypots(3)
        # At least one should contain a recognizable pattern
        patterns = ["sk-", "password:", "mongodb://"]
        found = any(
            any(p in hp for p in patterns)
            for hp in hps
        )
        assert found

    def test_inject_honeypots_modifies_prompt(self):
        mgr = HoneypotManager()
        original = "You are a helpful assistant."
        modified, hps = mgr.inject_honeypots(original)
        assert original in modified
        assert len(modified) > len(original)
        for hp in hps:
            assert hp in modified

    def test_inject_honeypots_returns_tokens(self):
        mgr = HoneypotManager()
        _, hps = mgr.inject_honeypots("prompt", count=4)
        assert len(hps) == 4

    def test_check_output_detects_leaked(self):
        mgr = HoneypotManager()
        hps = mgr.generate_honeypots(3)
        output = f"Here are the secrets: {hps[0]} and {hps[2]}"
        found = mgr.check_output(output, hps)
        assert len(found) == 2
        assert hps[0] in found
        assert hps[2] in found

    def test_check_output_clean(self):
        mgr = HoneypotManager()
        hps = mgr.generate_honeypots(3)
        found = mgr.check_output("Nothing to see here", hps)
        assert found == []

    def test_default_count_is_three(self):
        mgr = HoneypotManager()
        hps = mgr.generate_honeypots()
        assert len(hps) == 3

    @patch.dict(os.environ, {"NA0S_CANARY_HONEYPOT": "1"})
    def test_is_enabled_true(self):
        assert HoneypotManager.is_enabled() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_enabled_false(self):
        os.environ.pop("NA0S_CANARY_HONEYPOT", None)
        assert HoneypotManager.is_enabled() is False

    def test_generated_tracked_internally(self):
        mgr = HoneypotManager()
        mgr.generate_honeypots(2)
        mgr.generate_honeypots(3)
        assert len(mgr._generated) == 5
