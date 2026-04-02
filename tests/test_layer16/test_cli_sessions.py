"""Tests for CLI session management commands."""
import json
import sys
from unittest.mock import patch
import pytest
from na0s.cli import main
from na0s.predict import _reset_conversation_monitor


class TestCLISessionCreate:
    def setup_method(self):
        _reset_conversation_monitor()
    def teardown_method(self):
        _reset_conversation_monitor()

    def test_create_json(self, capsys):
        ret = main(["session", "create", "--json"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert "session_id" in data

    def test_create_plain(self, capsys):
        ret = main(["session", "create"])
        assert ret == 0
        out = capsys.readouterr().out.strip()
        assert len(out) == 36  # UUID4 length


class TestCLISessionList:
    def setup_method(self):
        _reset_conversation_monitor()
    def teardown_method(self):
        _reset_conversation_monitor()

    def test_list_empty(self, capsys):
        ret = main(["session", "list", "--json"])
        assert ret == 0
        assert json.loads(capsys.readouterr().out) == []

    def test_list_after_create(self, capsys):
        main(["session", "create"])
        capsys.readouterr()  # discard create output
        ret = main(["session", "list", "--json"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 1


class TestCLISessionInspect:
    def setup_method(self):
        _reset_conversation_monitor()
    def teardown_method(self):
        _reset_conversation_monitor()

    def test_inspect_existing(self, capsys):
        main(["session", "create", "--json"])
        sid = json.loads(capsys.readouterr().out)["session_id"]
        ret = main(["session", "inspect", sid, "--json"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["session_id"] == sid

    def test_inspect_prefix(self, capsys):
        main(["session", "create", "--json"])
        sid = json.loads(capsys.readouterr().out)["session_id"]
        ret = main(["session", "inspect", sid[:8], "--json"])
        assert ret == 0

    def test_inspect_not_found(self, capsys):
        ret = main(["session", "inspect", "nonexistent"])
        assert ret == 3


class TestCLISessionExpire:
    def setup_method(self):
        _reset_conversation_monitor()
    def teardown_method(self):
        _reset_conversation_monitor()

    def test_expire_existing(self, capsys):
        main(["session", "create", "--json"])
        sid = json.loads(capsys.readouterr().out)["session_id"]
        ret = main(["session", "expire", sid])
        assert ret == 0
        capsys.readouterr()
        main(["session", "list", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 0


class TestCLISessionCleanup:
    def setup_method(self):
        _reset_conversation_monitor()
    def teardown_method(self):
        _reset_conversation_monitor()

    def test_cleanup_dry_run(self, capsys):
        ret = main(["session", "cleanup", "--dry-run", "--json"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["dry_run"] is True

    def test_cleanup(self, capsys):
        ret = main(["session", "cleanup", "--json"])
        assert ret == 0


class TestCLISessionStats:
    def setup_method(self):
        _reset_conversation_monitor()
    def teardown_method(self):
        _reset_conversation_monitor()

    def test_stats_empty(self, capsys):
        ret = main(["session", "stats", "--json"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["active_sessions"] == 0

    def test_stats_after_create(self, capsys):
        main(["session", "create"])
        capsys.readouterr()
        ret = main(["session", "stats", "--json"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["active_sessions"] == 1


class TestCLIScanSession:
    def setup_method(self):
        _reset_conversation_monitor()
    def teardown_method(self):
        _reset_conversation_monitor()

    def test_scan_with_new_session(self, capsys):
        ret = main(["scan", "Hello world", "--new-session"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Session:" in captured.err or "session" in captured.err.lower()

    def test_scan_without_session_unchanged(self, capsys):
        ret = main(["scan", "Hello world"])
        assert ret == 0
