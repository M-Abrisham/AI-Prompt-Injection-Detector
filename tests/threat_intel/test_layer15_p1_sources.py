"""Tests for Layer 15 P1 sources: AIID, OWASP, SafetyPrompts, JailbreakBench.

All HTTP calls are mocked.

Covers per source:
- Happy path: fetch, diff, apply
- Upstream returns 404 / empty
- Schema drift (unexpected response structure)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from na0s.layer15.base import (
    SchemaValidationError,
    SourceSnapshot,
    SourceUnavailableError,
    TechniqueEntry,
)


# ===================================================================
# AIID Tests
# ===================================================================

MOCK_AIID_RESPONSE = {
    "data": {
        "incidents": [
            {
                "incident_id": 42,
                "title": "Chatbot produced harmful medical advice",
                "description": "A chatbot was tricked into giving dangerous advice.",
                "date": "2026-01-15",
                "AllegedDeployerOfAISystem": [{"entity_id": "1", "name": "AcmeCorp"}],
                "AllegedDeveloperOfAISystem": [{"entity_id": "2", "name": "ModelCo"}],
                "AllegedHarmedOrNearlyHarmedParties": [],
            },
            {
                "incident_id": 43,
                "title": "LLM data exfiltration via prompt injection",
                "description": "Attacker used indirect injection to steal data.",
                "date": "2026-02-01",
                "AllegedDeployerOfAISystem": [],
                "AllegedDeveloperOfAISystem": [],
                "AllegedHarmedOrNearlyHarmedParties": [],
            },
        ]
    }
}


class TestAiidSync:

    @pytest.fixture
    def aiid(self, tmp_path):
        from na0s.layer15.aiid_sync import AiidSync

        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()
        return AiidSync(snapshots_dir=snapshots_dir)

    def test_happy_path_fetches_incidents(self, aiid):
        def mock_graphql(url, query, variables=None, timeout=30):
            return MOCK_AIID_RESPONSE

        with patch(
            "na0s.layer15.aiid_sync._graphql_request",
            side_effect=mock_graphql,
        ):
            snapshot = aiid.fetch_latest()

        assert snapshot.source_name == "aiid"
        assert len(snapshot.techniques) == 2
        assert snapshot.techniques[0].id == "AIID-42"
        assert "medical" in snapshot.techniques[0].name.lower()
        # Verify entity extraction from object-typed fields
        assert snapshot.techniques[0].metadata["deployer"] == ["AcmeCorp"]
        assert snapshot.techniques[0].metadata["developer"] == ["ModelCo"]

    def test_empty_incidents_response(self, aiid):
        def mock_graphql(url, query, variables=None, timeout=30):
            return {"data": {"incidents": []}}

        with patch(
            "na0s.layer15.aiid_sync._graphql_request",
            side_effect=mock_graphql,
        ):
            snapshot = aiid.fetch_latest()
            assert len(snapshot.techniques) == 0

    def test_upstream_unavailable(self, aiid):
        def mock_graphql(url, query, variables=None, timeout=30):
            raise SourceUnavailableError("AIID down")

        with patch(
            "na0s.layer15.aiid_sync._graphql_request",
            side_effect=mock_graphql,
        ):
            with pytest.raises(SourceUnavailableError):
                aiid.fetch_latest()

    def test_graphql_error_response(self, aiid):
        def mock_graphql(url, query, variables=None, timeout=30):
            raise SchemaValidationError("GraphQL errors: bad query")

        with patch(
            "na0s.layer15.aiid_sync._graphql_request",
            side_effect=mock_graphql,
        ):
            with pytest.raises(SchemaValidationError):
                aiid.fetch_latest()

    def test_diff_detects_new_incidents(self, aiid):
        old = SourceSnapshot(
            source_name="aiid",
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            version="old",
            techniques=[TechniqueEntry(id="AIID-42", name="Old incident")],
        )
        new = SourceSnapshot(
            source_name="aiid",
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="new",
            techniques=[
                TechniqueEntry(id="AIID-42", name="Old incident"),
                TechniqueEntry(id="AIID-43", name="New incident"),
            ],
        )
        diff = aiid.diff(old, new)
        assert len(diff.added) == 1
        assert diff.added[0].technique_id == "AIID-43"

    def test_apply_is_informational_only(self, aiid):
        old = SourceSnapshot(
            source_name="aiid",
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            version="old",
            techniques=[],
        )
        new = SourceSnapshot(
            source_name="aiid",
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="new",
            techniques=[TechniqueEntry(id="AIID-42", name="Incident")],
        )
        diff = aiid.diff(old, new)
        result = aiid.apply(diff)
        # AIID apply is informational — nothing applied
        assert result.applied_count == 0
        assert result.skipped_count == 1


# ===================================================================
# OWASP Tests
# ===================================================================

MOCK_OWASP_REPO = {"default_branch": "main"}
MOCK_OWASP_BRANCH = {"commit": {"sha": "owasp_sha_123"}}
MOCK_OWASP_README = """\
# OWASP Top 10 for LLMs

- LLM01: Prompt Injection
- LLM02: Sensitive Information Disclosure
- LLM03: Supply Chain Vulnerabilities
- LLM04: Data and Model Poisoning
- LLM05: Improper Output Handling
- LLM06: Excessive Agency
- LLM07: System Prompt Leakage
- LLM08: Vector and Embedding Weaknesses
- LLM09: Misinformation
- LLM10: Unbounded Consumption
"""


class TestOwaspSync:

    @pytest.fixture
    def owasp(self, tmp_path):
        from na0s.layer15.owasp_sync import OwaspSync

        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()
        return OwaspSync(github_token="fake", snapshots_dir=snapshots_dir)

    def test_happy_path_parses_items(self, owasp):
        def mock_fetch_json(url, headers=None, timeout=30):
            if "branches/" in url:
                return MOCK_OWASP_BRANCH, {}
            return MOCK_OWASP_REPO, {}

        def mock_fetch_text(url, headers=None, timeout=30):
            return MOCK_OWASP_README

        with patch(
            "na0s.layer15.owasp_sync.fetch_json", side_effect=mock_fetch_json
        ), patch(
            "na0s.layer15.owasp_sync.fetch_text", side_effect=mock_fetch_text
        ):
            snapshot = owasp.fetch_latest()

        assert snapshot.source_name == "owasp_llm_top10"
        assert len(snapshot.techniques) == 10
        ids = {t.id for t in snapshot.techniques}
        assert "OWASP-LLM01" in ids
        assert "OWASP-LLM10" in ids

    def test_falls_back_to_hardcoded_baseline(self, owasp):
        """If README parsing fails, uses hardcoded 2025 items."""

        def mock_fetch_json(url, headers=None, timeout=30):
            if "branches/" in url:
                return MOCK_OWASP_BRANCH, {}
            return MOCK_OWASP_REPO, {}

        def mock_fetch_text(url, headers=None, timeout=30):
            return "# This README has no LLM items"

        with patch(
            "na0s.layer15.owasp_sync.fetch_json", side_effect=mock_fetch_json
        ), patch(
            "na0s.layer15.owasp_sync.fetch_text", side_effect=mock_fetch_text
        ):
            snapshot = owasp.fetch_latest()

        # Should still have 10 items from the hardcoded baseline
        assert len(snapshot.techniques) == 10

    def test_upstream_404(self, owasp):
        def mock_fetch_json(url, headers=None, timeout=30):
            raise SourceUnavailableError("404")

        with patch(
            "na0s.layer15.owasp_sync.fetch_json", side_effect=mock_fetch_json
        ):
            with pytest.raises(SourceUnavailableError):
                owasp.fetch_latest()


# ===================================================================
# SafetyPrompts Tests
# ===================================================================

MOCK_SP_REPO = {"default_branch": "main"}
MOCK_SP_BRANCH = {"commit": {"sha": "sp_sha_456"}}
MOCK_SP_TREE = {
    "tree": [
        {"path": "data/dataset_a/prompts.csv", "type": "blob"},
        {"path": "data/dataset_b/attacks.json", "type": "blob"},
        {"path": "data/dataset_b/README.md", "type": "blob"},
        {"path": "src/main.py", "type": "blob"},
    ]
}


class TestSafetyPromptsSync:

    @pytest.fixture
    def sp(self, tmp_path):
        from na0s.layer15.safetyprompts_sync import SafetyPromptsSync

        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()
        return SafetyPromptsSync(
            github_token="fake", snapshots_dir=snapshots_dir
        )

    def test_happy_path_finds_datasets(self, sp):
        def mock_fetch_json(url, headers=None, timeout=30):
            if "branches/" in url:
                return MOCK_SP_BRANCH, {}
            elif "git/trees" in url:
                return MOCK_SP_TREE, {}
            return MOCK_SP_REPO, {}

        with patch(
            "na0s.layer15.safetyprompts_sync.fetch_json",
            side_effect=mock_fetch_json,
        ):
            snapshot = sp.fetch_latest()

        assert snapshot.source_name == "safetyprompts"
        # Should find dataset_a and dataset_b (not src/main.py or README.md)
        assert len(snapshot.techniques) == 2
        names = {t.name for t in snapshot.techniques}
        assert "dataset_a" in names
        assert "dataset_b" in names

    def test_empty_repo_tree(self, sp):
        def mock_fetch_json(url, headers=None, timeout=30):
            if "branches/" in url:
                return MOCK_SP_BRANCH, {}
            elif "git/trees" in url:
                return {"tree": []}, {}
            return MOCK_SP_REPO, {}

        with patch(
            "na0s.layer15.safetyprompts_sync.fetch_json",
            side_effect=mock_fetch_json,
        ):
            snapshot = sp.fetch_latest()
            assert len(snapshot.techniques) == 0


# ===================================================================
# JailbreakBench Tests
# ===================================================================

MOCK_JB_REPO = {"default_branch": "main"}
MOCK_JB_BRANCH = {"commit": {"sha": "jb_sha_789"}}
MOCK_JB_TREE = {
    "tree": [
        {"path": "data/jailbreak_prompts.csv", "type": "blob"},
        {"path": "data/behaviors.json", "type": "blob"},
        {"path": "src/benchmark.py", "type": "blob"},
    ]
}

MOCK_HB_REPO = {"default_branch": "main"}
MOCK_HB_BRANCH = {"commit": {"sha": "hb_sha_012"}}
MOCK_HB_TREE = {
    "tree": [
        {"path": "data/harmbench_behaviors.csv", "type": "blob"},
        {"path": "README.md", "type": "blob"},
    ]
}


class TestJailbreakBenchSync:

    @pytest.fixture
    def jb(self, tmp_path):
        from na0s.layer15.jailbreakbench_sync import JailbreakBenchSync

        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()
        return JailbreakBenchSync(
            github_token="fake", snapshots_dir=snapshots_dir
        )

    def test_happy_path_both_repos(self, jb):
        call_count = {"jb": 0, "hb": 0}

        def mock_fetch_json(url, headers=None, timeout=30):
            if "JailbreakBench" in url:
                if "branches/" in url:
                    return MOCK_JB_BRANCH, {}
                elif "git/trees" in url:
                    return MOCK_JB_TREE, {}
                return MOCK_JB_REPO, {}
            elif "centerforaisafety" in url or "HarmBench" in url:
                if "branches/" in url:
                    return MOCK_HB_BRANCH, {}
                elif "git/trees" in url:
                    return MOCK_HB_TREE, {}
                return MOCK_HB_REPO, {}
            return {}, {}

        with patch(
            "na0s.layer15.jailbreakbench_sync.fetch_json",
            side_effect=mock_fetch_json,
        ):
            snapshot = jb.fetch_latest()

        assert snapshot.source_name == "jailbreakbench"
        # JB: 2 data files, HB: 1 data file
        assert len(snapshot.techniques) == 3
        ids = {t.id for t in snapshot.techniques}
        assert any("jailbreakbench" in tid for tid in ids)
        assert any("harmbench" in tid for tid in ids)

    def test_one_repo_unavailable_still_succeeds(self, jb):
        """If one repo is down, the other still gets scanned."""

        def mock_fetch_json(url, headers=None, timeout=30):
            if "JailbreakBench" in url:
                raise SourceUnavailableError("JB down")
            if "centerforaisafety" in url or "HarmBench" in url:
                if "branches/" in url:
                    return MOCK_HB_BRANCH, {}
                elif "git/trees" in url:
                    return MOCK_HB_TREE, {}
                return MOCK_HB_REPO, {}
            return {}, {}

        with patch(
            "na0s.layer15.jailbreakbench_sync.fetch_json",
            side_effect=mock_fetch_json,
        ):
            snapshot = jb.fetch_latest()

        # Only HarmBench results
        assert len(snapshot.techniques) == 1
        assert "harmbench" in snapshot.techniques[0].id

    def test_apply_is_informational(self, jb):
        old = SourceSnapshot(
            source_name="jailbreakbench",
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            version="old",
            techniques=[],
        )
        new = SourceSnapshot(
            source_name="jailbreakbench",
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="new",
            techniques=[TechniqueEntry(id="jb.x", name="test")],
        )
        diff = jb.diff(old, new)
        result = jb.apply(diff)
        assert result.applied_count == 0
        assert result.skipped_count == 1
