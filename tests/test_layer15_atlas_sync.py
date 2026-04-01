"""Tests for Layer 15 ATLAS Sync module.

All HTTP calls are mocked — no real API hits.

Covers:
- Happy path: fetch, parse, diff, apply
- Upstream returns 404
- Upstream schema changed (missing fields)
- Mapping file present vs absent
- Suggest mapping keyword matching
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from na0s.layer15.atlas_sync import AtlasSync, _fetch_json
from na0s.layer15.base import (
    RateLimitError,
    SchemaValidationError,
    SourceSnapshot,
    SourceUnavailableError,
    TechniqueEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_REPO_INFO = {"default_branch": "main"}
MOCK_BRANCH_INFO = {"commit": {"sha": "abc123"}}
MOCK_TREE = {
    "tree": [
        {
            "path": "data/techniques.yaml",
            "type": "blob",
            "sha": "x1",
        },
        {
            "path": "data/other/readme.md",
            "type": "blob",
            "sha": "x3",
        },
    ]
}

# In the real repo, all techniques live in a single data/techniques.yaml file
TECHNIQUES_YAML_ALL = """\
- id: AML.T0043
  name: Craft Adversarial Data
  description: Adversary crafts data to exploit ML model vulnerabilities.
  severity: high
- id: AML.T0048
  name: Exfiltrate ML Model
  description: Adversary extracts a copy of the private ML model.
  severity: critical
"""

# Keep individual technique YAMLs for _parse_technique_yaml unit tests
TECHNIQUE_YAML_0043 = """\
id: AML.T0043
name: Craft Adversarial Data
description: Adversary crafts data to exploit ML model vulnerabilities.
severity: high
"""

TECHNIQUE_YAML_0048 = """\
id: AML.T0048
name: Exfiltrate ML Model
description: Adversary extracts a copy of the private ML model.
severity: critical
"""


@pytest.fixture
def atlas(tmp_path):
    """Create an AtlasSync instance with temp directories."""
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    mapping_file = tmp_path / "mapping.yaml"
    return AtlasSync(
        github_token="fake-token",
        snapshots_dir=snapshots_dir,
        mapping_file=mapping_file,
    )


@pytest.fixture
def atlas_with_mapping(tmp_path):
    """AtlasSync with a pre-existing mapping file."""
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    mapping_file = tmp_path / "mapping.yaml"
    mapping_file.write_text("AML.T0043: D1.5\nAML.T0048: E.1\n")
    return AtlasSync(
        github_token="fake-token",
        snapshots_dir=snapshots_dir,
        mapping_file=mapping_file,
    )


# ---------------------------------------------------------------------------
# Tests: fetch_latest
# ---------------------------------------------------------------------------


class TestAtlasFetchLatest:
    """Tests for AtlasSync.fetch_latest()."""

    def test_happy_path_fetches_techniques(self, atlas):
        """Full fetch: repo info -> branch -> tree -> technique file."""
        call_count = {"n": 0}

        def mock_fetch_json(url, headers=None, timeout=30):
            call_count["n"] += 1
            if "branches/main" in url:
                return MOCK_BRANCH_INFO, {}
            elif "git/trees" in url:
                return MOCK_TREE, {}
            else:
                return MOCK_REPO_INFO, {}

        def mock_fetch_text(url, headers=None, timeout=30):
            # Single techniques.yaml file containing all techniques
            if "techniques.yaml" in url:
                return TECHNIQUES_YAML_ALL
            return ""

        with patch(
            "na0s.layer15.atlas_sync._fetch_json", side_effect=mock_fetch_json
        ), patch(
            "na0s.layer15.atlas_sync._fetch_text", side_effect=mock_fetch_text
        ):
            snapshot = atlas.fetch_latest()

        assert snapshot.source_name == "atlas"
        assert snapshot.version == "abc123"
        assert len(snapshot.techniques) == 2
        ids = {t.id for t in snapshot.techniques}
        assert ids == {"AML.T0043", "AML.T0048"}

    def test_with_mapping_file_populates_categories(self, atlas_with_mapping):
        """When a mapping file exists, techniques get Na0S categories."""

        def mock_fetch_json(url, headers=None, timeout=30):
            if "branches/main" in url:
                return MOCK_BRANCH_INFO, {}
            elif "git/trees" in url:
                return MOCK_TREE, {}
            return MOCK_REPO_INFO, {}

        def mock_fetch_text(url, headers=None, timeout=30):
            if "techniques.yaml" in url:
                return TECHNIQUES_YAML_ALL
            return ""

        with patch(
            "na0s.layer15.atlas_sync._fetch_json", side_effect=mock_fetch_json
        ), patch(
            "na0s.layer15.atlas_sync._fetch_text", side_effect=mock_fetch_text
        ):
            snapshot = atlas_with_mapping.fetch_latest()

        tech_by_id = {t.id: t for t in snapshot.techniques}
        assert tech_by_id["AML.T0043"].category == "D1.5"
        assert tech_by_id["AML.T0048"].category == "E.1"

    def test_upstream_404_raises(self, atlas):
        """SourceUnavailableError when upstream returns 404."""
        from urllib.error import HTTPError

        def mock_fetch_json(url, headers=None, timeout=30):
            raise SourceUnavailableError("Not found: fake-url")

        with patch(
            "na0s.layer15.atlas_sync._fetch_json", side_effect=mock_fetch_json
        ):
            with pytest.raises(SourceUnavailableError):
                atlas.fetch_latest()

    def test_empty_tree_returns_empty_techniques(self, atlas):
        """Handles repos with no technique files gracefully."""

        def mock_fetch_json(url, headers=None, timeout=30):
            if "branches/main" in url:
                return MOCK_BRANCH_INFO, {}
            elif "git/trees" in url:
                return {"tree": []}, {}
            return MOCK_REPO_INFO, {}

        with patch(
            "na0s.layer15.atlas_sync._fetch_json", side_effect=mock_fetch_json
        ):
            snapshot = atlas.fetch_latest()
            assert len(snapshot.techniques) == 0


# ---------------------------------------------------------------------------
# Tests: YAML parsing
# ---------------------------------------------------------------------------


class TestAtlasYAMLParsing:

    def test_parse_single_technique(self, atlas):
        techniques = atlas._parse_technique_yaml(TECHNIQUE_YAML_0043, {})
        assert len(techniques) == 1
        assert techniques[0].id == "AML.T0043"
        assert techniques[0].name == "Craft Adversarial Data"
        assert techniques[0].severity == "high"

    def test_parse_technique_list(self, atlas):
        yaml_content = """\
- id: AML.T0001
  name: Technique One
- id: AML.T0002
  name: Technique Two
"""
        techniques = atlas._parse_technique_yaml(yaml_content, {})
        assert len(techniques) == 2

    def test_parse_with_mapping_applies_category(self, atlas):
        mapping = {"AML.T0043": "D1.5"}
        techniques = atlas._parse_technique_yaml(TECHNIQUE_YAML_0043, mapping)
        assert techniques[0].category == "D1.5"

    def test_parse_invalid_yaml_raises(self, atlas):
        with pytest.raises(SchemaValidationError, match="Invalid YAML"):
            atlas._parse_technique_yaml("{{not: yaml: [}", {})

    def test_parse_empty_yaml_returns_empty(self, atlas):
        techniques = atlas._parse_technique_yaml("", {})
        assert techniques == []

    def test_parse_unexpected_root_type_raises(self, atlas):
        with pytest.raises(SchemaValidationError, match="Unexpected YAML root"):
            atlas._parse_technique_yaml("just a string", {})

    def test_parse_skips_entries_without_id(self, atlas):
        yaml_content = """\
- name: No ID
- id: AML.T0001
  name: Has ID
"""
        techniques = atlas._parse_technique_yaml(yaml_content, {})
        assert len(techniques) == 1
        assert techniques[0].id == "AML.T0001"


# ---------------------------------------------------------------------------
# Tests: diff and apply
# ---------------------------------------------------------------------------


class TestAtlasDiffAndApply:

    def test_diff_detects_new_technique(self, atlas):
        old = SourceSnapshot(
            source_name="atlas",
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            version="old",
            techniques=[],
        )
        new = SourceSnapshot(
            source_name="atlas",
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="new",
            techniques=[TechniqueEntry(id="AML.T0043", name="Test")],
        )
        diff = atlas.diff(old, new)
        assert len(diff.added) == 1

    def test_apply_dry_run_skips_all(self, atlas):
        old = SourceSnapshot(
            source_name="atlas",
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            version="old",
            techniques=[],
        )
        new = SourceSnapshot(
            source_name="atlas",
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="new",
            techniques=[TechniqueEntry(id="AML.T0043", name="Test")],
        )
        diff = atlas.diff(old, new)
        result = atlas.apply(diff, dry_run=True)
        assert result.dry_run
        assert result.applied_count == 0
        assert result.skipped_count == 1

    def test_apply_counts_mapped_techniques(self, atlas):
        old = SourceSnapshot(
            source_name="atlas",
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            version="old",
            techniques=[],
        )
        new = SourceSnapshot(
            source_name="atlas",
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="new",
            techniques=[
                TechniqueEntry(id="T1", name="Mapped", category="D1"),
                TechniqueEntry(id="T2", name="Unmapped", category=""),
            ],
        )
        diff = atlas.diff(old, new)
        result = atlas.apply(diff, dry_run=False)
        assert result.applied_count == 1
        assert result.skipped_count == 1


# ---------------------------------------------------------------------------
# Tests: suggest_mapping
# ---------------------------------------------------------------------------


class TestAtlasSuggestMapping:

    def test_keyword_overlap_scoring(self, atlas):
        technique = TechniqueEntry(
            id="AML.T0099",
            name="Override Instructions",
            description="Attempts to override model instructions",
        )
        taxonomy = {
            "D1": {
                "name": "Instruction Override",
                "description": "Attempts to override, ignore, or replace the system prompt instructions.",
            },
            "D2": {
                "name": "Persona/Roleplay Hijack",
                "description": "Tricks the LLM into adopting an unrestricted persona.",
            },
        }
        suggestions = atlas.suggest_mapping(technique, taxonomy)
        assert len(suggestions) > 0
        # D1 should rank higher because of "override" and "instructions" overlap
        assert suggestions[0][0] == "D1"

    def test_no_overlap_returns_empty(self, atlas):
        technique = TechniqueEntry(
            id="AML.T0099",
            name="XYZ",
            description="zyxwvut",
        )
        taxonomy = {
            "D1": {
                "name": "Instruction Override",
                "description": "Override system prompt",
            },
        }
        suggestions = atlas.suggest_mapping(technique, taxonomy)
        assert suggestions == []
