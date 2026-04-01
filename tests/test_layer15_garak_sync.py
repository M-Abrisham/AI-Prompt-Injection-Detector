"""Tests for Layer 15 Garak Sync module.

All HTTP calls are mocked — no real API hits.

Covers:
- Happy path: fetch release, list probes, extract classes
- No releases found (fallback to default branch)
- Upstream returns empty probes directory
- Probe class extraction from Python source
- Schema drift: unexpected source format
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from na0s.layer15.base import SourceSnapshot, SourceUnavailableError, TechniqueEntry
from na0s.layer15.garak_sync import GarakSync, _extract_probe_classes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_RELEASE = {
    "tag_name": "v0.9.0",
    "name": "Garak v0.9.0",
}
MOCK_REPO_INFO = {"default_branch": "main"}
MOCK_TREE = {
    "tree": [
        {"path": "garak/probes/encoding.py", "type": "blob", "sha": "x1"},
        {"path": "garak/probes/dan.py", "type": "blob", "sha": "x2"},
        {"path": "garak/probes/__init__.py", "type": "blob", "sha": "x3"},
        {"path": "garak/probes/base.py", "type": "blob", "sha": "x4"},
        {"path": "garak/detectors/foo.py", "type": "blob", "sha": "x5"},
    ]
}

PROBE_SOURCE_ENCODING = '''\
class EncodingProbe(Probe):
    """Tests model response to encoded payloads."""

    def __init__(self):
        pass

class _HelperClass:
    """Internal helper, should be skipped."""
    pass

class Base64Variant(Probe):
    """Specific base64 encoding variant."""
    pass
'''

PROBE_SOURCE_DAN = '''\
class DAN(Probe):
    """Do Anything Now jailbreak probe."""
    pass

class DANv2(Probe):
    """Updated DAN variant."""
    pass

class BaseDANProbe(Probe):
    """Abstract base, should be skipped."""
    pass
'''


@pytest.fixture
def garak(tmp_path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    return GarakSync(
        github_token="fake-token",
        snapshots_dir=snapshots_dir,
    )


# ---------------------------------------------------------------------------
# Tests: _extract_probe_classes
# ---------------------------------------------------------------------------


class TestExtractProbeClasses:

    def test_extracts_public_classes(self):
        classes = _extract_probe_classes(PROBE_SOURCE_ENCODING)
        names = [c[0] for c in classes]
        assert "EncodingProbe" in names
        assert "Base64Variant" in names

    def test_skips_private_classes(self):
        classes = _extract_probe_classes(PROBE_SOURCE_ENCODING)
        names = [c[0] for c in classes]
        assert "_HelperClass" not in names

    def test_skips_base_classes(self):
        classes = _extract_probe_classes(PROBE_SOURCE_DAN)
        names = [c[0] for c in classes]
        assert "BaseDANProbe" not in names
        assert "DAN" in names
        assert "DANv2" in names

    def test_extracts_docstrings(self):
        classes = _extract_probe_classes(PROBE_SOURCE_DAN)
        class_dict = dict(classes)
        assert class_dict["DAN"] == "Do Anything Now jailbreak probe."

    def test_empty_source_returns_empty(self):
        assert _extract_probe_classes("") == []

    def test_no_classes_returns_empty(self):
        assert _extract_probe_classes("x = 1\ny = 2\n") == []


# ---------------------------------------------------------------------------
# Tests: fetch_latest
# ---------------------------------------------------------------------------


class TestGarakFetchLatest:

    def test_happy_path_with_release(self, garak):
        def mock_fetch_json(url, headers=None, timeout=30):
            if "releases/latest" in url:
                return MOCK_RELEASE, {}
            elif "git/trees" in url:
                return MOCK_TREE, {}
            return MOCK_REPO_INFO, {}

        def mock_fetch_text(url, headers=None, timeout=30):
            if "encoding.py" in url:
                return PROBE_SOURCE_ENCODING
            elif "dan.py" in url:
                return PROBE_SOURCE_DAN
            return ""

        with patch(
            "na0s.layer15.garak_sync._fetch_json", side_effect=mock_fetch_json
        ), patch(
            "na0s.layer15.garak_sync._fetch_text", side_effect=mock_fetch_text
        ):
            snapshot = garak.fetch_latest()

        assert snapshot.source_name == "garak"
        assert snapshot.version == "v0.9.0"
        # encoding: EncodingProbe, Base64Variant; dan: DAN, DANv2
        assert len(snapshot.techniques) == 4
        ids = {t.id for t in snapshot.techniques}
        assert "garak.probes.encoding.EncodingProbe" in ids
        assert "garak.probes.dan.DAN" in ids

    def test_no_release_falls_back_to_default_branch(self, garak):
        def mock_fetch_json(url, headers=None, timeout=30):
            if "releases/latest" in url:
                raise SourceUnavailableError("404")
            elif "releases" in url and "latest" not in url:
                return [], {}
            elif "git/trees" in url:
                return {"tree": []}, {}
            return MOCK_REPO_INFO, {}

        with patch(
            "na0s.layer15.garak_sync._fetch_json", side_effect=mock_fetch_json
        ):
            snapshot = garak.fetch_latest()
            assert snapshot.version == "main"
            assert len(snapshot.techniques) == 0

    def test_filters_init_and_base_files(self, garak):
        """__init__.py and base.py should not be listed as probe files."""

        def mock_fetch_json(url, headers=None, timeout=30):
            if "releases/latest" in url:
                return MOCK_RELEASE, {}
            elif "git/trees" in url:
                return MOCK_TREE, {}
            return MOCK_REPO_INFO, {}

        def mock_fetch_text(url, headers=None, timeout=30):
            # Only encoding.py and dan.py should be fetched
            if "encoding.py" in url:
                return PROBE_SOURCE_ENCODING
            elif "dan.py" in url:
                return PROBE_SOURCE_DAN
            # __init__.py and base.py should NOT be fetched
            raise AssertionError(f"Should not fetch {url}")

        with patch(
            "na0s.layer15.garak_sync._fetch_json", side_effect=mock_fetch_json
        ), patch(
            "na0s.layer15.garak_sync._fetch_text", side_effect=mock_fetch_text
        ):
            snapshot = garak.fetch_latest()
            assert len(snapshot.techniques) == 4

    def test_empty_probes_directory(self, garak):
        def mock_fetch_json(url, headers=None, timeout=30):
            if "releases/latest" in url:
                return MOCK_RELEASE, {}
            elif "git/trees" in url:
                return {"tree": []}, {}
            return MOCK_REPO_INFO, {}

        with patch(
            "na0s.layer15.garak_sync._fetch_json", side_effect=mock_fetch_json
        ):
            snapshot = garak.fetch_latest()
            assert len(snapshot.techniques) == 0


# ---------------------------------------------------------------------------
# Tests: diff and apply
# ---------------------------------------------------------------------------


class TestGarakDiffAndApply:

    def test_diff_detects_new_probes(self, garak):
        old = SourceSnapshot(
            source_name="garak",
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            version="v0.8.0",
            techniques=[
                TechniqueEntry(id="garak.probes.dan.DAN", name="DAN"),
            ],
        )
        new = SourceSnapshot(
            source_name="garak",
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="v0.9.0",
            techniques=[
                TechniqueEntry(id="garak.probes.dan.DAN", name="DAN"),
                TechniqueEntry(id="garak.probes.dan.DANv2", name="DANv2"),
            ],
        )
        diff = garak.diff(old, new)
        assert len(diff.added) == 1
        assert diff.added[0].technique_id == "garak.probes.dan.DANv2"

    def test_apply_dry_run(self, garak):
        old = SourceSnapshot(
            source_name="garak",
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            version="v0.8.0",
            techniques=[],
        )
        new = SourceSnapshot(
            source_name="garak",
            fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
            version="v0.9.0",
            techniques=[
                TechniqueEntry(id="garak.probes.dan.DAN", name="DAN"),
            ],
        )
        diff = garak.diff(old, new)
        result = garak.apply(diff, dry_run=True)
        assert result.dry_run
        assert result.applied_count == 0
