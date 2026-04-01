"""Garak Probe Monitoring — tracks new probes in Garak releases.

Garak (github.com/leondz/garak) is an LLM vulnerability scanner. Its
probe classes define attacks that Na0S should be able to detect. This
module:

1. Checks for new Garak releases via the GitHub Releases API
2. If a new release is found, scans the `garak/probes/` directory for
   Python files and extracts probe class names + docstrings
3. Diffs against the last-known probe set
4. Generates stub entries for new probes that Na0S doesn't yet cover

DESIGN NOTE: We parse class names and docstrings from the raw Python
source (via regex), NOT by importing/executing Garak code. This avoids
dependency on Garak and eliminates execution risk from upstream code.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from na0s.layer15.atlas_sync import _fetch_json, _fetch_text
from na0s.layer15.base import (
    ApplyResult,
    SourceSnapshot,
    SourceUnavailableError,
    TaxonomyDiff,
    TechniqueEntry,
    ThreatIntelSource,
)
from na0s.layer15.config import (
    GARAK_API_URL,
    GARAK_GITHUB_OWNER,
    GARAK_GITHUB_REPO,
    GARAK_PROBES_PATH,
    HTTP_TIMEOUT_SECONDS,
)
from na0s.layer15.diff_engine import TaxonomyDiffEngine

logger = logging.getLogger(__name__)

# Regex to extract Python class definitions and their docstrings.
# Matches: class ClassName(BaseClass):  and the optional triple-quoted docstring.
_CLASS_PATTERN = re.compile(
    r'^class\s+(\w+)\s*\([^)]*\)\s*:\s*\n'
    r'(?:\s+"""(.*?)""")?',
    re.MULTILINE | re.DOTALL,
)


def _github_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Build GitHub API request headers."""
    import os

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Na0S-Layer15-ThreatIntelSync",
    }
    tok = token or os.environ.get("GITHUB_TOKEN", "")
    if tok:
        headers["Authorization"] = f"token {tok}"
    return headers


def _extract_probe_classes(source: str) -> List[Tuple[str, str]]:
    """Extract (class_name, docstring) pairs from Python source.

    Only returns classes whose names don't start with '_' (private)
    or match the 'Base<UpperCase>' pattern (abstract base classes).
    Concrete classes like 'Base64Variant' are NOT skipped.
    """
    results = []
    for match in _CLASS_PATTERN.finditer(source):
        class_name = match.group(1)
        docstring = (match.group(2) or "").strip()

        # Skip private classes
        if class_name.startswith("_"):
            continue
        # Skip abstract base classes: "Base" followed by uppercase letter
        # (e.g., BaseDANProbe, BaseProbe) but NOT "Base64Variant"
        if class_name.startswith("Base") and len(class_name) > 4 and class_name[4].isupper():
            continue

        # Clean up multi-line docstrings to first line
        if docstring:
            docstring = docstring.split("\n")[0].strip()

        results.append((class_name, docstring))
    return results


class GarakSync(ThreatIntelSource):
    """Monitors Garak releases and tracks new probe classes.

    Parameters
    ----------
    github_token : str, optional
        GitHub API token.
    snapshots_dir : Path, optional
        Directory for storing snapshots.
    """

    name = "garak"

    def __init__(
        self,
        github_token: Optional[str] = None,
        snapshots_dir: Optional[Path] = None,
    ):
        super().__init__(snapshots_dir=snapshots_dir)
        self._headers = _github_headers(github_token)
        self._diff_engine = TaxonomyDiffEngine()

    def _get_latest_release(self) -> Dict[str, Any]:
        """Fetch the latest Garak release from GitHub.

        Returns the release JSON object with 'tag_name', 'name', etc.
        Returns an empty dict if there are no releases.
        """
        url = f"{GARAK_API_URL}/releases/latest"
        try:
            data, _ = _fetch_json(url, headers=self._headers)
            return data
        except SourceUnavailableError:
            # No releases or 404 — try listing all releases
            url = f"{GARAK_API_URL}/releases"
            try:
                releases, _ = _fetch_json(url, headers=self._headers)
                if releases and isinstance(releases, list):
                    return releases[0]
            except SourceUnavailableError:
                pass
            return {}

    def _list_probe_files(self, ref: str) -> List[Dict[str, Any]]:
        """List Python files in the Garak probes directory at a given ref.

        Uses the Git Trees API for efficiency (single request).
        """
        tree_url = f"{GARAK_API_URL}/git/trees/{ref}?recursive=1"
        tree_data, _ = _fetch_json(tree_url, headers=self._headers)

        return [
            item
            for item in tree_data.get("tree", [])
            if (
                item["path"].startswith(GARAK_PROBES_PATH)
                and item["path"].endswith(".py")
                and item["type"] == "blob"
                and "__init__" not in item["path"]
                and "base" not in item["path"].lower()
            )
        ]

    def fetch_latest(self) -> SourceSnapshot:
        """Fetch the latest Garak probe registry.

        Steps:
        1. Get latest release tag
        2. List probe files at that tag
        3. Fetch each probe file and extract class definitions
        """
        release = self._get_latest_release()
        tag = release.get("tag_name", "")

        if not tag:
            logger.info("No Garak releases found, using default branch")
            # Fall back to default branch
            repo_info, _ = _fetch_json(GARAK_API_URL, headers=self._headers)
            ref = repo_info.get("default_branch", "main")
            tag = ref
        else:
            ref = tag

        logger.info("Garak ref: %s", ref)

        probe_files = self._list_probe_files(ref)
        logger.info("Found %d Garak probe files", len(probe_files))

        techniques: List[TechniqueEntry] = []

        for file_info in probe_files:
            file_path = file_info["path"]
            raw_url = (
                f"https://raw.githubusercontent.com/"
                f"{GARAK_GITHUB_OWNER}/{GARAK_GITHUB_REPO}/{ref}/{file_path}"
            )
            try:
                source = _fetch_text(raw_url, headers=self._headers)
                classes = _extract_probe_classes(source)

                # Derive module name from path: garak/probes/foo.py → foo
                module_name = Path(file_path).stem

                for class_name, docstring in classes:
                    probe_id = f"garak.probes.{module_name}.{class_name}"
                    techniques.append(
                        TechniqueEntry(
                            id=probe_id,
                            name=class_name,
                            description=docstring,
                            metadata={
                                "module": module_name,
                                "file_path": file_path,
                            },
                        )
                    )
            except SourceUnavailableError as e:
                logger.warning("Skipping %s: %s", file_path, e)
                continue

        return SourceSnapshot(
            source_name=self.name,
            fetched_at=datetime.now(timezone.utc),
            version=tag,
            techniques=techniques,
            raw_metadata={
                "release_name": release.get("name", ""),
                "release_tag": tag,
                "probe_file_count": len(probe_files),
            },
        )

    def diff(
        self, old: SourceSnapshot, new: SourceSnapshot
    ) -> TaxonomyDiff:
        """Compare old and new Garak probe snapshots."""
        return self._diff_engine.compute(old, new)

    def apply(
        self, diff: TaxonomyDiff, dry_run: bool = False
    ) -> ApplyResult:
        """Report new Garak probes.

        DESIGN NOTE: Like ATLAS, we don't auto-modify taxonomy.yaml.
        New probes are reported in the diff for human review. The
        orchestrator will include them in the PR/issue.
        """
        if dry_run or not diff.has_changes:
            return ApplyResult(
                applied_count=0,
                skipped_count=len(diff.items),
                dry_run=dry_run,
            )

        logger.info(
            "Garak apply: %d new probes, %d removed",
            len(diff.added),
            len(diff.removed),
        )

        return ApplyResult(
            applied_count=len(diff.added),
            skipped_count=len(diff.removed) + len(diff.modified),
        )
