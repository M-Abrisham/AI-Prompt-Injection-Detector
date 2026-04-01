"""OWASP LLM Top 10 Monitoring — detects version changes.

The OWASP LLM Top 10 is updated annually. This module monitors the
GitHub repo for version changes and cross-references with Na0S taxonomy
coverage.

DESIGN NOTE: This source changes infrequently (~annually), so we keep
it intentionally lightweight. We check the repo's README or version file
for the current version, diff against our last-known version, and alert
on changes.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    HTTP_TIMEOUT_SECONDS,
    OWASP_GITHUB_OWNER,
    OWASP_GITHUB_REPO,
)
from na0s.layer15.diff_engine import TaxonomyDiffEngine

logger = logging.getLogger(__name__)

OWASP_API_URL = (
    f"https://api.github.com/repos/{OWASP_GITHUB_OWNER}/{OWASP_GITHUB_REPO}"
)

# Known OWASP LLM Top 10 items (2025 version)
# Used as baseline; updated when we detect a new version.
OWASP_2025_ITEMS = {
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Supply Chain Vulnerabilities",
    "LLM04": "Data and Model Poisoning",
    "LLM05": "Improper Output Handling",
    "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage",
    "LLM08": "Vector and Embedding Weaknesses",
    "LLM09": "Misinformation",
    "LLM10": "Unbounded Consumption",
}


def _github_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Build GitHub API headers."""
    import os

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Na0S-Layer15-ThreatIntelSync",
    }
    tok = token or os.environ.get("GITHUB_TOKEN", "")
    if tok:
        headers["Authorization"] = f"token {tok}"
    return headers


class OwaspSync(ThreatIntelSource):
    """Monitors OWASP LLM Top 10 for version changes.

    Parameters
    ----------
    github_token : str, optional
        GitHub API token.
    snapshots_dir : Path, optional
        Directory for storing snapshots.
    """

    name = "owasp_llm_top10"

    def __init__(
        self,
        github_token: Optional[str] = None,
        snapshots_dir: Optional[Path] = None,
    ):
        super().__init__(snapshots_dir=snapshots_dir)
        self._headers = _github_headers(github_token)
        self._diff_engine = TaxonomyDiffEngine()

    def fetch_latest(self) -> SourceSnapshot:
        """Fetch the current OWASP LLM Top 10 items.

        Checks the repo for the latest commit and tries to read
        the item listing. Falls back to our hardcoded 2025 baseline
        if the repo structure is unexpected.
        """
        # Get latest commit SHA
        repo_info, _ = _fetch_json(OWASP_API_URL, headers=self._headers)
        default_branch = repo_info.get("default_branch", "main")

        branch_url = f"{OWASP_API_URL}/branches/{default_branch}"
        branch_info, _ = _fetch_json(branch_url, headers=self._headers)
        commit_sha = branch_info["commit"]["sha"]

        # Try to find the items listing from the repo
        items = self._try_parse_items_from_repo(default_branch)
        if not items:
            # Fall back to hardcoded baseline
            logger.info(
                "Could not parse OWASP items from repo, using 2025 baseline"
            )
            items = OWASP_2025_ITEMS

        techniques = [
            TechniqueEntry(
                id=f"OWASP-{item_id}",
                name=item_name,
                description=f"OWASP LLM Top 10 item: {item_name}",
                severity="high",
            )
            for item_id, item_name in sorted(items.items())
        ]

        return SourceSnapshot(
            source_name=self.name,
            fetched_at=datetime.now(timezone.utc),
            version=commit_sha[:8],
            techniques=techniques,
            raw_metadata={
                "commit_sha": commit_sha,
                "item_count": len(items),
            },
        )

    def _try_parse_items_from_repo(
        self, branch: str
    ) -> Optional[Dict[str, str]]:
        """Attempt to parse OWASP items from the repo's README.

        Returns a dict of {item_id: item_name} or None if parsing fails.
        This is best-effort — the README format may change.
        """
        readme_url = (
            f"https://raw.githubusercontent.com/"
            f"{OWASP_GITHUB_OWNER}/{OWASP_GITHUB_REPO}/{branch}/README.md"
        )
        try:
            readme = _fetch_text(readme_url, headers=self._headers)
        except SourceUnavailableError:
            return None

        # Look for LLM01-LLM10 patterns in the README
        items: Dict[str, str] = {}
        pattern = re.compile(r"(LLM\d{2})[:\s\-–—]+(.+?)(?:\n|$)")
        for match in pattern.finditer(readme):
            item_id = match.group(1)
            item_name = match.group(2).strip().rstrip("*#])")
            if item_name:
                items[item_id] = item_name

        return items if items else None

    def diff(
        self, old: SourceSnapshot, new: SourceSnapshot
    ) -> TaxonomyDiff:
        return self._diff_engine.compute(old, new)

    def apply(
        self, diff: TaxonomyDiff, dry_run: bool = False
    ) -> ApplyResult:
        """Alert on OWASP changes.

        OWASP changes are significant events — a changed item means
        the industry consensus on top threats has shifted.
        """
        if diff.has_changes:
            logger.warning(
                "OWASP LLM Top 10 has changed! %d modifications detected. "
                "This may require updating Na0S taxonomy tags.",
                len(diff.items),
            )

        return ApplyResult(
            applied_count=0,
            skipped_count=len(diff.items),
            dry_run=dry_run,
        )
