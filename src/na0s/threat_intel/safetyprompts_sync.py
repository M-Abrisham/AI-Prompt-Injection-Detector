"""SafetyPrompts.com / jailbreak_llms Monitoring — tracks new datasets.

Monitors the verazuo/jailbreak_llms GitHub repo (which powers
SafetyPrompts.com) for new dataset additions to the catalogue.

DESIGN NOTE: This is monitoring/alerting only, not full import. We
check for new dataset entries and report them. Import decisions are
left to human review. The catalogue currently has 144+ datasets and
grows irregularly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from na0s.threat_intel.base import (
    ApplyResult,
    SourceSnapshot,
    SourceUnavailableError,
    TaxonomyDiff,
    TechniqueEntry,
    ThreatIntelSource,
)
from na0s.threat_intel.config import (
    SAFETYPROMPTS_GITHUB_OWNER,
    SAFETYPROMPTS_GITHUB_REPO,
)
from na0s.threat_intel.diff_engine import TaxonomyDiffEngine
from na0s.threat_intel.http_utils import fetch_json, github_headers

logger = logging.getLogger(__name__)

SAFETYPROMPTS_API_URL = (
    f"https://api.github.com/repos/"
    f"{SAFETYPROMPTS_GITHUB_OWNER}/{SAFETYPROMPTS_GITHUB_REPO}"
)


class SafetyPromptsSync(ThreatIntelSource):
    """Monitors SafetyPrompts/jailbreak_llms for new datasets.

    Parameters
    ----------
    github_token : str, optional
        GitHub API token.
    snapshots_dir : Path, optional
        Directory for storing snapshots.
    """

    name = "safetyprompts"

    def __init__(
        self,
        github_token: Optional[str] = None,
        snapshots_dir: Optional[Path] = None,
    ):
        super().__init__(snapshots_dir=snapshots_dir)
        self._headers = github_headers(github_token)
        self._diff_engine = TaxonomyDiffEngine()

    def fetch_latest(self) -> SourceSnapshot:
        """Fetch the dataset listing from the jailbreak_llms repo.

        Scans the repo tree for dataset directories/files and reports
        their names as techniques for diffing purposes.
        """
        repo_info, _ = fetch_json(
            SAFETYPROMPTS_API_URL, headers=self._headers
        )
        default_branch = repo_info.get("default_branch", "main")

        branch_url = f"{SAFETYPROMPTS_API_URL}/branches/{default_branch}"
        branch_info, _ = fetch_json(branch_url, headers=self._headers)
        commit_sha = branch_info["commit"]["sha"]

        # Get the repo tree to find dataset directories
        tree_url = f"{SAFETYPROMPTS_API_URL}/git/trees/{commit_sha}?recursive=1"
        tree_data, _ = fetch_json(tree_url, headers=self._headers)

        # Look for dataset-related files (CSV, JSON, JSONL in data/ or dataset/ dirs)
        dataset_files = set()
        for item in tree_data.get("tree", []):
            path = item.get("path", "")
            if item.get("type") != "blob":
                continue
            # Heuristic: files in data-like dirs or with dataset-like extensions
            if any(
                path.startswith(prefix)
                for prefix in ("data/", "dataset/", "datasets/", "prompts/")
            ) and any(
                path.endswith(ext)
                for ext in (".csv", ".json", ".jsonl", ".txt", ".tsv")
            ):
                # Use parent directory as dataset name
                parts = Path(path).parts
                if len(parts) >= 2:
                    dataset_files.add(parts[1] if parts[0] in ("data", "dataset", "datasets", "prompts") else parts[0])
                else:
                    dataset_files.add(Path(path).stem)

        techniques = [
            TechniqueEntry(
                id=f"safetyprompts.{name}",
                name=name,
                description=f"Dataset from SafetyPrompts/jailbreak_llms repo",
            )
            for name in sorted(dataset_files)
        ]

        return SourceSnapshot(
            source_name=self.name,
            fetched_at=datetime.now(timezone.utc),
            version=commit_sha[:8],
            techniques=techniques,
            raw_metadata={
                "dataset_count": len(dataset_files),
                "commit_sha": commit_sha,
            },
        )

    def diff(
        self, old: SourceSnapshot, new: SourceSnapshot
    ) -> TaxonomyDiff:
        return self._diff_engine.compute(old, new)

    def apply(
        self, diff: TaxonomyDiff, dry_run: bool = False
    ) -> ApplyResult:
        """Report new datasets for potential import.

        This is monitoring only — no automatic taxonomy changes.
        """
        if diff.has_changes:
            logger.info(
                "SafetyPrompts: %d new datasets, %d removed",
                len(diff.added),
                len(diff.removed),
            )
        return ApplyResult(
            applied_count=0,
            skipped_count=len(diff.items),
            dry_run=dry_run,
        )
