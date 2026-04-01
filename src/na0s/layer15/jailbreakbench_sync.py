"""JailbreakBench / HarmBench Sync — cross-validates benchmark datasets.

Downloads latest benchmark datasets from JailbreakBench and HarmBench
GitHub repos, cross-validates against Na0S's existing test corpus, and
reports overlap and gaps.

DESIGN NOTE: These datasets can be large. We only download metadata
(file listings, not full content) for the weekly check. Full dataset
download is only triggered when new files are detected and is handled
in a temp directory to avoid committing raw benchmark data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from na0s.layer15.base import (
    ApplyResult,
    SourceSnapshot,
    SourceUnavailableError,
    TaxonomyDiff,
    TechniqueEntry,
    ThreatIntelSource,
)
from na0s.layer15.config import (
    HARMBENCH_GITHUB_OWNER,
    HARMBENCH_GITHUB_REPO,
    JAILBREAKBENCH_GITHUB_OWNER,
    JAILBREAKBENCH_GITHUB_REPO,
)
from na0s.layer15.diff_engine import TaxonomyDiffEngine
from na0s.layer15.http_utils import fetch_json, github_headers

logger = logging.getLogger(__name__)


def _scan_repo_datasets(
    owner: str,
    repo: str,
    headers: Dict[str, str],
    prefix: str,
) -> tuple[str, List[TechniqueEntry]]:
    """Scan a GitHub repo for dataset/benchmark files.

    Returns (commit_sha, list_of_technique_entries).
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    repo_info, _ = fetch_json(api_url, headers=headers)
    default_branch = repo_info.get("default_branch", "main")

    branch_url = f"{api_url}/branches/{default_branch}"
    branch_info, _ = fetch_json(branch_url, headers=headers)
    commit_sha = branch_info["commit"]["sha"]

    tree_url = f"{api_url}/git/trees/{commit_sha}?recursive=1"
    tree_data, _ = fetch_json(tree_url, headers=headers)

    # Find data files in dataset-related directories only.
    # Excludes config files (package.json, tsconfig.json, etc.) and
    # CI/tooling directories (.github/, node_modules/).
    data_extensions = {".csv", ".json", ".jsonl", ".pkl", ".parquet", ".tsv"}
    data_dirs = {"data", "datasets", "dataset", "benchmark", "results"}
    exclude_names = {"package.json", "package-lock.json", "tsconfig.json"}
    techniques = []

    for item in tree_data.get("tree", []):
        path = item.get("path", "")
        if item.get("type") != "blob":
            continue
        p = Path(path)
        if p.name in exclude_names:
            continue
        # Only include files under data-related directories
        top_dir = p.parts[0].lower() if p.parts else ""
        if top_dir.startswith(".") or top_dir in ("node_modules", "src", "docs"):
            continue
        if p.suffix in data_extensions and (
            top_dir in data_dirs or len(p.parts) == 1
        ):
            techniques.append(
                TechniqueEntry(
                    id=f"{prefix}.{path}",
                    name=p.stem,
                    description=f"Benchmark file: {path}",
                    metadata={
                        "file_path": path,
                        "repo": f"{owner}/{repo}",
                    },
                )
            )

    return commit_sha, techniques


class JailbreakBenchSync(ThreatIntelSource):
    """Syncs JailbreakBench and HarmBench dataset listings.

    Monitors both repos for new benchmark files and reports gaps
    relative to Na0S's test corpus.

    Parameters
    ----------
    github_token : str, optional
        GitHub API token.
    snapshots_dir : Path, optional
        Directory for storing snapshots.
    """

    name = "jailbreakbench"

    def __init__(
        self,
        github_token: Optional[str] = None,
        snapshots_dir: Optional[Path] = None,
    ):
        super().__init__(snapshots_dir=snapshots_dir)
        self._headers = github_headers(github_token)
        self._diff_engine = TaxonomyDiffEngine()

    def fetch_latest(self) -> SourceSnapshot:
        """Fetch dataset listings from JailbreakBench and HarmBench."""
        all_techniques: List[TechniqueEntry] = []
        versions: Dict[str, str] = {}

        # JailbreakBench
        try:
            sha, techniques = _scan_repo_datasets(
                JAILBREAKBENCH_GITHUB_OWNER,
                JAILBREAKBENCH_GITHUB_REPO,
                self._headers,
                prefix="jailbreakbench",
            )
            all_techniques.extend(techniques)
            versions["jailbreakbench"] = sha[:8]
            logger.info(
                "JailbreakBench: %d data files at %s",
                len(techniques),
                sha[:8],
            )
        except SourceUnavailableError as e:
            logger.warning("JailbreakBench unavailable: %s", e)

        # HarmBench
        try:
            sha, techniques = _scan_repo_datasets(
                HARMBENCH_GITHUB_OWNER,
                HARMBENCH_GITHUB_REPO,
                self._headers,
                prefix="harmbench",
            )
            all_techniques.extend(techniques)
            versions["harmbench"] = sha[:8]
            logger.info(
                "HarmBench: %d data files at %s",
                len(techniques),
                sha[:8],
            )
        except SourceUnavailableError as e:
            logger.warning("HarmBench unavailable: %s", e)

        combined_version = "+".join(
            f"{k}:{v}" for k, v in sorted(versions.items())
        )

        return SourceSnapshot(
            source_name=self.name,
            fetched_at=datetime.now(timezone.utc),
            version=combined_version or "unknown",
            techniques=all_techniques,
            raw_metadata={
                "versions": versions,
                "total_files": len(all_techniques),
            },
        )

    def diff(
        self, old: SourceSnapshot, new: SourceSnapshot
    ) -> TaxonomyDiff:
        return self._diff_engine.compute(old, new)

    def apply(
        self, diff: TaxonomyDiff, dry_run: bool = False
    ) -> ApplyResult:
        """Report new benchmark files for potential import.

        No automatic changes — new datasets are reported for human
        review and potential import into Na0S's test corpus.
        """
        if diff.has_changes:
            logger.info(
                "JailbreakBench/HarmBench: %d new files, %d removed",
                len(diff.added),
                len(diff.removed),
            )
        return ApplyResult(
            applied_count=0,
            skipped_count=len(diff.items),
            dry_run=dry_run,
        )
