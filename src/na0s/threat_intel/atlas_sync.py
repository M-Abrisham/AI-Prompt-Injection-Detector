"""MITRE ATLAS YAML Sync — fetches ATLAS technique definitions.

ATLAS (Adversarial Threat Landscape for AI Systems) defines attack
techniques against ML systems with IDs like AML.T0043. This module:

1. Fetches technique YAML files from the mitre/atlas GitHub repo via API
2. Parses technique IDs, names, descriptions
3. Maps ATLAS IDs to Na0S taxonomy categories using a mapping file
4. Produces a diff against the last-known snapshot
5. Flags unmapped techniques for manual review

DESIGN NOTE: We use the GitHub Contents API (not git clone) because:
- We only need the techniques directory, not the full repo
- API responses include ETags for conditional requests (304 Not Modified)
- Avoids needing git on the runner

Rate limits: 5,000 req/hr authenticated, 60 unauthenticated.
We fetch the tree in one call + individual files, staying well within limits.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from na0s.threat_intel.base import (
    ApplyResult,
    SchemaValidationError,
    SourceSnapshot,
    SourceUnavailableError,
    TaxonomyDiff,
    TechniqueEntry,
    ThreatIntelSource,
)
from na0s.threat_intel.config import (
    ATLAS_API_URL,
    ATLAS_MAPPING_FILE,
    ATLAS_RAW_URL,
    ATLAS_TECHNIQUES_PATH,
)
from na0s.threat_intel.diff_engine import TaxonomyDiffEngine
from na0s.threat_intel.http_utils import fetch_json, fetch_text, github_headers

logger = logging.getLogger(__name__)

# Lazy-loaded; only needed if YAML mapping file exists
_yaml = None


def _get_yaml():
    """Lazy import for PyYAML — avoids hard dependency at import time."""
    global _yaml
    if _yaml is None:
        try:
            import yaml

            _yaml = yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for ATLAS sync. "
                "Install it with: pip install PyYAML"
            )
    return _yaml



# Backward-compat aliases for tests that mock atlas_sync._fetch_json etc.
_github_headers = github_headers
_fetch_json = fetch_json
_fetch_text = fetch_text


class AtlasSync(ThreatIntelSource):
    """Syncs MITRE ATLAS technique definitions into Na0S taxonomy.

    Parameters
    ----------
    github_token : str, optional
        GitHub API token. Falls back to GITHUB_TOKEN env var.
    snapshots_dir : Path, optional
        Directory for storing snapshots.
    mapping_file : Path, optional
        YAML file mapping ATLAS IDs to Na0S categories.
    """

    name = "atlas"

    def __init__(
        self,
        github_token: Optional[str] = None,
        snapshots_dir: Optional[Path] = None,
        mapping_file: Optional[Path] = None,
    ):
        super().__init__(snapshots_dir=snapshots_dir)
        self.github_token = github_token
        self.mapping_file = mapping_file or ATLAS_MAPPING_FILE
        self._headers = _github_headers(github_token)
        self._mapping: Optional[Dict[str, str]] = None
        self._diff_engine = TaxonomyDiffEngine()

    def _load_mapping(self) -> Dict[str, str]:
        """Load ATLAS→Na0S mapping from YAML file.

        Returns a dict of {atlas_id: na0s_category_technique}.
        If the file doesn't exist, returns an empty dict (first run).
        """
        if self._mapping is not None:
            return self._mapping

        if not self.mapping_file.exists():
            logger.info(
                "No ATLAS mapping file at %s — all techniques will be UNMAPPED",
                self.mapping_file,
            )
            self._mapping = {}
            return self._mapping

        yaml = _get_yaml()
        with open(self.mapping_file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Expected format: {atlas_id: na0s_id} e.g. {"AML.T0043": "D1.5"}
        self._mapping = {str(k): str(v) for k, v in data.items()}
        logger.info("Loaded %d ATLAS→Na0S mappings", len(self._mapping))
        return self._mapping

    def fetch_latest(self) -> SourceSnapshot:
        """Fetch the latest ATLAS techniques from GitHub.

        Uses the Git Trees API to list all YAML files in the techniques
        directory, then fetches each one. This is more efficient than
        the Contents API for directories with many files.
        """
        # Step 1: Get the latest commit SHA on the default branch
        repo_info, _ = _fetch_json(ATLAS_API_URL, headers=self._headers)
        default_branch = repo_info.get("default_branch", "main")

        branch_url = f"{ATLAS_API_URL}/branches/{default_branch}"
        branch_info, _ = _fetch_json(branch_url, headers=self._headers)
        commit_sha = branch_info["commit"]["sha"]

        logger.info(
            "ATLAS default branch=%s, HEAD=%s", default_branch, commit_sha
        )

        # Step 2: Get the tree for the techniques directory
        tree_url = (
            f"{ATLAS_API_URL}/git/trees/{commit_sha}"
            f"?recursive=1"
        )
        tree_data, _ = _fetch_json(tree_url, headers=self._headers)

        technique_files = [
            item
            for item in tree_data.get("tree", [])
            if (
                item["path"].startswith(ATLAS_TECHNIQUES_PATH)
                and item["path"].endswith((".yaml", ".yml"))
                and item["type"] == "blob"
            )
        ]

        logger.info("Found %d ATLAS technique files", len(technique_files))

        # Step 3: Fetch and parse each technique file
        mapping = self._load_mapping()
        techniques: List[TechniqueEntry] = []

        for file_info in technique_files:
            file_path = file_info["path"]
            raw_url = f"{ATLAS_RAW_URL}/{default_branch}/{file_path}"
            try:
                content = _fetch_text(raw_url, headers=self._headers)
                parsed = self._parse_technique_yaml(content, mapping)
                if parsed:
                    techniques.extend(parsed)
            except (SourceUnavailableError, SchemaValidationError) as e:
                logger.warning("Skipping %s: %s", file_path, e)
                continue

        return SourceSnapshot(
            source_name=self.name,
            fetched_at=datetime.now(timezone.utc),
            version=commit_sha,
            techniques=techniques,
            raw_metadata={
                "default_branch": default_branch,
                "technique_file_count": len(technique_files),
            },
        )

    def _parse_technique_yaml(
        self,
        content: str,
        mapping: Dict[str, str],
    ) -> List[TechniqueEntry]:
        """Parse an ATLAS YAML technique file into TechniqueEntries.

        ATLAS technique files can contain a single technique or a list.
        Each technique has at minimum: id, name, description.

        Raises SchemaValidationError if the structure is unexpected.
        """
        yaml = _get_yaml()
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise SchemaValidationError(f"Invalid YAML: {e}")

        if data is None:
            return []

        # Normalize to list
        if isinstance(data, dict):
            # Single technique or a wrapper with 'techniques' key
            if "id" in data:
                items = [data]
            elif "techniques" in data:
                items = data["techniques"]
            else:
                # Might be a single technique keyed by ID
                items = []
                for key, val in data.items():
                    if isinstance(val, dict):
                        val.setdefault("id", key)
                        items.append(val)
        elif isinstance(data, list):
            items = data
        else:
            raise SchemaValidationError(
                f"Unexpected YAML root type: {type(data).__name__}"
            )

        techniques = []
        for item in items:
            if not isinstance(item, dict):
                logger.debug("Skipping non-dict item in ATLAS YAML: %r", item)
                continue

            tid = item.get("id", "")
            name = item.get("name", "")

            if not tid:
                logger.debug("Skipping ATLAS entry with no ID: %r", item)
                continue

            na0s_cat = mapping.get(tid, "")

            techniques.append(
                TechniqueEntry(
                    id=str(tid),
                    name=str(name),
                    description=str(item.get("description", "")),
                    severity=str(item.get("severity", "")),
                    category=na0s_cat,
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in ("id", "name", "description", "severity")
                    },
                )
            )

        return techniques

    def diff(
        self, old: SourceSnapshot, new: SourceSnapshot
    ) -> TaxonomyDiff:
        """Compare old and new ATLAS snapshots."""
        return self._diff_engine.compute(old, new)

    def apply(
        self, diff: TaxonomyDiff, dry_run: bool = False
    ) -> ApplyResult:
        """Apply ATLAS diff to local taxonomy.

        For now, this generates a report of what would change. Full
        taxonomy modification (writing back to taxonomy.yaml) requires
        careful schema validation and is gated behind dry_run=False.

        DESIGN NOTE: We do NOT auto-write to taxonomy.yaml. Instead,
        we save the diff report and let the GitHub Actions workflow
        open a PR for human review. This is intentional — automated
        taxonomy changes need human approval.
        """
        if dry_run or not diff.has_changes:
            return ApplyResult(
                applied_count=0,
                skipped_count=len(diff.items),
                dry_run=dry_run,
            )

        # In non-dry-run mode, we still only update the mapping file
        # and diff reports — not taxonomy.yaml directly.
        new_mappings = 0
        for item in diff.added:
            if item.na0s_mapping:
                new_mappings += 1

        logger.info(
            "ATLAS apply: %d new techniques (%d pre-mapped, %d need review)",
            len(diff.added),
            new_mappings,
            len(diff.unmapped),
        )

        return ApplyResult(
            applied_count=new_mappings,
            skipped_count=len(diff.items) - new_mappings,
        )

    def suggest_mapping(
        self, technique: TechniqueEntry, taxonomy_categories: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Suggest Na0S category mappings for an unmapped ATLAS technique.

        Uses keyword matching between the ATLAS technique description
        and Na0S category names/descriptions to produce ranked suggestions.

        Parameters
        ----------
        technique : TechniqueEntry
            The unmapped ATLAS technique.
        taxonomy_categories : dict
            The Na0S taxonomy categories dict (from taxonomy.yaml).

        Returns
        -------
        list of (category_id, score) tuples, sorted by score descending.
        """
        # Build a simple text representation of the technique
        technique_text = (
            f"{technique.name} {technique.description}".lower()
        )

        scores: List[Tuple[str, float]] = []
        for cat_id, cat_data in taxonomy_categories.items():
            cat_name = cat_data.get("name", "").lower()
            cat_desc = cat_data.get("description", "").lower()
            cat_text = f"{cat_name} {cat_desc}"

            # Simple keyword overlap scoring
            technique_words = set(technique_text.split())
            cat_words = set(cat_text.split())
            # Remove common stopwords
            stopwords = {
                "the", "a", "an", "is", "are", "to", "of", "and",
                "in", "for", "on", "with", "that", "this", "by",
            }
            technique_words -= stopwords
            cat_words -= stopwords

            if not technique_words or not cat_words:
                continue

            overlap = technique_words & cat_words
            # Jaccard-like score
            score = len(overlap) / len(technique_words | cat_words)
            if score > 0:
                scores.append((cat_id, round(score, 3)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:5]  # Top 5 suggestions
