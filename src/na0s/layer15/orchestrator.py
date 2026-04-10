
"""Layer 15 Orchestrator — runs all threat intel sources and produces reports.

This is the main entry point invoked by the GitHub Actions workflow and
the CLI. It:

1. Instantiates all enabled ThreatIntelSource implementations
2. Runs each source's sync() in sequence (to respect rate limits)
3. Aggregates results into a combined report
4. Generates a Markdown summary suitable for a GitHub PR or issue

DESIGN NOTE: Sources run sequentially, not in parallel, because they
share the GitHub API rate limit budget. If we ever move to per-source
rate tracking, we could parallelize with ThreadPoolExecutor.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from na0s.layer15.base import (
    SourceUnavailableError,
    SyncReport,
    ThreatIntelSource,
)
from na0s.layer15.config import REPO_ROOT, SNAPSHOTS_DIR
from na0s.layer15.diff_engine import TaxonomyDiffEngine

logger = logging.getLogger(__name__)


class Orchestrator:
    """Runs all threat intel sources and produces aggregated reports.

    Parameters
    ----------
    sources : list of ThreatIntelSource, optional
        Sources to sync. If None, uses the default set (ATLAS + Garak).
    dry_run : bool
        If True, no changes are applied to the taxonomy.
    output_dir : Path, optional
        Where to write reports. Defaults to data/threat_intel_reports/.
    """

    def __init__(
        self,
        sources: Optional[List[ThreatIntelSource]] = None,
        dry_run: bool = False,
        output_dir: Optional[Path] = None,
    ):
        self.sources = sources if sources is not None else self._default_sources()
        self.dry_run = dry_run
        self.output_dir = output_dir or REPO_ROOT / "data" / "threat_intel_reports"
        self._diff_engine = TaxonomyDiffEngine()

    @staticmethod
    def _default_sources() -> List[ThreatIntelSource]:
        """Instantiate the default set of sources (P0 + P1)."""
        from na0s.layer15.aiid_sync import AiidSync
        from na0s.layer15.atlas_sync import AtlasSync
        from na0s.layer15.garak_sync import GarakSync
        from na0s.layer15.jailbreakbench_sync import JailbreakBenchSync
        from na0s.layer15.owasp_sync import OwaspSync
        from na0s.layer15.safetyprompts_sync import SafetyPromptsSync

        return [
            AtlasSync(),
            GarakSync(),
            AiidSync(),
            OwaspSync(),
            JailbreakBenchSync(),
            SafetyPromptsSync(),
        ]

    def run(self) -> List[SyncReport]:
        """Execute all source syncs and save reports.

        Returns a list of SyncReport objects (one per source).
        Sources that fail are logged and skipped — partial success
        is better than total failure.
        """
        reports: List[SyncReport] = []

        from na0s.layer15.http_utils import check_rate_limit

        for source in self.sources:
            # Proactive rate limit check between sources
            if not check_rate_limit():
                logger.warning(
                    "GitHub API rate limit low — skipping remaining sources"
                )
                break

            logger.info("--- Syncing: %s ---", source.name)
            try:
                report = source.sync(dry_run=self.dry_run)
                reports.append(report)
                logger.info(
                    "%s: %d changes",
                    source.name,
                    len(report.diff.items),
                )
            except SourceUnavailableError as e:
                logger.error("Source %s unavailable: %s", source.name, e)
            except Exception:
                logger.exception("Unexpected error syncing %s", source.name)

        # Save reports
        if reports:
            self._save_reports(reports)

        return reports

    def _save_reports(self, reports: List[SyncReport]) -> None:
        """Save individual and combined reports to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Individual source reports
        for report in reports:
            self._diff_engine.save_report(report.diff, self.output_dir)

        # Combined summary
        combined_path = self.output_dir / f"combined_{ts}.json"
        combined = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": self.dry_run,
            "sources": [r.to_dict() for r in reports],
            "total_changes": sum(len(r.diff.items) for r in reports),
        }
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        logger.info("Combined report saved to %s", combined_path)

    def generate_pr_body(self, reports: List[SyncReport]) -> str:
        """Generate a Markdown body suitable for a GitHub PR or issue.

        Parameters
        ----------
        reports : list of SyncReport
            Results from run().

        Returns
        -------
        str
            Markdown-formatted PR body.
        """
        lines: List[str] = []
        lines.append("# Threat Intelligence Sync Report")
        lines.append("")
        lines.append(
            f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        lines.append(f"**Mode**: {'Dry run' if self.dry_run else 'Live'}")
        lines.append("")

        total_changes = sum(len(r.diff.items) for r in reports)
        if total_changes == 0:
            lines.append("No changes detected across any sources.")
            return "\n".join(lines)

        lines.append(f"**Total changes**: {total_changes}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Source | Version | Added | Removed | Modified | Duration |")
        lines.append("|---|---|---|---|---|---|")
        for r in reports:
            lines.append(
                f"| {r.source_name} "
                f"| `{r.diff.old_version or '(none)'}` → `{r.diff.new_version}` "
                f"| {len(r.diff.added)} "
                f"| {len(r.diff.removed)} "
                f"| {len(r.diff.modified)} "
                f"| {r.duration_seconds:.1f}s |"
            )
        lines.append("")

        # Unmapped items needing review
        all_unmapped = []
        for r in reports:
            all_unmapped.extend(r.diff.unmapped)
        if all_unmapped:
            lines.append("## ⚠️ Items Needing Manual Review")
            lines.append("")
            lines.append("These items could not be automatically mapped to Na0S taxonomy:")
            lines.append("")
            lines.append("| Source | ID | Name |")
            lines.append("|---|---|---|")
            for item in all_unmapped:
                lines.append(
                    f"| — | `{item.technique_id}` | {item.technique_name} |"
                )
            lines.append("")

        # Per-source details
        for r in reports:
            if r.diff.has_changes:
                lines.append(f"---")
                lines.append("")
                lines.append(self._diff_engine.to_markdown(r.diff))
                lines.append("")

        # Errors
        all_errors = []
        for r in reports:
            if r.result.errors:
                all_errors.extend(
                    (r.source_name, e) for e in r.result.errors
                )
        if all_errors:
            lines.append("## Errors")
            lines.append("")
            for source, error in all_errors:
                lines.append(f"- **{source}**: {error}")
            lines.append("")

        return "\n".join(lines)


def main() -> None:
    """CLI entry point for running the threat intel sync."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Na0S Layer 15 — External Threat Intelligence Sync"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without applying",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=["atlas", "garak", "aiid", "owasp", "jailbreakbench", "safetyprompts"],
        help="Specific sources to sync (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for reports",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print JSON report to stdout",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build source list
    sources = None
    if args.sources:
        from na0s.layer15.aiid_sync import AiidSync
        from na0s.layer15.atlas_sync import AtlasSync
        from na0s.layer15.garak_sync import GarakSync
        from na0s.layer15.jailbreakbench_sync import JailbreakBenchSync
        from na0s.layer15.owasp_sync import OwaspSync
        from na0s.layer15.safetyprompts_sync import SafetyPromptsSync

        source_map = {
            "atlas": AtlasSync,
            "garak": GarakSync,
            "aiid": AiidSync,
            "owasp": OwaspSync,
            "jailbreakbench": JailbreakBenchSync,
            "safetyprompts": SafetyPromptsSync,
        }
        sources = [source_map[s]() for s in args.sources]

    orchestrator = Orchestrator(
        sources=sources,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )
    reports = orchestrator.run()

    if args.json_output:
        combined = {
            "dry_run": args.dry_run,
            "sources": [r.to_dict() for r in reports],
            "total_changes": sum(len(r.diff.items) for r in reports),
        }
        print(json.dumps(combined, indent=2))
    else:
        print(orchestrator.generate_pr_body(reports))


if __name__ == "__main__":
    main()
