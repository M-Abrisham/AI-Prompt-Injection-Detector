#!/usr/bin/env python3
"""Harvest DRAFT F14 scenarios from a Layer-15 threat-intel snapshot (offline).

Thin CLI over :func:`na0s.eval.harvest.snapshot_to_scenarios`. It reads a
``{source}_snapshot.json`` written by a Layer-15 ``ThreatIntelSource``, runs its
techniques through the deterministic template pipeline (no network, no external
LLM), and writes provenance-traced DRAFT scenarios for human review. It NEVER
promotes — drafts land in ``data/eval/scenarios/_drafts/`` and must pass
``scripts/validate_draft_scenarios.py`` and a human before reaching ``v0.1/``.

All business logic lives in :mod:`na0s.eval.harvest.sources`; this file only
parses args, calls the harvester, and reports.

NOTE: the import path is the canonical ``na0s.layer15.*`` (this branch's package
name). A v1.0.0 rename to ``na0s.threat_intel`` is pending on ``main``; these
imports rename with it.

Usage::

    python scripts/harvest_from_l15_snapshot.py path/to/aiid_snapshot.json \\
        --source-slug aiid-2026-06 \\
        --origin https://incidentdatabase.ai \\
        --retrieved 2026-06-17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from na0s.eval.harvest import (
    DEFAULT_DRAFTS_DIR,
    IntelProvenance,
    IntelScenarioExtractor,
    snapshot_to_scenarios,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a Layer-15 threat-intel snapshot and write DRAFT F14 "
            "scenarios (offline, deterministic, no LLM)."
        )
    )
    parser.add_argument(
        "snapshot_path",
        type=Path,
        help="Path to a {source}_snapshot.json file.",
    )
    parser.add_argument(
        "--source-slug",
        required=True,
        help="Short kebab-case id for this intel source (draft filename stem).",
    )
    parser.add_argument(
        "--origin",
        required=True,
        help="Origin URL or dataset id, folded into each scenario description.",
    )
    parser.add_argument(
        "--retrieved",
        required=True,
        help="Retrieval date (ISO YYYY-MM-DD) when the snapshot was synced.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DRAFTS_DIR,
        help="Where to write the draft YAML (default: data/eval/scenarios/_drafts).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    provenance = IntelProvenance(
        source_slug=args.source_slug,
        origin=args.origin,
        retrieved=args.retrieved,
    )

    try:
        report = snapshot_to_scenarios(args.snapshot_path, provenance)
    except FileNotFoundError as exc:
        print(f"[harvest-l15] ERROR — {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[harvest-l15] ERROR — bad snapshot: {exc}", file=sys.stderr)
        return 2

    if report.scenarios:
        out_path = IntelScenarioExtractor().write_drafts(
            report.scenarios,
            output_dir=args.output_dir,
            source_slug=args.source_slug,
        )
        print(
            f"[harvest-l15] OK — {report.emitted_count} draft(s), "
            f"{report.skipped_count} skipped -> {out_path}"
        )
        for scenario in report.scenarios:
            print(f"  + emitted: {scenario.name}  [{scenario.attack_category}]")
    else:
        print(
            f"[harvest-l15] OK — {report.emitted_count} draft(s), "
            f"{report.skipped_count} skipped (nothing written)"
        )

    for skip in report.skipped:
        print(f"  - skipped: {skip.identifier}  ({skip.reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
