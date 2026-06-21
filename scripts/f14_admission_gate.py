#!/usr/bin/env python3
"""F14 scenario-admission gate CLI — REPORT-ONLY draft validation.

Validates DRAFT scenarios in ``data/eval/scenarios/_drafts/`` against the F14
admission contract (schema, taxonomy, exact + near-dup decontam, benign-twin,
soft trust) and prints a per-scenario ADMIT/REJECT report.

This tool NEVER writes to ``data/eval/scenarios/v0.1/`` and never promotes a
draft. Promotion stays a human PR step. Exit codes:

    0  every draft ADMITs
    1  at least one draft REJECTs
    2  configuration / data error (e.g. drafts dir missing)

Examples
--------
    python scripts/f14_admission_gate.py
    python scripts/f14_admission_gate.py /tmp/drafts --training-dir /tmp/train
    python scripts/f14_admission_gate.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when run directly from a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from na0s.eval.scenarios.admission_gate import (  # noqa: E402
    DEFAULT_NEAR_DUP_THRESHOLD,
    AdmissionReport,
    ScenarioAdmissionGate,
)

_DEFAULT_DRAFTS = _REPO_ROOT / "data" / "eval" / "scenarios" / "_drafts"
_DEFAULT_LIVE = _REPO_ROOT / "data" / "eval" / "scenarios" / "v0.1"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="f14_admission_gate",
        description="REPORT-ONLY admission gate for DRAFT F14 scenarios "
        "(never promotes; never writes v0.1/).",
    )
    p.add_argument(
        "drafts_dir",
        nargs="?",
        default=str(_DEFAULT_DRAFTS),
        help="Directory of DRAFT scenario YAMLs (default: data/eval/scenarios/_drafts).",
    )
    p.add_argument(
        "--training-dir",
        action="append",
        dest="training_dirs",
        default=None,
        metavar="DIR",
        help="Training-corpus dir to decontaminate against (repeatable). "
        "Default: data/raw + data/aggregated.",
    )
    p.add_argument(
        "--live-dir",
        default=str(_DEFAULT_LIVE),
        help="Live scenario set to decontaminate against (default: v0.1).",
    )
    p.add_argument(
        "--near-dup-threshold",
        type=float,
        default=DEFAULT_NEAR_DUP_THRESHOLD,
        help=f"Jaccard-proxy reject threshold (default {DEFAULT_NEAR_DUP_THRESHOLD}).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON instead of human-readable text.",
    )
    return p


def _report_to_dict(report: AdmissionReport) -> dict:
    return {
        "summary": {
            "total": report.total,
            "admitted": len(report.admitted),
            "rejected": len(report.rejected),
            "any_rejected": report.any_rejected,
            "training_sample_count": report.training_sample_count,
            "live_scenario_count": report.live_scenario_count,
            "decontam_methods": report.decontam_methods,
            "notes": report.notes,
        },
        "results": [
            {
                "name": r.name,
                "status": r.status,
                "checks_passed": r.checks_passed,
                "checks_failed": r.checks_failed,
                "reasons": r.reasons,
                "warnings": r.warnings,
            }
            for r in report.results
        ],
    }


def _print_human(report: AdmissionReport) -> None:
    print("F14 SCENARIO ADMISSION GATE (REPORT-ONLY — no promotion, no writes to v0.1/)")
    print("=" * 78)
    print(f"training samples : {report.training_sample_count}")
    print(f"live scenarios   : {report.live_scenario_count}")
    for method in report.decontam_methods:
        print(f"decontam         : {method}")
    for note in report.notes:
        print(f"note             : {note}")
    print("-" * 78)

    for r in report.results:
        marker = "ADMIT " if r.status == "ADMIT" else "REJECT"
        print(f"[{marker}] {r.name}")
        if r.checks_passed:
            print(f"         passed : {', '.join(r.checks_passed)}")
        for reason in r.reasons:
            print(f"         REJECT : {reason}")
        for warn in r.warnings:
            print(f"         warn   : {warn}")
    print("-" * 78)
    print(
        f"SUMMARY: {len(report.admitted)} admitted, "
        f"{len(report.rejected)} rejected of {report.total} draft(s)."
    )
    if report.any_rejected:
        print("RESULT: REJECT — at least one draft failed; do NOT promote.")
    else:
        print("RESULT: ADMIT — all drafts passed; safe for human review + promotion.")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    training_dirs = (
        [Path(d) for d in args.training_dirs] if args.training_dirs else None
    )
    gate = ScenarioAdmissionGate(
        training_dirs=training_dirs,
        live_dir=Path(args.live_dir),
        near_dup_threshold=args.near_dup_threshold,
    )

    try:
        report = gate.run(Path(args.drafts_dir))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_report_to_dict(report), indent=2))
    else:
        _print_human(report)

    return 1 if report.any_rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
