#!/usr/bin/env python3
"""Per-taxonomy-code coverage of the F14 eval-scenario corpus — ADVISORY report.

Reports which canonical attack classes the live eval set (``data/eval/scenarios/
v0.1/``) actually has scenario data for, and which are zero-data. The code
universe is derived AT RUNTIME from the taxonomy (data/taxonomy.yaml) — never
hardcoded.

This script's PRIMARY input is the SCENARIO corpus, NOT ``data/datasets.yaml``
``category``. The ``--datasets`` hook is a documented forward-compat no-op
(it will union the registry ``taxonomy_codes`` provenance axis once that becomes
a coverage axis); today it only loads + validates the registry as a smoke check.

Exit codes:

    0  default (advisory) — zero-data codes are a legitimate present-day state,
       so the report never fails the pipeline by default.
    1  --strict AND (any zero-data code OR any non-canonical observed code).
    2  configuration / data error (e.g. a scenario dir is missing).

Examples
--------
    python scripts/taxonomy_coverage.py
    python scripts/taxonomy_coverage.py --json
    python scripts/taxonomy_coverage.py --strict
    python scripts/taxonomy_coverage.py --scenario-dir data/eval/scenarios/v0.1
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

from na0s.eval.coverage import (  # noqa: E402
    STATUS_ZERO_DATA,
    CoverageReport,
    compute_taxonomy_coverage,
)

_DEFAULT_SCENARIO_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "v0.1"
_DEFAULT_DRAFTS_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "_drafts"
_DEFAULT_JSON_OUT = _REPO_ROOT / "benchmarks" / "results" / "taxonomy_coverage.json"
_DEFAULT_DATASETS = _REPO_ROOT / "data" / "datasets.yaml"
_DEFAULT_MATRIX = _REPO_ROOT / "docs" / "COVERAGE_MATRIX.md"

_MATRIX_BEGIN = "<!-- BEGIN taxonomy_coverage (auto) -->"
_MATRIX_END = "<!-- END taxonomy_coverage (auto) -->"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="taxonomy_coverage",
        description="ADVISORY per-taxonomy-code coverage of the F14 eval "
        "scenario corpus (reads scenarios, not datasets.yaml; never edits "
        "v0.1/).",
    )
    p.add_argument(
        "--scenario-dir",
        action="append",
        dest="scenario_dirs",
        default=None,
        metavar="DIR",
        help="Scenario dir to count (repeatable). "
        "Default: data/eval/scenarios/v0.1.",
    )
    p.add_argument(
        "--include-drafts",
        action="store_true",
        help="Also count data/eval/scenarios/_drafts (off by default).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help=f"Write the report JSON to {_DEFAULT_JSON_OUT.relative_to(_REPO_ROOT)} "
        "and print it to stdout.",
    )
    p.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help="Override the --json artifact destination (default: "
        f"{_DEFAULT_JSON_OUT.relative_to(_REPO_ROOT)}). Implies --json. Lets a "
        "test or ad-hoc run emit the artifact without clobbering the committed "
        "source-of-truth file.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any code is zero-data or any non-canonical code is "
        "observed (default is advisory exit 0).",
    )
    p.add_argument(
        "--write-matrix",
        action="store_true",
        help="Opt-in: write/replace an auto section in docs/COVERAGE_MATRIX.md "
        "(off by default; the matrix is owner-maintained — JSON is the source "
        "of truth).",
    )
    p.add_argument(
        "--datasets",
        nargs="?",
        const=str(_DEFAULT_DATASETS),
        default=None,
        metavar="PATH",
        help="Forward-compat NO-OP hook: load+validate data/datasets.yaml "
        "taxonomy_codes (registry-provenance axis). Does not change coverage "
        "today; the scenario corpus is the coverage input.",
    )
    return p


def _print_human(report: CoverageReport) -> None:
    print("F14 TAXONOMY COVERAGE (ADVISORY — reads scenarios, not datasets.yaml)")
    print("=" * 70)
    print(f"total scenarios : {report.total_scenarios}")
    print("-" * 70)
    print(f"{'Code':<6} {'Severity':<10} {'#scenarios':>10}  Status")
    print("-" * 70)
    for r in report.rows:
        print(
            f"{r.code:<6} {r.severity:<10} {r.scenario_count:>10}  {r.status}"
        )
    print("-" * 70)
    print(
        f"{len(report.covered_codes)}/{report.total_codes} codes covered, "
        f"{len(report.zero_data_codes)} zero-data"
    )
    if report.non_canonical:
        print("-" * 70)
        print("NON-CANONICAL attack_category values observed (surfaced, not "
              "dropped):")
        for code, count in report.non_canonical:
            print(f"  {code!r}: {count}")


def _render_matrix_section(report: CoverageReport) -> str:
    lines = [
        _MATRIX_BEGIN,
        "",
        "## Taxonomy coverage (auto-generated by scripts/taxonomy_coverage.py)",
        "",
        f"{len(report.covered_codes)}/{report.total_codes} canonical top-level "
        f"codes have eval-scenario data; {len(report.zero_data_codes)} are "
        "zero-data.",
        "",
        "| Code | Severity | #scenarios | Status |",
        "| --- | --- | --- | --- |",
    ]
    for r in report.rows:
        lines.append(
            f"| {r.code} | {r.severity} | {r.scenario_count} | {r.status} |"
        )
    lines.append("")
    lines.append(_MATRIX_END)
    return "\n".join(lines) + "\n"


def _write_matrix(report: CoverageReport, path: Path) -> None:
    """Replace (or append) the auto section, leaving owner-authored prose alone."""
    section = _render_matrix_section(report)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if _MATRIX_BEGIN in text and _MATRIX_END in text:
            head = text.split(_MATRIX_BEGIN, 1)[0]
            tail = text.split(_MATRIX_END, 1)[1]
            new_text = head + section + tail
        else:
            sep = "" if text.endswith("\n") else "\n"
            new_text = text + sep + "\n" + section
    else:
        new_text = section
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    scenario_dirs = (
        [Path(d) for d in args.scenario_dirs]
        if args.scenario_dirs
        else [_DEFAULT_SCENARIO_DIR]
    )
    if args.include_drafts:
        scenario_dirs.append(_DEFAULT_DRAFTS_DIR)

    # Forward-compat no-op: load + validate the registry provenance axis so a
    # broken datasets.yaml is at least surfaced, without changing coverage.
    if args.datasets is not None:
        try:
            from na0s.eval.registry import (
                load_registry,
                validate_registry_codes,
            )

            errs = validate_registry_codes(load_registry(Path(args.datasets)))
            if errs:
                for e in errs:
                    print(
                        f"WARN: datasets.yaml {e.source}: {e.code!r} {e.reason}",
                        file=sys.stderr,
                    )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    try:
        report = compute_taxonomy_coverage(scenario_dirs)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json or args.json_out:
        json_out = Path(args.json_out) if args.json_out else _DEFAULT_JSON_OUT
        payload = json.dumps(report.to_dict(), indent=2)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    else:
        _print_human(report)

    if args.write_matrix:
        _write_matrix(report, _DEFAULT_MATRIX)
        print(f"wrote matrix section -> {_DEFAULT_MATRIX}")

    if args.strict and (report.zero_data_codes or report.non_canonical):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
