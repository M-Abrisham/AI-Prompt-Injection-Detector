#!/usr/bin/env python3
"""Run Layer 16 detection baseline.

Usage:
    python scripts/run_baseline.py              # run and print results
    python scripts/run_baseline.py --save v1    # run, print, and save as v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo's src/ is importable when running from the repo root
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from na0s.layer16.testing.baseline_runner import BaselineRunner  # noqa: E402

_BASELINES_DIR = _REPO / "src" / "na0s" / "layer16" / "baselines"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Layer 16 detection baseline")
    parser.add_argument(
        "--save",
        metavar="VERSION",
        default=None,
        help="Save results as <VERSION>_baseline.json (e.g. --save v1)",
    )
    parser.add_argument(
        "--fixtures",
        metavar="DIR",
        default=None,
        help="Override the fixture directory path",
    )
    args = parser.parse_args()

    fixture_dir = Path(args.fixtures) if args.fixtures else None
    runner = BaselineRunner(fixture_dir=fixture_dir)

    print("Loading scenarios...")
    results = runner.run_full_suite()
    print()
    runner.print_summary(results)

    if args.save:
        out = _BASELINES_DIR / f"{args.save}_baseline.json"
        runner.save_baseline(results, out)
        print(f"\nBaseline saved to {out}")


if __name__ == "__main__":
    main()
