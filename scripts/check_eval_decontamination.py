#!/usr/bin/env python3
"""Eval decontamination check — block F14 scenarios from leaking into training data.

Walks the F14 scenario library under ``data/eval/scenarios/v0.1/`` and collects
the ``stable_id`` (SHA-256 of NFKC-normalized text) of every scenario. Then
walks the training-data roots (``data/processed/``, ``data/staging/``,
``data/aggregated/``) and computes the stable_id of each row's ``text`` column.
Any intersection is a contamination event — the candidate model was trained on
eval data and the promotion gate would give artificially-inflated metrics.

Usage:
    python scripts/check_eval_decontamination.py
    python scripts/check_eval_decontamination.py --training-roots data/processed/ data/staging/
    python scripts/check_eval_decontamination.py --strict

Exit codes:
    0 — no overlap
    1 — overlap found (one or more scenario stable_ids appear in training data)
    2 — configuration / IO error
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from na0s.eval.scenarios.loader import load_scenarios_dir  # noqa: E402
from na0s.eval.scenarios.schema import ScenarioType  # noqa: E402


_DEFAULT_SCENARIOS = _PROJECT_ROOT / "data" / "eval" / "scenarios" / "v0.1"
_DEFAULT_TRAINING_ROOTS = [
    _PROJECT_ROOT / "data" / "processed",
    _PROJECT_ROOT / "data" / "staging",
    _PROJECT_ROOT / "data" / "aggregated",
]


def compute_stable_id(text: str) -> str:
    """Return SHA-256 of NFKC + whitespace-collapsed text.

    Mirrors ``Na0SSample.__post_init__`` and ``Scenario._compute_stable_id``
    so training-row hashes compare equal to scenario hashes. Kept as a
    free function here to avoid a circular import of Na0SSample.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _collect_scenario_ids(scenarios_dir: Path) -> dict[str, tuple[str, Path]]:
    """Return ``{stable_id: (scenario_name, source_yaml_path)}``.

    Multi-turn scenarios contribute the stable_id of each turn's text in
    addition to the scenario-level id, so training rows that happen to
    match a single turn still get caught.
    """
    ids: dict[str, tuple[str, Path]] = {}
    for scn in load_scenarios_dir(scenarios_dir):
        source = Path(getattr(scn, "_source_file", scenarios_dir))
        if scn.type == ScenarioType.SINGLE_PROMPT and scn.payload:
            ids[compute_stable_id(scn.payload)] = (scn.name, source)
        elif scn.type == ScenarioType.MULTI_TURN:
            for idx, turn in enumerate(scn.turns):
                ids[compute_stable_id(turn.text)] = (
                    f"{scn.name}[turn={idx}]", source,
                )
        # stable_id-populated form (auto-computed in __post_init__)
        if scn.stable_id:
            ids.setdefault(scn.stable_id, (scn.name, source))
    return ids


def _iter_training_texts(roots: list[Path]):
    """Yield ``(text, source_path, row_num)`` for every text-bearing training row.

    Accepts CSV (expects a ``text`` column; case-insensitive) and JSONL
    (expects a ``text`` field). Silently skips non-matching files. Roots
    that don't exist are skipped — a training stage not yet populated is
    not a decontamination error.
    """
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".csv":
                yield from _iter_csv_texts(path)
            elif suffix in (".jsonl", ".ndjson"):
                yield from _iter_jsonl_texts(path)


def _iter_csv_texts(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        # Find the text column (case-insensitive).
        text_col = next(
            (c for c in reader.fieldnames if c.lower() == "text"),
            None,
        )
        if text_col is None:
            return
        for i, row in enumerate(reader, start=2):  # header is line 1
            text = row.get(text_col) or ""
            if text:
                yield text, path, i


def _iter_jsonl_texts(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") if isinstance(obj, dict) else None
            if text:
                yield text, path, i


def find_overlaps(
    scenarios_dir: Path,
    training_roots: list[Path],
) -> list[dict]:
    """Return list of overlap records (empty if no contamination)."""
    scenario_ids = _collect_scenario_ids(scenarios_dir)
    if not scenario_ids:
        return []
    overlaps: list[dict] = []
    for text, source, row in _iter_training_texts(training_roots):
        sid = compute_stable_id(text)
        hit = scenario_ids.get(sid)
        if hit is not None:
            scn_name, scn_source = hit
            overlaps.append({
                "stable_id": sid,
                "scenario_name": scn_name,
                "scenario_source": str(scn_source),
                "training_file": str(source),
                "training_row": row,
            })
    return overlaps


def main() -> int:
    args = _parse_args()

    scenarios_dir = Path(args.scenarios_dir)
    if not scenarios_dir.exists():
        print(f"ERROR: scenarios dir not found: {scenarios_dir}", file=sys.stderr)
        return 2

    training_roots = [Path(p) for p in (args.training_roots or _DEFAULT_TRAINING_ROOTS)]

    overlaps = find_overlaps(scenarios_dir, training_roots)

    print("=" * 70)
    print("  Na0S F14 Eval Decontamination Check")
    print("=" * 70)
    print(f"  Scenarios dir:  {scenarios_dir}")
    print(f"  Training roots: {[str(r) for r in training_roots]}")
    print()
    if not overlaps:
        print("  OK — no scenario stable_ids found in training data.")
        print("=" * 70)
        return 0

    print(f"  CONTAMINATION DETECTED: {len(overlaps)} overlap(s)")
    for o in overlaps:
        print(
            f"    scenario={o['scenario_name']}  "
            f"(from {o['scenario_source']})"
        )
        print(
            f"      -> appears in {o['training_file']} "
            f"at row {o['training_row']}"
        )
    print("=" * 70)
    return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that F14 eval scenarios are not in training data",
    )
    parser.add_argument(
        "--scenarios-dir", default=str(_DEFAULT_SCENARIOS),
        help=f"Scenario YAML dir (default: {_DEFAULT_SCENARIOS})",
    )
    parser.add_argument(
        "--training-roots", nargs="*", default=None,
        help=(
            "One or more directories to scan for training rows "
            "(default: data/processed, data/staging, data/aggregated)"
        ),
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Reserved for future use — currently any overlap is fatal.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
