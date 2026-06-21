#!/usr/bin/env python3
"""Eval decontamination check — block F14 scenarios from leaking into training data.

Walks the F14 scenario library under ``data/eval/scenarios/v0.1/`` and collects
the ``stable_id`` (SHA-256 of NFKC-normalized text) of every scenario. Then
walks the training-data roots (``data/processed/``, ``data/staging/``,
``data/aggregated/``) and computes the stable_id of each row's ``text`` column.
Any intersection is a contamination event — the candidate model was trained on
eval data and the promotion gate would give artificially-inflated metrics.

The exact-hash leg only catches verbatim copies. The optional ``--near-dup``
leg (MinHash + LSH, reusing ``na0s.dataset.near_duplicate``) additionally
catches paraphrase / light-edit leaks; it is warning-only unless ``--strict``.
It is off by default so the always-on CI gate stays fast on the full corpus.

A run that scans zero training rows fails loud (exit 2) rather than reporting
a false "clean" — an empty/unbuilt corpus cannot certify decontamination.

Usage:
    python scripts/check_eval_decontamination.py
    python scripts/check_eval_decontamination.py --training-roots data/processed/ data/staging/
    python scripts/check_eval_decontamination.py --near-dup            # warn on paraphrases
    python scripts/check_eval_decontamination.py --strict              # near-dup fatal too

Exit codes:
    0 — no overlap (near-dup matches, if any, were warning-only)
    1 — contamination: exact overlap, or near-dup overlap under --strict
    2 — configuration / IO error, or empty corpus (no --allow-empty-corpus)
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
from na0s.dataset.near_duplicate import (  # noqa: E402
    LSH_BANDS,
    LSH_ROWS_PER_BAND,
    MINHASH_JACCARD_THRESHOLD,
    jaccard_from_minhash,
    lsh_buckets,
    minhash_signature,
)


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


def _collect_scenario_texts(scenarios_dir: Path) -> list[tuple[str, str, Path]]:
    """Return ``[(text, scenario_name, source_yaml_path), ...]``.

    The text-level granularity mirrors ``_collect_scenario_ids`` (one entry
    per single-prompt payload and per multi-turn turn) so the near-dup leg
    catches a training row that paraphrases any individual turn.
    """
    rows: list[tuple[str, str, Path]] = []
    for scn in load_scenarios_dir(scenarios_dir):
        source = Path(getattr(scn, "_source_file", scenarios_dir))
        if scn.type == ScenarioType.SINGLE_PROMPT and scn.payload:
            rows.append((scn.payload, scn.name, source))
        elif scn.type == ScenarioType.MULTI_TURN:
            for idx, turn in enumerate(scn.turns):
                rows.append((turn.text, f"{scn.name}[turn={idx}]", source))
    return rows


def scan_exact(
    scenarios_dir: Path,
    training_roots: list[Path],
) -> tuple[list[dict], int, int]:
    """Run the exact stable_id overlap scan.

    Returns ``(overlaps, n_training_rows_scanned, n_scenario_ids)``. The row
    count lets the CLI distinguish "scanned a real corpus and found it clean"
    from "scanned nothing" (an empty/unbuilt corpus must not pass silently).
    """
    scenario_ids = _collect_scenario_ids(scenarios_dir)
    overlaps: list[dict] = []
    n_rows = 0
    if not scenario_ids:
        return overlaps, n_rows, 0
    for text, source, row in _iter_training_texts(training_roots):
        n_rows += 1
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
    return overlaps, n_rows, len(scenario_ids)


def find_overlaps(
    scenarios_dir: Path,
    training_roots: list[Path],
) -> list[dict]:
    """Return list of exact-overlap records (empty if no contamination).

    Thin back-compat wrapper around :func:`scan_exact`.
    """
    overlaps, _, _ = scan_exact(scenarios_dir, training_roots)
    return overlaps


def find_near_dup_overlaps(
    scenarios_dir: Path,
    training_roots: list[Path],
    threshold: float = MINHASH_JACCARD_THRESHOLD,
) -> list[dict]:
    """Return near-duplicate (paraphrase / light-edit) overlap records.

    Exact stable_id collisions are excluded here — they are already reported
    by :func:`scan_exact`; this leg surfaces the *additional* leaks that an
    exact hash misses (whitespace/casing/token-level edits, partial copies).

    Uses MinHash + LSH (``na0s.dataset.near_duplicate``) so cost is roughly
    linear in the training corpus rather than ``scenarios x rows``. The
    default ``threshold`` is the repo-canonical ``MINHASH_JACCARD_THRESHOLD``
    (0.8) — not a new arbitrary constant.
    """
    scenario_rows = _collect_scenario_texts(scenarios_dir)
    if not scenario_rows:
        return []
    # Pre-compute scenario signatures + their exact ids (to skip exact hits).
    scenario_sigs: list[list[int]] = []
    scenario_exact_ids: set[str] = set()
    bucket_to_scn: dict[int, list[int]] = {}
    for i, (text, _name, _src) in enumerate(scenario_rows):
        sig = minhash_signature(text)
        scenario_sigs.append(sig)
        scenario_exact_ids.add(compute_stable_id(text))
        for key in lsh_buckets(sig, LSH_BANDS, LSH_ROWS_PER_BAND):
            bucket_to_scn.setdefault(key, []).append(i)

    overlaps: list[dict] = []
    seen: set[tuple[int, str, int]] = set()
    for text, source, row in _iter_training_texts(training_roots):
        if compute_stable_id(text) in scenario_exact_ids:
            continue  # exact dup — already reported by scan_exact
        sig = minhash_signature(text)
        # LSH candidate set: scenarios sharing at least one band bucket.
        candidates: set[int] = set()
        for key in lsh_buckets(sig, LSH_BANDS, LSH_ROWS_PER_BAND):
            candidates.update(bucket_to_scn.get(key, ()))
        for i in candidates:
            sim = jaccard_from_minhash(sig, scenario_sigs[i])
            if sim >= threshold:
                dedup_key = (i, str(source), row)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                scn_text, scn_name, scn_source = scenario_rows[i]
                overlaps.append({
                    "scenario_name": scn_name,
                    "scenario_source": str(scn_source),
                    "training_file": str(source),
                    "training_row": row,
                    "jaccard": round(sim, 4),
                })
    return overlaps


def main() -> int:
    args = _parse_args()

    scenarios_dir = Path(args.scenarios_dir)
    if not scenarios_dir.exists():
        print(f"ERROR: scenarios dir not found: {scenarios_dir}", file=sys.stderr)
        return 2

    training_roots = [Path(p) for p in (args.training_roots or _DEFAULT_TRAINING_ROOTS)]
    near_dup_enabled = args.near_dup or args.strict

    exact_overlaps, n_rows, n_ids = scan_exact(scenarios_dir, training_roots)
    near_overlaps = (
        find_near_dup_overlaps(scenarios_dir, training_roots, args.near_dup_threshold)
        if near_dup_enabled else []
    )

    print("=" * 70)
    print("  Na0S F14 Eval Decontamination Check")
    print("=" * 70)
    print(f"  Scenarios dir:    {scenarios_dir}")
    print(f"  Training roots:   {[str(r) for r in training_roots]}")
    print(f"  Scenario ids:     {n_ids}")
    print(f"  Training rows:    {n_rows}")
    print(f"  Near-dup leg:     {'on (threshold=%s)' % args.near_dup_threshold if near_dup_enabled else 'off'}")
    print()

    # Empty/unbuilt corpus must fail loud — silently passing here would
    # certify "no contamination" against zero rows (false green).
    if n_ids > 0 and n_rows == 0 and not args.allow_empty_corpus:
        print(
            "  ERROR: scanned 0 training rows — corpus is empty or unbuilt; "
            "cannot certify decontamination.\n"
            "  (run scripts/process_data.py first, or pass --allow-empty-corpus)",
            file=sys.stderr,
        )
        print("=" * 70)
        return 2

    if exact_overlaps:
        print(f"  CONTAMINATION DETECTED (exact): {len(exact_overlaps)} overlap(s)")
        for o in exact_overlaps:
            print(f"    scenario={o['scenario_name']}  (from {o['scenario_source']})")
            print(f"      -> {o['training_file']} at row {o['training_row']}")

    if near_overlaps:
        label = "CONTAMINATION DETECTED (near-dup)" if args.strict else "WARNING (near-dup)"
        print(f"  {label}: {len(near_overlaps)} match(es) >= {args.near_dup_threshold}")
        for o in near_overlaps:
            print(
                f"    scenario={o['scenario_name']}  (from {o['scenario_source']})"
                f"  jaccard={o['jaccard']}"
            )
            print(f"      -> {o['training_file']} at row {o['training_row']}")

    fatal = bool(exact_overlaps) or (bool(near_overlaps) and args.strict)
    if not fatal:
        if near_overlaps:
            print("  (near-dup matches are warnings; pass --strict to make them fatal)")
        else:
            print("  OK — no scenario stable_ids found in training data.")
    print("=" * 70)
    return 1 if fatal else 0


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
        "--near-dup", action="store_true",
        help=(
            "Also run the MinHash/LSH near-duplicate leg (catches paraphrase / "
            "light-edit leaks that exact hashing misses). Warning-only unless "
            "--strict. Off by default to keep the always-on gate fast."
        ),
    )
    parser.add_argument(
        "--near-dup-threshold", type=float, default=MINHASH_JACCARD_THRESHOLD,
        help=(
            "Jaccard threshold for the near-dup leg "
            f"(default: repo-canonical MINHASH_JACCARD_THRESHOLD={MINHASH_JACCARD_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Enable the near-dup leg AND treat near-dup matches as fatal (exit 1).",
    )
    parser.add_argument(
        "--allow-empty-corpus", action="store_true",
        help="Do not fail (exit 2) when zero training rows are scanned.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
