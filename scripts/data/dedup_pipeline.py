#!/usr/bin/env python3
"""Deduplication pipeline wrapping near_duplicate.py with Na0SSample integration.

Processes JSONL and legacy CSV files, flagging duplicates (never deleting)
with ``is_duplicate=True`` and assigning ``near_dup_cluster`` IDs.

Usage::

    # JSONL input
    python scripts/data/dedup_pipeline.py --input data.jsonl --output deduped.jsonl

    # Legacy CSV input
    python scripts/data/dedup_pipeline.py --legacy-csv combined_data.csv --output deduped.jsonl

    # With options
    python scripts/data/dedup_pipeline.py --input data.jsonl --output deduped.jsonl \
        --simhash-threshold 3 --minhash-threshold 0.8 --max-rows 5000 --report stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Optional

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Also ensure scripts/ is importable for near_duplicate
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from near_duplicate import (
    simhash,
    hamming_distance,
    minhash_signature,
    jaccard_from_minhash,
    find_simhash_duplicates,
    find_minhash_duplicates,
    build_clusters,
)
from src.na0s.data_schema import Na0SSample, DataLabel

# ── Text normalisation ────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Collapse whitespace and strip for consistent dedup comparison."""
    return _WS_RE.sub(" ", text).strip()


# ── Core pipeline ─────────────────────────────────────────────────


def _assign_dedup_flags(
    samples: list[Na0SSample],
    simhash_threshold: int = 3,
    minhash_threshold: float = 0.8,
) -> dict:
    """Flag duplicates on a list of Na0SSample objects (in-place).

    Returns a stats dict.
    """
    t0 = time.time()
    n = len(samples)

    if n == 0:
        return {
            "total": 0,
            "exact_dups": 0,
            "near_dups": 0,
            "unique": 0,
            "clusters": 0,
            "dedup_rate": 0.0,
            "elapsed_sec": round(time.time() - t0, 3),
        }

    texts = [normalise_text(s.text) for s in samples]

    # ── Phase 1: exact duplicates ─────────────────────────────────
    seen_exact: dict[str, int] = {}
    exact_dup_indices: set[int] = set()
    for i, t in enumerate(texts):
        if t in seen_exact:
            exact_dup_indices.add(i)
        else:
            seen_exact[t] = i

    # ── Phase 2: SimHash near-duplicates ──────────────────────────
    sim_pairs = find_simhash_duplicates(texts, threshold=simhash_threshold)

    # ── Phase 3: MinHash near-duplicates ──────────────────────────
    min_pairs = find_minhash_duplicates(texts, threshold=minhash_threshold)

    # Combine all pairs (normalise to (i, j, score) with i < j)
    all_pairs_set: set[tuple[int, int]] = set()
    combined_pairs: list[tuple[int, int, float | int]] = []

    for i, j, dist in sim_pairs:
        pair = (min(i, j), max(i, j))
        if pair not in all_pairs_set:
            all_pairs_set.add(pair)
            combined_pairs.append((pair[0], pair[1], dist))

    for i, j, sim in min_pairs:
        pair = (min(i, j), max(i, j))
        if pair not in all_pairs_set:
            all_pairs_set.add(pair)
            combined_pairs.append((pair[0], pair[1], sim))

    # Also add exact dup pairs (pair each exact dup with its first occurrence)
    for idx in exact_dup_indices:
        first = seen_exact[texts[idx]]
        pair = (min(first, idx), max(first, idx))
        if pair not in all_pairs_set:
            all_pairs_set.add(pair)
            combined_pairs.append((pair[0], pair[1], 0))

    # ── Build clusters ────────────────────────────────────────────
    clusters = build_clusters(combined_pairs, n)

    # Assign cluster IDs and is_duplicate flags
    # In each cluster, the first index (lowest) is the representative;
    # all others are flagged as duplicates.
    near_dup_count = 0
    for ci, cluster in enumerate(clusters):
        cluster_id = f"cluster_{ci}"
        representative = cluster[0]  # lowest index
        for idx in cluster:
            samples[idx].near_dup_cluster = cluster_id
            if idx != representative:
                samples[idx].is_duplicate = True
                if idx in exact_dup_indices:
                    pass  # counted as exact
                else:
                    near_dup_count += 1

    # Flag remaining exact dups that aren't in a near-dup cluster
    for idx in exact_dup_indices:
        if not samples[idx].is_duplicate:
            samples[idx].is_duplicate = True

    exact_count = len(exact_dup_indices)
    # near_dup_count already excludes exact dups within clusters
    unique_count = n - exact_count - near_dup_count
    total_flagged = exact_count + near_dup_count

    stats = {
        "total": n,
        "exact_dups": exact_count,
        "near_dups": near_dup_count,
        "unique": unique_count,
        "clusters": len(clusters),
        "dedup_rate": round(total_flagged / n, 4) if n > 0 else 0.0,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    return stats


# ── JSONL pipeline ────────────────────────────────────────────────


def _sample_from_dict(d: dict) -> Na0SSample:
    """Reconstruct a Na0SSample from a JSONL dict."""
    label = d.get("label", "benign")
    if isinstance(label, str):
        label = DataLabel(label)
    from src.na0s.data_schema import DataSplit
    split_val = d.get("split", "train")
    if isinstance(split_val, str):
        split_val = DataSplit(split_val)
    return Na0SSample(
        text=d.get("text", ""),
        label=label,
        augmentation_type=d.get("augmentation_type"),
        technique_id=d.get("technique_id"),
        source=d.get("source"),
        source_id=d.get("source_id"),
        language=d.get("language", "en"),
        split=split_val,
        difficulty=d.get("difficulty"),
        license=d.get("license"),
        license_url=d.get("license_url"),
        attribution=d.get("attribution"),
        hf_dataset=d.get("hf_dataset"),
        quality_score=d.get("quality_score"),
        is_duplicate=d.get("is_duplicate", False),
        near_dup_cluster=d.get("near_dup_cluster"),
        created_at=d.get("created_at"),
    )


def dedup_jsonl_streaming(
    input_path: str,
    output_path: str,
    simhash_threshold: int = 3,
    minhash_threshold: float = 0.8,
    max_rows: Optional[int] = None,
) -> dict:
    """Process a JSONL file, flagging duplicates on Na0SSample objects.

    Reads all samples, flags duplicates (is_duplicate=True, near_dup_cluster),
    then writes all samples (none deleted) to output_path.

    Returns stats dict.
    """
    samples: list[Na0SSample] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(_sample_from_dict(d))

    stats = _assign_dedup_flags(samples, simhash_threshold, minhash_threshold)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    return stats


# ── Legacy CSV pipeline ──────────────────────────────────────────


def dedup_legacy_csv_streaming(
    csv_path: str,
    output_jsonl_path: str,
    simhash_threshold: int = 3,
    minhash_threshold: float = 0.8,
    max_rows: Optional[int] = None,
) -> dict:
    """Convert legacy CSV to Na0SSample JSONL while deduping.

    Reads the CSV (text, label, augmentation_type columns), converts each
    row to Na0SSample via from_legacy_csv_row(), runs dedup, writes JSONL.

    Returns stats dict.
    """
    samples: list[Na0SSample] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            try:
                sample = Na0SSample.from_legacy_csv_row(row)
                samples.append(sample)
            except ValueError:
                # Skip rows with unrecognised labels
                continue

    stats = _assign_dedup_flags(samples, simhash_threshold, minhash_threshold)

    os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    return stats


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dedup pipeline: flag duplicates on Na0SSample JSONL/CSV data.",
    )
    parser.add_argument("--input", dest="input_path", default=None,
                        help="Path to input JSONL file.")
    parser.add_argument("--legacy-csv", dest="legacy_csv", default=None,
                        help="Path to legacy CSV (text,label,augmentation_type).")
    parser.add_argument("--output", dest="output_path", required=True,
                        help="Path for output JSONL file.")
    parser.add_argument("--simhash-threshold", type=int, default=3,
                        help="SimHash Hamming distance threshold (default: 3).")
    parser.add_argument("--minhash-threshold", type=float, default=0.8,
                        help="MinHash Jaccard similarity threshold (default: 0.8).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows to process (for testing).")
    parser.add_argument("--report", dest="report_path", default=None,
                        help="Path to write JSON stats report.")
    args = parser.parse_args()

    if not args.input_path and not args.legacy_csv:
        parser.error("Provide --input (JSONL) or --legacy-csv (CSV).")

    if args.input_path and args.legacy_csv:
        parser.error("Provide only one of --input or --legacy-csv, not both.")

    if args.input_path:
        stats = dedup_jsonl_streaming(
            args.input_path,
            args.output_path,
            simhash_threshold=args.simhash_threshold,
            minhash_threshold=args.minhash_threshold,
            max_rows=args.max_rows,
        )
    else:
        stats = dedup_legacy_csv_streaming(
            args.legacy_csv,
            args.output_path,
            simhash_threshold=args.simhash_threshold,
            minhash_threshold=args.minhash_threshold,
            max_rows=args.max_rows,
        )

    print(json.dumps(stats, indent=2))

    if args.report_path:
        os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Report written to {args.report_path}")


if __name__ == "__main__":
    main()
