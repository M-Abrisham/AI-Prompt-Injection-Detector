#!/usr/bin/env python3
"""Merge taxonomy JSONL with existing combined_data.csv.

De-duplicates by text hash, assigns train/val/test splits, and writes
JSONL files with the full Na0SSample schema.

Usage:
    python scripts/merge_taxonomy.py \\
        --taxonomy data/staging/taxonomy_samples.jsonl \\
        --existing data/processed/combined_data.csv \\
        --output-dir data/processed/splits

    python scripts/merge_taxonomy.py \\
        --taxonomy data/staging/taxonomy_samples.jsonl \\
        --existing data/processed/combined_data.csv \\
        --output-dir data/processed/splits \\
        --split-ratio 0.8,0.1,0.1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys

# Path setup
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from na0s.dataset.schema import DataSplit, Na0SSample


def _text_hash(text: str) -> str:
    """SHA-256 hex digest of normalised text (stripped, lowered)."""
    normalised = text.strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _load_taxonomy_jsonl(path: str) -> list[dict]:
    """Load JSONL file, return list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _load_existing_csv(path: str) -> set[str]:
    """Load existing CSV and return set of text hashes for dedup."""
    hashes = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "")
            hashes.add(_text_hash(text))
    return hashes


def _load_existing_csv_as_samples(path: str) -> list[dict]:
    """Load existing CSV rows as Na0SSample dicts (legacy enriched)."""
    samples = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = Na0SSample.from_legacy_csv_row(row)
            samples.append(sample.to_dict())
    return samples


def _parse_split_ratio(ratio_str: str) -> tuple[float, float, float]:
    """Parse 'train,val,test' ratio string."""
    parts = [float(x.strip()) for x in ratio_str.split(",")]
    if len(parts) != 3:
        raise ValueError("split-ratio must have 3 values, got {}".format(len(parts)))
    total = sum(parts)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("split-ratio must sum to 1.0, got {}".format(total))
    return (parts[0], parts[1], parts[2])


def _assign_splits(
    samples: list[dict],
    ratios: tuple[float, float, float],
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Shuffle and split samples into train/val/test."""
    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits: dict[str, list[dict]] = {
        DataSplit.TRAIN.value: [],
        DataSplit.VAL.value: [],
        DataSplit.TEST.value: [],
    }

    for i, s in enumerate(samples):
        if i < n_train:
            s["split"] = DataSplit.TRAIN.value
            splits[DataSplit.TRAIN.value].append(s)
        elif i < n_train + n_val:
            s["split"] = DataSplit.VAL.value
            splits[DataSplit.VAL.value].append(s)
        else:
            s["split"] = DataSplit.TEST.value
            splits[DataSplit.TEST.value].append(s)

    return splits


def merge(
    taxonomy_path: str,
    existing_path: str | None,
    output_dir: str,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, int]:
    """Run the full merge pipeline. Returns counts per split."""
    # Load taxonomy samples
    taxonomy_samples = _load_taxonomy_jsonl(taxonomy_path)

    # Deduplicate against existing
    if existing_path and os.path.exists(existing_path):
        existing_hashes = _load_existing_csv(existing_path)
        before = len(taxonomy_samples)
        taxonomy_samples = [
            s for s in taxonomy_samples
            if _text_hash(s.get("text", "")) not in existing_hashes
        ]
        deduped = before - len(taxonomy_samples)
        print("Deduplication: removed {} duplicates from taxonomy".format(deduped))
    else:
        existing_hashes = set()

    # Also deduplicate within taxonomy itself
    seen = set()
    unique = []
    for s in taxonomy_samples:
        h = _text_hash(s.get("text", ""))
        if h not in seen:
            seen.add(h)
            unique.append(s)
    internal_dupes = len(taxonomy_samples) - len(unique)
    if internal_dupes:
        print("Internal dedup: removed {} duplicates within taxonomy".format(internal_dupes))
    taxonomy_samples = unique

    # Assign splits
    splits = _assign_splits(taxonomy_samples, split_ratio, seed)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    counts = {}
    for split_name, samples in splits.items():
        out_path = os.path.join(output_dir, "{}.jsonl".format(split_name))
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        counts[split_name] = len(samples)
        print("  {}: {} samples -> {}".format(split_name, len(samples), out_path))

    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge taxonomy JSONL with existing CSV"
    )
    parser.add_argument(
        "--taxonomy",
        required=True,
        help="Path to taxonomy_samples.jsonl",
    )
    parser.add_argument(
        "--existing",
        default=None,
        help="Path to existing combined_data.csv (for dedup)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for split JSONL files",
    )
    parser.add_argument(
        "--split-ratio",
        default="0.8,0.1,0.1",
        help="Train,val,test split ratio (default: 0.8,0.1,0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle (default: 42)",
    )
    args = parser.parse_args(argv)

    ratios = _parse_split_ratio(args.split_ratio)
    counts = merge(
        taxonomy_path=args.taxonomy,
        existing_path=args.existing,
        output_dir=args.output_dir,
        split_ratio=ratios,
        seed=args.seed,
    )
    total = sum(counts.values())
    print("\nTotal: {} samples across {} splits".format(total, len(counts)))
    return counts


if __name__ == "__main__":
    main()
