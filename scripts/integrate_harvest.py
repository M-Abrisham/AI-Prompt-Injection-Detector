#!/usr/bin/env python3
"""Convert harvested and scraped JSONL data into CSV for the training pipeline.

Reads JSONL files from ``data/harvest/`` and ``data/scraped/``, extracts
records that have ``text`` and ``label`` fields, and writes them to
``data/raw/harvested_samples.csv`` in the standard (text, label) format
expected by ``process_data.py``.

Usage::

    python scripts/integrate_harvest.py
    python scripts/integrate_harvest.py --harvest-dir data/harvest --scrape-dir data/scraped
    python scripts/integrate_harvest.py --output data/raw/harvested_samples.csv
    python scripts/integrate_harvest.py --min-confidence 0.6
    python scripts/integrate_harvest.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from glob import glob


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HARVEST_DIR = "data/harvest"
DEFAULT_SCRAPE_DIR = "data/scraped"
DEFAULT_OUTPUT = "data/raw/harvested_samples.csv"
DEFAULT_MIN_CONFIDENCE = 0.0  # accept all by default
DEFAULT_MIN_TEXT_LENGTH = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_jsonl(path):
    """Read a JSONL file and yield dicts, skipping malformed lines."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARNING: skipping malformed JSON at {path}:{lineno}",
                      file=sys.stderr)


def collect_harvest_records(harvest_dir, min_text_length):
    """Collect records from the harvest directory.

    The weekly harvest writes metadata about *discovered datasets*, not raw
    text samples.  However, ``new_datasets.jsonl`` may contain entries with
    ``description`` fields that can serve as lightweight samples for the
    "benign" class (they describe datasets, not inject anything).

    Returns a list of ``(text, label)`` tuples.
    """
    records = []
    jsonl_path = os.path.join(harvest_dir, "new_datasets.jsonl")
    if not os.path.isfile(jsonl_path):
        print(f"  No harvest JSONL found at {jsonl_path}")
        return records

    count = 0
    for entry in read_jsonl(jsonl_path):
        # Harvest entries are dataset metadata -- use description as text
        text = entry.get("description", "").strip()
        if not text or len(text) < min_text_length:
            continue
        # Harvest descriptions are benign (they describe datasets)
        label = 0
        records.append((text, label))
        count += 1

    print(f"  Harvest: {count} record(s) from {jsonl_path}")
    return records


def collect_scrape_records(scrape_dir, min_confidence, min_text_length):
    """Collect records from the scraped directory.

    The social scraper writes JSONL files with ``text``, ``label``, and
    ``confidence`` fields.  We read all JSONL files (merged + snapshots)
    and deduplicate by text content.

    Returns a list of ``(text, label)`` tuples.
    """
    records = []
    seen_texts = set()

    # Prefer the merged file first, then fall back to individual snapshots
    jsonl_files = []
    merged = os.path.join(scrape_dir, "merged_scrape.jsonl")
    if os.path.isfile(merged):
        jsonl_files.append(merged)
    else:
        # Collect individual scrape snapshot files
        jsonl_files.extend(sorted(glob(os.path.join(scrape_dir, "scrape_*.jsonl"))))

    # Also check static sub-directory
    static_dir = os.path.join(scrape_dir, "static")
    if os.path.isdir(static_dir):
        jsonl_files.extend(sorted(glob(os.path.join(static_dir, "*.jsonl"))))

    if not jsonl_files:
        print(f"  No scrape JSONL files found in {scrape_dir}")
        return records

    total = 0
    skipped_confidence = 0
    skipped_short = 0
    skipped_dup = 0

    for jsonl_path in jsonl_files:
        for entry in read_jsonl(jsonl_path):
            text = entry.get("text", "").strip()
            if not text:
                continue

            if len(text) < min_text_length:
                skipped_short += 1
                continue

            confidence = entry.get("confidence", 1.0)
            if confidence < min_confidence:
                skipped_confidence += 1
                continue

            # Deduplicate by normalized text
            norm = text.lower().strip()
            if norm in seen_texts:
                skipped_dup += 1
                continue
            seen_texts.add(norm)

            label = entry.get("label", 0)
            try:
                label = int(label)
            except (ValueError, TypeError):
                label = 0

            records.append((text, label))
            total += 1

    print(f"  Scrape: {total} record(s) from {len(jsonl_files)} file(s)")
    if skipped_confidence:
        print(f"    Skipped (low confidence): {skipped_confidence}")
    if skipped_short:
        print(f"    Skipped (too short):      {skipped_short}")
    if skipped_dup:
        print(f"    Skipped (duplicate):      {skipped_dup}")

    return records


def write_csv(records, output_path):
    """Write ``(text, label)`` records to a CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        for text, label in records:
            writer.writerow([text, label])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert harvested/scraped JSONL data to CSV for the training "
            "pipeline.  Output is written in (text, label) format compatible "
            "with process_data.py."
        ),
    )
    parser.add_argument(
        "--harvest-dir",
        default=DEFAULT_HARVEST_DIR,
        help=f"Harvest data directory (default: {DEFAULT_HARVEST_DIR}).",
    )
    parser.add_argument(
        "--scrape-dir",
        default=DEFAULT_SCRAPE_DIR,
        help=f"Scraped data directory (default: {DEFAULT_SCRAPE_DIR}).",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=(
            "Minimum confidence threshold for scraped records "
            f"(default: {DEFAULT_MIN_CONFIDENCE})."
        ),
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=DEFAULT_MIN_TEXT_LENGTH,
        help=(
            "Minimum text length to include a record "
            f"(default: {DEFAULT_MIN_TEXT_LENGTH})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and report counts without writing the output CSV.",
    )
    args = parser.parse_args(argv)

    print("Na0S Harvest-to-Training Integration")
    print(f"  Harvest dir:    {args.harvest_dir}")
    print(f"  Scrape dir:     {args.scrape_dir}")
    print(f"  Output:         {args.output}")
    print(f"  Min confidence: {args.min_confidence}")
    print(f"  Min text len:   {args.min_text_length}")
    print()

    all_records = []

    # Collect from harvest
    print("Collecting harvest data ...")
    harvest_recs = collect_harvest_records(
        args.harvest_dir, args.min_text_length,
    )
    all_records.extend(harvest_recs)

    # Collect from scrape
    print("Collecting scrape data ...")
    scrape_recs = collect_scrape_records(
        args.scrape_dir, args.min_confidence, args.min_text_length,
    )
    all_records.extend(scrape_recs)

    # Summary
    injection_count = sum(1 for _, lbl in all_records if lbl == 1)
    benign_count = sum(1 for _, lbl in all_records if lbl == 0)

    print()
    print(f"{'=' * 55}")
    print(f"Integration Summary")
    print(f"{'=' * 55}")
    print(f"  Total records:  {len(all_records)}")
    print(f"    Injection (1): {injection_count}")
    print(f"    Benign (0):    {benign_count}")
    print(f"  From harvest:   {len(harvest_recs)}")
    print(f"  From scrape:    {len(scrape_recs)}")
    print(f"{'=' * 55}")

    if not all_records:
        print("\nNo records to write.  Run weekly_harvest.py and/or "
              "social_scraper.py first to collect data.")
        return 0

    if args.dry_run:
        print("\n[DRY RUN] No output file written.")
        return 0

    write_csv(all_records, args.output)
    print(f"\nWrote {len(all_records)} records to {args.output}")
    print("This file will be picked up by process_data.py on the next run.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
