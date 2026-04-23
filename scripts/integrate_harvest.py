#!/usr/bin/env python3
"""Integrate harvest/scrape discoveries into the training pipeline.

Default behavior routes discovered records through ``quarantine.py --ingest``
using per-source JSONL artifacts.  Tier3/tier4 sources therefore cannot reach
training until they are validated and explicitly promoted.

Legacy behavior can still write directly to a CSV with
``--no-ingest-via-quarantine``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from glob import glob

from scripts import quarantine


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HARVEST_DIR = "data/harvest"
DEFAULT_SCRAPE_DIR = "data/scraped"
DEFAULT_OUTPUT = "data/raw/harvested_samples.csv"
DEFAULT_STAGING_DIR = "data/staging/integrate_harvest"
DEFAULT_MIN_CONFIDENCE = 0.0  # accept all by default
DEFAULT_MIN_TEXT_LENGTH = 10
DEFAULT_UNKNOWN_SOURCE = "social/unknown"


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
                print(
                    f"  WARNING: skipping malformed JSON at {path}:{lineno}",
                    file=sys.stderr,
                )


def collect_harvest_records(harvest_dir, min_text_length, include_descriptions=False):
    """Collect records from weekly harvest output.

    Harvest entries are metadata (arXiv abstracts, HF dataset descriptions,
    GitHub repo blurbs) — NOT labeled prompt samples. Historically this
    function emitted them as ``label=0`` (benign) records, which silently
    poisoned training: a paper abstract about jailbreaking would be fed to
    the model as a benign prompt, teaching it that text discussing attacks
    is safe.

    Default behaviour: return ``[]``. Set ``include_descriptions=True``
    (CLI: ``--include-harvest-descriptions``) to opt back into the legacy
    behaviour, e.g. for one-off corpus-bootstrap runs where the operator
    has reviewed the output and accepts the noise.
    """
    records = []

    if not include_descriptions:
        print(
            "  Harvest descriptions skipped (default). "
            "Pass --include-harvest-descriptions to ingest arXiv abstracts / "
            "HF descriptions as benign training data (warning: noisy)."
        )
        return records

    jsonl_path = os.path.join(harvest_dir, "new_datasets.jsonl")
    if not os.path.isfile(jsonl_path):
        print(f"  No harvest JSONL found at {jsonl_path}")
        return records

    count = 0
    for entry in read_jsonl(jsonl_path):
        text = str(entry.get("description", "")).strip()
        if not text or len(text) < min_text_length:
            continue

        source_hint = str(entry.get("source", "unknown")).strip().lower()
        source_id = f"harvest/{source_hint or 'unknown'}"
        records.append({
            "text": text,
            "label": 0,
            "source": source_id,
        })
        count += 1

    print(f"  Harvest: {count} record(s) from {jsonl_path}")
    return records


def collect_scrape_records(scrape_dir, min_confidence, min_text_length):
    """Collect records from social scraper output files.

    Reads merged scrape output first, otherwise falls back to snapshots.
    Deduplicates by normalized text and preserves per-row source IDs so trust
    tiers are resolved correctly during quarantine ingest.
    """
    records = []
    seen_texts = set()

    jsonl_files = []
    merged = os.path.join(scrape_dir, "merged_scrape.jsonl")
    if os.path.isfile(merged):
        jsonl_files.append(merged)
    else:
        jsonl_files.extend(sorted(glob(os.path.join(scrape_dir, "scrape_*.jsonl"))))

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
            text = str(entry.get("text", "")).strip()
            if not text:
                continue

            if len(text) < min_text_length:
                skipped_short += 1
                continue

            try:
                confidence = float(entry.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 0.0

            if confidence < min_confidence:
                skipped_confidence += 1
                continue

            norm = re.sub(r"\s+", " ", text.lower()).strip()
            if norm in seen_texts:
                skipped_dup += 1
                continue
            seen_texts.add(norm)

            label = entry.get("label", 0)
            try:
                label = int(label)
            except (ValueError, TypeError):
                label = 0
            if label not in (0, 1):
                label = 0

            source_id = str(entry.get("source", DEFAULT_UNKNOWN_SOURCE)).strip()
            if not source_id:
                source_id = DEFAULT_UNKNOWN_SOURCE

            records.append({
                "text": text,
                "label": label,
                "source": source_id,
            })
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
    """Write records to a legacy CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        for rec in records:
            writer.writerow([rec["text"], rec["label"]])


def write_jsonl(records, output_path):
    """Write records to JSONL."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sanitize_source(source_id):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("_")
    return safe or "unknown"


def ingest_via_quarantine(
    records,
    staging_dir,
    dry_run=False,
    quarantine_config=None,
):
    """Stage records per source and ingest with quarantine routing."""
    if quarantine_config:
        quarantine.TRUST_TIERS_PATH = quarantine_config

    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["source"]].append(rec)

    if not grouped:
        return {
            "sources": 0,
            "staged": 0,
            "quarantined": 0,
            "direct_pass": 0,
            "errors": 0,
        }

    os.makedirs(staging_dir, exist_ok=True)

    config = None
    if not dry_run:
        config = quarantine.load_trust_config()

    summary = {
        "sources": len(grouped),
        "staged": 0,
        "quarantined": 0,
        "direct_pass": 0,
        "errors": 0,
    }

    for source_id in sorted(grouped.keys()):
        source_records = grouped[source_id]
        safe_name = _sanitize_source(source_id)
        source_hash = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:8]
        staged_path = os.path.join(staging_dir, f"{safe_name}_{source_hash}.jsonl")
        write_jsonl(source_records, staged_path)
        summary["staged"] += len(source_records)
        print(
            f"  Staged {len(source_records)} record(s) for '{source_id}' "
            f"-> {staged_path}"
        )

        if dry_run:
            continue

        result = quarantine.ingest(staged_path, source_id, config=config)
        action = result.get("action")
        if action == "quarantined":
            summary["quarantined"] += 1
        elif action == "direct_pass":
            summary["direct_pass"] += 1
        else:
            summary["errors"] += 1

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Integrate harvest/scrape records into training. Default route "
            "is quarantine ingest; use --no-ingest-via-quarantine for "
            "legacy direct CSV output."
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
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=(
            "Legacy CSV output path used only when "
            "--no-ingest-via-quarantine is set."
        ),
    )
    parser.add_argument(
        "--staging-dir",
        default=DEFAULT_STAGING_DIR,
        help=(
            "Directory for per-source staged JSONL files prior to quarantine "
            f"ingest (default: {DEFAULT_STAGING_DIR})."
        ),
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
        "--ingest-via-quarantine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Route discovered records through scripts.quarantine ingest "
            "(default: true)."
        ),
    )
    parser.add_argument(
        "--quarantine-config",
        default=None,
        help=(
            "Optional trust tier config path passed to quarantine.py "
            "(default: data/trust_tiers.yaml)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and report counts without ingesting or writing outputs.",
    )
    parser.add_argument(
        "--include-harvest-descriptions",
        action="store_true",
        default=False,
        help=(
            "Opt in to ingesting weekly harvest descriptions (arXiv abstracts, "
            "HF dataset descriptions) as benign training data. Default OFF: "
            "harvest entries are metadata, not labeled prompts, and treating "
            "them as benign poisons training (e.g., paper abstracts about "
            "jailbreaking get labeled as safe text)."
        ),
    )
    args = parser.parse_args(argv)

    print("Na0S Harvest-to-Training Integration")
    print(f"  Harvest dir:            {args.harvest_dir}")
    print(f"  Scrape dir:             {args.scrape_dir}")
    print(f"  Min confidence:         {args.min_confidence}")
    print(f"  Min text len:           {args.min_text_length}")
    print(f"  Ingest via quarantine:  {args.ingest_via_quarantine}")
    if args.ingest_via_quarantine:
        print(f"  Staging dir:            {args.staging_dir}")
    else:
        print(f"  Legacy output CSV:      {args.output}")
    print()

    print("Collecting harvest data ...")
    harvest_recs = collect_harvest_records(
        args.harvest_dir,
        args.min_text_length,
        include_descriptions=args.include_harvest_descriptions,
    )

    print("Collecting scrape data ...")
    scrape_recs = collect_scrape_records(
        args.scrape_dir,
        args.min_confidence,
        args.min_text_length,
    )

    all_records = harvest_recs + scrape_recs
    injection_count = sum(1 for rec in all_records if rec["label"] == 1)
    benign_count = sum(1 for rec in all_records if rec["label"] == 0)
    unique_sources = sorted({rec["source"] for rec in all_records})

    print()
    print(f"{'=' * 65}")
    print("Integration Summary")
    print(f"{'=' * 65}")
    print(f"  Total records:         {len(all_records)}")
    print(f"    Injection (1):       {injection_count}")
    print(f"    Benign (0):          {benign_count}")
    print(f"  From harvest:          {len(harvest_recs)}")
    print(f"  From scrape:           {len(scrape_recs)}")
    print(f"  Distinct source IDs:   {len(unique_sources)}")
    print(f"{'=' * 65}")

    if not all_records:
        print(
            "\nNo records to ingest. Run weekly_harvest.py and/or "
            "social_scraper.py first."
        )
        return 0

    if args.ingest_via_quarantine:
        print("\nRouting records through quarantine ingest ...")
        summary = ingest_via_quarantine(
            all_records,
            args.staging_dir,
            dry_run=args.dry_run,
            quarantine_config=args.quarantine_config,
        )
        print("\nQuarantine ingest summary")
        print(f"  Sources staged:       {summary['sources']}")
        print(f"  Records staged:       {summary['staged']}")
        print(f"  Source buckets quarantined: {summary['quarantined']}")
        print(f"  Source buckets direct-pass: {summary['direct_pass']}")
        print(f"  Source buckets failed:      {summary['errors']}")

        if args.dry_run:
            print("\n[DRY RUN] No quarantine ingest executed.")
            return 0

        if summary["errors"] > 0:
            print("\nERROR: One or more source buckets failed quarantine ingest.")
            return 1

        print(
            "\nRun `python -m scripts.quarantine --validate-quarantined` "
            "and explicit promotion before `process_data.py`."
        )
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
