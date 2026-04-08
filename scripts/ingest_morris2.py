#!/usr/bin/env python3
"""Download and prepare the ComPromptMized / Morris II dataset for training.

Data sources (all from https://raw.githubusercontent.com/StavC/ComPromptMized/main/):
  1. Datasets/Jailbreaks.csv               — 70 raw worm payload texts
  2. DonkeyRail/Training_Samples/Experiment_Results_Virus.csv  — ~204K rows
  3. DonkeyRail/Training_Samples/Experiment_Results_Benign.csv — ~59K rows

Usage::

    python -m scripts.ingest_morris2 [--output-dir data/raw/morris2] [--max-samples 5000]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "data", "raw", "morris2")
DEFAULT_MAX_SAMPLES = 5000

_BASE_URL = "https://raw.githubusercontent.com/StavC/Here-Comes-the-AI-Worm/master/"
_JAILBREAKS_PATH = "Datasets/Jailbreaks.csv"
_VIRUS_PATH = "DonkeyRail/Training_Samples/Experiment_Results_Virus.csv"
_BENIGN_PATH = "DonkeyRail/Training_Samples/Experiment_Results_Benign.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    """Download *url* to *dest* using urllib (stdlib only)."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.isfile(dest):
        print(f"  [cached] {dest}")
        return
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def _text_hash(text: str) -> str:
    """Deterministic hash for sampling."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _parse_jailbreaks(path: str) -> list[dict]:
    """Parse Jailbreaks.csv and return records."""
    records = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row.get("JailBreak") or "").strip()
            if text:
                records.append({
                    "text": text,
                    "label": 1,
                    "source": "morris2_jailbreak",
                })
    return records


def _parse_donkeyrail_streaming(
    path: str,
    label: int,
    source: str,
    max_samples: int,
) -> list[dict]:
    """Stream-parse a DonkeyRail CSV, collecting (text, hash) pairs.

    Samples deterministically by sorting on text hash and taking the first
    *max_samples* entries.  Reads line-by-line to avoid loading the full
    ~30 MB file into memory.
    """
    candidates: list[tuple[str, str]] = []  # (hash, text)
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row.get("Reply") or "").strip()
            if not text:
                continue
            # Verify label column matches expectation (defensive)
            row_label = (row.get("Virus Label") or "").strip()
            if row_label not in ("", str(label)):
                continue
            h = _text_hash(text)
            candidates.append((h, text))

    # Deterministic sampling: sort by hash, take first N
    candidates.sort(key=lambda pair: pair[0])
    sampled = candidates[:max_samples]
    return [
        {"text": text, "label": label, "source": source}
        for _, text in sampled
    ]


def _write_jsonl(records: list[dict], path: str) -> None:
    """Write records as JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def ingest(output_dir: str = DEFAULT_OUTPUT_DIR, max_samples: int = DEFAULT_MAX_SAMPLES) -> dict:
    """Download, parse, sample, and write the combined JSONL.

    Returns summary stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Download raw CSVs
    jailbreaks_csv = os.path.join(output_dir, "Jailbreaks.csv")
    virus_csv = os.path.join(output_dir, "Experiment_Results_Virus.csv")
    benign_csv = os.path.join(output_dir, "Experiment_Results_Benign.csv")

    _download(_BASE_URL + _JAILBREAKS_PATH, jailbreaks_csv)
    _download(_BASE_URL + _VIRUS_PATH, virus_csv)
    _download(_BASE_URL + _BENIGN_PATH, benign_csv)

    # 2. Parse
    print("\nParsing Jailbreaks.csv ...")
    jailbreak_records = _parse_jailbreaks(jailbreaks_csv)
    print(f"  {len(jailbreak_records)} jailbreak payloads")

    print("Parsing Experiment_Results_Virus.csv (streaming) ...")
    virus_records = _parse_donkeyrail_streaming(
        virus_csv, label=1, source="morris2_virus_reply", max_samples=max_samples,
    )
    print(f"  {len(virus_records)} virus replies (sampled from up to {max_samples})")

    print("Parsing Experiment_Results_Benign.csv (streaming) ...")
    benign_records = _parse_donkeyrail_streaming(
        benign_csv, label=0, source="morris2_benign_reply", max_samples=max_samples,
    )
    print(f"  {len(benign_records)} benign replies (sampled from up to {max_samples})")

    # 3. Combine and write
    all_records = jailbreak_records + virus_records + benign_records
    combined_path = os.path.join(output_dir, "morris2_combined.jsonl")
    _write_jsonl(all_records, combined_path)
    print(f"\nWrote {combined_path}")

    # 4. Summary
    worm_count = sum(1 for r in all_records if r["label"] == 1)
    benign_count = sum(1 for r in all_records if r["label"] == 0)
    stats = {
        "total": len(all_records),
        "worm": worm_count,
        "benign": benign_count,
        "jailbreak_payloads": len(jailbreak_records),
        "virus_replies": len(virus_records),
        "benign_replies": len(benign_records),
    }

    print("\n--- Summary ---")
    print(f"  Total samples : {stats['total']}")
    print(f"  Worm (label=1): {stats['worm']}")
    print(f"  Benign (label=0): {stats['benign']}")
    print(f"  Jailbreak payloads: {stats['jailbreak_payloads']}")
    print(f"  Virus replies: {stats['virus_replies']}")
    print(f"  Benign replies: {stats['benign_replies']}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare the ComPromptMized / Morris II dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Max virus/benign samples to keep (default: {DEFAULT_MAX_SAMPLES}).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ingest(output_dir=args.output_dir, max_samples=args.max_samples)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
