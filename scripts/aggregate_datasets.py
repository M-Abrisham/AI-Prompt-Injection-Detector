#!/usr/bin/env python3
"""Download public prompt-injection datasets, normalize, deduplicate, and merge.

Downloads datasets from HuggingFace using the Parquet API (raw HTTP, no
``datasets`` library), normalizes every record to a standard JSONL schema,
deduplicates by text hash, and writes merged output files.

Output schema (one JSON object per line)::

    {"text": "...", "label": 0|1, "source": "dataset_name", "category": "benign|malicious|injection"}

Usage
-----
    python scripts/aggregate_datasets.py
    python scripts/aggregate_datasets.py --output-dir data/aggregated --force
    python scripts/aggregate_datasets.py --datasets imoxto_cleaned,gandalf_rct
    python scripts/aggregate_datasets.py --max-per-dataset 5000 --skip-gated
"""

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import time

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "id": "imoxto/prompt_injection_cleaned_dataset-v2",
        "name": "imoxto_cleaned",
        "text_col": "text",
        "label_col": "labels",
        "label_map": {0: 0, 1: 1},
        "splits": ["train"],
        "source": "imoxto",
        "max_samples": 50000,
    },
    {
        "id": "geekyrakshit/prompt-injection-dataset",
        "name": "geekyrakshit",
        "text_col": "prompt",
        "label_col": "label",
        "label_map": {0: 0, 1: 1},
        "splits": ["train", "test"],
        "source": "geekyrakshit",
        "max_samples": 50000,
    },
    {
        "id": "Lakera/gandalf-rct",
        "name": "gandalf_rct",
        "text_col": "prompt",
        "label_col": "success",
        "label_map": {"true": 1, "false": 0, True: 1, False: 0},
        "splits": ["trial"],
        "source": "gandalf",
        "filter": {"kind": "prompt"},
        "max_samples": 50000,
    },
    {
        "id": "microsoft/llmail-inject-challenge",
        "name": "llmail_inject",
        "text_col": "body",
        "label_col": None,
        "fixed_label": 1,
        "splits": ["Phase1"],
        "source": "llmail",
        "max_samples": 50000,
    },
    {
        "id": "hackaprompt/hackaprompt-dataset",
        "name": "hackaprompt",
        "text_col": "user_input",
        "label_col": None,
        "fixed_label": 1,
        "splits": ["train"],
        "source": "hackaprompt",
        "gated": True,
        "max_samples": 10000,
    },
]

RANDOM_SEED = 42

HF_PARQUET_API = "https://huggingface.co/api/datasets/{dataset_id}/parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_requests():
    """Raise a clear error if the ``requests`` library is not installed."""
    if requests is None:
        print(
            "ERROR: the 'requests' package is required. "
            "Install it with:  pip install requests",
            file=sys.stderr,
        )
        sys.exit(1)


def _get_hf_token():
    """Return a HuggingFace API token if available, else None.

    Checks (in order):
    1. ``HF_TOKEN`` environment variable
    2. ``HUGGING_FACE_HUB_TOKEN`` environment variable
    3. ``huggingface_hub`` cached token (from ``huggingface-cli login``)
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()

    # Try huggingface_hub cached credentials
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(token_path):
        try:
            with open(token_path, "r") as fh:
                token = fh.read().strip()
            if token:
                return token
        except OSError:
            pass

    return None


def _auth_headers():
    """Return authorization headers for HuggingFace API requests."""
    token = _get_hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _http_get_json(url, timeout=60):
    """GET *url* and return the parsed JSON body."""
    _ensure_requests()
    resp = requests.get(url, headers=_auth_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _http_get_bytes(url, timeout=180):
    """GET *url* and return raw bytes (with retry)."""
    _ensure_requests()
    last_exc = None
    headers = _auth_headers()
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except (requests.RequestException, IOError) as exc:
            last_exc = exc
            if attempt < 2:
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/2 after {wait}s: {exc}")
                time.sleep(wait)
    raise last_exc


def _read_parquet_bytes(raw_bytes):
    """Read Parquet bytes into a list of dicts.

    Tries ``pyarrow`` first, then falls back to ``pandas``.
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(io.BytesIO(raw_bytes))
        return table.to_pylist()
    except ImportError:
        pass

    try:
        import pandas as pd
        df = pd.read_parquet(io.BytesIO(raw_bytes))
        return df.to_dict(orient="records")
    except ImportError:
        pass

    raise ImportError(
        "Reading Parquet files requires either 'pyarrow' or 'pandas' "
        "(with a Parquet engine).  Install one of them:\n"
        "  pip install pyarrow   # or\n"
        "  pip install pandas pyarrow"
    )


def _write_jsonl(records, path):
    """Write a list of dicts as JSONL to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_text(text):
    """Lowercase, strip, and collapse all whitespace to single spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def text_hash(text):
    """Return SHA-256 hex digest of *normalize_text(text)*."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


# Keep private alias for internal use
_text_hash = text_hash


def _label_to_category(label):
    """Map a binary label to a human-readable category string."""
    return "injection" if label == 1 else "benign"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fetch_parquet_urls(dataset_id, split):
    """Fetch Parquet file URLs for a given dataset and split.

    Tries several HuggingFace Parquet API endpoint variations and returns a
    flat list of download URLs.
    """
    base = HF_PARQUET_API.format(dataset_id=dataset_id)

    # Strategy 1: /parquet/{config}/{split}  with config = "default"
    # Strategy 2: /parquet  (top-level, returns nested dict)
    # Strategy 3: /parquet/{split}  (some datasets use split at top level)
    urls_to_try = [
        f"{base}/default/{split}",
        f"{base}/{split}/{split}",
        base,
    ]

    last_exc = None
    for api_url in urls_to_try:
        try:
            data = _http_get_json(api_url)
            urls = _extract_parquet_urls(data, split)
            if urls:
                return urls
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(
        f"Could not fetch parquet URLs for {dataset_id} split={split}. "
        f"Last error: {last_exc}"
    )


def _extract_parquet_urls(data, split):
    """Extract a flat list of parquet URLs from an API response.

    The HuggingFace API can return:
    - A plain list of URL strings
    - A dict mapping config names to split dicts, which map to URL lists
    - A dict mapping split names to URL lists
    """
    # Case 1: plain list of strings
    if isinstance(data, list):
        urls = [u for u in data if isinstance(u, str) and u.endswith(".parquet")]
        if urls:
            return urls
        # Could be a list of dicts with a "url" key
        urls = [
            item["url"] for item in data
            if isinstance(item, dict) and "url" in item
        ]
        return urls

    # Case 2: dict
    if isinstance(data, dict):
        # Try: {config: {split: [urls]}}
        for config_name, splits_or_urls in data.items():
            if isinstance(splits_or_urls, dict) and split in splits_or_urls:
                val = splits_or_urls[split]
                if isinstance(val, list):
                    return [
                        u for u in val
                        if isinstance(u, str) and u.endswith(".parquet")
                    ] or [
                        item["url"] for item in val
                        if isinstance(item, dict) and "url" in item
                    ]

        # Try: {split: [urls]}
        if split in data:
            val = data[split]
            if isinstance(val, list):
                return [
                    u for u in val
                    if isinstance(u, str) and u.endswith(".parquet")
                ] or [
                    item["url"] for item in val
                    if isinstance(item, dict) and "url" in item
                ]

    return []


def download_and_read_parquet(url):
    """Download a single parquet file and return rows as a list of dicts."""
    raw = _http_get_bytes(url)
    return _read_parquet_bytes(raw)


def normalize_records(rows, config):
    """Convert raw rows to the standard schema using *config*.

    Parameters
    ----------
    rows : list[dict]
        Raw rows from parquet.
    config : dict
        Dataset configuration entry from the DATASETS registry.

    Returns
    -------
    list[dict]
        Records in standard ``{text, label, source, category}`` format.
    """
    text_col = config["text_col"]
    label_col = config.get("label_col")
    label_map = config.get("label_map", {})
    fixed_label = config.get("fixed_label")
    source = config["source"]
    row_filter = config.get("filter")

    records = []
    for row in rows:
        # Apply row-level filter if specified
        if row_filter:
            skip = False
            for key, expected in row_filter.items():
                val = row.get(key)
                if isinstance(val, str):
                    val = val.strip().lower()
                if isinstance(expected, str):
                    expected = expected.strip().lower()
                if val != expected:
                    skip = True
                    break
            if skip:
                continue

        # Extract text
        text = row.get(text_col)
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue

        # Extract label
        if label_col is not None:
            raw_label = row.get(label_col)
            if raw_label in label_map:
                label = label_map[raw_label]
            else:
                # Try string coercion for booleans / ints
                str_label = str(raw_label).strip().lower()
                if str_label in label_map:
                    label = label_map[str_label]
                else:
                    try:
                        label = int(raw_label)
                    except (TypeError, ValueError):
                        continue
        elif fixed_label is not None:
            label = fixed_label
        else:
            continue

        if label not in (0, 1):
            continue

        records.append({
            "text": text,
            "label": label,
            "source": source,
            "category": _label_to_category(label),
        })

    return records


def stratified_sample(records, max_samples, rng):
    """Sample up to *max_samples* records while preserving label ratio.

    Samples from label=0 and label=1 pools independently to maintain the
    original distribution as closely as possible.
    """
    if len(records) <= max_samples:
        return records

    by_label = {}
    for rec in records:
        by_label.setdefault(rec["label"], []).append(rec)

    total = len(records)
    sampled = []
    remaining_budget = max_samples

    labels_sorted = sorted(by_label.keys())
    for i, label in enumerate(labels_sorted):
        pool = by_label[label]
        if i < len(labels_sorted) - 1:
            # Proportional allocation
            n = max(1, round(max_samples * len(pool) / total))
            n = min(n, len(pool), remaining_budget)
        else:
            # Last label gets whatever is left
            n = min(len(pool), remaining_budget)
        sampled.extend(rng.sample(pool, n))
        remaining_budget -= n

    rng.shuffle(sampled)
    return sampled


def deduplicate(records):
    """Deduplicate records by SHA-256 of normalized text.

    Keeps the first occurrence (source priority is determined by insertion
    order).

    Returns
    -------
    tuple[list[dict], dict]
        (deduplicated records, stats dict with per-source dedup counts)
    """
    seen = set()
    deduped = []
    dup_counts = {}

    for rec in records:
        h = _text_hash(rec["text"])
        if h in seen:
            src = rec["source"]
            dup_counts[src] = dup_counts.get(src, 0) + 1
            continue
        seen.add(h)
        deduped.append(rec)

    return deduped, dup_counts


def fetch_dataset(config, max_per_dataset=None, skip_gated=True):
    """Download and normalize a single dataset.

    Parameters
    ----------
    config : dict
        An entry from the DATASETS registry.
    max_per_dataset : int or None
        Override for max_samples (CLI --max-per-dataset).
    skip_gated : bool
        If True, skip datasets that require authentication.

    Returns
    -------
    list[dict]
        Normalized records, sampled if needed.
    """
    name = config["name"]
    dataset_id = config["id"]
    is_gated = config.get("gated", False)

    if is_gated and skip_gated:
        print(f"  [skip] {name} is gated -- use --no-skip-gated to include")
        return []

    max_samples = max_per_dataset if max_per_dataset is not None else config.get("max_samples")
    rng = random.Random(RANDOM_SEED)

    all_rows = []
    for split in config["splits"]:
        print(f"  Fetching parquet URLs for split '{split}' ...")
        try:
            parquet_urls = fetch_parquet_urls(dataset_id, split)
        except Exception as exc:
            print(f"  WARNING: could not fetch parquet URLs for {dataset_id} "
                  f"split={split}: {exc}")
            continue

        print(f"  Found {len(parquet_urls)} parquet shard(s)")
        for i, purl in enumerate(parquet_urls):
            print(f"    Downloading shard {i + 1}/{len(parquet_urls)} ...")
            try:
                rows = download_and_read_parquet(purl)
                all_rows.extend(rows)
            except Exception as exc:
                print(f"    WARNING: failed to download shard {purl}: {exc}")
                continue
        print(f"  Loaded {len(all_rows)} raw rows from split '{split}'")

    if not all_rows:
        print(f"  WARNING: no rows loaded for {name}")
        return []

    # Normalize
    records = normalize_records(all_rows, config)
    print(f"  Normalized to {len(records)} records "
          f"(label=0: {sum(1 for r in records if r['label'] == 0)}, "
          f"label=1: {sum(1 for r in records if r['label'] == 1)})")

    # Sample
    if max_samples and len(records) > max_samples:
        records = stratified_sample(records, max_samples, rng)
        print(f"  Sampled down to {len(records)} records "
              f"(label=0: {sum(1 for r in records if r['label'] == 0)}, "
              f"label=1: {sum(1 for r in records if r['label'] == 1)})")

    return records


def merge_all(output_dir, datasets, force=False, max_per_dataset=None,
              skip_gated=True, dataset_filter=None):
    """Main orchestrator: download, normalize, sample, deduplicate, and write.

    Parameters
    ----------
    output_dir : str
        Directory for all output files.
    datasets : list[dict]
        The dataset registry.
    force : bool
        Re-download and overwrite existing output.
    max_per_dataset : int or None
        Override for per-dataset sample cap.
    skip_gated : bool
        Skip datasets marked as gated.
    dataset_filter : set[str] or None
        If set, only process datasets whose ``name`` is in this set.
    """
    merged_path = os.path.join(output_dir, "merged_train.jsonl")
    stats_path = os.path.join(output_dir, "stats.json")

    if os.path.exists(merged_path) and not force:
        print(f"[skip] {merged_path} already exists (use --force to overwrite)")
        return

    all_records = []
    per_source_raw = {}
    errors = 0

    for config in datasets:
        name = config["name"]
        source = config["source"]

        if dataset_filter and name not in dataset_filter:
            print(f"\n[{name}] skipped (not in --datasets filter)")
            continue

        print(f"\n[{name}] ({config['id']})")
        try:
            records = fetch_dataset(
                config,
                max_per_dataset=max_per_dataset,
                skip_gated=skip_gated,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            errors += 1
            continue

        if not records:
            continue

        per_source_raw[source] = per_source_raw.get(source, 0) + len(records)

        # Write per-source file
        source_path = os.path.join(output_dir, f"{config['name']}.jsonl")
        _write_jsonl(records, source_path)
        print(f"  Wrote {len(records)} records to {source_path}")

        all_records.extend(records)

    if not all_records:
        print("\nNo records collected. Nothing to write.")
        return

    total_before = len(all_records)
    print(f"\n{'=' * 60}")
    print(f"Total records before dedup: {total_before}")

    # Deduplicate
    deduped, dup_counts = deduplicate(all_records)
    total_after = len(deduped)

    print(f"Total records after dedup:  {total_after}")
    print(f"Duplicates removed:         {total_before - total_after}")
    if dup_counts:
        print("  Per-source duplicates removed:")
        for src, cnt in sorted(dup_counts.items()):
            print(f"    {src}: {cnt}")

    # Write merged file
    _write_jsonl(deduped, merged_path)
    print(f"\nWrote merged file: {merged_path} ({total_after} records)")

    # Per-source counts after dedup
    per_source_final = {}
    per_source_label = {}
    for rec in deduped:
        src = rec["source"]
        per_source_final[src] = per_source_final.get(src, 0) + 1
        key = (src, rec["label"])
        per_source_label[key] = per_source_label.get(key, 0) + 1

    # Write stats
    stats = {
        "total_before_dedup": total_before,
        "total_after_dedup": total_after,
        "duplicates_removed": total_before - total_after,
        "per_source_raw_counts": per_source_raw,
        "per_source_final_counts": per_source_final,
        "per_source_duplicates_removed": dup_counts,
        "per_source_label_counts": {
            f"{src}_label_{label}": cnt
            for (src, label), cnt in sorted(per_source_label.items())
        },
        "errors": errors,
    }
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"Wrote stats: {stats_path}")

    if errors:
        print(f"\nCompleted with {errors} error(s).", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download public prompt-injection datasets from HuggingFace, "
            "normalize to a standard JSONL schema, deduplicate, and merge."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/aggregated",
        help="Directory for output files (default: data/aggregated/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing output files.",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Override the per-dataset sample cap (default: use registry values).",
    )
    parser.add_argument(
        "--skip-gated",
        action="store_true",
        default=True,
        help="Skip gated datasets that require authentication (default: True).",
    )
    parser.add_argument(
        "--no-skip-gated",
        action="store_false",
        dest="skip_gated",
        help="Include gated datasets (requires HF authentication).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of dataset names to process "
            "(e.g. 'imoxto_cleaned,gandalf_rct'). Default: all."
        ),
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_filter = None
    if args.datasets:
        dataset_filter = {s.strip() for s in args.datasets.split(",") if s.strip()}
        known = {d["name"] for d in DATASETS}
        unknown = dataset_filter - known
        if unknown:
            print(
                f"WARNING: unknown dataset name(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(known))}",
                file=sys.stderr,
            )

    print("Na0S Dataset Aggregator")
    print(f"Output directory: {args.output_dir}")
    print(f"Force overwrite: {args.force}")
    print(f"Skip gated: {args.skip_gated}")
    if args.max_per_dataset is not None:
        print(f"Max per dataset: {args.max_per_dataset}")
    if dataset_filter:
        print(f"Dataset filter: {', '.join(sorted(dataset_filter))}")

    merge_all(
        output_dir=args.output_dir,
        datasets=DATASETS,
        force=args.force,
        max_per_dataset=args.max_per_dataset,
        skip_gated=args.skip_gated,
        dataset_filter=dataset_filter,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
