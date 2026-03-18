#!/usr/bin/env python3
"""Download HuggingFace datasets from the Na0S HF Dataset Registry.

Uses :data:`DATASET_REGISTRY` from ``hf_dataset_registry.py`` to download
each dataset via the ``datasets`` library, convert to Na0S sample format,
and write JSONL files to ``data/raw/hf/``.

Usage::

    # Download all registry datasets
    python -m scripts.data.download_hf_datasets --registry

    # Download a single dataset by HF ID
    python -m scripts.data.download_hf_datasets --dataset squad

    # Limit samples per dataset
    python -m scripts.data.download_hf_datasets --registry --max-samples 1000

    # Custom output directory
    python -m scripts.data.download_hf_datasets --registry --output-dir /tmp/hf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from scripts.data.hf_dataset_registry import (
    DATASET_REGISTRY,
    HFDatasetSpec,
    get_by_id,
    get_registry,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "data", "raw", "hf")
DEFAULT_MAX_SAMPLES = 50_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(iterable, desc=None, total=None):
    """Wrap an iterable with tqdm if available, otherwise pass through."""
    if TQDM_AVAILABLE:
        return tqdm(iterable, desc=desc, total=total)
    return iterable


def _sanitise_name(hf_id: str) -> str:
    """Turn a HuggingFace dataset ID into a safe filename stem."""
    return hf_id.replace("/", "__").replace("-", "_").lower()


def _resolve_label(row, spec: HFDatasetSpec) -> Optional[int]:
    """Resolve the integer label for a single row using the spec's label_map.

    Returns:
        An integer label (0 or 1), or None if the row cannot be labelled.
    """
    if spec.label_field is None:
        # No label field means we cannot infer label from data.
        # Caller must decide a fixed label from context (e.g. category).
        return None

    raw = row.get(spec.label_field)
    if raw is None:
        return None

    if spec.label_map:
        mapped = spec.label_map.get(str(raw))
        if mapped is not None:
            return int(mapped)
        # Try the raw value directly
        mapped = spec.label_map.get(raw)
        if mapped is not None:
            return int(mapped)

    # Fall back to int cast
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _default_label_for_category(category: str) -> int:
    """Return a sensible default label when no label_field is available."""
    if category in ("jailbreak", "red_team"):
        return 1
    return 0


def _write_jsonl(records: list, path: str) -> None:
    """Write records as JSONL to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core download / conversion
# ---------------------------------------------------------------------------

def download_spec(
    spec: HFDatasetSpec,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    auth_token: Optional[str] = None,
) -> int:
    """Download and convert a single dataset from a :class:`HFDatasetSpec`.

    Args:
        spec: The dataset specification.
        output_dir: Directory for output JSONL files.
        max_samples: Maximum number of samples to keep per dataset.
        auth_token: HuggingFace auth token for gated datasets.

    Returns:
        Number of records written.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
        Exception: Any download or conversion error.
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "The 'datasets' package is required. "
            "Install it with: pip install datasets"
        )

    fname = _sanitise_name(spec.hf_id) + ".jsonl"
    output_path = os.path.join(output_dir, fname)

    # Load from HuggingFace
    kwargs = {}
    if spec.config:
        kwargs["name"] = spec.config
    if spec.requires_auth and auth_token:
        kwargs["token"] = auth_token

    ds = load_dataset(spec.hf_id, split=spec.split, **kwargs)

    # Cap sample count
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    default_label = _default_label_for_category(spec.category)

    records = []
    for row in ds:
        text = row.get(spec.text_field)
        if not text:
            continue
        text = str(text).strip()
        if not text:
            continue

        label = _resolve_label(row, spec)
        if label is None:
            label = default_label

        records.append({
            "text": text,
            "label": label,
            "source": spec.hf_id,
            "category": spec.category,
        })

    _write_jsonl(records, output_path)
    return len(records)


def download_all(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    auth_token: Optional[str] = None,
    specs: Optional[List[HFDatasetSpec]] = None,
) -> dict:
    """Download all (or selected) datasets from the registry.

    Args:
        output_dir: Directory for output JSONL files.
        max_samples: Maximum number of samples per dataset.
        auth_token: HuggingFace auth token for gated datasets.
        specs: Specific specs to download. Defaults to the full registry.

    Returns:
        A dict mapping hf_id -> record count (or error string).
    """
    if specs is None:
        specs = get_registry()

    results = {}
    for spec in _progress(specs, desc="Downloading datasets", total=len(specs)):
        print(f"\n[{spec.hf_id}]", flush=True)
        try:
            n = download_spec(
                spec,
                output_dir=output_dir,
                max_samples=max_samples,
                auth_token=auth_token,
            )
            results[spec.hf_id] = n
            print(f"  {n} records written")
        except Exception as exc:
            results[spec.hf_id] = f"ERROR: {exc}"
            print(f"  ERROR: {exc}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets from the Na0S registry.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--registry",
        action="store_true",
        help="Download all datasets in the HF registry.",
    )
    group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Download a single dataset by HuggingFace ID.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for JSONL files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Max samples per dataset (default: {DEFAULT_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=None,
        help="HuggingFace auth token for gated datasets. "
             "Defaults to HF_TOKEN environment variable.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    auth_token = args.auth_token or os.environ.get("HF_TOKEN")

    if args.dataset:
        spec = get_by_id(args.dataset)
        if spec is None:
            print(f"ERROR: dataset '{args.dataset}' not found in registry.",
                  file=sys.stderr)
            return 1
        try:
            n = download_spec(
                spec,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                auth_token=auth_token,
            )
            print(f"\nDone. {n} records written.")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        results = download_all(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            auth_token=auth_token,
        )
        errors = sum(1 for v in results.values() if isinstance(v, str))
        total = sum(v for v in results.values() if isinstance(v, int))
        print(f"\nDone. {total} total records, {errors} error(s) "
              f"across {len(results)} datasets.")
        return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
