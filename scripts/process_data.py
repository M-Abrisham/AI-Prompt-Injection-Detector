"""Aggregate ALL raw datasets into a single training file.

Reads every CSV in data/raw/ and every JSONL in data/aggregated/,
data/harvest/, data/holdout/, and data/benchmark/, auto-detects
text/label columns, deduplicates by content hash, and writes
data/processed/combined_data.csv.
"""

import glob
import hashlib
import os
import re
import unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(ROOT, "data", "raw")
AGGREGATED_DIR = os.path.join(ROOT, "data", "aggregated")
HARVEST_DIR = os.path.join(ROOT, "data", "harvest")
HOLDOUT_DIR = os.path.join(ROOT, "data", "holdout")
BENCHMARK_DIR = os.path.join(ROOT, "data", "benchmark")
OUTPUT_PATH = os.path.join(ROOT, "data", "processed", "combined_data.csv")

# Candidate column names (checked in order of priority)
TEXT_CANDIDATES = ["text", "prompt", "instruction", "User Prompt", "body"]
LABEL_CANDIDATES = ["label", "labels"]


def _detect_column(df, candidates):
    """Return the first column name from *candidates* that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_csv(path):
    """Load a single CSV and normalise to (text, label)."""
    df = pd.read_csv(path)

    text_col = _detect_column(df, TEXT_CANDIDATES)
    if text_col is None:
        print(f"  WARN: skipping {os.path.basename(path)} "
              f"(no text column found among {TEXT_CANDIDATES})")
        return None

    label_col = _detect_column(df, LABEL_CANDIDATES)

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)

    if label_col is not None:
        out["label"] = pd.to_numeric(df[label_col], errors="coerce")
    else:
        # If there is no label column, skip -- we cannot train without labels
        print(f"  WARN: skipping {os.path.basename(path)} "
              f"(no label column found among {LABEL_CANDIDATES})")
        return None

    out = out.dropna(subset=["text", "label"])
    out["label"] = out["label"].astype(int)
    # Only keep valid binary labels
    out = out[out["label"].isin([0, 1])]
    return out


def _load_jsonl(path):
    """Load a single JSONL file and normalise to (text, label)."""
    df = pd.read_json(path, lines=True)

    text_col = _detect_column(df, TEXT_CANDIDATES)
    if text_col is None:
        print(f"  WARN: skipping {os.path.basename(path)} "
              f"(no text column found among {TEXT_CANDIDATES})")
        return None

    label_col = _detect_column(df, LABEL_CANDIDATES)

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)

    if label_col is not None:
        out["label"] = pd.to_numeric(df[label_col], errors="coerce")
    else:
        print(f"  WARN: skipping {os.path.basename(path)} "
              f"(no label column found among {LABEL_CANDIDATES})")
        return None

    out = out.dropna(subset=["text", "label"])
    out["label"] = out["label"].astype(int)
    out = out[out["label"].isin([0, 1])]
    return out


def _normalize_for_dedup(text):
    """Canonicalize text before hashing to improve duplicate detection.

    Uses Unicode NFKC normalization plus whitespace collapsing so that
    visually equivalent strings (e.g., fullwidth/compatibility variants)
    deduplicate to the same hash.
    """
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _text_hash(text):
    """Fast content hash for deduplication."""
    norm = _normalize_for_dedup(text)
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()


def merge_datasets():
    """Discover, load, deduplicate, and save all training data."""
    frames = []
    source_counts = {}

    # ── 1. Glob all CSVs in data/raw/ ────────────────────────────
    csv_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_paths:
        print("No CSV files found in data/raw/")

    for path in csv_paths:
        name = os.path.basename(path)
        df = _load_csv(path)
        if df is not None and len(df) > 0:
            source_counts[name] = len(df)
            frames.append(df)
            print(f"  [csv]  {name}: {len(df)} rows")

    # ── 2. Glob JSONL files in data/aggregated/, data/harvest/,
    #        data/holdout/, and data/benchmark/ ──────────────────
    for jsonl_dir in [AGGREGATED_DIR, HARVEST_DIR, HOLDOUT_DIR, BENCHMARK_DIR]:
        if not os.path.isdir(jsonl_dir):
            continue
        jsonl_paths = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))
        for path in jsonl_paths:
            name = os.path.relpath(path, ROOT)
            df = _load_jsonl(path)
            if df is not None and len(df) > 0:
                source_counts[name] = len(df)
                frames.append(df)
                print(f"  [jsonl] {name}: {len(df)} rows")

    if not frames:
        print("ERROR: No usable datasets found.")
        return None

    # ── 3. Concatenate ────────────────────────────────────────────
    combined = pd.concat(frames, axis=0, ignore_index=True)
    total_before = len(combined)

    # ── 4. Deduplicate by text hash ───────────────────────────────
    combined["_hash"] = combined["text"].apply(_text_hash)
    combined = combined.drop_duplicates(subset=["_hash"])
    # Stable idempotent ordering: sort by hash before writing.
    combined = combined.sort_values("_hash", kind="mergesort").reset_index(drop=True)
    combined = combined.drop(columns=["_hash"])
    dupes_removed = total_before - len(combined)

    # ── 5. Save ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    # ── 6. Report ─────────────────────────────────────────────────
    print(f"\n--- Per-source counts ---")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")
    print(f"\nTotal rows (before dedup): {total_before}")
    print(f"Duplicates removed:        {dupes_removed}")
    print(f"Final dataset size:        {len(combined)}")
    print(f"Label distribution:\n{combined['label'].value_counts().to_string()}")
    print(f"\nSaved to {OUTPUT_PATH}")
    return combined


if __name__ == "__main__":
    merge_datasets()
