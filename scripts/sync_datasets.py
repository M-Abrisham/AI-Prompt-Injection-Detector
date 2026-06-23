"""Sync datasets from data/datasets.yaml registry.

Reads the declarative registry, downloads each source (GitHub CSV or
HuggingFace), normalises columns to (text, label), and writes to
data/raw/<output>.csv.

Tracks dataset versions in data/datasets.lock so repeated runs skip
sources that haven't changed.  For HuggingFace repos the lock stores
the dataset commit SHA; for GitHub CSVs it stores a content hash.

Usage:
    python scripts/sync_datasets.py              # sync all
    python scripts/sync_datasets.py --force      # re-download everything
    python scripts/sync_datasets.py --only alpaca dolly  # sync specific sources
"""

import hashlib
import json
import os
import sys

import pandas as pd

from scripts.safe_yaml import safe_load_yaml

# Single owner of the ;-joined CSV-cell format for per-source taxonomy codes.
# Reused (not reinvented) so the stamp here and the read in process_data.py
# share one encoding. If na0s is not importable in the sync runtime, fall back
# to a local ";".join so sync never hard-depends on the package being on path.
try:
    from na0s.eval.registry import serialize_codes
except ImportError:  # pragma: no cover - defensive: sync runs from repo root
    def serialize_codes(codes):
        return ";".join(str(c) for c in (codes or []))

# Optional — gracefully degrade if not installed
try:
    from datasets import load_dataset
    from huggingface_hub import dataset_info

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(ROOT, "data", "datasets.yaml")
LOCK_PATH = os.path.join(ROOT, "data", "datasets.lock")


# ── Lock file helpers ─────────────────────────────────────────────


def _load_lock():
    if not os.path.exists(LOCK_PATH):
        return {}
    with open(LOCK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_lock(lock):
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2, sort_keys=True)
        f.write("\n")


def _content_hash(path):
    """SHA-256 of a local file (used for GitHub CSV freshness)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ── HuggingFace SHA lookup ────────────────────────────────────────


def _hf_sha(repo):
    """Get the latest commit SHA for a HuggingFace dataset repo."""
    if not HF_AVAILABLE:
        return None
    try:
        info = dataset_info(repo)
        return info.sha
    except Exception:
        return None


# ── Download functions ────────────────────────────────────────────


def _download_github_csv(cfg, output_path):
    """Download a CSV from a raw GitHub URL."""
    url = cfg["url"]
    text_col = cfg["text_column"]
    label_val = cfg["label"]

    df = pd.read_csv(url)
    if text_col not in df.columns:
        raise KeyError("Column '{}' not found in {}".format(text_col, url))

    out = pd.DataFrame({"text": df[text_col], "label": label_val})
    # Stamp the source's canonical taxonomy code(s) as constant provenance on
    # every row, carried through process_data.py into combined_data.csv. Empty
    # for untagged sources. (Scraped/harvest corpora are not synced here and
    # stay untagged -> "" downstream; no consumer should assume full coverage.)
    out["taxonomy_codes"] = serialize_codes(cfg.get("taxonomy_codes", []))
    out = out.dropna(subset=["text"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    return len(out)


def _download_huggingface(cfg, output_path):
    """Download a HuggingFace dataset and normalise to (text, label)."""
    if not HF_AVAILABLE:
        print("  SKIP (datasets / huggingface_hub not installed)")
        return 0

    repo = cfg["repo"]
    config_name = cfg.get("config")
    split = cfg.get("split", "train")
    text_col = cfg["text_column"]
    max_samples = cfg.get("max_samples")

    ds = load_dataset(repo, name=config_name, split=split)

    # Apply row-level filter if specified
    row_filter = cfg.get("filter")
    if row_filter:
        for key, val in row_filter.items():
            ds = ds.filter(lambda row, k=key, v=val: row.get(k) == v)

    # Cap sample count
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    # Extract text
    texts = [row[text_col] for row in ds if row.get(text_col)]

    # Resolve labels
    label_col = cfg.get("label_column")
    label_map = cfg.get("label_map")
    fixed_label = cfg.get("label")

    if label_col and label_map:
        # Build a robust lookup that handles bool/string mismatches
        # (YAML parses "true" as Python True, but HF may store "true" string)
        _lmap = {}
        for k, v in label_map.items():
            _lmap[k] = v
            _lmap[str(k).lower()] = v
            if isinstance(k, bool):
                _lmap["true" if k else "false"] = v
                # Also map None → same as False (e.g. gandalf_rct: None = not successful)
                if not k:
                    _lmap[None] = v
        labels = []
        for row in ds:
            raw = row.get(label_col)
            mapped = _lmap.get(raw)
            if mapped is None and raw is not None:
                mapped = _lmap.get(str(raw).lower())
            if mapped is not None:
                labels.append(mapped)
            else:
                labels.append(raw)
        # Trim to match texts length (filtered rows may differ)
        labels = labels[: len(texts)]
    elif fixed_label is not None:
        labels = [fixed_label] * len(texts)
    else:
        raise ValueError("Source '{}' needs label or label_column".format(repo))

    out = pd.DataFrame({"text": texts, "label": labels})
    # Stamp the source's canonical taxonomy code(s) as constant provenance on
    # every row (see _download_github_csv). Empty for untagged sources.
    out["taxonomy_codes"] = serialize_codes(cfg.get("taxonomy_codes", []))
    out = out.dropna(subset=["text"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    return len(out)


# ── Main sync logic ──────────────────────────────────────────────


def sync(force=False, only=None):
    registry = safe_load_yaml(REGISTRY_PATH)

    output_dir = os.path.join(ROOT, registry.get("output_dir", "data/raw"))
    sources = registry.get("sources", {})
    lock = _load_lock()
    updated = 0
    skipped = 0

    for name, cfg in sources.items():
        if only and name not in only:
            continue

        output_path = os.path.join(output_dir, cfg["output"])
        src_type = cfg["type"]

        # --- Freshness check ---
        if not force:
            if src_type == "huggingface":
                remote_sha = _hf_sha(cfg["repo"])
                locked_sha = lock.get(name, {}).get("sha")
                if (
                    remote_sha
                    and remote_sha == locked_sha
                    and os.path.exists(output_path)
                ):
                    print("  [skip] {} (up to date)".format(name))
                    skipped += 1
                    continue
            elif src_type == "github_csv":
                locked_hash = lock.get(name, {}).get("content_hash")
                if locked_hash and os.path.exists(output_path):
                    current_hash = _content_hash(output_path)
                    if current_hash == locked_hash:
                        print("  [skip] {} (up to date)".format(name))
                        skipped += 1
                        continue

        # --- Download ---
        print("  [sync] {} ...".format(name), end=" ")
        try:
            if src_type == "github_csv":
                count = _download_github_csv(cfg, output_path)
                lock[name] = {
                    "type": "github_csv",
                    "content_hash": _content_hash(output_path),
                    "rows": count,
                }
            elif src_type == "huggingface":
                count = _download_huggingface(cfg, output_path)
                sha = _hf_sha(cfg["repo"])
                lock[name] = {
                    "type": "huggingface",
                    "repo": cfg["repo"],
                    "sha": sha,
                    "rows": count,
                }
            else:
                print("unknown type '{}'".format(src_type))
                continue

            print("{} rows".format(count))
            updated += 1
        except Exception as e:
            print("ERROR: {}".format(e))

    _save_lock(lock)
    print("\nDone: {} updated, {} skipped".format(updated, skipped))


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    only = None
    if "--only" in args:
        idx = args.index("--only")
        only = set(args[idx + 1 :])

    print("Syncing datasets from {}".format(REGISTRY_PATH))
    sync(force=force, only=only)
