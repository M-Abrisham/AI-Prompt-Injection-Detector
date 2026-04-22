#!/usr/bin/env python3
"""Near-duplicate detection using SimHash and MinHash + LSH.

Pure Python implementation (no external dedup libraries). Detects
semantically similar duplicates that survive exact-match dedup.

Usage::

    python scripts/near_duplicate.py --data data/processed/combined_data.csv
    python scripts/near_duplicate.py --data data/processed/combined_data.csv --method minhash --threshold 0.8
    python scripts/near_duplicate.py --data data/processed/combined_data.csv --method simhash --threshold 3
    python scripts/near_duplicate.py --data data/processed/combined_data.csv --strategy keep_labeled --output deduped.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import struct
import sys
from collections import defaultdict
from typing import Iterator

import pandas as pd

def _find_project_root() -> str:
    """Walk upward from this file looking for pyproject.toml; fall back to cwd.

    The module was promoted from ``scripts/`` to ``src/na0s/dataset/``; a naive
    ``dirname(dirname(__file__))`` now points inside ``src/``, not the project
    root. Walking up until we find ``pyproject.toml`` keeps the DEFAULT_* paths
    anchored to the repo regardless of where the module lives.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()
        cur = parent


ROOT = _find_project_root()

DEFAULT_INPUT = os.path.join(ROOT, "data", "processed", "combined_data.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "processed", "combined_data_deduped.csv")
DEFAULT_REPORT = os.path.join(ROOT, "data", "staging", "near_duplicates_report.csv")

# SimHash defaults
SIMHASH_BITS = 64
SIMHASH_NGRAM = 3
SIMHASH_THRESHOLD = 3  # Hamming distance

# MinHash defaults
MINHASH_NUM_HASHES = 128
MINHASH_NGRAM = 3
MINHASH_JACCARD_THRESHOLD = 0.8

# LSH defaults  (bands * rows_per_band == num_hashes)
LSH_BANDS = 16
LSH_ROWS_PER_BAND = 8  # 16 * 8 = 128

# Chunked processing
CHUNK_SIZE = 50_000


# ── Character n-grams ──────────────────────────────────────────────

def _char_ngrams(text: str, n: int = 3) -> list[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


# ── Hash helpers ───────────────────────────────────────────────────

def _hash64(data: str) -> int:
    """Compute a 64-bit hash from a string."""
    digest = hashlib.md5(data.encode("utf-8", errors="replace")).digest()
    return struct.unpack("<Q", digest[:8])[0]


# ── SimHash ────────────────────────────────────────────────────────

def simhash(text: str, n: int = SIMHASH_NGRAM) -> int:
    """Compute a 64-bit SimHash fingerprint from character n-grams."""
    ngrams = _char_ngrams(text, n)
    if not ngrams:
        return 0

    v = [0] * SIMHASH_BITS
    for gram in ngrams:
        h = _hash64(gram)
        for i in range(SIMHASH_BITS):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(SIMHASH_BITS):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two integers."""
    return bin(a ^ b).count("1")


def _simhash_bit_partitions(fingerprint: int, num_partitions: int = 4) -> list[int]:
    """Split a 64-bit fingerprint into *num_partitions* blocks.

    Each block is ``64 // num_partitions`` bits wide.  Used for blocking:
    two fingerprints with Hamming distance <= ``num_partitions - 1`` must
    share at least one identical block (pigeonhole principle).
    """
    bits_per_block = SIMHASH_BITS // num_partitions
    mask = (1 << bits_per_block) - 1
    return [(fingerprint >> (i * bits_per_block)) & mask
            for i in range(num_partitions)]


def find_simhash_duplicates(
    texts: list[str],
    threshold: int = SIMHASH_THRESHOLD,
) -> list[tuple[int, int, int]]:
    """Find near-duplicate pairs using SimHash with bit-partition blocking.

    Uses the pigeonhole principle: for Hamming distance <= *threshold*,
    at least one of ``threshold + 1`` equal-width bit-partitions must be
    identical between the two fingerprints.  This gives O(n) average-case
    comparison instead of O(n^2).

    Returns list of (idx_a, idx_b, hamming_distance) tuples.
    """
    fingerprints = [simhash(t) for t in texts]
    n = len(fingerprints)

    # Number of partitions: threshold + 1 guarantees at least one match
    num_partitions = min(threshold + 1, SIMHASH_BITS)
    if num_partitions < 1:
        num_partitions = 1

    # Build blocking index: partition_id -> block_value -> [indices]
    blocks: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(num_partitions)]
    for idx, fp in enumerate(fingerprints):
        parts = _simhash_bit_partitions(fp, num_partitions)
        for pid, block_val in enumerate(parts):
            blocks[pid][block_val].append(idx)

    # Candidate pairs from blocking
    seen_pairs: set[tuple[int, int]] = set()
    duplicates: list[tuple[int, int, int]] = []

    for pid in range(num_partitions):
        for _block_val, indices in blocks[pid].items():
            if len(indices) < 2:
                continue
            for ii in range(len(indices)):
                for jj in range(ii + 1, len(indices)):
                    i, j = indices[ii], indices[jj]
                    pair = (min(i, j), max(i, j))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    dist = hamming_distance(fingerprints[i], fingerprints[j])
                    if dist <= threshold:
                        duplicates.append((pair[0], pair[1], dist))

    return duplicates


# ── MinHash ────────────────────────────────────────────────────────

def minhash_signature(text: str, num_hashes: int = MINHASH_NUM_HASHES,
                      n: int = MINHASH_NGRAM) -> list[int]:
    """Compute MinHash signature for a text."""
    ngrams = set(_char_ngrams(text, n))
    if not ngrams:
        return [0] * num_hashes

    sig = []
    for seed in range(num_hashes):
        min_hash = float("inf")
        for gram in ngrams:
            h = _hash64(f"{seed}:{gram}")
            if h < min_hash:
                min_hash = h
        sig.append(min_hash)
    return sig


def jaccard_from_minhash(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have same length")
    if not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ── LSH (Locality-Sensitive Hashing) ──────────────────────────────

def lsh_buckets(signature: list[int], bands: int, rows_per_band: int) -> list[int]:
    """Hash a MinHash signature into *bands* LSH buckets.

    Each band hashes *rows_per_band* consecutive elements of the signature
    into a single bucket key.  Two signatures that are identical in at
    least one band will be candidate pairs.

    Returns a list of *bands* bucket keys (one per band).
    """
    bucket_keys: list[int] = []
    for b in range(bands):
        start = b * rows_per_band
        end = start + rows_per_band
        band_slice = signature[start:end]
        # Hash the band slice to a single bucket key
        h = hashlib.md5(str(band_slice).encode("utf-8")).digest()
        bucket_keys.append(struct.unpack("<Q", h[:8])[0])
    return bucket_keys


def find_minhash_duplicates(
    texts: list[str],
    threshold: float = MINHASH_JACCARD_THRESHOLD,
    num_hashes: int = MINHASH_NUM_HASHES,
    bands: int | None = None,
    rows_per_band: int | None = None,
) -> list[tuple[int, int, float]]:
    """Find near-duplicate pairs using MinHash + LSH.

    Uses Locality-Sensitive Hashing to avoid O(n^2) pairwise comparison.
    Candidate pairs are identified via LSH band buckets, then verified
    with full Jaccard estimation.

    Returns list of (idx_a, idx_b, jaccard_similarity) tuples.
    """
    # Auto-compute bands/rows if not specified
    if bands is None or rows_per_band is None:
        # Pick bands * rows_per_band == num_hashes
        # Default: 16 bands x 8 rows for 128 hashes
        rows_per_band = rows_per_band or LSH_ROWS_PER_BAND
        bands = num_hashes // rows_per_band
        if bands < 1:
            bands = 1
            rows_per_band = num_hashes

    signatures = [minhash_signature(t, num_hashes) for t in texts]

    # Build LSH index: band_id -> bucket_key -> [indices]
    lsh_index: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(bands)]
    for idx, sig in enumerate(signatures):
        buckets = lsh_buckets(sig, bands, rows_per_band)
        for band_id, bkey in enumerate(buckets):
            lsh_index[band_id][bkey].append(idx)

    # Collect candidate pairs
    candidate_pairs: set[tuple[int, int]] = set()
    for band_id in range(bands):
        for _bkey, indices in lsh_index[band_id].items():
            if len(indices) < 2:
                continue
            for ii in range(len(indices)):
                for jj in range(ii + 1, len(indices)):
                    i, j = indices[ii], indices[jj]
                    candidate_pairs.add((min(i, j), max(i, j)))

    # Verify candidates
    duplicates: list[tuple[int, int, float]] = []
    for i, j in candidate_pairs:
        sim = jaccard_from_minhash(signatures[i], signatures[j])
        if sim >= threshold:
            duplicates.append((i, j, sim))

    return duplicates


# ── Duplicate clusters (union-find) ───────────────────────────────

def _build_groups(pairs: list[tuple[int, int, float | int]], n: int) -> list[set[int]]:
    """Build connected components from duplicate pairs using union-find."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, _ in pairs:
        union(i, j)

    groups: dict[int, set[int]] = defaultdict(set)
    for idx in range(n):
        root = find(idx)
        groups[root].add(idx)

    return [g for g in groups.values() if len(g) > 1]


def build_clusters(
    pairs: list[tuple[int, int, float | int]],
    n: int,
) -> list[list[int]]:
    """Build duplicate clusters from pairs.

    Returns a list of clusters, each cluster being a sorted list of
    indices that are near-duplicates of each other.
    """
    groups = _build_groups(pairs, n)
    return [sorted(g) for g in groups]


# ── Deduplication strategies ──────────────────────────────────────

def _pick_representative(
    group: set[int],
    texts: list[str],
    labels: list | None,
    strategy: str,
) -> int:
    """Pick the representative index from a duplicate group.

    Strategies:
        keep_first   -- keep the lowest index (first occurrence)
        keep_longest -- keep the longest text
        keep_labeled -- prefer labeled (label != None and label != -1)
                        samples; break ties by longest text
    """
    indices = sorted(group)

    if strategy == "keep_first":
        return indices[0]

    if strategy == "keep_longest":
        return max(indices, key=lambda idx: len(texts[idx]))

    if strategy == "keep_labeled":
        # Separate labeled from unlabeled
        labeled = []
        unlabeled = []
        for idx in indices:
            lbl = labels[idx] if labels is not None else None
            if lbl is not None and lbl != -1 and str(lbl).strip() != "":
                labeled.append(idx)
            else:
                unlabeled.append(idx)
        pool = labeled if labeled else unlabeled
        # Among the pool, pick longest
        return max(pool, key=lambda idx: len(texts[idx]))

    raise ValueError(f"Unknown dedup strategy: {strategy!r}")


# ── Chunked CSV reading ──────────────────────────────────────────

def iter_csv_chunks(
    path: str,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield chunks of a CSV file as DataFrames for memory efficiency."""
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        yield chunk


# ── Main deduplication pipeline ───────────────────────────────────

def deduplicate(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    report_path: str = DEFAULT_REPORT,
    method: str = "simhash",
    threshold: float | None = None,
    strategy: str = "keep_first",
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run near-duplicate detection and deduplication.

    Parameters
    ----------
    input_path : str
        Path to input CSV with at least a ``text`` column.
    output_path : str
        Path for deduplicated CSV output.
    report_path : str
        Path for duplicate cluster report CSV.
    method : str
        ``"simhash"`` or ``"minhash"``.
    threshold : float or None
        Distance threshold.  SimHash: Hamming distance (int).
        MinHash: Jaccard similarity (float).  None uses defaults.
    strategy : str
        ``"keep_first"``, ``"keep_longest"``, or ``"keep_labeled"``.
    chunk_size : int
        Rows per chunk when streaming the CSV.

    Returns
    -------
    dict
        Summary with counts, method, threshold, strategy, and clusters.
    """
    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Stream chunks and concatenate (memory-efficient for very large files)
    print(f"Loading {input_path} (chunk_size={chunk_size})...")
    chunks = list(iter_csv_chunks(input_path, chunk_size=chunk_size))
    df = pd.concat(chunks, ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str)
    texts = df["text"].tolist()
    n = len(texts)
    print(f"  {n} rows loaded")

    # Extract labels for keep_labeled strategy
    labels: list | None = None
    if "label" in df.columns:
        labels = df["label"].tolist()

    # Find duplicates
    if method == "simhash":
        thresh = int(threshold) if threshold is not None else SIMHASH_THRESHOLD
        print(f"Running SimHash (threshold: Hamming distance <= {thresh})...")
        pairs = find_simhash_duplicates(texts, threshold=thresh)
    elif method == "minhash":
        thresh = float(threshold) if threshold is not None else MINHASH_JACCARD_THRESHOLD
        print(f"Running MinHash + LSH (threshold: Jaccard >= {thresh})...")
        pairs = find_minhash_duplicates(texts, threshold=thresh)
    else:
        print(f"ERROR: Unknown method '{method}'. Use 'simhash' or 'minhash'.")
        sys.exit(1)

    print(f"  Found {len(pairs)} duplicate pairs")

    # Build clusters and pick representatives
    groups = _build_groups(pairs, n)
    clusters = build_clusters(pairs, n)
    to_remove: set[int] = set()
    report_rows: list[dict] = []

    for group in groups:
        representative = _pick_representative(group, texts, labels, strategy)
        for idx in group:
            if idx != representative:
                to_remove.add(idx)
                report_rows.append({
                    "removed_index": idx,
                    "removed_text": texts[idx][:200],
                    "kept_index": representative,
                    "kept_text": texts[representative][:200],
                    "group_size": len(group),
                    "strategy": strategy,
                })

    # Remove duplicates
    deduped = df.drop(index=list(to_remove)).reset_index(drop=True)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    deduped.to_csv(output_path, index=False)

    # Save report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    if report_rows:
        pd.DataFrame(report_rows).to_csv(report_path, index=False)

    # Save cluster report (JSON-like in CSV)
    cluster_rows = []
    for ci, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_rows.append({
                "cluster_id": ci,
                "index": idx,
                "text_preview": texts[idx][:200],
            })

    summary = {
        "input_rows": n,
        "duplicate_pairs": len(pairs),
        "duplicate_groups": len(groups),
        "duplicate_clusters": len(clusters),
        "rows_removed": len(to_remove),
        "output_rows": len(deduped),
        "method": method,
        "threshold": thresh,
        "strategy": strategy,
    }

    print(f"\n{'=' * 50}")
    print(f"Near-Duplicate Detection Summary")
    print(f"{'=' * 50}")
    print(f"  Method:          {method}")
    print(f"  Threshold:       {thresh}")
    print(f"  Strategy:        {strategy}")
    print(f"  Input rows:      {n}")
    print(f"  Duplicate pairs: {len(pairs)}")
    print(f"  Clusters:        {len(clusters)}")
    print(f"  Rows removed:    {len(to_remove)}")
    print(f"  Output rows:     {len(deduped)}")
    print(f"  Output:          {output_path}")
    if report_rows:
        print(f"  Report:          {report_path}")
    print(f"{'=' * 50}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Near-duplicate detection for training data.",
    )
    parser.add_argument("--data", "--input", "-i", dest="data",
                        default=DEFAULT_INPUT,
                        help="Path to input CSV file.")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help="Path for deduplicated output CSV.")
    parser.add_argument("--report", "-r", default=DEFAULT_REPORT,
                        help="Path for duplicate report CSV.")
    parser.add_argument("--method", choices=["simhash", "minhash"],
                        default="simhash",
                        help="Dedup method (default: simhash).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Distance threshold (SimHash: Hamming <=N, "
                             "MinHash: Jaccard >=N).")
    parser.add_argument("--strategy",
                        choices=["keep_first", "keep_longest", "keep_labeled"],
                        default="keep_first",
                        help="Which sample to keep from each cluster "
                             "(default: keep_first).")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Rows per CSV chunk (default: {CHUNK_SIZE}).")
    args = parser.parse_args()
    deduplicate(
        args.data, args.output, args.report,
        args.method, args.threshold, args.strategy,
        args.chunk_size,
    )


if __name__ == "__main__":
    main()
