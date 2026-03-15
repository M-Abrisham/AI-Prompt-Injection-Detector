#!/usr/bin/env python3
"""Near-duplicate detection using SimHash and MinHash.

Pure Python implementation (no external dedup libraries). Detects
semantically similar duplicates that survive exact-match dedup.

Usage::

    python scripts/near_duplicate.py --input data/processed/combined_data.csv
    python scripts/near_duplicate.py --input data/processed/combined_data.csv --method minhash --threshold 0.8
    python scripts/near_duplicate.py --input data/processed/combined_data.csv --method simhash --threshold 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import struct
import sys
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


# ── SimHash ─────────────────────────────────────────────────────────

def _char_ngrams(text: str, n: int = 3) -> list[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def _hash64(data: str) -> int:
    """Compute a 64-bit hash from a string."""
    digest = hashlib.md5(data.encode("utf-8", errors="replace")).digest()
    return struct.unpack("<Q", digest[:8])[0]


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


def find_simhash_duplicates(
    texts: list[str],
    threshold: int = SIMHASH_THRESHOLD,
) -> list[tuple[int, int, int]]:
    """Find near-duplicate pairs using SimHash.

    Returns list of (idx_a, idx_b, hamming_distance) tuples.
    """
    fingerprints = [simhash(t) for t in texts]
    duplicates = []

    # O(n^2) comparison — for large datasets, use band-based blocking
    n = len(fingerprints)
    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming_distance(fingerprints[i], fingerprints[j])
            if dist <= threshold:
                duplicates.append((i, j, dist))

    return duplicates


# ── MinHash ─────────────────────────────────────────────────────────

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


def find_minhash_duplicates(
    texts: list[str],
    threshold: float = MINHASH_JACCARD_THRESHOLD,
    num_hashes: int = MINHASH_NUM_HASHES,
) -> list[tuple[int, int, float]]:
    """Find near-duplicate pairs using MinHash.

    Returns list of (idx_a, idx_b, jaccard_similarity) tuples.
    """
    signatures = [minhash_signature(t, num_hashes) for t in texts]
    duplicates = []

    n = len(signatures)
    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_from_minhash(signatures[i], signatures[j])
            if sim >= threshold:
                duplicates.append((i, j, sim))

    return duplicates


# ── Deduplication ───────────────────────────────────────────────────

def _build_groups(pairs: list[tuple[int, int, float | int]], n: int) -> list[set[int]]:
    """Build connected components from duplicate pairs using union-find."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, _ in pairs:
        union(i, j)

    groups = defaultdict(set)
    for idx in range(n):
        root = find(idx)
        groups[root].add(idx)

    return [g for g in groups.values() if len(g) > 1]


def deduplicate(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    report_path: str = DEFAULT_REPORT,
    method: str = "simhash",
    threshold: float | None = None,
) -> dict:
    """Run near-duplicate detection and deduplication.

    Returns a summary dict with counts.
    """
    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    df["text"] = df["text"].fillna("").astype(str)
    texts = df["text"].tolist()
    n = len(texts)
    print(f"  {n} rows loaded")

    # Find duplicates
    if method == "simhash":
        thresh = int(threshold) if threshold is not None else SIMHASH_THRESHOLD
        print(f"Running SimHash (threshold: Hamming distance <= {thresh})...")
        pairs = find_simhash_duplicates(texts, threshold=thresh)
    elif method == "minhash":
        thresh = float(threshold) if threshold is not None else MINHASH_JACCARD_THRESHOLD
        print(f"Running MinHash (threshold: Jaccard >= {thresh})...")
        pairs = find_minhash_duplicates(texts, threshold=thresh)
    else:
        print(f"ERROR: Unknown method '{method}'. Use 'simhash' or 'minhash'.")
        sys.exit(1)

    print(f"  Found {len(pairs)} duplicate pairs")

    # Build groups and pick representatives
    groups = _build_groups(pairs, n)
    to_remove = set()
    report_rows = []

    for group in groups:
        # Keep the longest text as representative
        sorted_group = sorted(group, key=lambda idx: len(texts[idx]), reverse=True)
        representative = sorted_group[0]
        for idx in sorted_group[1:]:
            to_remove.add(idx)
            report_rows.append({
                "removed_index": idx,
                "removed_text": texts[idx][:200],
                "kept_index": representative,
                "kept_text": texts[representative][:200],
                "group_size": len(group),
            })

    # Remove duplicates
    deduped = df.drop(index=list(to_remove)).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    deduped.to_csv(output_path, index=False)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    if report_rows:
        pd.DataFrame(report_rows).to_csv(report_path, index=False)

    summary = {
        "input_rows": n,
        "duplicate_pairs": len(pairs),
        "duplicate_groups": len(groups),
        "rows_removed": len(to_remove),
        "output_rows": len(deduped),
        "method": method,
        "threshold": thresh,
    }

    print(f"\n{'=' * 50}")
    print(f"Near-Duplicate Detection Summary")
    print(f"{'=' * 50}")
    print(f"  Method:          {method}")
    print(f"  Threshold:       {thresh}")
    print(f"  Input rows:      {n}")
    print(f"  Duplicate pairs: {len(pairs)}")
    print(f"  Groups:          {len(groups)}")
    print(f"  Rows removed:    {len(to_remove)}")
    print(f"  Output rows:     {len(deduped)}")
    print(f"  Output:          {output_path}")
    if report_rows:
        print(f"  Report:          {report_path}")
    print(f"{'=' * 50}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Near-duplicate detection for training data.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", "-r", default=DEFAULT_REPORT)
    parser.add_argument("--method", choices=["simhash", "minhash"], default="simhash")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Distance threshold (SimHash: Hamming <=N, MinHash: Jaccard >=N)")
    args = parser.parse_args()
    deduplicate(args.input, args.output, args.report, args.method, args.threshold)


if __name__ == "__main__":
    main()
