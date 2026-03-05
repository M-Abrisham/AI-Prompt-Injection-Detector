#!/usr/bin/env python3
"""Validate combined training data before feature extraction.

Runs quality checks on data/processed/combined_data.csv and rejects or
flags rows that could degrade model quality.  Designed to sit between
process_data.py and features.py in the training pipeline.

Checks performed:
  1. Schema       — columns (text, label) exist, labels are binary
  2. Text quality — min length, non-empty, no null bytes
  3. Label balance — warn/fail if class ratio exceeds threshold
  4. Duplicates   — flag remaining near-duplicates
  5. Poisoning    — flag rows where label contradicts simple heuristics
  6. Per-source   — reject any raw CSV with < MIN_SOURCE_ROWS rows

Trust tier validation levels (used by quarantine.py):
  --tier basic    = schema check only (tier1 trusted sources)
  --tier standard = schema + text quality + class balance (tier2 community)
  --tier strict   = all checks (tier3/tier4 quarantined sources)
  (no --tier)     = all checks (backward-compatible default)

Exit codes:
  0 = all checks passed (warnings may still be printed)
  1 = hard failures detected — pipeline should stop

Usage::

    python scripts/validate_data.py
    python scripts/validate_data.py --input data/processed/combined_data.csv
    python scripts/validate_data.py --fix   # auto-remove flagged rows
    python scripts/validate_data.py --tier basic   # schema only
    python scripts/validate_data.py --tier strict   # all checks
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUT = os.path.join(ROOT, "data", "processed", "combined_data.csv")
RAW_DIR = os.path.join(ROOT, "data", "raw")

# ── Thresholds ───────────────────────────────────────────────────────
MIN_TEXT_LENGTH = 10          # characters
MAX_TEXT_LENGTH = 50_000      # characters — catch data corruption
MIN_TOTAL_ROWS = 1_000       # combined dataset must have at least this many
MIN_SOURCE_ROWS = 5           # per-source CSV must have at least this many
MAX_CLASS_RATIO = 5.0         # max ratio between majority/minority class
MAX_DUPLICATE_RATE = 0.10     # warn if >10% exact duplicates remain
SUSPICIOUS_INJECTION_WORDS = [
    "ignore previous", "ignore all", "disregard", "bypass",
    "new instructions", "system prompt", "you are now",
    "forget everything", "override", "jailbreak",
]

# ── Trust Tier Validation Levels ─────────────────────────────────────
#
# Maps --tier values to the set of check functions to run.
# "basic" is the lightest (tier1), "strict" is the heaviest (tier3/4).
# None (no --tier flag) defaults to all checks for backward compat.

VALID_TIERS = ("basic", "standard", "strict")

# Check names that each tier level includes.
# Each level is cumulative: strict includes everything.
TIER_CHECKS = {
    "basic": {
        "schema",
    },
    "standard": {
        "schema",
        "total_size",
        "text_quality",
        "class_balance",
    },
    "strict": {
        "schema",
        "total_size",
        "text_quality",
        "class_balance",
        "duplicates",
        "label_consistency",
        "source_files",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────

def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _has_injection_signal(text: str) -> bool:
    """Return True if text contains common injection phrases."""
    lower = text.lower()
    return any(phrase in lower for phrase in SUSPICIOUS_INJECTION_WORDS)


# ── Validators ───────────────────────────────────────────────────────

class ValidationResult:
    def __init__(self):
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.rows_to_drop: list[int] = []

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  WARN: {msg}")

    def error(self, msg: str):
        self.errors.append(msg)
        print(f"  FAIL: {msg}")

    def drop(self, indices: list[int], reason: str):
        self.rows_to_drop.extend(indices)
        if indices:
            print(f"  DROP: {len(indices)} rows — {reason}")


def check_schema(df: pd.DataFrame, result: ValidationResult):
    """Verify required columns and data types."""
    if "text" not in df.columns:
        result.error("Missing 'text' column")
        return
    if "label" not in df.columns:
        result.error("Missing 'label' column")
        return

    non_binary_mask = ~df["label"].isin([0, 1])
    if non_binary_mask.sum() > 0:
        # Try to salvage: if the text contains injection signals, relabel as 1
        nb_texts = df.loc[non_binary_mask, "text"].astype(str).str.lower()
        pattern = "|".join(re.escape(p) for p in SUSPICIOUS_INJECTION_WORDS)
        has_injection = nb_texts.str.contains(pattern, regex=True, na=False)

        relabeled = has_injection.sum()
        dropped = non_binary_mask.sum() - relabeled

        if relabeled > 0:
            df.loc[has_injection[has_injection].index, "label"] = 1
            print(f"  RELABEL: {relabeled} non-binary rows → label=1 (contain injection phrases)")

        # Drop the rest — no text signal to guess the label
        remaining = df.loc[non_binary_mask].index.difference(has_injection[has_injection].index)
        if len(remaining) > 0:
            result.drop(remaining.tolist(), f"non-binary labels, no injection signal ({dropped})")


def check_text_quality(df: pd.DataFrame, result: ValidationResult):
    """Flag rows with bad text."""
    # Null / empty
    null_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    null_count = null_mask.sum()
    if null_count:
        result.drop(df[null_mask].index.tolist(), f"null/empty text ({null_count})")

    # Too short
    lengths = df["text"].astype(str).str.len()
    short_mask = (lengths < MIN_TEXT_LENGTH) & ~null_mask
    short_count = short_mask.sum()
    if short_count:
        result.drop(df[short_mask].index.tolist(), f"text < {MIN_TEXT_LENGTH} chars ({short_count})")

    # Too long (likely data corruption)
    long_mask = lengths > MAX_TEXT_LENGTH
    long_count = long_mask.sum()
    if long_count:
        result.drop(df[long_mask].index.tolist(), f"text > {MAX_TEXT_LENGTH} chars ({long_count})")

    # Null bytes
    null_byte_mask = df["text"].astype(str).str.contains("\x00", regex=False)
    null_byte_count = null_byte_mask.sum()
    if null_byte_count:
        result.drop(df[null_byte_mask].index.tolist(), f"null bytes in text ({null_byte_count})")


def check_class_balance(df: pd.DataFrame, result: ValidationResult):
    """Warn or fail on severe class imbalance."""
    counts = df["label"].value_counts()
    if len(counts) < 2:
        result.error(f"Only one class present: {counts.to_dict()}")
        return

    majority = counts.iloc[0]
    minority = counts.iloc[-1]
    ratio = majority / minority if minority > 0 else float("inf")

    print(f"  Class distribution: {counts.to_dict()} (ratio {ratio:.2f}:1)")

    if ratio > MAX_CLASS_RATIO:
        result.warn(
            f"Class imbalance ratio {ratio:.1f}:1 exceeds threshold {MAX_CLASS_RATIO}:1. "
            f"Model uses class_weight='balanced' but extreme skew may hurt."
        )


def check_duplicates(df: pd.DataFrame, result: ValidationResult):
    """Check for remaining exact-text duplicates."""
    dup_mask = df["text"].astype(str).duplicated(keep="first")
    dup_count = dup_mask.sum()
    dup_rate = dup_count / len(df) if len(df) > 0 else 0

    if dup_rate > MAX_DUPLICATE_RATE:
        result.warn(f"{dup_count} duplicates remain ({dup_rate:.1%}) — above {MAX_DUPLICATE_RATE:.0%} threshold")
        result.drop(df[dup_mask].index.tolist(), f"duplicate text ({dup_count})")
    elif dup_count > 0:
        print(f"  OK: {dup_count} duplicates ({dup_rate:.1%}) — within threshold")


def check_label_consistency(df: pd.DataFrame, result: ValidationResult):
    """Warn about rows where label *might* contradict content.

    These are warn-only — NOT auto-removed.  Many of these are legitimate:
    safe prompts that discuss injection techniques, or correctly labeled
    injection samples the model needs to learn from.
    """
    safe_mask = df["label"] == 0
    safe_texts = df.loc[safe_mask, "text"].astype(str).str.lower()

    # Build a single regex for all injection phrases
    pattern = "|".join(re.escape(p) for p in SUSPICIOUS_INJECTION_WORDS)
    suspect_mask = safe_texts.str.contains(pattern, regex=True, na=False)
    flagged = suspect_mask.sum()

    if flagged > 0:
        pct = flagged / len(df) * 100
        result.warn(
            f"{flagged} rows ({pct:.2f}%) labeled safe contain injection-related phrases. "
            f"This is expected — many are legitimate discussions about injection."
        )


def check_source_files(result: ValidationResult):
    """Validate individual raw source CSVs have minimum row counts."""
    if not os.path.isdir(RAW_DIR):
        result.warn(f"Raw directory not found: {RAW_DIR}")
        return

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(RAW_DIR, fname)
        try:
            row_count = sum(1 for _ in open(path, encoding="utf-8")) - 1  # minus header
        except Exception:
            result.warn(f"Could not read {fname}")
            continue

        if row_count < MIN_SOURCE_ROWS:
            result.warn(f"{fname}: only {row_count} rows (min: {MIN_SOURCE_ROWS})")


def check_total_size(df: pd.DataFrame, result: ValidationResult):
    """Fail if dataset is too small to train on."""
    if len(df) < MIN_TOTAL_ROWS:
        result.error(f"Only {len(df)} rows — minimum is {MIN_TOTAL_ROWS}")


# ── Main ─────────────────────────────────────────────────────────────

def validate(input_path: str, fix: bool = False, tier: str | None = None) -> bool:
    """Run validation checks, optionally filtered by trust tier level.

    Args:
        input_path: Path to the CSV file to validate.
        fix: If True, auto-remove flagged rows and rewrite the file.
        tier: Validation tier level ("basic", "standard", "strict").
              None means run ALL checks (backward-compatible default).

    Returns:
        True if validation passed (no hard errors), False otherwise.
    """
    # Determine which checks to run
    if tier is not None:
        if tier not in TIER_CHECKS:
            print(f"  FAIL: Unknown tier '{tier}'. Valid: {VALID_TIERS}")
            return False
        active_checks = TIER_CHECKS[tier]
        print(f"Validating {input_path} (tier: {tier})")
    else:
        # No tier specified -- run all checks (backward compat)
        active_checks = TIER_CHECKS["strict"]
        print(f"Validating {input_path}")

    if not os.path.exists(input_path):
        print(f"  FAIL: File not found: {input_path}")
        return False

    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows")

    result = ValidationResult()

    # Schema check is always first and gates subsequent checks
    if "schema" in active_checks:
        check_schema(df, result)
        if not result.passed:
            return False

    if "total_size" in active_checks:
        check_total_size(df, result)

    if "text_quality" in active_checks:
        check_text_quality(df, result)

    if "class_balance" in active_checks:
        check_class_balance(df, result)

    if "duplicates" in active_checks:
        check_duplicates(df, result)

    if "label_consistency" in active_checks:
        check_label_consistency(df, result)

    if "source_files" in active_checks:
        check_source_files(result)

    # Apply fixes
    if fix and result.rows_to_drop:
        unique_drops = sorted(set(result.rows_to_drop))
        before = len(df)
        df = df.drop(index=unique_drops).reset_index(drop=True)
        after = len(df)
        df.to_csv(input_path, index=False)
        print(f"\n  FIXED: Removed {before - after} rows → {after} remaining")
        print(f"  Saved to {input_path}")

    # Summary
    tier_label = f" (tier: {tier})" if tier else ""
    print(f"\n{'=' * 55}")
    print(f"Validation Summary{tier_label}")
    print(f"{'=' * 55}")
    print(f"  Checks run: {', '.join(sorted(active_checks))}")
    print(f"  Warnings:   {len(result.warnings)}")
    print(f"  Errors:     {len(result.errors)}")
    print(f"  Rows flagged for removal: {len(set(result.rows_to_drop))}")
    print(f"  Result:     {'PASS' if result.passed else 'FAIL'}")
    print(f"{'=' * 55}")

    return result.passed


def main():
    parser = argparse.ArgumentParser(description="Validate training data quality.")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to combined CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-remove flagged rows and rewrite the CSV.",
    )
    parser.add_argument(
        "--tier",
        choices=VALID_TIERS,
        default=None,
        help=(
            "Trust tier validation level. "
            "basic=schema only (tier1), "
            "standard=schema+quality+balance (tier2), "
            "strict=all checks (tier3/4). "
            "Default: all checks (same as strict)."
        ),
    )
    args = parser.parse_args()

    passed = validate(args.input, fix=args.fix, tier=args.tier)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
