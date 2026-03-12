#!/usr/bin/env python3
"""Dataset trust scoring for Na0S quarantine pipeline.

Computes a numeric trust score (0.0-1.0) for each dataset source by
combining six dimensions:

    1. Source Reputation  (weight 0.30) — base tier + HF metadata
    2. Data Quality       (weight 0.25) — validation check pass rate
    3. Label Consistency  (weight 0.20) — suspicious label rate
    4. Freshness          (weight 0.10) — age decay
    5. Historical         (weight 0.10) — past promotion success rate
    6. Provenance         (weight 0.05) — metadata completeness

The composite score gates promotion decisions:

    >= 0.80  auto_promote     (tier1/tier2 only)
    >= 0.55  staging_eligible (can move quarantine -> staging)
    >= 0.30  quarantine_hold  (manual review required)
    <  0.30  auto_reject      (data removed with alert)

Usage::

    python scripts/trust_score.py --report
    python scripts/trust_score.py --gate --min-score 0.55
    python scripts/trust_score.py --score --source deepset/prompt-injections
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

from scripts import quarantine

# ── Weights ──────────────────────────────────────────────────────────────

DIMENSION_WEIGHTS = {
    "reputation": 0.30,
    "quality": 0.25,
    "label_consistency": 0.20,
    "freshness": 0.10,
    "historical": 0.10,
    "provenance": 0.05,
}

# ── Thresholds ───────────────────────────────────────────────────────────

THRESHOLDS = {
    "auto_promote": 0.80,
    "staging_eligible": 0.55,
    "quarantine_hold": 0.30,
}

# ── Base Tier Scores ─────────────────────────────────────────────────────

BASE_TIER_SCORES = {
    "tier1": 0.95,
    "tier2": 0.70,
    "tier3": 0.30,
    "tier4": 0.15,
}

# ── Hard Vetoes ──────────────────────────────────────────────────────────

HARD_VETOES = {
    "quality_zero": lambda d: d["quality"] == 0.0,
    "label_consistency_low": lambda d: d["label_consistency"] < 0.20,
}

# ── Quality Check Weights ────────────────────────────────────────────────

QUALITY_CHECKS = {
    "schema": {"weight": 0.30, "hard_fail": True},
    "text_quality": {"weight": 0.20, "hard_fail": False},
    "class_balance": {"weight": 0.15, "hard_fail": False},
    "duplicates": {"weight": 0.15, "hard_fail": False},
    "total_size": {"weight": 0.10, "hard_fail": True},
    "label_consistency": {"weight": 0.10, "hard_fail": False},
}

# Validation check sets per tier (mirrors validate_data.py)
TIER_CHECKS = {
    "basic": {"schema"},
    "standard": {"schema", "total_size", "text_quality", "class_balance"},
    "strict": {
        "schema", "total_size", "text_quality", "class_balance",
        "duplicates", "label_consistency", "source_files",
    },
}

SUSPICIOUS_INJECTION_WORDS = [
    "ignore previous", "ignore all", "disregard", "bypass",
    "new instructions", "system prompt", "you are now",
    "forget everything", "override", "jailbreak",
]

PROVENANCE_SIGNALS = [
    "has_license",
    "has_citation",
    "has_dataset_card",
    "has_source_url",
    "has_creation_date",
]

# ── Dimension Functions ──────────────────────────────────────────────────


def compute_reputation(source_id, tier, hf_metadata=None):
    """Compute source reputation score.

    Args:
        source_id: Source identifier.
        tier: Tier key (e.g. "tier1").
        hf_metadata: Optional dict with downloads, likes, has_dataset_card.

    Returns:
        float: Score 0.0-1.0.
    """
    base = BASE_TIER_SCORES.get(tier, 0.15)

    if hf_metadata:
        downloads = hf_metadata.get("downloads", 0)
        likes = hf_metadata.get("likes", 0)
        has_card = hf_metadata.get("has_dataset_card", False)

        pop_bonus = min(0.03, math.log10(max(downloads, 1)) / 200)
        like_bonus = min(0.01, likes / 1000)
        card_bonus = 0.01 if has_card else 0.0

        base = min(1.0, base + pop_bonus + like_bonus + card_bonus)

    return base


def compute_quality(failed_checks, tier):
    """Compute data quality score from validation results.

    Args:
        failed_checks: Set of check names that failed.
        tier: Tier key for determining active checks.

    Returns:
        float: Score 0.0-1.0.  Returns 0.0 on hard failures.
    """
    tier_key = "strict" if tier not in TIER_CHECKS else tier
    # Map tier key to validation level
    tier_to_level = {"tier1": "basic", "tier2": "standard",
                     "tier3": "strict", "tier4": "strict"}
    level = tier_to_level.get(tier, tier_key)
    active = TIER_CHECKS.get(level, TIER_CHECKS["strict"])

    score = 0.0
    total_weight = 0.0

    for check_name, cfg in QUALITY_CHECKS.items():
        if check_name not in active:
            continue
        total_weight += cfg["weight"]
        if check_name not in failed_checks:
            score += cfg["weight"]
        elif cfg["hard_fail"]:
            return 0.0

    return score / total_weight if total_weight > 0 else 0.5


def compute_label_consistency(rows):
    """Compute label consistency score.

    Checks for safe-labeled rows containing injection phrases and
    injection-labeled rows with no injection signals.

    Args:
        rows: List of dicts with "text" and "label" keys, or list of
              (text, label) tuples.

    Returns:
        float: Score 0.0-1.0.  1.0 = perfect consistency.
    """
    if not rows:
        return 0.5

    pattern = "|".join(re.escape(p) for p in SUSPICIOUS_INJECTION_WORDS)
    compiled = re.compile(pattern, re.IGNORECASE)

    safe_count = 0
    safe_suspect = 0
    inj_count = 0
    inj_benign = 0

    for row in rows:
        if isinstance(row, dict):
            text = str(row.get("text", ""))
            label = int(row.get("label", 0))
        else:
            text, label = str(row[0]), int(row[1])

        has_signals = bool(compiled.search(text))

        if label == 0:
            safe_count += 1
            if has_signals:
                safe_suspect += 1
        elif label == 1:
            inj_count += 1
            if not has_signals:
                inj_benign += 1

    suspect_rate = safe_suspect / safe_count if safe_count > 0 else 0.0
    benign_in_inj = inj_benign / inj_count if inj_count > 0 else 0.0

    # Combined mislabel estimate
    mislabel_rate = (suspect_rate + benign_in_inj * 0.5) / 1.5
    return max(0.0, 1.0 - mislabel_rate * 5)


def compute_freshness(last_updated, max_age_days=180):
    """Compute freshness score based on data age.

    Args:
        last_updated: datetime or ISO8601 string of last update.
        max_age_days: Age at which score decays to floor.

    Returns:
        float: Score 0.2-1.0.
    """
    if isinstance(last_updated, str):
        try:
            last_updated = datetime.fromisoformat(last_updated)
        except (ValueError, TypeError):
            return 0.5

    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)

    age_days = (datetime.now(timezone.utc) - last_updated).days

    if age_days <= 30:
        return 1.0
    elif age_days >= max_age_days:
        return 0.2
    else:
        return 1.0 - 0.8 * (age_days - 30) / (max_age_days - 30)


def compute_historical(source_id, log_entries):
    """Compute historical reliability from quarantine log.

    Uses Laplace-smoothed success rate of past promotions vs rejections.

    Args:
        source_id: Source identifier to filter log.
        log_entries: List of quarantine log dicts.

    Returns:
        float: Score 0.0-1.0.  0.5 = no history (neutral prior).
    """
    if not log_entries:
        return 0.5

    promotions = 0
    rejections = 0

    for entry in log_entries:
        if entry.get("source_id") != source_id:
            continue
        action = entry.get("action", "")
        if action in ("promote", "promote_to_production", "direct_pass"):
            promotions += 1
        elif action == "reject":
            rejections += 1

    total = promotions + rejections
    if total == 0:
        return 0.5

    # Laplace smoothing: (successes + 1) / (total + 2)
    return (promotions + 1) / (total + 2)


def compute_provenance(metadata):
    """Compute provenance completeness score.

    Args:
        metadata: Dict with provenance signal keys (has_license, etc).

    Returns:
        float: Score 0.0-1.0.
    """
    if not metadata:
        return 0.0

    present = sum(
        1 for sig in PROVENANCE_SIGNALS
        if metadata.get(sig, False)
    )
    return present / len(PROVENANCE_SIGNALS)


# ── Composite Score ──────────────────────────────────────────────────────


def compute_trust_score(
    source_id,
    tier,
    rows=None,
    failed_checks=None,
    metadata=None,
    log_entries=None,
    hf_metadata=None,
    last_updated=None,
):
    """Compute composite trust score for a dataset source.

    Args:
        source_id: Source identifier string.
        tier: Tier key (e.g. "tier1", "tier3").
        rows: List of (text, label) tuples or dicts for label checking.
        failed_checks: Set of validation check names that failed.
        metadata: Dict with provenance signals.
        log_entries: Quarantine log entries list.
        hf_metadata: Optional HuggingFace metadata dict.
        last_updated: datetime or ISO string of last dataset update.

    Returns:
        dict with keys: trust_score, dimensions, tier, gate_decision, reason.
    """
    if failed_checks is None:
        failed_checks = set()
    if metadata is None:
        metadata = {}
    if log_entries is None:
        log_entries = []
    if last_updated is None:
        last_updated = datetime.now(timezone.utc)

    dimensions = {
        "reputation": compute_reputation(source_id, tier, hf_metadata),
        "quality": compute_quality(failed_checks, tier),
        "label_consistency": compute_label_consistency(rows or []),
        "freshness": compute_freshness(last_updated),
        "historical": compute_historical(source_id, log_entries),
        "provenance": compute_provenance(metadata),
    }

    trust_score = sum(
        DIMENSION_WEIGHTS[k] * dimensions[k] for k in DIMENSION_WEIGHTS
    )

    gate_decision, reason = _apply_gate(trust_score, tier, dimensions)

    return {
        "trust_score": round(trust_score, 4),
        "dimensions": {k: round(v, 4) for k, v in dimensions.items()},
        "tier": tier,
        "gate_decision": gate_decision,
        "reason": reason,
    }


def _apply_gate(score, tier, dimensions):
    """Determine gate decision from score and tier constraints.

    Returns:
        tuple[str, str]: (decision, reason).
    """
    # Check hard vetoes first
    for veto_name, check_fn in HARD_VETOES.items():
        if check_fn(dimensions):
            return "reject", f"Hard veto: {veto_name}"

    # Tier-specific constraints: tier3/tier4 can never auto-promote
    can_auto_promote = tier in ("tier1", "tier2")

    if score >= THRESHOLDS["auto_promote"] and can_auto_promote:
        return (
            "auto_promote",
            f"Score {score:.3f} >= {THRESHOLDS['auto_promote']}",
        )
    elif score >= THRESHOLDS["staging_eligible"]:
        return (
            "staging_eligible",
            f"Score {score:.3f} >= {THRESHOLDS['staging_eligible']}",
        )
    elif score >= THRESHOLDS["quarantine_hold"]:
        return (
            "quarantine_hold",
            f"Score {score:.3f} >= {THRESHOLDS['quarantine_hold']} "
            f"(manual review required)",
        )
    else:
        return (
            "auto_reject",
            f"Score {score:.3f} < {THRESHOLDS['quarantine_hold']}",
        )


# ── CLI Helpers ──────────────────────────────────────────────────────────


def _load_data_rows(file_path):
    """Load rows from a CSV or JSONL file for label consistency scoring."""
    import csv as csv_mod

    rows = []
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as fh:
                reader = csv_mod.DictReader(fh)
                for row in reader:
                    if "text" in row and "label" in row:
                        rows.append(row)
        elif ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "text" in obj and "label" in obj:
                            rows.append(obj)
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return rows


def report(config=None):
    """Print trust scores for all known sources.

    Scans quarantine, staging, and aggregated directories, resolves
    tiers, and computes scores for each source found.
    """
    if config is None:
        config = quarantine.load_trust_config()

    log_entries = quarantine._load_log()
    sources_found = {}

    # Scan quarantine entries
    if os.path.isdir(quarantine.QUARANTINE_DIR):
        for name in sorted(os.listdir(quarantine.QUARANTINE_DIR)):
            entry_dir = os.path.join(quarantine.QUARANTINE_DIR, name)
            if not os.path.isdir(entry_dir):
                continue
            meta = quarantine._read_metadata(entry_dir)
            if meta:
                sources_found[meta.get("source_id", name)] = {
                    "dir": entry_dir,
                    "meta": meta,
                    "stage": "quarantine",
                }

    # Scan staging entries
    if os.path.isdir(quarantine.STAGING_DIR):
        for name in sorted(os.listdir(quarantine.STAGING_DIR)):
            entry_dir = os.path.join(quarantine.STAGING_DIR, name)
            if not os.path.isdir(entry_dir):
                continue
            meta = quarantine._read_metadata(entry_dir)
            if meta:
                sources_found[meta.get("source_id", name)] = {
                    "dir": entry_dir,
                    "meta": meta,
                    "stage": "staging",
                }

    if not sources_found:
        print("  No sources found in quarantine or staging.")
        return []

    results = []
    print(f"\n{'Source':<45} {'Tier':<6} {'Score':>6} {'Decision':<20} {'Stage':<12}")
    print(f"{'-' * 45} {'-' * 6} {'-' * 6} {'-' * 20} {'-' * 12}")

    for source_id, info in sorted(sources_found.items()):
        meta = info["meta"]
        tier = meta.get("tier", "tier3")

        # Find data files for label consistency
        data_files = [
            os.path.join(info["dir"], f)
            for f in os.listdir(info["dir"])
            if f.endswith((".csv", ".jsonl"))
        ]
        rows = []
        for df in data_files:
            rows.extend(_load_data_rows(df))

        # Determine failed checks from metadata
        failed = set()
        val_results = meta.get("validation_results") or {}
        for fname, vr in val_results.items():
            if isinstance(vr, dict) and not vr.get("passed", True):
                failed.add(vr.get("tier", "schema"))

        result = compute_trust_score(
            source_id=source_id,
            tier=tier,
            rows=rows,
            failed_checks=failed,
            metadata=meta,
            log_entries=log_entries,
            last_updated=meta.get("ingested_at"),
        )
        result["source_id"] = source_id
        result["stage"] = info["stage"]
        results.append(result)

        print(
            f"  {source_id:<43} {tier:<6} "
            f"{result['trust_score']:>5.3f} "
            f"{result['gate_decision']:<20} "
            f"{info['stage']:<12}"
        )

    print()
    return results


def gate(config=None, min_score=None):
    """Check that all sources meet minimum trust score.

    Args:
        config: Trust tier config (loaded if None).
        min_score: Minimum score threshold (defaults to staging_eligible).

    Returns:
        int: 0 if all pass, 1 if any fail.
    """
    if min_score is None:
        min_score = THRESHOLDS["staging_eligible"]

    results = report(config)

    blocked = [r for r in results if r["trust_score"] < min_score]
    if blocked:
        print(f"  GATE FAILED: {len(blocked)} source(s) below {min_score:.2f}")
        for r in blocked:
            print(f"    - {r['source_id']}: {r['trust_score']:.3f} ({r['reason']})")
        return 1

    print(f"  GATE PASSED: All {len(results)} source(s) >= {min_score:.2f}")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────


def build_parser():
    parser = argparse.ArgumentParser(
        description="Dataset trust scoring for Na0S quarantine pipeline.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--report",
        action="store_true",
        help="Print trust scores for all sources in quarantine/staging.",
    )
    group.add_argument(
        "--gate",
        action="store_true",
        help="Check all sources meet minimum trust score.",
    )
    group.add_argument(
        "--score",
        action="store_true",
        help="Score a single source by --source ID.",
    )
    parser.add_argument(
        "--source",
        metavar="SOURCE_ID",
        help="Source identifier (required with --score).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=f"Minimum score for --gate (default: {THRESHOLDS['staging_eligible']}).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to trust_tiers.yaml.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        quarantine.TRUST_TIERS_PATH = args.config

    try:
        config = quarantine.load_trust_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ERROR loading trust config: {exc}", file=sys.stderr)
        return 1

    if args.report:
        report(config)
        return 0

    elif args.gate:
        return gate(config, min_score=args.min_score)

    elif args.score:
        if not args.source:
            print("  ERROR: --source is required with --score", file=sys.stderr)
            return 1
        tier = quarantine.resolve_tier(args.source, config)
        log_entries = quarantine._load_log()
        result = compute_trust_score(
            source_id=args.source,
            tier=tier,
            log_entries=log_entries,
        )
        print(json.dumps(result, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
