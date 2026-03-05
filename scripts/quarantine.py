#!/usr/bin/env python3
"""Dataset quarantine manager for Na0S trust tier system.

Enforces tiered validation and quarantine policies for all data sources.
Datasets from untrusted sources (tier3/tier4) are held in data/quarantine/
until they pass validation and are explicitly promoted by a maintainer.

Workflow:
    1. Ingest: Data arrives from sync_datasets.py, social_scraper.py,
       or weekly_harvest.py.  quarantine.py checks the source's trust
       tier and routes accordingly:
         - tier1/tier2 -> data/raw/ or data/aggregated/ (direct)
         - tier3/tier4 -> data/quarantine/<source_name>/ (held)

    2. Review: A maintainer runs ``quarantine.py --review`` to list all
       quarantined datasets with their validation status.

    3. Promote: After manual inspection, run
       ``quarantine.py --promote <name>`` to move validated data out of
       quarantine into data/aggregated/.

    4. Reject: Run ``quarantine.py --reject <name>`` to permanently
       remove quarantined data that failed review.

All actions are logged to data/quarantine/quarantine_log.json.

Usage::

    python scripts/quarantine.py --ingest <file> --source <source_id>
    python scripts/quarantine.py --review
    python scripts/quarantine.py --promote <name>
    python scripts/quarantine.py --reject <name>
    python scripts/quarantine.py --status
    python scripts/quarantine.py --validate-quarantined
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from scripts.safe_yaml import safe_load_yaml

# ── Paths ─────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUST_TIERS_PATH = os.path.join(ROOT, "data", "trust_tiers.yaml")
QUARANTINE_DIR = os.path.join(ROOT, "data", "quarantine")
QUARANTINE_LOG = os.path.join(ROOT, "data", "quarantine", "quarantine_log.json")
RAW_DIR = os.path.join(ROOT, "data", "raw")
AGGREGATED_DIR = os.path.join(ROOT, "data", "aggregated")


# ── Trust Tier Resolution ─────────────────────────────────────────────────


def load_trust_config():
    """Load and return the trust tier configuration.

    Returns:
        dict: The parsed trust_tiers.yaml content.

    Raises:
        FileNotFoundError: If trust_tiers.yaml does not exist.
        ValueError: If the YAML is malformed.
    """
    return safe_load_yaml(TRUST_TIERS_PATH)


def resolve_tier(source_id, config):
    """Determine the trust tier for a given source identifier.

    Resolution order:
      1. Exact match in sources map
      2. Prefix/wildcard match (e.g., "reddit/*" matches "reddit/r/ChatGPT")
      3. Default to tier3 (New Discovery) for unknown sources

    Args:
        source_id: The source identifier string (e.g., HF repo name,
            "reddit/r/ChatGPT", "twitter").
        config: Parsed trust_tiers.yaml dict.

    Returns:
        str: Tier key (e.g., "tier1", "tier2", "tier3", "tier4").
    """
    sources = config.get("sources", {})

    # 1. Exact match
    if source_id in sources:
        return sources[source_id]

    # 2. Prefix/wildcard match
    #    Entries like "reddit/*" match any source starting with "reddit/"
    for pattern, tier in sources.items():
        if pattern.endswith("/*"):
            prefix = pattern[:-1]  # "reddit/*" -> "reddit/"
            if source_id.startswith(prefix):
                return tier

    # 3. Default to tier3 for unknown sources
    return "tier3"


def get_tier_config(tier_key, config):
    """Get the configuration dict for a specific tier.

    Args:
        tier_key: e.g. "tier1", "tier2", "tier3", "tier4"
        config: Parsed trust_tiers.yaml dict.

    Returns:
        dict: Tier configuration with keys: label, description,
              validation, quarantine, min_confidence.
    """
    tiers = config.get("tiers", {})
    if tier_key not in tiers:
        # Defensive fallback -- treat as strictest tier
        return {
            "label": "Unknown",
            "description": "Tier not found in configuration",
            "validation": "strict",
            "quarantine": True,
            "min_confidence": 0.6,
        }
    return tiers[tier_key]


# ── Quarantine Log ────────────────────────────────────────────────────────


def _load_log():
    """Load the quarantine log.  Returns a list of log entry dicts."""
    if not os.path.isfile(QUARANTINE_LOG):
        return []
    try:
        with open(QUARANTINE_LOG, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []


def _save_log(entries):
    """Write the quarantine log atomically."""
    os.makedirs(os.path.dirname(QUARANTINE_LOG), exist_ok=True)
    tmp_path = QUARANTINE_LOG + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp_path, QUARANTINE_LOG)


def _log_action(action, name, source_id, tier, details=None):
    """Append an entry to the quarantine log.

    Args:
        action: One of "ingest", "promote", "reject", "validate",
                "direct_pass", "expire_warning".
        name: Dataset name (directory name under quarantine).
        source_id: Original source identifier.
        tier: Tier key that was resolved.
        details: Optional dict with additional context.
    """
    entries = _load_log()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "name": name,
        "source_id": source_id,
        "tier": tier,
    }
    if details:
        entry["details"] = details
    entries.append(entry)
    _save_log(entries)


# ── File Hashing ──────────────────────────────────────────────────────────


def _file_sha256(path):
    """Compute SHA-256 of a file for integrity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Quarantine Metadata ──────────────────────────────────────────────────


def _write_metadata(quarantine_path, source_id, tier, file_hash, row_count):
    """Write a metadata.json alongside quarantined data.

    This metadata file records provenance so reviewers know where
    the data came from and what checks have been run.
    """
    meta = {
        "source_id": source_id,
        "tier": tier,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "file_sha256": file_hash,
        "row_count": row_count,
        "validation_status": "pending",
        "validation_results": None,
        "reviewed_by": None,
        "promoted_at": None,
    }
    meta_path = os.path.join(quarantine_path, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return meta_path


def _read_metadata(quarantine_path):
    """Read metadata.json from a quarantine directory."""
    meta_path = os.path.join(quarantine_path, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _update_metadata(quarantine_path, updates):
    """Merge updates into an existing metadata.json."""
    meta = _read_metadata(quarantine_path) or {}
    meta.update(updates)
    meta_path = os.path.join(quarantine_path, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ── Row Counting ──────────────────────────────────────────────────────────


def _count_rows(file_path):
    """Count data rows in a CSV or JSONL file."""
    if not os.path.isfile(file_path):
        return 0
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as fh:
                # Subtract 1 for header
                return max(0, sum(1 for _ in fh) - 1)
        elif ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        else:
            return 0
    except OSError:
        return 0


# ── Core Operations ───────────────────────────────────────────────────────


def ingest(file_path, source_id, config=None):
    """Ingest a dataset file, routing it based on its trust tier.

    For tier1/tier2 sources, the file is copied directly to data/raw/
    (CSV) or data/aggregated/ (JSONL).  For tier3/tier4, it goes to
    data/quarantine/<safe_name>/.

    Args:
        file_path: Path to the CSV or JSONL file to ingest.
        source_id: Source identifier for tier resolution.
        config: Pre-loaded trust tier config (loaded if None).

    Returns:
        dict: Result with keys: action, tier, destination, name.
    """
    if config is None:
        config = load_trust_config()

    if not os.path.isfile(file_path):
        print(f"  ERROR: File not found: {file_path}", file=sys.stderr)
        return {"action": "error", "message": f"File not found: {file_path}"}

    tier_key = resolve_tier(source_id, config)
    tier_cfg = get_tier_config(tier_key, config)
    file_hash = _file_sha256(file_path)
    row_count = _count_rows(file_path)
    basename = os.path.basename(file_path)

    # Sanitize name for directory use
    safe_name = (
        source_id.replace("/", "_").replace(" ", "_").replace(".", "_")
    )

    print(f"  Ingesting: {basename}")
    print(f"    Source:  {source_id}")
    print(f"    Tier:    {tier_key} ({tier_cfg['label']})")
    print(f"    Rows:    {row_count}")
    print(f"    SHA-256: {file_hash[:16]}...")

    if tier_cfg.get("quarantine", False):
        # Route to quarantine
        dest_dir = os.path.join(QUARANTINE_DIR, safe_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, basename)
        shutil.copy2(file_path, dest_path)
        _write_metadata(dest_dir, source_id, tier_key, file_hash, row_count)
        _log_action("ingest", safe_name, source_id, tier_key, {
            "file": basename,
            "rows": row_count,
            "sha256": file_hash,
            "destination": "quarantine",
        })
        print(f"    Action:  QUARANTINED -> {dest_dir}")
        return {
            "action": "quarantined",
            "tier": tier_key,
            "destination": dest_dir,
            "name": safe_name,
        }
    else:
        # Direct pass -- route to raw/ or aggregated/ based on file type
        ext = os.path.splitext(basename)[1].lower()
        if ext == ".csv":
            dest_dir = RAW_DIR
        elif ext == ".jsonl":
            dest_dir = AGGREGATED_DIR
        else:
            dest_dir = RAW_DIR

        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, basename)
        shutil.copy2(file_path, dest_path)
        _log_action("direct_pass", safe_name, source_id, tier_key, {
            "file": basename,
            "rows": row_count,
            "sha256": file_hash,
            "destination": dest_dir,
        })
        print(f"    Action:  DIRECT PASS -> {dest_path}")
        return {
            "action": "direct_pass",
            "tier": tier_key,
            "destination": dest_path,
            "name": safe_name,
        }


def run_tier_validation(file_path, tier_key, config=None):
    """Run validate_data.py with the appropriate --tier level.

    Args:
        file_path: Path to the data file to validate.
        tier_key: Trust tier key (tier1-tier4).
        config: Pre-loaded trust tier config (loaded if None).

    Returns:
        dict: Validation result with keys: passed, tier, output.
    """
    if config is None:
        config = load_trust_config()

    tier_cfg = get_tier_config(tier_key, config)
    validation_level = tier_cfg.get("validation", "strict")

    validate_script = os.path.join(ROOT, "scripts", "validate_data.py")
    cmd = [
        sys.executable, validate_script,
        "--input", file_path,
        "--tier", validation_level,
    ]

    print(f"  Running validation: --tier {validation_level}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=ROOT,
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        print(f"  Validation: {'PASS' if passed else 'FAIL'}")
        if not passed:
            # Print first 20 lines of output for context
            for line in output.strip().splitlines()[:20]:
                print(f"    {line}")
        return {"passed": passed, "tier": validation_level, "output": output}
    except subprocess.TimeoutExpired:
        print("  Validation: TIMEOUT")
        return {"passed": False, "tier": validation_level, "output": "Timeout"}
    except Exception as exc:
        print(f"  Validation: ERROR ({exc})")
        return {"passed": False, "tier": validation_level, "output": str(exc)}


def validate_quarantined(config=None):
    """Run validation on all quarantined datasets.

    Updates metadata.json for each with the validation results.
    Does NOT promote -- that requires explicit --promote.
    """
    if config is None:
        config = load_trust_config()

    if not os.path.isdir(QUARANTINE_DIR):
        print("  No quarantine directory found.")
        return []

    results = []
    for name in sorted(os.listdir(QUARANTINE_DIR)):
        entry_dir = os.path.join(QUARANTINE_DIR, name)
        if not os.path.isdir(entry_dir):
            continue

        meta = _read_metadata(entry_dir)
        if meta is None:
            print(f"  WARN: {name} -- no metadata.json, skipping")
            continue

        tier_key = meta.get("tier", "tier3")
        print(f"\n  Validating: {name} (tier: {tier_key})")

        # Find the data file
        data_files = [
            f for f in os.listdir(entry_dir)
            if f.endswith((".csv", ".jsonl"))
        ]
        if not data_files:
            print(f"    No data files found in {entry_dir}")
            _update_metadata(entry_dir, {
                "validation_status": "error",
                "validation_results": "No data files found",
            })
            continue

        # Validate each data file
        all_passed = True
        file_results = {}
        for data_file in data_files:
            data_path = os.path.join(entry_dir, data_file)
            vresult = run_tier_validation(data_path, tier_key, config)
            file_results[data_file] = vresult
            if not vresult["passed"]:
                all_passed = False

        # Update metadata
        status = "passed" if all_passed else "failed"
        _update_metadata(entry_dir, {
            "validation_status": status,
            "validation_results": {
                f: {"passed": r["passed"], "tier": r["tier"]}
                for f, r in file_results.items()
            },
            "validated_at": datetime.now(timezone.utc).isoformat(),
        })

        _log_action("validate", name, meta.get("source_id", ""), tier_key, {
            "status": status,
            "files_checked": len(data_files),
        })

        results.append({
            "name": name,
            "tier": tier_key,
            "status": status,
            "files": file_results,
        })

    return results


def review(config=None):
    """List all quarantined datasets with their status.

    Prints a formatted table to stdout.
    """
    if config is None:
        config = load_trust_config()

    if not os.path.isdir(QUARANTINE_DIR):
        print("  No quarantine directory found. Nothing to review.")
        return []

    entries = []
    for name in sorted(os.listdir(QUARANTINE_DIR)):
        entry_dir = os.path.join(QUARANTINE_DIR, name)
        if not os.path.isdir(entry_dir):
            continue

        meta = _read_metadata(entry_dir)
        if meta is None:
            entries.append({
                "name": name,
                "source_id": "unknown",
                "tier": "unknown",
                "status": "no metadata",
                "rows": 0,
                "ingested_at": "unknown",
            })
            continue

        # Check for staleness
        ingested_at = meta.get("ingested_at", "")
        stale_warning = ""
        if ingested_at:
            try:
                ing_dt = datetime.fromisoformat(
                    ingested_at.replace("Z", "+00:00")
                )
                age_days = (
                    datetime.now(timezone.utc) - ing_dt
                ).total_seconds() / 86400
                max_days = config.get("quarantine", {}).get(
                    "max_quarantine_days", 30
                )
                if age_days > max_days:
                    stale_warning = f" [STALE: {age_days:.0f}d]"
            except (ValueError, TypeError):
                pass

        entries.append({
            "name": name,
            "source_id": meta.get("source_id", "unknown"),
            "tier": meta.get("tier", "unknown"),
            "status": meta.get("validation_status", "pending")
                       + stale_warning,
            "rows": meta.get("row_count", 0),
            "ingested_at": ingested_at,
        })

    if not entries:
        print("  Quarantine is empty. No datasets pending review.")
        return []

    # Print formatted table
    print(f"\n{'=' * 80}")
    print("Quarantined Datasets")
    print(f"{'=' * 80}")
    print(f"{'Name':<30} {'Tier':<8} {'Status':<20} {'Rows':<8} {'Source'}")
    print(f"{'-' * 30} {'-' * 8} {'-' * 20} {'-' * 8} {'-' * 30}")
    for e in entries:
        print(
            f"{e['name']:<30} {e['tier']:<8} {e['status']:<20} "
            f"{e['rows']:<8} {e['source_id']}"
        )
    print(f"{'=' * 80}")
    print(f"Total: {len(entries)} dataset(s) in quarantine")

    return entries


def promote(name, config=None):
    """Promote a quarantined dataset to data/aggregated/.

    Moves validated data out of quarantine.  Requires that validation
    has passed (validation_status == "passed" in metadata).

    Args:
        name: Directory name under data/quarantine/.
        config: Pre-loaded trust tier config.

    Returns:
        dict: Result with keys: action, source, destination.
    """
    if config is None:
        config = load_trust_config()

    entry_dir = os.path.join(QUARANTINE_DIR, name)
    if not os.path.isdir(entry_dir):
        print(f"  ERROR: Quarantine entry not found: {name}", file=sys.stderr)
        return {"action": "error", "message": f"Not found: {name}"}

    meta = _read_metadata(entry_dir)
    if meta is None:
        print(f"  ERROR: No metadata for {name}", file=sys.stderr)
        return {"action": "error", "message": f"No metadata: {name}"}

    validation_status = meta.get("validation_status", "pending")
    if validation_status != "passed":
        print(
            f"  ERROR: Cannot promote {name} -- validation status is "
            f"'{validation_status}'. Run --validate-quarantined first.",
            file=sys.stderr,
        )
        return {
            "action": "error",
            "message": f"Validation status: {validation_status}",
        }

    # Move data files to aggregated/
    os.makedirs(AGGREGATED_DIR, exist_ok=True)
    moved_files = []
    for fname in os.listdir(entry_dir):
        if fname.endswith((".csv", ".jsonl")):
            src = os.path.join(entry_dir, fname)
            dst = os.path.join(AGGREGATED_DIR, fname)
            shutil.copy2(src, dst)
            moved_files.append(fname)
            print(f"  Promoted: {fname} -> {AGGREGATED_DIR}")

    if not moved_files:
        print(f"  WARN: No data files found to promote in {name}")
        return {"action": "error", "message": "No data files"}

    # Update metadata
    _update_metadata(entry_dir, {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": "promoted",
    })

    # Log
    source_id = meta.get("source_id", "unknown")
    tier_key = meta.get("tier", "tier3")
    _log_action("promote", name, source_id, tier_key, {
        "files": moved_files,
        "destination": AGGREGATED_DIR,
    })

    # Clean up quarantine entry (keep metadata as audit trail)
    for fname in os.listdir(entry_dir):
        fpath = os.path.join(entry_dir, fname)
        if fname.endswith((".csv", ".jsonl")):
            os.remove(fpath)
    print(f"  Cleaned quarantine entry: {entry_dir}")
    print(f"  Promotion complete for {name}")

    return {
        "action": "promoted",
        "source": entry_dir,
        "destination": AGGREGATED_DIR,
        "files": moved_files,
    }


def reject(name, reason="manual_rejection", config=None):
    """Reject and remove a quarantined dataset.

    Data files are deleted.  Metadata is preserved with rejection
    reason for audit trail.

    Args:
        name: Directory name under data/quarantine/.
        reason: Human-readable rejection reason.
        config: Pre-loaded trust tier config.

    Returns:
        dict: Result with keys: action, name, reason.
    """
    if config is None:
        config = load_trust_config()

    entry_dir = os.path.join(QUARANTINE_DIR, name)
    if not os.path.isdir(entry_dir):
        print(f"  ERROR: Quarantine entry not found: {name}", file=sys.stderr)
        return {"action": "error", "message": f"Not found: {name}"}

    meta = _read_metadata(entry_dir) or {}
    source_id = meta.get("source_id", "unknown")
    tier_key = meta.get("tier", "tier3")

    # Remove data files, keep metadata
    removed = []
    for fname in os.listdir(entry_dir):
        fpath = os.path.join(entry_dir, fname)
        if fname.endswith((".csv", ".jsonl")):
            os.remove(fpath)
            removed.append(fname)

    # Update metadata
    _update_metadata(entry_dir, {
        "validation_status": "rejected",
        "rejected_at": datetime.now(timezone.utc).isoformat(),
        "rejection_reason": reason,
    })

    _log_action("reject", name, source_id, tier_key, {
        "reason": reason,
        "files_removed": removed,
    })

    print(f"  Rejected: {name} ({len(removed)} file(s) removed)")
    print(f"  Reason: {reason}")
    return {"action": "rejected", "name": name, "reason": reason}


def status(config=None):
    """Print a summary of the trust tier system status.

    Shows tier distribution of known sources and quarantine counts.
    """
    if config is None:
        config = load_trust_config()

    sources = config.get("sources", {})
    tier_counts = {}
    for _src, tier_key in sources.items():
        tier_counts[tier_key] = tier_counts.get(tier_key, 0) + 1

    print(f"\n{'=' * 60}")
    print("Trust Tier System Status")
    print(f"{'=' * 60}")

    # Tier breakdown
    tiers = config.get("tiers", {})
    for tier_key in sorted(tiers.keys()):
        tier_cfg = tiers[tier_key]
        count = tier_counts.get(tier_key, 0)
        print(
            f"  {tier_key} ({tier_cfg['label']}): "
            f"{count} source(s), "
            f"validation={tier_cfg['validation']}, "
            f"quarantine={'yes' if tier_cfg['quarantine'] else 'no'}"
        )

    # Quarantine status
    quarantine_count = 0
    pending_count = 0
    passed_count = 0
    failed_count = 0
    if os.path.isdir(QUARANTINE_DIR):
        for name in os.listdir(QUARANTINE_DIR):
            entry_dir = os.path.join(QUARANTINE_DIR, name)
            if not os.path.isdir(entry_dir):
                continue
            quarantine_count += 1
            meta = _read_metadata(entry_dir)
            if meta:
                vs = meta.get("validation_status", "pending")
                if vs == "pending":
                    pending_count += 1
                elif vs == "passed":
                    passed_count += 1
                elif vs == "failed":
                    failed_count += 1

    print(f"\n  Quarantine: {quarantine_count} dataset(s)")
    print(f"    Pending:  {pending_count}")
    print(f"    Passed:   {passed_count} (ready for --promote)")
    print(f"    Failed:   {failed_count}")

    # Log stats
    log_entries = _load_log()
    if log_entries:
        action_counts = {}
        for entry in log_entries:
            a = entry.get("action", "unknown")
            action_counts[a] = action_counts.get(a, 0) + 1
        print(f"\n  Log entries: {len(log_entries)} total")
        for a, c in sorted(action_counts.items()):
            print(f"    {a}: {c}")

    print(f"{'=' * 60}")


# ── CLI ───────────────────────────────────────────────────────────────────


def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Dataset quarantine manager for Na0S trust tier system.  "
            "Routes data through tiered validation and quarantine."
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ingest",
        metavar="FILE",
        help="Ingest a data file, routing based on trust tier.",
    )
    group.add_argument(
        "--review",
        action="store_true",
        help="List all quarantined datasets and their status.",
    )
    group.add_argument(
        "--promote",
        metavar="NAME",
        help="Promote a validated quarantine entry to aggregated/.",
    )
    group.add_argument(
        "--reject",
        metavar="NAME",
        help="Reject and remove a quarantined dataset.",
    )
    group.add_argument(
        "--validate-quarantined",
        action="store_true",
        help="Run validation on all quarantined datasets.",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Print trust tier system status summary.",
    )

    parser.add_argument(
        "--source",
        metavar="SOURCE_ID",
        help=(
            "Source identifier for tier resolution (required with --ingest). "
            "E.g., 'deepset/prompt-injections', 'reddit/r/ChatGPT'."
        ),
    )
    parser.add_argument(
        "--reason",
        metavar="REASON",
        default="manual_rejection",
        help="Rejection reason (used with --reject).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=f"Path to trust_tiers.yaml (default: {TRUST_TIERS_PATH}).",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load config
    config_path = args.config
    if config_path:
        global TRUST_TIERS_PATH
        TRUST_TIERS_PATH = config_path

    try:
        config = load_trust_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ERROR loading trust config: {exc}", file=sys.stderr)
        return 1

    if args.ingest:
        if not args.source:
            print(
                "  ERROR: --source is required with --ingest",
                file=sys.stderr,
            )
            return 1
        result = ingest(args.ingest, args.source, config)
        return 0 if result.get("action") != "error" else 1

    elif args.review:
        review(config)
        return 0

    elif args.promote:
        result = promote(args.promote, config)
        return 0 if result.get("action") == "promoted" else 1

    elif args.reject:
        result = reject(args.reject, reason=args.reason, config=config)
        return 0 if result.get("action") == "rejected" else 1

    elif args.validate_quarantined:
        results = validate_quarantined(config)
        failed = sum(1 for r in results if r["status"] == "failed")
        print(
            f"\n  Validated {len(results)} dataset(s): "
            f"{len(results) - failed} passed, {failed} failed"
        )
        return 0

    elif args.status:
        status(config)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
