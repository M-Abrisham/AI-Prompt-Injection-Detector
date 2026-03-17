#!/usr/bin/env python3
"""Dataset quarantine manager for Na0S trust tier system.

Enforces tiered validation and quarantine policies for all data sources.
Datasets from untrusted sources (tier3/tier4) are held in data/quarantine/
until they pass validation, move through a staging layer for label quality
checks, and are explicitly promoted to production by a maintainer.

Workflow:
    1. Ingest: tier1/tier2 -> data/raw/ (direct), tier3/tier4 -> data/quarantine/
    2. Validate: --validate-quarantined runs tier-appropriate checks
    3. Stage: --promote-validated moves passed entries to data/staging/
    4. Stage Validate: --validate-staged runs label quality checks
    5. Promote: --promote-staged-validated moves to data/aggregated/ (production)

    Reject: Run ``quarantine.py --reject <name>`` to permanently
    remove quarantined data that failed review.

All actions are logged to data/quarantine/quarantine_log.json.

Usage::

    python scripts/quarantine.py --ingest <file> --source <source_id>
    python scripts/quarantine.py --review
    python scripts/quarantine.py --promote <name>
    python scripts/quarantine.py --promote-validated
    python scripts/quarantine.py --reject <name>
    python scripts/quarantine.py --status
    python scripts/quarantine.py --validate-quarantined
    python scripts/quarantine.py --validate-staged
    python scripts/quarantine.py --promote-to-production <name>
    python scripts/quarantine.py --promote-staged-validated
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
STAGING_DIR = os.path.join(ROOT, "data", "staging")


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
                "direct_pass", "expire_warning", "stage",
                "validate_staged", "promote_to_production".
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

    # Compute trust score
    from scripts.trust_score import compute_trust_score

    log_entries = _load_log()
    trust_result = compute_trust_score(
        source_id=source_id,
        tier=tier_key,
        log_entries=log_entries,
    )
    trust_score_val = trust_result["trust_score"]

    print(f"  Ingesting: {basename}")
    print(f"    Source:  {source_id}")
    print(f"    Tier:    {tier_key} ({tier_cfg['label']})")
    print(f"    Rows:    {row_count}")
    print(f"    SHA-256: {file_hash[:16]}...")
    print(f"    Trust:   {trust_score_val:.3f} ({trust_result['gate_decision']})")

    if tier_cfg.get("quarantine", False):
        # Route to quarantine
        dest_dir = os.path.join(QUARANTINE_DIR, safe_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, basename)
        shutil.copy2(file_path, dest_path)
        _write_metadata(dest_dir, source_id, tier_key, file_hash, row_count)
        _update_metadata(dest_dir, {
            "trust_score": trust_score_val,
            "trust_dimensions": trust_result["dimensions"],
            "trust_gate": trust_result["gate_decision"],
        })
        _log_action("ingest", safe_name, source_id, tier_key, {
            "file": basename,
            "rows": row_count,
            "sha256": file_hash,
            "destination": "quarantine",
            "trust_score": trust_score_val,
            "trust_gate": trust_result["gate_decision"],
        })
        print(f"    Action:  QUARANTINED -> {dest_dir}")
        return {
            "action": "quarantined",
            "tier": tier_key,
            "destination": dest_dir,
            "name": safe_name,
            "trust_score": trust_score_val,
            "trust_gate": trust_result["gate_decision"],
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
            "trust_score": trust_score_val,
            "trust_gate": trust_result["gate_decision"],
        })
        print(f"    Action:  DIRECT PASS -> {dest_path}")
        return {
            "action": "direct_pass",
            "tier": tier_key,
            "destination": dest_path,
            "name": safe_name,
            "trust_score": trust_score_val,
            "trust_gate": trust_result["gate_decision"],
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
    """Promote a quarantined dataset to data/staging/.

    Moves validated data from quarantine to the staging layer for
    additional label quality checks before production.  Requires that
    validation has passed (validation_status == "passed" in metadata).

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

    # Copy data files to staging/
    staging_entry_dir = os.path.join(STAGING_DIR, name)
    os.makedirs(staging_entry_dir, exist_ok=True)
    moved_files = []
    for fname in os.listdir(entry_dir):
        if fname.endswith((".csv", ".jsonl")):
            src = os.path.join(entry_dir, fname)
            dst = os.path.join(staging_entry_dir, fname)
            shutil.copy2(src, dst)
            moved_files.append(fname)
            print(f"  Staged: {fname} -> {staging_entry_dir}")

    if not moved_files:
        print(f"  WARN: No data files found to promote in {name}")
        return {"action": "error", "message": "No data files"}

    # Copy metadata.json to staging directory
    meta_src = os.path.join(entry_dir, "metadata.json")
    meta_dst = os.path.join(staging_entry_dir, "metadata.json")
    if os.path.isfile(meta_src):
        shutil.copy2(meta_src, meta_dst)

    # Update staging metadata
    _update_metadata(staging_entry_dir, {
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": "staged",
    })

    # Update quarantine metadata
    _update_metadata(entry_dir, {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": "staged",
    })

    # Log
    source_id = meta.get("source_id", "unknown")
    tier_key = meta.get("tier", "tier3")
    _log_action("stage", name, source_id, tier_key, {
        "files": moved_files,
        "destination": staging_entry_dir,
    })

    # Clean up quarantine entry data files (keep metadata as audit trail)
    for fname in os.listdir(entry_dir):
        fpath = os.path.join(entry_dir, fname)
        if fname.endswith((".csv", ".jsonl")):
            os.remove(fpath)
    print(f"  Cleaned quarantine entry: {entry_dir}")
    print(f"  Staging complete for {name}")

    return {
        "action": "promoted",
        "source": entry_dir,
        "destination": staging_entry_dir,
        "files": moved_files,
    }


def promote_validated(config=None):
    """Promote all quarantine entries that already passed validation.

    Returns:
        dict: Summary with counts for promoted/pending/failed/rejected/errors.
    """
    if config is None:
        config = load_trust_config()

    summary = {
        "eligible_entries": 0,
        "promoted": 0,
        "pending": 0,
        "failed": 0,
        "rejected": 0,
        "errors": 0,
    }

    if not os.path.isdir(QUARANTINE_DIR):
        print("  No quarantine directory found.")
        return summary

    for name in sorted(os.listdir(QUARANTINE_DIR)):
        entry_dir = os.path.join(QUARANTINE_DIR, name)
        if not os.path.isdir(entry_dir):
            continue

        data_files = [
            f for f in os.listdir(entry_dir)
            if f.endswith((".csv", ".jsonl"))
        ]
        if not data_files:
            continue

        summary["eligible_entries"] += 1
        meta = _read_metadata(entry_dir) or {}
        vstatus = meta.get("validation_status", "pending")

        if vstatus == "passed":
            # Check trust score gate before promoting
            trust_s = meta.get("trust_score")
            trust_gate = meta.get("trust_gate", "")
            if trust_gate == "auto_reject":
                summary["failed"] += 1
                print(
                    f"  BLOCKED: {name} trust score {trust_s:.3f} "
                    f"below staging threshold (auto_reject)"
                )
                continue
            result = promote(name, config)
            if result.get("action") == "promoted":
                summary["promoted"] += 1
            else:
                summary["errors"] += 1
        elif vstatus == "pending":
            summary["pending"] += 1
            print(
                f"  SKIP: {name} remains pending validation. "
                "Run --validate-quarantined first."
            )
        elif vstatus == "failed":
            summary["failed"] += 1
            print(
                f"  SKIP: {name} failed validation. "
                "Review and --reject or fix source data."
            )
        elif vstatus == "rejected":
            summary["rejected"] += 1
            print(f"  SKIP: {name} already rejected.")
        else:
            summary["errors"] += 1
            print(f"  SKIP: {name} has unknown validation status '{vstatus}'.")

    print("\n  Promotion summary")
    print(f"    Eligible entries: {summary['eligible_entries']}")
    print(f"    Promoted:         {summary['promoted']}")
    print(f"    Pending:          {summary['pending']}")
    print(f"    Failed:           {summary['failed']}")
    print(f"    Rejected:         {summary['rejected']}")
    print(f"    Errors:           {summary['errors']}")
    return summary


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


# ── Staging Operations ────────────────────────────────────────────────────


def _load_staged_data(file_path):
    """Load rows from a CSV or JSONL file for label quality checks.

    Returns:
        list[dict]: List of row dicts with at least 'text' and 'label' keys.
    """
    import csv

    ext = os.path.splitext(file_path)[1].lower()
    rows = []
    try:
        if ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rows.append(row)
        elif ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        print(f"    WARN: Could not load {file_path}: {exc}")
    return rows


def _check_label_quality(rows):
    """Run label quality checks on loaded data rows.

    Checks:
        a. Label distribution: both classes must be present
        b. Suspicious label flips: safe text with injection phrases,
           injection text without any injection signal
        c. Minimum row count >= 10

    Args:
        rows: List of row dicts with 'text' and 'label' keys.

    Returns:
        dict: Result with keys: passed, issues (list of strings).
    """
    issues = []

    # c. Minimum row count
    if len(rows) < 10:
        issues.append(f"Insufficient rows: {len(rows)} (minimum 10 required)")

    # Extract labels (handle various label column names)
    labels = []
    for row in rows:
        label = row.get("label", row.get("is_injection", row.get("class", None)))
        if label is not None:
            labels.append(str(label).strip().lower())

    if not labels:
        issues.append("No label column found in data")
        return {"passed": False, "issues": issues}

    # a. Label distribution -- both classes present
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        issues.append(
            f"Only one label class present: {unique_labels}. "
            "Both classes (e.g., 0/1 or safe/injection) required."
        )

    # b. Suspicious label flips
    injection_phrases = [
        "ignore previous", "ignore above", "disregard",
        "new instruction", "override", "system prompt",
        "forget everything", "you are now", "act as",
        "jailbreak", "bypass", "reveal your",
    ]
    suspicious_safe = 0
    suspicious_injection = 0
    for row in rows:
        text = str(row.get("text", row.get("prompt", ""))).lower()
        label = str(
            row.get("label", row.get("is_injection", row.get("class", "")))
        ).strip().lower()

        is_safe_label = label in ("0", "safe", "benign", "false", "no")
        is_injection_label = label in ("1", "injection", "malicious", "true", "yes")

        if is_safe_label:
            if any(phrase in text for phrase in injection_phrases):
                suspicious_safe += 1
        elif is_injection_label:
            has_signal = any(phrase in text for phrase in injection_phrases)
            # Also check for very short injection-labeled text (likely mislabeled)
            if not has_signal and len(text) < 20:
                suspicious_injection += 1

    if suspicious_safe > 0:
        issues.append(
            f"Found {suspicious_safe} safe-labeled row(s) containing "
            "injection phrases (possible label flip)"
        )
    if suspicious_injection > 0:
        issues.append(
            f"Found {suspicious_injection} injection-labeled row(s) with "
            "no injection signal and very short text (possible label flip)"
        )

    passed = len(issues) == 0
    return {"passed": passed, "issues": issues}


def validate_staged(config=None):
    """Run label quality validation on all datasets in data/staging/.

    For each staged entry directory:
      - Read metadata.json
      - Load CSV/JSONL data files
      - Run label quality checks
      - Update metadata with staging_validation_status and staging_validated_at

    Returns:
        list[dict]: Validation results for each staged entry.
    """
    if config is None:
        config = load_trust_config()

    if not os.path.isdir(STAGING_DIR):
        print("  No staging directory found.")
        return []

    results = []
    for name in sorted(os.listdir(STAGING_DIR)):
        entry_dir = os.path.join(STAGING_DIR, name)
        if not os.path.isdir(entry_dir):
            continue

        meta = _read_metadata(entry_dir)
        if meta is None:
            print(f"  WARN: {name} -- no metadata.json, skipping")
            continue

        print(f"\n  Validating staged: {name}")

        # Find data files
        data_files = [
            f for f in os.listdir(entry_dir)
            if f.endswith((".csv", ".jsonl"))
        ]
        if not data_files:
            print(f"    No data files found in {entry_dir}")
            _update_metadata(entry_dir, {
                "staging_validation_status": "error",
                "staging_validated_at": datetime.now(timezone.utc).isoformat(),
            })
            results.append({
                "name": name,
                "status": "error",
                "issues": ["No data files found"],
            })
            continue

        # Run label quality checks on all data files
        all_passed = True
        all_issues = []
        for data_file in data_files:
            data_path = os.path.join(entry_dir, data_file)
            rows = _load_staged_data(data_path)
            check_result = _check_label_quality(rows)
            if not check_result["passed"]:
                all_passed = False
            if check_result["issues"]:
                all_issues.extend(
                    [f"{data_file}: {issue}" for issue in check_result["issues"]]
                )
            print(
                f"    {data_file}: {'PASS' if check_result['passed'] else 'FAIL'}"
            )
            for issue in check_result["issues"]:
                print(f"      - {issue}")

        status_val = "passed" if all_passed else "failed"
        _update_metadata(entry_dir, {
            "staging_validation_status": status_val,
            "staging_validated_at": datetime.now(timezone.utc).isoformat(),
            "staging_validation_issues": all_issues,
        })

        source_id = meta.get("source_id", "unknown")
        tier_key = meta.get("tier", "tier3")
        _log_action("validate_staged", name, source_id, tier_key, {
            "status": status_val,
            "files_checked": len(data_files),
            "issues": all_issues,
        })

        results.append({
            "name": name,
            "status": status_val,
            "issues": all_issues,
        })

    return results


def promote_to_production(name, config=None):
    """Promote a validated staged dataset from data/staging/ to data/aggregated/.

    Requires that staging_validation_status == "passed" in metadata.

    Args:
        name: Directory name under data/staging/.
        config: Pre-loaded trust tier config.

    Returns:
        dict: Result with keys: action, source, destination.
    """
    if config is None:
        config = load_trust_config()

    entry_dir = os.path.join(STAGING_DIR, name)
    if not os.path.isdir(entry_dir):
        print(f"  ERROR: Staging entry not found: {name}", file=sys.stderr)
        return {"action": "error", "message": f"Not found: {name}"}

    meta = _read_metadata(entry_dir)
    if meta is None:
        print(f"  ERROR: No metadata for {name}", file=sys.stderr)
        return {"action": "error", "message": f"No metadata: {name}"}

    staging_status = meta.get("staging_validation_status", "pending")
    if staging_status != "passed":
        print(
            f"  ERROR: Cannot promote {name} to production -- "
            f"staging validation status is '{staging_status}'. "
            f"Run --validate-staged first.",
            file=sys.stderr,
        )
        return {
            "action": "error",
            "message": f"Staging validation status: {staging_status}",
        }

    # Copy data files to aggregated/
    os.makedirs(AGGREGATED_DIR, exist_ok=True)
    moved_files = []
    for fname in os.listdir(entry_dir):
        if fname.endswith((".csv", ".jsonl")):
            src = os.path.join(entry_dir, fname)
            dst = os.path.join(AGGREGATED_DIR, fname)
            shutil.copy2(src, dst)
            moved_files.append(fname)
            print(f"  Promoted to production: {fname} -> {AGGREGATED_DIR}")

    if not moved_files:
        print(f"  WARN: No data files found to promote in {name}")
        return {"action": "error", "message": "No data files"}

    # Update metadata
    _update_metadata(entry_dir, {
        "promoted_to_production_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": "production",
    })

    # Log
    source_id = meta.get("source_id", "unknown")
    tier_key = meta.get("tier", "tier3")
    _log_action("promote_to_production", name, source_id, tier_key, {
        "files": moved_files,
        "destination": AGGREGATED_DIR,
    })

    # Clean up staging entry data files (keep metadata as audit trail)
    for fname in os.listdir(entry_dir):
        fpath = os.path.join(entry_dir, fname)
        if fname.endswith((".csv", ".jsonl")):
            os.remove(fpath)
    print(f"  Cleaned staging entry: {entry_dir}")
    print(f"  Production promotion complete for {name}")

    return {
        "action": "promoted_to_production",
        "source": entry_dir,
        "destination": AGGREGATED_DIR,
        "files": moved_files,
    }


def promote_staged_validated(config=None):
    """Promote all staged entries that passed staging validation to production.

    Returns:
        dict: Summary with counts for promoted/pending/failed/errors.
    """
    if config is None:
        config = load_trust_config()

    summary = {
        "eligible_entries": 0,
        "promoted": 0,
        "pending": 0,
        "failed": 0,
        "errors": 0,
    }

    if not os.path.isdir(STAGING_DIR):
        print("  No staging directory found.")
        return summary

    for name in sorted(os.listdir(STAGING_DIR)):
        entry_dir = os.path.join(STAGING_DIR, name)
        if not os.path.isdir(entry_dir):
            continue

        data_files = [
            f for f in os.listdir(entry_dir)
            if f.endswith((".csv", ".jsonl"))
        ]
        if not data_files:
            continue

        summary["eligible_entries"] += 1
        meta = _read_metadata(entry_dir) or {}
        vstatus = meta.get("staging_validation_status", "pending")

        if vstatus == "passed":
            result = promote_to_production(name, config)
            if result.get("action") == "promoted_to_production":
                summary["promoted"] += 1
            else:
                summary["errors"] += 1
        elif vstatus in ("pending", None):
            summary["pending"] += 1
            print(
                f"  SKIP: {name} staging validation pending. "
                "Run --validate-staged first."
            )
        elif vstatus == "failed":
            summary["failed"] += 1
            print(
                f"  SKIP: {name} failed staging validation. "
                "Review label quality issues."
            )
        else:
            summary["errors"] += 1
            print(
                f"  SKIP: {name} has unknown staging validation "
                f"status '{vstatus}'."
            )

    print("\n  Staged promotion summary")
    print(f"    Eligible entries: {summary['eligible_entries']}")
    print(f"    Promoted:         {summary['promoted']}")
    print(f"    Pending:          {summary['pending']}")
    print(f"    Failed:           {summary['failed']}")
    print(f"    Errors:           {summary['errors']}")
    return summary


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

    # Staging status
    staging_count = 0
    staging_pending = 0
    staging_passed = 0
    staging_failed = 0
    if os.path.isdir(STAGING_DIR):
        for name in os.listdir(STAGING_DIR):
            entry_dir = os.path.join(STAGING_DIR, name)
            if not os.path.isdir(entry_dir):
                continue
            staging_count += 1
            meta = _read_metadata(entry_dir)
            if meta:
                svs = meta.get("staging_validation_status", "pending")
                if svs == "pending":
                    staging_pending += 1
                elif svs == "passed":
                    staging_passed += 1
                elif svs == "failed":
                    staging_failed += 1

    print(f"\n  Staging: {staging_count} dataset(s)")
    print(f"    Pending:  {staging_pending}")
    print(f"    Passed:   {staging_passed} (ready for --promote-to-production)")
    print(f"    Failed:   {staging_failed}")

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
        help="Promote a validated quarantine entry to staging/.",
    )
    group.add_argument(
        "--promote-validated",
        action="store_true",
        help="Promote all quarantine entries with validation_status=passed to staging.",
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
        "--validate-staged",
        action="store_true",
        help="Run label quality validation on all staged datasets.",
    )
    group.add_argument(
        "--promote-to-production",
        metavar="NAME",
        help="Promote a validated staging entry to production (aggregated/).",
    )
    group.add_argument(
        "--promote-staged-validated",
        action="store_true",
        help="Promote all staged entries that passed staging validation to production.",
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

    elif args.promote_validated:
        result = promote_validated(config)
        return 0 if result.get("errors", 0) == 0 else 1

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

    elif args.validate_staged:
        results = validate_staged(config)
        failed = sum(1 for r in results if r["status"] == "failed")
        print(
            f"\n  Staged validation: {len(results)} dataset(s): "
            f"{len(results) - failed} passed, {failed} failed"
        )
        return 0

    elif args.promote_to_production:
        result = promote_to_production(args.promote_to_production, config)
        return 0 if result.get("action") == "promoted_to_production" else 1

    elif args.promote_staged_validated:
        result = promote_staged_validated(config)
        return 0 if result.get("errors", 0) == 0 else 1

    elif args.status:
        status(config)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
