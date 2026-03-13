#!/usr/bin/env python3
"""Deploy trained model files and update KNOWN_HASHES.

Copies model.pkl and tfidf_vectorizer.pkl from data/processed/ into the
package directory (src/na0s/models/) and rewrites the KNOWN_HASHES dict
in src/na0s/models/__init__.py with fresh SHA-256 digests.

Usage::

    python scripts/deploy_model.py
    python scripts/deploy_model.py --rollback
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(ROOT, "data", "processed")
DEST_DIR = os.path.join(ROOT, "src", "na0s", "models")
INIT_PATH = os.path.join(DEST_DIR, "__init__.py")

MODEL_FILES = ["model.pkl", "tfidf_vectorizer.pkl"]
OPTIONAL_MODEL_FILES = ["structural_scaler.pkl"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _backup_file(dst):
    """Create two backups of *dst*: a timestamped one and a plain .bak.

    Returns the path of the timestamped backup so callers can verify it.
    Raises SystemExit(1) if either copy fails or the backup cannot be
    verified.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    fname = os.path.basename(dst)
    ts_bak = dst + f".{timestamp}.bak"
    plain_bak = dst + ".bak"

    try:
        shutil.copy2(dst, ts_bak)
    except OSError as exc:
        print(f"ERROR: could not create timestamped backup {ts_bak}: {exc}")
        sys.exit(1)

    try:
        shutil.copy2(dst, plain_bak)
    except OSError as exc:
        print(f"ERROR: could not create plain backup {plain_bak}: {exc}")
        sys.exit(1)

    # Verify timestamped backup exists and has the same size as the original
    if not os.path.exists(ts_bak):
        print(f"ERROR: timestamped backup {ts_bak} was not created.")
        sys.exit(1)

    orig_size = os.path.getsize(dst)
    bak_size = os.path.getsize(ts_bak)
    if orig_size != bak_size:
        print(
            f"ERROR: backup size mismatch for {fname}: "
            f"original={orig_size} bytes, backup={bak_size} bytes"
        )
        sys.exit(1)

    print(f"  Backed up {fname} -> {ts_bak}  ({bak_size} bytes)")
    print(f"  Backed up {fname} -> {plain_bak}")
    return ts_bak


def deploy(source_dir=None, dest_dir=None, init_path=None):
    """Copy model files from *source_dir* to *dest_dir* and update KNOWN_HASHES.

    All parameters default to the module-level constants; they are exposed
    here so that tests can redirect I/O to temporary directories without
    monkeypatching module globals.
    """
    if source_dir is None:
        source_dir = SOURCE_DIR
    if dest_dir is None:
        dest_dir = DEST_DIR
    if init_path is None:
        init_path = INIT_PATH

    # 1. Copy model files (required + optional)
    new_hashes = {}
    all_files = MODEL_FILES + [
        f for f in OPTIONAL_MODEL_FILES
        if os.path.exists(os.path.join(source_dir, f))
    ]
    for fname in all_files:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(dest_dir, fname)

        if not os.path.exists(src):
            print(f"ERROR: {src} not found. Run the training pipeline first.")
            sys.exit(1)

        # --- SHA-256 comparison: skip copy when source == destination ---
        if os.path.exists(dst):
            src_digest = _sha256(src)
            dst_digest = _sha256(dst)
            if src_digest == dst_digest:
                print(f"  {fname}: unchanged (sha256 identical), skipping copy")
                new_hashes[fname] = src_digest
                continue

            # Backup the existing destination before overwriting
            _backup_file(dst)

        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            print(f"ERROR: could not copy {src} -> {dst}: {exc}")
            sys.exit(1)

        digest = _sha256(dst)
        new_hashes[fname] = digest
        print(f"  Copied {fname}  sha256={digest[:16]}...")

    # 2. Update KNOWN_HASHES in __init__.py
    try:
        with open(init_path, "r") as f:
            content = f.read()
    except OSError as exc:
        print(f"ERROR: could not read {init_path}: {exc}")
        sys.exit(1)

    # Build replacement dict literal
    entries = ",\n".join(
        f'    "{fname}": "{digest}"' for fname, digest in sorted(new_hashes.items())
    )
    new_dict = "KNOWN_HASHES = {\n" + entries + ",\n}"

    # Replace the existing KNOWN_HASHES block
    updated = re.sub(
        r"KNOWN_HASHES\s*=\s*\{[^}]*\}",
        new_dict,
        content,
        count=1,
    )

    if not re.search(r"KNOWN_HASHES\s*=\s*\{", content):
        print("WARNING: KNOWN_HASHES block not found in __init__.py")
    elif updated == content:
        print("  KNOWN_HASHES unchanged (hashes match)")
    else:
        try:
            with open(init_path, "w") as f:
                f.write(updated)
        except OSError as exc:
            print(f"ERROR: could not write {init_path}: {exc}")
            sys.exit(1)
        print(f"  Updated KNOWN_HASHES in {init_path}")

    # 3. Summary
    print("\nDeployed model hashes:")
    for fname, digest in sorted(new_hashes.items()):
        print(f"  {fname}: {digest}")

    sys.exit(0)


def rollback(dest_dir=None):
    """Restore model files from their plain .bak counterparts.

    Looks for ``<dest_dir>/<fname>.bak`` for every file in MODEL_FILES and
    copies it back over the live file.  Exits with code 1 if any .bak is
    missing or a restore fails; exits with 0 on complete success.
    """
    if dest_dir is None:
        dest_dir = DEST_DIR

    all_ok = True
    for fname in MODEL_FILES + OPTIONAL_MODEL_FILES:
        live = os.path.join(dest_dir, fname)
        bak = live + ".bak"

        if not os.path.exists(bak):
            if fname in OPTIONAL_MODEL_FILES:
                continue  # Optional files may not have backups
            print(f"ERROR: rollback backup not found: {bak}")
            all_ok = False
            continue

        try:
            shutil.copy2(bak, live)
            print(f"  Restored {fname} from {bak}")
        except OSError as exc:
            print(f"ERROR: could not restore {fname}: {exc}")
            all_ok = False

    if not all_ok:
        sys.exit(1)

    print("\nRollback complete.")
    sys.exit(0)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Deploy trained model files to the package directory."
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Restore model files from .bak backups instead of deploying.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.rollback:
        rollback()
    else:
        deploy()
