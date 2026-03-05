#!/usr/bin/env python3
"""Deploy trained model files and update KNOWN_HASHES.

Copies model.pkl and tfidf_vectorizer.pkl from data/processed/ into the
package directory (src/na0s/models/) and rewrites the KNOWN_HASHES dict
in src/na0s/models/__init__.py with fresh SHA-256 digests.

Usage::

    python scripts/deploy_model.py
"""

import hashlib
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(ROOT, "data", "processed")
DEST_DIR = os.path.join(ROOT, "src", "na0s", "models")
INIT_PATH = os.path.join(DEST_DIR, "__init__.py")

MODEL_FILES = ["model.pkl", "tfidf_vectorizer.pkl"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def deploy():
    # 1. Copy model files
    new_hashes = {}
    for fname in MODEL_FILES:
        src = os.path.join(SOURCE_DIR, fname)
        dst = os.path.join(DEST_DIR, fname)

        if not os.path.exists(src):
            print(f"ERROR: {src} not found. Run the training pipeline first.")
            sys.exit(1)

        if os.path.exists(dst):
            shutil.copy2(dst, dst + ".bak")
            print(f"  Backed up {fname} \u2192 {fname}.bak")

        shutil.copy2(src, dst)
        digest = _sha256(dst)
        new_hashes[fname] = digest
        print(f"  Copied {fname}  sha256={digest[:16]}...")

    # 2. Update KNOWN_HASHES in __init__.py
    with open(INIT_PATH, "r") as f:
        content = f.read()

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
        with open(INIT_PATH, "w") as f:
            f.write(updated)
        print(f"  Updated KNOWN_HASHES in {INIT_PATH}")

    # 3. Summary
    print("\nDeployed model hashes:")
    for fname, digest in sorted(new_hashes.items()):
        print(f"  {fname}: {digest}")


if __name__ == "__main__":
    deploy()
