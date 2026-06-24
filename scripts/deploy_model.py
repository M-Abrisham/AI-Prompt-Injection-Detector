#!/usr/bin/env python3
"""Deploy trained model files and update KNOWN_HASHES.

Copies model.pkl and tfidf_vectorizer.pkl from data/processed/ into the
package directory (src/na0s/models/) and updates the KNOWN_HASHES dict in
src/na0s/models/__init__.py. The update MERGES fresh digests into the
existing KNOWN_HASHES, preserving entries for bundled pickle files that
are not re-emitted this run (e.g. model_embedding.pkl, structural_scaler.pkl).
Files copied this run override their prior digest; untouched bundled
entries are kept verbatim so every bundled *.pkl always retains its
authoritative hash (none of them ship sidecar .sha256 files).

Usage::

    python scripts/deploy_model.py
    python scripts/deploy_model.py --rollback
"""

import argparse
import ast
import hashlib
import os
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(ROOT, "data", "processed")
DEST_DIR = os.path.join(ROOT, "src", "na0s", "models")
INIT_PATH = os.path.join(DEST_DIR, "__init__.py")

MODEL_FILES = ["model.pkl", "tfidf_vectorizer.pkl"]
# char_tfidf_vectorizer.pkl is conditionally required: if the model was
# trained with character-level features (i.e. char_tfidf_vectorizer.pkl
# exists alongside model.pkl in source_dir), deployment MUST include it
# to avoid a feature-dimension mismatch at inference time.
OPTIONAL_MODEL_FILES = ["structural_scaler.pkl"]
CHAR_VECTORIZER = "char_tfidf_vectorizer.pkl"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sidecar(dst, digest):
    """Write a fresh plain-SHA-256 sidecar (``<dst>.sha256``) for *dst*.

    ``deploy_model`` copies the .pkl but not its sidecar, so without this the
    destination keeps whatever (possibly stale) sidecar shipped — the .sha256
    drifts from the .pkl it is supposed to guard (the F-AR6 bug).  Regenerate it
    from the freshly-deployed file every time, including the unchanged-skip path.

    Uses the canonical versioned format from ``safe_pickle._format_sidecar``
    (``v1:sha256:<digest>``) so it never drifts from what ``safe_dump`` writes;
    falls back to the legacy bare-hex form (still accepted by ``_parse_sidecar``)
    if na0s is not importable, so deployment never breaks on the sidecar.  The
    primary supply-chain anchor remains KNOWN_HASHES in __init__.py; this keeps
    the secondary/legacy sidecar path honest.
    """
    sidecar_path = dst + ".sha256"
    try:
        from na0s.integrity.safe_pickle import _format_sidecar
        content = _format_sidecar("sha256", digest)
    except Exception:
        content = digest  # legacy bare-hex form; _parse_sidecar accepts it
    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        print(f"ERROR: could not write sidecar {sidecar_path}: {exc}")
        sys.exit(1)
    print(f"  Wrote sidecar {os.path.basename(sidecar_path)}")


def _parse_known_hashes(content):
    """Extract the existing KNOWN_HASHES dict from *content* (init source).

    Returns a ``{filename: digest}`` dict, or ``{}`` if no parseable
    KNOWN_HASHES block is found. Uses the SAME stdlib ``ast`` approach as the
    rewriter (:func:`_rewrite_known_hashes`): :func:`ast.parse` the module,
    reuse :func:`_find_known_hashes_assign` to locate the assignment, and
    :func:`ast.literal_eval` the assignment's **value node** directly — no
    regex brace-matching. The old non-greedy regex (``\\{.*?\\}``) truncated at
    the first inner ``}`` (e.g. a comment or a future structured value), which
    made ``literal_eval`` fail → this returned ``{}`` → preserve-and-merge
    silently DROPPED every bundled entry. Operating on the AST value node makes
    inner braces irrelevant (none are ever counted).

    Fail-safe contract (unchanged): return ``{}`` — never raise — when *content*
    does not parse, has no ``KNOWN_HASHES`` assignment, or the value is not a
    str→str dict; only ``(ValueError, SyntaxError)`` are caught (no bare
    ``except``), so this never executes code from the init source.
    """
    try:
        tree = ast.parse(content)
    except (ValueError, SyntaxError):
        return {}
    node = _find_known_hashes_assign(tree)
    if node is None:
        return {}
    try:
        value = ast.literal_eval(node.value)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(value, dict):
        return {}
    # Keep only string->string entries (real digest records).
    return {
        k: v
        for k, v in value.items()
        if isinstance(k, str) and isinstance(v, str)
    }


def _find_known_hashes_assign(tree):
    """Return the ``ast.Assign`` node for ``KNOWN_HASHES = {...}`` or None.

    Walks the parsed module looking for an assignment whose (single) target
    is ``Name(id="KNOWN_HASHES")``.  Returns ``None`` gracefully when no such
    assignment exists, so the caller can warn-and-exit-0 for the absent-block
    case rather than raising ``StopIteration``/``KeyError``.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "KNOWN_HASHES":
                return node
    return None


def _render_known_hashes(new_hashes):
    """Render the deterministic ``KNOWN_HASHES = {...}`` literal text."""
    entries = ",\n".join(
        f'    "{fname}": "{digest}"'
        for fname, digest in sorted(new_hashes.items())
    )
    return "KNOWN_HASHES = {\n" + entries + ",\n}"


def _rewrite_known_hashes(content, new_hashes):
    """AST-locate the ``KNOWN_HASHES`` assignment and rewrite it in *content*.

    Rather than a brace-fragile regex (``KNOWN_HASHES\\s*=\\s*\\{[^}]*\\}``,
    which truncates at the first inner ``}`` — e.g. a ``# {…}`` comment or a
    future nested literal — and silently emits invalid Python), this parses
    *content* with :func:`ast.parse`, finds the ``KNOWN_HASHES`` assignment
    node, and splices a freshly-rendered literal over the **whole assignment
    statement's** source span (``lineno/col_offset`` .. ``end_lineno/
    end_col_offset``).  Brace counts inside the literal are irrelevant because
    no braces are ever counted.

    Returns one of:
      * ``(content, "absent")``  — no ``KNOWN_HASHES`` assignment found
        (caller warns + exits 0);
      * ``(updated, "ok")``      — rewrite succeeded AND the result was
        verified: ``ast.parse(updated)`` parses, and re-extracting the new
        ``KNOWN_HASHES`` value and ``ast.literal_eval``-ing it round-trips to
        ``new_hashes``;
      * ``(content, "error")``   — *content* does not parse, the rewritten
        module does not parse, or the round-trip verification does not match.
        The caller MUST fail closed (restore any backup, non-zero exit) and
        MUST NOT write — the original file is never overwritten with something
        that does not parse.

    This is a pure function: it performs NO I/O and never calls ``sys.exit``,
    so the caller owns the backup / restore / exit policy.

    ``libcst`` would give a formatting-preserving codemod, but it is not a
    dependency of this repo; stdlib ``ast`` is used so the deploy script grows
    no new third-party requirement.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        print(f"ERROR: existing __init__.py does not parse: {exc}")
        return content, "error"

    node = _find_known_hashes_assign(tree)
    if node is None:
        return content, "absent"

    new_literal = _render_known_hashes(new_hashes)

    # Splice over the entire assignment statement span. Python >= 3.8 always
    # populates end_lineno/end_col_offset (env is 3.14.4 — verified), so the
    # slice is exact regardless of how the dict body is formatted.
    lines = content.splitlines(keepends=True)
    start_off = _offset_of(lines, node.lineno, node.col_offset)
    end_off = _offset_of(lines, node.end_lineno, node.end_col_offset)
    updated = content[:start_off] + new_literal + content[end_off:]

    # --- Output verification (fail closed) -------------------------------
    try:
        verify_tree = ast.parse(updated)
    except SyntaxError as exc:
        print(f"ERROR: rewritten __init__.py would not parse: {exc}")
        return content, "error"

    verify_node = _find_known_hashes_assign(verify_tree)
    if verify_node is None:
        print("ERROR: rewritten __init__.py lost its KNOWN_HASHES assignment.")
        return content, "error"
    try:
        round_tripped = ast.literal_eval(verify_node.value)
    except (ValueError, SyntaxError) as exc:
        print(f"ERROR: rewritten KNOWN_HASHES does not literal_eval: {exc}")
        return content, "error"
    if round_tripped != dict(new_hashes):
        print(
            "ERROR: rewritten KNOWN_HASHES does not round-trip to the "
            "intended hashes; aborting without writing."
        )
        return content, "error"

    return updated, "ok"


def _offset_of(lines, lineno, col_offset):
    """Convert a 1-based *lineno* + 0-based *col_offset* into a string index
    into the joined *lines* (which were produced with ``keepends=True``)."""
    return sum(len(line) for line in lines[: lineno - 1]) + col_offset


def _restore_from_bak(path):
    """Restore *path* from its plain ``<path>.bak`` if that backup exists.

    Used on a verify/write failure so the live file is left equal to its
    pre-write content. Silent if no backup exists or the restore itself fails
    (the live file is then left as-is, which for the in-memory verify-failure
    path is the original untouched content)."""
    bak = path + ".bak"
    if os.path.exists(bak):
        try:
            shutil.copy2(bak, path)
        except OSError:
            pass


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

    # 0. Read the existing __init__.py and seed new_hashes from the existing
    #    KNOWN_HASHES so that bundled pickles we do NOT re-emit this run
    #    (e.g. model_embedding.pkl, structural_scaler.pkl when absent from
    #    source) retain their authoritative digest instead of being dropped.
    try:
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        print(f"ERROR: could not read {init_path}: {exc}")
        sys.exit(1)

    new_hashes = dict(_parse_known_hashes(content))

    # 1. Copy model files (required + conditionally-required + optional).
    #    Files copied this run OVERRIDE their preserved digest below.

    # If the model was trained with char-level features, the char vectorizer
    # is required for correct inference (15,029 dims vs 10,029 without it).
    char_vec_src = os.path.join(source_dir, CHAR_VECTORIZER)
    model_src = os.path.join(source_dir, "model.pkl")
    char_features_required = (
        os.path.exists(char_vec_src) and os.path.exists(model_src)
    )

    all_files = MODEL_FILES[:]
    if char_features_required:
        all_files.append(CHAR_VECTORIZER)
    all_files += [
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
                # Even when the .pkl is unchanged, refresh the sidecar: the
                # shipped one may be stale (the F-AR6 bug this closes).
                _write_sidecar(dst, src_digest)
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
        _write_sidecar(dst, digest)
        print(f"  Copied {fname}  sha256={digest[:16]}...")

    # 2. Update KNOWN_HASHES in __init__.py (content already read at step 0).
    # AST-locate the KNOWN_HASHES assignment and rewrite its value node,
    # verifying the result parses (ast.parse) and round-trips (ast.literal_eval).
    # This replaces the old brace-fragile re.sub, which truncated at the first
    # inner '}' and could ship invalid Python with no parse-verify.
    #
    # Determine intent first without writing, so the unchanged / absent paths
    # never touch the file or create a spurious backup.
    updated, status = _rewrite_known_hashes(content, new_hashes)

    if status == "absent":
        print("WARNING: KNOWN_HASHES block not found in __init__.py")
    elif status == "ok" and updated == content:
        print("  KNOWN_HASHES unchanged (hashes match)")
    elif status == "ok":
        # Back up __init__.py BEFORE mutating it (G4) so any failure — a bad
        # rewrite or a write error — is recoverable from the .bak. The rewrite
        # was already verified-valid above, but we keep the backup/restore
        # invariant so a corrupt file can never ship unrecoverably.
        _backup_file(init_path)
        try:
            with open(init_path, "w", encoding="utf-8") as f:
                f.write(updated)
        except OSError as exc:
            _restore_from_bak(init_path)
            print(f"ERROR: could not write {init_path}: {exc}")
            sys.exit(1)
        print(f"  Updated KNOWN_HASHES in {init_path}")
    else:
        # status == "error": _rewrite_known_hashes already printed the reason
        # (input did not parse, rewrite would not parse, or round-trip
        # mismatch). Fail closed: back up the original, restore it (a no-op for
        # the in-memory failure since nothing was written yet, but it leaves a
        # .bak whose content equals the untouched live file), and exit non-zero
        # WITHOUT writing the corrupt rewrite.
        _backup_file(init_path)
        _restore_from_bak(init_path)
        print("ERROR: KNOWN_HASHES rewrite failed verification; file unchanged.")
        sys.exit(1)

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
    all_rollback_files = MODEL_FILES + OPTIONAL_MODEL_FILES + [CHAR_VECTORIZER]
    for fname in all_rollback_files:
        live = os.path.join(dest_dir, fname)
        bak = live + ".bak"

        if not os.path.exists(bak):
            if fname in OPTIONAL_MODEL_FILES or fname == CHAR_VECTORIZER:
                continue  # Optional / conditionally-required files may not have backups
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
