#!/usr/bin/env python3
"""Drift check for docs/facts.yaml that tolerates volatile test counts.

The docs-drift CI gate regenerates ``docs/facts.yaml`` from source and used to
fail on ANY difference vs. the committed file (a blunt ``git diff --quiet``).
That made the gate hard-fail on every PR that merely adds or removes a test
file, because two facts are inherently volatile and grow on normal work:

  * ``test_files``        -- a glob count of ``tests/**/test_*.py``
  * ``test_count.count``  -- ``pytest --collect-only`` count

Pinning those exact integers into a committed file means any test-touching PR
drifts them, even though nothing a reviewer cares about changed (rules,
taxonomy, public API, constants). See ISS-08.

This checker keeps the gate strict for STABLE facts (any change there still
fails -- the author must re-run ``scripts/extract_facts.py``) but treats the
volatile count fields as a FLOOR: the regenerated value may be ``>=`` the
committed one without failing (tests only grow), while a DECREASE still fails
(it usually means tests were deleted or collection broke). All other fields,
including ``test_count.collection_errors`` and ``pytest_exit_code``, stay exact
-- a new collection error should surface, not be hidden.

Usage (CI, after ``extract_facts.py`` has regenerated the working-tree file)::

    python3 scripts/check_facts_drift.py        # exit 0 clean, 1 drift

Compares the committed ``HEAD:docs/facts.yaml`` against the regenerated
working-tree ``docs/facts.yaml``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTS_REL = "docs/facts.yaml"

# Leaf paths whose regenerated value may exceed the committed value without
# being treated as drift (tests grow). A *decrease* is still flagged.
FLOOR_PATHS: set[tuple] = {
    ("test_files",),
    ("test_count", "count"),
}

_MISSING = object()


def _flatten(node, prefix: tuple = ()) -> dict[tuple, object]:
    """Flatten a nested dict into {path_tuple: leaf_value}."""
    out: dict[tuple, object] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_flatten(v, prefix + (k,)))
    else:
        out[prefix] = node
    return out


def compare_facts(committed: dict, regenerated: dict) -> list[str]:
    """Return a list of human-readable drift reasons; empty means clean.

    Pure function over two parsed facts dicts so it is unit-testable without
    git or the filesystem.
    """
    reasons: list[str] = []
    cflat = _flatten(committed or {})
    rflat = _flatten(regenerated or {})
    for path in sorted(set(cflat) | set(rflat), key=lambda p: ".".join(map(str, p))):
        cv = cflat.get(path, _MISSING)
        rv = rflat.get(path, _MISSING)
        if cv == rv:
            continue
        dotted = ".".join(map(str, path))
        if path in FLOOR_PATHS:
            # Volatile count: growth is fine, a decrease (or non-numeric /
            # missing value) is real drift worth surfacing.
            if (
                isinstance(cv, (int, float))
                and isinstance(rv, (int, float))
                and not isinstance(cv, bool)
                and not isinstance(rv, bool)
                and rv >= cv
            ):
                continue
            reasons.append(
                f"{dotted}: regenerated {rv!r} is below committed {cv!r} "
                "(test count/files DECREASED -- tests removed or collection broke?)"
            )
        else:
            reasons.append(
                f"{dotted}: committed {cv!r} != regenerated {rv!r}"
            )
    return reasons


def _load(path_or_text: str, *, from_git: bool) -> dict:
    if from_git:
        text = subprocess.check_output(
            ["git", "show", f"HEAD:{path_or_text}"], cwd=str(REPO_ROOT), text=True
        )
    else:
        text = (REPO_ROOT / path_or_text).read_text()
    return yaml.safe_load(text) or {}


def main() -> int:
    committed = _load(FACTS_REL, from_git=True)
    regenerated = _load(FACTS_REL, from_git=False)
    reasons = compare_facts(committed, regenerated)

    if not reasons:
        print(
            "clean -- committed docs/facts.yaml matches source "
            "(volatile test counts allowed to grow within floor)."
        )
        return 0

    print("docs/facts.yaml drift detected in STABLE facts:")
    for r in reasons:
        print(f"  - {r}")
    print()
    print("Fix locally:")
    print("    python3 scripts/extract_facts.py")
    print("    git add docs/facts.yaml")
    print("    git commit --amend --no-edit  # or as a new commit")
    print()
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--", FACTS_REL], cwd=str(REPO_ROOT), text=True
        )
        if diff.strip():
            print("Diff:")
            print("```diff")
            print(diff, end="")
            print("```")
    except subprocess.CalledProcessError:
        pass
    return 1


if __name__ == "__main__":
    sys.exit(main())
