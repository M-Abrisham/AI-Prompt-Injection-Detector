"""Unit tests for the docs-drift facts gate's floor logic (ISS-08).

``scripts/check_facts_drift.py`` replaced a blunt ``git diff --quiet`` so that
test-adding PRs (which only grow the volatile ``test_files`` / ``test_count``
fields) no longer hard-fail the docs-drift CI gate, while real drift in stable
facts (rules, taxonomy, constants, exports) still fails. These tests pin that
contract by calling the pure ``compare_facts`` function directly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_facts_drift.py"
_spec = importlib.util.spec_from_file_location("check_facts_drift", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
compare_facts = _mod.compare_facts


def _facts(test_files=303, test_count=9449, collection_errors=0, rule_total=120):
    """A minimal facts.yaml-shaped dict with the fields the gate inspects."""
    return {
        "rule_count": {"literal": 101, "total_ast": rule_total},
        "taxonomy": {"category_count": 30, "technique_count_total": 278},
        "test_count": {"count": test_count, "collection_errors": collection_errors,
                       "pytest_exit_code": 0},
        "test_files": test_files,
    }


def test_identical_is_clean():
    assert compare_facts(_facts(), _facts()) == []


def test_test_files_growth_is_clean():
    # A PR that adds a test file: 303 -> 304. Must NOT be drift.
    assert compare_facts(_facts(test_files=303), _facts(test_files=304)) == []


def test_test_count_growth_is_clean():
    assert compare_facts(_facts(test_count=9449), _facts(test_count=9510)) == []


def test_test_files_decrease_is_drift():
    reasons = compare_facts(_facts(test_files=304), _facts(test_files=303))
    assert reasons, "a decrease in test_files must be flagged"
    assert any("test_files" in r and "DECREASED" in r for r in reasons)


def test_test_count_decrease_is_drift():
    reasons = compare_facts(_facts(test_count=9449), _facts(test_count=9000))
    assert any("test_count.count" in r for r in reasons)


def test_stable_fact_change_is_drift():
    # A real source change (e.g. a new rule) must still fail the gate exactly.
    reasons = compare_facts(_facts(rule_total=120), _facts(rule_total=121))
    assert any("rule_count.total_ast" in r for r in reasons)


def test_collection_error_appearing_is_drift():
    # collection_errors is NOT a floor field: a new collection error must surface.
    reasons = compare_facts(
        _facts(collection_errors=0), _facts(collection_errors=1)
    )
    assert any("collection_errors" in r for r in reasons)


def test_taxonomy_change_is_drift():
    a = _facts()
    b = _facts()
    b["taxonomy"]["technique_count_total"] = 279
    reasons = compare_facts(a, b)
    assert any("technique_count_total" in r for r in reasons)
