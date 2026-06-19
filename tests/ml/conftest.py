"""Shared fixtures + cross-test isolation guard for the ML (Layer 4/5) tests.

Several modules in this directory mock the optional ``sentence_transformers``
dependency at *import time* so the modules under test can be exercised without
the heavy package installed::

    if "sentence_transformers" not in sys.modules:
        sys.modules["sentence_transformers"] = MagicMock()

That guard only checks whether the name is *currently* in ``sys.modules`` -- it
does NOT check whether the real package is installed.  In CI (where
``sentence-transformers`` IS installed via ``.[dev]`` but has not been imported
yet at collection time) the guard passes and a ``MagicMock`` is injected,
**shadowing the real package for the rest of the process**.

The damage is twofold and order-dependent:

1.  The first module to ``from sentence_transformers import SentenceTransformer``
    *after* the mock lands -- ``na0s.ml.embedding_classifier`` -- binds its
    module-level ``SentenceTransformer`` name to the ``MagicMock``.  Its
    ``EmbeddingClassifier`` then produces all-``0.0`` embedding scores, so every
    downstream detection test that relies on the Layer-5 semantic boost (the
    ``tests/test_scan_*`` / ``tests/test_false_positives.py`` borderline cases)
    silently loses ~0.2 of composite risk and drops below the 0.55 threshold --
    exactly equivalent to ``NA0S_EMBEDDING_ENABLED=0``.
2.  Script modules that read ``_HAS_SENTENCE_TRANSFORMERS`` at import
    (``benchmark_embeddings`` etc.) see the mock as "installed" and behave
    incorrectly.

This conftest closes both holes WITHOUT weakening the mocking the ML tests rely
on:

* **Pre-warm** the real embedding modules at conftest import time -- which runs
  *before* the test modules in this directory are collected -- so their
  ``SentenceTransformer`` name is bound to the genuine class and can never be
  poisoned by a later mock injection.
* **Restore** ``sys.modules["sentence_transformers"]`` to its original value at
  the end of the session and reset the embedding-classifier singleton, so no
  ``MagicMock`` survives into the root-level detection tests that collect after
  ``tests/ml/``.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Snapshot the original sentence_transformers entry BEFORE any test module in
# this directory injects its MagicMock.  conftest.py is imported by pytest
# ahead of collecting the sibling test files, so this runs first.
# ---------------------------------------------------------------------------
_ORIGINAL_ST = sys.modules.get("sentence_transformers", "__absent__")

# Pre-warm the embedding modules so their module-level
# ``from sentence_transformers import SentenceTransformer`` binds the REAL class
# (or genuinely fails to import when the package is absent) -- never a MagicMock
# that a later test module slipped into sys.modules.
for _mod in ("na0s.ml.embedding_classifier", "na0s.ml.predict_embedding"):
    try:  # pragma: no cover - import side effects only
        importlib.import_module(_mod)
    except Exception:
        # Missing optional deps are fine; the point is to bind whatever the
        # real environment provides before any mock can shadow it.
        pass


def _restore_sentence_transformers() -> None:
    """Undo any MagicMock shadowing of ``sentence_transformers``.

    Returns ``sys.modules`` to its pre-collection state and rebinds the
    embedding modules (whose module-level ``SentenceTransformer`` name may have
    been bound to a mock if they were first imported under one) to the real
    class, then drops the cached singleton so the next caller rebuilds cleanly.
    """
    from unittest.mock import MagicMock

    current = sys.modules.get("sentence_transformers")
    leaked = isinstance(current, MagicMock)

    if not leaked:
        return

    # Restore the original module object (or remove the entry entirely if
    # sentence_transformers was genuinely absent before these tests ran).
    if _ORIGINAL_ST == "__absent__":
        sys.modules.pop("sentence_transformers", None)
    else:
        sys.modules["sentence_transformers"] = _ORIGINAL_ST

    # Rebind module-level SentenceTransformer references that may point at the
    # mock, and reset the embedding singleton so it is rebuilt against the
    # restored backend.
    for modname in ("na0s.ml.embedding_classifier", "na0s.ml.predict_embedding"):
        mod = sys.modules.get(modname)
        if mod is not None:
            try:  # pragma: no cover - reload side effects only
                importlib.reload(mod)
            except Exception:
                pass

    ec = sys.modules.get("na0s.ml.embedding_classifier")
    if ec is not None and hasattr(ec, "reset_singleton"):
        ec.reset_singleton()


@pytest.fixture(scope="package", autouse=True)
def _isolate_sentence_transformers_mock():
    """Package-scoped guard: strip any leaked sentence_transformers mock on exit.

    Autouse so it always arms; package-scoped so the cleanup runs once, when the
    last ``tests/ml`` test finishes — *before* the root-level detection suites
    (``tests/test_scan_*`` / ``tests/test_false_positives.py``) run — rather than
    at the very end of the session where it would be too late to help them.
    """
    yield
    _restore_sentence_transformers()
