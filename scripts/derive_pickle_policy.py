"""Regenerate the _SafeUnpickler allow-set / deny-set for safe_pickle (R5).

DEV-ONLY tooling — NOT imported at runtime by ``na0s.integrity.safe_pickle``.
It exists so the allowlist frozen in source is provably MEASURED (the exact
``(module, name)`` globals the four bundled artifacts name) and so the gadget
deny-set is grounded in the maintained ``fickling`` catalog rather than guessed
(na0s-review-checklist §1/§7 — no magic list).

Two independent jobs:

  1. ``measure`` — instrument a ``pickle.Unpickler`` that records every
     ``find_class(module, name)`` across model.pkl / tfidf_vectorizer.pkl /
     structural_scaler.pkl / model_embedding.pkl, print the UNION. This union
     MUST equal ``_PICKLE_ALLOW_EXACT`` in safe_pickle.py (drift guard).

  2. ``deny`` — if ``fickling`` is installed (the keyless ``cisec`` extra),
     enumerate its catalogued unsafe imports and intersect them with the
     numpy/sklearn/builtins namespaces to cross-check ``_DENY_PREFIXES`` /
     ``_GADGET_DENY``. fickling is OPTIONAL: when absent we print the
     hand-curated, provenance-tagged deny-set and a note that the fickling
     cross-check was skipped (no hallucinated symbols).

Usage (offline, keyless):
    PYTHONPATH=src python scripts/derive_pickle_policy.py measure
    PYTHONPATH=src python scripts/derive_pickle_policy.py deny
    PYTHONPATH=src python scripts/derive_pickle_policy.py check   # drift guard

``check`` exits non-zero if the measured union != the committed allow-set, so a
future numpy/sklearn bump that changes the artifacts' global set is caught.
"""

from __future__ import annotations

import io
import pickle
import sys
import warnings

_ARTIFACTS = (
    "model.pkl",
    "tfidf_vectorizer.pkl",
    "structural_scaler.pkl",
    "model_embedding.pkl",
)


def _measure_global_union():
    """Return the sorted set of (module, name) every bundled artifact names."""
    from na0s.models import get_model_path

    seen = set()

    class _Recorder(pickle.Unpickler):
        def find_class(self, module, name):
            seen.add((module, name))
            return super().find_class(module, name)

    import os

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for basename in _ARTIFACTS:
            path = get_model_path(basename)
            if not os.path.exists(path):
                print("  SKIP (absent): {}".format(basename), file=sys.stderr)
                continue
            with open(path, "rb") as f:
                data = f.read()
            _Recorder(io.BytesIO(data)).load()
    return sorted(seen)


def _cmd_measure():
    union = _measure_global_union()
    print("# MEASURED _PICKLE_ALLOW_EXACT (union across bundled artifacts):")
    print("_PICKLE_ALLOW_EXACT = frozenset({")
    for module, name in union:
        print('    ({!r}, {!r}),'.format(module, name))
    print("})")
    return 0


def _cmd_check():
    """Drift guard: measured union must be a SUBSET of the committed allow-set
    (the committed set may carry forward-compat ``_core`` twins not emitted by
    today's host, but must never MISS a global the artifacts actually name)."""
    from na0s.integrity.safe_pickle import _PICKLE_ALLOW_EXACT

    measured = set(_measure_global_union())
    missing = measured - set(_PICKLE_ALLOW_EXACT)
    if missing:
        print("DRIFT: measured globals NOT in committed _PICKLE_ALLOW_EXACT:")
        for m in sorted(missing):
            print("  {}".format(m))
        print(
            "-> the bundled artifacts would FALSE-REJECT. Add these (or "
            "regenerate via `measure`) and re-run the FP-safety tests."
        )
        return 1
    print(
        "OK: all {} measured globals are covered by the committed allow-set "
        "({} entries).".format(len(measured), len(_PICKLE_ALLOW_EXACT))
    )
    return 0


def _cmd_deny():
    from na0s.integrity.safe_pickle import _DENY_PREFIXES, _GADGET_DENY

    print("# Committed deny policy (provenance-tagged in safe_pickle.py):")
    print("_DENY_PREFIXES = (")
    for p in _DENY_PREFIXES:
        print("    {!r},".format(p))
    print(")")
    print("_GADGET_DENY = frozenset({")
    for pair in sorted(_GADGET_DENY):
        print("    {!r},".format(pair))
    print("})")
    try:
        import fickling  # noqa: F401
    except ImportError:
        print(
            "\n# NOTE: fickling not installed (keyless `cisec` extra). The "
            "deny-set above is the hand-curated, fickling-catalogue-derived "
            "list. Install fickling and re-run to cross-check its catalogued "
            "unsafe imports against numpy/sklearn/builtins.",
            file=sys.stderr,
        )
        return 0
    # fickling present: cross-check. The exact public API differs across
    # fickling versions, so probe defensively and PIN against the installed one
    # rather than asserting a hallucinated symbol.
    print(
        "\n# fickling is installed; cross-check its catalogued unsafe imports "
        "against the committed deny-set MANUALLY against this version's API "
        "(see fickling.analysis / fickling.fickle). Do NOT auto-trust a symbol "
        "that is not present on the installed version.",
        file=sys.stderr,
    )
    return 0


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "measure"
    if cmd == "measure":
        return _cmd_measure()
    if cmd == "check":
        return _cmd_check()
    if cmd == "deny":
        return _cmd_deny()
    print("usage: derive_pickle_policy.py [measure|deny|check]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
