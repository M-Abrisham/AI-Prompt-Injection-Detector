# src/na0s/replication_similarity.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.ml.replication_similarity"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.replication_similarity is deprecated; use na0s.ml.replication_similarity instead",
    DeprecationWarning,
    stacklevel=2,
)

# Import the canonical module and alias it into sys.modules at this path,
# so that `import na0s.replication_similarity` yields the real module object.
# This preserves module-level variable mutations in tests.
_canonical_name = "na0s.ml.replication_similarity"
if _canonical_name in _sys.modules:
    # Force reload so test-time sys.modules patches (e.g. fake torch) take effect
    _canonical = _importlib.reload(_sys.modules[_canonical_name])
else:
    _canonical = _importlib.import_module(_canonical_name)
_sys.modules[__name__] = _canonical
