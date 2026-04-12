# src/na0s/embedding_classifier.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.ml.embedding_classifier"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.embedding_classifier is deprecated; use na0s.ml.embedding_classifier instead",
    DeprecationWarning,
    stacklevel=2,
)

# Import the canonical module and alias it into sys.modules at this path,
# so that `import na0s.embedding_classifier` yields the real module object.
# This preserves module-level variable mutations in tests.
_canonical_name = "na0s.ml.embedding_classifier"
if _canonical_name in _sys.modules:
    # Force reload so test-time sys.modules patches (e.g. fake torch) take effect
    _canonical = _importlib.reload(_sys.modules[_canonical_name])
else:
    _canonical = _importlib.import_module(_canonical_name)
_sys.modules[__name__] = _canonical
