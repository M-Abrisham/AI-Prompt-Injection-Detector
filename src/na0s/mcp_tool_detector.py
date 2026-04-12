# src/na0s/mcp_tool_detector.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.detectors.mcp_tool"""
import warnings as _warnings
_warnings.warn("na0s.mcp_tool_detector is deprecated; use na0s.detectors.mcp_tool instead", DeprecationWarning, stacklevel=2)

from na0s.detectors import mcp_tool as _mod  # noqa: E402
from na0s.detectors.mcp_tool import *  # noqa: F401,F403,E402

globals().update({k: v for k, v in vars(_mod).items() if k.startswith("_") and not k.startswith("__")})
