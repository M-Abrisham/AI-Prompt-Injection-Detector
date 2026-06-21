"""Backward-compatibility shim: re-exports from layer2.numeric_decode."""
from ..obfuscation.numeric_decode import *  # noqa: F401,F403
from ..obfuscation.numeric_decode import (  # noqa: F401
    detect_numeric,
    detect_binary,
    detect_octal,
    detect_decimal,
    NumericDecodeResult,
)
