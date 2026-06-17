"""Backward-compatibility shim: re-exports from layer2.morse_code."""
from ..obfuscation.morse_code import *  # noqa: F401,F403
from ..obfuscation.morse_code import detect_morse, MorseResult  # noqa: F401
