"""Backward-compatibility shim: re-exports from layer2.ascii_art_detector."""
from ..obfuscation.ascii_art_detector import *  # noqa: F401,F403
from ..obfuscation.ascii_art_detector import detect_ascii_art, AsciiArtResult  # noqa: F401
