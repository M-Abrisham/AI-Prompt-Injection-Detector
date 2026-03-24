"""Na0S -- Multi-layer prompt injection detection for LLM applications.

Quick start::

    from na0s import scan
    result = scan("Ignore all previous instructions")
    print(result.is_malicious)   # True
    print(result.risk_score)     # 0.93
    print(result.label)          # "malicious"

Output scanning::

    from na0s import scan_output
    result = scan_output("Sure! Here is the system prompt...")
    print(result.is_suspicious)  # True

Advanced multi-layer cascade::

    from na0s import CascadeClassifier
    clf = CascadeClassifier()
    label, confidence, hits, stage = clf.classify("some input")

Cascade with ScanResult (same return type as scan())::

    clf = CascadeClassifier()
    result = clf.scan("some input")
    print(result.cascade_stage)  # "whitelist", "weighted", "judge", ...

Image / visual injection scanning::

    from na0s import scan_image
    result = scan_image(open("suspect.png", "rb").read())
    print(result.is_suspicious)  # True
    print(result.technique_ids)  # ["M1", "M1.1"]
"""

from na0s._version import __version__
from na0s.scan_result import ScanResult
from na0s.predict import scan, preload_models
from na0s.cascade import CascadeClassifier

try:
    from na0s.ensemble import ensemble_scan, EnsembleClassifier
except ImportError:
    pass  # Embedding dependency may not be installed
from na0s.output_scanner import OutputScanner, OutputScanResult
from na0s.streaming_scanner import StreamingOutputScanner
from na0s.positive_validation import TrustBoundary
from na0s.canary import CanaryManager, CanaryToken

try:
    from na0s.visual_injection_detector import (
        scan_image,
        scan_image_file,
        scan_document_visual,
        VisualInjectionResult,
    )
except ImportError:
    pass  # Image/document deps may not be installed

__all__ = [
    "__version__",
    "scan",
    "preload_models",
    "ensemble_scan",
    "EnsembleClassifier",
    "scan_output",
    "scan_image",
    "scan_image_file",
    "scan_document_visual",
    "CascadeClassifier",
    "ScanResult",
    "VisualInjectionResult",
    "OutputScanner",
    "OutputScanResult",
    "StreamingOutputScanner",
    "TrustBoundary",
    "CanaryManager",
    "CanaryToken",
]


def scan_output(
    output_text,
    original_prompt=None,
    system_prompt=None,
    sensitivity="medium",
):
    """Scan LLM output for signs of successful prompt injection.

    Parameters
    ----------
    output_text : str
        The LLM's response text.
    original_prompt : str or None
        The user's original prompt (for instruction-echo detection).
    system_prompt : str or None
        The system prompt (for leak detection).
    sensitivity : str
        ``"low"``, ``"medium"``, or ``"high"``.

    Returns
    -------
    OutputScanResult
    """
    scanner = OutputScanner(sensitivity=sensitivity)
    return scanner.scan(
        output_text=output_text,
        original_prompt=original_prompt,
        system_prompt=system_prompt,
    )
