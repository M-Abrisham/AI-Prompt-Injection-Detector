"""Layer 9 — LLM output scanning and post-generation validation.

Runs after the LLM produces a response.  Detects injections that slipped
past the input pipeline: leaked secrets, role-break phrases, system-prompt
extraction, PII, markdown/HTML injection, data-exfiltration URLs, and
paragraph-level drift.

Public API:
    OutputScanner, OutputScanResult  -- secret / PII / injection scanner
    StreamingOutputScanner           -- chunk-by-chunk SSE scanner
    PropagationScanner               -- re-runs input classifier on output
                                        to catch worm-style propagation
    DualDirectionScanner             -- composes input + output + propagation
    SegmentGrader                    -- grade LLM output paragraph-by-
                                        paragraph via OutputScanner
    RAGAttributionChecker            -- verify LLM output is grounded in
                                        retrieved context
    verify_attribution               -- convenience wrapper for the above
"""

from __future__ import annotations

from .scanner import OutputScanner, OutputScanResult
from .streaming import StreamingOutputScanner
from .propagation import PropagationScanner
from .dual import DualDirectionScanner
from .segment_grader import SegmentGrader
from .attribution import RAGAttributionChecker, verify_attribution

__all__ = [
    "DualDirectionScanner",
    "OutputScanResult",
    "OutputScanner",
    "PropagationScanner",
    "RAGAttributionChecker",
    "SegmentGrader",
    "StreamingOutputScanner",
    "verify_attribution",
]
