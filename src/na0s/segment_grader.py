"""Segment-level output grading — split output into paragraphs and grade each.

Each paragraph is independently scanned for injection evidence. If ANY segment
is flagged, the entire output is marked as compromised. This catches cases
where a long benign response contains a single injected paragraph.

Gated by ``NA0S_SEGMENT_GRADING=1`` env var (default: disabled).
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from na0s.output_scanner import OutputScanner


class SegmentGrader:
    """Grade LLM output at the paragraph level."""

    def __init__(self, scanner: Optional[OutputScanner] = None) -> None:
        self.scanner = scanner or OutputScanner()

    @staticmethod
    def is_enabled() -> bool:
        """Return True when ``NA0S_SEGMENT_GRADING`` is set to a truthy value."""
        return os.environ.get("NA0S_SEGMENT_GRADING", "").strip() in ("1", "true", "yes")

    def grade(
        self,
        output_text: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, object]:
        """Split *output_text* into paragraphs and scan each independently.

        Returns
        -------
        dict
            ``is_compromised``, ``overall_risk``, ``segments``, ``flags``.
        """
        if not output_text or not output_text.strip():
            return {
                "is_compromised": False,
                "overall_risk": 0.0,
                "segments": [],
                "flags": [],
            }

        # Split on double newlines (paragraph boundaries)
        raw_segments = re.split(r"\n\n+", output_text.strip())
        segments = [s.strip() for s in raw_segments if s.strip()]

        if not segments:
            return {
                "is_compromised": False,
                "overall_risk": 0.0,
                "segments": [],
                "flags": [],
            }

        segment_results: List[Dict[str, object]] = []
        all_flags: List[str] = []

        for seg_text in segments:
            result = self.scanner.scan(
                output_text=seg_text,
                system_prompt=system_prompt,
            )
            seg_dict = {
                "text": seg_text[:100],
                "risk_score": result.risk_score,
                "flags": list(result.flags),
                "is_suspicious": result.is_suspicious,
            }
            segment_results.append(seg_dict)

            if result.is_suspicious:
                for flag in result.flags:
                    prefixed = f"segment: {flag}"
                    if prefixed not in all_flags:
                        all_flags.append(prefixed)

        overall_risk = max(s["risk_score"] for s in segment_results)
        is_compromised = any(s["is_suspicious"] for s in segment_results)

        return {
            "is_compromised": is_compromised,
            "overall_risk": overall_risk,
            "segments": segment_results,
            "flags": all_flags,
        }
