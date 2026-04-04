"""Dual-direction scanner — combined input + output scanning.

Orchestrates both the OutputScanner (detecting injection success signals in
LLM output) and the PropagationScanner (detecting injection payloads in LLM
output targeting downstream systems), then cross-references the results to
determine if an input injection attempt actually succeeded.
"""

from __future__ import annotations

import threading
from typing import Optional

from na0s.output_scanner import OutputScanner, OutputScanResult
from na0s.propagation_scanner import PropagationScanner
from na0s.replication_similarity import replication_similarity


# ---------------------------------------------------------------------------
# DualDirectionScanner
# ---------------------------------------------------------------------------

class DualDirectionScanner:
    """Combined input/output scanning with cross-reference analysis.

    Parameters
    ----------
    output_scanner : OutputScanner or None
        Scanner for detecting injection success signals in output.
        Created with default settings if *None*.
    propagation_scanner : PropagationScanner or None
        Scanner for detecting injection payloads in output.
        Created with default settings if *None*.
    """

    def __init__(
        self,
        output_scanner: Optional[OutputScanner] = None,
        propagation_scanner: Optional[PropagationScanner] = None,
    ) -> None:
        self.output_scanner = output_scanner or OutputScanner()
        self.propagation_scanner = propagation_scanner or PropagationScanner()
        self._lock = threading.Lock()

    def scan(
        self,
        input_text: str,
        output_text: str,
        system_prompt: str = "",
    ) -> dict:
        """Run both output-side scanners and cross-reference results.

        Parameters
        ----------
        input_text : str
            The original user input (used for cross-referencing).
        output_text : str
            The LLM-generated output to scan.
        system_prompt : str
            Optional system prompt (used for leak detection).

        Returns
        -------
        dict
            ``output_scan``    – OutputScanResult as dict.
            ``propagation_scan`` – PropagationScanner result dict.
            ``cross_reference``  – Cross-reference analysis dict.
            ``overall_risk``     – Combined risk score float.
            ``is_suspicious``    – True if any scanner flagged the output.
        """
        with self._lock:
            return self._scan_impl(input_text, output_text, system_prompt)

    def _scan_impl(
        self, input_text: str, output_text: str, system_prompt: str
    ) -> dict:
        # 1. Run output scanner (detects injection success signals)
        output_result: OutputScanResult = self.output_scanner.scan(
            output_text=output_text,
            original_prompt=input_text,
            system_prompt=system_prompt or None,
        )

        # 2. Run propagation scanner (detects injection payloads in output)
        propagation_result: dict = self.propagation_scanner.scan(output_text)

        # 3. Cross-reference input and output
        cross_ref = self.cross_reference(
            input_result={"input_text": input_text},
            output_result={
                "output_scan": output_result.to_dict(),
                "propagation_scan": propagation_result,
                "output_text": output_text,
            },
        )

        # 4. Compute overall risk
        output_risk = output_result.risk_score
        propagation_risk = propagation_result.get("risk_score", 0.0)
        cross_ref_risk = cross_ref.get("cross_ref_score", 0.0)

        overall_risk = min(1.0, max(output_risk, propagation_risk, cross_ref_risk))

        is_suspicious = (
            output_result.is_suspicious
            or propagation_result.get("is_propagation_risk", False)
            or cross_ref.get("injection_succeeded", False)
        )

        return {
            "output_scan": output_result.to_dict(),
            "propagation_scan": propagation_result,
            "cross_reference": cross_ref,
            "overall_risk": round(overall_risk, 4),
            "is_suspicious": is_suspicious,
        }

    @staticmethod
    def cross_reference(input_result: dict, output_result: dict) -> dict:
        """Check if an input injection attempt succeeded in the output.

        Parameters
        ----------
        input_result : dict
            Must contain ``"input_text"`` key.
        output_result : dict
            Must contain ``"output_scan"`` and ``"propagation_scan"`` keys.

        Returns
        -------
        dict
            ``injection_succeeded`` – True if evidence suggests the
                injection was successful.
            ``cross_ref_score``     – float risk score from cross-referencing.
            ``evidence``            – list of evidence strings.
        """
        evidence: list = []
        cross_ref_score = 0.0

        output_scan = output_result.get("output_scan", {})
        propagation_scan = output_result.get("propagation_scan", {})
        output_text = output_result.get("output_text", "")
        input_text = input_result.get("input_text", "")

        # Check 1: Output scanner detected suspicious content
        if output_scan.get("is_suspicious", False):
            flags = output_scan.get("flags", [])
            for flag in flags:
                if not isinstance(flag, str):
                    continue
                if "Role break" in flag or "Compliance" in flag:
                    evidence.append(f"Output shows injection compliance: {flag}")
                    cross_ref_score = max(cross_ref_score, 0.7)
                elif "System prompt leak" in flag:
                    evidence.append(f"Output leaks system prompt: {flag}")
                    cross_ref_score = max(cross_ref_score, 0.8)
                elif "Secret pattern" in flag:
                    evidence.append(f"Output contains secrets: {flag}")
                    cross_ref_score = max(cross_ref_score, 0.9)

        # Check 2: Propagation scanner detected payload in output
        if propagation_scan.get("is_propagation_risk", False):
            evidence.append(
                "Output contains injection payload targeting downstream LLMs"
            )
            cross_ref_score = max(cross_ref_score, 0.8)

            # Worm patterns are especially dangerous
            worm = propagation_scan.get("worm_analysis", {})
            if worm.get("is_worm", False):
                evidence.append(
                    f"Self-replicating worm pattern detected: "
                    f"{', '.join(worm.get('matched_patterns', []))}"
                )
                cross_ref_score = max(cross_ref_score, 0.95)

        # Check 3: Input-output replication similarity (BLEU/ROUGE-L)
        if input_text and output_text:
            sim = replication_similarity(output_text, input_text)
            combined = sim["combined"]
            if combined >= 0.7:
                evidence.append(
                    "High input-output replication similarity "
                    f"(combined={combined:.2f}, BLEU={sim['bleu']:.2f}, "
                    f"ROUGE-L={sim['rouge_l']:.2f})"
                )
                cross_ref_score = max(cross_ref_score, 0.9)
            elif combined >= 0.55:
                evidence.append(
                    "Moderate replication similarity "
                    f"(combined={combined:.2f}, BLEU={sim['bleu']:.2f}, "
                    f"ROUGE-L={sim['rouge_l']:.2f})"
                )
                cross_ref_score = max(cross_ref_score, 0.75)

        # Check 4: Both scanners flagged — very high confidence
        both_flagged = (
            output_scan.get("is_suspicious", False)
            and propagation_scan.get("is_propagation_risk", False)
        )
        if both_flagged:
            evidence.append(
                "Both output and propagation scanners flagged — "
                "high-confidence injection success"
            )
            cross_ref_score = min(1.0, cross_ref_score + 0.1)

        injection_succeeded = cross_ref_score >= 0.5

        return {
            "injection_succeeded": injection_succeeded,
            "cross_ref_score": round(cross_ref_score, 4),
            "evidence": evidence,
        }
