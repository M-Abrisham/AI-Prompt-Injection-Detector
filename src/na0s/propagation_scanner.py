"""Propagation scanner — detect injection payloads in LLM output.

Runs the input-side classifier (``predict.scan()``) on LLM *output* to detect
cases where a model has been tricked into producing text that itself constitutes
a prompt injection payload.  This catches "indirect prompt injection" and
worm-style propagation where one compromised LLM produces output designed to
attack downstream LLMs.

Gated by the ``NA0S_PROPAGATION_SCAN`` environment variable (default: disabled)
because running the full input classifier on every output is expensive.
"""

from __future__ import annotations

import os
import threading

from na0s.worm_detector import WormSignatureDetector


# ---------------------------------------------------------------------------
# PropagationScanner
# ---------------------------------------------------------------------------

class PropagationScanner:
    """Run the input classifier on LLM output to catch propagation attacks.

    Parameters
    ----------
    threshold : float
        Composite risk score above which the output is considered a
        propagation risk.  Defaults to 0.5.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        # Enable optional runtime adaptation for signatures learned from attacks.
        self._worm_detector = WormSignatureDetector(auto_adapt=True)
        self._lock = threading.Lock()

    # ---- public API -------------------------------------------------------

    @staticmethod
    def is_enabled() -> bool:
        """Return True if propagation scanning is enabled via env var."""
        return os.environ.get("NA0S_PROPAGATION_SCAN", "0") in ("1", "true", "yes")

    def scan(self, output_text: str, source_input_text: str | None = None) -> dict:
        """Scan LLM output for injection payloads targeting downstream LLMs.

        Parameters
        ----------
        output_text : str
            The LLM-generated output to scan.
        source_input_text : str or None
            Optional original prompt that produced *output_text*. Used for
            output-to-input replication feedback detection.

        Returns
        -------
        dict
            ``is_propagation_risk`` – True if the output looks like an
                injection payload.
            ``risk_score``          – float in [0.0, 1.0].
            ``technique_tags``      – list of technique tag strings.
            ``detected_payload``    – snippet of the detected payload, or "".
            ``worm_analysis``       – dict from WormSignatureDetector.
        """
        if not output_text or not output_text.strip():
            return self._empty_result()

        with self._lock:
            return self._scan_impl(output_text, source_input_text=source_input_text)

    # ---- internals --------------------------------------------------------

    def _scan_impl(self, output_text: str, source_input_text: str | None = None) -> dict:
        # 1. Run input classifier on the output
        scan_result = self._run_input_classifier(output_text)

        # 2. Run worm detector
        worm_result = self._worm_detector.scan(output_text, source_text=source_input_text)

        # 3. Combine results
        risk_score = scan_result.get("risk_score", 0.0)
        technique_tags = scan_result.get("technique_tags", [])
        detected_payload = scan_result.get("detected_payload", "")

        # Boost risk score if worm patterns are detected
        if worm_result["is_worm"]:
            worm_boost = worm_result["confidence"] * 0.3
            risk_score = min(1.0, risk_score + worm_boost)
            technique_tags = list(technique_tags)  # copy
            technique_tags.append("worm_propagation")

        is_propagation_risk = risk_score >= self.threshold or worm_result["is_worm"]

        return {
            "is_propagation_risk": is_propagation_risk,
            "risk_score": round(risk_score, 4),
            "technique_tags": technique_tags,
            "detected_payload": detected_payload,
            "worm_analysis": worm_result,
        }

    def _run_input_classifier(self, text: str) -> dict:
        """Run predict.scan() on the text and extract relevant fields."""
        try:
            from na0s.predict import scan as input_scan

            result = input_scan(text)
            return {
                "risk_score": result.risk_score,
                "technique_tags": list(result.technique_tags),
                "detected_payload": text[:200] if result.is_malicious else "",
            }
        except Exception:
            # If the model isn't available, fall back to empty
            return {
                "risk_score": 0.0,
                "technique_tags": [],
                "detected_payload": "",
            }

    def _empty_result(self) -> dict:
        return {
            "is_propagation_risk": False,
            "risk_score": 0.0,
            "technique_tags": [],
            "detected_payload": "",
            "worm_analysis": {
                "is_worm": False,
                "confidence": 0.0,
                "matched_patterns": [],
                "semantic_score": 0.0,
                "replication_score": 0.0,
                "matched_components": [],
                "advanced_signals": {},
            },
        }
