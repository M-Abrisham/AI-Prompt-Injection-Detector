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

import logging
import os

from na0s.worm.detector import WormSignatureDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORM_BOOST_FACTOR = 0.3
"""How much worm confidence contributes to overall risk score."""

PAYLOAD_SNIPPET_LEN = 200
"""Max characters captured for the detected_payload field."""


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
        # auto_adapt=False: attacker-controlled output must NOT train the
        # detector — that's a poisoning vector.  Auto-adapt should only run
        # on *confirmed* malicious input after human review.
        # reconstruction_window=1 (stateless, _history_limit==0): consistent
        # with the input-path get_worm_detector() singleton (WD-3) so a prior
        # output-scan turn cannot poison a later benign output-scan via the
        # cross-turn reconstruction buffer.
        self._worm_detector = WormSignatureDetector(auto_adapt=False, reconstruction_window=1)

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

        # No global lock needed: WormSignatureDetector handles its own
        # thread safety, and _run_input_classifier is stateless.
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
            worm_boost = worm_result["confidence"] * WORM_BOOST_FACTOR
            risk_score = min(1.0, risk_score + worm_boost)
            technique_tags.append("worm_propagation")
            # Surface the taxonomy code so the worm hit is traceable to IM1.6.
            for _tid in worm_result.get("technique_ids", []):
                if _tid not in technique_tags:
                    technique_tags.append(_tid)

        # Single decision criterion: boosted risk_score only.
        # No separate `or worm_result["is_worm"]` — that double-counted
        # the worm signal and made the threshold meaningless.
        is_propagation_risk = risk_score >= self.threshold

        return {
            "is_propagation_risk": is_propagation_risk,
            "risk_score": round(risk_score, 4),
            "technique_tags": technique_tags,
            "detected_payload": detected_payload,
            "worm_analysis": worm_result,
        }

    def _run_input_classifier(self, text: str) -> dict:
        """Run predict.scan() on the text and extract relevant fields.

        Fail-closed: if the classifier errors, return high risk with an
        error flag so the caller doesn't silently pass through attacks.
        """
        try:
            from na0s.predict import scan as input_scan

            result = input_scan(text)

            # Use rule_hits for payload snippet if available, otherwise
            # fall back to text prefix.
            if result.is_malicious and result.rule_hits:
                detected_payload = str(result.rule_hits[0])[:PAYLOAD_SNIPPET_LEN]
            elif result.is_malicious:
                detected_payload = text[:PAYLOAD_SNIPPET_LEN]
            else:
                detected_payload = ""

            return {
                "risk_score": result.risk_score,
                "technique_tags": list(result.technique_tags),
                "detected_payload": detected_payload,
            }
        except Exception:
            # Fail-closed: classifier unavailable = assume risk, not safety.
            logger.exception("Input classifier failed on propagation scan")
            return {
                "risk_score": 1.0,
                "technique_tags": ["classifier_error"],
                "detected_payload": "",
            }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "is_propagation_risk": False,
            "risk_score": 0.0,
            "technique_tags": [],
            "detected_payload": "",
            "worm_analysis": WormSignatureDetector.empty_result(),
        }
