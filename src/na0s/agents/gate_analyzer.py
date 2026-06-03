"""Gate failure diagnostic agent.

Reads gate evaluation JSON outputs (canary, shadow, F14) and generates
human-friendly diagnostic summaries for iMessage alerts. Integrates Claude API
for intelligent root cause analysis and fix recommendations.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from na0s.agents.claude_gate_analyzer import ClaudeGateAnalyzer, GateCacheManager

logger = logging.getLogger(__name__)


class GateAnalyzer:
    """Analyzes gate evaluation results and generates diagnostics.

    Integrates Claude API for intelligent root cause analysis when gates fail.
    Claude analysis is optional and gracefully degraded if API key is unavailable.
    """

    def __init__(self, data_dir: str = "data", use_claude: bool = True):
        """Initialize gate analyzer with optional Claude integration.

        Args:
            data_dir: Directory containing gate result files
            use_claude: Whether to enable Claude API analysis
        """
        self.data_dir = Path(data_dir)
        self.canary_path = self.data_dir / "canary" / "canary_results.json"
        self.shadow_path = self.data_dir / ".." / "models" / "shadow_results.json"
        self.f14_path = self.data_dir / ".." / "models" / "f14_gate_results.json"

        # Initialize Claude analyzer and cache manager
        self.use_claude = use_claude
        if use_claude:
            self.cache_manager = GateCacheManager()
            self.claude_analyzer = ClaudeGateAnalyzer(cache_manager=self.cache_manager)
        else:
            self.cache_manager = None
            self.claude_analyzer = None

    def check_canary(self) -> Optional[Dict[str, Any]]:
        """Read and analyze canary evaluation results.

        Returns:
            Dict with keys: {passed, verdict, tpr, tnr, fpr, errors, summary}
        """
        if not self.canary_path.exists():
            logger.warning(f"Canary results not found: {self.canary_path}")
            return None

        try:
            with open(self.canary_path) as f:
                data = json.load(f)

            passed = data.get("passed", False)
            metrics = data.get("metrics", {})
            errors = data.get("errors", [])

            tpr = metrics.get("tpr", 0)
            tnr = metrics.get("tnr", 0)
            fpr = metrics.get("fpr", 0)

            summary = f"Canary: TPR {tpr:.1%}, TNR {tnr:.1%}, FPR {fpr:.1%}"
            if not passed:
                error_count = len(errors)
                summary += f" — {error_count} classification errors"

            return {
                "passed": passed,
                "verdict": "PASSED" if passed else "FAILED",
                "tpr": tpr,
                "tnr": tnr,
                "fpr": fpr,
                "error_count": len(errors),
                "errors": errors[:3],  # Top 3 errors
                "summary": summary,
            }
        except Exception as e:
            logger.error(f"Error reading canary results: {e}")
            return None

    def check_shadow(self) -> Optional[Dict[str, Any]]:
        """Read and analyze shadow evaluation results.

        Returns:
            Dict with keys: {passed, verdict, fpr_delta, recall_delta, failures, summary}
        """
        if not self.shadow_path.exists():
            logger.warning(f"Shadow results not found: {self.shadow_path}")
            return None

        try:
            with open(self.shadow_path) as f:
                data = json.load(f)

            verdict = data.get("verdict", "UNKNOWN")
            passed = verdict == "PASS"

            gates = data.get("gates", [])
            failures = data.get("failures", [])

            # Extract FPR and recall deltas from gates
            fpr_delta = None
            recall_delta = None
            for gate in gates:
                if "FPR" in gate.get("gate", ""):
                    fpr_delta = gate.get("actual", 0)
                if "Recall" in gate.get("gate", ""):
                    recall_delta = gate.get("actual", 0)

            summary = f"Shadow: {verdict}"
            if fpr_delta is not None:
                summary += f" — FPR Δ {fpr_delta:+.1%}"
            if recall_delta is not None:
                summary += f", Recall Δ {recall_delta:+.1%}"

            if failures:
                summary += f" — {len(failures)} gate(s) failed"

            return {
                "passed": passed,
                "verdict": verdict,
                "fpr_delta": fpr_delta,
                "recall_delta": recall_delta,
                "failure_count": len(failures),
                "failures": failures[:3],
                "summary": summary,
            }
        except Exception as e:
            logger.error(f"Error reading shadow results: {e}")
            return None

    def check_f14(self) -> Optional[Dict[str, Any]]:
        """Read and analyze F14 promotion gate results.

        Returns:
            Dict with keys: {passed, verdict, overall_tpr, regressions, summary}
        """
        if not self.f14_path.exists():
            logger.warning(f"F14 results not found: {self.f14_path}")
            return None

        try:
            with open(self.f14_path) as f:
                data = json.load(f)

            verdict = data.get("verdict", "UNKNOWN")
            passed = verdict == "PASS"

            overall = data.get("overall", {})
            overall_tpr = overall.get("tpr", 0)

            baseline_comp = data.get("baseline_comparison", {})
            regressions = baseline_comp.get("regressions", [])

            summary = f"F14: {verdict}"
            if overall_tpr is not None:
                summary += f" — Overall TPR {overall_tpr:.1%}"

            if regressions:
                summary += f" — {len(regressions)} category regress(ions)"
                for reg in regressions[:2]:
                    cat = reg.get("category", "?")
                    delta = reg.get("delta", 0)
                    summary += f" ({cat}: {delta:+.1%})"

            return {
                "passed": passed,
                "verdict": verdict,
                "overall_tpr": overall_tpr,
                "regression_count": len(regressions),
                "regressions": regressions[:5],
                "summary": summary,
            }
        except Exception as e:
            logger.error(f"Error reading F14 results: {e}")
            return None

    def analyze_gate_with_claude(
        self, gate_type: str, failure_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze a gate failure using Claude API for root cause analysis.

        Calls Claude API to generate intelligent root cause analysis and
        fix recommendations. Results are cached for offline review.

        Args:
            gate_type: Type of gate (canary, shadow, f14)
            failure_data: Gate failure data to analyze

        Returns:
            Analysis result with root_cause and fix_specificity, or None if disabled
        """
        if not self.use_claude or not self.claude_analyzer:
            return None

        return self.claude_analyzer.analyze_gate(gate_type, failure_data)

    def diagnose_failures(self) -> Dict[str, Any]:
        """Run all gate checks and compile failure report.

        Calls Claude API for root cause analysis when gates fail (if enabled).

        Returns:
            Dict with all gate results and overall summary for iMessage
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "canary": self.check_canary(),
            "shadow": self.check_shadow(),
            "f14": self.check_f14(),
            "claude_analysis": {},
        }

        # Determine which gate(s) failed
        failed_gates = []
        if results["canary"] and not results["canary"]["passed"]:
            failed_gates.append("Canary")
        if results["shadow"] and not results["shadow"]["passed"]:
            failed_gates.append("Shadow")
        if results["f14"] and not results["f14"]["passed"]:
            failed_gates.append("F14")

        # Call Claude API for each failed gate to get root cause analysis
        if self.use_claude and self.claude_analyzer:
            if results["canary"] and not results["canary"]["passed"]:
                canary_failure = {
                    "tpr": results["canary"].get("tpr"),
                    "tnr": results["canary"].get("tnr"),
                    "fpr": results["canary"].get("fpr"),
                    "error_count": results["canary"].get("error_count"),
                    "errors": results["canary"].get("errors", []),
                }
                analysis = self.analyze_gate_with_claude("canary", canary_failure)
                if analysis:
                    results["claude_analysis"]["canary"] = analysis

            if results["shadow"] and not results["shadow"]["passed"]:
                shadow_failure = {
                    "verdict": results["shadow"].get("verdict"),
                    "fpr_delta": results["shadow"].get("fpr_delta"),
                    "recall_delta": results["shadow"].get("recall_delta"),
                    "failure_count": results["shadow"].get("failure_count"),
                    "failures": results["shadow"].get("failures", []),
                }
                analysis = self.analyze_gate_with_claude("shadow", shadow_failure)
                if analysis:
                    results["claude_analysis"]["shadow"] = analysis

            if results["f14"] and not results["f14"]["passed"]:
                f14_failure = {
                    "verdict": results["f14"].get("verdict"),
                    "overall_tpr": results["f14"].get("overall_tpr"),
                    "regression_count": results["f14"].get("regression_count"),
                    "regressions": results["f14"].get("regressions", []),
                }
                analysis = self.analyze_gate_with_claude("f14", f14_failure)
                if analysis:
                    results["claude_analysis"]["f14"] = analysis

        if not failed_gates:
            results["overall_verdict"] = "ALL_PASSED"
            results["message"] = "✅ All gates passed. Ready for deployment approval."
        else:
            results["overall_verdict"] = "FAILED"
            gate_str = ", ".join(failed_gates)
            results["message"] = f"❌ {gate_str} gate(s) failed. Waiting for manual review."

        return results

    def write_report(self, report_dir: str = "data/approval_queue/failure_reports") -> Optional[str]:
        """Write failure report to disk if gates failed.

        Args:
            report_dir: Directory to write reports to

        Returns:
            Path to written report, or None if all gates passed
        """
        results = self.diagnose_failures()

        if results["overall_verdict"] == "ALL_PASSED":
            return None

        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"failure_{timestamp}.json"

        try:
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Wrote failure report to {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Error writing report: {e}")
            return None

    def format_message(self) -> str:
        """Format gate analysis results for iMessage.

        Includes Claude's root cause analysis and fix specificity when available.

        Returns:
            Human-readable message for user approval or retry decision
        """
        results = self.diagnose_failures()

        lines = []
        if results["canary"]:
            lines.append(f"Canary: {results['canary']['verdict']}")
            if not results["canary"]["passed"]:
                for err in results["canary"]["errors"]:
                    technique = err.get("technique", "?")
                    lines.append(f"  • {technique}: misclassified")
                # Include Claude analysis if available
                if "canary" in results.get("claude_analysis", {}):
                    analysis = results["claude_analysis"]["canary"]
                    if analysis.get("root_cause"):
                        lines.append(f"  Root Cause: {analysis['root_cause']}")
                    if analysis.get("fix_specificity"):
                        lines.append(f"  Fix: {analysis['fix_specificity']}")

        if results["shadow"]:
            lines.append(f"Shadow: {results['shadow']['verdict']}")
            # Include Claude analysis if available
            if "shadow" in results.get("claude_analysis", {}):
                analysis = results["claude_analysis"]["shadow"]
                if analysis.get("root_cause"):
                    lines.append(f"  Root Cause: {analysis['root_cause']}")
                if analysis.get("fix_specificity"):
                    lines.append(f"  Fix: {analysis['fix_specificity']}")

        if results["f14"]:
            lines.append(f"F14: {results['f14']['verdict']}")
            for reg in results["f14"]["regressions"][:3]:
                cat = reg.get("category", "?")
                delta = reg.get("delta", 0)
                lines.append(f"  • {cat}: TPR {delta:+.1%}")
            # Include Claude analysis if available
            if "f14" in results.get("claude_analysis", {}):
                analysis = results["claude_analysis"]["f14"]
                if analysis.get("root_cause"):
                    lines.append(f"  Root Cause: {analysis['root_cause']}")
                if analysis.get("fix_specificity"):
                    lines.append(f"  Fix: {analysis['fix_specificity']}")

        if results["overall_verdict"] == "ALL_PASSED":
            return "✅ All gates passed. Ready for deployment approval."

        return "\n".join([results["message"]] + lines)
