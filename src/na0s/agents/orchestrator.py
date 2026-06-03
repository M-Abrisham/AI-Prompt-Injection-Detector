"""Main orchestrator for Na0S agent automation pipeline.

Coordinates gate analysis, quarantine review, deployment approval,
and synthetic validation via Claude + OpenClaw iMessage integration.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from .gate_analyzer import GateAnalyzer
from .quarantine_reviewer import QuarantineReviewer
from .deploy_approver import DeployApprover
from .synthetic_validator import SyntheticValidator
from .openclaw_bridge import OpenClawBridge
from .approvals_sync import ApprovalsSync

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Main orchestrator for agent-driven Na0S pipeline."""

    def __init__(self, data_dir: str = "data", openclaw_url: str = "http://localhost:3000", use_claude: bool = True):
        """Initialize orchestrator with all agent components.

        Args:
            data_dir: Root data directory path
            openclaw_url: OpenClaw API endpoint
            use_claude: Whether to enable Claude API analysis in gate analyzer
        """
        self.data_dir = Path(data_dir)
        self.openclaw = OpenClawBridge(base_url=openclaw_url)

        # Initialize agent modules
        self.gate_analyzer = GateAnalyzer(data_dir=data_dir, use_claude=use_claude)
        self.quarantine_reviewer = QuarantineReviewer(data_dir=data_dir)
        self.deploy_approver = DeployApprover(data_dir=data_dir)
        self.synthetic_validator = SyntheticValidator(data_dir=data_dir)

        # Cloud->local transport for deploy-approval requests (git mail-drop)
        self.approvals_sync = ApprovalsSync(data_dir=data_dir)

    def run_gate_analysis(self) -> bool:
        """Analyze gate failures and send alert if any gate failed.

        Returns:
            True if all gates passed or alerts sent successfully
        """
        logger.info("Running gate analysis...")
        results = self.gate_analyzer.diagnose_failures()
        message = self.gate_analyzer.format_message()

        if results["overall_verdict"] == "ALL_PASSED":
            logger.info("All gates passed")
            return True

        # Log Claude analysis results at INFO level
        if results.get("claude_analysis"):
            for gate_type, analysis in results["claude_analysis"].items():
                if analysis:
                    root_cause = analysis.get("root_cause", "N/A")
                    fix = analysis.get("fix_specificity", "N/A")
                    logger.info(f"Claude analysis ({gate_type}): root_cause='{root_cause}', fix='{fix}'")

        # Send alert
        if not self.openclaw.send_message(message):
            logger.error("Failed to send gate analysis alert")
            return False

        # Write failure report
        report_path = self.gate_analyzer.write_report()
        if report_path:
            logger.info(f"Failure report written to {report_path}")

        return True

    def run_quarantine_review(self) -> bool:
        """Review quarantine backlog and send recommendations.

        Returns:
            True if review completed or no pending entries
        """
        logger.info("Running quarantine review...")
        message = self.quarantine_reviewer.format_message()

        if "empty" in message.lower():
            logger.info("Quarantine backlog is empty")
            return True

        # Send review summary
        if not self.openclaw.send_message(message):
            logger.error("Failed to send quarantine review")
            return False

        # Write the review report before waiting (so it's persisted even if the
        # user never replies this cycle).
        report_path = self.quarantine_reviewer.write_review_report()
        if report_path:
            logger.info(f"Review report written to {report_path}")

        # Poll for the user's promote/reject decision and execute it.
        logger.info("Waiting for user quarantine review actions...")
        response = self.openclaw.poll_replies(timeout=600)
        if not response:
            logger.info("No quarantine action this cycle; leaving entries pending")
            return True

        success, result_message = self.quarantine_reviewer.handle_user_response(response)
        self.openclaw.send_message(result_message)
        if not success:
            logger.warning(f"Quarantine action not completed: {result_message}")

        return True

    def run_synthetic_validation(self) -> bool:
        """Validate synthetic data quality.

        Returns:
            True if validation completed
        """
        logger.info("Running synthetic data validation...")
        report = self.synthetic_validator.compile_validation_report()
        message = self.synthetic_validator.format_message()

        # Send validation summary
        if not self.openclaw.send_message(message):
            logger.error("Failed to send synthetic validation alert")
            return False

        # Write validation report
        report_path = self.synthetic_validator.write_report()
        if report_path:
            logger.info(f"Validation report written to {report_path}")

        # Nothing flagged → no decision to wait on.
        if not report or report.get("flagged_count", 0) == 0:
            return True

        # Poll for the user's remove/keep decision and execute it.
        logger.info("Waiting for user synthetic-data decision...")
        response = self.openclaw.poll_replies(timeout=600)
        if not response:
            logger.info("No synthetic-data action this cycle; keeping all samples")
            return True

        acted, removed, result_message = self.synthetic_validator.handle_user_response(response)
        self.openclaw.send_message(result_message)
        if acted and removed:
            logger.info(f"Removed {removed} flagged synthetic sample(s)")

        return True

    def run_deployment_approval(self) -> bool:
        """Pull any cloud deploy request, notify the user, and act on the reply.

        Pulls the latest approval request from the cloud mail-drop branch
        (read-only), alerts the user once per request, polls for the reply, then
        executes the decision and relays the result back over iMessage.

        Returns:
            True if there was nothing pending or the decision was handled OK.
        """
        logger.info("Checking for pending deployment...")

        # Pull the latest request from the cloud mail-drop branch (read-only).
        self.approvals_sync.sync_pending()

        deployment = self.deploy_approver.get_pending_deployment()
        if not deployment:
            logger.info("No pending deployment")
            return True

        # Notify once per unique request; re-prompts would otherwise spam the
        # user every monitoring cycle while we wait for a reply.
        if not self.approvals_sync.already_notified(deployment):
            message = self.deploy_approver.format_approval_message(deployment)
            if not self.openclaw.send_message(message):
                logger.error("Failed to send deployment approval request")
                return False
            self.approvals_sync.mark_notified(deployment)
        else:
            logger.info("Approval already sent for this request; polling for reply")

        logger.info("Waiting for user deployment approval...")

        # Poll for user response. No reply -> leave un-finalized so the next
        # monitoring cycle keeps waiting (it won't re-notify, see above).
        response = self.openclaw.poll_replies(timeout=600)  # 10 minute timeout
        if not response:
            logger.warning("No deployment approval response received")
            return False

        # handle_approval returns (success, message); relay the message back to
        # the user and use the boolean to detect failure (was previously
        # treated as a single truthy value, so failures looked like successes).
        success, result_message = self.deploy_approver.handle_approval(response)
        self.openclaw.send_message(result_message)

        # This request is decided either way; don't prompt for it again.
        self.approvals_sync.mark_finalized(deployment)

        if not success:
            logger.error(f"Deployment action did not succeed: {result_message}")
            return False

        return True

    def run_full_pipeline(self) -> bool:
        """Run complete agent orchestration pipeline.

        Executes in order:
        1. Gate analysis → report failures
        2. Quarantine review → get promote/reject decisions
        3. Synthetic validation → flag low-quality samples
        4. Deployment approval → wait for user approval

        Returns:
            True if all steps completed successfully
        """
        logger.info("Starting full pipeline orchestration...")

        # Step 1: Gate analysis
        if not self.run_gate_analysis():
            logger.error("Gate analysis failed")
            return False

        # Step 2: Quarantine review
        if not self.run_quarantine_review():
            logger.error("Quarantine review failed")
            return False

        # Step 3: Synthetic validation
        if not self.run_synthetic_validation():
            logger.error("Synthetic validation failed")
            return False

        # Step 4: Deployment approval (only if all prior gates passed)
        gates_result = self.gate_analyzer.diagnose_failures()
        if gates_result["overall_verdict"] == "ALL_PASSED":
            if not self.run_deployment_approval():
                logger.error("Deployment approval failed")
                return False

        logger.info("Pipeline orchestration completed")
        return True

    def run_continuous_monitoring(self, interval: int = 300) -> None:
        """Run orchestrator in continuous monitoring loop.

        Checks for pending actions every N seconds.

        Args:
            interval: Check interval in seconds (default 5 min)
        """
        logger.info(f"Starting continuous monitoring loop (interval={interval}s)")

        while True:
            try:
                # Check each agent in sequence
                if self.gate_analyzer.diagnose_failures()["overall_verdict"] == "FAILED":
                    self.run_gate_analysis()

                if self.quarantine_reviewer.compile_review_summary():
                    self.run_quarantine_review()

                # Always run deployment approval: it pulls the cloud mail-drop
                # first, so there may be a request even with no local file yet.
                self.run_deployment_approval()
            except KeyboardInterrupt:
                logger.info("Continuous monitoring stopped")
                return
            except Exception as e:
                # A transient error in one cycle must not kill the daemon.
                logger.error(f"Error in monitoring cycle (continuing): {e}")

            time.sleep(interval)
