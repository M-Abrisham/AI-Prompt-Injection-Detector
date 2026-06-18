"""Deployment approval agent.

Polls for pending deployments, sends approval iMessage, and executes
deploy_model.py only after user approval via OpenClaw.

Phase 6: Auto-Execute Decisions
- On "approve" response: executes deploy_model.py and captures output
- On "reject" response: marks deployment as rejected
- Handles execution failures with retry and error notification
- Updates pending_deploy.json with execution metadata

Phase 7: Approval History
- Records all deployment decisions to approval_history.jsonl
- Tracks execution results, timing, and errors
"""

import hmac
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .approval_history import ApprovalHistoryManager
from .approvals_sync import ApprovalsSync

logger = logging.getLogger(__name__)

# Matches an authorize reply: the literal word "approve" followed by the secret
# nonce. Case-insensitive on the verb; the token must be present (a bare
# "approve" does NOT match group 1 to anything, so it is rejected upstream).
_APPROVE_RE = re.compile(r"^approve\s+(\S+)\s*$", re.IGNORECASE)


class DeployApprover:
    """Manages deployment approval workflow."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.approval_queue_dir = self.data_dir / "approval_queue"
        self.pending_deploy_path = self.approval_queue_dir / "pending_deploy.json"
        self.history = ApprovalHistoryManager(data_dir=data_dir)

    def get_pending_deployment(self) -> Optional[Dict[str, Any]]:
        """Check if there's a pending deployment waiting for approval.

        Returns:
            Pending deployment dict with status='pending', or None
        """
        if not self.pending_deploy_path.exists():
            return None

        try:
            with open(self.pending_deploy_path) as f:
                data = json.load(f)

            if data.get("status") == "pending":
                return data

            return None
        except Exception as e:
            logger.error(f"Error reading pending_deploy.json: {e}")
            return None

    def _deploy_command(self) -> str:
        """Return the literal command ``execute_deploy`` will run (for display)."""
        repo_root = self.data_dir.resolve().parent
        deploy_script = repo_root / "scripts" / "deploy_model.py"
        return f"{sys.executable} {deploy_script}"

    def format_approval_message(
        self, deployment: Dict[str, Any], nonce: Optional[str] = None
    ) -> str:
        """Format deployment summary for approval iMessage.

        Args:
            deployment: Pending deployment dict
            nonce: Per-request secret challenge. When supplied, the message
                instructs the user to reply ``approve <nonce>`` and warns that a
                bare/incorrect code is rejected. The nonce is the only thing
                that authorizes the deploy, so it must travel ONLY to the
                approver's device (never logged to the gateway).

        Returns:
            Human-readable summary for user approval
        """
        gates = deployment.get("gates", {})
        summary = deployment.get("summary", "")

        lines = ["✅ All gates passed. Ready to deploy new model.\n"]

        # Bind the approval to THIS request: short id + the model/candidate
        # identity so the user can confirm what they are authorizing.
        rid = ApprovalsSync.request_id(deployment)
        candidate = (
            deployment.get("candidate_path")
            or deployment.get("model")
            or deployment.get("candidate")
            or "unknown candidate"
        )
        lines.append(f"Request: {rid}")
        lines.append(f"Candidate: {candidate}")
        lines.append(f"Command: {self._deploy_command()}")

        # Gate summary
        if gates.get("canary"):
            canary = gates["canary"]
            if canary.get("passed"):
                tpr = canary.get("tpr", 0)
                tnr = canary.get("tnr", 0)
                lines.append(f"Canary ✓: TPR {tpr:.1%}, TNR {tnr:.1%}")

        if gates.get("shadow"):
            shadow = gates["shadow"]
            if shadow.get("passed"):
                fpr_delta = shadow.get("fpr_delta", 0)
                recall_delta = shadow.get("recall_delta", 0)
                lines.append(f"Shadow ✓: FPR {fpr_delta:+.1%}, Recall {recall_delta:+.1%}")

        if gates.get("decontam"):
            if gates["decontam"].get("passed"):
                lines.append("Decontam ✓: No eval scenario contamination")

        if gates.get("f14"):
            f14 = gates["f14"]
            if f14.get("passed"):
                tpr = f14.get("overall_tpr", 0)
                lines.append(f"F14 ✓: {tpr:.1%} overall TPR")

        if summary:
            lines.append(f"\nNotes: {summary}")

        if nonce:
            lines.append(
                f"\nTo AUTHORIZE, reply EXACTLY `approve {nonce}`. "
                "A missing or incorrect code is REJECTED. Reply `reject` to cancel."
            )
        else:
            # No challenge available (e.g. legacy/standalone call) — never imply
            # a bare "approve" authorizes; require the code.
            lines.append(
                "\nTo AUTHORIZE, reply `approve <code>` using the code from this "
                "request. A missing or incorrect code is REJECTED. "
                "Reply `reject` to cancel."
            )
        return "\n".join(lines)

    def execute_deploy(
        self,
        retry_on_failure: bool = True,
        expected_content_hash: Optional[str] = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """Run deploy_model.py to copy candidate to production.

        Executes the deployment script and captures output. On failure, retries once.

        Args:
            retry_on_failure: Whether to retry once on failure (default True)
            expected_content_hash: TOCTOU guard. If provided, ``pending_deploy.json``
                is re-read and re-hashed immediately before the subprocess; if it
                no longer matches the hash captured at notify time, the deploy is
                ABORTED (no subprocess) — the artifact was mutated after the user
                authorized it.

        Returns:
            Tuple of (success: bool, result_dict: dict) where result_dict contains:
                - stdout: script output
                - stderr: script errors (if any)
                - returncode: exit code
                - error_message: human-readable error (if failed)
                - execution_time: time taken in seconds
                - retry_count: number of retries attempted
        """
        import time

        result_dict = {
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error_message": None,
            "execution_time": 0,
            "retry_count": 0,
        }

        # TOCTOU re-verify: the artifact must be byte-identical (modulo volatile
        # status keys) to what the user saw at notify time. Abort BEFORE any
        # subprocess runs if it changed, so a swapped candidate cannot ride an
        # already-granted approval.
        if expected_content_hash is not None:
            current = self.get_pending_deployment()
            if current is None or ApprovalsSync.content_hash(current) != expected_content_hash:
                msg = (
                    "Deploy ABORTED: pending_deploy.json changed after approval "
                    "(content hash mismatch); refusing to run with a mutated artifact."
                )
                logger.error(msg)
                result_dict["error_message"] = msg
                return False, result_dict

        max_retries = 2 if retry_on_failure else 1
        start_time = time.time()

        # data_dir is "<repo>/data", so its parent is the repo root. Use the
        # running interpreter (sys.executable) and an absolute script path so
        # this works under launchd, where bare "python" and a relative cwd fail.
        repo_root = self.data_dir.resolve().parent
        deploy_script = repo_root / "scripts" / "deploy_model.py"

        for attempt in range(max_retries):
            try:
                logger.info(f"Executing deploy_model.py (attempt {attempt + 1}/{max_retries})...")
                result = subprocess.run(
                    [sys.executable, str(deploy_script)],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                result_dict["stdout"] = result.stdout
                result_dict["stderr"] = result.stderr
                result_dict["returncode"] = result.returncode
                result_dict["execution_time"] = time.time() - start_time
                result_dict["retry_count"] = attempt

                if result.returncode == 0:
                    logger.info("Deploy succeeded")
                    return True, result_dict

                # Attempt failed, will retry
                error_msg = result.stderr.split('\n')[0] if result.stderr else "Unknown error"
                logger.warning(f"Deploy attempt {attempt + 1} failed: {error_msg}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")

            except subprocess.TimeoutExpired:
                error_msg = "Deployment script execution timed out (5 min limit exceeded)"
                logger.error(error_msg)
                result_dict["error_message"] = error_msg
                result_dict["execution_time"] = time.time() - start_time
                result_dict["retry_count"] = attempt
                if attempt == max_retries - 1:
                    return False, result_dict

            except Exception as e:
                error_msg = f"Execution error: {str(e)}"
                logger.error(error_msg)
                result_dict["error_message"] = error_msg
                result_dict["execution_time"] = time.time() - start_time
                result_dict["retry_count"] = attempt
                if attempt == max_retries - 1:
                    return False, result_dict

        # All retries exhausted
        result_dict["error_message"] = (
            f"Deploy failed after {max_retries} attempt(s). "
            f"Last error: {result_dict['stderr'].split(chr(10))[0] if result_dict['stderr'] else 'Unknown'}"
        )
        return False, result_dict

    def update_status(
        self, status: str, note: str = "", execution_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update pending_deploy.json status and execution metadata.

        Args:
            status: "pending", "approved", "rejected", or "failed"
            note: Optional status note
            execution_result: Optional dict with stdout, stderr, returncode, error_message, etc.

        Returns:
            True if update succeeded
        """
        deployment = self.get_pending_deployment()
        if not deployment:
            logger.warning("No pending deployment to update")
            return False

        try:
            deployment["status"] = status
            deployment["status_updated_at"] = datetime.now().isoformat()

            if note:
                deployment["status_note"] = note

            # Record execution metadata if provided
            if execution_result:
                deployment["executed_at"] = datetime.now().isoformat()
                deployment["execution_result"] = {
                    "success": status == "approved",
                    "returncode": execution_result.get("returncode"),
                    "error_message": execution_result.get("error_message"),
                    "stdout_lines": len(execution_result.get("stdout", "").splitlines()),
                    "stderr_lines": len(execution_result.get("stderr", "").splitlines()),
                    "execution_time_seconds": execution_result.get("execution_time", 0),
                    "retry_count": execution_result.get("retry_count", 0),
                }
                if execution_result.get("stderr"):
                    # Store first 500 chars of stderr for debugging
                    deployment["execution_result"]["stderr_preview"] = (
                        execution_result["stderr"][:500]
                    )

            with open(self.pending_deploy_path, "w") as f:
                json.dump(deployment, f, indent=2)

            logger.info(f"Updated deployment status to: {status}")
            return True
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return False

    def handle_approval(
        self,
        user_response: str,
        expected_nonce: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Process user approval response and execute deployment if approved.

        Authorization requires the reply to parse as ``approve <token>`` AND the
        token to match ``expected_nonce`` under a constant-time compare. A bare
        ``approve``, a wrong/stale/empty token, or any other text is REJECTED and
        NEVER triggers a deploy — this is what defeats a forged/replayed
        ``approve`` injected at the gateway, which carries no secret nonce.

        When authorized: re-verifies the artifact against ``content_hash``
        (TOCTOU) and, if unchanged, executes deploy_model.py and updates status.
        When ``reject``: marks the deployment rejected.

        Args:
            user_response: Raw reply text from the user.
            expected_nonce: The per-request secret minted at notify time. Required
                to authorize; ``None`` means no challenge is on file and authorize
                is impossible.
            content_hash: Artifact hash captured at notify time, forwarded to
                ``execute_deploy`` as the TOCTOU guard.

        Returns:
            Tuple of (success: bool, message: str) where message is iMessage notification
        """
        response = (user_response or "").strip()
        response_lower = response.lower()

        match = _APPROVE_RE.match(response)
        # An "approve-shaped" reply is anything starting with the approve verb,
        # including a bare/token-less "approve". All of these are authorization
        # ATTEMPTS and must be routed through the nonce check — never to the
        # generic "unexpected response" path — so a forged bare "approve" is
        # explicitly REJECTED rather than ambiguously handled.
        is_approve_attempt = response_lower == "approve" or response_lower.startswith(
            "approve "
        ) or response_lower.startswith("approve\t")

        if match or is_approve_attempt:
            token = match.group(1) if match else ""
            # Constant-time compare; reject if there is no challenge on file or
            # the token does not match the CURRENT request's nonce. A bare
            # "approve" has an empty token and never matches a real nonce.
            authorized = bool(expected_nonce) and bool(token) and hmac.compare_digest(
                token, expected_nonce
            )
            if not authorized:
                logger.warning("Approval reply carried a missing/incorrect nonce; REJECTED")
                self.history.record_action(
                    action_type="deploy",
                    status="rejected",
                    approved_by="user",
                    reason="Approval reply had an invalid or stale authorization code",
                    execution_result="skipped",
                )
                return False, (
                    "⛔ Not approved: the authorization code was missing or incorrect. "
                    "Reply EXACTLY `approve <code>` using the code from the request."
                )

            logger.info("User approved deployment with valid nonce. Executing...")

            success, exec_result = self.execute_deploy(
                retry_on_failure=True, expected_content_hash=content_hash
            )

            if success:
                self.update_status("approved", "User approved and deployed", exec_result)

                # Record to approval history
                self.history.record_action(
                    action_type="deploy",
                    status="approved",
                    approved_by="user",
                    reason="User approved deployment via iMessage",
                    execution_result="success",
                    execution_time_seconds=exec_result.get("execution_time", 0),
                )

                message = (
                    f"✅ Model deployed successfully\n"
                    f"Execution time: {exec_result.get('execution_time', 0):.1f}s\n"
                    f"Retries: {exec_result.get('retry_count', 0)}"
                )
                logger.info(message)
                return True, message
            else:
                self.update_status("failed", "Deploy execution failed", exec_result)

                # Record failed deployment to history
                error_msg = exec_result.get("error_message", "Unknown error")
                self.history.record_action(
                    action_type="deploy",
                    status="approved",
                    approved_by="user",
                    reason="User approved but execution failed",
                    execution_result="failed",
                    execution_time_seconds=exec_result.get("execution_time", 0),
                    error=error_msg,
                )

                message = (
                    f"❌ Deploy failed: {error_msg}\n"
                    f"Retries attempted: {exec_result.get('retry_count', 0)}\n"
                    f"Execution time: {exec_result.get('execution_time', 0):.1f}s"
                )
                logger.error(message)
                return False, message

        elif response_lower == "reject":
            self.update_status("rejected", "User rejected deployment")

            # Record rejection to history
            self.history.record_action(
                action_type="deploy",
                status="rejected",
                approved_by="user",
                reason="User rejected deployment via iMessage",
                execution_result="skipped",
            )

            message = "✅ Deployment rejected and cancelled"
            logger.info(message)
            return True, message

        else:
            logger.warning(f"Unexpected response: {user_response}")
            return False, f"❓ Unexpected response: {user_response}. Expected 'approve' or 'reject'"

    def check_and_notify(self) -> Optional[str]:
        """Check for pending deployment and return approval message.

        Returns:
            Approval message if deployment pending, None otherwise
        """
        deployment = self.get_pending_deployment()
        if not deployment:
            return None

        return self.format_approval_message(deployment)
