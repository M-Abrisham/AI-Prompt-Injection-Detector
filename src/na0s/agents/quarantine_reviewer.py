"""Quarantine backlog reviewer agent.

Monitors data/quarantine/ for pending/stale validation and generates
recommendations for promotion, rejection, or revision via iMessage.

Phase 6: Auto-Execute Decisions
- On "promote <NAME>" response: executes quarantine.py --promote <name>
- On "reject <NAME>" response: executes quarantine.py --reject <name>
- Captures script output and sends status confirmation via iMessage
- Handles execution failures gracefully with error logging
- Updates quarantine metadata with action history

Phase 7: Approval History
- Records all quarantine decisions (promote/reject) to approval_history.jsonl
- Tracks execution results, timing, and errors
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import random
import time

from .approval_history import ApprovalHistoryManager

logger = logging.getLogger(__name__)


class QuarantineReviewer:
    """Reviews quarantine backlog and recommends actions."""

    # Max age before flagging as "stale" (days)
    STALE_THRESHOLD_DAYS = 20

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.quarantine_dir = self.data_dir / "quarantine"
        self.quarantine_log = self.quarantine_dir / "quarantine_log.json"
        self.history = ApprovalHistoryManager(data_dir=data_dir)

    def get_pending_entries(self) -> List[Dict[str, Any]]:
        """Scan quarantine directory for pending validation.

        Returns:
            List of metadata dicts for datasets with validation_status='pending'
        """
        pending = []

        if not self.quarantine_dir.exists():
            logger.warning(f"Quarantine directory not found: {self.quarantine_dir}")
            return pending

        for source_dir in self.quarantine_dir.iterdir():
            if not source_dir.is_dir():
                continue

            metadata_path = source_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)

                if metadata.get("validation_status") == "pending":
                    metadata["source_name"] = source_dir.name
                    metadata["metadata_path"] = str(metadata_path)
                    pending.append(metadata)
            except Exception as e:
                logger.error(f"Error reading {metadata_path}: {e}")

        return pending

    def identify_stale_entries(
        self, entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter entries that have been pending too long.

        Args:
            entries: List of pending metadata dicts

        Returns:
            Entries where ingested_at is older than STALE_THRESHOLD_DAYS
        """
        stale = []
        now = datetime.now()

        for entry in entries:
            ingested_str = entry.get("ingested_at")
            if not ingested_str:
                continue

            try:
                ingested = datetime.fromisoformat(ingested_str)
                age_days = (now - ingested).days

                if age_days > self.STALE_THRESHOLD_DAYS:
                    entry["age_days"] = age_days
                    stale.append(entry)
            except Exception as e:
                logger.error(f"Error parsing ingested_at: {e}")

        return sorted(stale, key=lambda x: x.get("age_days", 0), reverse=True)

    def sample_entries(self, entries: List[Dict[str, Any]], per_entry: int = 5) -> Dict[str, Any]:
        """Sample rows from each quarantine entry for human review.

        Args:
            entries: List of metadata dicts
            per_entry: Number of rows to sample per entry

        Returns:
            Dict mapping entry name to list of sampled rows
        """
        samples = {}

        for entry in entries:
            source_name = entry.get("source_name", "?")
            data_path = Path(entry.get("metadata_path", "")).parent / "data.jsonl"

            if not data_path.exists():
                logger.warning(f"Data file not found: {data_path}")
                continue

            try:
                # Read all rows to sample
                rows = []
                with open(data_path) as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))

                # Random sample
                sampled = random.sample(rows, min(per_entry, len(rows)))
                samples[source_name] = {
                    "count": len(rows),
                    "sampled": sampled,
                    "metadata": entry,
                }
            except Exception as e:
                logger.error(f"Error sampling from {data_path}: {e}")

        return samples

    def compile_review_summary(self) -> Optional[Dict[str, Any]]:
        """Compile full quarantine review summary.

        Returns:
            Dict with pending entries, stale entries, and samples, or None if none pending
        """
        pending = self.get_pending_entries()
        if not pending:
            logger.info("No pending quarantine entries")
            return None

        stale = self.identify_stale_entries(pending)
        samples = self.sample_entries(stale[:5])  # Review top 5 stale entries

        return {
            "timestamp": datetime.now().isoformat(),
            "total_pending": len(pending),
            "stale_count": len(stale),
            "stale_entries": stale[:10],
            "samples": samples,
            "message": self._format_message_summary(pending, stale, samples),
        }

    def _format_message_summary(
        self,
        pending: List[Dict[str, Any]],
        stale: List[Dict[str, Any]],
        samples: Dict[str, Any],
    ) -> str:
        """Format review summary for iMessage.

        Args:
            pending: All pending entries
            stale: Stale entries (>20 days)
            samples: Sampled rows from stale entries

        Returns:
            Formatted message string
        """
        lines = [f"📋 Quarantine Review: {len(pending)} datasets pending validation"]

        if stale:
            lines.append(f"⚠️  {len(stale)} datasets are stale (>{self.STALE_THRESHOLD_DAYS} days)")

            # Brief summary of stale entries
            for entry in stale[:3]:
                name = entry.get("source_name", "?")
                age = entry.get("age_days", "?")
                tier = entry.get("tier", "?")
                lines.append(f"  • {name} (Tier {tier}, {age}d old)")

        if samples:
            lines.append("\nSample recommendations:")
            for source_name, sample_info in list(samples.items())[:3]:
                metadata = sample_info["metadata"]
                trust_score = metadata.get("trust_score", "?")
                lines.append(f"  {source_name}: trust {trust_score}")
                lines.append(f"    → Review samples & reply 'promote {source_name.upper()}' or 'reject {source_name.upper()}'")

        lines.append("\nReply with: promote <NAME>, reject <NAME>, or 'skip review'")
        return "\n".join(lines)

    def write_review_report(
        self, report_dir: str = "data/approval_queue/quarantine_reviews"
    ) -> Optional[str]:
        """Write review report to disk.

        Args:
            report_dir: Directory to write reports to

        Returns:
            Path to written report, or None if no pending entries
        """
        summary = self.compile_review_summary()
        if not summary:
            return None

        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"review_{timestamp}.json"

        try:
            with open(report_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Wrote review report to {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Error writing report: {e}")
            return None

    def _get_entry_info(self, entry_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a quarantine entry.

        Args:
            entry_name: Name of quarantine entry

        Returns:
            Metadata dict or None if not found
        """
        entry_dir = self.quarantine_dir / entry_name
        if not entry_dir.exists():
            return None

        metadata_path = entry_dir / "metadata.json"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for {entry_name}: {e}")
            return None

    def _update_entry_metadata(self, entry_name: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for a quarantine entry, including action history.

        Args:
            entry_name: Name of quarantine entry
            updates: Dict with fields to update

        Returns:
            True if update succeeded
        """
        entry_dir = self.quarantine_dir / entry_name
        metadata_path = entry_dir / "metadata.json"

        if not metadata_path.exists():
            logger.warning(f"No metadata for {entry_name}")
            return False

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            metadata.update(updates)

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error updating metadata for {entry_name}: {e}")
            return False

    def execute_action(self, action: str, entry_name: str) -> tuple[bool, Dict[str, Any]]:
        """Execute promote/reject action on a quarantine entry via quarantine.py script.

        Calls scripts/quarantine.py with appropriate flags, captures output,
        and updates entry metadata with action history.

        Args:
            action: "promote" or "reject"
            entry_name: Name of quarantine entry

        Returns:
            Tuple of (success: bool, result_dict: dict) where result_dict contains:
                - action: the action performed
                - entry_name: the entry name
                - stdout: script output
                - stderr: script errors
                - returncode: exit code
                - error_message: human-readable error (if failed)
                - execution_time: time taken in seconds
        """
        if action not in ("promote", "reject"):
            logger.error(f"Unknown action: {action}")
            return False, {"action": action, "entry_name": entry_name, "error_message": "Invalid action"}

        # Verify entry exists
        entry_info = self._get_entry_info(entry_name)
        if not entry_info:
            error_msg = f"Quarantine entry not found: {entry_name}"
            logger.error(error_msg)
            return False, {
                "action": action,
                "entry_name": entry_name,
                "error_message": error_msg,
            }

        start_time = time.time()
        result_dict = {
            "action": action,
            "entry_name": entry_name,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error_message": None,
            "execution_time": 0,
        }

        try:
            logger.info(f"Executing quarantine.py --{action} {entry_name}...")

            # Build command
            cmd = [
                "python",
                "scripts/quarantine.py",
                f"--{action}",
                entry_name,
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.data_dir.parent),  # Go to repo root
                capture_output=True,
                text=True,
                timeout=300,
            )

            result_dict["stdout"] = result.stdout
            result_dict["stderr"] = result.stderr
            result_dict["returncode"] = result.returncode
            result_dict["execution_time"] = time.time() - start_time

            if result.returncode != 0:
                error_lines = result.stderr.split('\n') if result.stderr else ["Unknown error"]
                result_dict["error_message"] = error_lines[0]
                logger.error(f"Action {action} failed: {result_dict['error_message']}")
                return False, result_dict

            # Update entry metadata with action history
            action_record = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "actor": "approver_agent",
                "result": "success",
            }

            # Initialize action_history if not present
            current_metadata = self._get_entry_info(entry_name)
            if current_metadata:
                action_history = current_metadata.get("action_history", [])
                action_history.append(action_record)
                self._update_entry_metadata(entry_name, {"action_history": action_history})

            logger.info(f"Action {action} succeeded for {entry_name}")
            return True, result_dict

        except subprocess.TimeoutExpired:
            error_msg = f"{action} operation timed out (5 min limit exceeded)"
            result_dict["error_message"] = error_msg
            result_dict["execution_time"] = time.time() - start_time
            logger.error(error_msg)
            return False, result_dict

        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            result_dict["error_message"] = error_msg
            result_dict["execution_time"] = time.time() - start_time
            logger.error(error_msg)
            return False, result_dict

    def format_message(self) -> str:
        """Format quarantine status for iMessage.

        Returns:
            Human-readable status message
        """
        summary = self.compile_review_summary()
        if not summary:
            return "✅ Quarantine backlog is empty. All pending datasets have been reviewed."
        return summary["message"]

    def handle_user_response(self, user_response: str) -> tuple[bool, str]:
        """Process user response for quarantine actions (promote/reject).

        Parses response like "promote DATASET_A" or "reject DATASET_B",
        executes the appropriate quarantine.py command, and returns status.

        Args:
            user_response: User input like "promote D" or "reject E"

        Returns:
            Tuple of (success: bool, message: str) where message is iMessage notification
        """
        response = user_response.lower().strip()

        # Parse command: "promote <name>" or "reject <name>"
        parts = response.split()
        if len(parts) < 2:
            return False, f"❓ Invalid format. Use 'promote <NAME>' or 'reject <NAME>'"

        action = parts[0]
        entry_name = parts[1].lower()

        if action not in ("promote", "reject"):
            return False, f"❓ Unknown action '{action}'. Use 'promote' or 'reject'"

        # Execute the action
        success, result = self.execute_action(action, entry_name)

        if success:
            # Record to approval history
            self.history.record_action(
                action_type=action,
                status="approved",
                approved_by="user",
                reason=f"User approved {action} for dataset {entry_name}",
                execution_result="success",
                execution_time_seconds=result.get("execution_time", 0),
                metadata={"dataset": entry_name},
            )

            if action == "promote":
                message = (
                    f"✅ Dataset {entry_name.upper()} promoted to staging\n"
                    f"Execution time: {result.get('execution_time', 0):.1f}s"
                )
            else:  # reject
                message = (
                    f"✅ Dataset {entry_name.upper()} rejected and removed\n"
                    f"Execution time: {result.get('execution_time', 0):.1f}s"
                )
            logger.info(message)
            return True, message
        else:
            # Record failed action to history
            error_msg = result.get("error_message", "Unknown error")
            self.history.record_action(
                action_type=action,
                status="approved",
                approved_by="user",
                reason=f"User approved {action} but execution failed",
                execution_result="failed",
                execution_time_seconds=result.get("execution_time", 0),
                error=error_msg,
                metadata={"dataset": entry_name},
            )

            message = (
                f"❌ {action.capitalize()} failed: {error_msg}\n"
                f"Dataset: {entry_name.upper()}"
            )
            logger.error(message)
            return False, message
