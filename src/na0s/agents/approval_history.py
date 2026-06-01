"""Approval history tracking and analytics.

Maintains thread-safe JSONL log of all approval/rejection/deployment decisions
with timestamps, metadata, and execution results. Provides querying and
aggregation APIs for dashboard and reporting.

Features:
- Thread-safe JSONL writes with atomic operations
- Auto-rotation: archives entries older than 365 days
- Querying by date range, action type, status
- Statistics: success rates, execution times, action counts
- History exports for different action types
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import fcntl

logger = logging.getLogger(__name__)


class ApprovalHistoryManager:
    """Manages approval history with thread-safe JSONL persistence."""

    # Retention policy: keep last 365 days in active log
    RETENTION_DAYS = 365

    def __init__(self, data_dir: str = "data"):
        """Initialize history manager.

        Args:
            data_dir: Root data directory path
        """
        self.data_dir = Path(data_dir)
        self.approval_queue_dir = self.data_dir / "approval_queue"
        self.approval_history_path = self.approval_queue_dir / "approval_history.jsonl"
        self.archive_dir = self.approval_queue_dir / "approval_history_archive"

        # Ensure directories exist
        self.approval_queue_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Thread lock for file operations
        self._lock = threading.RLock()

    def record_action(
        self,
        action_type: str,
        status: str,
        approved_by: Optional[str] = None,
        reason: Optional[str] = None,
        execution_result: Optional[str] = None,
        execution_time_seconds: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record an approval action to history.

        Args:
            action_type: Type of action ("deploy", "promote", "reject", "quarantine", etc.)
            status: Status ("approved", "rejected", "failed", "pending")
            approved_by: User who approved (optional, defaults to "system")
            reason: Human-readable reason for action
            execution_result: Result of execution ("success", "failed", "timeout", etc.)
            execution_time_seconds: Seconds taken to execute
            error: Error message if failed
            metadata: Additional metadata dict

        Returns:
            True if record was written successfully
        """
        try:
            # Build record
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "action_type": action_type,
                "status": status,
                "approved_by": approved_by or "system",
                "reason": reason or "",
                "execution_result": execution_result,
                "execution_time_seconds": execution_time_seconds,
                "error": error,
            }

            # Add optional metadata
            if metadata:
                record.update(metadata)

            # Write atomically with file locking
            with self._lock:
                # Use fcntl file locking on Unix-like systems
                with open(self.approval_history_path, "a") as f:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        f.write(json.dumps(record) + "\n")
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except (OSError, IOError):
                        # Fallback: just write without lock (Windows compatibility)
                        f.write(json.dumps(record) + "\n")

            logger.info(
                f"Recorded {action_type} action: status={status}, "
                f"approved_by={approved_by or 'system'}"
            )

            # Auto-rotate history if needed
            self._rotate_history()

            return True

        except Exception as e:
            logger.error(f"Error recording action: {e}")
            return False

    def get_recent(self, days: int = 30, action_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent approval records.

        Args:
            days: Number of days to look back (default 30)
            action_type: Optional filter by action type

        Returns:
            List of approval records, most recent first
        """
        try:
            if not self.approval_history_path.exists():
                return []

            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            records = []

            with self._lock:
                with open(self.approval_history_path) as f:
                    for line in f:
                        if not line.strip():
                            continue

                        try:
                            record = json.loads(line)

                            # Filter by date
                            if record.get("timestamp", "") < cutoff_date:
                                continue

                            # Filter by action type
                            if action_type and record.get("action_type") != action_type:
                                continue

                            records.append(record)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipped malformed history line: {line[:50]}")
                            continue

            # Return most recent first
            return sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)

        except Exception as e:
            logger.error(f"Error reading recent records: {e}")
            return []

    def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get approval statistics for a date range.

        Args:
            days: Number of days to analyze (default 30)

        Returns:
            Dict with stats: total_approvals, success_rate, avg_execution_time, by_action_type
        """
        try:
            records = self.get_recent(days=days)

            if not records:
                return {
                    "total_approvals": 0,
                    "success_rate": 0.0,
                    "avg_execution_time_seconds": 0.0,
                    "by_action_type": {},
                    "by_status": {},
                }

            # Calculate stats
            total = len(records)
            successful = len([r for r in records if r.get("status") == "approved"])
            success_rate = (successful / total * 100) if total > 0 else 0.0

            # Execution times
            exec_times = [
                r["execution_time_seconds"]
                for r in records
                if r.get("execution_time_seconds") is not None
            ]
            avg_exec_time = (sum(exec_times) / len(exec_times)) if exec_times else 0.0

            # By action type
            by_action = {}
            for record in records:
                action = record.get("action_type", "unknown")
                if action not in by_action:
                    by_action[action] = {"count": 0, "approved": 0}
                by_action[action]["count"] += 1
                if record.get("status") == "approved":
                    by_action[action]["approved"] += 1

            # By status
            by_status = {}
            for record in records:
                status = record.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1

            return {
                "total_approvals": total,
                "successful_approvals": successful,
                "success_rate": round(success_rate, 2),
                "avg_execution_time_seconds": round(avg_exec_time, 2),
                "by_action_type": {
                    k: {
                        "count": v["count"],
                        "approved": v["approved"],
                        "success_rate": round(v["approved"] / v["count"] * 100, 2)
                        if v["count"] > 0
                        else 0,
                    }
                    for k, v in by_action.items()
                },
                "by_status": by_status,
            }

        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
            return {}

    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get all deployment approval history.

        Returns:
            List of deployment records, most recent first
        """
        return self.get_recent(days=self.RETENTION_DAYS, action_type="deploy")

    def get_quarantine_history(self) -> List[Dict[str, Any]]:
        """Get all quarantine action history (promote/reject/review).

        Returns:
            List of quarantine records, most recent first
        """
        history = []
        for action in ["promote", "reject", "quarantine_review"]:
            history.extend(self.get_recent(days=self.RETENTION_DAYS, action_type=action))

        return sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)

    def _rotate_history(self) -> bool:
        """Rotate history: archive entries older than RETENTION_DAYS.

        Returns:
            True if rotation occurred or was not needed
        """
        try:
            if not self.approval_history_path.exists():
                return True

            cutoff_date = (datetime.utcnow() - timedelta(days=self.RETENTION_DAYS)).isoformat()
            active_records = []
            archived_records = []

            with self._lock:
                # Read all records
                with open(self.approval_history_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            timestamp = record.get("timestamp", "")

                            if timestamp < cutoff_date:
                                archived_records.append(record)
                            else:
                                active_records.append(record)
                        except json.JSONDecodeError:
                            # Keep malformed lines in active file
                            active_records.append(line.strip())

                # If nothing to archive, skip
                if not archived_records:
                    return True

                # Write to archive file (append mode)
                archive_path = self.archive_dir / f"approval_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
                with open(archive_path, "w") as f:
                    for record in archived_records:
                        if isinstance(record, dict):
                            f.write(json.dumps(record) + "\n")
                        else:
                            f.write(record + "\n")

                # Rewrite active log
                with open(self.approval_history_path, "w") as f:
                    for record in active_records:
                        if isinstance(record, dict):
                            f.write(json.dumps(record) + "\n")
                        else:
                            f.write(record + "\n")

                logger.info(
                    f"Rotated {len(archived_records)} old entries to {archive_path.name}"
                )

            return True

        except Exception as e:
            logger.error(f"Error rotating history: {e}")
            return False

    def clear_history(self) -> bool:
        """Clear all history (use with caution in tests).

        Returns:
            True if cleared successfully
        """
        try:
            with self._lock:
                if self.approval_history_path.exists():
                    self.approval_history_path.unlink()
            logger.warning("Cleared all approval history")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False
