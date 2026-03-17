"""Audit logging for LLM Judge invocations.

Writes one JSONL record per judge call.  Controlled by the
``NA0S_JUDGE_AUDIT`` environment variable (set to ``1`` to enable).
"""

import json
import os
import threading
import time
from pathlib import Path


_DEFAULT_LOG_PATH = os.path.join("data", "audit", "judge_audit.jsonl")


class JudgeAuditLogger:
    """Append-only JSONL audit log for judge invocations."""

    def __init__(self):
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_enabled() -> bool:
        return os.getenv("NA0S_JUDGE_AUDIT", "") == "1"

    def get_log_path(self) -> str:
        return os.getenv("NA0S_JUDGE_AUDIT_LOG", _DEFAULT_LOG_PATH)

    def log_invocation(
        self,
        input_hash: str,
        verdict: str,
        confidence: float,
        reasoning: str,
        model: str,
        latency_ms: float,
        error: str = "",
    ) -> None:
        """Write a single audit record (JSONL) if auditing is enabled."""
        if not self.is_enabled():
            return

        record = {
            "timestamp": time.time(),
            "input_hash": input_hash,
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "model": model,
            "latency_ms": latency_ms,
            "error": error,
        }

        log_path = self.get_log_path()
        with self._lock:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_recent(self, n: int) -> list[dict]:
        """Return the last *n* audit records from the log file."""
        log_path = self.get_log_path()
        if not os.path.isfile(log_path):
            return []

        with self._lock:
            with open(log_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

        entries: list[dict] = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries
