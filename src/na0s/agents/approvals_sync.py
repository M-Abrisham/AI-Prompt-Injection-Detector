"""Git mail-drop sync: pull deploy-approval requests from cloud to local.

The weekly retrain runs in GitHub Actions (a throwaway cloud machine) and
writes ``data/approval_queue/pending_deploy.json``. That file is published onto
a dedicated branch (``config.APPROVALS_BRANCH``); the local daemon has no other
way to see it. This module closes that cloud->local handoff.

The sync is **read-only with respect to the working tree**: it uses
``git fetch`` followed by ``git show <remote>/<branch>:<path>`` and never checks
the branch out, so it cannot disturb whatever branch the daemon's repo is on.

A small state file (``.approvals_state.json``) records which requests have
already been notified and finalized, so the daemon notifies the user *once* per
request and never re-prompts for a request that was already approved/rejected.
"""

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

from na0s import config

logger = logging.getLogger(__name__)

# Path (relative to repo root) where the cloud publishes the request.
PENDING_REL_PATH = "data/approval_queue/pending_deploy.json"

# Keys that change locally as a request is acted on; excluded from the request
# identity so the same cloud request keeps a stable id across status updates.
_VOLATILE_KEYS = frozenset(
    {
        "status",
        "status_updated_at",
        "status_note",
        "executed_at",
        "execution_result",
    }
)


class ApprovalsSync:
    """Fetches deploy-approval requests from the cloud mail-drop branch."""

    def __init__(
        self,
        data_dir: str = "data",
        branch: Optional[str] = None,
        remote: Optional[str] = None,
        repo_root: Optional[str] = None,
    ):
        """Initialize the approvals sync.

        Args:
            data_dir: Root data directory (matches the rest of the agents).
            branch: Mail-drop branch name (default from config.APPROVALS_BRANCH).
            remote: Git remote to fetch from (default from config.APPROVALS_REMOTE).
            repo_root: Repo root override; auto-detected via ``git`` if omitted.
        """
        self.data_dir = Path(data_dir)
        self.branch = branch or config.APPROVALS_BRANCH
        self.remote = remote or config.APPROVALS_REMOTE
        self.repo_root = Path(repo_root) if repo_root else self._detect_repo_root()
        self.pending_path = self.data_dir / "approval_queue" / "pending_deploy.json"
        self.state_path = self.data_dir / "approval_queue" / ".approvals_state.json"

    # ------------------------------------------------------------------ git --

    def _detect_repo_root(self) -> Path:
        """Resolve the repository root via ``git rev-parse``.

        Falls back to the data dir's parent (or CWD) if git is unavailable.
        """
        start = self.data_dir.resolve().parent if self.data_dir.is_absolute() else Path.cwd()
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(start),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip())
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not detect repo root via git: {e}")
        return start

    def _git(self, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
        """Run a git command in the repo root and return the completed process."""
        return subprocess.run(
            ["git", *args],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    # ------------------------------------------------------------- identity --

    @staticmethod
    def request_id(request: Dict[str, Any]) -> str:
        """Return a stable 16-char id for a request, ignoring volatile status."""
        basis = {k: v for k, v in request.items() if k not in _VOLATILE_KEYS}
        blob = json.dumps(basis, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # ---------------------------------------------------------------- state --

    def _load_state(self) -> Dict[str, List[str]]:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                data.setdefault("notified", [])
                data.setdefault("finalized", [])
                return data
            except Exception as e:
                logger.warning(f"Could not read approvals state, resetting: {e}")
        return {"notified": [], "finalized": []}

    def _save_state(self, state: Dict[str, List[str]]) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Could not write approvals state: {e}")

    def already_notified(self, request: Dict[str, Any]) -> bool:
        """True if the user was already alerted about this exact request."""
        return self.request_id(request) in self._load_state()["notified"]

    def mark_notified(self, request: Dict[str, Any]) -> None:
        """Record that the user has been alerted about this request."""
        state = self._load_state()
        rid = self.request_id(request)
        if rid not in state["notified"]:
            state["notified"].append(rid)
            self._save_state(state)

    def mark_finalized(self, request: Dict[str, Any]) -> None:
        """Record that this request reached a final decision (approve/reject)."""
        state = self._load_state()
        rid = self.request_id(request)
        if rid not in state["finalized"]:
            state["finalized"].append(rid)
            self._save_state(state)

    # ----------------------------------------------------------------- sync --

    def fetch_remote_request(self) -> Optional[Dict[str, Any]]:
        """Fetch the mail-drop branch and read the request file (read-only).

        Returns:
            Parsed request dict, or None if the branch/file is absent or invalid.
        """
        fetch = self._git("fetch", self.remote, self.branch)
        if fetch.returncode != 0:
            logger.info(
                f"No approvals branch '{self.branch}' on '{self.remote}' yet "
                f"(or fetch failed): {fetch.stderr.strip()}"
            )
            return None

        ref = f"{self.remote}/{self.branch}:{PENDING_REL_PATH}"
        show = self._git("show", ref)
        if show.returncode != 0:
            logger.debug(f"No {PENDING_REL_PATH} on {ref}: {show.stderr.strip()}")
            return None

        try:
            return json.loads(show.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Remote approval request is not valid JSON: {e}")
            return None

    def sync_pending(self) -> Optional[Dict[str, Any]]:
        """Pull a *new* pending request from the cloud into the local queue.

        Materializes the request into the local ``pending_deploy.json`` so
        ``DeployApprover`` can consume it unchanged.

        Returns:
            The request dict if a new pending request was synced, else None.
        """
        request = self.fetch_remote_request()
        if not request:
            return None

        if request.get("status") != "pending":
            logger.debug("Remote request is not pending; nothing to sync")
            return None

        rid = self.request_id(request)
        if rid in self._load_state()["finalized"]:
            logger.debug(f"Approval request {rid} already finalized; skipping")
            return None

        try:
            self.pending_path.parent.mkdir(parents=True, exist_ok=True)
            self.pending_path.write_text(json.dumps(request, indent=2))
            logger.info(f"Synced pending deploy request {rid} from {self.remote}/{self.branch}")
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Could not write synced request locally: {e}")
            return None

        return request
