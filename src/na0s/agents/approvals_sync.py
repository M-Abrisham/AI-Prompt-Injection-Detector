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
import hmac
import json
import logging
import os
import secrets
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

from na0s import config

logger = logging.getLogger(__name__)

# Path (relative to repo root) where the cloud publishes the request.
PENDING_REL_PATH = "data/approval_queue/pending_deploy.json"

# Environment variable holding the shared HMAC key used to sign/verify the
# mail-drop request. When set on BOTH the cloud producer and the local daemon,
# only requests signed with this key are accepted.
HMAC_KEY_ENV = "NA0S_AGENT_APPROVAL_HMAC_KEY"

# Module-level latch so the "signing DISABLED" warning is logged loudly but
# only once per process, not on every poll of the mail-drop branch.
_unsigned_mode_warned = False


def _approval_key() -> Optional[bytes]:
    """Return the configured HMAC key as bytes, or None if unset/empty.

    The key is read from the ``NA0S_AGENT_APPROVAL_HMAC_KEY`` environment
    variable. When it is absent the mail-drop runs in (backward-compatible)
    unsigned mode — see ``sign_request`` / ``verify_request`` callers.
    """
    raw = os.environ.get(HMAC_KEY_ENV)
    if not raw:
        return None
    return raw.encode("utf-8")


def _canonical_request_bytes(request: Dict[str, Any]) -> bytes:
    """Canonical JSON of a request EXCLUDING its ``signature`` field, as bytes.

    Canonicalization is ``json.dumps`` with ``sort_keys=True`` and the compact
    ``(",", ":")`` separators so the signed bytes are stable under key
    reordering and insignificant whitespace differences.
    """
    without_sig = {k: v for k, v in request.items() if k != "signature"}
    blob = json.dumps(without_sig, sort_keys=True, separators=(",", ":"))
    return blob.encode("utf-8")


def sign_request(request: Dict[str, Any], key) -> str:
    """Compute the HMAC-SHA256 signature (hex) of ``request``.

    The signature covers the canonical JSON of the request with any existing
    ``signature`` field removed, so a request can be (re)signed idempotently.

    Args:
        request: The request dict to sign.
        key: The shared secret as ``bytes`` or ``str``.

    Returns:
        Lowercase hex digest of the HMAC-SHA256.
    """
    key_bytes = key.encode("utf-8") if isinstance(key, str) else key
    return hmac.new(key_bytes, _canonical_request_bytes(request), hashlib.sha256).hexdigest()


def verify_request(request: Dict[str, Any], key) -> bool:
    """Constant-time-verify a request's embedded ``signature`` against ``key``.

    Recomputes the HMAC over the request (excluding ``signature``) and compares
    it to the embedded ``signature`` using ``hmac.compare_digest``.

    Returns:
        True iff a ``signature`` is present and matches. False when the
        ``signature`` field is missing/empty or does not match.
    """
    embedded = request.get("signature")
    if not embedded or not isinstance(embedded, str):
        return False
    expected = sign_request(request, key)
    return hmac.compare_digest(expected, embedded)


def sign_pending_request(path) -> str:
    """Sign an existing ``pending_deploy.json`` in place using the env key.

    Reads the JSON at ``path``, computes the HMAC-SHA256 ``signature`` over its
    canonical form (excluding any prior ``signature``), writes the signed dict
    back, and returns the signature hex.

    Raises:
        RuntimeError: if ``NA0S_AGENT_APPROVAL_HMAC_KEY`` is not configured.
    """
    key = _approval_key()
    if key is None:
        raise RuntimeError(
            f"{HMAC_KEY_ENV} is not set; cannot sign {path}. "
            "Configure the shared HMAC secret before signing approval requests."
        )
    path = Path(path)
    request = json.loads(path.read_text())
    signature = sign_request(request, key)
    request["signature"] = signature
    path.write_text(json.dumps(request, indent=2))
    return signature

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

    @staticmethod
    def content_hash(request: Dict[str, Any]) -> str:
        """SHA-256 of the canonical (non-volatile) request content.

        Used to pin the artifact at notify time so a TOCTOU mutation of
        ``pending_deploy.json`` between notification and execution is detected.
        Volatile status keys are excluded so a legitimate status update (e.g.
        marking ``approved``) does not look like tampering, while any change to
        the substantive payload (candidate path, gates, command, …) does.
        """
        basis = {k: v for k, v in request.items() if k not in _VOLATILE_KEYS}
        blob = json.dumps(basis, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- state --

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                data.setdefault("notified", [])
                data.setdefault("finalized", [])
                data.setdefault("challenges", {})
                return data
            except Exception as e:
                logger.warning(f"Could not read approvals state, resetting: {e}")
        return {"notified": [], "finalized": [], "challenges": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
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
        # A finalized request can never be approved again: drop its nonce so a
        # leaked/replayed code is useless even before the file is cleaned up.
        state.get("challenges", {}).pop(rid, None)
        self._save_state(state)

    # ----------------------------------------------------------- challenge --

    def issue_challenge(self, request: Dict[str, Any]) -> Dict[str, str]:
        """Mint (or return the existing) per-request approval challenge.

        At notify time we bind a fresh secret ``nonce`` and a ``content_hash``
        of the request to its ``request_id`` and persist them in the state
        file. Only a reply carrying this exact nonce authorizes the deploy, and
        the artifact must still hash to ``content_hash`` when execution runs
        (TOCTOU guard). The nonce is the secret delivered ONLY to the approver's
        device, so a forged/bare ``approve`` on the gateway cannot satisfy it.

        Idempotent: re-notifying the same request returns the same challenge so
        the code the user already holds stays valid.
        """
        state = self._load_state()
        challenges = state.setdefault("challenges", {})
        rid = self.request_id(request)
        existing = challenges.get(rid)
        if existing and existing.get("nonce") and existing.get("content_hash"):
            return existing
        challenge = {
            "nonce": secrets.token_hex(16),
            "content_hash": self.content_hash(request),
        }
        challenges[rid] = challenge
        self._save_state(state)
        return challenge

    def get_challenge(self, request: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Return the persisted ``{nonce, content_hash}`` for a request, if any."""
        challenge = self._load_state().get("challenges", {}).get(self.request_id(request))
        if challenge and challenge.get("nonce") and challenge.get("content_hash"):
            return challenge
        return None

    def consume_challenge(self, request: Dict[str, Any]) -> None:
        """Invalidate a request's nonce so it is strictly single-use."""
        state = self._load_state()
        if state.get("challenges", {}).pop(self.request_id(request), None) is not None:
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
            request = json.loads(show.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Remote approval request is not valid JSON: {e}")
            return None

        if not self._authenticate_request(request):
            return None
        return request

    def _authenticate_request(self, request: Dict[str, Any]) -> bool:
        """Enforce HMAC signing policy on a fetched mail-drop request.

        Policy:
          * If a key is configured (``NA0S_AGENT_APPROVAL_HMAC_KEY`` set) the
            request MUST carry a valid ``signature`` or it is rejected — a
            SECURITY warning is logged and the request is dropped (never
            materialized or notified).
          * If NO key is configured the mail-drop runs in backward-compatible
            unsigned mode: a loud one-time WARNING is logged and the request is
            accepted, so the live flow is not broken before the shared secret is
            deployed to both the cloud producer and this daemon.

        Returns:
            True if the request may proceed; False if it must be rejected.
        """
        global _unsigned_mode_warned
        key = _approval_key()
        if key is None:
            if not _unsigned_mode_warned:
                logger.warning(
                    "SECURITY: mail-drop request signing is DISABLED — %s is "
                    "not set, so unsigned approval requests are accepted. Set "
                    "the shared HMAC secret on the cloud producer AND this "
                    "daemon to enforce request authenticity.",
                    HMAC_KEY_ENV,
                )
                _unsigned_mode_warned = True
            return True

        if not verify_request(request, key):
            reason = (
                "missing signature"
                if not request.get("signature")
                else "signature mismatch"
            )
            logger.warning(
                "SECURITY: rejecting mail-drop approval request (%s); a forged "
                "or tampered pending_deploy.json will not be materialized or "
                "notified.",
                reason,
            )
            return False
        return True

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
