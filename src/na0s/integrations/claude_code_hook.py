"""Na0S guard as a **Claude Code native hook** (keyless — runs on your subscription).

This is the *subscription* sibling of :mod:`na0s.integrations.agent_sdk`.  The
Agent SDK adapter wires Na0S into a programmatic ``ClaudeAgentOptions(hooks=...)``
host (which needs the SDK's own auth).  This module instead wires Na0S into
**Claude Code's own** ``settings.json`` hooks, which Claude Code runs LOCALLY as
shell commands on the user's machine — so they need **NO API key**, only a
Claude Code subscription.

How Claude Code native hooks work (verified against the official docs)
---------------------------------------------------------------------
Sources (fetched 2026-06-20):
  * Hooks reference: https://code.claude.com/docs/en/hooks
  * Hooks guide:     https://code.claude.com/docs/en/hooks-guide

A ``"type": "command"`` hook is a plain local process Claude Code spawns; it
talks purely over **stdin / stdout / stderr / exit code** and makes no Anthropic
API call.  Claude Code passes the hook a JSON object on **stdin**.  For a
``PreToolUse`` hook the load-bearing fields are::

    {
      "session_id": "abc123",
      "cwd": "/Users/me/project",
      "hook_event_name": "PreToolUse",
      "tool_name": "Bash",
      "tool_input": { "command": "rm -rf /tmp/build" }
    }

and for ``PostToolUse`` the same plus a ``tool_response`` (the tool result).

This bridge reads that JSON, maps ``tool_name`` + ``tool_input`` 1:1 onto the
REUSED :func:`na0s.integrations.agent_sdk.serialize_tool_input` (and
``tool_response`` onto :func:`serialize_tool_response`), runs ``na0s.scan()`` /
``na0s.scan_output()`` locally, and emits Claude Code's decision shape.

Decision shapes (verified)
--------------------------
PreToolUse uses ``hookSpecificOutput`` (NOT the legacy top-level
``decision: "block"`` pattern, which is for ``UserPromptSubmit``/``PostToolUse``/
``Stop``).  To deny::

    {
      "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "..."
      }
    }

To allow: exit 0 with no JSON (or ``{}``).

Important docs rule: **do not mix channels.**  If the hook exits 2, Claude Code
treats it as a blocking error, reads the *stderr* text as the reason, and
**ignores any stdout JSON**.  So we pick ONE channel: print the structured
``hookSpecificOutput`` deny JSON to stdout and exit 0 (the machine-parsed,
recommended path).  The exit-2 + stderr path is only a *fallback* used when JSON
cannot be emitted (e.g. fail-closed on a serialization error).

PostToolUse cannot deny (the tool already ran).  Per the docs it can inject
context the model then sees; this bridge emits an ``additionalContext`` warning
via ``hookSpecificOutput`` so the model treats a suspicious tool output as
untrusted data.

Keyless guarantee
-----------------
This module never imports ``claude_agent_sdk`` and never reads
``ANTHROPIC_API_KEY``.  It only calls Na0S's local ``scan`` / ``scan_output``.
That is the whole point: it runs inside Claude Code's local hook runtime on the
user's subscription, with no credentials of its own.

Wire it (``.claude/settings.json``)::

    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Bash|Write|Edit",
            "hooks": [
              { "type": "command",
                "command": "python -m na0s.integrations.claude_code_hook" }
            ]
          }
        ]
      }
    }

Run directly::

    python -m na0s.integrations.claude_code_hook            # fail-OPEN (default)
    python -m na0s.integrations.claude_code_hook --fail-closed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Callable, Dict, Optional, TextIO

from na0s.integrations.agent_sdk import (
    _format_output_warning,
    _format_reason,
    serialize_tool_input,
    serialize_tool_response,
)

__all__ = [
    "build_pretooluse_decision",
    "build_posttooluse_decision",
    "process_hook_event",
    "main",
]

logger = logging.getLogger("na0s.integrations.claude_code_hook")


# ---------------------------------------------------------------------------
# Pure decision builders (no I/O — unit-testable with plain dicts)
# ---------------------------------------------------------------------------

def build_pretooluse_decision(
    event: Dict[str, Any],
    scan: Optional[Callable[..., Any]] = None,
    block_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Decide a PreToolUse event.

    Returns the Claude Code ``hookSpecificOutput`` DENY dict when the tool call
    is judged malicious, else ``None`` (caller emits nothing / allows).

    ``scan`` defaults to :func:`na0s.scan` (lazy import, so importing this
    module loads no model and needs no key).  ``block_threshold=None`` delegates
    the decision to Na0S's own calibrated ``result.is_malicious`` (no second,
    arbitrary cutoff — same rationale as the Agent SDK adapter); a float gates on
    ``result.risk_score >= block_threshold`` (an explicit, host-chosen policy).
    """
    event_name = event.get("hook_event_name", "PreToolUse")
    tool_name = event.get("tool_name", "") or ""
    tool_input = event.get("tool_input", {}) or {}

    text = serialize_tool_input(tool_name, tool_input)
    # The tool-name prefix alone is no attacker-controlled surface -> allow.
    stripped = text.strip()
    if tool_name:
        stripped = stripped.replace(f"[tool:{tool_name}]", "", 1).strip()
    if not stripped:
        return None

    fn = scan
    if fn is None:
        from na0s import scan as _scan  # lazy: no SDK / no model / no key at import
        fn = _scan
    result = fn(text)

    if block_threshold is None:
        blocked = bool(getattr(result, "is_malicious", False))
    else:
        blocked = float(getattr(result, "risk_score", 0.0)) >= block_threshold
    if not blocked:
        return None

    reason = _format_reason(
        getattr(result, "label", "malicious"),
        float(getattr(result, "risk_score", 0.0)),
        getattr(result, "technique_tags", None),
        getattr(result, "rule_hits", None),
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }


def build_posttooluse_decision(
    event: Dict[str, Any],
    scan_output: Optional[Callable[..., Any]] = None,
    block_threshold: Optional[float] = None,
    sensitivity: str = "medium",
) -> Optional[Dict[str, Any]]:
    """Decide a PostToolUse event.

    PostToolUse cannot deny (the tool already ran), so on a suspicious
    ``tool_response`` it returns a ``hookSpecificOutput`` carrying an
    ``additionalContext`` warning that tells the model to treat the output as
    untrusted data; else ``None`` (no annotation).
    """
    event_name = event.get("hook_event_name", "PostToolUse")
    tool_response = event.get("tool_response")

    text = serialize_tool_response(tool_response)
    if not text.strip():
        return None

    fn = scan_output
    if fn is None:
        from na0s import scan_output as _scan_output  # lazy
        fn = _scan_output
    result = fn(text, sensitivity=sensitivity)

    if block_threshold is None:
        flagged = bool(getattr(result, "is_suspicious", False))
    else:
        flagged = float(getattr(result, "risk_score", 0.0)) >= block_threshold
    if not flagged:
        return None

    warning = _format_output_warning(
        float(getattr(result, "risk_score", 0.0)),
        getattr(result, "flags", None),
        getattr(result, "technique_ids", None),
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": warning,
        },
    }


def process_hook_event(
    event: Dict[str, Any],
    scan: Optional[Callable[..., Any]] = None,
    scan_output: Optional[Callable[..., Any]] = None,
    block_threshold: Optional[float] = None,
    sensitivity: str = "medium",
) -> Optional[Dict[str, Any]]:
    """Route a parsed hook event by ``hook_event_name`` to the right builder.

    Returns the decision dict to print, or ``None`` to allow with no output.
    Unknown / unhandled events return ``None`` (allow).
    """
    name = event.get("hook_event_name", "PreToolUse")
    if name == "PreToolUse":
        return build_pretooluse_decision(
            event, scan=scan, block_threshold=block_threshold
        )
    if name == "PostToolUse":
        return build_posttooluse_decision(
            event, scan_output=scan_output, block_threshold=block_threshold,
            sensitivity=sensitivity,
        )
    return None


# ---------------------------------------------------------------------------
# CLI entry point (the actual stdin/stdout/exit-code bridge)
# ---------------------------------------------------------------------------

# Exit codes per the Claude Code docs.  Exit 0 = no objection (we emit our
# decision as stdout JSON).  Exit 2 = blocking error; Claude Code reads stderr
# as the reason and IGNORES stdout — used here ONLY as the fail-closed fallback.
_EXIT_OK = 0
_EXIT_BLOCK = 2


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="na0s.integrations.claude_code_hook",
        description=(
            "Na0S guard as a Claude Code native (keyless) hook. Reads the hook "
            "event JSON on stdin and emits a PreToolUse deny / PostToolUse "
            "warning decision on stdout. Runs na0s.scan() locally; no API key."
        ),
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help=(
            "On an internal error (bad JSON, scan crash) DENY the tool call "
            "instead of the default fail-OPEN (allow). Use for high-assurance "
            "setups where a guard error should not silently let a call through."
        ),
    )
    parser.add_argument(
        "--block-threshold",
        type=float,
        default=None,
        help=(
            "Optional explicit risk-score cutoff in [0,1]. Omit to delegate to "
            "Na0S's own calibrated is_malicious / is_suspicious decision."
        ),
    )
    parser.add_argument(
        "--sensitivity",
        default="medium",
        help="Output-scan sensitivity forwarded to scan_output (PostToolUse).",
    )
    return parser.parse_args(argv)


def main(
    argv=None,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
    scan: Optional[Callable[..., Any]] = None,
    scan_output: Optional[Callable[..., Any]] = None,
) -> int:
    """Run the bridge once over one hook event from stdin. Returns an exit code.

    Designed for both ``python -m`` use (defaults read ``sys.stdin`` and write
    ``sys.stdout``) and tests (inject ``stdin``/``stdout`` ``StringIO`` and stub
    ``scan``/``scan_output`` — no network, no SDK, no key).

    Failure policy: by default the bridge **fails OPEN** — any internal error
    (malformed stdin JSON, a scan crash) results in *allow* (exit 0, no output)
    so a guard bug never bricks the user's Claude Code session.  ``--fail-closed``
    flips this to deny-on-error via the docs' exit-2 + stderr fallback channel.
    """
    # argv=None -> read process args (the `python -m` path); tests pass an
    # explicit list (often []) so they never pick up the test runner's argv.
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        raw = stdin.read()
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise ValueError("hook event JSON must be an object")

        decision = process_hook_event(
            event,
            scan=scan,
            scan_output=scan_output,
            block_threshold=args.block_threshold,
            sensitivity=args.sensitivity,
        )
        if decision is not None:
            # Single channel: structured JSON on stdout, exit 0. (Per docs, an
            # exit-2 would make Claude Code IGNORE this stdout JSON, so we never
            # mix them on the success path.)
            json.dump(decision, stdout)
            stdout.write("\n")
        return _EXIT_OK
    except Exception as exc:  # noqa: BLE001 — a guard must never crash the session
        logger.exception(
            "na0s Claude Code hook errored; failing %s",
            "closed" if args.fail_closed else "open",
        )
        if args.fail_closed:
            # Fallback deny channel: reason on stderr + exit 2 (NOT stdout JSON,
            # which the docs say is ignored when exit code is 2).
            stderr.write(f"Na0S guard error; denying by --fail-closed policy: {exc}\n")
            return _EXIT_BLOCK
        # Fail OPEN: allow, emit nothing.
        return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess smoke test
    raise SystemExit(main())
