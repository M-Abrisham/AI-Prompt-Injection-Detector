"""Na0S host-integration adapters.

This sub-package exposes Na0S's local detection pipeline as guards for
third-party agent runtimes.  Adapters here are PURE-LOCAL: they run
``na0s.scan()`` / ``na0s.scan_output()`` on tool I/O text and return an
allow/deny decision.  They make NO LLM/API call of their own and never read
``ANTHROPIC_API_KEY``.

Currently provided:

* :mod:`na0s.integrations.agent_sdk` — PreToolUse + PostToolUse hooks for the
  Claude Agent SDK (``anthropics/claude-agent-sdk-python``).  Importing this
  module does NOT require ``claude_agent_sdk`` to be installed.
"""

from na0s.integrations.agent_sdk import (
    make_na0s_pretooluse_hook,
    make_na0s_posttooluse_hook,
    register_na0s_hooks,
    serialize_tool_input,
    serialize_tool_response,
)

__all__ = [
    "make_na0s_pretooluse_hook",
    "make_na0s_posttooluse_hook",
    "register_na0s_hooks",
    "serialize_tool_input",
    "serialize_tool_response",
]
