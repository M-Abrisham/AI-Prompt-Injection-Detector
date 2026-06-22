"""Na0S MCP guard server (Model Context Protocol).

This sub-package exposes Na0S's local detection pipeline as a **FastMCP guard
server**: an MCP server whose tools (``scan_text`` / ``check_tool_call`` /
``check_tool_response``) let an MCP host run Na0S over its prompt text and tool
I/O before/after a tool executes.  It realizes the "embeddable defensive SDK"
positioning at the protocol layer.

PURE-LOCAL by design
--------------------
The guard tools only call ``na0s.scan()`` / ``na0s.scan_output()`` and the local
MCP supply-chain detector.  They make **no** LLM/API call of their own and never
read ``ANTHROPIC_API_KEY``.  The MCP SDK here is the *protocol* layer, not an LLM
client — no credentials are needed to run the guard.

No hard dependency on the MCP SDK
---------------------------------
Importing this package never imports ``mcp`` (the FastMCP SDK) at top level, so
``import na0s.mcp`` works whether or not the SDK is installed.  ``mcp.server.
fastmcp.FastMCP`` is imported lazily (try/except) only inside
:func:`build_guard_server`, mirroring
``na0s.integrations.agent_sdk.register_na0s_hooks``.  Build the server with::

    from na0s.mcp import build_guard_server

    server = build_guard_server()   # raises an actionable ImportError if the
                                    # MCP SDK is not installed (pip install na0s[mcp])
    server.run()                    # FastMCP stdio transport

The factory accepts injected ``scan`` / ``scan_output`` / ``scan_tools`` /
``baseline`` so the guard logic is unit-testable with stubs and no network.
"""

from na0s.mcp.baseline import ToolBaselineStore
from na0s.mcp.server import build_guard_server

__all__ = [
    "build_guard_server",
    "ToolBaselineStore",
]
