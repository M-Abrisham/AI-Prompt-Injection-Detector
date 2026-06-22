"""FastMCP guard server factory for Na0S.

:func:`build_guard_server` returns a FastMCP server exposing three guard tools an
MCP host can call to run Na0S's local detection pipeline over its prompt text and
tool I/O:

* ``scan_text(text)`` — run ``na0s.scan()`` over free text (a prompt, a RAG
  chunk, a retrieved document) and return an allow/deny decision.
* ``check_tool_call(tool_name, tool_input, tools=None)`` — run ``na0s.scan()``
  over the serialized tool arguments AND, behind the ``_HAS_MCP_SUPPLY_CHAIN``
  flag, run the MCP supply-chain detector (tool poisoning / rug-pull / typosquat)
  over the advertised tool manifest.  Returns an allow/deny decision in the same
  shape ``na0s.integrations.agent_sdk`` uses.
* ``check_tool_response(tool_response)`` — run ``na0s.scan_output()`` over a
  tool's result to flag prompt-injection / exfiltration smuggled back through
  untrusted tool output.

PURE-LOCAL / NO API KEY
-----------------------
Every guard tool calls only ``na0s.scan`` / ``na0s.scan_output`` /
``na0s.predict.scan_tools`` and the local supply-chain detector.  No LLM/API call
is made by the guard; ``ANTHROPIC_API_KEY`` is never read.  The MCP SDK is the
protocol transport, not an LLM client.

Optional MCP SDK
----------------
``mcp.server.fastmcp.FastMCP`` is imported lazily inside the factory under
``try/except ImportError`` with an actionable ``pip install na0s[mcp]`` message,
mirroring ``na0s.integrations.agent_sdk.register_na0s_hooks`` (:517-520).  This
module is therefore importable WITHOUT the MCP SDK installed — only *calling*
:func:`build_guard_server` requires it.

Threshold discipline
---------------------
The block decision delegates to Na0S's already-calibrated
``DECISION_THRESHOLD`` (predict.py) by default — no new magic cutoff is invented.
A host may pass ``block_threshold`` for an explicit, host-chosen policy.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

# Same-package + detectors-tier imports only.  These do NOT import the MCP SDK or
# top-level ``na0s`` at module top, so ``import na0s.mcp.server`` is SDK-free.
from na0s.integrations.agent_sdk import (
    serialize_tool_input,
    serialize_tool_response,
)
from na0s.mcp.baseline import ToolBaselineStore

logger = logging.getLogger("na0s.mcp.server")

__all__ = ["build_guard_server"]


# The MCP supply-chain detector (tool poisoning / rug-pull / typosquat) is an
# optional, local detector.  Wire it behind a feature flag exactly like
# predict.py wires _HAS_MCP_TOOL_DETECTOR, so the guard degrades gracefully if
# the detector module is unavailable.
try:
    from na0s.detectors.mcp_supply_chain import (
        get_mcp_supply_chain_weight as _get_mcp_supply_chain_weight,
        scan_tool_supply_chain as _scan_tool_supply_chain,
    )

    _HAS_MCP_SUPPLY_CHAIN = True
except ImportError:  # pragma: no cover - detector ships with the package
    _HAS_MCP_SUPPLY_CHAIN = False


def _resolve_threshold() -> float:
    """Na0S's calibrated DECISION_THRESHOLD (lazy — no top-level na0s import)."""
    from na0s.predict import DECISION_THRESHOLD

    return DECISION_THRESHOLD


def _allow() -> Dict[str, Any]:
    """The allow decision (mirrors the agent_sdk PreToolUse allow == {})."""
    return {"decision": "allow", "is_malicious": False, "risk_score": 0.0}


def build_guard_server(
    scan: Optional[Callable[..., Any]] = None,
    scan_output: Optional[Callable[..., Any]] = None,
    scan_tools: Optional[Callable[..., Any]] = None,
    baseline: Optional[ToolBaselineStore] = None,
    fail_open: bool = True,
    block_threshold: Optional[float] = None,
    name: str = "na0s-guard",
):
    """Build a FastMCP guard server backed by Na0S's local pipeline.

    Parameters
    ----------
    scan
        Free-text / tool-input scan function.  Defaults to :func:`na0s.scan`
        (imported lazily so building the server needs no model preload and the
        module stays SDK-free).  Inject a stub in tests.
    scan_output
        Tool-output scan function.  Defaults to :func:`na0s.scan_output` (lazy).
    scan_tools
        MCP manifest scanner.  Defaults to :func:`na0s.predict.scan_tools`
        (lazy) — the T1.x tool-manifest seam.  NOTE: ``scan_tools`` is NOT in
        ``na0s.__all__``; it is imported from ``na0s.predict`` on demand.
    baseline
        Optional :class:`~na0s.mcp.baseline.ToolBaselineStore` of approved tool
        descriptions, used by the supply-chain detector for rug-pull detection.
        When ``None``, rug-pull is skipped (poisoning + typosquat still run).
    fail_open
        On an unexpected error inside a guard tool, ALLOW (return the allow
        shape) and log, rather than denying.  Mirrors the agent_sdk hook
        contract.  Set ``False`` to fail CLOSED (deny on guard error) for
        higher-assurance deployments.
    block_threshold
        ``None`` (default): the block decision is the pipeline's own
        ``is_malicious`` (i.e. the calibrated ``DECISION_THRESHOLD`` is the
        single source of truth — no new magic constant).  Pass a float in
        ``[0, 1]`` for an explicit host-chosen ``risk_score >= block_threshold``
        policy.

    Returns
    -------
    FastMCP
        A server with ``scan_text`` / ``check_tool_call`` / ``check_tool_response``
        registered as MCP tools.  Run it with ``server.run()``.

    Raises
    ------
    ImportError
        If the MCP SDK (``pip install na0s[mcp]``) is not installed.  The rest of
        this module remains importable and the guard logic is testable without
        the SDK by inspecting the registered callables.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # lazy, optional
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "build_guard_server() needs the MCP SDK (FastMCP) to build the "
            "guard server: pip install na0s[mcp]. The guard logic itself is "
            "importable and testable without the SDK."
        ) from exc

    mcp = FastMCP(name)

    # Resolve the block threshold once (None => delegate to is_malicious).
    _threshold = block_threshold

    def _blocked(result: Any) -> bool:
        """Block iff is_malicious (default) or risk_score >= host threshold."""
        if _threshold is None:
            return bool(getattr(result, "is_malicious", False))
        return float(getattr(result, "risk_score", 0.0)) >= _threshold

    def _scan_fn() -> Callable[..., Any]:
        if scan is not None:
            return scan
        from na0s import scan as _scan  # lazy: no model load at import

        return _scan

    def _scan_output_fn() -> Callable[..., Any]:
        if scan_output is not None:
            return scan_output
        from na0s import scan_output as _scan_output  # lazy

        return _scan_output

    def _scan_tools_fn() -> Callable[..., Any]:
        if scan_tools is not None:
            return scan_tools
        # scan_tools is NOT in na0s.__all__ — import from na0s.predict (the
        # documented manifest seam), not the top-level package.
        from na0s.predict import scan_tools as _scan_tools

        return _scan_tools

    # -- scan_text ---------------------------------------------------------

    @mcp.tool()
    def scan_text(text: str) -> Dict[str, Any]:
        """Scan free text (prompt / RAG chunk / retrieved doc) for prompt injection.

        Returns an allow/deny decision: ``{"decision": "allow"|"deny",
        "is_malicious": bool, "risk_score": float, "label": str,
        "techniques": [...], "rules": [...]}``.  Treats ``text`` as untrusted
        DATA — it is scanned, never executed.
        """
        try:
            if not text or not str(text).strip():
                return _allow()
            result = _scan_fn()(str(text))
            blocked = _blocked(result)
            return {
                "decision": "deny" if blocked else "allow",
                "is_malicious": bool(blocked),
                "risk_score": float(getattr(result, "risk_score", 0.0)),
                "label": getattr(result, "label", "safe"),
                "techniques": list(getattr(result, "technique_tags", []) or [])[:10],
                "rules": [str(h) for h in (getattr(result, "rule_hits", []) or [])][:10],
            }
        except Exception:  # noqa: BLE001 - a guard must never crash the host
            logger.exception(
                "na0s scan_text guard errored; failing %s",
                "open" if fail_open else "closed",
            )
            if fail_open:
                return _allow()
            return {
                "decision": "deny",
                "is_malicious": True,
                "risk_score": 1.0,
                "label": "blocked",
                "reason": "Na0S guard error; denying by fail-closed policy.",
            }

    # -- check_tool_call ---------------------------------------------------

    @mcp.tool()
    def check_tool_call(
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Inspect an MCP tool call before it runs (allow/deny).

        Two complementary checks:

        1. **Argument injection** — serialize ``tool_name`` + ``tool_input`` and
           run ``na0s.scan()`` over the attacker-controllable argument values.
        2. **Supply-chain** (behind ``_HAS_MCP_SUPPLY_CHAIN``) — run the MCP
           supply-chain detector over the advertised ``tools`` manifest (tool
           poisoning, typosquat) and, when an approved ``baseline`` is set,
           rug-pull.  Also runs ``scan_tools`` for the T1.x manifest signal.

        The call is DENIED if the argument scan is malicious OR the supply-chain
        detector judges the advertised manifest malicious.  Two distinct uses of
        the supply-chain signal, by design:

        * ``supply_chain_weight`` is the CAPPED (<=0.30) fusion contribution —
          the soft, corroborating value that ``predict.py`` would add to a
          *free-text* composite (a lone hit must never dominate that composite).
        * The manifest DENY decision uses the detector's OWN ``risk_score`` (a
          dedicated manifest check, where the supply-chain detector IS the
          primary signal), denying when it reaches ``DECISION_THRESHOLD`` — i.e.
          the detector's high/critical severity.  This is the same calibrated
          threshold, NOT a new magic constant.  The capped weight is ALSO added
          to ``arg_risk`` so a near-miss argument scan plus a manifest hit can
          corroborate over the line.

        Returns the agent_sdk-style allow/deny decision dict.
        """
        try:
            tool_input = tool_input or {}
            manifest = tools or []

            # ---- (1) argument-injection scan over the serialized tool input ---
            arg_risk = 0.0
            arg_blocked = False
            arg_result = None
            text = serialize_tool_input(tool_name or "", tool_input)
            stripped = text.strip()
            if tool_name:
                stripped = stripped.replace("[tool:{}]".format(tool_name), "", 1).strip()
            if stripped:
                arg_result = _scan_fn()(text)
                arg_risk = float(getattr(arg_result, "risk_score", 0.0))
                arg_blocked = _blocked(arg_result)

            # ---- (2) supply-chain + manifest scan over the advertised tools ---
            sc_weight = 0.0
            sc_native_risk = 0.0
            sc_techniques: List[str] = []
            sc_indicators: List[str] = []
            manifest_techniques: List[str] = []
            if manifest:
                threshold = (
                    _threshold if _threshold is not None else _resolve_threshold()
                )

                # T1.x manifest seam (na0s.predict.scan_tools).
                try:
                    known = baseline.known_names() if baseline else None
                    manifest_results = _scan_tools_fn()(manifest, known_tools=known)
                    for mr in manifest_results or []:
                        manifest_techniques.extend(
                            getattr(mr, "technique_ids", []) or []
                        )
                except Exception:  # noqa: BLE001 - manifest scan is best-effort
                    logger.exception("na0s scan_tools manifest seam errored")

                # Supply-chain lifecycle detector (poisoning / rug-pull / typosquat).
                if _HAS_MCP_SUPPLY_CHAIN:
                    bl = baseline.as_baseline() if baseline else None
                    known_names = baseline.known_names() if baseline else None
                    sc_results = _scan_tool_supply_chain(
                        manifest, baseline=bl, known_names=known_names
                    )
                    for sc in sc_results or []:
                        w = _get_mcp_supply_chain_weight(sc)
                        if w > sc_weight:
                            sc_weight = w
                        if sc.risk_score > sc_native_risk:
                            sc_native_risk = sc.risk_score
                        if sc.risk_score > 0.0:
                            sc_techniques.extend(getattr(sc, "technique_ids", []) or [])
                            sc_indicators.extend(
                                getattr(sc, "risk_indicators", []) or []
                            )

                # DENY on the detector's own manifest verdict (it is the primary
                # signal for a dedicated manifest check), OR on the corroborated
                # additive floor (capped weight on top of argument risk).
                manifest_risk = max(sc_native_risk, arg_risk + sc_weight)
                sc_blocked = manifest_risk >= threshold
            else:
                sc_blocked = False
                manifest_risk = arg_risk

            blocked = bool(arg_blocked or sc_blocked)
            risk_score = max(arg_risk, manifest_risk if manifest else arg_risk)

            techniques = []
            for t in list(getattr(arg_result, "technique_tags", []) or []) \
                    + manifest_techniques + sc_techniques:
                if t not in techniques:
                    techniques.append(t)

            reasons: List[str] = []
            if arg_blocked:
                reasons.append(
                    "argument-injection (risk={:.2f})".format(arg_risk)
                )
            if manifest and sc_blocked and not arg_blocked:
                reasons.append(
                    "supply-chain (weight={:.2f}): {}".format(
                        sc_weight, "; ".join(sc_indicators[:3])
                    )
                )

            return {
                "decision": "deny" if blocked else "allow",
                "is_malicious": blocked,
                "risk_score": float(min(risk_score, 1.0)),
                "techniques": techniques[:10],
                "supply_chain_weight": round(sc_weight, 4),
                "indicators": sc_indicators[:10],
                "reason": "; ".join(reasons) if reasons else "",
            }
        except Exception:  # noqa: BLE001 - never crash the host
            logger.exception(
                "na0s check_tool_call guard errored; failing %s",
                "open" if fail_open else "closed",
            )
            if fail_open:
                return _allow()
            return {
                "decision": "deny",
                "is_malicious": True,
                "risk_score": 1.0,
                "reason": "Na0S guard error; denying by fail-closed policy.",
            }

    # -- check_tool_response -----------------------------------------------

    @mcp.tool()
    def check_tool_response(tool_response: Any) -> Dict[str, Any]:
        """Scan a tool's RESULT for injection / exfiltration in untrusted output.

        A tool's result is untrusted data (a web page, a file read, an MCP
        payload) that may smuggle a prompt-injection or data-exfiltration
        attempt back to the model.  This runs ``na0s.scan_output()`` over it and
        flags it; it cannot un-run the tool, so the decision is advisory
        (``flagged``) plus the redacted text the host may substitute.
        """
        try:
            text = serialize_tool_response(tool_response)
            if not text.strip():
                return {"decision": "allow", "is_suspicious": False, "risk_score": 0.0}
            result = _scan_output_fn()(text)
            if _threshold is None:
                flagged = bool(getattr(result, "is_suspicious", False))
            else:
                flagged = float(getattr(result, "risk_score", 0.0)) >= _threshold
            return {
                "decision": "deny" if flagged else "allow",
                "is_suspicious": bool(flagged),
                "risk_score": float(getattr(result, "risk_score", 0.0)),
                "flags": list(getattr(result, "flags", []) or [])[:10],
                "techniques": list(getattr(result, "technique_ids", []) or [])[:10],
                "redacted_text": getattr(result, "redacted_text", "") or "",
            }
        except Exception:  # noqa: BLE001 - never crash the host
            logger.exception(
                "na0s check_tool_response guard errored; failing %s",
                "open" if fail_open else "closed",
            )
            if fail_open:
                return {"decision": "allow", "is_suspicious": False, "risk_score": 0.0}
            return {
                "decision": "deny",
                "is_suspicious": True,
                "risk_score": 1.0,
                "reason": "Na0S guard error; denying by fail-closed policy.",
            }

    # Expose the underlying callables for unit tests (the FastMCP @tool decorator
    # may wrap them, so tests can reach the raw guard logic without the SDK's
    # transport).  These are plain functions; calling them runs the guard.
    mcp._na0s_scan_text = scan_text  # type: ignore[attr-defined]
    mcp._na0s_check_tool_call = check_tool_call  # type: ignore[attr-defined]
    mcp._na0s_check_tool_response = check_tool_response  # type: ignore[attr-defined]

    return mcp
