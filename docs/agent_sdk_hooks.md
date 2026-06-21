# Na0S guards for the Claude Agent SDK

Na0S can act as **PreToolUse** and **PostToolUse** guards for the
[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python), so an
agent host blocks malicious tool **inputs** before a tool runs and flags/redacts
malicious tool **outputs** after a tool runs — at action time, not just at
prompt-scan time. This is the "embeddable defensive SDK" use case: a host
embeds Na0S and Na0S inspects the agent's tool I/O, which is exactly the
indirect prompt-injection surface Na0S defends.

Adapter module: `na0s.integrations.agent_sdk`.

## No API key required to run the guard

The guard is **pure-local**. It only calls `na0s.scan()` / `na0s.scan_output()`
on the tool I/O text and returns an allow/deny decision. It makes **no** LLM or
API call of its own and never reads `ANTHROPIC_API_KEY`.

The Agent SDK *itself* needs credentials to drive the agent loop (an API key, a
cloud provider such as Bedrock/Vertex/Azure, or — when run via the Claude Code
runtime — your Claude Code subscription session). That is the host's concern and
is independent of Na0S. You can register and run the Na0S guard with zero Na0S
credentials.

The module also never imports `claude_agent_sdk` at import time, so adding the
guard does not make the SDK a hard dependency of Na0S — `import na0s` works
whether or not the SDK is installed.

## Quick start: register both guards

```python
import anyio
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from na0s.integrations.agent_sdk import register_na0s_hooks

async def main():
    options = ClaudeAgentOptions(
        # Guards EVERY tool by default. Pass matcher="Bash" to scope it,
        # or pre_kwargs/post_kwargs to tune (see below).
        hooks=register_na0s_hooks(),
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Read notes.txt and summarize it")
        async for message in client.receive_response():
            print(message)

anyio.run(main)
```

If a tool input is judged malicious, the PreToolUse guard returns the SDK's
`permissionDecision: "deny"` shape and the tool call is blocked; the
`permissionDecisionReason` (label, risk score, techniques) is surfaced to the
model. If a tool output is judged suspicious, the PostToolUse guard appends an
`additionalContext` warning (and optionally replaces the output with a redacted
version), telling the model to treat the output as untrusted data.

## Wire the callables yourself

`register_na0s_hooks()` is a convenience. You can build and place the hooks
manually for finer matcher control:

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from na0s.integrations.agent_sdk import (
    make_na0s_pretooluse_hook,
    make_na0s_posttooluse_hook,
)

options = ClaudeAgentOptions(hooks={
    "PreToolUse":  [HookMatcher(matcher="Bash|Write", hooks=[make_na0s_pretooluse_hook()])],
    "PostToolUse": [HookMatcher(hooks=[make_na0s_posttooluse_hook(redact=True)])],
})
```

`HookMatcher.matcher` is matched against the **tool name** only. `None`/`""`/`"*"`
matches all tools; a pure `[A-Za-z0-9_|]` string is an exact name (with `|`
alternation); anything else is treated as a regex (e.g. `^mcp__` to guard all
MCP tools).

## Tuning

### PreToolUse — `make_na0s_pretooluse_hook(...)`

| Option | Default | Meaning |
| --- | --- | --- |
| `scan` | `na0s.scan` | Local scan function (injectable for tests). |
| `block_threshold` | `None` | `None` ⇒ block on `result.is_malicious` (Na0S's own calibrated `DECISION_THRESHOLD = 0.55`). A float ⇒ host policy `risk_score >= block_threshold`. |
| `scan_fields` | `None` | Only scan these `tool_input` keys (narrows the surface, cuts false positives for hosts that know their tool schemas). Default: scan all argument values. |
| `max_chars` | `64000` | Cap on serialized text fed to `scan()`. |
| `fail_open` | `True` | On a guard error, allow the call (don't brick the agent). Set `False` to fail closed (deny on error) for high-assurance deployments. |

**Why the default delegates to `is_malicious`.** Na0S already calibrates and
applies `DECISION_THRESHOLD` inside `scan()`. Re-thresholding `risk_score` in the
adapter would introduce a *second*, arbitrary cutoff that drifts from the
pipeline's calibration. So by default the adapter trusts `result.is_malicious`
and introduces no magic constant of its own. `block_threshold` exists only for a
host that deliberately wants a stricter/looser, explicitly-owned policy.

### PostToolUse — `make_na0s_posttooluse_hook(...)`

| Option | Default | Meaning |
| --- | --- | --- |
| `scan_output` | `na0s.scan_output` | Output-side detector (`OutputScanResult`: `is_suspicious`, `risk_score`, `flags`, `redacted_text`, `technique_ids`). |
| `block_threshold` | `None` | `None` ⇒ flag on `result.is_suspicious`; float ⇒ `risk_score >= block_threshold`. |
| `redact` | `False` | When suspicious, replace the output with `result.redacted_text` via `updatedToolOutput` (the model never sees the raw payload). When `False`, the raw output is left intact and only annotated. |
| `sensitivity` | `"medium"` | Forwarded to `na0s.scan_output(..., sensitivity=...)`. |
| `fail_open` | `True` | On error, leave the output untouched. |

**PostToolUse cannot block.** The tool has already executed by the time the
PostToolUse hook runs, so the SDK does not allow `permissionDecision: deny`
there. The guard therefore *flags* a suspicious output (via `additionalContext`)
and can *redact* it (via `updatedToolOutput`) — it cannot un-run the tool. Use
PreToolUse to actually block dangerous calls.

## How tool I/O is serialized

- **PreToolUse:** the guard flattens the tool argument **values** (the
  attacker-controlled surface) into one string, with high-signal fields
  (`command`, `content`, `query`, `url`, `prompt`, `file_path`, ...) first and
  the tool name as a low-weight `[tool:NAME]` prefix. `scan_fields` narrows this.
- **PostToolUse:** the `tool_response` (dict, string, or Anthropic content
  blocks) is coerced to text and passed to `scan_output()`.

Both are truncated to `max_chars` (default 64 KB) since the detector's signal is
local.

## Security notes

- Tool inputs and outputs are treated as **untrusted data**, never as
  instructions to the guard. This is the indirect-injection surface Na0S exists
  to defend.
- Hooks are exception-safe: a scan error never propagates into the agent loop
  (the SDK warns that an unhandled hook exception can interrupt the agent). The
  failure mode is configurable (`fail_open`).

## Testing

The adapter is unit-tested entirely with mocked SDK payload dicts and stubbed
scan functions — no real SDK, no network, no API key:

```
PYTHONPATH=src python3 -m pytest tests/integrations/ -q --tb=short
```
