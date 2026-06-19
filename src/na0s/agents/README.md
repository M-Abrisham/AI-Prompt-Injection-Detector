# Na0S Agent Orchestration System

Intelligent automation for the Na0S pipeline using Claude Agents + OpenClaw iMessage integration.

## Architecture

```
GitHub Actions (auto-retrain.yml)
    ↓
    Runs pipeline, writes gates JSONs
    ↓
    Writes pending_deploy.json (instead of auto-deploy)
    Writes failure_reports/ if gates fail
    ↓
Claude Agent (local Mac, via launchd)
    ├── gate_analyzer: Diagnose gate failures
    ├── quarantine_reviewer: Review stale quarantine entries
    ├── deploy_approver: Wait for user approval before deploy
    ├── synthetic_validator: QA generated samples
    └── orchestrator: Coordinate all agents
    ↓
OpenClaw Gateway (local iMessage bridge)
    ↓
iMessage → User's phone/Mac
```

## Agent Modules

### 1. GateAnalyzer
Reads and analyzes gate evaluation results with optional Claude API integration:
- **Input**: `data/canary/canary_results.json`, `models/shadow_results.json`, `models/f14_gate_results.json`
- **Output**: Failure diagnosis + iMessage alert (with root cause analysis if Claude enabled)
- **Actions**: Flags which gates failed and why; uses Claude API to identify root causes and suggest fixes
- **Claude API**: Calls Claude API for intelligent analysis when gates fail (cached to avoid redundant calls)

### 2. QuarantineReviewer
Reviews quarantine backlog for stale/pending entries:
- **Input**: `data/quarantine/*/metadata.json`
- **Output**: Stale entries flagged, recommendations for promote/reject
- **Actions**: Samples rows for human review, sends iMessage

### 3. DeployApprover
Waits for user approval before deploying:
- **Input**: `data/approval_queue/pending_deploy.json`
- **Output**: Approval/rejection decision
- **Actions**: Executes `deploy_model.py` on approval

### 4. SyntheticValidator
QAs synthetic data quality:
- **Input**: `data/holdout/malicious_holdout.jsonl`
- **Output**: Flags low-quality samples
- **Actions**: Removes flagged samples before training

### 5. PipelineOrchestrator
Coordinates all agents:
- **Input**: Pipeline state
- **Output**: Complete automation workflow
- **Actions**: Runs agents in sequence or continuous loop

## Security activation

The deploy-approval path ships hardened but defaults to a permissive mode for
backward compatibility. Turn on the two protections below on the **local
daemon** (and, for signing, the cloud producer) before relying on it.

The defenses are layered. The **per-request nonce** and the **mail-drop HMAC
signature** are the PRIMARY gates that actually authorize a deploy. The **sender
allowlist** is defense-in-depth only — an iMessage handle is spoofable, so it
never bypasses the nonce; it just adds a fail-closed check in front of it.

### Mail-drop request signing (primary)

GitHub Actions (the cloud producer) signs each `pending_deploy.json` with an
HMAC key; the local daemon verifies it. Until the key is set on BOTH ends,
signing runs in **warn-and-accept** mode (a SECURITY warning is logged but the
request is not dropped), so set it to actually enforce authenticity.

```bash
# 1. Generate a shared key (32 random bytes, hex).
openssl rand -hex 32

# 2. Cloud producer — store it as a GitHub Actions secret.
gh secret set NA0S_AGENT_APPROVAL_HMAC_KEY

# 3. Local daemon — export the SAME value where the daemon runs.
export NA0S_AGENT_APPROVAL_HMAC_KEY="<the-same-hex-key>"
```

Both sides must hold the identical key. A mismatched or missing key means
unsigned/unverifiable requests; once both are set, unsigned requests are
dropped instead of accepted.

### Sender allowlist (defense-in-depth)

Restrict which iMessage handles may authorize a deploy. Set a comma-separated
list of phone numbers / emails on the **local daemon**:

```bash
export NA0S_AGENT_APPROVAL_ALLOWED_SENDERS="+15551234567,user@icloud.com"
```

When set, a reply whose sender is missing or not on the list is **rejected
fail-closed** before the nonce is even checked. An allowlisted sender still has
to supply the correct `approve <nonce>` — the allowlist cannot substitute for
the nonce. Leaving the variable empty (the default) disables this gate and
preserves nonce-only behavior. Matching is case- and whitespace-insensitive.

## Installation

1. Install dependencies:
```bash
pip install anthropic>=0.50.0 requests>=2.31
```

2. Set up Anthropic API key:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

3. Optional: Install real OpenClaw for production iMessage delivery:
```bash
# Download from https://openclaw.app (macOS app)
# Once installed, copy to /Applications/OpenClaw.app
```

4. For testing/demo: Use mock OpenClaw (included):
```bash
python scripts/mock_openclaw.py  # Runs on localhost:3000
```

## Usage

### Run individual agents:
```bash
# Gate analysis
python scripts/run_agent.py --gate-analyzer

# Quarantine review
python scripts/run_agent.py --quarantine-review

# Deployment approval
python scripts/run_agent.py --deploy-approval

# Synthetic validation
python scripts/run_agent.py --synthetic-validation
```

### Run full pipeline:
```bash
python scripts/run_agent.py --full-pipeline
```

### Run continuous monitoring (via launchd):
```bash
python scripts/run_agent.py --continuous --poll-interval 300
```

## OpenClaw iMessage Bridge

The `OpenClawBridge` class handles intelligent switching between real OpenClaw (production) and mock OpenClaw (testing) with automatic fallback.

### Three Operation Modes

**1. Auto Mode (Default)**
```python
from na0s.agents.openclaw_bridge import OpenClawBridge

# Auto-detect available OpenClaw
bridge = OpenClawBridge(mode="auto")
# → Uses real OpenClaw if running/installed
# → Falls back to mock if real unavailable
```

**2. Mock Mode (Testing)**
```python
# Always use mock (for CI/testing)
bridge = OpenClawBridge(mode="mock")
# → Requires mock server running on localhost:3000
```

**3. Real Mode (Production)**
```python
# Use only real OpenClaw (error if unavailable)
bridge = OpenClawBridge(mode="real")
# → Fails if OpenClaw not installed/running
```

### Configuration

Via environment variables:
```bash
# Operation mode: "auto" (default), "mock", "real"
export OPENCLAW_MODE="auto"

# API endpoint (default: http://localhost:3000)
export OPENCLAW_BASE_URL="http://localhost:3000"

# HTTP timeout in seconds (default: 30)
export OPENCLAW_TIMEOUT="30"

# Port overrides
export OPENCLAW_REAL_PORT="3000"
export OPENCLAW_MOCK_PORT="3000"
```

Or in code:
```python
bridge = OpenClawBridge(
    base_url="http://localhost:3000",
    timeout=30,
    mode="auto"
)
```

### How Mode Detection Works

**Auto Mode Process:**
1. Try HTTP health check: `GET http://localhost:3000/health`
2. If responds → Use real OpenClaw
3. If fails → Check if app installed: `/Applications/OpenClaw.app`
4. If not → Fallback to mock
5. If both fail → Error in "real" mode, fallback in "auto"

### Sending Messages

```python
# Send message (uses active mode: real or mock)
success = bridge.send_message("🚨 Deployment approved! Running now...")

# Send and wait for reply
reply = bridge.poll_replies(timeout=300)  # 5 min wait

# Send with approval workflow
approval = bridge.send_message_with_approval(
    "Deploy production model?",
    timeout=600,
    expected_replies=["approve", "reject"]
)
```

### Troubleshooting

**iMessages not arriving?**
- Check if real OpenClaw is running: `curl http://localhost:3000/health`
- Check if app is installed: `ls /Applications/OpenClaw.app`
- Check logs: `bridge.active_mode` tells you which mode is active
- Fall back to mock for testing: `OPENCLAW_MODE=mock python script.py`

**Error: "OpenClaw mode set to 'real' but real OpenClaw not available"**
- Either install real OpenClaw app, OR
- Switch to auto mode: `OPENCLAW_MODE=auto`
- Or use mock for testing: `OPENCLAW_MODE=mock`

**Switching modes at runtime**
```python
# Create bridge in auto mode (tries real, falls back to mock)
bridge = OpenClawBridge(mode="auto")

# Logs which mode is active
print(bridge.active_mode)  # "real" or "mock"
```

## Setup Daemon (macOS)

1. Install launchd service:
```bash
mkdir -p ~/Library/LaunchAgents
cp launch/na0s-agent.plist ~/Library/LaunchAgents/com.na0s.agent.plist
launchctl load ~/Library/LaunchAgents/com.na0s.agent.plist
```

2. View logs:
```bash
tail -f /var/log/na0s-agent.log
tail -f /var/log/na0s-agent-error.log
```

3. Stop daemon:
```bash
launchctl unload ~/Library/LaunchAgents/com.na0s.agent.plist
```

## Data Structures

### pending_deploy.json
```json
{
  "type": "deploy_approval",
  "requested_at": "<ISO-8601>",
  "candidate_path": "data/processed/",
  "gates": {
    "canary": {"passed": true},
    "shadow": {"passed": true},
    "decontam": {"passed": true},
    "f14": {"passed": true}
  },
  "status": "pending"
}
```

### failure_reports
```
data/approval_queue/failure_reports/
├── canary_YYYYMMDD_HHMMSS.json
├── shadow_YYYYMMDD_HHMMSS.json
└── f14_YYYYMMDD_HHMMSS.json
```

## Intelligent Gate Analysis with Claude API

The `GateAnalyzer` can optionally integrate Claude API for intelligent root cause analysis and fix recommendations. When a gate fails, Claude analyzes the failure data and generates actionable insights.

### Configuration

Enable Claude analysis by default:
```python
from na0s.agents.orchestrator import PipelineOrchestrator

# Claude enabled (default)
orchestrator = PipelineOrchestrator(use_claude=True)

# Disable Claude fallback (graceful degradation)
orchestrator = PipelineOrchestrator(use_claude=False)
```

### Environment Variables

```bash
# Required for Claude API calls
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: specify cache directory (default: data/cache/gate_analysis)
export GATE_CACHE_DIR="data/cache/gate_analysis"
```

### Example iMessage Output with Claude Analysis

```
❌ Canary gate failed. Waiting for manual review.
Canary: FAILED
  • Prompt Injection: misclassified
  Root Cause: Model may have learned spurious correlation with quotes in prompts
  Fix: Increase adversarial training on quote-injection variants
Shadow: PASS
```

### Caching and Performance

Claude analysis results are cached in `data/cache/gate_analysis/` to avoid redundant API calls:
- Cache key is based on gate type and failure data hash
- Identical failures reuse cached analysis (instant feedback)
- Cache is persistent across runs for offline review

### Error Handling

If Claude analysis fails or ANTHROPIC_API_KEY is unset:
- System gracefully degrades to metric-based summaries
- iMessage still includes gate verdicts and metric deltas
- No API errors propagate to the user

## OpenClaw Integration

The `OpenClawBridge` class wraps OpenClaw's local HTTP API:

```python
from na0s.agents.openclaw_bridge import OpenClawBridge

bridge = OpenClawBridge(base_url="http://localhost:3000")

# Send message
bridge.send_message("Deploy approved!")

# Wait for reply
reply = bridge.poll_replies(timeout=600)

# Register slash command
bridge.register_skill("approve", "Approve deployment", my_handler)
```

## Testing

Run agent tests:
```bash
pytest tests/agents/ -v
```

## Next Steps

- [x] Integrate Claude API for intelligent analysis (gate failure root causes)
- [ ] Add approval history tracking
- [ ] Implement quarantine action execution (promote/reject)
- [ ] Add retry logic for failed deployments
- [ ] Create dashboard for approval queue status
