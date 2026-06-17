 # Na0S Agent Team

This document defines a practical multi-agent team for operating Na0S.
Each "agent" is a role with clear ownership, inputs, outputs, and commands.

## Mission

Ship detection improvements quickly without regressing:
- malicious recall floor
- false positive ceiling
- runtime stability
- taxonomy consistency

## Team Roster

### 1) Orchestrator Agent
- Owns: intake, prioritization, handoffs, final go/no-go.
- Inputs: issue reports, benchmark deltas, CI failures.
- Outputs: short execution plan, owner assignment, release decision.
- Runs:
  - `pytest -q`
  - `python scripts/benchmark.py`

### 2) Threat Intel Agent
- Owns: new attack patterns from community and dataset sources.
- Inputs: social scraping output, fresh incidents, adversarial samples.
- Outputs: normalized candidate samples with taxonomy tags.
- Runs:
  - `python scripts/social_scraper.py`
  - `python scripts/weekly_harvest.py`
  - `python scripts/integrate_harvest.py`

### 3) Rule Engineer Agent
- Owns: Layer-1/Layer-2 detection logic, context suppression safety.
- Inputs: missed attacks and false positives from benchmarks.
- Outputs: rule updates + targeted tests.
- Touches:
  - `src/na0s/layer1/rules_registry.py`
  - `src/na0s/layer1/context.py`
  - `src/na0s/layer2/obfuscation.py`
- Runs:
  - `pytest -q tests/rules/test_rules.py tests/test_scan_integration.py`

### 4) ML & Embedding Agent
- Owns: feature/model training, threshold calibration, embedding behavior.
- Inputs: processed datasets and hard negatives.
- Outputs: refreshed models + calibration artifacts.
- Runs:
  - `python scripts/process_data.py`
  - `python scripts/features.py`
  - `python scripts/model.py`
  - `python scripts/threshold_sweep.py`

### 5) Validation Agent
- Owns: regression gating and performance/quality acceptance.
- Inputs: proposed code/model changes.
- Outputs: pass/fail report with exact regressions.
- Runs:
  - `pytest -q`
  - `pytest -q tests/test_benchmark_regression.py tests/test_false_positives.py`
  - `python scripts/benchmark.py`

### 6) Release Agent
- Owns: changelog, docs consistency, package/release workflow safety.
- Inputs: validated changes from all agents.
- Outputs: release notes and publish-ready branch.
- Runs:
  - `pytest -q tests/test_publish_workflow.py`
  - `python -m na0s --help`

## Operating Protocol

1. Orchestrator opens a task with target metric changes.
2. Threat Intel and Rule Engineer work in parallel on detection gaps.
3. ML Agent recalibrates only if rule-only fix is insufficient.
4. Validation Agent blocks merge on recall/FPR/runtime regressions.
5. Release Agent publishes only after all required checks pass.

## Handoff Template

Use this exact structure for each handoff:

```text
Task:
Scope:
Evidence:
Changed Files:
Validation Run:
Residual Risk:
Next Owner:
```

## Minimum Merge Gate

A change is merge-ready only if all are true:
- `pytest -q` passes
- benchmark regression suite passes
- no unexpected increase in false positives
- taxonomy tags remain correct for affected techniques

## Concurrent Work: Worktree & Branch Discipline (added 2026-06-16)

Multiple agents run against this repo at the same time. **Rule #1: never switch branches in the primary checkout `/Users/mehrnoosh/Na0S`** — a branch switch there rewrites tracked files and destroys other agents' *uncommitted* work. Each agent works in its own `git worktree`. Full protocol lives in `CLAUDE.md` → "Multi-Agent Worktree Discipline".

### How to start work safely
```bash
git worktree add .claude/worktrees/<task> <base-branch>     # or /tmp/na0s_<task>
cd .claude/worktrees/<task>
PYTHONPATH="$PWD/src" python3 -m pytest tests/<area>/ -q     # test from the worktree
git add <explicit paths only>                               # NEVER `git add -A` / `git add .`
git commit -m "<scoped message>"
git worktree remove .claude/worktrees/<task>                # when merged/abandoned
```

### Hot shared files — coordinate, edit only in your worktree
`src/na0s/predict.py`, `src/na0s/cascade.py`, `src/na0s/layer1/rules_registry.py`, `src/na0s/_voting.py`, `src/na0s/__init__.py`. Multiple roles touch these (Rule Engineer, ML, refactors). Editing them in the shared checkout is the main collision source.

### Live worktree/branch map (snapshot 2026-06-16 — re-verify with `git worktree list`)
- `/Users/mehrnoosh/Na0S` — PRIMARY checkout, on `hardening/d8-context-window`. Read-mostly; do NOT branch-switch here.
- `/private/tmp/na0s_d8_wt` — d8 context-window agent (detached HEAD, locked).
- `.claude/worktrees/agent-aa0756264bec936ab`, `.../agent-aa7b8ad42df33657a` — dependabot dependency bumps.

### Incident that prompted this protocol (2026-06-16)
A branch switch in the PRIMARY checkout (→ `hardening/d8-context-window`) wiped a refactor agent's uncommitted import-canonicalization edits to `cascade.py`/`predict.py`. The d8 agent's then-uncommitted WIP was subsequently committed under the wrong message (commit `f6b3ab8`, "refactor: repoint core pipeline imports…", which actually contains D8 context-window code). Fix forward: isolate in worktrees, scope every `git add`, keep the primary checkout on a stable branch.

