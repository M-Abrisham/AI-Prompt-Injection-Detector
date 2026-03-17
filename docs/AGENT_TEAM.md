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
  - `pytest -q tests/test_rules.py tests/test_scan_integration.py`

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

