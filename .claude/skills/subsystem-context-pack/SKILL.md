---
name: subsystem-context-pack
description: >-
  Packs a single Na0S subsystem (a layerN/ directory, detectors/, rag/, canary/,
  or eval/) into one compressed, token-budgeted file with repomix, to seed a
  subagent with only that subsystem's context. Use when handing a bounded slice
  of the codebase to a parallel auditor/reviewer agent, when a task needs a
  subsystem's shape without its full source, or when bootstrapping a
  per-subsystem skill. Keeps each agent's context window small and identical
  across the 4-agent fan-out.
---

# Subsystem Context Pack

Give a subagent exactly one subsystem, not the whole tree. `repomix` packs a
glob of files into a single Markdown/XML blob with a token count, so each
parallel auditor gets a small, identical, budget-bounded context.

No install state — run via `npx repomix@latest`. Add `repomix-output.*` to
`.gitignore`; never commit the packed output.

> **Verify flags against `npx repomix@latest --help` before relying on them** —
> repomix flag names have drifted across versions (`--compress`,
> `--token-count-tree`, `--skill-generate`, budget flags have all changed names
> at points). Treat the commands below as the shape, not gospel. (See the
> hallucinated-API and version-drift sections of `na0s-review-checklist`.)

## Repo layout these globs target
- Layers: `src/na0s/layer0/ … layer16/` (8–19 files each)
- Detectors: `src/na0s/detectors/` (~14 security detectors)
- Other subsystems: `src/na0s/rag/`, `src/na0s/canary/`, `src/na0s/eval/`
- Core pipeline (top-level): `src/na0s/cascade.py`, `src/na0s/predict.py`
- Tests mirror source: `tests/test_layer16/`, `tests/parsers/office/`, etc.

## Common packs

Feed one layer + its tests to a context-window subagent, signatures only:
```
npx repomix@latest --include "src/na0s/layer16/**,tests/test_layer16/**" \
  --style markdown --compress -o /tmp/layer16.md
```
`--compress` keeps function/class signatures and drops bodies (~70% fewer
tokens) — right when an agent needs the *shape* of a subsystem, not every line.

Pack all detectors for a cross-detector audit, and see what dominates the budget:
```
npx repomix@latest --include "src/na0s/detectors/**" --style xml \
  --token-count-tree 500 -o /tmp/detectors.xml
```

Budget-guard a pack so an agent prompt can't blow the window:
```
npx repomix@latest --include "src/na0s/cascade.py,src/na0s/predict.py" \
  --token-budget 60000 --stdout
```

Pack only the changed files for a focused review handoff:
```
git diff --name-only main | npx repomix@latest --stdin --style markdown -o /tmp/diff-pack.md
```

## Make all 4 agents pack identically
Pin a shared config so every agent slices the same way:
```
npx repomix@latest --init     # writes repomix.config.json
```
Set `output.style: "markdown"`, `output.compress: true`, and a default `include`.
Commit `repomix.config.json`; gitignore the output.

## Bootstrap a per-subsystem skill
repomix can emit Agent-Skills format directly from a packed subset:
```
npx repomix@latest --include "src/na0s/layer16/**" --compress --skill-generate layer16
```
Use this only as a *starting draft* — hand-edit the generated SKILL.md to the
spec (name/description/<500 lines) before relying on it.

## Guardrails
- **Never pipe a pack to an external service.** Packs contain proprietary
  detector logic; keep them local (`/tmp` or a gitignored path).
- Prefer `--compress` for audits; use full source only when an agent must edit.
- One subsystem per pack — the point is bounded context, not a whole-repo dump.
