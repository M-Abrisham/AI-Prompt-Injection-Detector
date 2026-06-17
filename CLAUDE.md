# Na0S — Claude Code Instructions

## Agent & Task Management

- Use **no more than 4 parallel agents** at a time. More creates notification noise that buries useful output.
- When using background agents (`run_in_background: true`), do NOT print individual agent completion notifications inline. Wait until all agents finish, then give **one consolidated summary** of all findings.
- Each agent must have a clear, bounded scope. Prefer 2 focused agents over 5 broad ones.
- If an agent takes longer than expected, do NOT launch duplicate agents for the same task.

## Multi-Agent Worktree Discipline (CRITICAL — agents share this checkout)

Several Claude agents may operate on this repo at once (check `git worktree list`). Branch switches and broad `git add` in a shared checkout silently destroy other agents' uncommitted work — this has already caused lost work and a mislabeled commit (a branch switch in the primary checkout wiped one agent's edits; another agent's WIP was then committed under the wrong message).

- **NEVER switch branches (`git checkout <branch>` / `git switch`) in the PRIMARY checkout `/Users/mehrnoosh/Na0S`.** Switching rewrites tracked files and clobbers neighbors' uncommitted edits. Treat the primary checkout as read-mostly and leave it on its current branch.
- **Work in your OWN worktree:** `git worktree add .claude/worktrees/<task> <base-branch>` (or `/tmp/na0s_<task>`), then `cd` into it. Run tests with `PYTHONPATH=<worktree>/src python3 -m pytest …`. The two dependabot agents and the d8 agent already follow this — everyone must.
- **Scope every commit:** stage only your task's explicit paths (`git add <paths>`), NEVER `git add -A` / `git add .`. Run `git status` and confirm every staged file is yours before committing, so a neighbor's changes aren't swept in.
- **Hot shared files** (`predict.py`, `cascade.py`, `layer1/rules_registry.py`, `_voting.py`): only edit them inside your own worktree, then rebase — never in the shared checkout. See `docs/AGENT_TEAM.md` for per-role file ownership and the live branch/worktree map.

## Testing

- After completing all code changes, **always run the full test suite** and confirm zero regressions before reporting completion: `python3 -m pytest tests/ -q --tb=line`
- Na0S has ~8,000+ tests. A passing run takes ~15 minutes. Use `--tb=line` for concise failure output.
- For package-specific work, run targeted tests first: `python3 -m pytest tests/canary/ -v` or `tests/ml/` etc.
- Never weaken test assertions to make them pass. Fix the code, not the test.

## Problem Solving

- For unfamiliar or exploratory tasks (CTF, debugging unknown issues), produce a **concrete working attempt within the first 2 responses**. Do not loop through file analysis without actionable output.
- Break open-ended problems into explicit phases with deliverables at each phase.

## Code Conventions

### Source organization
- **New modules go into sub-packages**, NOT as top-level files in `src/na0s/`. The organized packages are:
  - `canary/` — canary token detection and management
  - `detectors/` — specialized prompt injection detectors
  - `fusion/` — ensemble fusion, complexity routing, score aggregation
  - `integrity/` — supply chain integrity, model security, validation
  - `judge/` — LLM judge-based classification
  - `ml/` — ML classifiers, embeddings, model inference
  - `rag/` — RAG pipeline security, output scanning, propagation detection
  - `worm/` — worm signature detection
  - `parsers/office/` — office document parsing (DOCX/XLSX/PPTX/ODF/OLE)
  - `layer0/` through `layer16/` — detection layers
- Core pipeline files (`predict.py`, `cascade.py`, `config.py`, `signal_boost.py`, etc.) stay at top level
- Backward-compat shims exist at old module paths (e.g., `na0s.canary_alert` redirects to `na0s.canary.alert`). Do NOT add code to shim files.
- New layers register via `try/except ImportError` + `_HAS_*` feature flags in `predict.py` or `cascade.py`
- All HTTP utilities for Layer 15+ live in `layer15/http_utils.py`

### Test organization
- Tests mirror the source package structure: `tests/canary/`, `tests/judge/`, `tests/ml/`, etc.
- New tests go in the matching subdirectory, NOT as flat files at `tests/` root
- `tests/test_layer16/` and `tests/parsers/office/` are the reference patterns
- Core pipeline tests (test_cascade*.py, test_predict*.py, test_scan_*.py) stay at `tests/` root
- Use `pytest` with `unittest.mock` for mocking. Never hit real APIs in tests.

### Branch naming
- `feat/<description>` — new features
- `fix/<description>` — bug fixes
- `refactor/<description>` — code reorganization
- `hardening/<description>` — security/robustness improvements
- `ci/<description>` — CI/CD changes
- `docs/<description>` — documentation only

## Commits

- Keep commit messages short — no Co-Authored-By lines
- One logical change per commit
