# Na0S — Claude Code Instructions

## Agent & Task Management

- Use **no more than 4 parallel agents** at a time. More creates notification noise that buries useful output.
- When using background agents (`run_in_background: true`), do NOT print individual agent completion notifications inline. Wait until all agents finish, then give **one consolidated summary** of all findings.
- Each agent must have a clear, bounded scope. Prefer 2 focused agents over 5 broad ones.
- If an agent takes longer than expected, do NOT launch duplicate agents for the same task.

## Testing

- After completing all code changes, **always run the full test suite** and confirm zero regressions before reporting completion: `python3 -m pytest tests/ -q --tb=line`
- Na0S has ~8,000+ tests. A passing run takes ~15 minutes. Use `--tb=line` for concise failure output.
- For layer-specific work, run targeted tests first: `python3 -m pytest tests/test_layerN/ -v`
- Never weaken test assertions to make them pass. Fix the code, not the test.

## Problem Solving

- For unfamiliar or exploratory tasks (CTF, debugging unknown issues), produce a **concrete working attempt within the first 2 responses**. Do not loop through file analysis without actionable output.
- Break open-ended problems into explicit phases with deliverables at each phase.

## Code Conventions

- Layers use directories under `src/na0s/` (e.g., `layer0/`, `layer1/`, `layer16/`)
- New layers register via `try/except ImportError` + `_HAS_*` feature flags in `predict.py` or `cascade.py`
- All HTTP utilities for Layer 15+ live in `layer15/http_utils.py`
- Tests live in `tests/` (flat files like `test_layer15_*.py` or directories like `tests/test_layer16/`)
- Use `pytest` with `unittest.mock` for mocking. Never hit real APIs in tests.

## Commits

- Keep commit messages short — no Co-Authored-By lines
- One logical change per commit
