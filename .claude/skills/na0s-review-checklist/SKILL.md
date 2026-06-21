---
name: na0s-review-checklist
description: >-
  Reviews AI-generated or AI-modified code against 13 documented failure
  patterns — hallucinated APIs, import blindness, hollow tests, silent refactor
  destruction, arbitrary security thresholds, mocked-CLI smoke gaps, env/VS Code
  blind spots, and destructive autonomous actions. Use when reviewing a diff,
  auditing tests, debugging generated code, before committing or opening a PR,
  and when spawning review/debug/refactor subagents — paste the relevant
  section's "agent snippet" into the subagent's prompt.
---

# Na0S Code-Review Failure Checklist

Run any AI-produced code through these 13 sections before trusting it. The core
failure mode is **confident plausible wrongness** — code that looks expert-level
but is subtly wrong. Polished output reads as authoritative; treat every output
as a review target, not a deliverable. For Na0S (security infrastructure),
**subtly wrong is worse than obviously broken.**

When spawning a subagent, paste the section numbers relevant to its scope into
its prompt, using the **agent snippet** under each section.

---

## 1. Hallucinated APIs and method signatures
- Verify every unfamiliar call against actual docs/source, not memory.
- Check library version compatibility (rapidly-changing AI libs especially).
- Watch for plausible names that don't exist (`findByEmail` vs `getUserByEmail`
  for the same API). Hallucination rates rise for niche languages/libs.
- Red flag: an "apology + fix" that introduces two *new* invented methods.

**Agent snippet:** "Before trusting any method/function call, grep-verify it
exists in the codebase or installed library (`pip show`, `npm list`). Flag any
call you couldn't verify — don't assume it's correct because the name sounds right."

## 2. Import blindness
- Uses something without importing it; imports something never used; imports
  from a wrong path; assumes `from na0s.x.y import Z` resolves without checking.

**Agent snippet:** "Grep every import — confirm each path and name resolves.
Confirm every used name has an import or local definition. Flag unused AND
missing imports."

## 3. Silent project-structure assumptions
- Assumes file paths, module names, folder layout without checking.
- References files that don't exist; assumes `src/` vs flat layout.

**Agent snippet:** "Before writing path-dependent code, `ls`/`find` to verify
structure. Cite the specific files you confirmed. Never fabricate a path from
convention."

## 4. Tests that test nothing (most insidious)
- "Inverted dual" tests: mock everything, assert the mock came back.
- Tests that still pass when the code under test is deleted.
- 100% coverage testing peripheral features, zero for the core.
- **CLI/entrypoint scripts with mocked collaborators are the worst offenders.**
  Live example: `scripts/shadow_evaluate.py` (a CI promotion gate) had 52 green
  unit tests yet crashed on first real run — a 10000-vs-10029 TF-IDF/structural
  feature-dimension mismatch from the L5 merge. The tests mocked
  `load_production_model`, so the real path was never exercised.

**Agent snippet:** "For each test ask: 'Would this fail if I broke the feature it
claims to test?' Comment out the code-under-test and confirm the test fails —
prove it has teeth. Don't test the mock. **For any script in `scripts/`, any
`__main__`, anything CI invokes: do not trust green unit tests — invoke it
end-to-end against realistic inputs at least once. If it loads a model/DB/
resource, the real load must happen in your smoke run, not a mock.**"

## 5. Silent refactoring destruction
- Removes functionality it "didn't understand": thread sync, error handling,
  edge cases, transaction caps, rate limits, validators "that existed for no reason."
- Speculative cascades — "preventively" fixing working code.

**Agent snippet:** "Always `git diff` original vs rewrite before accepting.
Never accept a full-file rewrite blindly. Flag any removed constant, guard, sync
primitive, or error handler with 'WHY was this removed?' If the answer isn't in
the task, restore it."

## 6. Incomplete error handling
- Happy path only; generic `except Exception` swallowing real errors; no cleanup
  on failure (handles, connections, locks); missing null/empty/malformed handling.

**Agent snippet:** "For every external call (network, DB, subprocess, file IO):
what happens on failure? Is the resource cleaned up? Is the error caught at the
right layer? Flag bare excepts and any resource opened without with/try-finally."

## 7. Default thresholds and magic numbers in security code
- Round-number defaults that look professional but were never justified. Live
  example: `GATE_MAX_FPR_INCREASE = 0.01` and `GATE_MAX_RECALL_DROP = 0.005` in
  `scripts/shadow_evaluate.py` — 1% FPR / 0.5% recall slack, no statistical test,
  no attack-cost-asymmetry calibration, no per-severity stratification, no link
  to the threat model. For a defensive SDK these are likely 2–5x too loose.
- Common offenders: `0.5`, `0.8`, `0.95`, `0.99`, `0.01`, `0.05`, `30`/`60`,
  `100`/`1000`.

**Agent snippet:** "When you hit a numeric threshold in security code (FPR,
recall, similarity, confidence, p-value, rate-limit, timeout, retry, batch),
grep for its origin commit and any justification. If it's a round number with no
inline rationale, no threat-model link, no calibration script, flag it:
'Likely-arbitrary default — verify against threat model and attack-cost
asymmetry.' Recommend tightening 2–5x or a calibration sweep before treating it
as production-grade."

## 8. Dependency / package / environment blindness
- Doesn't check if a package is installed or conflicts; ignores Apple Silicon
  arch issues; assumes the right interpreter (VS Code may point at a venv it
  doesn't know about). "Installed it but not found" = usually the wrong venv.

**Agent snippet:** "Before suggesting an install: is it already installed
(`pip show X`)? Will it conflict? Is the user on Apple Silicon? Which interpreter
is active (`which python3`, `.venv/bin/python --version`)? State environment
assumptions explicitly."

## 9. VS Code blind spots
- Doesn't know settings.json, installed extensions, or which interpreter VS Code
  uses; integrated terminal differs from external (shell-init); extension
  conflicts are a common root cause.

**Agent snippet:** "If the task touches VS Code config/keybindings/extensions,
ask what's installed and what settings.json says first. If something 'should
work but doesn't,' check interpreter path + extension conflicts before blaming code."

## 10. Outdated syntax / deprecated patterns
- Training cutoff means recent versions are unknown; may use deprecated APIs with
  un-patchable CVEs; may use the wrong year in date handling.

**Agent snippet:** "Verify versions (`pip list`) before version-dependent code.
For any library updated in the last 12 months, fetch current docs rather than
relying on trained knowledge. Check the CHANGELOG for deprecations."

## 11. Dead code after refactor + shim traps
- Copied instead of moved; duplicate definitions; orphaned imports; test files
  never cleaned up.
- **Shim/wrapper traps:** when an old path (`scripts/X.py`) becomes a thin
  re-export while real code moves to `src/na0s/X.py`, every build-system dep list
  referencing the old path (dvc.yaml `deps:`, Makefile prereqs, workflow
  path-triggers, pre-commit paths) now tracks only the wrapper → silent stale
  outputs / skipped CI. Caught as a P1 on PR #79.
- **Smoke FIRST, wire SECOND.** Wiring allegedly-functional-but-never-run code
  into a pipeline is worse than leaving it dead — it blocks the pipeline that
  depended on its absence (the F1 / `shadow_evaluate.py` lesson).

**Agent snippet:** "After any refactor: grep old names/paths across the whole
repo; delete orphaned imports; confirm no duplicate defs; confirm the old file is
actually removed. If a wrapper/shim is created at the old path, ALSO grep
dvc.yaml, .github/workflows/, Makefile, .pre-commit-config.yaml, pyproject.toml
for that path and add the real module path. Before wiring any unused code path
into a pipeline (CI gate, cron, hook), run it end-to-end with realistic inputs
FIRST; if it crashes, fix or descope before wiring."

## 12. Overconfidence / premature victory
- Same confidence whether certain or guessing; "everything works!" when nothing
  was fixed; self-review always says "looks excellent"; claims a tool ran when it
  didn't; won't flag ambiguity — silently assumes.

**Agent snippet:** "Never declare success without running the actual
verification — show the command and its output, not a summary. If you hit a
setback, say so plainly. Flag assumptions explicitly. If torn between two
options, ask rather than guess."

## 13. Destructive autonomous actions (never without confirmation)
- `rm -rf`, `git reset --hard`, `git push --force`, DB wipes,
  `--accept-data-loss`, CI-bypass tricks, dev-only code landing on main.
- An expert engineer refuses these and pushes back.

**Agent snippet:** "Before ANY destructive op (rm, reset, force-push, DB write,
uninstall), stop and ask the user. No exceptions. Never chain destructive ops.
If a CI check is in the way, fix the root cause — don't bypass."

---

## Reviewer reminders (before handing output to the user)
- **Smoke-test every CLI script the work touches** — don't trust green units;
  run it end-to-end and paste real stdout/stderr.
- **Surface hidden assumptions** — ask "what are you assuming about my
  environment/structure/versions?" before non-trivial work.
- **Read the diff, not just the new code.** Never accept a full-file replace blindly.
- **Run the code before trusting the explanation** — explanations describe what
  code *should* do, not always what it *does*.
- **Slot-machine rule** — save state before a long autonomous run; accept or
  discard the whole result; don't engage half-good output.

## Meta-pattern
Failures scale with trust. The productivity gains exist only for those who've
learned where AI judgment can't be trusted. Treat every output as a code-review
target, not a deliverable.
