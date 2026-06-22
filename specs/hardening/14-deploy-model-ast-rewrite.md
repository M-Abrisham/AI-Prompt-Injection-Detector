---
item: 14 — M4(a)
title: "Replace brace-fragile re.sub KNOWN_HASHES rewrite with AST (stdlib) slice-rewrite + ast.literal_eval verify"
priority_tier: P1 (robustness / data-corruption footgun; no attacker precondition; builds on item #1)
class: supply-chain / integrity (NOT a prompt-injection attack class)
depends_on: [1]          # M4(b) — preserve/merge KNOWN_HASHES (specs/hardening/01-deploy-model-known-hashes-drop.md)
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]
applicable_qs: [Q1, Q2, Q3, Q4, Q8, Q10]
na_steps: [5, 9, 10]
na_qs: [Q5, Q6, Q7, Q9]
touches_predict_py: false    # predict.py:84 only imports KNOWN_HASHES; this item rewrites the PRODUCER, not the consumer — no predict.py edit
touches_cascade_py: false    # cascade.py holds no KNOWN_HASHES ref; benefits via the loader, no parity edit
---

# Item 14 — M4(a): AST-based `KNOWN_HASHES` rewrite + `ast.literal_eval` verify

## 0. Root cause (confirmed against source — line numbers verified 2026-06-22)

`scripts/deploy_model.py` rewrites the `KNOWN_HASHES` dict in
`src/na0s/models/__init__.py` with a **single regex substitution** whose body class is
`[^}]*` — it matches everything up to the **first** `}`, not the matching closing brace.
The current code (line numbers re-verified, header KEY-REF said 159-170; the actual block is
**158-170**):

```python
# scripts/deploy_model.py:158-170
# Build replacement dict literal
entries = ",\n".join(
    f'    "{fname}": "{digest}"' for fname, digest in sorted(new_hashes.items())
)                                                                       # :159-161
new_dict = "KNOWN_HASHES = {\n" + entries + ",\n}"                       # :162
# Replace the existing KNOWN_HASHES block
updated = re.sub(
    r"KNOWN_HASHES\s*=\s*\{[^}]*\}",                                     # :166  <-- brace-fragile
    new_dict,
    content,
    count=1,
)                                                                       # :165-170
```

**Empirically reproduced (not guessed)** — given a `KNOWN_HASHES` block that contains a `}`
anywhere inside it (e.g. inside a trailing comment, a future nested literal, or a `# {…}`
annotation), the `[^}]*` class stops at that first inner brace and `re.sub` produces **invalid
Python**:

```text
KNOWN_HASHES = {
    "model.pkl": "xx",
} brace in comment          <-- regex closed here, trailing garbage left behind
    "tfidf_vectorizer.pkl": "bb",
}
SOMETHING_ELSE = 1
```

`ast.parse()` on that output raises `SyntaxError: unmatched '}'`. The deploy script does **not
parse-verify** what it writes (`scripts/deploy_model.py:176-183` writes `updated` straight to
disk), so a corrupt `__init__.py` would ship silently and break **every** `import na0s.models`
(and therefore `import na0s`) at next load — a hard, repo-wide ImportError.

**Why AST is the correct fix (empirically validated):**
`ast.parse(src)` + `ast.walk` locates the `KNOWN_HASHES` `Assign` node; against the real file the
value node spans **lines 26-31** (`src/na0s/models/__init__.py:26-31`). `ast.get_source_segment`
extracts the exact dict literal and `ast.literal_eval` parses it safely (no code execution) →
returns a 4-entry dict. The rewrite then slices the source on the node's `lineno`/`end_lineno`
(or `col_offset`/`end_col_offset`) and substitutes a freshly-rendered literal — brace count
inside the literal is irrelevant because we never count braces. A final `ast.parse(updated)` (and
optionally a re-`literal_eval` of the new block) **verifies** the output before it is written.

**Relationship to item #1 (M4b, `specs/hardening/01-…md`):** item #1 already plans to *parse the
existing dict via `ast.literal_eval` and seed `new_hashes` from it* (see §3A step 1 of spec 01,
`scripts/deploy_model.py:158` insertion point — same line). Item #1 fixes **what data goes in**
(preserve-and-merge); item #14 fixes **how the file is rewritten** (AST slice instead of
brace-fragile `re.sub`) **and verifies the write parses**. They touch the same 158-170 block, so
**#14 builds on #1** and should land after it to avoid a merge conflict and to share the
`ast.literal_eval`-of-existing-dict reader. This is a **robustness / data-corruption footgun, not
an exploit** — no attacker precondition.

---

## 1. Step 1 — Current system vs. what it SHOULD be; gaps & edge cases (Q1)

**Current:** rewrite = `re.sub(r"KNOWN_HASHES\s*=\s*\{[^}]*\}", new_dict, content, count=1)`
(`:165-170`); no verification that the result is valid Python; presence-check is a second regex
`re.search(r"KNOWN_HASHES\s*=\s*\{", content)` (`:172`).

**What it SHOULD be:** rewrite must (a) locate the dict via the Python grammar (AST), not a
brace-class regex; (b) replace **only** the value-node source span; (c) **verify** the rewritten
module parses (`ast.parse`) and that the new dict round-trips (`ast.literal_eval`) before writing;
(d) fail closed (non-zero exit, original file untouched) if verification fails.

**Gaps / edge cases:**
- G1 — Nested/inner `}` (comment, future set/dict literal, type annotation) corrupts the file
  today; the headline robustness gap (reproduced above).
- G2 — Multi-line / reordered / extra-whitespace dict bodies: `[^}]*` already handles newlines
  greedily-to-first-brace, but any inner brace breaks it; AST handles arbitrary formatting.
- G3 — **No post-write parse check** — a malformed rewrite ships and breaks `import na0s`. The fix
  MUST add an `ast.parse(updated)` guard and roll back / abort on failure.
- G4 — Backup/rollback survival (item-specific scope): if the AST rewrite is going to mutate
  `__init__.py`, the script must back up the **init file** (not just the .pkl artefacts) so a bad
  rewrite is recoverable. Today `_backup_file` (`:45-85`) backs up only the copied `.pkl` files
  (`:138`); `__init__.py` is overwritten in place with **no backup** (`:177-183`). Add an init
  backup + restore-on-verify-failure path.
- G5 — Idempotency: a second identical deploy must produce a byte-identical (or semantically
  identical, modulo deterministic ordering) `__init__.py`; the existing `updated == content` skip
  (`:174-175`) must be preserved by the AST path.
- G6 — Absent-block path: `test_does_not_touch_init_when_no_block_present`
  (`tests/test_deploy_model.py:390-408`) requires that a file with **no** `KNOWN_HASHES` warns and
  still exits 0 — the AST walk must return "not found" gracefully (no `KeyError`/`StopIteration`).
- G7 — `libcst` is **NOT installed** in this environment (verified: `import libcst` →
  `ModuleNotFoundError`; `python3 --version` → 3.14.4; `import ast` OK). Therefore the
  implementation MUST default to **stdlib `ast`** and treat `libcst` as an *optional* nicety only
  (see §3). No new hard dependency may be added to a deploy script.

---

## 2. Step 2 — Roadmap / Taxonomy / README / Coverage Matrix cross-reference

- `ROADMAP_V2.md:1360` — *"`deploy_model.py` — backup/rollback paths, `KNOWN_HASHES` regex
  replacement, failure-mode assertions. Priority: P1."* This item is the *"`KNOWN_HASHES` regex
  replacement"* + *"failure-mode assertions"* sub-clause (item #1 covers the data-preservation
  clause). Tick / annotate with the fix SHA once landed (MEMORY: Roadmap-Todo Sync).
- `ROADMAP_V2.md:1289, 1347` — L13 narrative states *"`deploy_model.py` … programmatically updates
  `KNOWN_HASHES`"* / *"backup + rollback with programmatic `KNOWN_HASHES` update"*. After this fix
  the "programmatic update" is AST-safe; update the narrative wording only if it claims regex.
- `ROADMAP_V2.md:2840` — `_backup_file()` (.pkl backup) already marked DONE; this item **extends**
  backup coverage to the init file (G4) — note that as a follow-on, do not re-touch the .pkl path.
- README / `THREAT_TAXONOMY.md` / Coverage Matrix: **no change** — internal deploy-script
  robustness, not a user-facing detector or taxonomy-tagged attack class.
- Disambiguation: the "M4" at `ROADMAP_V2.md:1515` (hardware-key-signed bot commits) and taxonomy
  "M4 (Code)" (`:1241,1275`) are unrelated; "M4(a)" here is the supply-chain audit's own numbering.

---

## 3. Step 3 — Root-cause implementation plan (numbered)

Replace the brace-fragile `re.sub` rewrite with an **AST locate → slice-replace → verify** path,
on top of item #1's preserve-and-merge `new_hashes`. Stdlib `ast` only; `libcst` optional.

### 3A. Recommended — stdlib `ast` slice-rewrite + verify (one logical change)

1. **Add a `_rewrite_known_hashes(content, new_hashes)` helper** in `scripts/deploy_model.py`
   (alphabetical import: add `import ast` to the block at `:14-20`). It must:
   1. `tree = ast.parse(content)`; walk for the `ast.Assign` whose target is `Name(id="KNOWN_HASHES")`
      (matches the real node at `src/na0s/models/__init__.py:26-31`).
   2. If not found → return `(content, "absent")` so the caller can warn-and-exit-0 (preserves
      G6 / `tests/test_deploy_model.py:390-408`).
   3. Extract the existing literal with `ast.get_source_segment(content, node.value)` and verify it
      with `ast.literal_eval(...)` — this is the **input verify** (also the reader item #1 needs).
   4. Render the new literal text deterministically (reuse item #1's `entries`/`new_dict` builder,
      `:159-162`) and splice it into `content` by the value node's `lineno/col_offset` …
      `end_lineno/end_col_offset` (Python ≥3.8 always sets `end_*`; env is 3.14.4 — verified).
   5. **Output verify (fail closed):** `ast.parse(updated)`; then re-extract the new value segment
      and `ast.literal_eval` it; assert it equals `new_hashes` (or its sorted form). If any check
      raises → do **not** write; return a sentinel so the caller aborts with a non-zero exit and the
      original file is left intact / restored from backup (G3).
2. **Back up `__init__.py` before mutating it (G4):** call `_backup_file(init_path)` (the existing
   `:45-85` helper works on any path) right before the write at `:177`. On a verify failure or write
   `OSError`, restore from the `.bak` and `sys.exit(1)`.
3. **Replace the `:164-183` block**: swap the `re.sub` (`:165-170`) for `_rewrite_known_hashes(...)`,
   keep the `updated == content` "unchanged" skip (`:174-175`) and the absent-block warning
   (`:172-173`) — but drive both off the helper's return, not off a second regex.
4. Keep copy/skip/.pkl-backup logic (`:120-148`) and rollback (`:193-227`) unchanged.

### 3B. Optional — `libcst` codemod (NOT the default)

`libcst` would give a formatting-preserving codemod, but it is **not installed** (verified) and a
deploy script must not grow a hard third-party dep. If ever adopted, gate it behind a
`try: import libcst except ImportError:` fallback to the §3A `ast` path. **Recommendation: ship
3A only**; mention 3B in the docstring as a future option, do not implement it now.

### 3C. Why not "just fix the regex"

A balanced-brace regex in Python `re` is impossible (no recursion); `regex` module is another dep.
`[^}]*` → `.*?` with `re.DOTALL` still cannot tell an inner `}` from the closing one. AST is the
only correct, dependency-free option — and it doubles as the `literal_eval` verify the item asks
for.

---

## 4. Step 4 — Wiring / pipeline parity (Q8)

- **predict.py (Q8 — consumer only, no edit):** `src/na0s/predict.py:84` `from .models import …,
  KNOWN_HASHES`; `:343` `KNOWN_HASHES.get("model.pkl", "")`. predict.py merely *reads* the dict at
  import time; this item changes only how the dict is *written to source*. No predict.py change.
  The relevant guarantee is that the rewritten `__init__.py` still **imports** — that is exactly
  what the §3A output `ast.parse` verify enforces, end-to-end protecting predict.py's import.
- **cascade.py (Q8 — does not apply):** `src/na0s/cascade.py` holds no `KNOWN_HASHES` reference;
  no parity edit. It benefits transitively (a valid `__init__.py` keeps `na0s.models` importable).
- **safe_pickle:** unchanged — `na0s.integrity.safe_pickle` consumes `KNOWN_HASHES`
  (`src/na0s/integrity/safe_pickle.py:38,221-222`); the dict's *runtime shape* is unchanged by an
  AST rewrite, so the integrity path is untouched. Do not edit the `na0s.safe_pickle` shim
  (CLAUDE.md rule).
- **Retrain integration caller:** `tests/test_retrain_integration.py:186,202` calls `deploy(...)`
  with a stub `__init__.py` (`:199` `init_path.write_text('KNOWN_HASHES = {\n  "model.pkl": "old",\n}')`)
  — the AST path must keep that test green (it is a clean single-line-per-entry literal).

---

## 5. Step 5 — Harvester audit
**N/A — robustness fix to a deploy script's source-rewrite; no harvested dataset, no detector, no
training corpus is involved.** The only "data" is the bundled `.pkl` inventory (a build artefact).

---

## 6. Step 6 — Test plan (Code + Use-Case / behavior) — Q4 applies

All deploy-script tests stay in `tests/test_deploy_model.py` (core-pipeline script → `tests/`
root per CLAUDE.md; the file already holds 29 tests). The import-survival use-case test (T5) may
extend `tests/test_model_versioning.py` (`:8` already imports `KNOWN_HASHES`).

**Code-level tests (the robustness bug, directly):**

- **T1 — nested-brace survives (HEADLINE; must be red on `main`, green on branch):** seed a temp
  `__init__.py` whose `KNOWN_HASHES` block contains an inner `}` (e.g. a trailing
  `# layout: {a:b}` comment on an entry line). Run `deploy(...)`. Assert the rewritten file
  **`ast.parse`-s without `SyntaxError`** and that `KNOWN_HASHES` round-trips via `ast.literal_eval`
  to the expected entries. Document red→green in the docstring (proven: the current `re.sub`
  produces `SyntaxError: unmatched '}'`).
- **T2 — output is verified before write / fail-closed (G3):** monkeypatch the literal renderer (or
  inject a deliberately malformed `new_hashes` rendering) so the rewritten text would be invalid
  Python; assert `deploy()` exits non-zero **and** the original `__init__.py` is left byte-identical
  (restored from backup), i.e. a bad rewrite never ships. Assert exception type / exit code, not
  just "raised".
- **T3 — init-file backup created + restored on failure (G4):** assert a `__init__.py.bak`
  (and/or timestamped) is created before the rewrite; in the T2 failure path assert the live file
  equals the backup content.
- **T4 — idempotency (G5):** run `deploy()` twice with identical inputs; assert the `__init__.py`
  is byte-identical after the second run (the `updated == content` skip path holds under AST).
- **T5b — absent-block path preserved (G6):** the existing
  `test_does_not_touch_init_when_no_block_present` (`tests/test_deploy_model.py:390-408`) must stay
  green under the AST walk (warn + exit 0, no `StopIteration`/`KeyError`).
- **T6 — exactly-one-block invariant preserved:** the existing `test_replaces_existing_block`
  (`tests/test_deploy_model.py:386-388`, the KEY-REF) must stay green — after rewrite exactly one
  `KNOWN_HASHES = {` remains. Keep this test; AST rewrite of a single node guarantees it.

**Use-case / behavior tests (the consequence — `import na0s.models` survives):**

- **T5 — rewritten module imports (durable, end-to-end):** after a `deploy()` over a temp copy of
  the **real** `src/na0s/models/__init__.py` content (with a nested-brace variant), `importlib`
  the rewritten file from a temp path (or `compile()` it) and assert `KNOWN_HASHES` is the expected
  dict — proves the producer never emits a module that breaks `import na0s.models` →
  `predict.py:84`.
- **T7 — real-file shape guard (no mutation):** assert that parsing the *actual*
  `src/na0s/models/__init__.py` with the new helper locates the node at the expected span and
  `literal_eval`s to the 4 live entries (`model.pkl`, `structural_scaler.pkl`,
  `model_embedding.pkl`, `tfidf_vectorizer.pkl` — verified `:26-31`). Read-only; no write to the
  real file.

**Anti-hollow-test discipline (na0s-review-checklist):** every assertion checks a concrete value
(parses-without-error, exact dict equality, exit code, backup byte-equality), never bare
"no exception". T1 documents red→green. **No magic thresholds** — this item introduces zero numeric
constants (AST spans + exit codes only), so the "no arbitrary threshold" checklist item is
satisfied vacuously.

**CLI / suite smoke step (mandatory):**
- Targeted: `python3 -m pytest tests/test_deploy_model.py tests/test_model_versioning.py tests/test_retrain_integration.py -v`.
- CLI smoke (real, not mocked): `python3 scripts/deploy_model.py` writes the **real** init →
  destructive; instead smoke `deploy()` against a temp-dir copy in a test, and run
  `python3 scripts/deploy_model.py --rollback` only against a sandboxed temp `dest_dir` via
  `rollback(dest_dir=...)`. Confirm `import scripts.deploy_model` still succeeds (catches an
  `import ast` typo) — covered by `TestImport.test_module_imports`
  (`tests/test_deploy_model.py:50-57`).
- Full suite before reporting done: `python3 -m pytest tests/ -q --tb=line` (CLAUDE.md ~15 min,
  zero net regressions). Verify against MAIN, not the stale editable install (na0s-debugging /
  MEMORY): if `na0s.models`/`na0s.integrity` resolve oddly, run with `PYTHONPATH=<worktree>/src`.

---

## 7. Step 7 — Cleanup / refactor per conventions (Q2)

- Fix lives entirely in `scripts/deploy_model.py` + tests. Add `import ast` alphabetically to
  `:14-20`. Extract the rewrite into a `_rewrite_known_hashes()` helper (testable in isolation,
  mirrors the existing `_sha256` / `_backup_file` helper style at `:37-85`).
- Remove the now-dead `import re` only if **no** other `re.` use remains after the swap — verify:
  the presence-check at `:172` and the rewrite at `:165-170` are the only `re.` calls; if both move
  to AST, drop `import re` (`:17`). If item #1 still uses a regex to read the old dict, keep `re`.
- Update the module docstring (`:2-12`) and the inline comment `# Replace the existing KNOWN_HASHES
  block` (`:164`) to say "AST-locate the `KNOWN_HASHES` assignment and rewrite its value node,
  verifying the result parses".
- Do **not** commit the pre-existing untracked scratch files (`_skeptic_test_out.txt`,
  `_xfail_run.txt`, `pyt_out.txt`, `_skeptic_test_out.txt`, `logs/`) — leave them untracked; scope
  every `git add` (MEMORY: Multi-Agent Worktree Discipline).
- `scripts/deploy_model.py` is a 4-scripts v1.0.0-rename candidate (MEMORY: v1 restructure) — keep
  this change minimal; do not fold in the unrelated rename.

---

## 8. Step 8 — Roadmap update

- Tick / annotate `ROADMAP_V2.md:1360` (`deploy_model.py … KNOWN_HASHES regex replacement,
  failure-mode assertions, P1`) — the *regex replacement* and *failure-mode assertion* clauses are
  this item; cite the fix SHA (MEMORY: Roadmap-Todo Sync). Cross-reference item #1's SHA as the
  dependency.
- If `ROADMAP_V2.md:1289/1347` wording still implies a regex rewrite, update to "AST-safe
  programmatic `KNOWN_HASHES` update".

---

## 9. Step 9 — README / Benchmark
**N/A — no user-facing surface and no metric/benchmark number changes.** This is an internal
deploy-script rewrite-correctness fix; it changes no detector output and no eval score.

---

## 10. Step 10 — Taxonomy / Coverage Matrix / scorer thresholds
**N/A — supply-chain robustness fix, not an attack class.** No taxonomy code (the script does not
classify prompts), no Coverage Matrix row, no per-attack scorer threshold. **No numeric threshold
is introduced**, so the "no arbitrary magic threshold" checklist item is satisfied vacuously.

---

## 11. Step 11 — PR / test gate

- Branch: `hardening/deploy-model-ast-rewrite` (off `main`, **after** item #1 lands or rebased onto
  it; use a worktree, never branch-switch the primary checkout — MEMORY: Worktree Discipline).
- One logical commit, short message, no Co-Authored-By:
  `hardening(deploy): AST-rewrite KNOWN_HASHES + parse-verify, replace brace-fragile re.sub (M4a)`.
- PR gate: full `tests/` suite green, zero net regressions; **T1 must be red on the pre-fix tree
  and green on the branch** (attach before/after as proof the robustness test bites).
- Skills/agents: `github-pr-prep` (self-review + PR body), `github-pr-review` (precision-first
  review), `github-ci-fix` only if CI goes red.

---

## Q&A self-check

- **Q1 — Can Na0S handle the target?** Not today: any inner `}` in the `KNOWN_HASHES` block makes
  the `re.sub` (`scripts/deploy_model.py:165-170`) emit invalid Python (reproduced:
  `SyntaxError: unmatched '}'`), and the script does not parse-verify before writing
  (`:176-183`) → a corrupt `__init__.py` ships and breaks `import na0s.models`. After §3A
  (AST slice-rewrite + `ast.parse`/`literal_eval` verify + init backup) and §6 tests + full-suite
  green, yes.
- **Q2 — Cleanup?** One file + tests; §7 covers `import re` removal (conditional), docstring/comment
  fixes, and the don't-commit-scratch guard. No clutter introduced.
- **Q3 — Pipeline wiring correct?** Yes by construction — the bug is in the *producer*; the
  consumer (`predict.py:84` import) is protected because the new `ast.parse` output-verify
  guarantees the rewritten module still imports. No new wiring.
- **Q4 — Tested for code AND use-case?** Yes — T1–T4/T6 (script-level robustness/verify/backup),
  T5/T5b/T7 (use-case: rewritten module imports, absent-block, real-file shape guard).
- **Q5 — Harvester audit?** N/A — no harvested dataset.
- **Q6 — Taxonomy + Coverage Matrix?** N/A — not an attack class.
- **Q7 — Scorer scores it correctly?** N/A — no scorer / no threshold.
- **Q8 — predict.py / cascade.py references?** predict.py: `:84` import + `:343` read — *consumer
  only, no edit* (its import is what the output-verify protects). cascade.py: no `KNOWN_HASHES`
  ref, no edit.
- **Q9 — Harvester agent harvests this type?** N/A.
- **Q10 — Other checks:** (a) `libcst` is NOT installed (verified) → MUST default to stdlib `ast`,
  no new hard dep; (b) Python is 3.14.4 → `end_lineno`/`end_col_offset` always present (verified)
  for safe slicing; (c) extend backup coverage to `__init__.py` (today only `.pkl` is backed up,
  `:138`); (d) fail-closed on verify failure (do not ship a file that does not parse); (e) keep the
  existing `test_replaces_existing_block` (`:386-388`) and `test_does_not_touch_init_when_no_block_present`
  (`:390-408`) green.

---

## Agent / skill assignment (inject na0s-review-checklist into every agent prompt)

| Step / scope | Agent / skill |
|---|---|
| Root-cause confirm + read deploy/loader source (Steps 0–2) | `silent-failure-hunter` (the un-verified write at `:176-183`), `l3-l5-code-auditor` (model-loader consumer path) |
| Implementation 3A — AST helper + verify + init backup (Steps 3–4) | `Plan` → primary implementer; `na0s-review-checklist` injected; `na0s-debugging` for env / MAIN-vs-editable + the libcst-absent / py3.14 `end_*` confirmation |
| Integrity correctness (Step 4, import-survival) | `security-research-auditor` + skill `security-review` (confirm the rewrite cannot inject executable code — `literal_eval` only, never `eval`/`exec`) |
| Test authoring (Step 6) | implementer + `na0s-review-checklist` (anti-hollow-test section) |
| Cleanup / refactor (Step 7) | implementer; `l3-l5-code-auditor` second-pass |
| Roadmap sync (Step 8) | implementer (MEMORY: Roadmap-Todo Sync) |
| PR / CI (Step 11) | skills `github-pr-prep`, `github-pr-review`, `github-ci-fix`; agents `pr-review-toolkit:*` |
| Layer 9–11 integrity context (if integrity module needs a parity look) | `layer-9-11-auditor` |

Skills explicitly **not** used and why: `data-harvesting`, `cron-scheduling`,
`eval-scenario-curation`, `detector-authoring`, `llm-judge`, `eval-harness`, `intel-harvest`,
`incident-to-scenario`, `detector-failure-analysis` — none apply (no harvest, no cron, no scenario,
no new detector, no judge, no eval/recall delta).

---

## Execution preconditions / dependencies

- **Depends-on: #1 (M4b — `specs/hardening/01-deploy-model-known-hashes-drop.md`).** #1 introduces
  the `ast.literal_eval`-of-existing-dict reader and the preserve-and-merge `new_hashes` at the
  same `:158` insertion point; #14 swaps the *rewrite mechanism* (regex → AST slice) and adds the
  *output verify*. Land #14 after #1 (or rebase onto it) to share the reader and avoid a conflict
  in the 158-183 block.
- **Blocks / enables:** makes any future `KNOWN_HASHES` edit (e.g. the 384-dim `model_embedding.pkl`
  rebuild, `ROADMAP_V2.md:767`) safe against accidental file corruption.
- Environment: dedicated git worktree off `main`; `libcst` absent → stdlib `ast` path only; verify
  imports with `PYTHONPATH=<worktree>/src` (editable install points at the d8 checkout).

---

## Definition of done

- [ ] `_rewrite_known_hashes()` (stdlib `ast`) replaces the `re.sub` rewrite at
      `scripts/deploy_model.py:165-170`; no balanced-brace regex remains.
- [ ] Output is verified with `ast.parse` + `ast.literal_eval` before writing; rewrite fails closed
      (non-zero exit, original file intact/restored) on any verify failure.
- [ ] `__init__.py` is backed up before mutation and restored on failure (G4).
- [ ] T1 (nested-brace survives) is red on the pre-fix tree, green on the branch (proof attached).
- [ ] T2 (fail-closed verify), T3 (init backup/restore), T4 (idempotency), T5/T5b/T7 (import
      survives / absent-block / real-file shape) added and passing.
- [ ] Existing `test_replaces_existing_block` (`:386-388`) and
      `test_does_not_touch_init_when_no_block_present` (`:390-408`) stay green; full suite
      `python3 -m pytest tests/ -q --tb=line` green, zero net regressions.
- [ ] `import re` removed iff no `re.` use remains; module docstring/comment updated to AST wording.
- [ ] No new hard dependency added (`libcst` optional/unused; stdlib `ast` only).
- [ ] `ROADMAP_V2.md:1360` ticked/annotated with the fix SHA; #1 cited as dependency.
- [ ] PR opened on a `hardening/` branch; held-out tests pass before merge; no scratch files committed.
