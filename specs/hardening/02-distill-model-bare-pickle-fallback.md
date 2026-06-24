---
item: 2
title: "H1 — distill_model.py ImportError fallback uses bare pickle / no sidecar"
priority_tier: H1 (high — supply-chain / integrity)
category: supply-chain / model-integrity hardening
depends_on: []           # self-contained; no other hardening item must land first
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]
na_steps: [5, 9, 10]
applicable_qs: [Q1, Q2, Q3, Q4, Q10]
na_qs: [Q5, Q6, Q7, Q8, Q9]
files_touched:
  - scripts/distill_model.py            # remove fallback, hoist hard import
  - tests/integrity/test_distill_model_pickle.py   # NEW
roadmap_refs:
  - ROADMAP_V2.md:747   # distill_model.py line in L5 file tree
  - ROADMAP_V2.md:736,754  # "safe_dump/safe_load with SHA-256 sidecars" claim
---

# H1 — distill_model.py ImportError fallback uses bare pickle / no sidecar

## 0. Confirmed root cause (refs verified against live file)

Original KEY REFS named `scripts/distill_model.py:8-12,30-40,243-267,271,301`.
After opening the file the lines drifted slightly; **corrected** locations:

| Concern | Original ref | Actual line(s) | Note |
|---|---|---|---|
| Module docstring / usage | 8-12 | 8-13 | accurate |
| sklearn dependency guard (`_HAS_SKLEARN`) | 30-40 | 30-40 | accurate; **unrelated** to this bug (sklearn guard is fine, it hard-fails in `main()` at 243-250) |
| sklearn hard-fail in `main()` | — | 243-250 | this is the *correct* pattern the pickle path should mirror |
| **Bare-pickle fallback block** | 243-267 | **254-267** | THE BUG |
| `safe_load(args.tfidf_features)` call | 271 | 271 | accurate |
| `safe_dump(student, args.output)` call | 301 | 301 | accurate |

**The defect (lines 254-267):**

```python
# Try to use safe_pickle if available, fall back to regular pickle
try:
    from na0s.safe_pickle import safe_load, safe_dump
except ImportError:
    import pickle

    def safe_load(path):
        with open(path, "rb") as f:
            return pickle.load(f)          # <-- arbitrary-code-execution unpickle, no integrity check

    def safe_dump(obj, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)            # <-- writes model WITHOUT a .sha256/.hmac sidecar
```

Two distinct supply-chain harms in the `except` branch:

1. **Load side (271):** bare `pickle.load` deserializes the teacher-features `.pkl`
   with **no integrity verification**. `na0s.integrity.safe_pickle.safe_load`
   (real impl at `src/na0s/integrity/safe_pickle.py:295-363`) validates pickle
   magic bytes (`_validate_pickle_magic`, line 128), resolves an expected digest
   from `KNOWN_HASHES` / HMAC / SHA-256 sidecar (`_resolve_expected_hash`, 214),
   and constant-time-compares before unpickling (335). The fallback skips ALL of
   that — a tampered `features.pkl` executes arbitrary code on load.

2. **Dump side (301):** bare `pickle.dump` writes the distilled model with **no
   sidecar**. `safe_dump` (real impl 247-291) always emits a `v1:` sidecar
   (HMAC-SHA256 when `NA0S_PICKLE_KEY` is set, else plain SHA-256). A model
   produced by the fallback has no sidecar, so any *downstream* `safe_load` of
   it raises `FileNotFoundError` ("No integrity hash available", line 239) —
   a silent supply-chain hole that surfaces only at deploy/load time.

**Why the fallback is also pointless (dead-defensive code):**
`na0s.safe_pickle` is a backward-compat shim (`src/na0s/safe_pickle.py:1-14`)
that re-exports `na0s.integrity.safe_pickle`. Both import paths resolve in the
real env (verified: `PYTHONPATH=src:. python3 -c "from na0s.safe_pickle import
safe_load, safe_dump"` → OK; canonical path → OK). The `except ImportError`
branch therefore only fires when **`na0s` is not importable at all** — i.e. the
script is being run completely outside the package, in which case silently
downgrading to unverified pickle is exactly the wrong behavior. Every sibling
training script already does the right thing: a top-level hard import with **no
fallback** — `scripts/features.py:18`, `scripts/model.py:19`,
`scripts/model_embedding.py:30`, `scripts/canary_eval.py:34`,
`scripts/optimize_threshold.py:21`, `scripts/build_faiss_index.py:45`,
`scripts/features_embedding.py:33`. `distill_model.py` is the lone outlier.

**Ideal state:** `distill_model.py` imports `safe_load`/`safe_dump` at module
top level with no fallback, matching its 7 sibling scripts; if `na0s` is not
importable the script aborts with a clear error and non-zero exit, and **no
bare `pickle` import remains anywhere in the file**.

---

## 1. Explore: current vs ideal, gaps & edge cases  — APPLICABLE

**Current behavior (gap):** lazy, in-`main()` import with a silent bare-pickle
fallback. Loads unverified pickles; writes sidecar-less models.

**Ideal:** top-level `from na0s.safe_pickle import safe_load, safe_dump`
(sibling convention) — or the canonical `from na0s.integrity.safe_pickle import
safe_load, safe_dump`. **Decision: use the canonical path** `na0s.integrity.safe_pickle`
to avoid the shim's `DeprecationWarning` (shim raises it at import,
`src/na0s/safe_pickle.py:7-11`) and because CLAUDE.md says new code should
target organized sub-packages. (Siblings still use the shim; migrating them is
out of scope for this item — note it as follow-up, do not touch them here.)

**Edge cases to cover in the plan/tests:**
- E1: `na0s` not importable → script must abort at import time (ModuleNotFoundError
  surfaces) — NOT silently use bare pickle. After the fix this is automatic
  because the import is top-level and unguarded.
- E2: legit `features.pkl` with a valid sidecar still loads through the real
  `safe_load` (no regression to the happy path).
- E3: tampered `features.pkl` (sidecar digest mismatch) is **rejected** with the
  real `safe_load` `ValueError("Integrity check failed ...")` — the whole point.
- E4: `safe_dump(student, output)` now writes BOTH the `.pkl` and a sidecar
  (`.sha256` keyless / `.hmac` when `NA0S_PICKLE_KEY` set).
- E5: **No bare `import pickle` / `pickle.load` / `pickle.dump` token remains**
  in the file (grep-level assertion in the test).

---

## 2. Roadmap / taxonomy / README / coverage cross-read  — APPLICABLE (partial)

- **ROADMAP_V2.md:747** lists `distill_model.py` in the L5 file tree.
- **ROADMAP_V2.md:736 & 754** both assert L5 "Model files load via
  `safe_dump`/`safe_load` with SHA-256 sidecars." The current fallback
  *violates* that documented invariant for the distillation path. The fix makes
  the code match the roadmap claim — this is a doc-truth reconciliation, not a
  new feature.
- No dedicated H1 / "bare-pickle" line exists in ROADMAP_V2.md or
  BENCHMARK_SPRINT.md yet (grep clean). Step 8 will add a checked-off entry.
- Taxonomy/Coverage Matrix: see Step 10 (N/A) — this is loader integrity, not a
  detection class with a TPR/FPR row.

---

## 3. Root-cause implementation plan (numbered)  — APPLICABLE

1. **Delete** the fallback block `scripts/distill_model.py:254-267` in its
   entirety (the comment line 254 through the `safe_dump` def ending line 267).
2. **Hoist** a single hard import to the top-level import group (alongside the
   existing `import numpy as np` at line 23). Add:
   `from na0s.integrity.safe_pickle import safe_load, safe_dump`
   Place it after the third-party `import numpy as np` and before
   `logger = logging.getLogger(__name__)` (line 25), matching the first-party
   import position used by siblings (e.g. `scripts/model.py:19`).
3. **Remove** the now-unused `import os`? — **NO**: `os` is still used by
   `os.makedirs(...)` at line 300 (the dump dir-create). Keep `import os`.
   Verify with grep that `os.` still has a live use after the edit (it does:
   line 300). Do not over-prune.
4. **Confirm** nothing else in the file references the local fallback symbols.
   The only call sites are `safe_load` (271) and `safe_dump` (301); both now
   bind to the real package functions. No signature change (`safe_load(path)` /
   `safe_dump(obj, path)` match the real API at
   `src/na0s/integrity/safe_pickle.py:295` and `:247`). Verified — no
   hallucinated args.
5. **Leave the sklearn guard untouched** (lines 30-40 + 243-250). It is a
   *correct* hard-fail-in-`main()` pattern and is unrelated to the pickle bug.
   Do NOT refactor it in this item (keep the diff minimal and reviewable).
6. **Optional belt-and-suspenders (low priority, only if review asks):** the
   top-level import failing produces a raw traceback. Siblings accept that. If a
   friendlier message is wanted, wrap module run in `main()` is overkill — leave
   as raw `ModuleNotFoundError`, which is unambiguous and matches siblings.
   Decision: **do not add** a second guard; minimal diff.

**Resulting diff shape:** ~ +1 import line at top, −14 lines (254-267 block
removed). Net negative LOC. No behavior change on the happy path; the dangerous
silent-downgrade path is eliminated.

---

## 4. Implement + wire (predict.py / cascade.py parity)  — APPLICABLE (no pipeline wiring)

This is a **standalone training/distillation script**, not a runtime detector.
There is nothing to wire into `predict.py` / `cascade.py` (see Q8). The "wiring"
that matters here is that the script now depends on the SAME integrity contract
(`na0s.integrity.safe_pickle`) that the runtime model loaders use
(`src/na0s/ml/predict_embedding.py:46`, `src/na0s/dataset/hard_negatives.py:24`)
— so a model emitted by `distill_model.py` carries a sidecar that the runtime
`safe_load` can verify. That is the end-to-end correctness link to assert in
Step 6 / Q4.

Agent: **l3-l5-code-auditor** owns the edit (L5/distillation territory), with
**silent-failure-hunter** reviewing that the removed `except` didn't mask any
other failure mode. Inject `na0s-review-checklist` (sections: hallucinated APIs,
import blindness, silent refactor destruction) into both prompts.

---

## 5. Harvester audit / harvested datasets  — N/A

N/A — this is loader/serialization integrity for a training script; there is no
attack-string dataset to harvest. (Crafted-malicious-pickle datasets are
explicitly scoped to item 8, not item 2.) The test instead synthesizes its own
tiny fixtures (a valid `.pkl`+sidecar and a tampered `.pkl`) in a tmp dir.

---

## 6. Tests: Code + Use-Case  — APPLICABLE

New file: **`tests/integrity/test_distill_model_pickle.py`** (mirrors source —
integrity concern, lands under `tests/integrity/`; reference pattern:
`tests/integrity/test_safe_pickle.py` and `tests/test_deploy_model.py`).
Use `unittest` + `unittest.mock`, no network, no real model dirs.

**A. Code-level (structural / regression-proof):**
- `test_no_bare_pickle_fallback`: read the source of `scripts/distill_model.py`
  and assert it contains **no** `import pickle`, no `pickle.load`, no
  `pickle.dump`, and no `except ImportError` guarding the safe_pickle import.
  (Source-text assertion so the fallback can't silently return.)
- `test_safe_pickle_imported_at_top`: assert the module source contains
  `from na0s.integrity.safe_pickle import safe_load, safe_dump` at top level
  (regex anchored, not inside a function body).
- `test_module_imports_and_symbols`: `import scripts.distill_model as mod`;
  assert `mod.safe_load` and `mod.safe_dump` are the SAME objects as
  `na0s.integrity.safe_pickle.safe_load/safe_dump` (`is` identity) — proves the
  fallback definitions are gone and the real impl is bound.
- `test_no_import_guard_swallows_failure` (E1): simulate `na0s` missing by
  removing `na0s.integrity.safe_pickle` from `sys.modules` and inserting a
  `meta_path` finder that raises `ModuleNotFoundError` for that name, then
  `importlib.reload(scripts.distill_model)` and assert it raises (NOT a silent
  success). Restore `sys.modules`/`meta_path` in `tearDown`.

**B. Use-Case / behavior (end-to-end loader contract):**
- `test_legit_pickle_roundtrips` (E2/E4): `safe_dump` (via the bound `mod.safe_dump`)
  a small `(X, y)` tuple to a tmp path; assert a sidecar file exists
  (`<path>.sha256` keyless); `mod.safe_load` it back and assert equality. Proves
  the happy path still works AND a sidecar is now produced.
- `test_tampered_pickle_rejected` (E3): write a valid pickle + matching sidecar,
  then flip a byte in the `.pkl` (leave sidecar stale); assert `mod.safe_load`
  raises `ValueError` whose message contains "Integrity check failed". Proves
  the integrity gate the fallback used to bypass is now enforced through the
  distill path.
- `test_sidecarless_pickle_rejected` (regression for the dump-side harm): write
  a bare pickle with NO sidecar and NOT in `KNOWN_HASHES`; assert `mod.safe_load`
  raises `FileNotFoundError` mentioning "No integrity hash available". Proves a
  model the OLD fallback would have produced is correctly refused by the new
  loader contract.

**Assertion discipline (anti-hollow):** every test asserts a concrete outcome
(identity, file existence, specific exception type + message substring) — no
bare `assertTrue(True)` / smoke-only tests. Numbers used are not thresholds;
they are fixture sizes. No magic security threshold introduced (this item adds
none).

**Smoke step (CLI):** run
`PYTHONPATH=src:. python3 scripts/distill_model.py --help` and assert exit 0 +
usage text (the import now executes at module load, so `--help` proves the
top-level import resolves). This is the mocked-CLI-gap guard from the checklist.

Agents: **l3-l5-code-auditor** authors tests; **silent-failure-hunter** reviews
that E1/E3 actually fail loudly. Skill: **na0s-debugging** if reload/meta_path
mechanics misbehave under the full suite (import-cache traps).

---

## 7. Cleanup / refactor per conventions  — APPLICABLE

- The fix itself IS the cleanup: removes a dead, dangerous fallback; aligns the
  script with its 7 siblings.
- Confirm no orphaned `import os` / `import sys` after edit — `os` stays (line
  300 dump dir), `sys` stays (line 248/250 sklearn-guard stderr + exit). Do not
  prune in-use imports.
- New test goes in `tests/integrity/` (matching sub-package), NOT `tests/` root
  — per CLAUDE.md test-org rule. `__init__.py` already exists there.
- **Follow-up (do NOT do in this item, log only):** the 7 sibling scripts import
  the deprecated shim `na0s.safe_pickle`; a separate `refactor/` branch should
  migrate them to `na0s.integrity.safe_pickle`. Out of scope here to keep the
  diff atomic.

---

## 8. Roadmap update (cite SHA on completion)  — APPLICABLE

- Add a checked line under the L5 / supply-chain hardening section of
  ROADMAP_V2.md (near :747): "H1 distill_model.py — removed bare-pickle
  ImportError fallback; now hard-imports `na0s.integrity.safe_pickle`; emits +
  verifies sidecars. (SHA: <fill at commit>)".
- This also makes the existing :736/:754 "safe_dump/safe_load with SHA-256
  sidecars" claim TRUE for the distillation path (previously violated).
- Per the Roadmap-Todo Sync memory: the todo + its check-off both live in
  ROADMAP_V2.md; cite the commit SHA when pushed.

---

## 9. README / Benchmark updates  — N/A

N/A — no public-API surface, metric, or benchmark number changes. This is an
internal script integrity fix with zero detection-behavior impact; README
quickstart and BENCHMARK_SPRINT numbers are unaffected. (If the L5 README
paragraph ever enumerated a "bare pickle" caveat it would need editing, but it
does not — it already claims sidecars, which the fix now honors.)

---

## 10. Taxonomy + Coverage Matrix + per-feature thresholds  — N/A

N/A — `distill_model.py` is a training utility, not a detector with a taxonomy
code or a COVERAGE_MATRIX TPR/FPR row. No per-attack scorer threshold applies.
(Supply-chain *detection* taxonomy lives in `scripts/taxonomy/supply_chain.py`
and is exercised by item 8's crafted-pickle work, not by this loader fix.) No
threshold is added by this change, so there is nothing to justify.

---

## 11. PR + held-out test gate  — APPLICABLE

- Branch: `hardening/distill-bare-pickle` off `main` (rename is DONE on main;
  `na0s.integrity.safe_pickle` exists there — verified). Work in a git worktree
  per the multi-agent discipline; do NOT branch-switch the primary checkout.
- PR body: cite the root cause (silent bare-pickle downgrade), the 14-line
  removal, the sibling-parity rationale, and the new integrity tests.
- **Gate:** targeted first — `python3 -m pytest tests/integrity/ -v` (must be
  green, incl. the new file) — then the FULL suite
  `python3 -m pytest tests/ -q --tb=line` with zero net regressions (CLAUDE.md
  mandate; ~8000 tests, ~15 min). Confirm against MAIN-equivalent env
  (`PYTHONPATH=<worktree>/src`) per the na0s-debugging trap, since the editable
  install may point at a stale checkout.
- Skill **github-pr-prep** to assemble the PR; **github-ci-fix** only if CI goes
  red. Use **pr-review-toolkit:review-pr** / **github-pr-review** for the review
  pass; inject `na0s-review-checklist` into the reviewer prompt.

---

## Q&A self-check

- **Q1 — Can Na0S handle the target (threat/bug) + suite green?**
  After fix: yes. The bug (silent unverified-pickle downgrade) is removed; the
  script now hard-fails on missing `na0s` and routes all (de)serialization
  through the integrity-checked loader. Suite must stay green (Step 11 gate).
- **Q2 — Cleanup done?** Yes — dead fallback removed, test in correct sub-dir,
  sibling-shim migration logged as separate follow-up (not gold-plated here).
- **Q3 — Pipeline wiring correct?** No runtime pipeline wiring needed (training
  script). The relevant "wiring" — shared integrity contract with runtime
  loaders — is asserted by the sidecar roundtrip + downstream-load tests (Step 6).
- **Q4 — Tested for code AND use-case?** Yes — code-level (no bare pickle, top
  import, symbol identity, loud import failure) + use-case (legit roundtrip with
  sidecar, tampered rejected, sidecarless rejected, CLI `--help` smoke).
- **Q5 — Harvester audit?** N/A — no harvested dataset for a loader-integrity fix.
- **Q6 — Taxonomy + Coverage Matrix?** N/A — not a detector row.
- **Q7 — Scorer thresholds?** N/A — no scorer / no threshold added.
- **Q8 — predict.py / cascade.py references to target?**
  N/A — grep confirms `distill` appears only in `scripts/distill_model.py`,
  `tests/ml/test_l5_advanced.py`, and ROADMAP prose; neither `predict.py` nor
  `cascade.py` references the distill script or its pickle path. No parity edit.
- **Q9 — Harvester harvests this type?** N/A — not an intel/attack type.
- **Q10 — Other correctness checks:**
  (a) confirm `import os`/`import sys` remain in use post-edit (they do: 300 / 248);
  (b) confirm `safe_load`/`safe_dump` arg signatures match the real impl
  (verified: `:295`, `:247`) — no hallucinated kwargs;
  (c) confirm the new test's `meta_path`/`sys.modules` mutation is fully restored
  in `tearDown` so it can't poison the rest of the ~8000-test run
  (import-cache trap from na0s-debugging).

---

## Agent / skill team (per step)

| Step / area | Owner agent | Reviewer / support | Skills (inject na0s-review-checklist into every prompt) |
|---|---|---|---|
| 0-2 root-cause confirm | l3-l5-code-auditor | security-research-auditor | security-review |
| 3-4 implement | l3-l5-code-auditor | silent-failure-hunter | na0s-debugging |
| 6 tests | l3-l5-code-auditor | silent-failure-hunter | na0s-debugging, eval-harness (suite gate only) |
| 7 cleanup | l3-l5-code-auditor | layer-9-11-auditor (integrity adjacency) | — |
| 8 roadmap | (author) | — | — |
| 11 PR + CI | github-pr-prep | github-pr-review / pr-review-toolkit:review-pr | github-ci-fix |

`layer-9-11-auditor` is looped in for step 7 because `safe_pickle` lives in the
integrity (L10/L11) subsystem and it owns that review surface.

---

## Execution preconditions / dependencies

- **Depends-on: none.** Self-contained; does not require any other hardening
  item to land first. `na0s.integrity.safe_pickle` already exists on `main`
  (verified import OK), so the target import resolves immediately.
- **Not blocked by item 8** (crafted-malicious-pickle datasets): item 8 is a
  *detection* concern; this item is a *loader* fix. They can land in any order.
  If item 8 lands first it provides extra adversarial fixtures, but this item's
  tests synthesize their own — no hard dependency.
- Must be done in a git worktree off `main` (multi-agent discipline); verify the
  env with `PYTHONPATH=<worktree>/src` (editable install may be stale).

---

## Definition of done

- [ ] Fallback block `scripts/distill_model.py:254-267` removed entirely.
- [ ] Top-level `from na0s.integrity.safe_pickle import safe_load, safe_dump`
      added near line 23-25.
- [ ] No `import pickle` / `pickle.load` / `pickle.dump` / `except ImportError`
      (around the safe_pickle import) remains anywhere in the file (grep clean).
- [ ] `import os` and `import sys` confirmed still in use (no broken pruning).
- [ ] `scripts/distill_model.py --help` exits 0 (CLI smoke).
- [ ] New `tests/integrity/test_distill_model_pickle.py` covers: no-bare-pickle,
      top-import, symbol identity, loud import-failure (E1), legit roundtrip +
      sidecar (E2/E4), tampered rejected (E3), sidecarless rejected (dump-side).
- [ ] All new tests assert concrete outcomes (no hollow assertions).
- [ ] `python3 -m pytest tests/integrity/ -v` green.
- [ ] Full suite `python3 -m pytest tests/ -q --tb=line` — zero net regressions
      vs main-equivalent baseline.
- [ ] ROADMAP_V2.md H1 line added + checked off with commit SHA.
- [ ] PR opened off `main` via worktree; review pass done; CI green before merge.
