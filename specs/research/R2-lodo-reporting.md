---
item: "R2 — LODO honest-accuracy reporting"
dedup_status: "ALREADY-TRACKED — duplicate of BM-13 (BENCHMARK_SPRINT.md:182) + the DI-decontam track (spec 17). CI plumbing already DONE (IMP-2, BENCHMARK_SPRINT.md:217-218)."
applicable_steps: [1, 2, 3, "5 (audit-only)", 6, 8, 9, 10]
na_steps: [4, 7, 11-as-GitHub]
skills: [eval-harness, na0s-review-checklist]
depends_on: ["spec 17 (DI-1/DI-2 sealed corpus on main; DI-3 retrain)", "DVC corpus access (dvc pull OR auto-retrain.yml CI)"]
local_only: true
scope: THIN — execution is a one-line merge into BM-13, NOT a new module
---

# R2 — LODO honest-accuracy reporting (Leave-One-Dataset-Out)

> **PLAN-ONLY spec.** The only write at execution time is a one-line edit into
> BM-13 in `BENCHMARK_SPRINT.md` (and, if the optional harness extension is
> chosen, a `--lodo` flag in `scripts/technique_analysis.py`). **LOCAL ONLY — no
> GitHub push/PR at any step.** All file:line claims verified against worktree
> `/Users/mehrnoosh/Na0S-wt/research-items` (branch `research/local-items`, base
> advanced via #476).

## TL;DR — what execution actually does

R2 is **not new work**. It is a truth-up: fold "per-fold LODO recall + Wilson/
bootstrap CI + in/out-of-distribution gap reporting" into the existing **BM-13**
checklist item so the LODO task explicitly inherits the CI machinery that is
already shipped (`wilson_ci`, `bootstrap_ci`, `rogan_gladen_ci`). The *execution*
deliverable is one merged bullet in BM-13. Everything heavier (the actual LODO
retrain loop) is owned by **BM-13 itself** and **spec 17 (DI-3 retrain)**, both
of which are blocked on DVC corpus access — so R2 must NOT silently re-scope into
"build the LODO harness". Its honest job is to make BM-13 specify *how* LODO will
report numbers (per-fold, CI-bounded, with an in/out-of-distribution gap), not to
run it.

---

## DEDUP STATUS (re-verified against THIS worktree)

| Claim | Verified location | Status |
|---|---|---|
| LODO is already a roadmap item | `BENCHMARK_SPRINT.md:182` `**BM-13**: LODO evaluation (Leave-One-Dataset-Out)` — "Train on N-1 datasets, test on held-out one / Standard evaluation inflates AUC by 8.4%" | **DUP — BM-13 owns LODO.** |
| CI computation is DONE | `BENCHMARK_SPRINT.md:217` (IMP-2): "Confidence interval computation — DONE (2026-06-20): `src/na0s/judge/calibration.py` (`bootstrap_ci`, `rogan_gladen_ci`) wired into `scripts/benchmark.py`". Functions confirmed: `calibration.py:122` `bootstrap_ci`, `:152` `rogan_gladen_ci`, `:39` `confusion_metrics`; Wilson at `scripts/technique_analysis.py:69` `wilson_ci`. | **DONE — no CI code to write.** |
| The "CIs DONE (ROADMAP_V2.md:718)" ref in the task | `ROADMAP_V2.md:718` is in the **Layer 1** section, NOT a CI line. The real CI-done record is the IMP-2 bullet above. The task's `:718` is a stale pointer — **note in the merge, do not cite :718.** | **Pointer-correction needed.** |
| DI-6 / DI-1…DI-6 task list | NOT present verbatim in this worktree's `BENCHMARK_SPRINT.md` (lines 26-31 here are CP-1/CP-2, not DI). The DI track lives in `specs/hardening/17-di-decontam-retrain.md` (DI-1/DI-2/DI-5 done on main, **DI-3 retrain open**) and `ROADMAP_V2.md:1395` (Item 17). | **DI-6 is a conceptual ID, not a live checkbox here.** R2 relates to the same decontam/honest-number theme but is its own thin item. |

**Conclusion:** R2 is ALREADY-TRACKED. Execution = annotate BM-13 (and only
optionally extend the harness). No duplicate roadmap line should be created.

---

## Step 1 — RELOAD skills + explore current system vs ideal

**Skills to reload at execution:** `eval-harness` (router for
`technique_analysis.py` / `benchmark.py` / `calibrate_judge.py`, the metric-reading
rules, the decontam precondition) and `na0s-review-checklist` (§4 hollow tests,
§7 arbitrary thresholds, §11 smoke-first, §12 no premature victory).

**What "LODO honest reporting" SHOULD be (ideal):**
For each source dataset *Dᵢ* in the training corpus, train the classifier on the
N-1 other datasets, evaluate on the held-out *Dᵢ*, and report **per-fold recall
with a CI**, plus the **in-distribution vs out-of-distribution gap** (the
standard-eval-minus-LODO delta — the ~8.4% AUC inflation BM-13 cites). The point
is to surface generalization, not the train-on-test upper bound.

**What the system IS today (gap):**
- `scripts/technique_analysis.py` scores the **shipped frozen ensemble** via
  `from na0s.predict import scan` (`technique_analysis.py:617`). It does **NOT
  train** — so it can measure held-out recall but cannot do a true LODO *retrain*
  loop on its own.
- `scripts/model.py:56` `train_model()` reads `combined_data.csv` and does a
  single stratified `train_test_split` (`model.py:123`) + 5-fold CV
  (`model.py:106-107`) — **k-fold CV is NOT LODO** (folds are random, not
  per-source). No `--lodo` / leave-one-*dataset*-out path exists
  (grep for `lodo|leave.one|held-out dataset` in `technique_analysis.py`,
  `benchmark.py`, `model.py` → 0 hits).
- The corpus **does** carry the per-source labels LODO needs:
  `scripts/process_data.py:143` builds `source_counts` keyed by dataset name and
  carries "per-source taxonomy provenance" (`process_data.py:71-72, 108-110`)
  through `combined_data.csv` — so a future LODO loop has its fold key. ✅
- CI machinery is **fully present and keyless** (`bootstrap_ci`,
  `rogan_gladen_ci`, `wilson_ci`) — nothing to build.

**Gap vs ideal (the honest one-line):** BM-13 says "Train on N-1, test on
held-out" but does **not** specify (a) per-fold CI reporting, (b) the
in/out-of-distribution **gap** as the headline, or (c) that it must reuse the
already-shipped CI functions. R2 closes that specification gap in BM-13.

## Step 2 — Read roadmap / taxonomy / README / coverage matrix / benchmark

- **Roadmap/benchmark:** BM-13 (`BENCHMARK_SPRINT.md:182-184`); IMP-2 CI-done
  (`BENCHMARK_SPRINT.md:217-218`); BM-1/BM-2 number blocks are flagged
  contaminated-pending-DI-3 by spec 17 (`specs/hardening/17-di-decontam-retrain.md:74`).
- **Coverage matrix:** `docs/COVERAGE_MATRIX.md` headline is **measured @ 0.55**
  from `technique_analysis.json` (overall recall 57.1%, benign FPR 1.2%). These
  are **standard-eval (in-distribution-ish)** numbers — LODO would be the
  out-of-distribution complement, so the matrix's "Basis = measured" rows are the
  baseline R2's gap is measured *against*.
- **Taxonomy:** N/A as a content edit — R2 is eval infrastructure, it adds no
  attack-class row to `data/taxonomy.yaml`. (See Step 10.)
- **README:** benchmark table is the standard-eval numbers; R2 does not change
  them, only adds the framing that LODO numbers (when produced) are the honest
  generalization figure.

## Step 3 — Root-cause + numbered LOCAL implementation plan

**Root cause of the "gap":** BM-13 was written as a one-line aspiration before
the CI machinery existed (pre-2026-06-20 IMP-2). Now that `bootstrap_ci` /
`rogan_gladen_ci` / `wilson_ci` are shipped, BM-13 is under-specified: it never
says LODO must *use* them or report the distribution gap. R2 = make BM-13 say so.

**LOCAL implementation plan (minimal, the actual execution):**

1. **Edit BM-13** in `BENCHMARK_SPRINT.md:182-184` to add three sub-bullets,
   each grounded in an existing symbol (no new APIs):
   - `Per-fold LODO recall with Wilson CI (scripts/technique_analysis.py:69 wilson_ci) — one fold per source dataset (fold key = process_data.py:143 source_counts).`
   - `Bootstrap CI on the blended per-fold metric (src/na0s/judge/calibration.py:122 bootstrap_ci) — reuse the IMP-2 plumbing, do NOT reimplement.`
   - `Report the in-distribution vs out-of-distribution GAP (standard-eval recall − LODO recall) as the headline honesty signal — this is the ~8.4% AUC inflation BM-13 already cites.`
2. **Add a cross-reference line** under BM-13 pointing at spec 17 (DI-3 retrain)
   and noting the **DVC-corpus precondition** — LODO cannot run locally without
   `dvc pull` or `auto-retrain.yml` CI (same blocker as DI-3,
   `specs/hardening/17-di-decontam-retrain.md:32`). State that LODO and DI-3
   share the retrain loop, so they should be executed together, not twice.
3. **Correct the stale pointer:** the orchestration prompt's "CIs DONE
   (ROADMAP_V2.md:718)" is wrong; the real record is IMP-2. Reference IMP-2 in
   the merge, not `:718`.

**OPTIONAL harness extension (only if execution scope is upgraded beyond the
one-liner — flag to the user first):**

4. Add a `--lodo` mode to `scripts/technique_analysis.py` that, *given a corpus
   on disk*, iterates source datasets, shells the existing retrain
   (`scripts/model.py:train_model`) on N-1, and re-runs the existing
   `analyze_by_category` against the held-out fold, emitting per-fold
   `recall_ci` via the existing `wilson_ci` and a `lodo_gap` field vs the frozen
   standard run. **No new threshold constants** — reuse `_DEFAULT_THRESHOLD =
   0.55` (`model.py:32`) and the harness `--threshold` default (0.55,
   `technique_analysis.py:580`). Re-uses `_MIN_SAMPLES = 100` (`model.py:29`) as
   the per-fold floor; a fold below it is **skipped with a stamped warning**, not
   silently gated (parity with the `--max-samples`/`truncated` discipline at
   `technique_analysis.py:597-601`). This is BM-13's real body and is
   **DVC/retrain-blocked**; spec it, do not run it locally.

## Step 4 — Wire new capability into predict.py / cascade.py

**N/A — R2 is an eval-reporting/governance item, not a detector.** It touches no
runtime classification path; `predict.py`/`cascade.py` neither know nor need to
know about LODO. (The optional harness extension lives in `scripts/`, off the
runtime path.)

## Step 5 — Harvested-dataset audit

**Audit-only / mostly N/A.** R2 is eval-integrity infra, not an attack class, so
there is no "attack dataset to move into an isolated test file." The relevant
audit is on the **corpus structure, not new data**: confirm the fold key exists —
`process_data.py:143` `source_counts` and the per-source provenance carried into
`combined_data.csv` (`process_data.py:71-72, 108-110`). That provenance **is**
what makes LODO possible; if a future harvested source lands without a `source`
tag, LODO loses a fold key. So the only harvester check is: **every ingested
source must stamp its dataset name** (already done by `sync_datasets.py` per the
provenance comment). No harvester cron change needed for R2.

## Step 6 — Tests for Code + behavior

Tests apply **only if the optional harness extension (Step 3.4) is built**;
if execution is the pure one-line BM-13 merge, this is **N/A — doc-only edit,
covered by the docs-drift gate (Step 9)**.

If the `--lodo` flag is added, mirror the existing pattern in the **flat**
`tests/test_technique_analysis.py` (core-pipeline tests stay at `tests/` root per
CLAUDE.md; this file already exists with `TestWilsonCI` at line 383):
- **Code test (has teeth — §4):** feed a tiny 2-source synthetic corpus where
  source-A and source-B have *disjoint* attack strings; assert the LODO loop
  produces 2 folds, each fold's held-out source absent from that fold's training
  set (prove leave-one-out actually leaves it out — break it by NOT excluding the
  fold and confirm the test fails).
- **CI test:** assert each fold emits a `recall_ci` 2-tuple from `wilson_ci` and
  that `lodo_gap = standard_recall − lodo_recall` is finite and sign-correct on a
  rigged fold where the held-out source is unseen-hard (gap > 0).
- **Skip-floor test:** a fold with n < `_MIN_SAMPLES` is **skipped + stamped**,
  not gated (mirror `truncated` handling).
- **No hollow mocks (§4):** do NOT mock `train_model`/`scan` away to the point the
  retrain never runs; the smoke step (Step "CLI smoke") must exercise the real
  retrain on the synthetic corpus at least once.
- **No arbitrary thresholds (§7):** the only numbers are the inherited 0.55 and
  `_MIN_SAMPLES=100`; assert tests cite their origin, introduce none new. There is
  **no similarity cutoff** in R2 (so the short-injection-string recalibration
  caveat does not bite here — note that explicitly).

## Step 7 — File/dir cleanup & refactor

**N/A for the one-line merge.** If the optional flag is built: the new code goes
**into the existing** `scripts/technique_analysis.py` (no new top-level module),
tests into the existing `tests/test_technique_analysis.py` — both already exist,
so no new dir, no shim, no cleanup debt created.

## Step 8 — Update the roadmap

- `BENCHMARK_SPRINT.md` BM-13 (`:182-184`): merge the three CI/gap sub-bullets +
  the spec-17/DVC cross-reference (Step 3.1-3.3). Do **not** check BM-13 `[x]` —
  the LODO *run* is still blocked on the corpus; only its *specification* is
  tightened.
- `ROADMAP_V2.md`: no new line (R2 is folded into BM-13, not a standalone
  roadmap entry — avoids the duplicate the dedup status warns against). If a
  cross-link to spec 17 / BM-13 already exists post-#476, leave it; if absent,
  add a one-line "LODO honest-reporting → BM-13" pointer near the DI/Item-17
  region (`ROADMAP_V2.md:1395`).
- Cite the local commit SHA in the BM-13 bullet only if an actual local commit is
  made (per roadmap-sync rule). **No GitHub.**

## Step 9 — README + Benchmark accuracy

- **README:** no number change. R2 only adds (in BM-13) the framing that the
  README's standard-eval table is in-distribution and the honest generalization
  figure is the future LODO number. Touch README only if it currently *claims*
  LODO is done (it does not — grep `lodo` in README = 0 hits).
- **Benchmark:** the BM-13 edit is the benchmark update. Keep the **docs-drift
  gate green** — if `docs/facts.yaml` carries any benchmark counts, the BM-13
  prose edit must not contradict them (a doc-only edit; run the docs-drift check
  in the smoke step).

## Step 10 — Taxonomy + Coverage Matrix + thresholds

- **Taxonomy:** **N/A — R2 adds no attack-class code.** It is eval infra; no
  `data/taxonomy.yaml` row, no dup risk.
- **Coverage Matrix:** one honest annotation only — `docs/COVERAGE_MATRIX.md`'s
  "measured @ 0.55" headline is **standard-eval**; note (when LODO runs) that the
  measured rows are in-distribution and the LODO number is the out-of-distribution
  complement. Per spec 17, holdout-sourced cells are already flagged for
  re-baseline post-DI-3; R2 piggybacks that note, it does not move a cell now.
- **Thresholds:** no new threshold. R2 reuses 0.55 (`config.DEFAULT_THRESHOLD`,
  `model.py:32`, harness default `:580`) and `_MIN_SAMPLES=100`. **No similarity
  cutoff is introduced** — so the "MUST re-calibrate similarity cutoffs for short
  injection strings" rule is **N/A here** (state this so a future reader doesn't
  hunt for a cutoff that doesn't exist).

## Step 11 — NO GitHub

**Enforced.** All work local. No push, no PR, no `gh`. The user authorizes git
commit/push from worktrees only on explicit say-so; R2 stays at the working-tree
/ local-commit boundary at most.

---

## Q&A self-checks

- **Q1 (can Na0S handle R2 / run the suite?):** R2 is reporting infra, not a
  detector — there is nothing to "scan." The relevant verification is the harness
  + suite smoke (below), not a `scan()` run. The CI functions it leans on already
  pass their unit tests (`tests/test_technique_analysis.py::TestWilsonCI`,
  `tests/judge/test_benchmark_metrics.py`).
- **Q2 (cleanup):** No clutter — one-line doc merge, or code into existing
  files; no new module/dir/shim.
- **Q3 (pipeline wiring):** N/A — off the runtime path; correctly NOT wired into
  predict.py/cascade.py.
- **Q4 (tested for code AND use-case):** Code test only if the optional flag is
  built (Step 6); the one-line merge is covered by the docs-drift gate.
- **Q5 (harvester audit):** Audit-only — the fold-key provenance
  (`process_data.py:143` `source_counts`) is the harvester dependency; no new
  dataset, no cron change. N/A as "move dataset to isolated test file."
- **Q6 (taxonomy+coverage, no dups):** R2 adds no taxonomy code; coverage gets one
  in/out-of-distribution annotation. No dup — BM-13 already owns LODO.
- **Q7 (does the scorer score R2 right?):** The scorer (`technique_analysis.py`
  recall + Wilson CI) is exactly the machinery LODO reuses; per-fold it scores
  held-out recall correctly. The only addition is the `lodo_gap` field. ✅
- **Q8 (do predict.py/cascade.py reference R2?):** **N/A — eval infra, no runtime
  reference, by design.**
- **Q9 (does the harvester agent harvest this type?):** **N/A — R2 is not an
  attack type to harvest.**
- **Q10 (other correctness):** (a) Confirm LODO and DI-3 share the retrain loop —
  do not build two; (b) confirm DVC corpus is the hard blocker before claiming
  any LODO *number* (`specs/hardening/17-di-decontam-retrain.md:32`); (c) keep the
  no-precision/F1 posture — LODO reports **recall + CI + gap**, not a blended F1
  (matches `technique_analysis.py:655-660` rationale: TP and FP come from
  independent pools).

---

## CLI / suite smoke step (na0s-review-checklist §4/§12 — mandatory at execution)

1. **Doc-only path:** run the docs-drift / facts gate after the BM-13 edit
   (e.g. `python3 -m pytest tests/ -q -k "facts or drift or docs"` and any
   `scripts/check_*facts*`), confirm green. Show the command + output, don't
   summarize.
2. **If the `--lodo` flag is built:**
   `python3 scripts/technique_analysis.py --help` (confirm `--lodo` parses),
   then a real end-to-end smoke on the tiny synthetic 2-source corpus —
   the **retrain must actually run**, not be mocked (§4 / §11 smoke-first), then
   `python3 -m pytest tests/test_technique_analysis.py -q`.
3. **Full suite** before declaring done: `python3 -m pytest tests/ -q --tb=line`
   (~15 min, ~8000 tests) — confirm zero regressions (CLAUDE.md).

---

## Execution preconditions / dependencies

- **Base off the advanced branch** (`research/local-items` / post-#476), NOT the
  stale primary checkout — same lesson as spec 17 (which warns base-off-main or
  reintroduce the DI-1 leak). Verify renames/CI plumbing against this worktree's
  `src/`, with `PYTHONPATH=<this-worktree>/src` if the editable install points
  elsewhere.
- **CI machinery present** (verified): `calibration.py:122 bootstrap_ci`,
  `:152 rogan_gladen_ci`, `technique_analysis.py:69 wilson_ci` — keyless,
  numpy-only.
- **DVC corpus** — the only true blocker for an *actual* LODO run (per-fold
  retrain). `dvc pull` OR run inside `auto-retrain.yml`. Without it, R2 can only
  **specify** LODO, not produce numbers. The one-line BM-13 merge has **no**
  corpus dependency.
- **Depends-on:** spec 17 (DI-3 sealed-corpus retrain shares the loop);
  `data/processed/combined_data.csv` carrying the per-source fold key.
- **No API key** required at any step (keyless throughout).

## Definition of done

- [ ] **(Required)** BM-13 (`BENCHMARK_SPRINT.md:182-184`) carries the three
      CI/gap sub-bullets + the spec-17/DVC cross-reference; the stale
      `ROADMAP_V2.md:718` pointer is replaced by the IMP-2 reference.
- [ ] BM-13 is **not** checked `[x]` (LODO run still corpus-blocked); only its
      spec is tightened.
- [ ] No duplicate roadmap line created; ROADMAP_V2 cross-link to BM-13 present
      (added only if absent post-#476).
- [ ] Docs-drift / facts gate green after the edit (command + output shown).
- [ ] **(Optional, only if flag built)** `--lodo` parses, retrains on N-1 in a
      real smoke run, emits per-fold `recall_ci` + `lodo_gap`, reuses 0.55 /
      `_MIN_SAMPLES` with no new thresholds; teeth-proven tests in the existing
      `tests/test_technique_analysis.py`; full suite zero-regression.
- [ ] Coverage-matrix in/out-of-distribution annotation added (no cell moved).
- [ ] **LOCAL ONLY** — no push, no PR, no `gh` at any step.
