---
item: R3 — Sealed-corpus retrain
classification: Eval-integrity / supply-chain / governance (NOT a prompt-injection attack class)
dedup_status: ALREADY-TRACKED (THIN execution wrapper). Canonical item = DI-3, fully specced in `specs/hardening/17-di-decontam-retrain.md`. This R3 spec is the LOCAL EXECUTION-CHECKLIST companion to #17 — it does NOT re-derive #17's design; it (a) re-verifies the dedup/dependency state against THIS worktree, (b) names the DVC + `auto-retrain.yml` execution path, (c) reconciles the now-MOVED DI-3 tracking box.
applicable_steps: [1, 2, 3 (lightly), 6, 7, 8, 9, 10 (lightly), 11]
na_steps: [4, 5]
applicable_qs: [Q1, Q2, Q3, Q4, Q5, Q6, Q10]
na_qs: [Q7, Q8, Q9]
skills: na0s-review-checklist, eval-harness, data-harvesting, cron-scheduling
agents: silent-failure-hunter (decontam/empty-corpus guard), security-research-auditor (integrity chain + harvester decontam), l3-l5-code-auditor (ML re-baseline), layer-9-11-auditor (L11 KNOWN_HASHES↔bytes)
depends_on: [hardening #1 (KNOWN_HASHES hash-of-record path), hardening #14 (deploy_model AST rewrite + sealed-corpus attestation — VERIFIED LANDED in this worktree, see §Dedup), DVC corpus access]
local_only: true  # NO GitHub push/PR at any step. All work stays local.
---

# R3 — Sealed-corpus retrain (LOCAL execution checklist for DI-3 / spec #17)

> **PLAN-ONLY artefact.** This file is the only write. It does not modify source, roadmap, #17, tests, or run git.
> **LOCAL-ONLY at execution.** No push, no PR, no `gh` at any step until the user explicitly says so.

## 0. One-sentence purpose
DI-3 is the single open piece of the data-integrity track: the bundled classifier in `src/na0s/models/` was fit on a corpus that included the "held-out" eval sets, so every published ML number is a train-on-test upper bound — R3 is the thin task of *actually retraining on the now-sealed corpus and re-stamping the integrity chain*, which can only be executed where the DVC corpus is hydrated (locally via `dvc pull` or in `auto-retrain.yml` CI), then checking off the DI-3 box and re-baselining the numbers.

---

## Dedup status — RE-VERIFIED against THIS worktree (base advanced via #476)

| Claim in the task / spec #17 | State in THIS worktree (`research/local-items`, post-#476) | Action |
|---|---|---|
| DI-3 tracked at `BENCHMARK_SPRINT.md:28` (DI-1…DI-6 list) | **STALE REF — the DI-1/DI-2/DI-3 checkbox block is GONE from `BENCHMARK_SPRINT.md`.** `grep -niE 'DI-[0-9]|decontam|retrain|sealed'` over the file returns only the unrelated judge-calibration line 221. #476 ("docs(roadmap): record hardening campaign …") rewrote the sprint doc and dropped the DI-N boxes. | Step 8 must NOT edit a non-existent box. The live tracking surface is now ROADMAP_V2 + the spec dir (next rows). |
| Canonical open item | `ROADMAP_V2.md:1395` — `[ ] Item 17 — DI-1→DI-3 decontamination + sealed-corpus retrain + sidecar/KNOWN_HASHES refresh … Spec: specs/hardening/17-di-decontam-retrain.md` (unchecked). | THIS is the box R3 execution checks off (with the local-retrain commit SHA, once executed). |
| Cross-link present? | `specs/hardening/README.md:31` already cross-links #17→"only DI-3 retrain open"; `:59-60` already records "cannot fully run locally: corpus is DVC-tracked … needs `dvc pull` or `auto-retrain.yml` CI, base off main." | The DVC/CI execution note R3 was asked to "add" **already exists** in the hardening README. R3 adds the same note to the `research/` track for symmetry; no new claim. |
| #17 `depends_on: [1, 14]` | **#14 (deploy_model AST-rewrite) has LANDED in this worktree.** `scripts/deploy_model.py:147` now AST-locates `KNOWN_HASHES` and rewrites it (NOT the brace-fragile `re.sub(r"KNOWN_HASHES\s*=\s*\{[^}]*\}")` that #17 §1 cited at lines 165-170), and `:7-8` documents it "preserv[es] entries for bundled pickle files that are not re-emitted this run (e.g. model_embedding.pkl, structural_scaler.pkl)." So G5 (the `model_embedding.pkl` refresh gap #17 flagged) is **already resolved by code** — preserved, not clobbered. | One #17 precondition is satisfied; R3 records it so execution does not re-do #14's work. #1 status still to confirm at execution. |
| DI-1 (corpus fix) | DONE here: `scripts/process_data.py:33 TRAINING_JSONL_DIRS = [AGGREGATED_DIR, HARVEST_DIR]`; loop `:162`; "DELIBERATELY EXCLUDED" comment `:159-160`. `tests/test_no_holdout_leakage.py` present (3 tests, incl. the skip-guarded `test_no_holdout_string_in_combined_csv:61`). | Verified — branch base is NOT the contaminated `process_data`. No re-fix needed. |
| Shipped weights still contaminated? | YES — `src/na0s/models/model.pkl` digest in `KNOWN_HASHES` is still `057280f9…`, `tfidf_vectorizer.pkl` `347b2b4e…` (`src/na0s/models/__init__.py:26-31`) — unchanged contaminated-corpus hashes. Sidecars present only for `model.pkl` + `tfidf_vectorizer.pkl`. | The live bug R3 fixes is real and present. |

**Conclusion:** R3 = ALREADY-TRACKED, BLOCKED. It is NOT a new design item — #17 owns the design. R3's only original content is the LOCAL-execution checklist + the re-verified dedup above (notably: the DI-3 box MOVED, and #14 already landed). Do **not** author a competing plan; execute #17 §3 steps 2-8 with the corrections noted here.

---

## Steps (instantiated; N/A marked)

### Step 1 — Reload skills + explore (APPLIES)
- Reload: **`eval-harness`** (technique_analysis + benchmark re-baseline), **`data-harvesting`** (Q5 decontam stage), **`cron-scheduling`** (`auto-retrain.yml` is the execution vehicle), **`na0s-review-checklist`** (inject into every spawned agent).
- Explore vs ideal (already done in §Dedup): system today = sealed *code/corpus* (DI-1/2) + sealed *deploy path* (#14 AST rewrite) but **contaminated shipped `.pkl` + `KNOWN_HASHES`**. Ideal = weights fit on `TRAINING_JSONL_DIRS` only, hashes refreshed, numbers re-baselined.

### Step 2 — Roadmap / Taxonomy / README / Coverage / benchmark read (APPLIES)
Read for full picture: `ROADMAP_V2.md:1395` (the box), `specs/hardening/17-di-decontam-retrain.md` (the design), `specs/hardening/README.md:31,59-60` (cross-link + DVC note), `BENCHMARK_SPRINT.md` (confirm DI boxes are gone — do NOT hunt for them), `BENCHMARK_RESULTS.md` + `README.md` + `docs/facts.yaml` (the contaminated numbers that must be re-baselined). Taxonomy (`data/taxonomy.yaml`) is **untouched** — no attack class added.

### Step 3 — Root-cause implementation plan (APPLIES, but DEFERS to #17 §3)
Root cause = shipped weights predate the DI-1 fix; `KNOWN_HASHES` still pins contaminated digests. The full numbered plan lives in **#17 §3 steps 2-8**. R3 does not duplicate it; the LOCAL execution recipe is in §"LOCAL implementation plan" below, which is #17 §3 with this worktree's two corrections (box moved; #14 landed → don't touch `model_embedding.pkl` deploy wiring, it already preserves it).

### Step 4 — Implement from root cause / wire into pipeline (N/A for THIS spec)
**N/A — this is a PLAN-ONLY artefact; no source/pipeline change is authored here.** (And per #17 §4: DI-3 touches NO `predict.py`/`cascade.py` logic beyond an optional `_TRAINED_SKLEARN` bump if the retrain's sklearn version differs — there is no detector to wire.)

### Step 5 — Review harvested datasets → isolated test file → train (N/A as a *new* dataset task; folds into Q5)
**N/A — R3 does not import a new attack dataset.** The harvester relevance is purely the *decontam control* (Q5): `data/harvest/` IS a `TRAINING_JSONL_DIRS` member (`process_data.py:33`), so harvested rows feeding the retrain must be decontaminated against `data/holdout/` + `data/benchmark/` + F14 scenarios. That is an *audit*, not a dataset move — see Q5.

### Step 6 — File/dir cleanup & refactor (APPLIES, light)
- `model_embedding.pkl` deploy-list inconsistency (#17 G5) is **already resolved** by #14's `deploy_model.py` (preserves non-re-emitted entries) — verify at execution, do NOT re-wire.
- Note the ~25 stale `.claude/worktrees/*` copies of `check_eval_decontamination.py` (#17 §6) for a separate `git worktree prune` pass — out of scope here.
- No new `src/na0s/` module; the one new test goes under `tests/integrity/` (mirrors `src/na0s/integrity/`), per code-org rules — see §Test plan.

### Step 7 — Update roadmap; check off (APPLIES)
- `ROADMAP_V2.md:1395` `[ ] Item 17` → `[x]` **only after a real local retrain commit exists**, citing that LOCAL commit SHA. Until DVC corpus is hydrated, leave unchecked and mark BLOCKED.
- `specs/hardening/README.md:31` "only DI-3 retrain open" → update on completion.
- Do NOT re-add a DI-3 box to `BENCHMARK_SPRINT.md` (#476 deliberately removed the DI-N list).

### Step 8 — README + Benchmark fixes (APPLIES)
Re-baselined sealed-model numbers flow into `docs/facts.yaml`, `README.md` benchmark table, `BENCHMARK_RESULTS.md`; OLD numbers annotated "contaminated (pre-DI-3)". Correct the `ROADMAP_V2.md` dataset-pipeline prose (#17 §7: it still describes `process_data.py` aggregating "holdout/, benchmark/" — now WRONG per `TRAINING_JSONL_DIRS`). The docs-drift CI gate (`docs/facts.yaml`) must stay green AND honest.

### Step 9 — (folded into Step 8 above)

### Step 10 — Taxonomy + Coverage Matrix + thresholds (APPLIES, lightly)
- **Taxonomy:** no new code, no duplicate — `data/taxonomy.yaml` untouched (holdout categories C1/D1–D8/E1/E2/O1/P1 already mapped).
- **Coverage Matrix:** every cell sourced from holdout recall was measured against the contaminated model → downgrade those rows to "estimated / re-baselined (post-DI-3)" until re-run by the eval harness.
- **Thresholds (na0s-review-checklist: justify every number):** reuse existing `_DEFAULT_THRESHOLD = 0.55` (`scripts/model.py`) and the CI-calibrated `decision_threshold.json` path (`scripts/optimize_threshold.py` at `NA0S_TARGET_FPR=0.012`, `auto-retrain.yml:125`). **Introduce NO new magic number.** Do NOT re-tune the threshold on the now-sealed holdout in the same pass — that re-contaminates the operating point; any sweep needs a third split. (Per the task's similarity-cutoff caution: R3 introduces *no* similarity cutoff — there is no short-injection-string matcher here; the only "matching" is exact-string/decontam set-membership in `check_eval_decontamination.py` / `test_no_holdout_leakage.py`, which need no cutoff.)

### Step 11 — NO GitHub (APPLIES, hard rule)
All retrain, deploy, re-baseline, doc, and roadmap edits stay **LOCAL**. The `auto-retrain.yml` path, if used as the execution vehicle, runs in CI but R3's deliverable (checked box + re-baselined docs + new test) is committed locally only; **no push / no PR** until the user authorizes.

---

## LOCAL implementation plan (numbered — #17 §3 with this-worktree corrections)

> Base off **`main`** in a dedicated `git worktree` (DI-1/2 + #14 already there). Never touch the primary checkout.

1. **Verify DI-1/#14 present (cheap, do first).** Confirm `process_data.py:33 TRAINING_JSONL_DIRS`, `deploy_model.py:147` AST rewrite, `tests/test_no_holdout_leakage.py` passes. Confirm hardening **#1** landed (KNOWN_HASHES hash-of-record path). No code change.
2. **Hydrate the sealed corpus (THE BLOCKER).** Either (a) `dvc pull` → `data/raw/` + `data/aggregated/` populated, then run the chain locally:
   `python -m scripts.process_data` → `python -m scripts.validate_data --fix` → `python -m na0s.dataset.hard_negatives` → `python -m scripts.features` → `python -m scripts.model`;
   OR (b) drive the whole thing through **`.github/workflows/auto-retrain.yml`** (`workflow_dispatch`), whose chain is verified present: `sync_datasets` → `integrate_harvest` → `quarantine` promote → `gen_all_datasets`/`generate_taxonomy_samples` → `process_data` → `validate_data` → `na0s.dataset.hard_negatives` → `features` → `model` → `optimize_threshold` (`NA0S_TARGET_FPR=0.012`) → `canary_eval` → `shadow_evaluate` → `check_eval_decontamination` → `f14_promotion_gate` → `deploy_model` (`:381`, gated on approval) → local `git add src/na0s/models/` + commit (`:408-410`). Record which path was used.
3. **Prove decontam of THIS build.** After `combined_data.csv` exists: `pytest tests/test_no_holdout_leakage.py -v` (now NON-skipped because the CSV + eval JSONLs are present — `test_no_holdout_string_in_combined_csv:69` only asserts when both exist) AND `python scripts/check_eval_decontamination.py`. **Extend the decontam scan to `data/harvest`** (Q5) so a harvested row equal to a holdout row cannot silently re-contaminate. Capture both outputs as the attestation.
4. **Train + capture in-distribution metrics** (`scripts/model.py` writes `data/processed/training_metrics.json`; saves `model.pkl` + fresh `.sha256`/`.hmac` sidecar via `safe_dump`). Reuse existing `_DEFAULT_THRESHOLD=0.55` / `_MIN_SAMPLES=100` — no new threshold.
5. **Re-baseline OUT-OF-SAMPLE numbers** (the whole point): `python scripts/technique_analysis.py` (per-category holdout recall, benign FPR, evasion) + `python scripts/benchmark.py --dataset all` (deepset F1/P/R/AUC, alpaca/dolly FPR). Record old (contaminated) vs new (sealed) side-by-side; **expect recall to DROP** — that drop is the honest signal, not a regression to "fix."
6. **Refresh the integrity chain.** `python scripts/deploy_model.py` copies `model.pkl` + `tfidf_vectorizer.pkl` (+`structural_scaler.pkl` if regenerated) into `src/na0s/models/`, backs up the old ones, and AST-rewrites `KNOWN_HASHES` (`:147`) — **preserving `model_embedding.pkl`'s entry** (no longer a gap). Verify end-to-end: `safe_load` the new `model.pkl` with no `Integrity check failed` ValueError (proves `KNOWN_HASHES` ↔ new bytes).
7. **Stamp provenance** (DI-5 hook, already-built): with `NA0S_MODEL_PROVENANCE=1` record the provenance JSON via `na0s.integrity.model_provenance` (training date, dataset version, sample count, sealed-corpus attestation). Bump `_TRAINED_SKLEARN` in `src/na0s/predict.py` only if the sklearn version changed.
8. **Flow re-baselined numbers into docs** (Step 8) and **check off the box** (Step 7) — LOCAL commit only.

---

## Test plan (Code + behavior)

**Code-level (deterministic, no API, no hollow asserts):**
- `tests/test_no_holdout_leakage.py` — make `test_no_holdout_string_in_combined_csv` run **non-skipped** (corpus + eval JSONLs present during retrain) and assert it passes against the freshly-built CSV.
- New `tests/integrity/test_known_hashes_match_bundled.py` (mirrors `src/na0s/integrity/`): for every key in `KNOWN_HASHES`, assert on-disk `src/na0s/models/<file>` SHA-256 == dict value AND `na0s.integrity.safe_pickle.safe_load(get_model_path(f))` returns non-None without raising. Loads + verifies — not `assert True`. Permanently pins "shipped bytes == hash-of-record" so a future retrain that forgets `deploy_model.py` fails CI.
- `scripts/deploy_model.py` tests: add a case asserting `KNOWN_HASHES` is AST-rewritten to the new digests after a deploy into a temp dest, AND that `model_embedding.pkl`'s entry is PRESERVED (regression-guards #14's preserve behavior).

**Use-case / behavior:**
1. *Legit sealed model loads:* after deploy, `na0s.predict` `safe_load`s new `model.pkl`; a benign + a malicious prompt classify with sane scores (full `scan()` runs green).
2. *Tampered file rejected:* flip one byte of deployed `model.pkl` → `safe_load` raises `ValueError("Integrity check failed …")` — proves refreshed `KNOWN_HASHES` is load-bearing.
3. *Decontam attestation:* `check_eval_decontamination.py` exits 0; `test_no_holdout_leakage.py` passes against the new corpus + `data/harvest`.
4. *Re-baseline sanity:* technique_analysis + benchmark produce FINITE numbers; previously-inflated categories are lower-or-equal (soft documented expectation, not a hard threshold).

**Suite + CLI smoke (mandatory, na0s-review-checklist #4):**
- `python3 -m pytest tests/ -q --tb=line` — 0 regressions (~9k tests, ~15 min).
- CLI: `na0s scan "ignore previous instructions"` returns valid JSON with a sane score on the NEW model (proves the deployed artefact loads through the real entry point, not just unit tests).

---

## Q&A self-check
- **Q1 — Can Na0S handle the target?** Partially: the *code/corpus/deploy* paths handle it (DI-1/2 + #14); the *shipped artefact* does not until DI-3 retrains. Fix = LOCAL plan steps 2-6; verify = full suite + CLI smoke.
- **Q2 — Cleanup done?** Step 6: `model_embedding.pkl` gap already resolved by #14 (verify, don't re-wire); stale worktree copies noted for a separate prune; one new test under `tests/integrity/`.
- **Q3 — Pipeline wiring correct?** The retrain chain is fully wired in `auto-retrain.yml` (verified, incl. the gated `deploy_model.py:381`); R3 verifies `KNOWN_HASHES ↔ bytes` and extends the decontam scan to `data/harvest`.
- **Q4 — Tested for code AND use-case?** Yes — code (hash-match, leakage, deploy-preserve) + use-case (legit load, tampered-reject, decontam attestation, re-baseline sanity, CLI smoke).
- **Q5 — Harvester audit:** APPLIES. `data/harvest/` is a `TRAINING_JSONL_DIRS` member → audit (via `data-harvesting` skill's decontam stage) that harvested rows are decontaminated against `data/holdout/` + `data/benchmark/` + F14 scenarios BEFORE landing in `harvest/`; add `data/harvest` to the decontam-gate scan (plan step 3). Audit + gate only — no model fine-tune requested.
- **Q6 — Taxonomy + Coverage Matrix:** Step 10 — re-baseline holdout-sourced coverage rows; no taxonomy duplicates (no code added).
- **Q7 — Does the scorer score this right?** N/A — there is no per-attack-class scorer for "data contamination"; the relevant scorer is the eval harness (technique_analysis / benchmark), re-RUN in plan step 5, not modified.
- **Q8 — predict.py / cascade.py references?** N/A — neither references "decontamination"/"retrain". `predict.py` only *consumes* the artefact (`MODEL_PATH = get_model_path("model.pkl")`, `safe_load`, `_TRAINED_SKLEARN`); the only possible edit is bumping `_TRAINED_SKLEARN` if sklearn version changes (plan step 7). `cascade.py` — no reference.
- **Q9 — Does the harvester agent harvest this type?** N/A as a harvest TARGET — R3 is not an attack dataset; the harvester's relevance is solely its decontam control surface (covered by Q5).
- **Q10 — Other checks:** (a) branch off `main` not the contaminated primary checkout (else DI-1 regresses); (b) confirm DVC corpus availability before claiming DI-3 done; (c) confirm hardening **#1** landed (hash-of-record path); (d) docs-drift gate green after re-baseline; (e) DI-3 box has MOVED to `ROADMAP_V2.md:1395` — do not edit a non-existent `BENCHMARK_SPRINT.md` box.

---

## Execution preconditions / dependencies (the BLOCKERs)
1. **DVC corpus access** — `dvc pull` works locally OR the retrain runs in `auto-retrain.yml` CI where `sync_datasets` hydrates it. `data/*` is gitignored (`.gitignore:2`, plus `data/holdout/*.jsonl:31`, `data/benchmark/*.jsonl:33`, `data/raw/*.csv:48`), so the corpus, `combined_data.csv`, and thresholds **only materialize at retrain time** — DI-3 **cannot be executed on this laptop without DVC**, only specced. THIS IS WHY R3 IS BLOCKED.
2. **Branch off `main`** (DI-1/2 + #14 present), fresh `git worktree`, never the primary checkout (`hardening/rag-poison-wiring`, which still has the contaminated `process_data`).
3. **Hardening #1 landed** — defines the `KNOWN_HASHES` hash-of-record path this retrain stamps.
4. **Hardening #14 landed** — VERIFIED in this worktree (`deploy_model.py:147` AST rewrite + entry preservation). One #17 dependency already satisfied.
5. **`NA0S_PICKLE_KEY` decision** — if set, `safe_dump` writes `.hmac` (stronger) instead of `.sha256` sidecars; `KNOWN_HASHES` is plain SHA-256 regardless. Record which.

## Definition of done
- [ ] Branch off `main`; DI-1/2 + #14 verified present (not reintroduced); #1 confirmed landed.
- [ ] DVC corpus hydrated (record path: `dvc pull` vs `auto-retrain.yml`).
- [ ] Sealed `combined_data.csv` rebuilt; `test_no_holdout_leakage.py` runs NON-skipped and passes; `check_eval_decontamination.py` exits 0 over processed+staging+aggregated+**harvest**.
- [ ] New `model.pkl` (+`tfidf_vectorizer.pkl`, +`structural_scaler.pkl` if regenerated) deployed; `KNOWN_HASHES` AST-rewritten to new bytes; `model_embedding.pkl` entry preserved; sidecars refreshed.
- [ ] `safe_load` loads new artefacts with no `Integrity check failed`; byte-flip test rejects a tampered file.
- [ ] Out-of-sample numbers re-baselined (technique_analysis + benchmark); OLD numbers annotated "contaminated (pre-DI-3)"; deltas recorded.
- [ ] `docs/facts.yaml` / `README.md` / `BENCHMARK_RESULTS.md` updated; `ROADMAP_V2.md:1395` Item 17 checked with LOCAL commit SHA; dataset-pipeline prose corrected; `specs/hardening/README.md:31` updated. (Do NOT re-add a DI-3 box to `BENCHMARK_SPRINT.md`.)
- [ ] New `tests/integrity/test_known_hashes_match_bundled.py` added (loads + verifies; not hollow); deploy-preserve test added.
- [ ] Provenance stamped (`NA0S_MODEL_PROVENANCE=1`); `_TRAINED_SKLEARN` bumped only if sklearn changed.
- [ ] `python3 -m pytest tests/ -q --tb=line` green (0 regressions); `na0s scan` CLI smoke passes on new model.
- [ ] ALL work LOCAL — no push, no PR, no `gh` (Step 11).
