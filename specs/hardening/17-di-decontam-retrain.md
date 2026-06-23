---
item: 17
title: DI-1→DI-3 decontamination + sealed-corpus retrain + sidecar/KNOWN_HASHES refresh
priority_tier: P1 (data/eval integrity — every published ML/ensemble number is currently a train-on-test upper bound)
depends_on: [1, 14]   # #1 deploy_model/KNOWN_HASHES drop must define the hash-of-record path; #14 supplies the sealed-corpus provenance/attestation contract this retrain stamps
applicable_steps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]   # this is THE one true data/eval item — full template applies; + Q1, Q2, Q3, Q4, Q5, Q6, Q9, Q10
na_steps: []   # no template step is N/A for this item; Q7/Q8 are the only Q's marked N/A (per-attack scorer + predict.py/cascade.py refs) — see §Q&A
classification: Supply-chain / data-integrity (decontaminate the ML training corpus from the eval sets, retrain on the sealed corpus, refresh the integrity sidecar/KNOWN_HASHES, re-baseline every number) — NOT a prompt-injection attack class, but it is the single item where dataset/harvester/coverage/eval all legitimately apply
---

# Item 17 — DI-1→DI-3 decontamination + sealed-corpus retrain + sidecar/KNOWN_HASHES refresh

## 0. Root cause (one sentence)
The bundled classifier weights in `src/na0s/models/` were fitted on a corpus that included the three "held-out" eval sets verbatim (the old `scripts/process_data.py` globbed `data/holdout/` and `data/benchmark/` into `combined_data.csv`), so every ML/ensemble/threshold number ever cited is a train-on-test upper bound — and although the *code* fix (DI-1) and the *guards* (DI-2/DI-5) have since landed on `main`, **no model has actually been retrained on the now-sealed corpus**: the shipped `.pkl` files are unchanged since 2026-04-02 and `KNOWN_HASHES` still pins the contaminated-corpus digests.

---

## 1. KEY REFS — confirmed current line numbers (verified against the real files; the task's refs had drifted — corrections noted)

| Ref (as given) | Reality on `main` (HEAD) | Correction / state |
|----------------|--------------------------|--------------------|
| `BENCHMARK_SPRINT.md:26-31` (DI-1…DI-6 task list) | **Confirmed present, lines 26–31** — DI-1/DI-2/DI-3 are still rendered as unchecked `[ ]` boxes | The checkboxes are STALE: DI-1 and DI-2 are in fact already implemented on `main` (see below). DI-3 is genuinely open. Step 8 must reconcile these boxes against reality. |
| `scripts/process_data.py:137` ("globs `data/holdout/` and `data/benchmark/` into training") | On the **primary checkout** (`hardening/rag-poison-wiring`) line 137 still reads `for jsonl_dir in [AGGREGATED_DIR, HARVEST_DIR, HOLDOUT_DIR, BENCHMARK_DIR]:` — the contaminated form. On **`main`** this is already fixed: a `TRAINING_JSONL_DIRS = [AGGREGATED_DIR, HARVEST_DIR]` constant (main line 33) and the loop at main lines 144–147 `for jsonl_dir in TRAINING_JSONL_DIRS:` with an explicit "HOLDOUT_DIR / BENCHMARK_DIR are DELIBERATELY EXCLUDED" comment. | **DI-1 is DONE on `main`.** The ref pointed at the pre-fix branch. **This branch must base off `main`, not the primary checkout**, or it will reintroduce the leak. |
| `data/eval/*` | `data/eval/scenarios/v0.1/*.yaml` (F14 library) + `data/eval/scenarios/_drafts/` + `data/eval/trust_priors.yaml`. The ML eval sets are NOT under `data/eval/` — they live in `data/holdout/{malicious,safe}_holdout.jsonl` and `data/benchmark/{adversarial_evasion,test_sample}.jsonl`. | Two distinct decontam surfaces exist: (a) **F14 scenario** leakage — covered by `scripts/check_eval_decontamination.py` (already on `main`); (b) **ML holdout/benchmark** leakage — covered by the new `tests/test_no_holdout_leakage.py` (already on `main`). This item is about (b)'s *consequence*: the model still embodies the leak until retrained. |
| `scripts/deploy_model.py` | Confirmed; deploys `MODEL_FILES = ["model.pkl", "tfidf_vectorizer.pkl"]` (line 28), conditionally `char_tfidf_vectorizer.pkl` (34/107–115), optionally `structural_scaler.pkl` (33), `_sha256` (37–42), `_backup_file` (45–85), rewrites `KNOWN_HASHES` via `re.sub(r"KNOWN_HASHES\s*=\s*\{[^}]*\}", …)` (165–170), `rollback()` (193–227). | `deploy_model.py` already programmatically rewrites `KNOWN_HASHES` AND backs up the prior artefact. **It does NOT write/refresh the `.sha256` sidecars** — those are written only by `na0s.integrity.safe_pickle.safe_dump` at train time (model.py:225). Note: `model_embedding.pkl` is in `KNOWN_HASHES` but NOT in `deploy_model.MODEL_FILES`/`OPTIONAL_MODEL_FILES`, so deploy_model never refreshes its hash — a latent gap this item must decide on (see §3 step 6). |
| `src/na0s/models/__init__.py` | Confirmed; `KNOWN_HASHES` dict lines 26–31 (4 entries: model/structural_scaler/model_embedding/tfidf_vectorizer), `get_model_path` 34–36. Identical bytes on `main` and the primary checkout. | These four digests are the **contaminated-corpus hashes** (model.pkl `057280f9…`, tfidf `347b2b4e…`). DI-3 must replace at least model.pkl + tfidf_vectorizer.pkl (and decide structural_scaler/model_embedding). |

Supporting facts (verified, load-bearing):
- **Shipped weights are pre-fix.** `src/na0s/models/*.pkl` all carry mtime `Apr 2 2026`; the DI-1 code fix and `tests/test_no_holdout_leakage.py` postdate them. So the *live* model is provably the contaminated one.
- **Integrity chain.** `safe_pickle.safe_load` (src/na0s/integrity/safe_pickle.py:295–363) resolves the expected hash via `_resolve_expected_hash` (214–244) in priority order: hardcoded `KNOWN_HASHES` → `.hmac` sidecar → `.sha256` sidecar. For the four bundled files the **hardcoded `KNOWN_HASHES` wins** (basename in dict → returns at line 221). ⇒ A retrain that updates the `.sha256` sidecars but NOT `KNOWN_HASHES` would still verify against the OLD hash and the new model would fail to load. **`KNOWN_HASHES` is the authoritative refresh target**, sidecars are secondary.
- **Sidecars present:** only `model.pkl.sha256` and `tfidf_vectorizer.pkl.sha256` exist on disk; `structural_scaler.pkl` and `model_embedding.pkl` have NO sidecar (they ride on `KNOWN_HASHES` alone).
- **Training corpus is NOT on disk.** `data/processed/` is empty locally; `data/raw/` is empty; `data/aggregated/` holds only `morris2/`. The corpus is DVC-tracked (`data/raw.dvc`, `docs/MODEL_PROVENANCE.md` "training data lives outside the repo (DVC-tracked)"). ⇒ **DI-3's retrain cannot run on this laptop** without a `dvc pull` or running it inside `auto-retrain.yml` (which does `sync_datasets` → `process_data` → `features` → `model`). This is the single biggest execution precondition.
- **Retrain wiring already exists.** `.github/workflows/auto-retrain.yml` runs the full chain (`scripts.sync_datasets` → `scripts.integrate_harvest` → quarantine promote → `scripts.process_data` → `scripts.validate_data` → `na0s.dataset.hard_negatives` → `scripts.features` → `scripts.model` → `scripts.canary_eval` → `scripts/shadow_evaluate.py` → `scripts/deploy_model.py`). `scripts/model.py` saves via `safe_dump` (line 225) and writes `data/processed/training_metrics.json` (219–222). So DI-3 is mostly an *invocation + verification + re-baseline* task, not greenfield.
- **DI-5 already shipped:** `docs/MODEL_PROVENANCE.md` + `src/na0s/integrity/model_provenance.py` (gated on `NA0S_MODEL_PROVENANCE=1`) exist on `main`. The retrain must STAMP a fresh provenance record (training date, dataset version, sample count, sealed-corpus attestation), not invent a new mechanism.
- **DI-4 (retire `canary_eval.csv` as the deploy gate) is OUT OF SCOPE for this item** — it is its own roadmap line and the F14 gate (`scripts/f14_admission_gate.py` / F14-v0.1) is the replacement track. This spec only RECORDS that the current canary gate is rule-circular and must NOT be cited as a held-out number; it does not rip it out.

---

## 2. Gap vs ideal

| # | Today | Ideal |
|---|-------|-------|
| G1 | DI-1 code fix landed on `main`, but BENCHMARK_SPRINT.md still shows `[ ] DI-1/DI-2`. | Checkboxes reflect reality; DI-1/DI-2 marked done with SHA; DI-3 tracked as the remaining open item. |
| G2 | Live `.pkl` weights + `KNOWN_HASHES` are the **contaminated-corpus** artefacts (Apr-2 mtime; digests `057280f9…`/`347b2b4e…`). | Weights fitted on the **sealed corpus** (`TRAINING_JSONL_DIRS` only, no holdout/benchmark rows); `KNOWN_HASHES` + sidecars carry the NEW digests. |
| G3 | Every cited ML/ensemble number (deepset F1, holdout recall 0.685, evasion 0.485, AUC, FPR) was measured against a contaminated model. | Re-baseline ALL of them against the sealed model; mark prior numbers contaminated; publish honest deltas. |
| G4 | No CI step asserts "the retrained model was NOT trained on holdout/benchmark rows" beyond `test_no_holdout_leakage.py` (which guards the *code/corpus*, not the *shipped weights*). | A retrain gate that (a) regenerates `combined_data.csv`, (b) re-runs `test_no_holdout_leakage.py` + `check_eval_decontamination.py`, (c) trains, (d) verifies the new `KNOWN_HASHES` matches the new `.pkl`, before any PR. |
| G5 | `model_embedding.pkl` is in `KNOWN_HASHES` but never refreshed by `deploy_model.py` (not in its file lists). | Decide: either add it to `deploy_model` file lists and refresh it, OR document that it is a static/optional artefact not part of the retrain and leave its hash untouched (must be a conscious, recorded decision — see step 6). |
| G6 | `docs/facts.yaml` / README / BENCHMARK_RESULTS.md carry the contaminated numbers; the docs-drift CI gate (`docs/facts.yaml`) will pin them. | Re-baselined numbers flow into facts.yaml/README/BENCHMARK_RESULTS so the docs gate stays green AND honest. |

---

## 3. Root-cause implementation plan (numbered)

> **Base off `main`** (DI-1/DI-2/DI-5 already there). Work in a dedicated worktree on branch `hardening/di-decontam-retrain`. NEVER touch the primary checkout's tree.

1. **Reconcile the DI-1/DI-2 state (cheap, do first).** Confirm on `main`: `scripts/process_data.py` has `TRAINING_JSONL_DIRS = [AGGREGATED_DIR, HARVEST_DIR]` (line 33) and the loop at 144–147; `tests/test_no_holdout_leakage.py` passes. Mark DI-1/DI-2 done in BENCHMARK_SPRINT.md (step 8). No code change here — just verification + roadmap truth-up.

2. **Hydrate the sealed corpus.** Either:
   - (a) `dvc pull` the training data into `data/raw/`+`data/aggregated/`, then run locally:
     `python -m scripts.process_data` → confirm the printed per-source list contains **no** `data/holdout/*` or `data/benchmark/*` rows → `python -m scripts.validate_data --fix` → `python -m na0s.dataset.hard_negatives` → `python -m scripts.features` → `python -m scripts.model`; OR
   - (b) drive the retrain through `auto-retrain.yml` (`workflow_dispatch`) in CI where the corpus is hydrated by `sync_datasets`.
   Record which path was used (reproducibility). The script entry points are confirmed present (`scripts/process_data.py`, `scripts/features.py`, `scripts/model.py`, `na0s.dataset.hard_negatives`).

3. **Prove decontamination of THIS corpus build.** After `process_data` writes `data/processed/combined_data.csv`, run BOTH guards against the freshly-built CSV:
   - `python -m pytest tests/test_no_holdout_leakage.py -v` (the `test_no_holdout_string_in_combined_csv` test only asserts when both the CSV and the eval JSONLs are on disk — so this is the meaningful, non-skipped run).
   - `python scripts/check_eval_decontamination.py --training-roots data/processed data/staging data/aggregated` (exit 0 required) — catches F14-scenario leakage too.
   Capture both outputs into the PR body as the decontamination attestation.

4. **Train + capture metrics.** `scripts/model.py` already prints CV acc/AUC, raw vs calibrated acc, TPR/FPR sweep, ROC-AUC, PR-AUC, Brier, ECE, FNR@0.55, confusion matrix, and writes `data/processed/training_metrics.json` (lines 197–222). These are the *in-distribution* numbers. The model is saved via `safe_dump` (line 225) → writes `model.pkl` + a fresh `.sha256` (or `.hmac` if `NA0S_PICKLE_KEY` set) sidecar. **No magic thresholds introduced** — reuse the existing `_DEFAULT_THRESHOLD = 0.55` (model.py:32) and `_MIN_SAMPLES = 100` (model.py:29); do not add new ones.

5. **Re-baseline the OUT-OF-SAMPLE numbers (the whole point).** Run the eval harness against the NEW model — these sets are now genuinely held out:
   - `python scripts/technique_analysis.py` (two-sided recall + benign-FPR) → per-category recall on `data/holdout/malicious_holdout.jsonl`, benign FPR on `safe_holdout.jsonl`, evasion detection on `data/benchmark/adversarial_evasion.jsonl`.
   - `python scripts/benchmark.py --dataset all` → deepset F1/P/R/AUC, alpaca/dolly FPR.
   Record old (contaminated) vs new (sealed) side by side. **Expect recall to DROP** vs the inflated 0.685 / 0.485 — that drop is the honest signal, not a regression to fix by re-contaminating.

6. **Refresh the integrity chain (the "sidecar/KNOWN_HASHES refresh" half of the title).**
   - Run `python scripts/deploy_model.py` — copies `model.pkl`+`tfidf_vectorizer.pkl` (+conditionally char/structural) from `data/processed/` into `src/na0s/models/`, backs up the old ones (`_backup_file`), and rewrites `KNOWN_HASHES` (deploy_model.py:165–170).
   - **Decide `model_embedding.pkl` + `structural_scaler.pkl`:** if the retrain regenerates `structural_scaler.pkl`, it IS in `OPTIONAL_MODEL_FILES` (deploy_model.py:33) and will be refreshed. `model_embedding.pkl` is NOT in any deploy_model list → its `KNOWN_HASHES` entry will be left at the old digest. Two valid resolutions: (i) leave it (document that the embedding model is a separate, non-retrained artefact — preferred if no embedding training ran), or (ii) add it to `deploy_model.MODEL_FILES`/optional and refresh. Pick (i) unless an embedding retrain is in scope; record the choice in MODEL_PROVENANCE.
   - **Verify the chain end-to-end:** after deploy, `python -c "from na0s.predict import _load_model_and_vectorizer; ..."` (or the existing model-load path) must `safe_load` the new `model.pkl` WITHOUT an `Integrity check failed` ValueError — proving `KNOWN_HASHES` matches the new bytes. This is the §Q4 use-case smoke test.

7. **Stamp provenance (DI-5 hook, already-built mechanism).** With `NA0S_MODEL_PROVENANCE=1`, record a provenance JSON via `na0s.integrity.model_provenance` (training date, training_script=`scripts/model.py`, dataset_version, sample_count, sealed-corpus attestation). Update `docs/MODEL_PROVENANCE.md` if the sklearn training version changed (and bump `_TRAINED_SKLEARN` in `src/na0s/predict.py` line ~340 if so).

8. **Flow re-baselined numbers into docs (keep the docs-drift gate green AND honest).** Update `docs/facts.yaml`, README benchmark table, and `BENCHMARK_RESULTS.md` / `BENCHMARK_SPRINT.md` "Current Benchmark Numbers" with the new sealed-model numbers; explicitly annotate the OLD numbers as "contaminated (pre-DI-3)".

---

## 4. Exact files / functions to change

| File | Change | Why |
|------|--------|-----|
| `src/na0s/models/model.pkl`, `tfidf_vectorizer.pkl` (+ `structural_scaler.pkl` if regenerated), and their `.sha256` sidecars | Replaced with sealed-corpus artefacts (binary, via `deploy_model.py`) | The contaminated weights are the live bug |
| `src/na0s/models/__init__.py` | `KNOWN_HASHES` dict (lines 26–31) rewritten by `deploy_model.py` | Authoritative hash-of-record must match new bytes or `safe_load` rejects them |
| `data/processed/training_metrics.json` | Regenerated by `scripts/model.py` (not committed if gitignored; cited in PR) | In-distribution baseline record |
| `BENCHMARK_SPRINT.md` (lines 26–31, 152–157) | DI-1/DI-2 → done w/ SHA; DI-3 progress; mark BM-1/BM-2 numbers contaminated→re-baselined | Roadmap truth-up + honest numbers |
| `docs/facts.yaml`, `README.md`, `BENCHMARK_RESULTS.md` | Re-baselined numbers | Docs-drift gate + public credibility |
| `docs/MODEL_PROVENANCE.md` | New training date / dataset version / sealed attestation; sklearn version + `_TRAINED_SKLEARN` if changed | DI-5 record |
| `ROADMAP_V2.md` | Check off DI-3; cite commit SHA | Roadmap-sync rule |
| (CI, optional) `.github/workflows/auto-retrain.yml` | Insert a "decontamination assert" step (`pytest tests/test_no_holdout_leakage.py` + `check_eval_decontamination.py`) BEFORE `scripts.model` | G4 — make the retrain prove decontam every cadence, not just this once |

**No source-logic file (`predict.py`/`cascade.py`/detectors) changes** beyond the optional `_TRAINED_SKLEARN` bump — this is a data/artefact/CI item.

---

## 5. Test plan (Code + Use-Case)

**Code-level (deterministic, no API):**
- `tests/test_no_holdout_leakage.py` — already on `main`; this item makes `test_no_holdout_string_in_combined_csv` run *non-skipped* by ensuring the corpus + eval JSONLs are present during the retrain. Assert it passes against the freshly-built CSV (not just structurally).
- New `tests/integrity/test_known_hashes_match_bundled.py` (mirrors source tree under `tests/integrity/`): for every file in `KNOWN_HASHES`, assert the on-disk `src/na0s/models/<file>` SHA-256 == the dict value, AND that `na0s.integrity.safe_pickle.safe_load(get_model_path(f))` returns a non-None object without raising. This permanently pins "shipped bytes == hash-of-record" so a future retrain that forgets `deploy_model.py` fails CI. (Not hollow: it loads + verifies, it doesn't just `assert True`.)
- `scripts/deploy_model.py` already has tests (per BENCHMARK/ROADMAP); add a case asserting `KNOWN_HASHES` is rewritten to the new digests after a deploy into a temp dest dir (`deploy(source_dir=…, dest_dir=…, init_path=…)` supports redirection — deploy_model.py:88–101).

**Use-case / behavior (the §Q4 reframe — end-to-end of the integrity/loader change):**
1. *Legit sealed model loads:* after deploy, `na0s.predict` model-load path `safe_load`s the new `model.pkl` and a benign + a malicious prompt classify with sane scores (full `scan()` runs green).
2. *Tampered file is rejected:* flip one byte of the deployed `model.pkl` → `safe_load` raises `ValueError("Integrity check failed …")` (safe_pickle.py:346) — proves the refreshed `KNOWN_HASHES` is actually load-bearing.
3. *Decontam attestation:* `scripts/check_eval_decontamination.py` exits 0 and `test_no_holdout_leakage.py` passes against the new corpus.
4. *Re-baseline sanity:* `technique_analysis.py` + `benchmark.py` produce numbers; assert they are *finite and lower-or-equal* on the previously-inflated categories (recall on holdout no longer impossibly high) — a soft documented expectation, not a hard pass/fail threshold.

**Suite + CLI smoke (mandatory):**
- `python3 -m pytest tests/ -q --tb=line` — zero regressions (per CLAUDE.md; ~15 min, ~9,077 tests).
- CLI smoke: `na0s scan "ignore previous instructions"` returns valid JSON with a sane score using the NEW model (proves the deployed artefact is loadable through the real entry point, not just unit tests — failure-checklist #4).

---

## 6. Cleanup / refactor (Step 7)

- **Worktree hygiene:** the search surfaced ~25 stale copies of `scripts/check_eval_decontamination.py` under `.claude/worktrees/*` — these are abandoned agent worktrees, NOT part of `main`. Do not touch them here, but note for a separate `git worktree prune` housekeeping pass (Step 2 cleanliness).
- **Stray artefacts in primary checkout** (`_skeptic_test_out.txt`, `_xfail_run.txt`, `pyt_out.txt`, `logs/`) are uncommitted scratch on the rag-poison branch — not this branch's concern; do not stage them.
- **`model_embedding.pkl` in `KNOWN_HASHES` but absent from `deploy_model` file lists** (G5) is a real latent inconsistency; resolving it (document-or-wire) IS in scope for this item.
- No new module is created in `src/na0s/`; the one new test goes under `tests/integrity/` mirroring `src/na0s/integrity/` (code-organization rule).

---

## 7. Roadmap / README / Benchmark updates (Steps 8–9)

- BENCHMARK_SPRINT.md: check `[x] DI-1`, `[x] DI-2` (cite the `main` SHAs that introduced `TRAINING_JSONL_DIRS` + `test_no_holdout_leakage.py`); update `[~] DI-3` → `[x]` on merge with this branch's SHA; rewrite the BM-1/BM-2 number blocks (lines 148–157) to "contaminated (pre-DI-3)" → new sealed numbers.
- ROADMAP_V2.md: check off DI-3 line + SHA (the dataset-pipeline section ~line 1289 describes `process_data.py` "aggregates … holdout/, benchmark/" — that prose is now WRONG and must be corrected to match `TRAINING_JSONL_DIRS`).
- README + BENCHMARK_RESULTS.md + `docs/facts.yaml`: new numbers; old numbers annotated contaminated.

---

## 8. Taxonomy + Coverage Matrix + thresholds (Step 10 / Q6 — APPLIES, lightly)

- **Taxonomy:** No new attack class — `data/taxonomy.yaml` is untouched. The holdout categories (C1/D1–D8/E1/E2/O1/P1) already map to taxonomy codes; the retrain doesn't add codes, it re-measures recall per existing code. ✅ no duplicates introduced.
- **Coverage Matrix:** the per-category recall rows that were "measured" against the contaminated model become *re-measured against the sealed model* — every COVERAGE_MATRIX cell sourced from holdout recall must be flagged "re-baselined (post-DI-3)" or downgraded from measured→estimated until re-run. This is the one place the coverage matrix genuinely moves.
- **Thresholds:** reuse existing `_DEFAULT_THRESHOLD=0.55` (model.py:32, predict `DECISION_THRESHOLD`) — DO NOT re-tune the threshold on the now-sealed holdout in the same pass (that would re-contaminate the threshold). Any future threshold sweep must use a third split. **No new magic number is introduced by this item.**

---

## Q&A self-check

- **Q1 — Can Na0S handle the target?** Partially: the *code* path (DI-1/DI-2) handles it on `main`; the *shipped artefact* does not until DI-3 retrains. Fix = steps 2–6; verify = full suite + CLI smoke (§5).
- **Q2 — Cleanup done?** §6: stale worktree copies + scratch files noted (not this branch's), `model_embedding.pkl`/`deploy_model` inconsistency resolved here.
- **Q3 — Pipeline wiring correct?** Retrain chain already wired in `auto-retrain.yml`; this item adds a decontam-assert step before training (G4) and verifies `KNOWN_HASHES↔bytes` wiring (step 6).
- **Q4 — Tested for code AND use-case?** Yes — §5 code tests (hash-match, leakage) + use-case (legit load, tampered-reject, decontam attestation, re-baseline sanity, CLI smoke).
- **Q5 — HARVESTER AUDIT:** APPLIES. The harvester feeds `data/harvest/` → which IS a training dir (`TRAINING_JSONL_DIRS`). Audit that the harvester/`data-harvesting` skill decontaminates harvested rows against `data/holdout/`+`data/benchmark/`+F14 scenarios BEFORE they land in `harvest/` (the `data-harvesting` skill's "decontaminate against training + live eval data" stage). If a harvested row equals a holdout row, the retrain silently re-contaminates. Add this assertion to the decontam-gate step (step 3 covers it by scanning `data/aggregated` + `data/processed`; extend to `data/harvest`). No model fine-tune is requested — audit + gate only.
- **Q6 — Taxonomy + Coverage Matrix:** §8 — re-baseline coverage rows; no taxonomy duplicates.
- **Q7 — Does the scorer score this the way it should?** N/A — there is no per-attack-class scorer for "data contamination"; the relevant "scorer" is the eval harness (`technique_analysis.py`/`benchmark.py`), which is re-run in step 5, not modified.
- **Q8 — predict.py / cascade.py references?** Effectively N/A — neither references "decontamination". `predict.py` only *consumes* the artefact (`MODEL_PATH = get_model_path("model.pkl")` line 247, `safe_load` 331–332, `_TRAINED_SKLEARN` ~340). The only possible predict.py edit is bumping `_TRAINED_SKLEARN` if the retrain's sklearn version differs (step 7). `cascade.py` — no reference.
- **Q9 — Does the harvester agent harvest this type?** The harvester harvests *attack intel/datasets*, not "decontamination"; but its decontam stage is exactly the Q5 control surface. The `data-harvesting` skill already owns "decontaminate against training + live eval data" — verify it covers the holdout/benchmark sets, not just F14 scenarios.
- **Q10 — Other checks:** (a) confirm the branch is based off `main` not the stale primary checkout (else DI-1 regresses); (b) confirm DVC corpus availability before claiming DI-3 done; (c) `model_embedding.pkl` hash decision recorded; (d) docs-drift gate green after re-baseline.

---

## Agent / skill team (inject `na0s-review-checklist` into every spawned agent)

| Step(s) | Owner | Skills to load |
|---------|-------|----------------|
| 1, 8 (roadmap reconcile) | `Plan` + author | na0s-review-checklist |
| 2–4 (corpus hydrate, decontam-prove, train) | `silent-failure-hunter` (catch a silently-skipped leakage test / empty-corpus train) | `eval-harness`, `na0s-debugging`, `data-harvesting` |
| 3, Q5/Q9 (harvester decontam audit) | `security-research-auditor` | `data-harvesting`, `eval-scenario-curation` |
| 5 (re-baseline numbers) | `l3-l5-code-auditor` (ML stream) | `eval-harness` |
| 6–7 (integrity chain + provenance) | `security-research-auditor` + `layer-9-11-auditor` (L11 supply chain) | `security-review` |
| CI gate (G4) | author | `github-ci-fix`, `cron-scheduling` (auto-retrain.yml is a cron workflow) |
| 11 (PR) | `pr-review-toolkit:review-pr` / `github-pr-prep` | `github-pr-review` |

---

## Execution preconditions / dependencies

1. **Branch off `main`** (commit where `TRAINING_JSONL_DIRS` + `tests/test_no_holdout_leakage.py` exist) — NOT off `hardening/rag-poison-wiring` (still contaminated). Use a fresh `git worktree`.
2. **#1 must land** — the deploy_model/KNOWN_HASHES "hash-of-record" path is the refresh target; if #1 changes how hashes are stored, DI-3 stamps the new form.
3. **#14 must land** — supplies the sealed-corpus provenance/attestation contract this retrain fills in (per-row "withheld from training + rule authorship" attestation, DI-5 lineage).
4. **DVC corpus access** — `dvc pull` works OR the retrain runs in `auto-retrain.yml` CI where `sync_datasets` hydrates it. Without this, DI-3 cannot be *executed*, only specced.
5. **`NA0S_PICKLE_KEY`** decision — if set, `safe_dump` writes `.hmac` sidecars (stronger) instead of `.sha256`; KNOWN_HASHES is plain-SHA256 regardless. Record which.

## Definition of done

- [ ] Branch based off `main`; DI-1/DI-2 verified present (not reintroduced).
- [ ] Sealed `combined_data.csv` rebuilt; `test_no_holdout_leakage.py` runs **non-skipped** and passes; `check_eval_decontamination.py` exits 0 over processed+staging+aggregated+harvest.
- [ ] New `model.pkl`(+`tfidf_vectorizer.pkl`, +`structural_scaler.pkl` if regenerated) deployed; `KNOWN_HASHES` rewritten to match the new bytes; `.sha256`/`.hmac` sidecars refreshed.
- [ ] `safe_load` loads the new artefacts with no `Integrity check failed`; byte-flip test rejects a tampered file.
- [ ] `model_embedding.pkl` hash decision recorded in MODEL_PROVENANCE.
- [ ] Out-of-sample numbers re-baselined (technique_analysis + benchmark); old numbers annotated "contaminated"; deltas in PR.
- [ ] `docs/facts.yaml` / README / BENCHMARK_RESULTS / BENCHMARK_SPRINT / ROADMAP_V2 updated (DI-3 checked + SHA; process_data prose corrected).
- [ ] New `tests/integrity/test_known_hashes_match_bundled.py` added (loads + verifies, not hollow).
- [ ] (Optional but recommended) decontam-assert step added to `auto-retrain.yml` before training.
- [ ] `python3 -m pytest tests/ -q --tb=line` green (0 regressions); `na0s scan` CLI smoke passes on the new model.
- [ ] PR opened; merges only after held-out CI is green.
