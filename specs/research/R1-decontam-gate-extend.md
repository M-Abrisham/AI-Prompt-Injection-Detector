# R1 — Extend the decontamination CI gate (EXTENDS DI-2)

| Field | Value |
|---|---|
| **Item** | R1 — Extend the decontamination CI gate (eval-integrity / supply-chain infra) |
| **Dedup status** | EXTENDS the shipped exact+near-dup gate (`scripts/check_eval_decontamination.py`, wired at `.github/workflows/auto-retrain.yml:179-182`). NOT a duplicate. See "Dedup re-verification" — the BENCHMARK_SPRINT.md:27 / auto-retrain.yml:176-179 / ROADMAP:1780 line cites in the task prompt are STALE; corrected below. |
| **Type** | EVAL-INTEGRITY / SUPPLY-CHAIN infra (NOT a prompt-injection attack class) |
| **Applicable orchestration steps** | 1, 2, 3, 4 (eval-infra wiring only, no predict/cascade), 6, 7, 8, 9, 10 (coverage/benchmark only) |
| **N/A steps** | 5 (harvested attack datasets), 11 is a constraint not a step |
| **Applicable Q-checks** | Q1, Q2, Q3 (gate wiring), Q4, Q10 |
| **N/A Q-checks** | Q5, Q6, Q7, Q8, Q9 (all attack-dataset / taxonomy / scorer / predict-cascade specific) |
| **Skills to reload (step 1)** | `eval-harness`, `eval-scenario-curation`, `data-harvesting`, `detector-failure-analysis`, `na0s-review-checklist` |
| **Agents to assign** | `security-research-auditor` (decontam-math + threshold calibration), `l3-l5-code-auditor` (embedding-leg wiring review), one general implementer agent (BFF + per-source legs). Max 4 parallel per CLAUDE.md. |
| **Depends-on** | Existing `scripts/check_eval_decontamination.py`; `na0s.dataset.near_duplicate`; `na0s.ml._st_loader.load_pinned_sentence_transformer` (optional embedding leg); built training corpus (`data/processed/*.csv`) — gitignored & EMPTY locally, so the embedding/BFF legs MUST degrade gracefully on an empty corpus. |
| **Execution policy** | **LOCAL ONLY. No GitHub push / PR / merge at any step until the user explicitly says so.** |

---

## Dedup re-verification (against THIS worktree)

The task prompt's line cites are stale; re-verified the real state in `/Users/mehrnoosh/Na0S-wt/research-items`:

- **Core gate ships and is wired.** `scripts/check_eval_decontamination.py` (380 lines) implements: exact `stable_id` overlap (`scan_exact`, line 171), an optional MinHash/LSH near-dup leg (`find_near_dup_overlaps`, line 214), empty-corpus fail-loud (line 301-309, exit 2), and a `--strict` escalation. It is wired into `.github/workflows/auto-retrain.yml` as step `decontam` at **lines 179-182** (NOT 176-179 as the prompt said), gated on canary+shadow PASSED, and its `outcome` gates the F14 promotion step (`auto-retrain.yml:190`).
- **BENCHMARK_SPRINT.md:27 does NOT contain DI-2.** Line 27 is `make bench-fast` in CI. There is no `DI-` token anywhere in the current `BENCHMARK_SPRINT.md` (grep clean). The DI-1..DI-6 decontam tasks live only in the "Benchmark Integrity" auto-memory note, not the committed file. **Action for execution:** re-add the DI-2/R1 task line to `BENCHMARK_SPRINT.md` under a new "### Decontamination Integrity" subsection rather than editing a non-existent line.
- **ROADMAP_V2.md embedding_fn TODO is at line 1844** (not 1780). The exact text: the F14 admission gate "ships a labeled MinHash/Jaccard proxy + an `embedding_fn` hook" and lists "true embedding-cosine decontam" as a Remaining-for-v0.4 item. The hook itself is already in `src/na0s/eval/scenarios/admission_gate.py` (`embedding_fn` param at line 175, `_check_near_dup_decontam` cosine leg at lines 364-381). **So the embedding leg is half-built in the admission gate but ABSENT from the CI decontam script** — that is exactly R1(b)'s gap.
- **#476:** grep for `#476` in `ROADMAP_V2.md` returns nothing in this worktree; the base did NOT advance a cross-link here. R1 still needs to create its own roadmap entry.

Conclusion: R1 is a genuine EXTENSION (three new legs) of a live gate, not a rebuild. No duplicate work.

---

## Step 1 — Reload skills + explore current system vs ideal

**Reload:** `eval-harness` (decontam is its precondition section), `eval-scenario-curation` (owns the F14 scenario library the gate scans), `data-harvesting` (decontam is a pipeline stage), `detector-failure-analysis` (N/A for finding misses but reload for the threshold-justification discipline), `na0s-review-checklist` (mandatory — §1 hallucinated APIs, §4 hollow tests, §7 arbitrary thresholds, §11 dead-code-after-refactor, §12 premature victory).

**Current system (verified):** the gate has exactly two leak-detection legs:
1. Exact `stable_id` (SHA-256 of NFKC + whitespace-collapsed text) — `scan_exact`, catches verbatim copies only.
2. MinHash/LSH Jaccard near-dup at `MINHASH_JACCARD_THRESHOLD = 0.8` (`near_duplicate.py:62`) — `find_near_dup_overlaps`, off by default (`--near-dup`/`--strict`), warning-only unless `--strict`.

**What it SHOULD be (R1 scope):** three additional, independent leak-detection legs that each catch a class the existing two miss:
- **(a) 13-gram BFF presence-fraction** — catches partial / spliced copies where a long contiguous substring of an eval scenario is embedded inside a larger training row (or vice-versa). Exact-hash misses this (different whole-string hash); MinHash at 0.8 Jaccard can miss it when the shared span is a small fraction of a long row. A 13-gram presence-fraction (fraction of an eval scenario's 13-grams that appear in the training corpus's 13-gram set) directly measures span overlap. ("BFF" = the Allen-AI / "Big Friendly Filter" dedup convention of n-gram-set membership; here implemented pure-Python, no external lib.)
- **(b) Local embedding-cosine leg** — catches semantic paraphrase that survives both exact-hash and token-Jaccard (synonym swaps, re-ordering, translation-back). Closes the `embedding_fn` TODO by reusing the SAME pinned offline model the admission gate's hook expects (`na0s.ml._st_loader.load_pinned_sentence_transformer`, `all-MiniLM-L6-v2`, `predict_embedding.py:63,193`). Must be OPT-IN (`--embedding`) so the always-on CI gate stays fast and dependency-light, exactly as `--near-dup` already is.
- **(c) Per-source-dataset overlap stats** — today the report prints one global "Training rows: N" count. It should break overlaps and row-counts down per training file/source so an operator can see WHICH dataset is the contamination vector (e.g. a single newly-synced HuggingFace source leaking), not just that contamination exists.

**Edge cases to design for (from review-checklist + corpus reality):**
- Corpus is EMPTY locally (`data/processed/` has no CSV in this worktree — gitignored). All three new legs MUST inherit the existing empty-corpus fail-loud (line 301) and never crash on zero rows.
- `sentence-transformers` may be absent → the embedding leg must `try/except ImportError`, print a skip notice, and NOT fail the gate (mirror admission_gate's `embedding_fn raised → skip cosine, keep proxy` at line 372-376).
- **Short injection strings** (many F14 payloads are < 13 tokens). A 13-gram leg on a string shorter than 13 tokens has ZERO 13-grams → presence-fraction is undefined. Must be handled (skip those scenarios in the BFF leg, fall through to exact+MinHash which already cover short strings) and DOCUMENTED, not silently 0.0 or 1.0.

---

## Step 2 — Read the governing docs (done; pointers for execution)

- **Roadmap:** `ROADMAP_V2.md:1507` (L13 dataset lifecycle prose names the gate), `:1844` (embedding_fn TODO), `:1827` (F14-v0.1 wired the gate). The L13 prose at :1507 will need a one-clause update to mention the three new legs.
- **Coverage Matrix:** `docs/COVERAGE_MATRIX.md` — decontam is infra, not a detection row; verify there is no attack-row to add (there isn't; see Step 10).
- **Benchmark:** `BENCHMARK_SPRINT.md` — add the Decontamination-Integrity task line (Step 9).
- **Source to read end-to-end before editing:** `scripts/check_eval_decontamination.py` (the file under change), `src/na0s/dataset/near_duplicate.py:184-269` (`minhash_signature`, `lsh_buckets`, `jaccard_from_minhash`, `_char_ngrams` at line ~73 — REUSE for n-grams, do not reinvent), `src/na0s/ml/_st_loader.py:72` (`load_pinned_sentence_transformer` signature — it takes the `SentenceTransformer` CLASS as first arg), `src/na0s/eval/scenarios/admission_gate.py:339-393` (the reference cosine implementation + `_cosine` helper to mirror).

---

## Step 3 — Root-cause gap analysis

**Gap:** the CI gate is a 2-leg filter (exact + token-Jaccard). Three real leak modes pass it undetected:
1. **Span/partial copy** — eval payload pasted as a fragment inside a longer training row. Root cause: whole-string hashing + Jaccard-on-full-row both dilute a partial match. → Leg (a) 13-gram presence-fraction.
2. **Semantic paraphrase** — same attack, reworded. Root cause: lexical methods are blind to meaning; the embedding leg was designed (`embedding_fn` hook) but never landed in the CI script, only the admission gate. → Leg (b) embedding-cosine.
3. **No source attribution** — when contamination IS found, the operator can't tell which dataset to quarantine. Root cause: the report aggregates globally. → Leg (c) per-source stats.

All three are additive: they extend `scan_exact`/`find_near_dup_overlaps`'s reporting and add two new `find_*` functions. None weakens the existing legs.

---

## Step 4 — Numbered LOCAL implementation plan

> Wiring note: this is eval infra. **predict.py / cascade.py parity is N/A** — the decontam gate never runs inside the detection pipeline; it runs in `auto-retrain.yml` and as a standalone CLI. (Confirmed: no `predict`/`cascade` import touches `check_eval_decontamination`.)

1. **Reuse, don't reinvent, the n-gram primitive.** In `scripts/check_eval_decontamination.py`, import `_char_ngrams` is char-level — R1 needs WORD 13-grams. Add a small private `_word_ngrams(text, n=13)` helper (NFKC + whitespace-collapsed split, same normalization as `compute_stable_id`) so the BFF leg is consistent with the exact leg's canonicalization. Justify n=13 (see Step 10).
2. **Leg (a) — `find_bff_overlaps(scenarios_dir, training_roots, *, n=13, min_presence_fraction=...)`.** Build the SET of all training 13-grams once (streaming over `_iter_training_texts`, the existing iterator — do not re-read files). For each scenario text from `_collect_scenario_texts` (existing, line 153) with ≥ n tokens, compute presence-fraction = |scenario 13-grams ∩ training 13-grams| / |scenario 13-grams|. Flag if ≥ threshold. Scenarios with < n tokens are SKIPPED in this leg (recorded in a `skipped_short` count for the report) and remain covered by exact+MinHash. Memory note: the training 13-gram set can be large; build it as a Python `set[int]` of `hash(gram) & mask` (64-bit) to bound memory, accepting a negligible collision rate (document it).
3. **Leg (b) — `find_embedding_overlaps(scenarios_dir, training_roots, *, embedding_fn, threshold=...)`.** Mirror `admission_gate._check_near_dup_decontam` (lines 339-393): embed every scenario text and every training row via `embedding_fn`, cosine-compare, flag ≥ threshold. `embedding_fn` defaults to a lazy loader that calls `load_pinned_sentence_transformer(SentenceTransformer, DEFAULT_EMBEDDING_MODEL)` inside a `try/except ImportError` → returns None → leg prints "sentence-transformers unavailable; embedding leg skipped" and is a no-op (NOT a failure). Reuse a `_cosine(a, b)` helper copied from admission_gate (or import it if it is exported — verify; if private, copy with attribution comment). To bound O(scenarios × rows) cost, gate this leg behind `--embedding` AND reuse the existing MinHash LSH bucketing to pre-filter candidate rows (only embed training rows that share an LSH band with some scenario) — OR cap with a documented `--embedding-max-rows`. Default OFF.
4. **Leg (c) — per-source stats.** Extend `scan_exact` to also return a `per_source: dict[str, {rows: int, overlaps: int}]` keyed by `str(training_file)`. Thread the per-file row counter that already exists (`n_rows`) into a per-path counter. Print a sorted "Per-source overlap" table in `main()` after the global summary. This is pure reporting; it changes no exit-code logic.
5. **CLI flags** (mirror the existing `--near-dup` family, all OFF by default to keep the always-on gate fast/deterministic):
   - `--bff` (enable leg a, warning-only), `--bff-threshold FLOAT` (default justified in Step 10), `--bff-n INT` (default 13).
   - `--embedding` (enable leg b, warning-only), `--embedding-threshold FLOAT` (default 0.85 — the same value the admission gate already uses, `DEFAULT_NEAR_DUP_THRESHOLD = 0.85` at `admission_gate.py:79`; NOT a new arbitrary number).
   - `--strict` already exists — extend it so BFF/embedding matches become FATAL too (today it only escalates near-dup). Keep exact overlaps always-fatal.
   - `--per-source` defaults ON for the report (free, no perf cost) OR is always printed (preferred — no flag needed).
6. **Exit-code semantics (unchanged contract):** exact overlap → exit 1 (always). BFF/embedding overlap → warning (exit 0) unless `--strict` → exit 1. Empty corpus → exit 2 (existing). Config/IO error → exit 2. Document the new legs in the module docstring's "Exit codes" block.
7. **CI wiring (auto-retrain.yml):** keep the always-on step (`auto-retrain.yml:179-182`) as exact-only for speed. Add a SECOND, separate step (or extend with `--near-dup --bff`) that runs the heavier legs **warning-only** so a paraphrase leak is surfaced in logs without flapping the deploy gate on a similarity heuristic (review-checklist §7 — do not block a deploy on an un-calibrated cosine cutoff). The embedding leg stays out of the always-on CI path unless a corpus + model are present; document why.

---

## Exact files / functions to change

| File | Change |
|---|---|
| `scripts/check_eval_decontamination.py` | ADD `_word_ngrams`, `find_bff_overlaps`, `find_embedding_overlaps`, `_load_default_embedding_fn`, `_cosine`; EXTEND `scan_exact` return to include `per_source`; EXTEND `main()` printing + `_parse_args()` (`--bff`, `--bff-threshold`, `--bff-n`, `--embedding`, `--embedding-threshold`); EXTEND module docstring + Exit-codes block. |
| `.github/workflows/auto-retrain.yml` | ADD a warning-only heavier-leg step (`--near-dup --bff`), or extend step `decontam` carefully; keep exact-only as the always-on hard block (lines 179-182). |
| `tests/eval/scripts/test_check_eval_decontamination.py` | ADD test classes for the three new legs (see Step 6). |
| `ROADMAP_V2.md` | Update L13 prose (:1507) to name the 3 new legs; check off the `embedding_fn` TODO (:1844) with the local commit SHA; add an R1 line. |
| `BENCHMARK_SPRINT.md` | Add "### Decontamination Integrity" subsection with the R1/DI-2 line. |

**No new module under `src/`** — this is a script-level extension of an existing script. (Per CLAUDE.md, new *modules* go in sub-packages; this is additive functions in an existing `scripts/` file, which is the correct home for the CLI gate alongside the rest of the decontam logic. If the n-gram/embedding helpers grow reusable, a follow-up can promote them to `na0s.dataset.near_duplicate` — note it, don't pre-factor.)

---

## Step 5 — Harvested-dataset audit

**N/A — R1 is eval-integrity infra, not an attack class.** There is no "decontamination dataset" to harvest; the inputs are the existing F14 scenario library (`data/eval/scenarios/v0.1/`) and the training corpus (`data/processed/*.csv`). No harvester tuning needed. (The thing R1 *protects* is the harvest→eval pipeline's integrity — covered by the gate itself.)

---

## Step 6 — Test plan (Code + behavior)

Extend `tests/eval/scripts/test_check_eval_decontamination.py` (existing patterns: `tmp_path` CSV/JSONL fixtures, `cdc` import at line 27). Tests MUST have teeth (review-checklist §4 — would fail if the leg were deleted):

**Leg (a) BFF:**
- `test_bff_catches_partial_span` — training row = "<200 chars of filler> <verbatim eval payload> <more filler>"; exact + MinHash(0.8) both MISS (assert `find_overlaps` empty, `find_near_dup_overlaps` empty), but `find_bff_overlaps` flags it. (This is the load-bearing test — it proves BFF catches what the other two legs miss.)
- `test_bff_skips_short_scenarios` — scenario with < 13 tokens is reported in `skipped_short`, not flagged, not crashed.
- `test_bff_clean_when_disjoint` — no shared 13-gram → empty.
- `test_bff_threshold_boundary` — fraction exactly at vs just below threshold.

**Leg (b) embedding:**
- `test_embedding_catches_paraphrase` — inject a stub `embedding_fn` (deterministic vectors: identical text → same vec, paraphrase → cos ≥ threshold, unrelated → low cos) so the test is KEYLESS and offline (no real model download). Assert paraphrase flagged, unrelated not.
- `test_embedding_skips_when_unavailable` — `embedding_fn=None` (or one that raises ImportError) → leg returns empty + prints skip, gate does NOT fail. (Mirrors admission_gate's skip-on-raise contract.)
- `test_embedding_exact_excluded` — an exact dup is not double-reported by the embedding leg.

**Leg (c) per-source:**
- `test_per_source_attribution` — two training files, one contaminated; assert `per_source` keys both, with the right `overlaps` count on the contaminated one and 0 on the clean one.

**CLI / exit-code (CLI smoke — review-checklist §4/§12, mandatory):**
- `test_cli_strict_makes_bff_fatal` — `--bff --strict` on a BFF-only overlap → exit 1; without `--strict` → exit 0 with a warning line.
- `test_cli_empty_corpus_still_fails_loud` — empty corpus + `--bff --embedding` → exit 2 (new legs don't bypass the existing guard).

**End-to-end smoke (NOT mocked — review-checklist §4/§11):** run the real CLI against the real `data/eval/scenarios/v0.1/` and a tiny temp training CSV containing one paraphrase + one verbatim-span row, with `--near-dup --bff --embedding` (stub embedding via env or `--embedding` with model present). Paste real stdout. Confirm: exact catches the verbatim whole-string if present, BFF catches the span, per-source table renders, embedding leg either runs or prints a clean skip. This is the "don't wire dead code" gate (review-checklist §11).

---

## Step 7 — Cleanup / refactor

- Keep all decontam logic in the one script; do NOT scatter copies of `_cosine`/n-gram helpers — if `_cosine` is importable from `admission_gate`, import it; if private, copy ONCE with a `# mirror of admission_gate._cosine` comment and a note to unify in a follow-up.
- Q2: confirm no stray top-level `_*.txt` scratch files are added (the worktree root already has many untracked `_*.txt` — do not add more; write any scratch to the session scratchpad).
- Verify `_iter_training_texts` is reused by all three new legs (single corpus pass where possible) — no duplicate file-walking.

---

## Step 8 — Roadmap update

- Check off `ROADMAP_V2.md:1844` `embedding_fn` "true embedding-cosine decontam" sub-item, citing the local commit SHA (LOCAL commit only; no push).
- Add an R1 line under the L13 / Benchmark-Integrity area: "R1 — extended `check_eval_decontamination.py` with 13-gram BFF presence-fraction, opt-in local embedding-cosine, and per-source overlap stats; wired warning-only into auto-retrain.yml." Cite the local SHA.
- Update the L13 prose sentence at `:1507` that enumerates the gate's legs ("exact stable_id + optional `--near-dup` MinHash/LSH leg") → add "+ `--bff` 13-gram presence-fraction + optional `--embedding` cosine + per-source attribution."
- Per the Roadmap-Todo-Sync memory: every new todo also goes into ROADMAP_V2.md — done here.

---

## Step 9 — README / Benchmark updates

- `BENCHMARK_SPRINT.md`: add a "### Decontamination Integrity" subsection (the DI-* tasks are currently ONLY in auto-memory, missing from the file) and the R1 line. Do NOT invent a fake line-27 edit — the prompt's cite was stale.
- `README.md`: grep for any decontam mention; if it lists the gate's legs, append the new ones. If README doesn't mention the gate (likely), no change — do not manufacture content.
- `docs/data_pipeline.md` / `CHANGELOG.md`: if a decontam section exists, add a one-line entry; otherwise skip (do not create new doc files — CLAUDE.md forbids unsolicited docs).

---

## Step 10 — Taxonomy / Coverage / thresholds

- **Taxonomy (`data/taxonomy.yaml`):** N/A — decontam is not an attack class; no taxonomy code to add. (R1's own item-prompt confirms taxonomy codes are N/A for R-items.)
- **Coverage Matrix (`docs/COVERAGE_MATRIX.md`):** N/A as a detection row — there is no attack to score. The only Coverage tie-in is that R1 *strengthens the trustworthiness* of every COVERAGE_MATRIX recall number (decontam is the precondition per the eval-harness skill). No row edit.
- **Thresholds — MUST be locally re-calibrated, justify every number (review-checklist §7):**
  - **n = 13** for the BFF leg: 13-gram is the de-facto dedup standard (GPT-3 / C4 / Allen-AI BFF all use contiguous 13-token spans as the partial-copy unit) — cite that prior art in the docstring. It is a documented convention, NOT a tuned magic number; expose `--bff-n` so it is overridable.
  - **`--embedding-threshold` = 0.85** — REUSE the value already shipped as `admission_gate.DEFAULT_NEAR_DUP_THRESHOLD = 0.85` (`admission_gate.py:79`). Justify by inheritance, not by re-inventing a cutoff. Note explicitly that 0.85 was set for the admission gate's near-dup proxy and SHOULD be re-validated for short injection strings before it is ever made FATAL — until then the embedding leg is warning-only.
  - **`--bff-threshold`** — do NOT ship a round-number default blind. Calibrate locally: sweep presence-fraction over the existing F14 v0.1 scenarios vs a small known-clean training sample and pick the floor that (i) flags the seeded partial-span fixtures and (ii) zero-FPs on the disjoint benign scenarios. Record the sweep in the test/docstring. If the corpus is unavailable locally (it is empty here), ship the leg with a CONSERVATIVE default (high fraction, e.g. catch only near-total spans) and mark the precise calibration as a CI-time follow-up — never fabricate a precision number from no data (review-checklist §7, §12). State this honestly in the spec/commit.

---

## Q&A self-checks

- **Q1 (can Na0S handle it / run the suite):** Yes — run `python scripts/check_eval_decontamination.py --near-dup --bff` end-to-end + `python3 -m pytest tests/eval/scripts/test_check_eval_decontamination.py -q`, then the full suite per CLAUDE.md (`python3 -m pytest tests/ -q --tb=line`, ~15 min, expect zero net regressions vs the known env-only failures).
- **Q2 (cleanup):** Yes — single script, reused iterators, no new scratch files, helpers not duplicated. See Step 7.
- **Q3 (pipeline wiring):** Wiring is into `auto-retrain.yml` (the gate's pipeline), warning-only for the heavy legs; predict/cascade parity is **N/A** (decontam never runs in the detection path — verified no import edge).
- **Q4 (tested for code AND use-case):** Yes — per-leg unit tests with teeth + a real-CLI end-to-end smoke that proves BFF catches a span the other legs miss. See Step 6.
- **Q5 (harvester audit):** **N/A — R1 is eval-integrity infra, not an attack dataset; nothing to harvest.**
- **Q6 (taxonomy + coverage):** **N/A — decontam is not an attack class; no taxonomy code / coverage row.** (Confirmed no dup row needed.)
- **Q7 (scorer scores it right):** **N/A — there is no per-attack scorer for an infra gate; the "scorer" here is the gate's own exact/BFF/cosine legs, tested in Step 6.**
- **Q8 (predict.py/cascade.py reference it):** **N/A — and they SHOULD NOT; the decontam gate is a CI/CLI tool, deliberately outside the runtime detection pipeline.**
- **Q9 (harvester agent harvests this type):** **N/A — not a harvestable data type.**
- **Q10 (other correctness check):** Empty-corpus fail-loud must still fire with the new legs enabled; `sentence-transformers`-absent must skip-not-crash; short-string (<13-token) scenarios must be handled in the BFF leg; all three covered in Step 6.

---

## Execution preconditions / dependencies

1. Work in an **isolated git worktree** off `main` (Multi-Agent Worktree Discipline memory) — never branch-switch this primary checkout. Branch name: `hardening/r1-decontam-gate-extend` (hardening = robustness improvement, per CLAUDE.md).
2. Verify symbols against THIS worktree, not the editable-install env (the install may point at the d8 checkout). Use `PYTHONPATH=<worktree>/src` when importing `na0s.*`.
3. Confirm `na0s.dataset.near_duplicate` exports `minhash_signature`, `lsh_buckets`, `jaccard_from_minhash`, `MINHASH_JACCARD_THRESHOLD`, `LSH_BANDS`, `LSH_ROWS_PER_BAND` (verified: lines 62-66, 184-269).
4. Confirm `na0s.ml._st_loader.load_pinned_sentence_transformer(st_class, model_name, …)` signature (verified line 72) — it takes the `SentenceTransformer` CLASS, imported at the caller's scope.
5. Training corpus (`data/processed/*.csv`) is **gitignored and empty locally** — all legs must degrade gracefully; full calibration of `--bff-threshold` deferred to a CI run with a real corpus (stated honestly, not faked).
6. `sentence-transformers` may be absent — embedding leg must `try/except ImportError`.

---

## Definition of done

- [ ] `scripts/check_eval_decontamination.py` gains `find_bff_overlaps`, `find_embedding_overlaps`, per-source stats in `scan_exact`/`main`, and the 5 new CLI flags; module docstring + Exit-codes block updated.
- [ ] Every new threshold is justified inline (n=13 prior art; 0.85 inherited from admission_gate; `--bff-threshold` calibrated-or-conservatively-deferred with the reason stated). No bare round-number magic.
- [ ] New legs are OFF by default; exact leg stays the always-on hard block; `--strict` escalates the new legs to fatal.
- [ ] Empty-corpus exit-2 and `sentence-transformers`-absent skip both still hold with new legs on.
- [ ] `auto-retrain.yml` runs the heavy legs warning-only without destabilizing the deploy gate.
- [ ] New tests added with teeth (each fails if its leg is deleted); the BFF "catches-a-span-the-others-miss" test present; KEYLESS stub embedding (no model download in tests).
- [ ] Real-CLI end-to-end smoke run, stdout pasted in the PR/commit body (review-checklist §4/§11) — not just green units.
- [ ] Full suite green (zero net regressions vs known env-only failures).
- [ ] ROADMAP_V2.md (`:1507`, `:1844`, new R1 line) + BENCHMARK_SPRINT.md (new Decontamination-Integrity subsection) updated, citing the LOCAL commit SHA.
- [ ] **All work LOCAL — no git push, no PR, no merge until the user explicitly authorizes.**
