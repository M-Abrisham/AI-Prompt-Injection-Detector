---
item: R4
title: Embeddings → safetensors / .npz (allow_pickle=False) — array-artifact persistence
classification: Supply-chain / eval-integrity infrastructure (NOT a prompt-injection attack class)
dedup_status: ALREADY-TRACKED — fully absorbed by Item 13 (specs/hardening/13-format-migration-skops-safetensors.md). The array/embeddings half of R4 = Item 13 §3.1 step 2 `safe_arrays.py` (file lines 86-92) + §2 array-target rows (FAISS labels, (X,y) feature matrix, teacher_preds) + §3.3 write-path steps. Re-verified against THIS worktree (base advanced via #476 / 81db256): #476 added NO independent R4 roadmap line and NO R4 cross-link into Item 13 — R4 has no standalone tracking row; it lives inside Item 13's line 1391.
scope: THIN. Execution = MERGE into Item 13, CONDITIONAL on an embedding-cache / raw-array artifact actually shipping. Verify object-free FIRST (done below: none ships today).
applicable_steps: [1, 2, 3, 7, 8, 9]   # explore, roadmap/refs, root-cause, cleanup, roadmap-update, README/benchmark — all THIN / mostly verification
na_steps: [4, 5, 6, 10, 11]            # implement-now (no artifact to migrate), harvester, new tests for shipping code, taxonomy/coverage, PR — see N/A lines
skills: [subsystem-context-pack, na0s-review-checklist]
depends_on: [Item 13, Item 5]
local_only: true   # NO GitHub — no push / PR at any step
---

# R4 — Embeddings → safetensors / `.npz` (`allow_pickle=False`)

## 0. One-sentence framing
R4 is the **array/tensor sliver** of the broader pickle-elimination migration: persist raw
numpy/embedding arrays as `np.savez(allow_pickle=False)` / `scipy.sparse.save_npz` / safetensors
instead of pickle — but **no such array artifact ships in Na0S today**, so R4 is a
forward-looking helper that is already fully specified inside Item 13 and only becomes
load-bearing IF an embedding-cache or raw-array artifact is later persisted.

## Dedup re-verification (against THIS worktree)
- **Item 13 owns it.** `specs/hardening/13-format-migration-skops-safetensors.md`:
  - §3.1 step 2 (file lines **86-92**): a NEW `src/na0s/integrity/safe_arrays.py` —
    `save_arrays`/`load_arrays` (`np.savez` + `np.load(..., allow_pickle=False)`),
    `save_sparse`/`load_sparse` (`scipy.sparse.save_npz`/`load_npz`), and a "(Optional, gated)
    safetensors path for pure float-tensor embeddings if/when an embedding cache is persisted —
    currently none ships, so this is a thin forward-looking helper, NOT a load-bearing change."
    **That sentence IS R4.**
  - §2 array-target rows: FAISS labels (`np.savez(allow_pickle=False)`), the `(X,y)` feature
    matrix (`scipy.sparse.save_npz` for `X` + `np.savez` for `y`), `teacher_preds.npy`
    (explicit `allow_pickle=False`).
  - §3.3 write-path steps 8/11 and §0/§1.42 ("the 'numpy/embeddings→.npz' half of the title is
    **forward-looking**").
  - ROADMAP_V2.md:**1391** — Item 13 title literally reads "numpy/embeddings→`.npz`/safetensors
    (`allow_pickle=False`)". ROADMAP_V2.md:**2763** still lists `safetensors | Secure model format
    | L11` as aspirational; Item 13 Step 8 reconciles that row. **No separate R4 row exists.**
- **#476 (81db256) check:** present in this worktree's `git log`; it recorded the 8-PR hardening
  campaign but added **no** R4 line and **no** R4 cross-link to Item 13. Dedup unchanged.
- **Object-free precondition (verified in THIS worktree):**
  - `src/na0s/models/` ships only 4 pickles + 2 sidecars (`model.pkl`, `model_embedding.pkl`,
    `structural_scaler.pkl`, `tfidf_vectorizer.pkl`); **zero** `.npz`/`.npy`/`.safetensors`.
  - `find src -name '*.npz' -o -name '*.npy' -o -name '*.safetensors'` → **empty**.
  - The only `safetensors` strings in source are HF-loader hardening hints
    (`use_safetensors=True`) in `ml/promptguard.py:50`, `ml/late_chunking.py:48/340-341`,
    `ml/promptguard_classifier.py:69` — **not** a Na0S-persisted array cache.
  - `model_embedding.pkl` is a **pickled sklearn classifier** → it is the **skops** target
    (Item 13 §C / model_embedding row), NOT the array/`.npz` target. R4 does not touch it.
  - **Conclusion: no embedding-cache / raw-array artifact ships → R4's execution body is INERT.**
    Execution = keep R4 as the documented forward-looking helper inside Item 13's
    `safe_arrays.py`, to be activated the moment an array artifact is persisted.

---

## Applicable steps (THIN — verification + cross-link, not net-new code)

### Step 1 — Explore current rules around R4
- Grep map of every raw-array persistence surface (all already enumerated in Item 13 §1/§2,
  **line refs re-checked against THIS worktree — Item 13's refs have drifted**):
  | Surface | THIS-worktree line | Format today | R4/Item-13 target |
  |---|---|---|---|
  | FAISS labels | `ml/faiss_classifier.py:196` writes labels via `safe_dump` (Item 5 already gated them; comment at `:178-182`) | gated pickle | `save_arrays`(labels, `allow_pickle=False`) |
  | `(X,y)` feature matrix | `scripts/features.py:314` `safe_dump((X,y), FEATURES_PATH)` (vec/char/scaler at `:302/306/310`) | pickle | `save_sparse(X)` + `save_arrays(y)` |
  | teacher predictions | `scripts/distill_model.py:263` `np.load(args.teacher_predictions)` (no explicit `allow_pickle`) | `.npy` (default-safe) | explicit `np.load(..., allow_pickle=False)` |
  | embedding cache | **does not exist** | n/a | safetensors helper, gated, forward-looking |
  > **Drift note for the executor:** Item 13's spec cites `features.py:164/168/172/176` and
  > `distill_model.py:276`; in THIS worktree they are `:302/306/310/314` and `:263`. Re-grep at
  > execution time; do not trust the stale Item-13 line numbers.
- API existence confirmed in THIS worktree: `scipy.sparse.save_npz` ✓, `numpy.savez` ✓,
  `numpy.load(..., allow_pickle=False)` ✓ (no hallucinated symbols).

### Step 2 — Roadmap / Taxonomy / README / Coverage / benchmark for the gap
- **Roadmap:** R4's gap (= "ship arrays as `.npz`/safetensors") is ROADMAP_V2.md:1391 (Item 13)
  + the aspirational row at :2763. Action for R4: **do not add a new roadmap line**; instead, when
  Item 13 lands, ensure its checkbox text explicitly names the array/`safe_arrays.py` half so R4 is
  visibly subsumed. (Roadmap edit happens under Item 13 Step 8, not separately for R4.)
- **Taxonomy / Coverage Matrix:** N/A reference — no attack class (see Step 10).
- **README / Benchmark:** the only doc surface is "models ship in a non-executable format" — owned
  by Item 13 Step 9; R4 adds nothing until an array artifact ships.

### Step 3 — Root-cause implementation plan (LOCAL; the R4-specific delta over Item 13)
R4 introduces **no new module and no new load-bearing code** beyond Item 13. Its plan is a
3-point *activation contract* on Item 13's `safe_arrays.py`:
1. **Confirm `safe_arrays.py` carries the array path** (Item 13 §3.1 step 2): `save_arrays`/
   `load_arrays` (npz, `allow_pickle=False`) + `save_sparse`/`load_sparse` + a **gated, unused**
   `save_tensors`/`load_tensors` safetensors stub behind `_HAS_SAFETENSORS`. The safetensors stub
   ships **disabled** (no caller) until an embedding cache exists — wiring unused code into a hot
   path is forbidden (na0s-review-checklist §11 "smoke FIRST, wire SECOND").
2. **`allow_pickle=False` is the invariant.** Every array reader R4 owns MUST pass
   `allow_pickle=False` explicitly (numpy's default is safe on ≥1.16.3, but R4 makes it explicit
   so a future numpy default-flip or an object-dtype array fails CLOSED, not silently). Reuse the
   digest sidecar from `safe_pickle.py` (gate runs BEFORE `np.load`).
3. **Activation trigger (the ONLY moment R4 becomes load-bearing):** if/when a future task
   persists an embedding cache or raw embedding matrix, it MUST route through `safe_arrays`
   (`save_tensors`/`save_arrays`), NOT `np.save`/pickle/`torch.save`. Until then, R4 = the
   documented helper + this contract. **No similarity-cutoff / threshold is introduced by R4**
   (it persists arrays; it does not score them) — so the "short-injection-string similarity must
   be re-calibrated" caveat does not apply here (no such number exists in R4).

### Step 7 — Cleanup / refactor
- R4 adds nothing to clean up beyond Item 13's two new `integrity/` modules. The one R4-specific
  hygiene item: when `safe_arrays.py` lands, ensure the safetensors helper is **import-guarded**
  (`_HAS_SAFETENSORS`) and **not added to `pyproject.toml` as a dep** while it has no caller
  (Item 13 §3.4 step 14 already says "add `safetensors` only if used; else defer — do not add an
  unused dep"). R4 enforces that "defer" until activation.

### Step 8 — Roadmap update (LOCAL)
- No standalone R4 checkbox. When Item 13 is checked off, append a half-line to its roadmap entry:
  "(incl. array/embeddings `safe_arrays.py` `.npz`/safetensors — R4 subsumed)". Reconcile the
  aspirational `safetensors` row at ROADMAP_V2.md:2763 (Item 13 Step 8 already owns this; R4 just
  verifies the word "embeddings/arrays" appears). Cite the local commit SHA if any lands.

### Step 9 — README / Benchmark
- **No benchmark parity gate for R4 in isolation** (no shipped array artifact → no `predict_proba`
  path changes → nothing to parity-test). Item 13's parity gate (the skops model swap) is the
  real gate; R4 rides on it. README change: none until an array artifact ships; then the single
  line "embedding/array caches ship as `.npz`/safetensors (non-executable)" is added under Item 13.

---

## N/A steps (honest justifications)

- **Step 4 — Implement-now & wire into predict.py/cascade.py.**
  N/A — there is NO array artifact to migrate today and no array load site in the `scan()` hot
  path; `safe_arrays.py` is created and unit-tested by Item 13, but R4 adds no wiring because the
  safetensors/embedding-cache caller does not exist. Wiring unused code would violate
  na0s-review-checklist §11.
- **Step 5 — Harvester / dataset audit.**
  N/A — R-items are eval-integrity/supply-chain infra, not attack classes; R4 specifically
  serializes the model's own array bytes, not harvested prompt-injection intel. No HuggingFace/
  arXiv/GitHub harvest, no F14 scenario, no decontam dataset applies. (Scope note: R1/R2/R3 touch
  the decontam pipeline; R4 does not.)
- **Step 6 — New tests for shipping behavior.**
  N/A as net-new — the `safe_arrays` round-trip/refusal/tamper tests live in Item 13's test plan
  (`tests/integrity/`, Item 13 §5 tests 4 & 5: round-trip equality + `allow_pickle=False` refusal
  of an object-array npz + crafted-malicious fixture proving no sentinel side-effect). R4 adds a
  test ONLY when an array artifact actually ships (see "when activated" below). A smoke import of
  `safe_arrays` symbols is folded into Item 13 §5 test 13.
- **Step 10 — Taxonomy + Coverage Matrix + per-feature thresholds.**
  N/A — a serialization-format change introduces no detectable attack class, no taxonomy code, no
  scored threshold, and no similarity cutoff. R4 persists arrays; it never scores them.
- **Step 11 — PR / GitHub.**
  N/A — LOCAL ONLY per the directive; and there is no R4-only PR — any code lands inside Item 13's
  branch (`hardening/format-migration-skops-safetensors`). No push/PR at any execution step.

---

## Q&A self-check
- **Q1 — Can Na0S handle R4 (run scan/suite)?** N/A as a detector — R4 is persistence, not
  detection; the relevant suite gate is Item 13's parity benchmark, which R4 has no independent
  effect on (no shipped array artifact).
- **Q2 — Cleanup done / clutter?** Yes — R4 adds no new files; it constrains Item 13's
  `safe_arrays.py` safetensors helper to stay gated+unwired until an array artifact ships.
- **Q3 — Pipeline wiring correct?** N/A — no array load site in `scan()`; nothing to wire until
  activation.
- **Q4 — Tested for code AND use-case?** Code: covered by Item 13's `safe_arrays` tests. Use-case:
  N/A until an array artifact exists (no behavior to exercise).
- **Q5 — Harvester audit.** N/A — model-array bytes, not harvested intel.
- **Q6 — Taxonomy + coverage (no dups)?** N/A — no attack class; and the dedup audit above
  confirms NO duplicate R4 row exists (it is one row: Item 13).
- **Q7 — Does the scorer score R4 right?** N/A — re-serialization is verdict-preserving; R4
  introduces no scorer and no threshold.
- **Q8 — Do predict.py / cascade.py reference R4?** N/A — neither references a `.npz`/safetensors
  array load (they load the skops/pickle estimator, which is Item 13's surface, not R4's). Grep
  confirms no `np.load`/`save_npz`/`safetensors` array-cache call in `predict.py`/`cascade.py`.
- **Q9 — Harvester agent harvest this type?** N/A — not a harvestable intel type.
- **Q10 — Other correctness check.** (i) Object-free precondition re-verified: no array artifact
  ships → R4 inert (this is the gate before any work). (ii) `allow_pickle=False` must be explicit
  on every R4 reader (fail-closed). (iii) safetensors dep deferred until a caller exists (no unused
  dep). (iv) Re-grep Item-13 line refs at execution — they drifted in THIS worktree.

---

## When R4 becomes ACTIVE (the only future-work trigger)
The moment a task persists an embedding cache or a raw embedding/array matrix:
1. Route the writer through `na0s.integrity.safe_arrays` (`save_tensors` if pure-float tensors →
   safetensors; else `save_arrays`/`save_sparse` → npz), NEVER `np.save`/`torch.save`/pickle.
2. Add `safetensors>=0.4,<1` to `pyproject.toml` (documented min/max row) ONLY then; import-guard
   `_HAS_SAFETENSORS` with fail-closed-on-bundled-but-missing.
3. Add a round-trip + tamper + `allow_pickle=False`-refusal test in `tests/integrity/` mirroring
   Item 13 §5 tests 4-5, plus a parity test that the cached embeddings reproduce the live-encode
   values within float tolerance (`atol=1e-9`).
4. Smoke: `python3 -c "from na0s.integrity.safe_arrays import save_arrays, load_arrays, save_sparse, load_sparse; print('ok')"`
   then `python3 -m pytest tests/integrity -q --tb=line`, then full `pytest tests/ -q --tb=line`.

---

## Execution preconditions / dependencies
- **Depends-on Item 13** — R4 has no body of its own; its helper (`safe_arrays.py`) and tests are
  authored under Item 13. Do not open an R4-only branch.
- **Depends-on Item 5** — Item 5 already gated the FAISS-label pickle
  (`faiss_classifier.py:196` now writes via `safe_dump`), so R4's npz superseding of that surface
  must coordinate with Item 5 to avoid double-touching the same line (Item 13 §3.2 step 6 note).
- **Gating precondition: object-free check** — before any R4 code work, re-run
  `find src -name '*.npz' -o -name '*.npy' -o -name '*.safetensors'`; if still empty (true today),
  R4 stays a documented helper and NO code is written.
- **Env:** `safetensors` is NOT installed and is absent from `pyproject.toml` — do not add it until
  an array caller exists. Verify symbols against MAIN (`PYTHONPATH=<worktree>/src`), not the stale
  editable install. Work in a git worktree; never branch-switch the primary checkout.
- **Keyless:** SHA-256 sidecar / KNOWN_HASHES only; no `NA0S_PICKLE_KEY` requirement.
- **LOCAL ONLY** — no push / PR / GitHub at any step.

## Definition of done
- [ ] Object-free precondition re-verified at execution time (no `.npz`/`.npy`/`.safetensors`
      array artifact ships) — if true, R4 = documented helper, no net-new code; record that.
- [ ] Confirmed `na0s.integrity.safe_arrays` (under Item 13) carries `save_arrays`/`load_arrays`
      (`allow_pickle=False`, digest gate BEFORE `np.load`), `save_sparse`/`load_sparse`, and a
      **gated, unwired** safetensors helper (`_HAS_SAFETENSORS`).
- [ ] `safetensors` NOT added to `pyproject.toml` while it has no caller (deferred per Item 13 §3.4).
- [ ] Item 13 line refs re-grepped in THIS worktree (drift confirmed: `features.py:302/306/310/314`,
      `distill_model.py:263`, `faiss_classifier.py:196`).
- [ ] Roadmap: when Item 13 lands, its checkbox text explicitly names the array/`safe_arrays.py`
      half so R4 is visibly subsumed; aspirational `safetensors` row (ROADMAP_V2.md:2763) reconciled.
- [ ] No standalone R4 PR; any code is part of Item 13's branch; merge-to-main confirmed with user.
- [ ] LOCAL-only throughout — no GitHub.

## Skills to reload at execution (Step 1)
- `subsystem-context-pack` — pack `src/na0s/integrity/` + `src/na0s/ml/` + `tests/integrity/` for
  a bounded auditor context.
- `na0s-review-checklist` — inject §1 (hallucinated APIs), §2 (imports), §7 (thresholds — here:
  confirm R4 has NONE), §11 (smoke-first-wire-second) into any subagent.
- N/A skills: `data-harvesting`, `eval-scenario-curation`, `incident-to-scenario`,
  `detector-authoring`, `cron-scheduling` (no harvest / scenario / detector / cron surface).

## Agents to assign
- `layer-9-11-auditor` — L11 supply-chain integrity is R4's domain (the `safe_arrays.py` array path).
- `l3-l5-code-auditor` — owns the ML/array load sites (`faiss_classifier`, `features.py`,
  `distill_model.py`) if/when activated.
- `silent-failure-hunter` — verify `allow_pickle=False` fails CLOSED and the unwired safetensors
  helper never silently activates.
- (No `github-pr-*` agent — LOCAL only, and R4 ships inside Item 13's branch.)
