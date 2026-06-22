---
item: M4(b)
title: "deploy_model.py drops model_embedding.pkl (and structural_scaler.pkl) from KNOWN_HASHES on every deploy"
priority_tier: P1 (live correctness bug, no attacker precondition; one-line fix + regression test)
class: supply-chain / integrity (NOT a prompt-injection attack class)
depends_on: []          # none — self-contained; does not block on other hardening items
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]
applicable_qs: [Q1, Q2, Q3, Q4, Q8, Q10]
na_steps: [5, 9, 10]
na_qs: [Q5, Q6, Q7, Q9]
touches_predict_py: true     # predict.py:84 imports KNOWN_HASHES; :343 reads it; :897-903 loads embedding model
touches_cascade_py: false    # cascade.py uses the same loader path but holds no KNOWN_HASHES reference of its own
---

# M4(b) — `deploy_model.py` drops `model_embedding.pkl` from `KNOWN_HASHES`

## 0. Root cause (confirmed against source — line numbers verified 2026-06-22)

`scripts/deploy_model.py` rebuilds the **entire** `KNOWN_HASHES` dict from only the
set of files it actually copies, then blows away the existing dict via a single
`re.sub`. Any bundled pickle that is hashed-but-not-copied is silently deleted
from the dict.

Evidence chain (all verified, not guessed):

1. `scripts/deploy_model.py:28` — `MODEL_FILES = ["model.pkl", "tfidf_vectorizer.pkl"]`.
2. `scripts/deploy_model.py:33-34` — `OPTIONAL_MODEL_FILES = ["structural_scaler.pkl"]`,
   `CHAR_VECTORIZER = "char_tfidf_vectorizer.pkl"`.
   **`model_embedding.pkl` appears in NONE of these lists.**
3. `scripts/deploy_model.py:103,113-119` — `new_hashes = {}` is populated **only** from
   `all_files` = `MODEL_FILES` (+ char vectorizer if present in source, + optional files if
   present in source). `model_embedding.pkl` is never added.
4. `scripts/deploy_model.py:158-170` — `re.sub(r"KNOWN_HASHES\s*=\s*\{[^}]*\}", new_dict, content, count=1)`
   **replaces the whole dict literal** with `new_dict`, which is built from `sorted(new_hashes.items())`.
   So after a deploy run the dict contains exactly the copied files and nothing else.
5. `src/na0s/models/__init__.py:26-31` — current `KNOWN_HASHES` has 4 entries:
   `model.pkl`, `structural_scaler.pkl`, `model_embedding.pkl`, `tfidf_vectorizer.pkl`.
   Bundled-on-disk inventory matches exactly (verified: `src/na0s/models/*.pkl` =
   `model.pkl, model_embedding.pkl, structural_scaler.pkl, tfidf_vectorizer.pkl`).
6. **No sidecar exists for `model_embedding.pkl`** (verified: no
   `src/na0s/models/model_embedding.pkl.sha256` and no `.hmac`). So the hardcoded
   `KNOWN_HASHES` entry is its **only** integrity source.
7. `src/na0s/integrity/safe_pickle.py:214-244` — `_resolve_expected_hash()` checks
   `KNOWN_HASHES` first (`:221-222`); if the basename is absent **and** no sidecar exists,
   it raises `FileNotFoundError` (`:239-244`).
8. `src/na0s/ml/predict_embedding.py:60` — `MODEL_PATH = get_model_path("model_embedding.pkl")`;
   `:195` — `classifier = safe_load(MODEL_PATH)` inside `load_models()`.

**Net effect:** the first time anyone runs `python scripts/deploy_model.py` after this code
shipped, the `model_embedding.pkl` entry is deleted from `KNOWN_HASHES`. Because there is no
sidecar, the next `safe_load("model_embedding.pkl")` raises `FileNotFoundError`. That exception
is swallowed by predict.py (`src/na0s/predict.py:897-903`, bare `except Exception: embedding_score = 0.0`),
so Layer-5's embedding signal goes **silently inert** — recall degrades with zero error surfaced.
`structural_scaler.pkl` has the same class of bug but is partially masked because it lives in
`OPTIONAL_MODEL_FILES` (preserved *iff* it is present in `data/processed/` at deploy time, which
is not guaranteed) — it must be re-verified, not assumed safe.

**Secondary defect (same root cause, opposite direction):** the regex
`KNOWN_HASHES\s*=\s*\{[^}]*\}` (`:166`) uses `[^}]*` and is therefore brittle to any future
nested `}` and to multi-line content; it is the same single-`re.sub`-rebuild pattern that
makes the drop possible. The fix below removes the rebuild-from-scratch behavior, which also
defuses this brittleness.

This is a **live correctness bug with no attacker precondition** — it is not an exploit, it is
a maintenance footgun that destroys an integrity record and silently disables a detector.

---

## 1. Step 1 — Explore current system vs. what it SHOULD be; gaps & edge cases

**Current behavior:** `deploy()` rebuilds `KNOWN_HASHES` only from copied files →
hashed-but-not-copied bundled pkls are erased.

**What it SHOULD be:** `deploy()` must **preserve** existing `KNOWN_HASHES` entries for bundled
pkls it does not touch, and **update/add** entries for files it copies. The dict must remain a
superset that covers every `*.pkl` actually bundled in `src/na0s/models/`. The invariant is:

> Every `*.pkl` file present in `src/na0s/models/` MUST have an entry in `KNOWN_HASHES`
> (since none of them ship sidecars), so `safe_load` always resolves a hardcoded hash.

**Gaps / edge cases to cover:**
- G1 — Drop of `model_embedding.pkl` (the headline bug).
- G2 — Same drop risk for `structural_scaler.pkl` when it is absent from `data/processed/` at
  deploy time (it only survives today because it happens to be in `OPTIONAL_MODEL_FILES` *and*
  present in source). Confirm and lock down.
- G3 — Future-proofing: any new bundled pkl added later (e.g. the planned 384-dim
  `model_embedding.pkl` rebuild, ROADMAP_V2.md:767) must not be silently droppable.
- G4 — Idempotency: running `deploy()` twice in a row must not change `KNOWN_HASHES` for
  unchanged files (the existing `unchanged (sha256 identical)` skip path at `:128-135` already
  populates `new_hashes` for unchanged copied files, but says nothing about non-copied ones).
- G5 — No regression to the existing 19 passing tests in `tests/test_deploy_model.py`
  (esp. `TestKnownHashesReplacement`, `TestCharVectorizerRequired`).

---

## 2. Step 2 — Roadmap / Taxonomy / README / Coverage Matrix cross-reference

- `ROADMAP_V2.md:1360` — *"`deploy_model.py` — backup/rollback paths, `KNOWN_HASHES` regex
  replacement, failure-mode assertions. Priority: P1."* This item is the concrete instance of
  that open P1 task; check it off (or add a sub-bullet) citing the fix SHA.
- `ROADMAP_V2.md:767` — planned 384-dim `model_embedding.pkl` rebuild explicitly notes it must
  *"update `KNOWN_HASHES['model_embedding.pkl']`"*. That future task is **blocked from being
  safe** until this drop bug is fixed (a rebuild + deploy today would erase the entry it is
  trying to update). Note the dependency in that roadmap line.
- `ROADMAP_V2.md:2840` — the `_backup_file()` backup feature is already marked DONE; do not
  re-touch it.
- README / `THREAT_TAXONOMY.md` / Coverage Matrix: **no change** — this is an internal deploy
  script integrity bug, not a user-facing detector or a taxonomy-tagged attack class.
- Disambiguation note for the implementer: the "M4" in `ROADMAP_V2.md:1515` (hardware-key-signed
  bot commits) and the taxonomy "M4 (Code)" category (`ROADMAP_V2.md:1241,1275`) are **unrelated**
  to this item; "M4(b)" here is the supply-chain audit's own numbering for the deploy-script drop.

---

## 3. Step 3 — Root-cause implementation plan (numbered)

The fix is to **merge into the existing dict instead of rebuilding it**, plus add a
self-checking invariant. Two viable approaches; the spec recommends 3A (smaller diff, lowest risk).

### 3A. Recommended — preserve-and-merge (one logical change)

1. In `scripts/deploy_model.py`, before building the replacement literal (`:158`), **parse the
   existing `KNOWN_HASHES` out of `content`** (the init file already read at `:151-153`) into a
   dict, and **start `new_hashes` from that existing dict** rather than from `{}`.
   - Parse via a narrow regex over the dict body capturing `"<name>": "<64-hex>"` pairs, or via
     `ast.literal_eval` on the extracted dict literal (preferred — no hand-rolled hex parsing).
     `ast.literal_eval` is safe (no code execution) and exact.
   - Files actually copied this run **override** their prior entries (update semantics);
     untouched bundled files **retain** their prior entries (preserve semantics).
2. Keep the existing copy/skip/backup logic (`:120-148`) unchanged — only the seeding of
   `new_hashes` and the final dict rebuild change.
3. Optionally tighten the destructive regex at `:166` from `\{[^}]*\}` to a non-greedy
   `\{.*?\n\}` with `re.DOTALL` anchored on the closing-brace-on-own-line that
   `new_dict` itself emits (`:162` writes `",\n}"`). This is a *defensive* nicety; the
   preserve-merge in step 1 is what actually fixes the bug. Justify any regex change in the diff
   comment; do not introduce it as a "magic" pattern without rationale.

### 3B. Alternative — declare `model_embedding.pkl` as a known bundled artifact

Add `model_embedding.pkl` (and confirm `structural_scaler.pkl`) to an explicit
`BUNDLED_PKLS = MODEL_FILES + OPTIONAL_MODEL_FILES + [CHAR_VECTORIZER, "model_embedding.pkl"]`
and, for any bundled pkl that is **not** present in `source_dir` but **is** present in
`dest_dir`, re-hash the existing destination file and carry its digest forward into
`new_hashes`. This keeps the rebuild-from-scratch shape but guarantees coverage.

**Why 3A over 3B:** 3A is a true root-cause fix (stop destroying data); 3B re-derives data that
should simply be preserved and adds a second code path that can drift. Pick 3A unless review
finds the init file is not always parseable (it always is — it is generated source we control).

### 3C. Belt-and-suspenders invariant test (independent of which approach is chosen)

Add a standalone regression test asserting the **product invariant** directly against the real
package: every `*.pkl` in `src/na0s/models/` is a key in `na0s.models.KNOWN_HASHES`. This guards
G3 (future bundled files) regardless of the deploy script's internals and is the cheapest
durable safety net. (See test plan §6.)

---

## 4. Step 4 — Wiring / pipeline parity

- **predict.py (Q8 — applies):** `src/na0s/predict.py:84` imports `KNOWN_HASHES`; `:343` reads
  `KNOWN_HASHES.get("model.pkl", "")` for the model-version tag; `:897-903` loads the Layer-5
  embedding classifier through the loader that calls `safe_load("model_embedding.pkl")`. The fix
  restores the integrity entry so this path stops silently degrading. **No predict.py code change
  is required** — predict.py is the *consumer* that the bug breaks; fixing the producer
  (deploy_model.py) is sufficient. The bare `except Exception` at `:901-902` is what *masked*
  the failure; flag it in the PR as a known silent-failure smell but do **not** change its
  behavior in this item (out of scope; could regress FP posture). Optionally file a follow-up to
  log-on-swallow there.
- **cascade.py (Q8 — does not apply to the fix):** `src/na0s/cascade.py:100,195,804-806` load the
  same embedding model via `predict_embedding.load_models()`, so cascade benefits automatically
  once the hash is restored. cascade.py holds no `KNOWN_HASHES` reference of its own, so there is
  no parity edit to make — verifying it loads is a *test*, not a code change.
- **safe_pickle:** unchanged. `na0s.safe_pickle` is a shim → `na0s.integrity.safe_pickle`
  (verified). Do not add code to the shim (CLAUDE.md rule).

---

## 5. Step 5 — Harvester audit

**N/A — this is a deploy-script integrity bug, not a detector trained on harvested data; no
dataset is involved in the fix.** (The only "data" is the bundled pkl inventory, which is a
build artifact, not a harvested corpus.)

---

## 6. Step 6 — Test plan (Code + Use-Case / behavior) — Q4 applies

All new tests go in `tests/test_deploy_model.py` (core-pipeline deploy script → stays at
`tests/` root per CLAUDE.md test-org rules; existing tests already live there). The invariant
test (T4) may live in `tests/test_model_versioning.py` (already imports `KNOWN_HASHES`,
`tests/test_model_versioning.py:8`) or a new `tests/test_known_hashes_coverage.py` — prefer
extending `test_model_versioning.py` to avoid a near-empty new file.

**Code-level tests (the bug, directly):**

- **T1 — regression for the drop (headline):** Build a temp `__init__.py` whose `KNOWN_HASHES`
  contains `model.pkl`, `tfidf_vectorizer.pkl`, **and `model_embedding.pkl`** (a known 64-hex
  value). Place only `model.pkl` + `tfidf_vectorizer.pkl` in `source_dir` (i.e. simulate a normal
  TF-IDF redeploy that does NOT re-emit the embedding model). Run `deploy(...)`. Assert the
  resulting init **still contains** `"model_embedding.pkl": "<original 64-hex>"` unchanged.
  This test MUST FAIL on the current code and PASS after the fix (state this in the docstring).
- **T2 — structural_scaler preserved when absent from source (G2):** Same harness; `KNOWN_HASHES`
  seeded with `structural_scaler.pkl`; do not place it in `source_dir`. Assert its entry survives.
- **T3 — copied file's hash is UPDATED, not just preserved:** Seed `model.pkl` with a stale hash
  `"deadbeef…"`; place a different `model.pkl` in source; assert the post-deploy hash equals the
  real SHA-256 of the deployed file (not the stale one) — proves merge does update, not freeze.
- **T3b — idempotency (G4):** Run `deploy()` twice with identical inputs; assert `KNOWN_HASHES`
  byte-identical after the second run (no spurious churn).

**Use-case / behavior tests (the consequence — end-to-end loader behavior):**

- **T4 — product invariant (G3, durable):** With no mocking, assert
  `set(every *.pkl in src/na0s/models/) ⊆ set(na0s.models.KNOWN_HASHES.keys())`. Resolve the dir
  via `importlib.resources.files("na0s.models")` (matches `get_model_path`'s own resolution at
  `src/na0s/models/__init__.py:34-36`). This is the test that would have caught the regression
  at the source-of-truth and protects every future bundled pkl.
- **T5 — safe_load round-trips model_embedding.pkl via the hardcoded hash:** Call
  `na0s.integrity.safe_pickle.safe_load(get_model_path("model_embedding.pkl"))` and assert it
  returns an object (no `FileNotFoundError`). Skip-guard with `importorskip` for any optional
  unpickle deps (sklearn/numpy) per the project's optional-dep test discipline (MEMORY:
  Test-Env Optional Deps). This proves the integrity record actually resolves for the real file.
- **T6 — negative control (tamper still rejected):** Assert `safe_load` on a byte-mutated copy of
  `model_embedding.pkl` raises `ValueError("Integrity check failed")` (per
  `safe_pickle.py:335-351`). Confirms the fix does not weaken integrity — a legit file loads, a
  tampered file is rejected.

**Anti-hollow-test discipline (na0s-review-checklist):** every assertion checks a concrete value
(exact 64-hex digest, membership, exception type), not just "no exception". T1 explicitly
documents red→green. No magic thresholds anywhere (this item has none).

**CLI / suite smoke step (mandatory):**
- Targeted: `python3 -m pytest tests/test_deploy_model.py tests/test_model_versioning.py -v`
- CLI smoke (real, not mocked): run `python3 scripts/deploy_model.py` is **destructive** (writes
  the real init) — instead smoke `deploy()` against a temp dir copy in a test, and additionally
  run `python3 scripts/deploy_model.py --rollback` only in a sandboxed temp `dest_dir` via the
  `rollback(dest_dir=...)` param to confirm the parser + entrypoint still wire (covered by
  existing `TestImport.test_parser_rollback_flag`).
- Full suite before reporting done: `python3 -m pytest tests/ -q --tb=line` (CLAUDE.md: ~15 min,
  zero net regressions). Verify against MAIN semantics, not the stale editable install
  (na0s-debugging / MEMORY: editable-install points at d8 checkout — run with
  `PYTHONPATH=<this-worktree>/src` if `na0s.integrity`/`na0s.models` resolve oddly).

---

## 7. Step 7 — Cleanup / refactor per conventions (Q2)

- The fix lives entirely in `scripts/deploy_model.py` (a 4-scripts file already on the v1.0.0
  rename radar per MEMORY: v1 restructure 4/7). Keep the change minimal and self-contained; do
  not fold in the unrelated script renames.
- If approach 3A introduces `import ast`, add it alphabetically to the existing import block
  (`scripts/deploy_model.py:14-20`).
- Remove the now-stale doc claim in `scripts/deploy_model.py:4-6` if it implies the dict is
  rebuilt; update the module docstring to say "merges fresh digests into the existing
  `KNOWN_HASHES`, preserving entries for bundled files not re-emitted this run."
- No dead code, no leftover scratch files (`_skeptic_test_out.txt`, `_xfail_run.txt`, `pyt_out.txt`
  in the working tree are pre-existing untracked noise — do **not** add to them; do not commit them).

---

## 8. Step 8 — Roadmap update

- Tick / annotate `ROADMAP_V2.md:1360` (`deploy_model.py … KNOWN_HASHES regex replacement,
  failure-mode assertions, P1`) with the fix SHA once landed (MEMORY: Roadmap-Todo Sync — every
  todo carries its commit SHA when pushed).
- Add a one-line note under `ROADMAP_V2.md:767` (384-dim rebuild) that the rebuild is now safe to
  perform because deploy no longer drops the entry it updates.

---

## 9. Step 9 — README / Benchmark

**N/A — no user-facing surface, no metric, and no benchmark number changes.** The embedding
signal being restored *could* change recall on an embedding-sensitive eval slice, but only on
machines that had already run the buggy deploy; the shipped repo's `KNOWN_HASHES` is currently
intact (4/4 entries verified), so there is no benchmark delta to record from the fix itself.

---

## 10. Step 10 — Taxonomy / Coverage Matrix / scorer thresholds

**N/A — supply-chain integrity bug, not an attack class.** No taxonomy code (the script does not
classify prompts), no Coverage Matrix row, and no per-attack scorer threshold is involved. The
item introduces **no numeric threshold** of its own (the only constants touched, if any, are the
existing file-name lists), so the "no arbitrary magic threshold" checklist item is satisfied
vacuously.

---

## 11. Step 11 — PR / test gate

- Branch: `fix/deploy-model-known-hashes-drop` (off `main`; do not branch-switch the primary
  checkout — use a worktree per MEMORY: Multi-Agent Worktree Discipline).
- One logical commit (CLAUDE.md), short message, no Co-Authored-By line:
  `fix(deploy): preserve KNOWN_HASHES entries for bundled pkls not re-emitted (M4b)`.
- PR gate: **held-out tests must pass before merge** — full `tests/` suite green with zero net
  regressions; the new T1 must demonstrably go red on `main` and green on the branch (attach the
  before/after in the PR body as proof the regression test bites).
- Use `github-pr-prep` to self-review the diff and generate the PR description; `github-pr-review`
  (precision-first) for the review pass; `github-ci-fix` only if CI goes red.

---

## Q&A self-check

- **Q1 — Can Na0S handle the target?** Not today: a single `deploy_model.py` run erases
  `model_embedding.pkl` from `KNOWN_HASHES` → `safe_load` raises `FileNotFoundError` →
  Layer-5 embedding signal silently zeroed (`predict.py:897-903`). After the §3A fix + §6 tests
  + full-suite green, yes.
- **Q2 — Cleanup?** Fix is one file + tests; §7 lists the docstring tidy and the "don't touch the
  untracked scratch files" guard. No clutter introduced.
- **Q3 — Pipeline wiring correct?** Yes by construction — the bug is in the producer; the consumer
  (`predict.py`/`cascade.py` via `safe_load`) is already wired and starts working again once the
  hash is preserved. No new wiring needed.
- **Q4 — Tested for code AND use-case?** Yes — T1–T3b (code/script), T4–T6 (use-case: invariant,
  real `safe_load` round-trip, tamper-rejection).
- **Q5 — Harvester audit?** N/A — no harvested dataset.
- **Q6 — Taxonomy + Coverage Matrix?** N/A — not an attack class.
- **Q7 — Scorer scores it correctly?** N/A — no scorer / no threshold.
- **Q8 — predict.py / cascade.py references?** predict.py: yes (`:84` import, `:343` read,
  `:897-903` load) — consumer only, no edit. cascade.py: loads same model via
  `predict_embedding.load_models()` (`:100,195,804-806`) — no `KNOWN_HASHES` ref, no edit.
- **Q9 — Harvester agent harvests this type?** N/A.
- **Q10 — Other checks:** (a) confirm `structural_scaler.pkl` survives an absent-from-source
  deploy (T2); (b) add the durable bundled-pkl ⊆ KNOWN_HASHES invariant (T4) so future bundled
  files can't silently regress; (c) verify no sidecars exist for the bundled pkls (confirmed:
  none for `model_embedding.pkl`), which is *why* the hardcoded entry is load-bearing.

---

## Agent / skill assignment (inject na0s-review-checklist into every agent prompt)

| Step / scope | Agent / skill |
|---|---|
| Root-cause confirm + read deploy/loader source (Steps 0–2) | `silent-failure-hunter` (the swallowed `FileNotFoundError` at predict.py:901), `l3-l5-code-auditor` (Layer-5 embedding loader path) |
| Implementation 3A (Steps 3–4) | `Plan` → primary implementer; `na0s-review-checklist` injected; `na0s-debugging` for env/MAIN-vs-editable verification |
| Integrity / safe_pickle correctness (Step 4, T5/T6) | `security-research-auditor` + skill `security-review` |
| Test authoring (Step 6) | implementer + `na0s-review-checklist` (anti-hollow-test section); skill `eval-harness` only if a recall-delta check is wanted (optional, not required) |
| Cleanup / refactor (Step 7) | implementer; `l3-l5-code-auditor` second-pass |
| Roadmap sync (Step 8) | implementer (MEMORY: Roadmap-Todo Sync) |
| PR / CI (Step 11) | skills `github-pr-prep`, `github-pr-review`, `github-ci-fix`; agents `pr-review-toolkit:*` |
| Layer 9–11 integrity context (if integrity module needs a parity look) | `layer-9-11-auditor` |

Skills explicitly **not** used and why: `data-harvesting`, `cron-scheduling`,
`eval-scenario-curation`, `detector-authoring`, `llm-judge` — none apply (no harvest, no cron, no
scenario, no new detector, no judge).

---

## Execution preconditions / dependencies

- **Depends-on: none.** Self-contained one-file fix; does not block on any other hardening item.
- **Blocks / enables:** `ROADMAP_V2.md:767` (384-dim `model_embedding.pkl` rebuild) should land
  **after** this fix, because a rebuild-then-deploy on today's code would erase the very entry it
  intends to update.
- Environment: run in a dedicated git worktree off `main`; verify imports with
  `PYTHONPATH=<worktree>/src` (editable install points at the d8 checkout).

---

## Definition of done

- [ ] `deploy()` preserves `KNOWN_HASHES` entries for bundled pkls not re-emitted this run
      (§3A), and updates entries for files it copies.
- [ ] T1 regression test is red on `main`, green on the branch (proof attached to PR).
- [ ] T2, T3, T3b, T4, T5, T6 added and passing.
- [ ] Durable invariant test (T4): every `*.pkl` in `src/na0s/models/` ∈ `KNOWN_HASHES`.
- [ ] `safe_load("model_embedding.pkl")` resolves via hardcoded hash (no `FileNotFoundError`);
      tampered copy still rejected (T6).
- [ ] Module docstring in `scripts/deploy_model.py` corrected to describe merge semantics.
- [ ] Full suite `python3 -m pytest tests/ -q --tb=line` green, zero net regressions.
- [ ] `ROADMAP_V2.md:1360` ticked/annotated with the fix SHA; `:767` dependency note added.
- [ ] PR opened on a `fix/` branch; held-out tests pass before merge; no scratch files committed.
