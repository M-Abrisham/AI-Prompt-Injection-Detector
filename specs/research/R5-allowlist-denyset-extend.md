---
item: R5
title: "Allowlist deny-set + fickling gadget catalog (EXTENDS hardening #04)"
class: RESEARCH item — SUPPLY-CHAIN / GOVERNANCE infrastructure (NOT a prompt-injection attack class)
extends: specs/hardening/04-toctou-allowlist-unpickler.md  (Defect-2 residual only — see §0)
red_team_sibling: specs/hardening/08-l11-adversarial-stress-tests.md
dedup_status: >
  EXTENDS Item #04. RE-VERIFIED against THIS worktree (research-items, off `hardening/rag-poison-wiring`):
  #04's Defect-1 (TOCTOU read-once buffer) is ALREADY SHIPPED here — `_read_file_bytes`,
  `_verify_buffer_digest`, and the `pickle.load(io.BytesIO(data))` tail all exist in
  `src/na0s/integrity/safe_pickle.py` (lines 189-204, 839-891, 894-964). What #04 still owes —
  the restricted `find_class` allowlist (Defect-2) — is ABSENT (line 964 is bare
  `pickle.load(io.BytesIO(data))`; `git grep -E "find_class|Unpickler|PICKLE_ALLOWLIST"` over
  src/ returns ZERO relevant hits). R5 is therefore NOT a duplicate of #04: it is the *policy
  half* of #04's still-unbuilt allowlist — (i) replace the blanket `numpy.*` / `sklearn.*`
  prefix-trust at 04...md:128-130 with an enumerated numpy gadget HARD-DENY set, and (ii) ground
  the policy in the fickling gadget catalog, scoped to the residual pickles that survive #13's
  skops/safetensors migration. No overlap with #08 (which only TESTS the mechanism). The base
  advanced via #476 (PickleBall now at ROADMAP_V2.md:2781, was :2528; Item-4 row at :1382) — the
  cross-links exist but the mechanism does not.
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]      # + Q1, Q2, Q3, Q4, Q8, Q10
na_steps: [5, 9, 10]                              # + Q5, Q6, Q7, Q9
status: PLAN-ONLY (no source/test/roadmap edits in this pass — spec file is the only write)
canonical_file: src/na0s/integrity/safe_pickle.py
shim: src/na0s/safe_pickle.py  (do NOT edit — redirects to canonical)
skills: [na0s-review-checklist, subsystem-context-pack]
depends_on:
  - "#04 allowlist scaffold — R5 supplies the POLICY (deny-set + catalog grounding) the #04
     mechanism enforces. If #04's `find_class` hook is co-landed, R5 fills its allowlist/deny
     constant; if not, R5 lands the constant + the minimal `_SafeUnpickler` together."
  - "#13 (format-migration-skops-safetensors, ROADMAP_V2.md:1382-area) — SCOPES R5: the deny-set
     only has to cover the pickles that REMAIN after #13. Soft dep — R5 ships independently and is
     simply narrowed (fewer residual pickles) once #13 lands."
  - "fickling >= 0.1 (NOT installed; #11 `cisec` extra adds it) — used OFFLINE at spec-execution
     time to cross-check the deny-set against fickling's gadget catalog. NOT a runtime import."
local_only: "NO GitHub at any execution step — no push/PR until the user says so."
---

# R5 — Allowlist deny-set + fickling gadget catalog (EXTENDS #04)

## 0. What changed under this worktree (re-verify before executing)

Spec #04 was authored against `hardening/rag-poison-wiring` BEFORE the read-once buffer landed.
On THIS worktree the buffer half is already done, so R5's surface is strictly the **policy** of
the still-missing `find_class` allowlist. Verified facts (cite on execution):

| #04 claim | State in THIS worktree | Consequence for R5 |
|---|---|---|
| Defect-1 TOCTOU (3 opens) | **FIXED** — `_read_file_bytes` (189-204), `_verify_buffer_digest` (839-891), `pickle.load(io.BytesIO(data))` (964) | Out of scope — do NOT re-plan the buffer. |
| `_NumpyCompatUnpickler` exists only on `91944d6` | **Still absent**; but the numpy `_core`↔`core` reality is real | Both `numpy._core.*` AND `numpy.core.*` are LIVE in the artifacts (see §1) — the policy must allow both, deny the gadgets under each. |
| Defect-2 restricted `find_class` | **ABSENT** — line 964 is bare `pickle.load(io.BytesIO(data))`; no `Unpickler` subclass anywhere in src | R5's deliverable. |
| #04 §6 "allow `numpy.core.*`/`numpy._core.*`/`sklearn.*`/`scipy.*` by prefix" (04...md:128-130) | the blanket-prefix idea is exactly what R5 must HARDEN | Replace blanket prefix-trust with prefix-allow **minus an enumerated numpy gadget deny-set**. |

`fickling` and `modelscan` are **not installed** (`pip show` → not found); `numpy` is **2.4.4**;
`scipy` is **not** in the artifact global set (measured, §1). `char_tfidf_vectorizer.pkl` is
referenced by `predict.py:291` but is **not on disk** and **not in KNOWN_HASHES** (optional).

## 1. Explore — the measured global set (the empirical ground truth)

Step-1 exploration already run (reproduce on execution with `PYTHONPATH=src`):
instrument a `pickle.Unpickler` subclass that records every `(module, name)` `find_class`
receives during a successful load of each bundled artifact, take the UNION. Result on numpy 2.4.4:

```
numpy                          :: dtype, float64, ndarray
numpy._core.multiarray         :: scalar
numpy._core.numeric            :: _frombuffer
numpy.core.multiarray          :: _reconstruct, scalar      # ← legacy alias, STILL emitted
sklearn.calibration            :: CalibratedClassifierCV, _CalibratedClassifier
sklearn.feature_extraction.text:: TfidfTransformer, TfidfVectorizer
sklearn.isotonic               :: IsotonicRegression
sklearn.linear_model._logistic :: LogisticRegression
sklearn.preprocessing._data    :: StandardScaler
```

Top-level prefixes touched: **only `numpy` and `sklearn`** (no `scipy`, no `builtins`, no `os`).
Artifacts: `model.pkl`→`CalibratedClassifierCV`, `tfidf_vectorizer.pkl`→`TfidfVectorizer`,
`structural_scaler.pkl`→`StandardScaler`, `model_embedding.pkl`→`CalibratedClassifierCV`. All four
load clean. This is the **measured allowlist** — not a guessed one (na0s-review-checklist §1/§7).

## 2. Gap vs ideal

| | Current (this worktree) | Ideal (R5) |
|---|---|---|
| `find_class` policy | none — bare `pickle.load(io.BytesIO(data))` (964) resolves ANY global the opcode stream names | restricted unpickler: prefix-allow `numpy.*`/`sklearn.*` **minus** an enumerated numpy/sklearn **gadget deny-set**, exact-allow `builtins.{int,float,...}` only if measured, deny everything else |
| numpy trust | #04 plan = blanket `numpy.*` prefix (04...md:128-130) | numpy submodule gadgets (`numpy.testing._private.utils.runstring`/`assert_*`, `numpy.lib.utils`, `numpy.core._exceptions`, `numpy.f2py`, `numpy.distutils`, `numpy.ctypeslib`, etc.) HARD-DENIED even though their parent prefix is allowed — a prefix-allow MUST be intersected with a deny-set |
| policy provenance | ad-hoc | grounded in fickling's gadget catalog (the maintained, CVE-tracked list of pickle RCE primitives) + the measured set; deny-set documented as *fickling-derived*, not invented |
| scope | "all pickles forever" | scoped to **residual post-#13 pickles** — the artifacts that are NOT migrated to skops/.npz/safetensors keep this gate; migrated ones don't reach `pickle.load` at all |
| failure mode | n/a | denied global → `na0s.integrity_audit` `find_class_blocked` event + `pickle.UnpicklingError` (a `ValueError` subclass, so `predict.py`'s `except (ValueError, …)` guards still degrade gracefully) |

**Why a deny-set ON TOP of a prefix-allow (the core R5 insight).** A pure `numpy.*` prefix-allow is
unsafe: numpy ships callables that are RCE/abuse gadgets reachable under its own namespace (e.g.
`numpy.testing._private.utils.runstring` executes a string; `numpy.ctypeslib`/`numpy.f2py` reach
native code; `numpy.distutils`/`numpy.lib.utils.safe_eval` historically wrap `eval`). fickling's
catalog tracks these. So the policy is **(prefix-allow ∩ NOT deny-set) ∪ exact-allow**, where the
deny-set is the fickling-catalogued numpy (and sklearn) gadgets. This is the precise hardening of
04...md:128-130's blanket trust.

## 3. Root-cause implementation plan (numbered, LOCAL-only)

All edits confined to `src/na0s/integrity/safe_pickle.py` (+ a test file + a regeneration helper).
Shim `src/na0s/safe_pickle.py` is NOT touched. **No GitHub at any step.**

1. **Build the deny-set offline, fickling-grounded (the "catalog" half).** Add `fickling>=0.1` to
   the `cisec` optional-deps extra (already slated by #11 — do NOT create a new extra). In a
   `scripts/` one-off helper (`scripts/derive_pickle_policy.py`, NEW, dev-only, NOT imported at
   runtime), enumerate fickling's catalogued unsafe imports (via `fickling`'s analysis API —
   `is_likely_safe`/`check_safety` + its internal unsafe-import list; pin the exact symbol names
   AGAINST the installed fickling version, do not hallucinate them) and intersect with the numpy /
   sklearn namespaces to produce the concrete `_GADGET_DENY` tuple. Emit the tuple as text the
   spec executor pastes into `safe_pickle.py` with a provenance comment (`# fickling vX.Y catalog,
   derived <date>; regenerate via scripts/derive_pickle_policy.py`). This makes the deny-set
   *measured + catalog-backed*, never a magic list (na0s-review-checklist §1, §7).

2. **Freeze the measured allowlist** (the §1 union) as a module-level `frozenset`
   `_PICKLE_ALLOW_EXACT: frozenset[tuple[str,str]]` plus an allow-prefix tuple
   `_PICKLE_ALLOW_PREFIXES = ("numpy.", "numpy", "sklearn.", "scipy.")` (keep `scipy.` as
   forward-headroom for sklearn internals even though unused today — justify in a comment;
   it is allow-by-prefix, still gated by the deny-set, so it adds no gadget exposure).

3. **Define the numpy/sklearn gadget HARD-DENY** `_GADGET_DENY: frozenset[tuple[str,str]]`
   (exact pairs) + `_DENY_PREFIXES` (module prefixes denied outright regardless of allow-prefix):
   `numpy.testing`, `numpy.f2py`, `numpy.distutils`, `numpy.ctypeslib`, `numpy.lib.utils`,
   `numpy.core._exceptions` (and the `_core` twin), plus the universal RCE primitives already
   named in #04 (`os`, `posix`, `nt`, `subprocess`, `builtins.eval`, `builtins.exec`,
   `builtins.__import__`, `builtins.getattr`, `builtins.setattr`, `importlib`, `sys`, `socket`,
   `pty`, `code`, `commands`). Each entry carries an inline `# fickling-catalogued` or
   `# measured-absent` provenance tag. The deny check runs BEFORE the prefix-allow so a denied
   submodule cannot ride in on `numpy.*`.

4. **Introduce `_SafeUnpickler(pickle.Unpickler)`** with `find_class(self, module, name)`:
   a. Apply the numpy `_core`↔`core` remap FIRST (both directions are live per §1 — preserve the
      cross-numpy-version behavior the #04 plan called for; remap then policy-check the *target*).
   b. If `(module, name)` ∈ `_GADGET_DENY` OR `module` startswith any `_DENY_PREFIXES` → deny.
   c. Else if `(module, name)` ∈ `_PICKLE_ALLOW_EXACT` OR `module`/its remap startswith any
      `_PICKLE_ALLOW_PREFIXES` → `super().find_class(module, name)`.
   d. Else → deny.
   Deny path: emit `_audit` `find_class_blocked` JSON (`event`, `module`, `name`) and raise
   `pickle.UnpicklingError(f"blocked global {module}.{name}")`.

5. **Wire it into the existing read-once tail.** Change ONLY line 964 from
   `return pickle.load(io.BytesIO(data))` to `return _SafeUnpickler(io.BytesIO(data)).load()`.
   Everything upstream (read-once buffer, magic check, dual digest verify, sklearn-warning
   suppression `with warnings.catch_warnings()` block 957-963) is preserved verbatim. The
   `_SafeUnpickler(...).load()` MUST stay INSIDE the `catch_warnings()` block so the
   `InconsistentVersionWarning` suppression (the C1 Intel-mac fix) is not regressed.

6. **Scope-gate to residual pickles (the #13 dependency).** Add a one-line comment at the deny-set
   anchoring it to "pickles surviving the skops/safetensors migration (#13)". No code branch on
   #13 — migrated artifacts simply never reach `safe_load`/`pickle.load`, so the gate is a no-op
   for them. Document that when #13 lands, re-run `scripts/derive_pickle_policy.py` to shrink the
   allow-set to whatever residual pickle types remain (keeps the allowlist minimal).

7. **No new env flag, no threshold.** Always-on, like the existing magic-byte check. No similarity
   cutoff is involved (this is set-membership, not fuzzy matching) — the na0s-review-checklist
   "similarity cutoffs must be re-calibrated" caveat is **N/A here**; the only "numbers" are exact
   module/name strings, each provenance-tagged.

## 4. Exact files / functions to change

- `src/na0s/integrity/safe_pickle.py` (canonical) ONLY:
  - ADD `_PICKLE_ALLOW_EXACT`, `_PICKLE_ALLOW_PREFIXES`, `_GADGET_DENY`, `_DENY_PREFIXES`
    (module-level frozensets/tuples, provenance-commented).
  - ADD `_SafeUnpickler(pickle.Unpickler)` with the `find_class` policy of §3.4.
  - EDIT exactly line 964: `pickle.load(io.BytesIO(data))` → `_SafeUnpickler(io.BytesIO(data)).load()`
    (inside the existing `catch_warnings()` block — no other line in `safe_load` changes).
  - `import pickle` / `import io` already present (45, 41) — no new stdlib import.
- `scripts/derive_pickle_policy.py` (NEW, dev-only regeneration helper; NOT runtime-imported) —
  fickling-catalog enumeration + measured-load instrumentation → prints the frozensets.
- `tests/integrity/test_safe_pickle_allowlist_denyset.py` (NEW, mirrors source pkg per CLAUDE.md).
- **NO change** to `predict.py` / `cascade.py` source (transparent loader hardening — Q8).
- **NO change** to the shim `src/na0s/safe_pickle.py` (CLAUDE.md forbids).

## 5. Pipeline wiring (Step 4 / Q3 / Q8)

`safe_load` is the single chokepoint. `predict.py:88` `from .integrity.safe_pickle import safe_load`
(direct canonical import, NOT via shim on this worktree — confirmed); call sites `predict.py:379,
380, 457, 507` load vectorizer/model/scaler/char-vectorizer. `cascade.py` reaches the model only
transitively through `predict.py`'s cached loaders — **no direct `safe_load`**, so parity is
AUTOMATIC. No `_HAS_*` flag, no dual-registration: this is loader hardening, not a detection signal.
(Verify on execution: `git grep -n safe_load src/na0s/cascade.py` → expect empty.)

## 6. Test plan — Code + Use-Case (Step 6 / Q4)

New `tests/integrity/test_safe_pickle_allowlist_denyset.py`. Reuse the `tempfile.TemporaryDirectory`
+ `patch.dict(os.environ, {"NA0S_PICKLE_KEY": ...})` fixtures from `tests/integrity/test_safe_pickle.py`.
Every test asserts an observable outcome (return value / raised type+message / audit record) — NO
hollow tests (na0s-review-checklist §4).

**A. Deny-set rejects gadgets (adversarial `__reduce__`):**
1. `test_reduce_os_system_blocked` — `class Evil: __reduce__ -> (os.system, ("touch SENTINEL",))`;
   `safe_dump` it (writes a VALID sidecar over the malicious pickle = simulates a hash-bypass /
   sidecar-rewrite adversary), then `safe_load` → must raise `pickle.UnpicklingError` naming
   `os.system`, AND assert the sentinel file was NEVER created (rejection is PRE-execution).
2. `test_reduce_builtins_eval_blocked`, `test_reduce_builtins_exec_blocked`,
   `test_reduce_subprocess_popen_blocked` — same pattern.
3. `test_numpy_gadget_blocked` — craft a `__reduce__` targeting a fickling-catalogued numpy gadget
   (e.g. `numpy.testing._private.utils.runstring` or whatever the installed fickling version flags)
   → blocked DESPITE `numpy.` being an allow-prefix. This is the R5-specific proof that the
   prefix-allow is intersected with the deny-set (the 04...md:128-130 hardening). Pin the exact
   gadget name against the installed numpy/fickling — do NOT assert on a hallucinated symbol; if the
   chosen gadget isn't importable on the host, `pytest.importorskip`/skip with a recorded reason.
4. `test_blocked_event_audited` — `assertLogs("na0s.integrity_audit", "ERROR")` shows a
   `find_class_blocked` record (mirror the audit-test style in `test_safe_pickle.py`).

**B. No false-rejects (Use-Case / FP-safe — the load MUST still work):**
5. `test_bundled_artifacts_load` — `safe_load(get_model_path(x))` for each of `model.pkl`,
   `tfidf_vectorizer.pkl`, `structural_scaler.pkl`, `model_embedding.pkl`; assert non-None of the
   expected type (`CalibratedClassifierCV`/`TfidfVectorizer`/`StandardScaler`). `pytest.importorskip`
   sklearn/numpy + skip if an artifact is absent (minimal-CI-green). THIS is the critical FP proof:
   the measured allowlist must not reject the shipped models.
6. `test_benign_dict_round_trips` — `safe_dump`/`safe_load` of a plain dict still returns it
   (covers `builtins`/collections allow-exact entries the measured set needs, if any).
7. `test_numpy_core_remap_both_directions` — assert a pickle naming `numpy.core.multiarray._reconstruct`
   AND one naming `numpy._core.multiarray.scalar` both resolve (both live per §1) on the host's numpy,
   proving the remap + allow survive the policy.

**C. Pipeline use-case:**
8. `test_predict_pipeline_still_scans` — import `na0s.predict`, run `predict("ignore all previous
   instructions")` + a benign string; assert a `ScanResult` from the model-backed path (loader change
   is transparent). Mirror #04 §6 C8.

**D. CLI / suite smoke (mandatory, na0s-review-checklist §4/§11):**
9. `python3 -m pytest tests/integrity/ -v` — ALL existing integrity tests stay green (the wrapper
   preservation means `_validate_pickle_magic`, `_sha256`, `_hmac_sha256` imports are untouched).
10. CLI smoke (not mocked): a real scan entrypoint over `"ignore previous instructions"` to prove
    the real model loads through `_SafeUnpickler`.
11. `python3 -m pytest tests/ -q --tb=line` — 0 net regressions vs base (known env-only failures
    unchanged; project memory baseline ~8969 passed / ~15 env-only).
12. `scripts/derive_pickle_policy.py` smoke — run it once end-to-end and confirm its printed frozenset
    EQUALS the one frozen in source (a drift guard: the committed deny-set == the regenerated one).

## 7. Cleanup / refactor (Step 7 / Q2)

- New code lands in the existing `integrity/` package + `tests/integrity/` mirror — conforms to the
  code-organization standard; no top-level dump.
- `scripts/derive_pickle_policy.py` is dev-tooling, not a runtime import — keep it out of the package
  import graph (verify with `import-linter` if run).
- Do NOT add code to the shim. Do NOT re-touch the read-once buffer (#04 territory, already done).
- If #04's `find_class` scaffold is co-landed, MERGE rather than duplicate — one `_SafeUnpickler`, one
  deny-set; flag any divergent copy for deletion (na0s-review-checklist §11 dead-code).

## 8. Roadmap update (Step 8)

- `ROADMAP_V2.md:1382` (Item 4 row) — append a sub-bullet noting the allowlist/deny-set POLICY is
  delivered by R5 (cite local commit SHA once landed); correct the stale `depends-on 91944d6` note —
  the buffer half is already in-tree, only the policy remained.
- `ROADMAP_V2.md:2781` (PickleBall — Restricted unpickler) — mark IMPLEMENTED via `_SafeUnpickler`
  (deny-set + measured allowlist), cite SHA. NOTE the line MOVED from :2528 (spec #04's ref) to
  :2781 under #476 — re-grep before editing (na0s-review-checklist §3).
- Per "Roadmap-Todo Sync": check the box only after the full suite is green; cite the commit SHA.

## 9. README / Benchmark (Step 9)

**N/A — no detector recall/FPR change.** The two-sided recall harness, `benchmark.py`, and
COVERAGE_MATRIX are untouched (loader hardening, not a detection signal). Optional one-line CHANGELOG
/ L11 prose note (model after `ROADMAP_V2.md:1289`) about the restricted-unpickler deny-set —
low-priority, no numbers.

## 10. N/A steps & Q&A (honest justifications)

- **Step 5 — N/A** — No HARVESTED ATTACK DATASET. R-items are supply-chain/governance infra; the only
  "dataset" is crafted malicious pickles authored in-test (adversarial `__reduce__`) — a binary RCE
  artifact, not a prompt-injection text sample the F14 harvester ingests. Do NOT route through
  `data-harvesting`/`eval-scenario-curation`.
- **Step 9 (benchmark) — N/A** — no recall/FPR delta (see §9).
- **Step 10 — N/A** — Taxonomy codes + COVERAGE_MATRIX classify prompt-injection attack classes; an
  unpickler RCE deny-set is a CWE-502 supply-chain control with no `data/taxonomy.yaml` code, no
  coverage-matrix row, and no per-attack scorer threshold.
- **Q1 — APPLICABLE** — "Can Na0S handle it?" Currently NO (unrestricted `find_class` at 964). Fix per
  §3; prove with §6 tests + full suite green.
- **Q2 — APPLICABLE** — cleanup per §7 (no shim edit; merge-not-duplicate with #04; dev-script kept
  out of the runtime graph).
- **Q3 — APPLICABLE** — single `safe_load` chokepoint; predict/cascade parity automatic (§5).
- **Q4 — APPLICABLE** — Code (§6 A: gadget/numpy-gadget rejection + audit) AND use-case (§6 B: bundled
  artifacts still load, FP-safe; §6 C: full `predict()` scans).
- **Q5 — N/A** — no harvested dataset; crafted pickles only (see Step 5).
- **Q6 — N/A** — not a taxonomy/coverage concern (see Step 10).
- **Q7 — N/A** — `safe_load` is binary refuse/allow; there is no per-attack score or similarity
  threshold to calibrate (set-membership, exact strings only).
- **Q8 — APPLICABLE / YES** — `predict.py:88` imports `safe_load` (canonical, not shim here), calls at
  379/380/457/507; no `predict.py` edit needed (transparent). `cascade.py` has NO direct `safe_load`
  (grep-verify empty) → inherits the hardening via predict's cached loaders.
- **Q9 — N/A** — the harvester agent does not harvest pickle-RCE samples.
- **Q10 — APPLICABLE** — extra checks: (a) the deny-set must be intersected with the allow-prefix so a
  numpy gadget can't ride in on `numpy.*` (the §6 A3 proof — R5's whole point); (b) `pickle.UnpicklingError`
  (ValueError subclass) keeps `predict.py`'s `except` guards degrading gracefully; (c) numpy `_core`↔`core`
  remap preserved both directions (§6 B7) so artifacts load across numpy 1.26/2.4; (d) `_SafeUnpickler.load()`
  stays inside the `catch_warnings()` block (no regression of the sklearn-warning suppression / C1 fix);
  (e) the committed deny-set == the regenerated one (§6 D12 drift guard).

## 11. Agent / skill team per step (inject `na0s-review-checklist` into EACH subagent prompt)

| Step / area | Agent | Checklist injection focus |
|---|---|---|
| Plan authoring (this spec) | `Plan` + skill `na0s-review-checklist` | hallucinated-API + arbitrary-threshold + import-blindness |
| §1 measured set + §3.1 fickling catalog derivation | `l3-l5-code-auditor` (model-load path) | "no magic list" — deny-set must be MEASURED + fickling-catalog-backed, not guessed; verify every fickling symbol against the installed version |
| §3.3-3.4 deny-set + `_SafeUnpickler` | `layer-9-11-auditor` (L11 owner) + `silent-failure-hunter` | prefix-allow ∩ deny-set ordering; silently-swallowed `UnpicklingError`?; silent-refactor-destruction |
| §5 wiring (single-line tail change) | `layer-9-11-auditor` | one-line diff discipline; keep `catch_warnings()` block intact |
| §6 tests | `silent-failure-hunter` + skill `na0s-debugging` | hollow-tests + mocked-CLI-smoke-gap; sentinel-untouched proves pre-execution reject |
| §6 D suite gate | skill `na0s-debugging` (verify-against-MAIN, PYTHONPATH=src) | full-suite-green; env-only-failure parity |
| §7 cleanup | `l3-l5-code-auditor` | merge-not-duplicate with #04; dev-script out of import graph |
| §8 roadmap | `Plan` | re-grep moved line numbers (:2528→:2781) before editing |

(`subsystem-context-pack`: pack `src/na0s/integrity/**,tests/integrity/**` `--compress` to seed each
auditor with only the L11 integrity slice — bounded context, local `/tmp` output only.)

## Execution preconditions / dependencies

1. **Work in a dedicated git worktree** off the correct base (per multi-agent worktree discipline —
   never branch-switch the primary checkout). Verify symbols with `PYTHONPATH=<worktree>/src` (the
   editable install points elsewhere).
2. **#04 coordination:** if #04's `find_class` scaffold is co-landed, R5 fills its allowlist/deny
   constant (one `_SafeUnpickler`); if #04 hasn't landed, R5 lands the minimal `_SafeUnpickler` +
   the policy together (the buffer half is ALREADY in-tree, so this is small).
3. **#13 is a SOFT dep** — R5 ships independently; it only shrinks once #13 reduces the residual
   pickle surface. Do NOT block on #13.
4. **`fickling` is NOT installed** — install it locally (the `cisec` extra, keyless, offline) ONLY to
   run `scripts/derive_pickle_policy.py`; it is NOT a runtime dependency of `safe_pickle.py`. Pin the
   exact fickling catalog symbol names against the installed version — no hallucinated API.
5. **Bundled artifacts present locally** for the FP-safety test (§6 B5) — confirmed on disk
   (`model.pkl`/`tfidf_vectorizer.pkl`/`structural_scaler.pkl`/`model_embedding.pkl`); else
   `importorskip` + note the CI-matrix follow-up. `char_tfidf_vectorizer.pkl` is optional (absent).
6. **No API key, no network at runtime** (loader hardening is fully local/keyless). fickling's
   discovery is offline against the local catalog.
7. **LOCAL ONLY — NO GitHub** (no push/PR) at any step until the user explicitly says so.

## Definition of done

- [ ] `_SafeUnpickler` replaces the bare `pickle.load(io.BytesIO(data))` at `safe_pickle.py:964`
      (inside the existing `catch_warnings()` block); read-once buffer + dual digest verify untouched.
- [ ] Deny-set is MEASURED (the §1 union) + fickling-catalog-derived (regen recipe in
      `scripts/derive_pickle_policy.py`), each entry provenance-tagged — no magic list.
- [ ] Prefix-allow is INTERSECTED with the gadget deny-set: a `numpy.*` gadget (`numpy.testing…`,
      `numpy.f2py`, etc.) is BLOCKED despite the allow-prefix (§6 A3) — the 04...md:128-130 hardening.
- [ ] `os.system`/`eval`/`exec`/`subprocess.Popen` + any non-allowed global → `pickle.UnpicklingError`
      + audited `find_class_blocked` event; sentinel proves PRE-execution rejection (§6 A).
- [ ] Zero false-rejects: all 4 bundled artifacts still `safe_load` to the expected sklearn types under
      the host numpy (§6 B5); `numpy._core`↔`core` remap preserved both directions (§6 B7).
- [ ] All existing `tests/integrity/*.py` still pass (helper signatures + read-once path preserved).
- [ ] `predict.py`/`cascade.py` unchanged; full `predict()` smoke + CLI smoke load the real model
      through `_SafeUnpickler`; sklearn `InconsistentVersionWarning` suppression NOT regressed.
- [ ] `scripts/derive_pickle_policy.py` regenerates a deny-set EQUAL to the committed one (drift guard).
- [ ] `python3 -m pytest tests/ -q --tb=line` — 0 net regressions vs base.
- [ ] `ROADMAP_V2.md:1382` (Item 4) + `:2781` (PickleBall) updated with the local commit SHA; stale
      `depends-on 91944d6` note corrected; moved line numbers re-grepped before editing.
- [ ] NO GitHub push/PR — all work stays LOCAL until the user authorizes it.
