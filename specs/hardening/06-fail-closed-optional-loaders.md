---
item: 6
title: "M1 — fail-closed optional loaders (re-raise integrity ValueError)"
priority_tier: P0 (supply-chain / integrity)
depends_on: []          # self-contained; no other hardening item must land first
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]
na_steps: [5, 9, 10]
applicable_qs: [Q1, Q2, Q3, Q4, Q8, Q10]
na_qs: [Q5, Q6, Q7, Q9]
touches: ["src/na0s/predict.py", "src/na0s/ml/predict_embedding.py"]
flag: NA0S_FAIL_CLOSED
---

# Item 6 — M1: fail-closed optional loaders (re-raise integrity `ValueError`)

> **Naming disambiguation (read first).** This item's label "M1" is the
> *hardening-series* item number. It is **NOT** ROADMAP_V2.md's "M1 — two-key
> provenance gate on harvested attacks" (`ROADMAP_V2.md:1473`), which is a
> different feature. Do not conflate them. The roadmap home for *this* work is
> the **Supply-chain / model integrity** section (see Step 8).

## 1. Root cause (confirmed against source, 2026-06-22)

`na0s.integrity.safe_pickle.safe_load()` is the integrity gate. On a tampered
file it raises `ValueError("Integrity check failed for … File may be tampered.")`
(`src/na0s/integrity/safe_pickle.py:346-351`), and on a missing hash it raises
`FileNotFoundError` (`safe_pickle.py:239-244`), and on a bad magic byte
`ValueError` (`safe_pickle.py:128-157`). (`na0s.safe_pickle` is a shim →
`na0s.integrity.safe_pickle`, `src/na0s/safe_pickle.py:13`.)

The **mandatory** loaders correctly let that exception propagate
(`_get_cached_models`, `src/na0s/predict.py:306-307` — no `try/except`). So a
tampered `model.pkl` / `tfidf_vectorizer.pkl` already fails closed.

The **optional** loaders do not. They wrap `safe_load()` in a bare
`except Exception:` and degrade silently to a sentinel:

| Loader | File | safe_load call | swallow site | sentinel |
|---|---|---|---|---|
| `_get_cached_scaler` | `structural_scaler.pkl` | `predict.py:371` | `predict.py:372-375` | `_cached_scaler = False` → returns `None` |
| `_get_cached_char_vectorizer` | `char_tfidf_vectorizer.pkl` | `predict.py:403` | `predict.py:404-407` | `_cached_char_vectorizer = False` → returns `None` |
| `_get_cached_embedding_structural_scaler` | `embedding_structural_scaler.pkl` | `predict_embedding.py:122-124` | `predict_embedding.py:125-131` | `_cached_…scaler = False` → returns `None` |

Downstream, `_transform()` (`predict.py:435-442`) and `_concat_structural_features()`
(`predict_embedding.py:159-175`) treat `None` as "feature absent" and proceed.
**Net effect:** an attacker who tampers with an *optional* `.pkl` triggers the
integrity `ValueError`, which is caught and discarded — the scan silently runs
with a degraded feature set instead of refusing. That is **fail-open on a
supply-chain integrity violation**: exactly the class of silent failure the
`silent-failure-hunter` and `na0s-review-checklist` (§ "silent refactor
destruction" / "swallowed exceptions") flag.

There is a legitimate reason the `except` exists: a *missing* file is a
backward-compat signal (pre-L3 / pre-L4 / pre-structural-concat models), and
that path is already handled **before** `safe_load` by the `os.path.isfile`
guards (`predict.py:367`, `:399`; `predict_embedding.py:118`). So by the time
`safe_load` is reached the file **exists** — any exception it raises is an
*integrity* signal, not an absence signal. The fix must distinguish
"file absent → degrade" (keep) from "file present but tampered → raise" (new).

**Line-ref drift note:** the KEY REFS cited `predict_embedding.py:121-131,195`
and `predict.py:306-307,370-407,421-435`. Verified current as of HEAD on
`hardening/rag-poison-wiring`: scaler swallow is `predict.py:372-375` (within the
stated `370-407` window), char-vec swallow `404-407`, embedding swallow
`125-131`, embedding `load_models` safe_load `predict_embedding.py:195`. The
`421-435` window in the original ref is `_transform`'s body; the actual
structural-feature branch is `435-442`. No material drift.

## 2. Gap vs. ideal

- **Ideal:** every `safe_load` integrity failure on a *present* file fails
  closed by default (re-raise). A missing optional file still degrades
  gracefully (current correct behavior preserved). An operator may *opt out*
  of fail-closed for the optional loaders only — never for the mandatory
  model — via an explicit env flag, for niche air-gapped recovery.
- **Gap:** all three optional loaders fail open unconditionally; there is no
  `NA0S_FAIL_CLOSED` flag anywhere in the tree (verified: `grep -rn
  NA0S_FAIL_CLOSED src tests scripts` → 0 hits; the `fail_closed` hits in
  `tests/fusion/test_ensemble.py` and `tests/test_technique_analysis.py` are
  unrelated — a `ScanResult` label and the harness coverage gate).

### Edge cases to cover

1. Optional file **absent** → still `None`, no raise (backward compat — pre-L3/L4 model).
2. Optional file **present but tampered** → integrity `ValueError` → **re-raise** (default).
3. Optional file present, tampered, **`NA0S_FAIL_CLOSED=0`** → log + degrade to `None` (explicit opt-out only).
4. Non-integrity error during load (e.g. a `pickle` deserialization crash on a *valid* hash) — must still re-raise under fail-closed, because a structurally broken-but-authentic scaler is itself a config error worth surfacing. (Decide: treat all exceptions uniformly under the flag; do **not** special-case exception type — `safe_load` itself already type-discriminates.)
5. Cache poisoning: the current code caches `False` after a swallow, so a later legit reload never happens. Under fail-closed we must **not** cache a sentinel after a raise — leave `_cached_* = None` so a corrected file can load on retry.
6. Mandatory loader (`_get_cached_models`) and `predict_embedding.load_models` (`predict_embedding.py:195`) must **never** be made opt-out — they are already fail-closed; the flag must not weaken them.

## 3. Default decision (justify the magic value — there is none)

`NA0S_FAIL_CLOSED` defaults to **`1` (enabled)**. Rationale: this is a security
SDK (`project_positioning`); the secure default is fail-closed. The only number
here is the default flag value, justified by the threat model (a tampered model
artifact must not silently downgrade detection). No numeric threshold is
introduced — this is a boolean gate, so the `na0s-review-checklist` "arbitrary
threshold" rule is satisfied by construction.

## 4. Implementation plan (root-cause, numbered)

1. **Add the flag in one place.** Add to `src/na0s/config.py` (the documented
   home for tunables, `config.py:1-11` pattern):
   ```python
   # Fail closed on a tampered OPTIONAL model artifact (integrity ValueError).
   # 1 (default) = re-raise; 0 = log + degrade (air-gapped recovery only).
   FAIL_CLOSED: bool = os.getenv("NA0S_FAIL_CLOSED", "1") not in ("0", "false", "False")
   ```
   Read it at call time (not import time) inside a tiny helper so tests can
   `monkeypatch.setenv` without re-importing — i.e. define
   `def _fail_closed() -> bool: return os.getenv("NA0S_FAIL_CLOSED", "1") not in ("0","false","False")`
   in `config.py` and import that. (Evaluating at import time would make the
   flag untestable per the `na0s-review-checklist` "env blind spot" item.)
2. **`predict.py:_get_cached_scaler`** (`:370-375`): replace the bare
   `except Exception:` so it re-raises when `_fail_closed()` and the file
   existed (it did — guarded at `:367`). On opt-out, keep the existing
   `logger.warning` + `_cached_scaler = False`. On re-raise path, do **not**
   set `_cached_scaler = False` (leave `None`; edge case 5).
3. **`predict.py:_get_cached_char_vectorizer`** (`:402-407`): identical
   treatment.
4. **`predict_embedding.py:_get_cached_embedding_structural_scaler`**
   (`:121-131`): identical treatment. Import the helper as
   `from na0s.config import _fail_closed` (module already imports `os`,
   `predict_embedding.py:41`).
5. **Do not touch** `_get_cached_models` (`predict.py:306-307`) or
   `load_models` (`predict_embedding.py:195`) — already fail-closed; only add a
   one-line comment asserting that invariant so a future refactor doesn't add a
   `try/except` around them.
6. **Message hygiene:** when re-raising, let the original `safe_load`
   `ValueError` propagate unchanged (it already names the path + "File may be
   tampered"). When degrading on opt-out, the warning must say *why* it is
   degrading and that `NA0S_FAIL_CLOSED=0` is set, so the downgrade is auditable
   (ties into the existing `na0s.integrity_audit` logger, `safe_pickle.py:41`).

### Exact files / functions to change
- `src/na0s/config.py` — add `_fail_closed()` helper + `FAIL_CLOSED` const.
- `src/na0s/predict.py` — `_get_cached_scaler`, `_get_cached_char_vectorizer`.
- `src/na0s/ml/predict_embedding.py` — `_get_cached_embedding_structural_scaler`.
- No new module (fits CLAUDE.md "core pipeline files stay at top level"; the
  flag is config, not a new sub-package).

## Step-by-step orchestration (template steps 1-11)

- **Step 1 — Explore current rules around target.** DONE above (§1-2): three
  optional loaders swallow `safe_load`; mandatory loaders already fail closed.
- **Step 2 — Roadmap / taxonomy / README / coverage for the picture.** Roadmap
  home = Supply-chain / model integrity section (alongside the existing
  HMAC-SHA256 sidecar work referenced from `tests/integrity/test_safe_pickle.py`
  docstring). No taxonomy/coverage row applies (Step 10 N/A).
- **Step 3 — Root-cause plan.** § 4 above.
- **Step 4 — Implement + WIRE (predict.py + cascade.py parity).** Parity check:
  `cascade.py` consumes the scaler via the **same** `_get_cached_scaler`
  imported from `predict.py` (`cascade.py:23`, used at `cascade.py:431`), so
  fixing the loader fixes both paths automatically — *no separate cascade edit
  needed* and none should be added (avoid drift). Confirm by grepping callsites:
  `predict.py:637-638,702-703,1536-1537` and `cascade.py:431` all route through
  the patched helpers. This is the key wiring fact (Q3/Q8).
- **Step 5 — HARVESTER AUDIT.** **N/A** — the "dataset" here is a tampered
  binary artifact, not harvested intel; nothing for the threat-intel harvester
  to ingest. (Distinct from item 8, where the dataset *is* crafted malicious
  pickles authored in-tree.)
- **Step 6 — Tests (Code + use-case).** § "Test plan" below.
- **Step 7 — Cleanup / refactor.** Remove the now-redundant `import os` shadow
  inside `_get_cached_models` (`predict.py:294`) only if it is dead given the
  module-level `import os` at `predict.py:63` — verify it is redundant before
  deleting (it is a local re-import). Keep the change minimal; do not gold-plate
  the surrounding cache logic. De-clutter: none of the stray top-level files
  (`_skeptic_test_out.txt`, `pyt_out.txt`, `_xfail_run.txt`) are in scope.
- **Step 8 — Roadmap update.** Add a checked item under Supply-chain / model
  integrity: "fail-closed optional loaders — re-raise integrity `ValueError` on
  a tampered present optional `.pkl`, `NA0S_FAIL_CLOSED` opt-out (default 1)";
  cite the merge SHA when landed (per `feedback_roadmap_sync`).
- **Step 9 — README / Benchmark.** README: add `NA0S_FAIL_CLOSED` to the env-var
  table if one exists; otherwise a one-line note in the security/config section.
  Benchmark: **N/A** — no recall/FPR change (the loaders' *success* path is
  unchanged; only the *tampered* path changes from silent-degrade to raise).
- **Step 10 — Taxonomy / Coverage / thresholds.** **N/A** — this is an
  integrity/supply-chain control, not a prompt-injection attack class; it maps
  to no taxonomy.yaml leaf and no COVERAGE_MATRIX row, and introduces no scorer
  threshold.
- **Step 11 — PR + held-out gate.** § "PR / test-gate" below.

## Test plan (Code + Use-case) — Step 6 / Q4

New isolated test file: **`tests/integrity/test_fail_closed_loaders.py`**
(mirrors source: integrity/supply-chain → `tests/integrity/`, per CLAUDE.md test
org; reuses the `safe_dump`-then-overwrite tamper idiom proven in
`tests/integrity/test_l11_safe_pickle_fixes.py:196-203` and
`tests/integrity/test_safe_pickle.py`).

Tamper recipe (grounded, not hollow): for each optional path, write a *valid*
pickle + matching sidecar via `safe_dump(obj, path)`, point the loader's
`*_PATH` module global at it (monkeypatch `na0s.predict.SCALER_PATH` etc.), reset
the cache (`_cached_scaler = None` / `_reset_embedding_structural_scaler_cache`,
`predict_embedding.py:135-139`), then overwrite the `.pkl` bytes with different
content so the SHA-256 no longer matches → `safe_load` raises
`ValueError(match="Integrity check failed")`.

Code-level tests (per loader × scaler / char-vec / embedding-scaler):
1. `test_<loader>_absent_returns_none_no_raise` — file missing → `None`, no
   exception (backward-compat preserved). (Edge case 1.)
2. `test_<loader>_tampered_raises_by_default` — present+tampered, flag unset →
   `pytest.raises(ValueError, match="Integrity check failed")`. (Edge case 2 —
   the headline assertion required by the item: "tampered structural_scaler.pkl
   → scan() raises".)
3. `test_<loader>_tampered_degrades_when_opt_out` — `monkeypatch.setenv
   ("NA0S_FAIL_CLOSED","0")` → returns `None`, emits the audit warning
   (`caplog`), and **does not** poison the cache to `False` permanently in a way
   that blocks a later good reload. (Edge cases 3 + 5.)
4. `test_<loader>_legit_load_unchanged` — untampered file loads the real object
   (no regression on the success path).
5. `test_fail_closed_default_is_enabled` — `_fail_closed()` is `True` with the
   env unset; `False` for `"0"`/`"false"`. (Asserts the default decision § 3.)

Use-case / behavior test (the end-to-end "scan() raises"):
6. `test_scan_raises_on_tampered_structural_scaler` — monkeypatch
   `na0s.predict.SCALER_PATH` to a `safe_dump`'d-then-tampered file, reset the
   scaler cache, call the real `scan("ignore previous instructions")` and assert
   it propagates `ValueError`. Pair it with
   `test_scan_succeeds_when_scaler_absent` (path → nonexistent file) to prove
   the legit degrade path still yields a normal `ScanResult` (FP-safe: a benign
   prompt still scans SAFE; an injection still scans MALICIOUS via the mandatory
   model). This satisfies the `na0s-review-checklist` "no hollow tests" + CLI/
   suite smoke rule.

No assertion-light tests: every test asserts either a raised type+message or a
concrete returned object / `ScanResult` field, plus a `caplog` check on the
opt-out warning.

## Smoke step (CLI / suite — required)

1. Targeted first: `python3 -m pytest tests/integrity/ -v` (fast, proves the
   new file + no regression in existing safe_pickle tests).
2. CLI smoke (real, not mocked): tamper a temp scaler, point `NA0S` at it, and
   run the package CLI entry on a benign prompt — confirm it exits non-zero /
   surfaces the integrity error (verifies the raise reaches the top level, not
   just the unit). If no CLI flag exposes the path, smoke via a 3-line
   `python -c` that imports `na0s` and calls `scan()` with the monkeypatched
   path inside a `tmp` dir.
3. Full suite last (CLAUDE.md mandate): `python3 -m pytest tests/ -q --tb=line`
   — confirm zero regressions before reporting done. Note the ~15-min runtime;
   verify against MAIN env (`PYTHONPATH=<worktree>/src`) per `na0s-debugging`
   to avoid the stale editable-install trap.

## Q&A self-check

- **Q1 — Can Na0S handle the target?** Not yet for optional loaders (fails
  open). After § 4 it fails closed by default; full suite must stay green.
- **Q2 — Cleanup done?** Step 7: minimal, no top-level dumps; stray
  `*_out.txt` files out of scope.
- **Q3 — Pipeline wiring correct?** Yes — single shared helper means predict +
  cascade get the fix without duplication (`cascade.py:23,431`). No separate
  cascade edit.
- **Q4 — Tested for code AND use-case?** Yes — per-loader unit tests + the
  end-to-end `scan() raises` behavior test.
- **Q5 — Harvester audit.** **N/A** — artifact tamper, nothing to harvest.
- **Q6 — Taxonomy / Coverage.** **N/A** — integrity control, no attack-class
  taxonomy/coverage row.
- **Q7 — Scorer.** **N/A** — no per-attack score; boolean integrity gate.
- **Q8 — predict.py / cascade.py refs?** Yes — predict.py owns all three
  optional loaders (one is in `predict_embedding`); cascade.py reuses
  `_get_cached_scaler`. Both covered.
- **Q9 — Harvester agent harvests this type?** **N/A** — not harvestable intel.
- **Q10 — Other checks.** Mutation/concurrency: the loaders use
  double-checked locking under `_scaler_cache_lock` etc.; ensure the re-raise
  happens *inside* the lock without leaving the lock holding a sentinel
  (it raises, lock releases via `with`, cache stays `None`). Verify no test
  relies on the old swallow behavior (grep the existing scaler tests in
  `tests/structural/`, `tests/ml/test_l5_structural_concat.py`,
  `tests/test_subword_features.py` — they use *valid* fixtures, so unaffected;
  confirm none feed a tampered file expecting `None`).

## Agent / skill team (inject `na0s-review-checklist` into every subagent prompt)

| Step / concern | Agent / skill |
|---|---|
| Lead plan + decomposition | `Plan` |
| Find every swallowed-exception / fail-open loader site, confirm no other optional `safe_load` swallows exist | `silent-failure-hunter` |
| Integrity / supply-chain correctness of the re-raise + flag semantics | `security-research-auditor` + skill `security-review` |
| L3/L5 structural-feature loader code review (`predict.py` scaler/char-vec, `predict_embedding`) | `l3-l5-code-auditor` |
| L9-L11 integrity layer cross-check (safe_pickle contract unchanged) | `layer-9-11-auditor` |
| Test authoring + tamper-fixture correctness, full-suite green, env-trap avoidance | skills `eval-harness`, `na0s-debugging` |
| PR prep + self-review + CI gate | `pr-review-toolkit:review-pr`, skills `github-pr-prep`, `github-ci-fix` |
| Checklist enforcement on the diff | skill `na0s-review-checklist` |

`cron-scheduling` / `data-harvesting` skills: **N/A** for this item (no
scheduled job, no harvest).

## Execution preconditions / dependencies

- **Depends-on: none.** Self-contained; touches only `config.py` + two loaders.
- **Soft ordering with item 1 (import-linter):** if item 1 lands first its
  `vulture`/`deptry` job will *not* flag this change; if this lands first the
  contract still passes (config const + helper are referenced). No hard
  dependency either way.
- **Env:** verify against MAIN, not the d8 editable install
  (`PYTHONPATH=<worktree>/src`) — the optional `.pkl` artifacts and
  `na0s.integrity.safe_pickle` exist on main.
- **Worktree:** do the work in an isolated git worktree on a
  `hardening/fail-closed-optional-loaders` branch off `main` (per
  `project_multi_agent_worktree`); never branch-switch the primary checkout.

## Definition of done

- [ ] `NA0S_FAIL_CLOSED` flag added in `config.py` with a call-time helper
      (default enabled), justified default (no arbitrary numeric threshold).
- [ ] All three optional loaders re-raise the integrity `ValueError` on a
      present-but-tampered file by default; missing-file degrade preserved;
      opt-out (`=0`) logs an auditable warning and degrades.
- [ ] Re-raise path does **not** poison the cache (`_cached_* = None`, retryable).
- [ ] Mandatory loaders (`_get_cached_models`, `load_models`) left fail-closed,
      with an invariant comment guarding against future regression.
- [ ] `tests/integrity/test_fail_closed_loaders.py` — per-loader unit tests +
      end-to-end `scan() raises on tampered structural_scaler.pkl` + benign
      FP-safe degrade test; all non-hollow.
- [ ] `python3 -m pytest tests/integrity/ -v` green; CLI smoke shows the raise
      reaches top level; full `tests/` suite green, zero regressions.
- [ ] README env note + ROADMAP_V2 item checked with merge SHA.
- [ ] PR opened; held-out / full-suite gate passes before merge; merge-to-main
      confirmed with the user (per memory `feedback_no_git_commit`).
