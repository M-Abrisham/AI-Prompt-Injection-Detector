---
item: 12
title: worm/detector.py joblib guard → canonical 3-tier integrity hierarchy
priority_tier: P1 (supply-chain / arbitrary-code-execution surface; pickle-via-joblib load of an attacker-influenceable model path)
depends_on: []            # self-contained. OPTIONAL coordination with item 5 (shared-helper extraction) — see §8.
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]    # template steps + Q1, Q2, Q3, Q4, Q8, Q10
na_steps: [5, 9, 10]                            # + Q5, Q6, Q7, Q9 (see N/A justifications)
classification: Wiring/Integrity (supply-chain). The worm engine is a DETECTOR, but THIS change hardens the integrity of its model LOAD, not worm-detection logic.
---

# Item 12 — `worm/detector.py` joblib guard → canonical 3-tier integrity hierarchy

## 0. Root cause (one sentence)
`_WormCorpusClassifier` deserializes its model with a hand-rolled **single-tier** guard
(`joblib.load(path)` after a bespoke plain-SHA-256 `.sha256` sidecar check at
`src/na0s/worm/detector.py:604–658`), duplicating — and weaker than — the project's canonical
**3-tier** loader `na0s.integrity.safe_pickle.safe_load` (hardcoded `KNOWN_HASHES` → HMAC-SHA256
sidecar → plain SHA-256 sidecar, with pre-hash pickle-magic validation, atomic writes, constant-time
compare, and audit logging), so the worm corpus model is the only ML artifact in the repo not routed
through the shared trust hierarchy.

---

## 1. KEY REFS — confirmed current line numbers (verified against the real file, not guessed)

**Ref drift:** the item's KEY REFS cited `607–651`. Confirmed actual: the `_load_model()` body is
**`detector.py:604–658`**; the bespoke integrity block (sidecar existence + read + SHA-256 compare) is
**`619–644`**; the `joblib.load` deserialize is **line 647**; `predict_proba` consumes it at **686**;
the matching **write** side (`train()` → `joblib.dump` + bespoke `.sha256` write) is **`698–730`**
(specifically `joblib.dump` at **726**, `_hash_file` at **727**, sidecar write at **728–729**).
`_hash_file` helper = **595–602**.

| Concern | File:line | Current behavior | Tier coverage |
|---|---|---|---|
| Load | `worm/detector.py:647` (`obj = joblib.load(path)`) | bespoke `.sha256`-only gate at `619–644`, then `joblib.load` | **Tier 3 only** (plain SHA-256 sidecar) |
| Save | `worm/detector.py:726–729` (`joblib.dump` + `_hash_file` + write `.sha256`) | writes plain SHA-256 sidecar; **never** HMAC, **never** KNOWN_HASHES | **Tier 3 only** |
| Hash helper | `worm/detector.py:595–602` (`_hash_file`) | re-implements `_sha256` already in `safe_pickle.py:57–62` | duplicate logic |
| Default model path | `worm/detector.py:584–586` | `~/.na0s/models/worm_classifier.joblib` (user-trained; not bundled) | — |

**Canonical infra confirmed** at `src/na0s/integrity/safe_pickle.py`:
- Module docstring (`:7–24`) literally documents the **3-tier "Trust hierarchy"** this item wants.
- `safe_load(path)` (`:295`): `_validate_pickle_magic` (`:128`, pre-hash, fails fast) → `_resolve_expected_hash`
  (`:214`: KNOWN_HASHES `models/__init__.py:26` → `.hmac` → `.sha256`) → `hmac.compare_digest` (`:335`) →
  `pickle.load`. Raises `ValueError` on mismatch, `FileNotFoundError` when no digest source exists.
- `safe_dump(obj, path)` (`:247`): pickles + writes HMAC sidecar when `NA0S_PICKLE_KEY` set (warns + writes
  `.sha256` otherwise); atomic temp-file writes (`:160–180`); POSIX permission warnings (`:183`); structured
  JSON audit to `na0s.integrity_audit` (`:280`).
- `KNOWN_HASHES` (`models/__init__.py:26–31`) — hardcoded SHA-256s live INSIDE the pip-signed Python source,
  so an attacker who tampers a `.pkl` cannot also forge the expected hash.

**Empirically verified (this environment, sklearn + joblib present):**
1. `pickle.load` on a `joblib.dump`'d file **FAILS** (`UnpicklingError: invalid load key, '\x0b'`):
   joblib frames large numpy arrays in its own container even though the first 2 bytes are a valid
   PROTO opcode (`0x80 0x05`). → **`safe_load` CANNOT read an existing `.joblib` artifact** — the
   format must change, not just the loader.
2. A sklearn `Pipeline([TfidfVectorizer(ngram_range=(1,2)), LogisticRegression])` (exactly what
   `train()` builds at `detector.py:715–718`) **round-trips cleanly through `safe_dump`/`safe_load`**
   (plain pickle) with `predict_proba` intact and a `.sha256` sidecar written. → migrating off joblib
   to `safe_dump`/`safe_load` is the clean, dependency-free path; joblib is NOT required for this model.

**Wiring confirmed:**
- `predict.py` / `cascade.py` contain **zero** direct `_WormCorpusClassifier`/`worm_classifier` references
  (grep clean). The worm engine reaches the input cascade only via the full `WormSignatureDetector.scan()`
  (ROADMAP WD-1/cascade-parity), and the corpus classifier is consumed inside `scan()` at
  `detector.py:1602` (`self._corpus_classifier.predict_proba(text)`); output-side via
  `output/propagation.py:18,52,96`. So this load-integrity change is **transitively** reached, no new
  predict/cascade flag (see Q8).
- `na0s.safe_pickle` (top-level) is a **deprecation shim** → `na0s.integrity.safe_pickle`. New code must
  import the canonical path (CLAUDE.md: never add to shims).
- `verify_file_digest` / `write_digest_sidecar` do **NOT** yet exist in `safe_pickle.py` (item 5 proposes
  adding them; not landed). This item therefore targets the **already-shipped** `safe_dump`/`safe_load`
  directly (zero new dependency) — see §8 for optional item-5 alignment.

---

## 2. Gap vs. ideal

| Aspect | Current (`worm/detector.py`) | Ideal (canonical) |
|---|---|---|
| Trust tiers | **Tier 3 only** — plain SHA-256 sidecar an attacker with write access can rewrite alongside the model | **All 3 tiers** — KNOWN_HASHES (most trusted, source-signed) → HMAC-SHA256 (forge-proof without the key) → SHA-256 (legacy fallback) |
| Compare | `actual_hash != expected_hash` (`:632`) — **non-constant-time** string compare (timing side-channel) | `hmac.compare_digest` (`safe_pickle.py:335`) |
| Pre-deserialize validation | none — sidecar checked, then straight to `joblib.load` | `_validate_pickle_magic` fails fast on malformed bytes BEFORE any hash/deserialize |
| Write atomicity | `joblib.dump` + separate sidecar `open().write()` (`:726–729`) — crash mid-write ⇒ model/sidecar skew | atomic temp-file + `os.replace()` for both (`safe_pickle.py:160–180`) |
| HMAC / key support | none (cannot use `NA0S_PICKLE_KEY`) | `safe_dump` emits `.hmac` when key set; `safe_load` requires the key to verify it |
| Audit logging | `logger.warning`/`logger.info` only | structured JSON to `na0s.integrity_audit` (dump/load/failure events) |
| Duplicate code | bespoke `_hash_file` (`:595–602`) re-implements `safe_pickle._sha256` | one source of truth |

**Edge cases the plan must cover:**
- **Existing on-disk `.joblib` artifacts** (e.g. from a prior `train()` run, or `~/.na0s/models/worm_classifier.joblib`):
  the format changes from joblib to plain-pickle `.pkl`, so old artifacts will not be found at the new path
  and the classifier stays inert (`predict_proba → 0.0`). ROADMAP WD-10 confirms **no worm model ships today**
  (`~/.na0s/models/worm_classifier.joblib` absent) → migration breaks nothing in the default install. Plan must
  still handle a stale-`.joblib`-present host (refuse to silently load the wrong format; degrade to inert).
- **Missing sidecar / not in KNOWN_HASHES**: `safe_load` raises `FileNotFoundError` — must be caught and degraded
  to inert (`_pipeline = None`, `predict_proba → 0.0`), never crash `scan()`.
- **Tampered model**: `safe_load` raises `ValueError` — same catch → inert + audit-logged.
- **Object lacks `predict_proba`**: the existing minimal-validation check (`:648–651`) must be preserved AFTER
  the safe load.
- **Keyless host** (project memory: subscription-only, no `NA0S_PICKLE_KEY`): SHA-256 sidecar path must work;
  `safe_dump` already warns + writes `.sha256`. **Do NOT** introduce any code path that REQUIRES the key.
- **FP-safety**: refusal removes only the *optional* `corpus_classifier_score` signal (one of 7 Bayes inputs,
  `detector.py:759`); the regex/semantic/replication heads still fire. Refusal must not flip any benign verdict.
  No threshold change.

---

## 3. Root-cause implementation plan (numbered, by location)

### 3.1 Load path — `_load_model()` (`detector.py:604–658`)
1. Replace the bespoke block (`619–647`) with a call to the canonical loader:
   `from na0s.integrity.safe_pickle import safe_load` (lazy import inside the method or module-top behind
   the existing `_HAS_JOBLIB`/new guard — see step 5). Keep the `os.path.isfile(path)` early-return (`:614–617`).
2. Wrap `obj = safe_load(path)` in `try/except (ValueError, FileNotFoundError, OSError, EOFError,
   AttributeError, TypeError, KeyError)` → on any, `logger.warning(... exc_info=True)` and leave
   `self._pipeline = None` (inert). `ValueError`/`FileNotFoundError` are the explicit
   tamper / no-digest signals from `safe_load`; the rest cover deserialize failures.
3. **Preserve** the `predict_proba` capability check (`:648–651`) AFTER the load: if the loaded object
   lacks a callable `predict_proba`, log + leave `_pipeline = None`.
4. Delete the bespoke sidecar-existence / read / `!=` compare (`619–644`) and the now-unused `_hash_file`
   (`595–602`) — one integrity implementation, no timing side-channel, all 3 tiers for free.

### 3.2 Save path — `train()` (`detector.py:698–730`)
5. Replace `joblib.dump(pipeline, self._model_path)` + `_hash_file` + sidecar write (`726–729`) with
   `from na0s.integrity.safe_pickle import safe_dump; safe_dump(pipeline, self._model_path)`. `safe_dump`
   writes the `.hmac` (key set) or `.sha256` (keyless) sidecar atomically and audit-logs. Keep
   `os.makedirs(model_dir, exist_ok=True)` (`:724–725`).
6. **Filename/format change**: the artifact is now a plain pickle, not joblib. Change `_DEFAULT_MODEL_PATH`
   (`584–586`) from `worm_classifier.joblib` → `worm_classifier.pkl` so format and extension agree (and so the
   stale-joblib host of §2 cleanly mismatches the new path and stays inert rather than half-loading). Update
   the docstring at `:573–579` accordingly. (Confirm via grep that no other module/script hardcodes the old
   `.joblib` name — `scripts/train_worm_classifier.py` uses the class default and the `--data`/path flow, so it
   inherits the change; verify in Step 4.)

### 3.3 Dependency guard
7. The `_HAS_JOBLIB` flag (`detector.py:67–74`) and `_HAS_SKLEARN` flag gate train/load. After migration,
   **load** no longer needs joblib (uses `safe_pickle` → `pickle`); **train** still needs sklearn to *fit*
   but no longer needs joblib to *persist*. Update `_load_model`'s guard from `if not _HAS_JOBLIB` (`:611`)
   to a guard that does not require joblib (e.g. gate only on `safe_pickle` importability, which is always
   present in-package). Update `train()`'s `_HAS_JOBLIB` RuntimeError (`712–713`) — replace the joblib
   requirement with nothing (safe_dump always available) or keep `_HAS_SKLEARN` (still required to fit).
   *Net*: removes a hard joblib dependency from the worm load path — a small robustness win, not a magic number.

### 3.4 Optional cleanup (Step 7)
8. If item 5 has landed `verify_file_digest`/`write_digest_sidecar`, prefer the *direct* `safe_dump`/`safe_load`
   here anyway (the model IS a pickle we own end-to-end) — the helper variant is for non-pickle artifacts
   (torch). No need to consume item 5's helpers; document the choice.

---

## 4. Step-by-step template instantiation

**Step 1 — Explore current rules around the target.** DONE (§1–2): exact lines confirmed, the canonical
3-tier infra mapped, the joblib-vs-pickle format incompatibility proven empirically, wiring confirmed.

**Step 2 — Roadmap/Taxonomy/README/Coverage for the gaps.** ROADMAP_V2 **L11 section (line 1107)** already
documents `safe_pickle`'s 3-tier hierarchy as the canonical control "Used by 20+ call sites" — the worm
classifier is a missing call site. ROADMAP **WD-10 (line 1045)** confirms no worm model ships and
`scripts/train_worm_classifier.py` exists; the migration must keep WD-10's eventual train/calibrate flow
working through `safe_dump`. No README number changes (internal control). Coverage Matrix: N/A (Step 10).

**Step 3 — Root-cause plan.** DONE (§3).

**Step 4 — Implement + wire (parity).** Per §3.1–3.4. Parity: no predict.py/cascade.py edit — the corpus
classifier is consumed transitively inside `WormSignatureDetector.scan()` (`detector.py:1602`), which both
entry points already call (ROADMAP WD-1 + cascade parity). VERIFY the transitive path stays green and that
`scripts/train_worm_classifier.py` (calls `_WormCorpusClassifier().train()` → `WormSignatureDetector().scan()`)
still runs end-to-end with the new `.pkl` artifact. No new `_HAS_*` flag added to predict/cascade.

**Step 5 — Harvester audit.** **N/A — supply-chain integrity item; the "dataset" here is the corpus model's
serialized BYTES, not harvested prompt-injection intel.** (The worm *corpus* itself is WD-10's concern, a
separate item; this item only changes how that future corpus model is persisted/loaded. No HF/arXiv/GitHub
harvest, no taxonomy tagging, no decontam applies to a load-time integrity gate.)

**Step 6 — Tests (Code + Use-Case).** See §5.

**Step 7 — Cleanup/refactor.** (a) Delete bespoke `_hash_file` (`595–602`) and the inline sidecar/compare
block (`619–644`) — de-dup against `safe_pickle`. (b) Drop the worm load path's hard joblib dependency
(§3.3). (c) Rename artifact `.joblib`→`.pkl` for format honesty (§3.2). (d) `worm/detector.py` is canonical
(not a shim); the top-level `worm_detector.py` shim is untouched (CLAUDE.md). No file moves (worm/ is already
the v1.0.0 home).

**Step 8 — Roadmap update.** Add this item under the **Worm / Self-Replication Detection** section (near
WD-10) AND/OR the L11 integrity section as a new checkbox ("worm corpus model routed through canonical
3-tier `safe_pickle`"); on landing, check off citing the commit SHA. Note the joblib→pickle format change so
WD-10 (train/calibrate) authors target the new `safe_dump` path.

**Step 9 — README/Benchmark.** **N/A — internal integrity hardening; no user-facing behavior or metric
change.** Recall/FPR unchanged (refusal only drops an optional, currently-dormant signal). If a SECURITY.md
threat-model table exists, optionally add the "tampered worm corpus model ⇒ refused pre-deserialize, not
executed" row.

**Step 10 — Taxonomy + Coverage Matrix + per-feature thresholds.** **N/A — this is a load-time integrity
gate, not a detectable attack *class* with a taxonomy code or a scored threshold.** The worm attack class
itself (IM1.6 / AML.T0061) is already covered (ROADMAP WD-4/WD-6); THIS change does not alter detection,
scoring, or coverage of any attack — it changes how a model file is verified. No COVERAGE_MATRIX row, no
scorer threshold, no magic number introduced (the only constants are the SHA-256/HMAC digests + `1<<16`
chunk size already in `safe_pickle.py`; the Bayes `_LR_MULTIPLIERS` corpus weight `45.0` at `detector.py:759`
is untouched).

**Step 11 — PR + held-out test gate.** Branch `hardening/worm-3tier-integrity` off `main` (worktree;
never branch-switch the primary checkout). Require full `pytest tests/ -q --tb=line` green + the new/updated
worm + integrity targeted tests green before merge. `github-pr-prep` → `github-pr-review`. No merge-to-main
without user confirmation (per memory).

---

## 5. Test plan (Code + Use-Case) — `tests/worm/test_worm_corpus_classifier.py` (extend) + `tests/integrity/`

Mirror source layout: corpus-classifier tests stay in `tests/worm/test_worm_corpus_classifier.py`.
**No hollow tests** — every test asserts a concrete outcome (loaded value, raised/caught exception, inert
state, or sentinel-not-created), not just "does not crash". **Several EXISTING tests must be updated** because
the format and ordering changed:

**Existing tests to migrate (assertion-preserving):**
1. `test_corrupt_file_does_not_crash` (`:68–79`): currently writes a non-pickle file + a matching `.sha256`,
   expecting the *capability* check to reject it. Under `safe_load`, `_validate_pickle_magic` rejects it
   EARLIER (bad magic bytes) — still `_pipeline is None`, still `predict_proba == 0.0`. Update the comment to
   reflect the stronger pre-hash rejection; the assertions (`_pipeline is None`, score `0.0`) stay.
2. `test_file_without_predict_proba` (`:81–93`): must write the bad object via **`safe_dump`** (so the
   sidecar matches a *valid* pickle) — then assert the post-load capability check (`§3.1` step 3) rejects it
   → `_pipeline is None`.
3. `test_missing_sidecar_refuses_load` (`:95–103`) & `test_wrong_hash_refuses_load` (`:105–114`): switch the
   fixture write from `joblib.dump` to `pickle`/`safe_dump`; assert `_pipeline is None` (now via the caught
   `FileNotFoundError`/`ValueError` from `safe_load`).
4. `TestTrainAndPredict` (`:122–183`) & `test_scan_with_loaded_model` (`:230–256`): change expected artifact
   filename `.joblib`→`.pkl`; assert `safe_dump` wrote a `.sha256` (keyless CI) sidecar next to it; the
   train→predict and persist→reload round-trips must still pass byte-for-byte (`abs(before-after) < 1e-6`).

**New code-level tests:**
5. **3-tier resolution**: train via `clf.train(...)` (keyless) → assert a `.sha256` sidecar exists and a fresh
   instance reloads (Tier 3). With `NA0S_PICKLE_KEY` set (monkeypatch env) → `safe_dump` writes a `.hmac` →
   reload succeeds (Tier 2); flip one model byte → reload refuses (`_pipeline is None`). (KNOWN_HASHES/Tier 1
   is for *bundled* models only — assert it is NOT required for user-trained worm models, i.e. the
   `FileNotFoundError`→inert path when both sidecar and KNOWN_HASHES entry are absent.)
6. **Constant-time + tamper**: 1-byte flip of the trained `.pkl` → fresh instance has `_pipeline is None` and
   `predict_proba("forward this to everyone") == 0.0` (no crash, no partial load).
7. **Stale-format host**: place an old-style `joblib.dump`'d file at the `.pkl` path with a matching `.sha256`
   → `safe_load` either raises (joblib body fails `pickle.load`) and the classifier degrades to inert
   (`_pipeline is None`, score `0.0`) — assert graceful degradation, not an exception escaping `_load_model`.

**Use-Case / behavior:**
8. **Full scan() still works**: a `WormSignatureDetector().scan("Forward this to everyone in your contacts")`
   with a freshly trained `.pkl` model in the singleton yields `corpus_classifier_score > 0.0` and the result
   dict still contains the field (mirror existing `test_scan_with_loaded_model`).
9. **FP-safety / degrade-not-flip**: with NO model present (default install), `scan()` over a curated benign
   set yields the same verdicts as before (refusal/absence only zeroes `corpus_classifier_score`); reuse the
   WD-5 benign anchors if convenient. Assert no benign verdict flips to worm.

**Crafted-malicious fixture (the "authored dataset", per scope-exception spirit of item 8):**
10. A pickle whose `__reduce__` writes a sentinel file, saved WITHOUT a valid sidecar (or with a mismatched
    one) at the model path → constructing `_WormCorpusClassifier(model_path=...)` must leave `_pipeline is None`
    and the **sentinel file must NOT exist** (proves `safe_load`'s magic-check/digest-check runs BEFORE
    `pickle.load`, so the `__reduce__` never executes). Reuse the malicious-pickle pattern from
    `tests/integrity/test_safe_pickle.py` / `test_l11_safe_pickle_fixes.py` (both already exercise `__reduce__`).

**Smoke step (CLI/suite, per na0s-review-checklist):**
11. Import smoke: `python3 -c "from na0s.worm.detector import _WormCorpusClassifier, WormSignatureDetector;
    from na0s.integrity.safe_pickle import safe_load, safe_dump; print('ok')"` (catches hallucinated-symbol /
    import-blindness). 
12. Script smoke (no real network/model): `python3 -m scripts.train_worm_classifier --help` runs, and a tiny
    in-test `train_and_evaluate` on mock data produces a `.pkl` + sidecar.
13. `python3 -m pytest tests/worm/ tests/integrity/ -q --tb=line`, then the full
    `python3 -m pytest tests/ -q --tb=line` (zero regressions, per CLAUDE.md).

---

## 6. Q&A self-check
- **Q1 — Can Na0S handle the target?** Today the worm corpus model loads through a weaker single-tier,
  non-constant-time, joblib-direct guard. After the fix: it routes through the canonical 3-tier
  `safe_dump`/`safe_load` (KNOWN_HASHES → HMAC → SHA-256), magic-validated pre-deserialize, tampered/absent
  artifacts refused + degraded to inert, full suite green. Verified by §5 tests 5–10 + full suite.
- **Q2 — Cleanup done?** Bespoke `_hash_file` + inline sidecar/compare deleted (de-dup); hard joblib
  dependency dropped from the load path; artifact renamed `.joblib`→`.pkl` for format honesty; canonical
  imports only (no shim writes). `worm/` already at its v1.0.0 home — no file move.
- **Q3 — Pipeline wiring correct?** Yes — hardening lives inside `_load_model`/`train`; the corpus signal is
  consumed transitively in `WormSignatureDetector.scan()` (`detector.py:1602`), which predict.py and cascade.py
  already invoke (WD-1 + cascade parity). No new flag; verify transitive path + the train script stay green.
- **Q4 — Tested for code AND use-case?** Yes — code-level 3-tier/tamper/stale-format (5–7), behavior full
  `scan()` (8), FP-safety degrade-not-flip (9), crafted-malicious pre-deserialize gate (10), smokes (11–13).
- **Q5 — Harvester audit.** N/A — the "dataset" is the model's serialized bytes, not harvested intel; the worm
  *corpus* is WD-10's separate concern.
- **Q6 — Taxonomy + Coverage Matrix.** N/A — load-time integrity gate, not a taxonomy-coded attack class; the
  worm attack class (IM1.6/AML.T0061) is already covered by WD-4/WD-6 and is unchanged here.
- **Q7 — Scorer.** N/A — no per-attack threshold changes; refusal removes only the optional
  `corpus_classifier_score` Bayes input (weight `45.0`, `detector.py:759`, untouched); scoring math unchanged.
- **Q8 — predict.py / cascade.py references?** Indirect only — zero direct `_WormCorpusClassifier` refs (grep
  clean); reached transitively via `WormSignatureDetector.scan()` (worm wired on input by WD-1 + cascade
  parity; output-side via `output/propagation.py`). No direct predict/cascade edit; only a green-import +
  transitive-path verification.
- **Q9 — Harvester agent harvest this type?** N/A — not a harvestable intel type (it is a serialization
  control). (WD-10's harvester already seeds worm *corpus* discovery; orthogonal to this item.)
- **Q10 — Other correctness checks.** (i) Keyless host must verify via `.sha256` — never require
  `NA0S_PICKLE_KEY`. (ii) `safe_load` cannot read a legacy `.joblib` → the rename + inert-degrade keeps a
  stale host safe (test 7). (iii) Refusal must never flip a benign verdict (test 9, FP-safe). (iv) Preserve
  the post-load `predict_proba` capability check. (v) Confirm no other module hardcodes the old `.joblib`
  filename (grep gate). (vi) The numpy-pickle cross-version concern (project memory `_NumpyCompatUnpickler`)
  lives on a different branch and is OUT of scope — but note that a TF-IDF+LR pipeline pickles only numpy
  arrays + sklearn estimators, which `safe_load`'s plain `pickle.load` already round-trips here (verified §1);
  if a future host hits a numpy-2-pickle mismatch, that is tracked separately (project memory), not this item.

---

## 7. Agent / skill assignment (inject na0s-review-checklist into every subagent prompt)

| Step | Owner | Why |
|------|-------|-----|
| 1–2 explore + roadmap | `security-research-auditor` + skill `na0s-debugging` | map the worm loader vs `safe_pickle` against MAIN (`PYTHONPATH=<worktree>/src`); confirm WD-10/L11 roadmap state |
| 3.1–3.3 load/save/guard migration | `layer-9-11-auditor` + skill `security-review` | L11 supply-chain integrity is this auditor's domain; route to canonical 3-tier loader |
| degrade-on-refusal audit | `silent-failure-hunter` | ensure tamper/absent/stale-format ⇒ inert-degrade (not a silent pass of a tampered file, not a crash in `scan()`) |
| 3.2 format/filename + train-script parity | `l3-l5-code-auditor` + skill `na0s-debugging` | the classifier is an sklearn pipeline (L4/L5 ML idiom); verify `scripts/train_worm_classifier.py` end-to-end |
| 6 tests (assertion-rich, migrate existing) | `l3-l5-code-auditor` + skill `security-review` | mirror `tests/worm/` + reuse `tests/integrity/` malicious-pickle pattern; no hollow tests |
| 8 roadmap | `Plan` | check off + cite SHA under Worm + L11 sections |
| 11 PR | `github-pr-prep` → `github-pr-review` (`pr-review-toolkit:review-pr`) + skill `github-ci-fix` | PR prep, review, drive CI green |

**N/A skills for this item:** `eval-harness`, `data-harvesting`, `eval-scenario-curation`,
`incident-to-scenario`, `cron-scheduling`, `detector-authoring` — no eval/harvest/scenario/cron/new-detector
surface in a model-load integrity migration.

---

## 8. Execution preconditions / dependencies
- **Depends-on: none.** Self-contained — `na0s.integrity.safe_pickle.safe_dump`/`safe_load` already ship and
  are battle-tested (20+ call sites; `tests/integrity/test_safe_pickle.py`).
- **Optional coordination with item 5** (`05-route-raw-loaders-safeload`): item 5 may extract
  `verify_file_digest`/`write_digest_sidecar` and even suggests (its Step 7) refactoring THIS worm block to a
  shared helper. They are non-conflicting: item 12 targets the *already-shipped* `safe_dump`/`safe_load`
  (the worm model is a pickle we own end-to-end, so no torch-style "verify-without-deserialize" helper is
  needed). If item 5 lands first, no rebase pain; if item 12 lands first, item 5's optional worm-refactor
  becomes a no-op. **Neither blocks the other.**
- Work in a dedicated git worktree on branch `hardening/worm-3tier-integrity` off `main` (multi-agent worktree
  discipline; never branch-switch the primary checkout).
- Verify symbols against MAIN, not the stale editable install (`PYTHONPATH=<worktree>/src`).
- Keyless: do NOT introduce any code path that REQUIRES `NA0S_PICKLE_KEY` (project memory: subscription-only).
- No worm corpus model ships by default (ROADMAP WD-10) → migration is behavior-neutral on a clean install.

## 9. Definition of done
- [ ] `_WormCorpusClassifier._load_model` (`detector.py:604–658`) loads via `na0s.integrity.safe_pickle.safe_load`; bespoke sidecar/compare block (`619–644`) and `_hash_file` (`595–602`) deleted.
- [ ] `_WormCorpusClassifier.train` (`698–730`) persists via `safe_dump`; bespoke `joblib.dump` + sidecar write removed.
- [ ] Default artifact path renamed `worm_classifier.joblib` → `worm_classifier.pkl` (`584–586`) + docstring updated; no other module hardcodes the old name (grep verified).
- [ ] Load path no longer hard-requires joblib (`_HAS_JOBLIB` guard at `:611` updated); train still requires sklearn to fit.
- [ ] Post-load `predict_proba` capability check preserved; tamper ⇒ `ValueError`, absent digest ⇒ `FileNotFoundError`, both caught ⇒ inert (`_pipeline is None`, `predict_proba → 0.0`).
- [ ] Crafted-malicious `__reduce__` pickle test proves the gate fires PRE-deserialize (sentinel never written).
- [ ] All 3 tiers exercised by tests: KNOWN_HASHES-absent-is-fine (user model), HMAC (key set), SHA-256 (keyless); constant-time compare via `safe_load`.
- [ ] Existing `tests/worm/test_worm_corpus_classifier.py` fixtures migrated (`.joblib`→`.pkl`, `joblib.dump`→`safe_dump`, magic-byte ordering comment) — assertions preserved, not weakened.
- [ ] FP-safe: default-install (no model) verdicts unchanged; no benign verdict flips; no magic threshold added.
- [ ] `scripts/train_worm_classifier.py` runs end-to-end producing a `.pkl` + sidecar (smoke).
- [ ] `pytest tests/worm tests/integrity` green, then full `pytest tests/ -q --tb=line` zero regressions.
- [ ] ROADMAP_V2 checked off with commit SHA (Worm + L11 sections); joblib→pickle format change noted for WD-10.
- [ ] PR opened; merge gated on green held-out/full suite; main-merge confirmed with user.
