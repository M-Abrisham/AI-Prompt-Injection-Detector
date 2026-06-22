---
item: 5
title: Route 3 zero-integrity raw loaders through safe_load (supply-chain integrity)
priority_tier: P1 (supply-chain / arbitrary-code-execution surface)
depends_on: []            # self-contained; no other hardening item must land first
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]   # template steps + Q1, Q2, Q3, Q4, Q8, Q10
na_steps: [5, 9, 10]                            # + Q5, Q6, Q7, Q9 (see N/A justifications)
classification: Wiring/Integrity (supply-chain), NOT a prompt-injection attack class
---

# Item 5 — Route 3 zero-integrity raw loaders through `safe_load`

## 0. Root cause (one sentence)
Three ML persistence loaders deserialize attacker-influenceable model artifacts with raw
`pickle.load` / `torch.load` and **no integrity check**, so a tampered file on the model path
executes arbitrary code at load time — they must verify a digest (via `integrity.safe_pickle`,
the existing trust hierarchy) *before* deserializing, exactly like `worm/detector.py` already does.

---

## 1. KEY REFS — confirmed current line numbers (verified, not guessed)

| # | File | Loader call | Current line | Format | Integrity today |
|---|------|-------------|--------------|--------|-----------------|
| A | `src/na0s/ml/faiss_classifier.py` | `self._labels = pickle.load(f)` | **219** (ref said 219 ✅) | pickle (`.labels.pkl`) | **none** — raw `open()`+`pickle.load` |
| B | `src/na0s/ml/stacking_classifier.py` | `data = pickle.load(fh)  # noqa: S301` | **130** (ref said 130 ✅) | pickle | **none** — raw `open()`+`pickle.load`, S301 silenced |
| C | `src/na0s/ml/embedding_adapter.py` | `self._adapter.load_state_dict(torch.load(path, map_location="cpu"))` | **437** (ref said 437 ✅) | **torch** state-dict (NOT pickle) | **none** — raw `torch.load`, `weights_only` not set |
| T | `src/na0s/worm/detector.py` | `obj = joblib.load(path)` | **647** (`_load_model` body **604–658**; ref said 607–651, slight drift) | joblib (pickle) | **HAS .sha256 sidecar gate** — the reference template |

- The matching **write** sides (must emit a sidecar so the new load can verify):
  - A: `faiss_classifier.py:197–199` `save()` does `pickle.dump(self._labels, …)` (no sidecar).
  - B: `stacking_classifier.py:118–125` `save()` does `pickle.dump({"model",…}, …)` (no sidecar).
  - C: `embedding_adapter.py:407–419` `save()` does `torch.save(self._adapter.state_dict(), path)` (no sidecar).
- **safe_load infra confirmed** at `src/na0s/integrity/safe_pickle.py`:
  - `safe_load(path)` (line 295): validates pickle magic bytes → resolves expected digest via
    KNOWN_HASHES (hardcoded, `models/__init__.py:26`) → HMAC sidecar (`.hmac`) → SHA-256 sidecar
    (`.sha256`), constant-time compares, then `pickle.load`. Raises `ValueError` on mismatch,
    `FileNotFoundError` when no digest source exists.
  - `safe_dump(obj, path)` (line 247): pickles + writes HMAC sidecar (when `NA0S_PICKLE_KEY` set)
    else SHA-256 sidecar; atomic temp-file writes; permission warnings.
  - Trust hierarchy + audit logging already in place (`na0s.integrity_audit`).
- **No `safe_torch_load` exists** (grep confirmed). `safe_load` is pickle-specific (`_validate_pickle_magic`
  rejects a torch zip archive). Ref C therefore needs a *torch-aware* sidecar-verify helper, not `safe_load`.
- The `_NumpyCompatUnpickler` from project memory lives on a DIFFERENT unpushed branch
  (`ci/test-optional-dep-guards`, commit 91944d6) and is **not on this branch** — do not assume it.
- Shim note: `na0s.safe_pickle` (top-level) is a deprecation shim → `na0s.integrity.safe_pickle`.
  New code must import from the **canonical** `na0s.integrity.safe_pickle` (CLAUDE.md: never add to shims).

---

## 2. Gap vs. ideal

| Loader | Gap (current) | Ideal (target) |
|--------|---------------|----------------|
| A faiss labels | `pickle.load` of `<index>.labels.pkl` with zero verification; an attacker who can write the model dir gets RCE on first KNN query | `save()` calls `safe_dump(self._labels, labels_path)`; `load()` calls `safe_load(labels_path)`; missing/mismatched sidecar ⇒ refuse + disable classifier (set `_init_failed`), never crash the scan |
| B stacking | `pickle.load` of meta-learner dict; same RCE surface; S301 lint silenced rather than fixed | `save()` → `safe_dump`; `load()` → `safe_load`; refuse on tamper; drop the `# noqa: S301` because the real fix replaces the unsafe call |
| C adapter | `torch.load` deserializes a pickle inside the zip with full `__reduce__` power; no digest, no `weights_only` | `save()` writes a SHA-256/HMAC sidecar via the same helper API; `load()` verifies the sidecar against the file bytes **before** `torch.load`, and passes `weights_only=True` where the installed torch supports it (defense-in-depth, not a substitute for the digest) |
| T worm | already gated — **reference pattern only** (no change) | n/a — used as the design template; optionally refactor its bespoke `_hash_file`/sidecar block to reuse the shared helper (Step 7, optional) |

Edge cases to cover in the plan:
- **Legacy artifacts with no sidecar** (already on disk from a prior `save()`): must NOT silently load.
  Behavior = refuse + log + degrade (FAISS/stacking/adapter are all optional, env-gated, graceful-degrading
  signals — refusal disables the optional boost, never breaks `scan()`).
- **Keyless host** (project memory: subscription-only, no `NA0S_PICKLE_KEY`): SHA-256 sidecar path must work;
  `safe_dump` already warns + writes a `.sha256`. No HMAC required.
- **torch not installed** (this env): adapter path must import-guard so the new helper is only exercised
  when torch is present; tests `importorskip("torch")`.
- **FP-safety**: integrity refusal must not flip any benign verdict — refusal only removes an *optional*
  positive signal; the main TF-IDF/embedding verdict is unaffected. No threshold changes.

---

## 3. Root-cause implementation plan (numbered, by file)

### 3.1 Shared torch-aware verify helper (for Ref C)
Add to `src/na0s/integrity/safe_pickle.py` (canonical module — NOT the shim):
1. `verify_file_digest(path) -> None` — extract the digest-resolution + constant-time compare
   logic currently inline in `safe_load` (lines 306–351) into a reusable function that raises on
   tamper but does NOT deserialize. Refactor `safe_load` to call it (no behavior change to existing
   callers; covered by existing `tests/integrity/test_safe_pickle.py`).
2. `write_digest_sidecar(path) -> str` — extract the sidecar-write half of `safe_dump`
   (lines 259–278) so torch artifacts (already written by `torch.save`) can get a matching sidecar
   without re-pickling. Returns the sidecar path written.
   *Justification for refactor*: avoids duplicating the trust-hierarchy logic into `embedding_adapter.py`;
   keeps ONE source of truth for the digest format (`v1:algo:digest`) and the HMAC/SHA-256 decision.
   (No magic numbers introduced — reuses existing `1<<16` chunk size and `hmac.compare_digest`.)

### 3.2 Ref A — `faiss_classifier.py`
3. In `save()` (lines 197–199): replace `with open(labels_path,"wb"): pickle.dump(...)` with
   `from na0s.integrity.safe_pickle import safe_dump; safe_dump(self._labels, labels_path)`.
4. In `load()` (lines 217–219): replace `with open(labels_path,"rb"): pickle.load(f)` with
   `from na0s.integrity.safe_pickle import safe_load; self._labels = safe_load(labels_path)`.
5. Confirm `_ensure_loaded()` (lines 233–264) already wraps `self.load()` in `try/except Exception`
   (it does, line 261) → a `ValueError`/`FileNotFoundError` from `safe_load` sets `_init_failed=True`
   and `classify()` returns the inert SAFE dict (lines 290–298). No new error handling needed; document it.
6. Keep the FAISS index binary (`write_index`/`read_index`, line 195/215) as-is — that's faiss's own
   binary format, not pickle; out of scope for this item (note it as a follow-up risk if faiss
   ever deserializes Python objects, but `IndexFlatIP` does not).

### 3.3 Ref B — `stacking_classifier.py`
7. In `save()` (lines 118–125): replace raw `pickle.dump({"model":…,"trained":…}, fh)` with
   `safe_dump({"model": self._model, "trained": self._trained}, path)`.
8. In `load()` (lines 127–133): replace `data = pickle.load(fh)  # noqa: S301` with
   `data = safe_load(path)`; remove the now-obsolete `# noqa: S301`.
9. Wrap the `load()` call site behavior: stacking is invoked from the cascade meta-learner path;
   confirm its caller tolerates a raised exception (audit caller via Step 4/agent). If the caller
   does NOT guard, add a local `try/except (ValueError, FileNotFoundError)` that logs + leaves
   `self._trained=False` so `is_available()` (line 137–139) returns False and the ensemble degrades.

### 3.4 Ref C — `embedding_adapter.py`
10. In `save()` (lines 407–419): after `torch.save(...)` (line 418), call
    `write_digest_sidecar(path)` so the artifact ships with a verifiable sidecar.
11. In `load()` (lines 421–439): before `torch.load` (line 437), call `verify_file_digest(path)`;
    on raise, log + leave `self._adapter=None` (degrade) rather than propagate, OR propagate if the
    caller guards — decide via caller audit. Then call
    `torch.load(path, map_location="cpu", weights_only=True)` guarded by a version check
    (`weights_only` exists in torch ≥ 1.13; fall back to `weights_only=False` only with a warning on
    older torch). *Justification*: `weights_only=True` blocks arbitrary `__reduce__` execution; it is
    defense-in-depth ON TOP of the digest, not a replacement (a state-dict of tensors loads fine under it).

### 3.5 Wiring / parity (Step 4)
12. No change to `predict.py` / `cascade.py` import lines is required: these three modules are reached
    *transitively* via `ml/predict_embedding.py` (`from .faiss_classifier import get_faiss_classifier`,
    line 51), which `predict.py` gates behind `_HAS_EMBEDDING_CLASSIFIER` (lines 195–204, used line 897)
    and `cascade.py` imports as `load_models`/`classify_prompt_embedding` (line 100). The hardening lives
    inside the load/save methods, so the existing wiring carries it automatically — VERIFY this transitive
    path stays green, do not add redundant flags. (See Q3/Q8.)

---

## 4. Step-by-step template instantiation

**Step 1 — Explore current rules around the target.** DONE above (Section 1–2): all 3 loaders + the
worm template + the `safe_pickle` infra confirmed at exact lines; transitive pipeline path mapped.

**Step 2 — Roadmap/Taxonomy/README/Coverage for the gaps.** ROADMAP_V2 L5 section (lines 736, 754)
already *claims* "Model files load via `safe_dump`/`safe_load` with SHA-256 sidecars" — this is
**aspirational/inaccurate** for these 3 loaders and must be reconciled (the claim is true for
`model.pkl`/`model_embedding.pkl`/scalers via `predict.py` + `predict_embedding.py`, but FALSE for
faiss labels, stacking, adapter). Step 8 fixes the wording or adds the gap as a tracked item.

**Step 3 — Root-cause plan.** DONE (Section 3).

**Step 4 — Implement + wire (parity).** Per 3.1–3.5. Parity check: the three loaders are optional,
env-gated, transitively-wired signals; no new predict/cascade flag needed, only verification that the
transitive path still imports and the existing `_HAS_*` guards still degrade gracefully on refusal.

**Step 5 — Harvester audit.** **N/A — supply-chain integrity item; the "dataset" here is a model
artifact's bytes, not harvested prompt-injection intel.** No HuggingFace/arXiv/GitHub harvest applies.
(Contrast item 8, where malicious pickles are *authored* test fixtures — here we only need 2–3 crafted
tamper fixtures, written in the test, not harvested.)

**Step 6 — Tests (Code + Use-Case).** See Section 5.

**Step 7 — Cleanup/refactor.** (a) The shared-helper extraction (3.1) de-duplicates digest logic.
(b) Optional: refactor `worm/detector.py:595–658` bespoke `_hash_file` + sidecar block to reuse
`verify_file_digest` — keeps one integrity implementation; gate behind "no behavior change + existing
worm tests green". (c) Remove the `# noqa: S301` in stacking (3.3.8). (d) These 3 files are top-level-shim
targets per ROADMAP_V2 (lines 687–698, 821) for the v1.0.0 move to `ml/`; this item does NOT move them
(out of scope) but the touched code should be ready for that move (canonical imports only).

**Step 8 — Roadmap update.** Add/locate item in ROADMAP_V2 "integrity / L11" or L5 section; on landing,
check it off citing the commit SHA. Correct the L5 over-claim noted in Step 2.

**Step 9 — README/Benchmark.** **N/A — internal integrity hardening, no user-facing behavior or metric
change; recall/FPR unchanged because refusal only drops an optional positive signal.** (If a SECURITY.md
threat-model table exists, add the "tampered model artifact ⇒ refused, not executed" row — minor, optional.)

**Step 10 — Taxonomy + Coverage Matrix + per-feature thresholds.** **N/A — this is not a detectable
attack *class* with a taxonomy code or a scored threshold; it is a load-time integrity gate.** No
COVERAGE_MATRIX row, no scorer threshold. (Confirmed: the matrix tracks prompt-injection technique
families, not supply-chain controls.) No magic number is introduced — the only constants are the reused
SHA-256/HMAC digests and chunk size already in `safe_pickle.py`.

**Step 11 — PR + held-out test gate.** Branch `hardening/route-raw-loaders-safeload` off `main`.
Require full `pytest tests/ -q --tb=line` green + the new targeted tests green before merge. Use
`github-pr-prep` then `github-pr-review`. Do not merge to main without confirmation (per memory).

---

## 5. Test plan (Code + Use-Case) — `tests/integrity/` and `tests/ml/`

Mirror source layout: helper tests in `tests/integrity/test_safe_pickle.py` (extend), loader tests in
`tests/ml/`. No hollow tests — every test asserts a concrete outcome (loaded value, raised exception
type, or degraded state), not just "no crash".

**Code-level (helpers, 3.1):**
1. `verify_file_digest` passes for a freshly `safe_dump`-ed file; raises `ValueError` after a 1-byte flip;
   raises `FileNotFoundError` when no sidecar + not in KNOWN_HASHES.
2. `write_digest_sidecar` writes a `.sha256` (keyless) / `.hmac` (with `NA0S_PICKLE_KEY`) that
   `verify_file_digest` then accepts; assert the sidecar content matches the recomputed digest.
3. Regression: existing `tests/integrity/test_safe_pickle.py` + `test_l11_safe_pickle_fixes.py` stay green
   (proves the refactor of `safe_load`/`safe_dump` is behavior-preserving).

**Use-Case / behavior (Refs A/B/C):**
4. **A faiss round-trip**: `save()` → `load()` of labels succeeds and returns equal labels (skip if
   faiss-cpu absent). Tamper the `.labels.pkl` bytes → `load()` raises; via `_ensure_loaded()` the
   classifier sets `_init_failed` and `classify()` returns the inert SAFE dict (assert label=="SAFE",
   score==0.0). Missing sidecar (legacy file) → same refusal.
5. **B stacking round-trip**: `save()`/`load()` restores `_model` + `_trained`; tampered pickle → refusal;
   assert `is_available()` is False after a refused load (degraded, not crashed). Assert the predicted
   label/proba path is byte-identical to a raw-pickle baseline for the legit case (no behavior drift).
6. **C adapter round-trip** (`importorskip("torch")`): `save()` writes weights + sidecar; `load()` restores
   and `eval()`-mode adapter produces the same forward output as before save; tampered weights file → refusal
   before `torch.load`; assert `weights_only=True` is used on torch ≥ 1.13 (introspect call or version-gate).
7. **Negative/FP-safety**: a legit artifact still loads on a keyless host (no `NA0S_PICKLE_KEY`) via SHA-256
   sidecar; assert no warning-to-error escalation breaks loading.
8. **Crafted-malicious fixture** (the "authored dataset"): a pickle whose `__reduce__` would write a sentinel
   file — assert that with the sidecar gate, `safe_load`/`verify_file_digest` raises BEFORE unpickling so the
   sentinel is never created (proves the gate runs pre-deserialization). Reuse/extend the malicious-pickle
   pattern from `tests/integrity/test_l11_safe_pickle_fixes.py` if present.

**Smoke step (CLI/suite, per checklist):**
9. `python3 -c "from na0s.integrity.safe_pickle import safe_load, safe_dump, verify_file_digest, write_digest_sidecar; print('ok')"`
   (import smoke — catches the hallucinated-symbol / import-blindness failure modes).
10. `python3 -m pytest tests/integrity/ tests/ml/ tests/worm/ -q --tb=line` then the full
    `python3 -m pytest tests/ -q --tb=line` (zero regressions, per CLAUDE.md).

---

## 6. Q&A self-check

- **Q1 — Can Na0S handle the target?** Not today (3 raw loaders = RCE on tamper). After the fix: yes —
  tampered/sidecar-less artifacts are refused pre-deserialize and the optional signal degrades; verified by
  tests 4–8 + full suite.
- **Q2 — Cleanup done?** Helper extraction de-dups digest logic; `# noqa: S301` removed; optional worm
  refactor to shared helper; canonical imports only (no shim writes). These 3 files remain pending the
  v1.0.0 `ml/` move (tracked separately in ROADMAP_V2 — out of scope here).
- **Q3 — Pipeline wiring correct?** Yes — hardening is inside `load`/`save`; reached transitively via
  `predict_embedding.py` (faiss line 51) under `_HAS_EMBEDDING_CLASSIFIER` (predict.py 195–204/897) and
  `cascade.py` line 100. No new flag; verify transitive import stays green.
- **Q4 — Tested for code AND use-case?** Yes — helpers (1–3) + per-loader round-trip/tamper/degrade (4–8)
  + FP-safety (7) + smoke (9–10).
- **Q5 — Harvester audit.** N/A — model-artifact bytes, not harvested intel.
- **Q6 — Taxonomy + Coverage Matrix.** N/A — integrity gate, not a taxonomy-coded attack class.
- **Q7 — Scorer.** N/A — no per-attack threshold; refusal removes an optional signal, scoring unchanged.
- **Q8 — predict.py / cascade.py references?** Indirect only (transitive via `predict_embedding.py`); no
  direct symbol change needed in predict/cascade — only a green-import verification.
- **Q9 — Harvester agent harvest this type?** N/A — not a harvestable intel type.
- **Q10 — Other correctness checks.** (i) torch-absent env must import cleanly (guard + importorskip).
  (ii) Keyless host must use SHA-256 sidecar. (iii) Refusal must never flip a benign verdict (FP-safe).
  (iv) Reconcile the L5 ROADMAP over-claim (Step 2/8). (v) Confirm no other raw `pickle.load`/`torch.load`
  in `ml/` slipped the net (grep gate in CI).

---

## 7. Agent / skill assignment (inject na0s-review-checklist into every subagent prompt)

| Step | Owner | Why |
|------|-------|-----|
| 1–2 explore + roadmap | `security-research-auditor` + skill `na0s-debugging` | map loaders + safe_pickle infra against MAIN (PYTHONPATH=<worktree>/src) |
| 3.1 helper refactor | `l3-l5-code-auditor` + skill `security-review` | extract `verify_file_digest`/`write_digest_sidecar` w/o behavior drift |
| 3.2–3.3 faiss/stacking (ml/, L4/L5) | `l3-l5-code-auditor` | these live in the L4/L5 ML path |
| 3.4 adapter + torch `weights_only` | `l3-l5-code-auditor` + skill `security-review` | torch-specific deserialization hardening |
| 3.5 wiring/parity verify | `layer-9-11-auditor` + skill `na0s-debugging` | integrity = L11 concern; confirm transitive predict/cascade path |
| degrade-on-refusal audit | `silent-failure-hunter` | ensure refusal degrades (not silent-passes a tampered file, not crashes scan) |
| 6 tests | `l3-l5-code-auditor` (assertion-rich, no hollow tests) | mirror tests/ml + tests/integrity |
| 8 roadmap | `Plan` | check off + cite SHA; fix L5 over-claim |
| 11 PR | `github-pr-prep` → `github-pr-review` (`pr-review-toolkit:*`) + skill `github-ci-fix` | PR prep, review, drive CI green |

N/A skills for this item: `eval-harness`, `data-harvesting`, `cron-scheduling`, `eval-scenario-curation`,
`incident-to-scenario` (no eval/harvest/cron/scenario surface in a load-time integrity gate).

---

## 8. Execution preconditions / dependencies
- **Depends-on: none.** Self-contained; `integrity.safe_pickle` already exists and is battle-tested.
- Work in a dedicated git worktree on branch `hardening/route-raw-loaders-safeload` off `main`
  (multi-agent worktree discipline; never branch-switch the primary checkout).
- Verify symbols against MAIN, not the stale editable install (`PYTHONPATH=<worktree>/src`).
- torch is optional and absent in CI/dev here — adapter tests must `importorskip("torch")`.
- Keyless: do NOT introduce any code path that REQUIRES `NA0S_PICKLE_KEY` (SHA-256 sidecar must suffice).

## 9. Definition of done
- [ ] `verify_file_digest` + `write_digest_sidecar` added to `na0s.integrity.safe_pickle`; `safe_load`/`safe_dump` refactored to reuse them; existing safe_pickle tests green.
- [ ] `faiss_classifier.save/load` use `safe_dump`/`safe_load` (labels); tamper ⇒ refusal ⇒ `_init_failed` degrade.
- [ ] `stacking_classifier.save/load` use `safe_dump`/`safe_load`; `# noqa: S301` removed; tamper ⇒ `is_available()` False.
- [ ] `embedding_adapter.save` writes a sidecar; `load` verifies digest BEFORE `torch.load` and uses `weights_only=True` (torch ≥ 1.13); tamper ⇒ refusal.
- [ ] Transitive predict.py/cascade.py embedding path imports + degrades cleanly (no new flag); smoke import passes.
- [ ] Crafted-malicious-pickle test proves the gate fires PRE-deserialize (sentinel never written).
- [ ] FP-safe: legit keyless load works; no benign verdict changes; no magic threshold added.
- [ ] `pytest tests/integrity tests/ml tests/worm` green, then full `pytest tests/ -q --tb=line` zero regressions.
- [ ] ROADMAP_V2 item checked off with commit SHA; L5 "safe_dump/safe_load" over-claim reconciled.
- [ ] PR opened; merge gated on green held-out/full suite; main-merge confirmed with user.
