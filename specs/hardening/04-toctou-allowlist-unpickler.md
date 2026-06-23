---
item: 4
title: "TOCTOU read-once-buffer + allowlist find_class Unpickler"
priority_tier: P1 (supply-chain integrity hardening; core loader)
depends_on:
  - "ci/test-optional-dep-guards (commit 91944d6) — must land/merge first; it introduces _NumpyCompatUnpickler, the hook this item extends. If it is NOT merged, this item must ADD the unpickler subclass from scratch."
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]   # + Q1, Q2, Q3, Q4, Q8, Q10
na_steps: [5, 9, 10]                           # + Q5, Q6, Q7, Q9
status: PLAN-ONLY (no source/test edits in this pass)
canonical_file: src/na0s/integrity/safe_pickle.py
shim: src/na0s/safe_pickle.py (do NOT edit — redirects to canonical)
---

# Item 4 — TOCTOU read-once-buffer + allowlist `find_class` Unpickler

## 0. KEY-REF reconciliation (line numbers verified on branch `hardening/rag-poison-wiring`)

Opened `src/na0s/integrity/safe_pickle.py` in full. The refs are accurate **except** the
`_NumpyCompatUnpickler` reference, which has drifted:

| Ref claim | Verified location (current branch) | Status |
|---|---|---|
| `:59` (`_sha256` opens the file to hash) | line 59 — `with open(path, "rb") as f:` inside `_sha256()` | OK |
| `:91` (`_hmac_sha256` opens the file to hash) | line 91 — `with open(path, "rb") as f:` inside `_hmac_sha256()` | OK |
| `:128-157` (`_validate_pickle_magic` opens the file to sniff 2 magic bytes) | lines 128–157 — `with open(path, "rb") as f: header = f.read(2)` at 134 | OK |
| `:295-363` (`safe_load`) | lines 295–363; the final unpickle is `with open(path, "rb") as f: return pickle.load(f)` at **362–363** | OK |
| `_NumpyCompatUnpickler (grep src/)` | **NOT PRESENT on this branch.** `git grep` for `NumpyCompat\|find_class\|Unpickler` over tracked `*.py` returns 0 hits. | CORRECTION |

**Correction detail.** `_NumpyCompatUnpickler` exists only on the unpushed branch
`ci/test-optional-dep-guards` at commit `91944d6` ("fix(integrity): load numpy-2-pickled
models on numpy<2 hosts"). On that branch it lives in `safe_pickle.py` directly after the
cache dicts (a `pickle.Unpickler` subclass overriding `find_class` to remap
`numpy._core.*` → `numpy.core.*` on numpy<2 hosts), and `safe_load` ends with
`return _NumpyCompatUnpickler(f).load()` instead of `pickle.load(f)`. That branch ALSO
adds (a) `pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)` in `safe_dump` and (b) an
sklearn `InconsistentVersionWarning` suppression block around the final load. **Neither
branch is merged to `main`.** This spec therefore plans for BOTH realities:

- **Path A (preferred):** `ci/test-optional-dep-guards` lands first → this item *extends*
  the existing `_NumpyCompatUnpickler.find_class` with an allowlist guard and converts the
  load to read-once-buffer.
- **Path B (fallback):** if that branch is abandoned → this item *introduces* the
  restricted unpickler subclass itself (keeping the numpy `_core` remap, since the shipped
  artifacts genuinely need it), plus the allowlist + read-once buffer.

## 1. Root cause (two coupled defects)

**Defect 1 — TOCTOU (CWE-367) over the pickle bytes.** `safe_load` opens `path` up to
**three separate times**: once in `_validate_pickle_magic` (line 134), once inside
`_cached_sha256`/`_cached_hmac_sha256` → `_sha256`/`_hmac_sha256` (lines 59 / 91) to compute
the integrity digest, and a **third** time at line 362 to feed `pickle.load`. The integrity
digest is computed over bytes read in open #2, but the bytes actually *executed* by the
unpickler are read in open #3. A local attacker with write access to the model directory
(or who wins a symlink/rename race) can swap the file *between* the verified hash and the
final `open()`, so a tampered/malicious pickle is unpickled even though the hash check
"passed". The verify-then-reopen pattern is the classic time-of-check/time-of-use window.
Worse, the mtime-gated cache (`_sha256_cache`/`_hmac_cache`, lines 48–49, 65–73, 97–105)
widens the window: a second `safe_load` of the same path can skip hashing entirely and trust
a cached digest while the on-disk bytes differ (mtime can be reset to the cached value with
`os.utime`).

**Defect 2 — unrestricted `find_class` (CWE-502 residual).** Even with a perfect hash, the
final `pickle.load` (or `_NumpyCompatUnpickler` on the dep-guard branch) calls the default
`Unpickler.find_class`, which can import and resolve **any** module/attribute the pickle
opcode stream names (e.g. `os.system`, `subprocess.Popen`, `builtins.eval`). The hash gate
is the *only* thing standing between a malicious `.pkl` and arbitrary code execution; if the
hash is ever bypassed (Defect 1, a sidecar-rewrite on the SHA-256 fallback tier, or a future
hash break) there is no second line of defense. Defense-in-depth requires the unpickler to
**allowlist** the globals the legitimate sklearn/numpy artifacts actually need and reject the
rest — this is the "Restricted unpickler" / PickleBall pattern already namechecked in
`ROADMAP_V2.md:2528` ("PickleBall (CCS 2025) — Restricted unpickler").

## 2. Gap vs ideal (what it IS vs what it SHOULD be)

| | Current | Ideal |
|---|---|---|
| Reads of the pickle bytes | 3 independent `open()` calls; hashed bytes ≠ executed bytes | **Read once** into an in-memory buffer; magic-check, hash, AND unpickle all operate on the SAME buffer |
| `find_class` policy | Default — resolves any global named in the stream | **Allowlist** of `(module, name)` pairs the bundled artifacts need; everything else → `UnpicklingError` |
| Cache correctness | mtime-gated; can be defeated by `os.utime` | Hashing the in-memory buffer makes the cache advisory only for the *check*, but the executed bytes are always the buffer (TOCTOU closed regardless of cache) |
| Failure mode on disallowed global | n/a (no policy) | Audit-logged `integrity_audit` event + `pickle.UnpicklingError` with the blocked `(module, name)` |
| numpy `_core` remap | only on dep-guard branch | preserved INSIDE the allowlisted `find_class` (remap first, then allowlist-check the remapped target) |

## 3. Root-cause implementation plan (numbered)

All edits are confined to **one canonical file**: `src/na0s/integrity/safe_pickle.py`.
The shim `src/na0s/safe_pickle.py` is NOT touched (it `import`s the canonical module).

1. **Read-once buffer.** Add a private `_read_file_bytes(path) -> bytes` that does a single
   `with open(path, "rb") as f: return f.read()`. Refactor `safe_load` to call it ONCE near
   the top and thread the resulting `data: bytes` through every downstream step.

2. **Buffer-based magic check.** Add `_validate_pickle_magic_bytes(data: bytes)` operating on
   `data[:2]` (reuse the exact opcode logic at lines 137–157). Keep the existing
   path-based `_validate_pickle_magic` as a thin wrapper (`open → read(2) → _..._bytes`) so
   the public/tested helper signature at `test_l11_safe_pickle_fixes.py` is preserved
   (`_validate_pickle_magic` is imported there — do NOT break it).

3. **Buffer-based hashing.** Add `_sha256_bytes(data)` and `_hmac_sha256_bytes(data, key)`
   that hash the in-memory buffer (same `hashlib`/`hmac` calls, no `open`). In `safe_load`,
   compute the digest from `data`, NOT from a fresh file read. Keep `_sha256`/`_hmac_sha256`
   (path-based) and the mtime caches intact for `safe_dump` and external callers — only
   `safe_load`'s verification path switches to the buffer. (The cache becomes a pure
   optimization that can no longer cause a TOCTOU mismatch because the executed bytes are the
   buffer, not a re-read file.)

4. **Buffer-based unpickle.** Replace the final `with open(path, "rb") as f: return
   pickle.load(f)` (lines 362–363) with `io.BytesIO(data)` fed to the restricted unpickler:
   `return _SafeUnpickler(io.BytesIO(data)).load()`. Add `import io`. The exact same bytes
   that were hashed are the bytes unpickled — TOCTOU window eliminated.

5. **Allowlist `find_class`.** Introduce `_SafeUnpickler(pickle.Unpickler)` (Path B) OR extend
   `_NumpyCompatUnpickler.find_class` (Path A). Logic, in order:
   a. Apply the numpy `_core` → `core` remap (preserve existing behavior).
   b. Check `(module, name)` (post-remap) against `_PICKLE_ALLOWLIST`.
   c. If allowed → `super().find_class(module, name)`.
   d. If denied → emit an `na0s.integrity_audit` `event: "find_class_blocked"` record and
      raise `pickle.UnpicklingError(f"blocked global {module}.{name}")`.

6. **Derive the allowlist EMPIRICALLY, not by guessing** (no hallucinated symbols). Build it
   by instrumenting a load of the 4 bundled artifacts (`model.pkl`, `tfidf_vectorizer.pkl`,
   `structural_scaler.pkl`, `char_tfidf_vectorizer.pkl` — paths from `predict.py:227-230`;
   `model_embedding.pkl` from `KNOWN_HASHES` `models/__init__.py:26-30`): subclass
   `pickle.Unpickler`, log every `(module, name)` `find_class` receives during a successful
   load, take the UNION across all artifacts under both numpy 1.26 and numpy 2.4, and freeze
   that set as `_PICKLE_ALLOWLIST` (a module-level `frozenset`). This guarantees zero
   false-rejects on legit artifacts. Document in a comment that the set was *measured* and
   how to regenerate it (a `scripts/` one-off or a test helper), so it is not an arbitrary
   magic list. Restrict by module-prefix where the symbol set is open-ended (e.g. allow
   `numpy.core.*` / `numpy._core.*` / `sklearn.*` / `scipy.*` modules but still deny
   `os`, `posix`, `nt`, `subprocess`, `builtins.eval`, `builtins.exec`, `builtins.__import__`,
   `importlib`, `sys` outright). The allowlist is a *prefix+exact* hybrid, justified by the
   measured global set.

7. **Audit + error parity.** Reuse the existing `_audit` logger (line 41) for the
   `find_class_blocked` event, matching the JSON shape of the other audit events
   (`event`, `path` if threadable, `module`, `name`). Raise `pickle.UnpicklingError`
   (a `ValueError` subclass) so existing `except (ValueError, ...)` guards in `predict.py`
   (the `try/except` around `_cached_*` loads, e.g. lines 295–307, 367–373, 399–405) keep
   degrading gracefully rather than crashing the pipeline.

8. **No new env flag / no threshold.** This is always-on hardening; there is no tunable
   number and no env gate. (If a kill-switch is ever wanted, gate it behind an env var that
   *defaults to ON* — but the plan is to ship it unconditionally, like the existing
   magic-byte check.)

## 4. Exact files / functions to change

- `src/na0s/integrity/safe_pickle.py` ONLY:
  - ADD `import io` (top, with the other stdlib imports, lines 27–36).
  - ADD `_read_file_bytes(path)`, `_validate_pickle_magic_bytes(data)`, `_sha256_bytes(data)`,
    `_hmac_sha256_bytes(data, key)`.
  - ADD `_PICKLE_ALLOWLIST` (frozenset) + the allowed module-prefix tuple + the hard-deny set.
  - ADD/EXTEND `_SafeUnpickler` (Path B) or `_NumpyCompatUnpickler.find_class` (Path A).
  - EDIT `safe_load` (295–363): single read-once → buffer-based magic → buffer-based hash →
    `io.BytesIO(data)` → restricted unpickler.
  - PRESERVE `_validate_pickle_magic`, `_sha256`, `_hmac_sha256`, `_cached_*`, `safe_dump`,
    all helper signatures (imported by tests) — wrap, don't replace.
- **No changes** to `predict.py`/`cascade.py` source: they call `safe_load` unchanged
  (Q8 below). The hardening is transparent to callers.

## 5. Pipeline wiring (Step 4 / Q3 / Q8)

`safe_load` is the single chokepoint. Wiring is already complete and does NOT need new
registration:
- `predict.py:83` `from .safe_pickle import safe_load` (via shim → `integrity.safe_pickle`);
  call sites `predict.py:306,307,371,403` load model/vectorizer/scaler/char-vectorizer.
- Other callers: `ml/predict_embedding.py`, `dataset/hard_negatives.py`,
  `layer15/benchmark_analyzer.py`, `layer15/atlas_sync.py`, `eval/harvest/taxonomy.py`,
  `eval/scenarios/loader.py`, `models/__init__.py`.
- `cascade.py` reaches the model load through `predict.py`'s cached loaders (no direct
  `safe_load`). **Parity is automatic** because both predict and cascade go through the same
  `safe_load`. No detector-style `_HAS_*` flag or dual-registration is needed — this is a
  loader hardening, not a new detection signal.

## 6. Test plan — Code + Use-Case (Step 6 / Q4)

New file: `tests/integrity/test_safe_pickle_toctou_allowlist.py` (mirrors the source package
per CLAUDE.md; sibling of the existing `tests/integrity/test_safe_pickle.py`). Use the
established `tempfile.TemporaryDirectory` + `patch.dict(os.environ, {"NA0S_PICKLE_KEY": ...})`
fixtures from the existing tests. NO hollow assertions — every test asserts an observable
outcome (return value, raised type+message, or audit-log record).

**A. TOCTOU read-once (Code):**
1. `test_toctou_swap_between_hash_and_load_blocked` — `safe_dump` a benign object; monkeypatch
   `open` (or `_read_file_bytes`) so the SECOND read returns malicious bytes, prove the
   loaded object equals the FIRST (verified) bytes OR raises — i.e. the executed bytes are the
   hashed bytes. Concretely: patch `builtins.open` to return original bytes on the hash path
   and a malicious pickle on a later open; with the read-once fix there IS no later open, so
   the test asserts the malicious payload is never returned. (Pre-fix this test FAILS; that is
   the regression guard.)
2. `test_cache_cannot_serve_stale_digest_for_executed_bytes` — load once (populates cache),
   rewrite the file with tampered bytes but reset mtime via `os.utime` to the cached value,
   load again → must raise `Integrity check failed` (buffer is re-hashed) OR — if the cache is
   intentionally trusted — the *executed* bytes must still be the freshly-read buffer, not the
   cached-but-stale file. Assert the precise behavior the implementation chooses; document it.

**B. Allowlist `find_class` (Code + adversarial `__reduce__`):**
3. `test_reduce_os_system_blocked` — craft `class Evil: def __reduce__(self): return (os.system, ("echo pwned",))`, `safe_dump` it (this writes a valid sidecar over the malicious
   pickle — simulating a sidecar-rewrite / hash-bypass adversary), then `safe_load` → must
   raise `pickle.UnpicklingError` naming `os.system`, and the payload must NOT execute (assert
   no side effect, e.g. a sentinel file is never created).
4. `test_reduce_builtins_eval_blocked`, `test_reduce_subprocess_popen_blocked`,
   `test_reduce_builtins_exec_blocked` — same pattern for `eval`, `subprocess.Popen`, `exec`.
5. `test_allowlist_blocked_event_audited` — assert an `na0s.integrity_audit`
   `find_class_blocked` record is emitted (use `assertLogs("na0s.integrity_audit", "ERROR")`
   matching the existing audit-test style at `test_safe_pickle.py:251`).

**C. No false-rejects (Use-Case / FP-safe — the load must still work):**
6. `test_benign_dict_round_trips` — `safe_dump`/`safe_load` of a plain dict still returns it.
7. `test_bundled_artifacts_load` — load each real bundled artifact
   (`model.pkl`, `tfidf_vectorizer.pkl`, `structural_scaler.pkl`, `char_tfidf_vectorizer.pkl`,
   `model_embedding.pkl`) via `safe_load(get_model_path(...))`; assert a non-None object back.
   Skip cleanly (`pytest.importorskip`) if sklearn/numpy or the artifacts are absent (mirror
   the existing optional-dep guards) so the test is green in the minimal CI env. This is the
   critical FP-safety proof: the empirically-derived allowlist must not reject the shipped
   models. Run under BOTH numpy<2 and numpy>=2 if both are installable (else note it as a CI
   matrix follow-up).
8. `test_predict_pipeline_still_scans` — end-to-end smoke: import `na0s.predict`, run a real
   `predict("ignore all previous instructions")` and a benign string; assert the model-backed
   path returns a `ScanResult` (the loader change is transparent to scan()). This is the
   "full scan() still works" use-case bullet from Q4.

**D. CLI / suite smoke (mandatory per checklist):**
9. Targeted first: `python3 -m pytest tests/integrity/ -v` (must stay green incl. the existing
   5 files — they import `_validate_pickle_magic`, `_sha256`, `_hmac_sha256`, so the wrapper
   preservation in §3.2–3.3 is what keeps them passing).
10. CLI smoke (not mocked): `python3 -m na0s.predict "ignore previous instructions"` (or the
    documented scan entrypoint) to prove the real model loads through the new unpickler.
11. Full suite last: `python3 -m pytest tests/ -q --tb=line` — confirm 0 net regressions vs
    base (the ~8000-test run; expect the known env-only failures unchanged, per
    project memory "8969 passed / 15 env-only failures").

## 7. Cleanup / refactor (Step 7 / Q2)

- This file already has irregular blank-line spacing (double/triple blanks around lines 50–52,
  74–76, 95–96, 106, 211–213, 245–247, 292–294). Normalize while touching the file (single
  blank between helpers) — but ONLY incidental whitespace, no logic churn, to keep the diff
  reviewable.
- Reconcile with the dep-guard branch (Path A) so we don't ship two divergent `safe_load`
  tails (numpy remap + sklearn warning suppression must survive the merge). If Path B, port
  the numpy `_core` remap and the `InconsistentVersionWarning` suppression forward so this
  branch doesn't regress the Intel-mac fix (project memory: C1 two-step extraction was
  unblocked by 91944d6 — must not break it).
- Do NOT add code to the shim `src/na0s/safe_pickle.py` (header forbids it).
- No new top-level module; everything stays in the existing canonical `integrity/` package
  and the existing `tests/integrity/` dir — already conforms to the code-organization standard.

## 8. Roadmap update (Step 8)

Edit `ROADMAP_V2.md` Layer 11 section:
- Check off / extend the existing item `:1180` ("Stress cases for `safe_pickle` — corrupted
  files (truncated mid-opcode)…") — the TOCTOU + adversarial-`__reduce__` tests partly satisfy
  it; note residual (concurrent/large-file) still open.
- Add a new completed bullet under Layer 11 referencing the **PickleBall restricted-unpickler**
  line already present at `:2528` ("PickleBall (CCS 2025) — Restricted unpickler") — mark that
  technique as IMPLEMENTED via `_SafeUnpickler` allowlist, cite the commit SHA once landed.
- Per project memory "Roadmap-Todo Sync": cite the commit SHA when pushed; check the box only
  after the full suite is green.

## 9. README / Benchmark (Step 9)

N/A for benchmark/recall numbers (no detection-recall change). README/CHANGELOG: add one line
under the L11 supply-chain description noting read-once-buffer + restricted-unpickler
allowlist (the existing L11 prose at `ROADMAP_V2.md:1107,1172` is the model wording). No
COVERAGE_MATRIX or recall-harness change. Mark the README touch optional/low-priority.

## 10. N/A steps & Q&A (honest justifications)

- **Step 5 — N/A** — Harvested datasets: there is no external dataset for this; the "dataset"
  is *crafted* malicious pickles authored in the test file (adversarial `__reduce__`), per the
  item-8-style exception. No harvester involvement.
- **Step 9 (benchmark portion) — N/A** — Loader hardening changes no detector recall/FPR; the
  two-sided recall harness and benchmark.py are untouched.
- **Step 10 — N/A** — Taxonomy codes & COVERAGE_MATRIX classify *prompt-injection attack
  classes*; an unpickler RCE is a supply-chain/CWE-502 concern with no `data/taxonomy.yaml`
  code and no coverage-matrix row. No per-attack scorer threshold applies.
- **Q1 — APPLICABLE** — "Can Na0S handle it?": currently NO (TOCTOU window + unrestricted
  find_class). Fix per §3; prove with §6 tests + full suite green.
- **Q2 — APPLICABLE** — cleanup per §7 (whitespace + branch reconciliation; no merged-branch
  clutter introduced).
- **Q3 — APPLICABLE** — wiring is the single `safe_load` chokepoint; predict/cascade parity
  automatic (§5).
- **Q4 — APPLICABLE** — tested for Code (§6 A/B) AND use-case (§6 C: tampered rejected, legit
  loads, full scan() works).
- **Q5 — N/A** — no harvested dataset; crafted pickles only.
- **Q6 — N/A** — not a taxonomy/coverage-matrix concern (see Step 10).
- **Q7 — N/A** — no scorer scores a pickle RCE; binary reject/accept, not a risk score.
- **Q8 — APPLICABLE / YES** — `predict.py` references `safe_load` (`:83` import; `:306,307,
  371,403` calls). It loads through the shim → canonical. No `predict.py` edit needed.
  `cascade.py` has no direct `safe_load`; it loads via `predict.py`'s cached loaders →
  inherits the hardening.
- **Q9 — N/A** — the harvester agent does not harvest pickle-RCE samples; out of scope.
- **Q10 — APPLICABLE** — extra checks: (a) ensure `pickle.UnpicklingError` (ValueError
  subclass) keeps `predict.py`'s `except` guards degrading gracefully; (b) verify the
  empirical allowlist under BOTH numpy 1.26 and 2.4 so the numpy-2-pickled artifacts load on a
  numpy<2 host (the dep-guard concern); (c) confirm the mtime cache cannot reintroduce TOCTOU
  after the buffer switch.

## 11. Agent / skill team per step (inject `na0s-review-checklist` into EACH)

| Step / area | Agent / skill | Checklist injection focus |
|---|---|---|
| Plan authoring | `Plan` + skill `security-review` | hallucinated-API + arbitrary-threshold + import-blindness sections |
| Explore + root cause (1–3) | `security-research-auditor`, skill `na0s-debugging` | verify-against-MAIN not stale editable-install; cite file:line |
| Allowlist derivation (§3.6) | `l3-l5-code-auditor` (model-load path) | "no magic list" — allowlist must be *measured* from real artifacts |
| Implement loader (4) | `silent-failure-hunter` (swallowed `UnpicklingError`?) + `layer-9-11-auditor` (L11 owner) | silent-refactor-destruction + hollow-test sections |
| Tests (6) | `silent-failure-hunter` + skill `na0s-debugging` | hollow-tests + mocked-CLI-smoke-gap sections |
| Eval/suite gate (6 D) | skill `eval-harness` (confirm no recall delta), `github-ci-fix` | full-suite-green; env-only-failure parity |
| Cleanup/refactor (7) | `l3-l5-code-auditor` | whitespace-only diff discipline; don't touch shim |
| PR (11) | skill `github-pr-prep` then `pr-review-toolkit:review-pr` | secret-scan + risk; require held-out tests green |

## Execution preconditions / dependencies

1. **`ci/test-optional-dep-guards` (commit `91944d6`) SHOULD land first** (Path A). It
   introduces `_NumpyCompatUnpickler` + the sklearn-warning suppression + `HIGHEST_PROTOCOL`
   dump that this item extends. If it is dropped, switch to Path B (this item adds the
   restricted unpickler from scratch AND ports the numpy `_core` remap forward so the
   Intel-mac/numpy<2 load is not regressed).
2. Work in a dedicated git worktree off `main` (or off the dep-guard branch if Path A), per
   the multi-agent worktree discipline — never branch-switch the primary checkout.
3. Verify imports against the renamed tree using `PYTHONPATH=<worktree>/src` (the editable
   install points at the d8 checkout; `na0s.integrity.safe_pickle` must resolve there).
4. Bundled artifacts present locally for the FP-safety test (§6 C7); else `importorskip` and
   note the CI-matrix follow-up.
5. No API key required at any step (loader hardening is fully local/keyless).

## Definition of done

- [ ] `safe_load` reads the pickle bytes EXACTLY ONCE; magic-check, hash, and unpickle all
      operate on that one buffer (TOCTOU window closed; verified by §6 A1).
- [ ] Restricted `find_class` allowlist rejects `os.system`/`eval`/`exec`/`subprocess.Popen`
      and any non-allowlisted global with `pickle.UnpicklingError` + an audited
      `find_class_blocked` event (§6 B).
- [ ] Allowlist is EMPIRICALLY derived from the 5 bundled artifacts under numpy 1.26 AND 2.4;
      zero false-rejects (§6 C7 passes; documented regeneration recipe, no magic list).
- [ ] All 5 existing `tests/integrity/*.py` files still pass (helper signatures preserved).
- [ ] New `tests/integrity/test_safe_pickle_toctou_allowlist.py` added; all new tests pass and
      each asserts an observable outcome (no hollow tests).
- [ ] `predict.py`/`cascade.py` unchanged; full `predict()` smoke + CLI smoke still load the
      real model through the new unpickler.
- [ ] numpy `_core` remap + sklearn `InconsistentVersionWarning` suppression preserved (no
      regression of commit 91944d6 / the C1 Intel-mac unblock).
- [ ] `python3 -m pytest tests/ -q --tb=line` shows 0 net regressions vs base (known env-only
      failures unchanged).
- [ ] `ROADMAP_V2.md` L11 items `:1180` / PickleBall `:2528` checked/updated with the commit
      SHA; CHANGELOG line added.
- [ ] PR opened off the correct base; merge gated on held-out/full-suite green; no force-push
      / no merge-to-main without confirmation.
