---
item: 8
title: "L11 tests: adversarial __reduce__ rejection + KNOWN_HASHES precedence + stress"
priority_tier: P2 (test-coverage hardening; closes an RCE-shaped behavioral gap once #4 lands)
class: TEST item — supply-chain / integrity (NOT a prompt-injection attack class)
depends_on: [4]          # #4 introduces the RestrictedUnpickler/find_class allowlist that #8 asserts
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]
na_steps: [5, 9, 10]
applicable_qs: [Q1, Q2, Q3, Q4, Q8, Q10]
na_qs: [Q5, Q6, Q7, Q9]
dataset_note: "The 'dataset' = a small set of CRAFTED malicious pickles authored in-test (not harvested). Step5/Q5 reframed accordingly — still N/A as a HARVESTER concern."
---

# Item 8 — L11 tests: adversarial `__reduce__` rejection + `KNOWN_HASHES` precedence + stress

## 0. Root cause (one sentence)
`safe_load()` verifies a file's integrity digest but then deserializes with **bare `pickle.load(f)`** (`src/na0s/integrity/safe_pickle.py:363`), so a pickle that PASSES integrity (e.g. attacker holds the key, or owns a `.sha256` sidecar, or KNOWN_HASHES is stale) still runs arbitrary `__reduce__` on load; there is **no `find_class` allowlist** anywhere in the tree and **no test** that (a) rejects a `__reduce__` RCE payload, (b) proves the `hardcoded > HMAC > SHA-256` precedence in `_resolve_expected_hash`, or (c) stress-exercises truncated/large/concurrent files — this item adds those tests (the rejection mechanism itself lands in item #4).

---

## 1. KEY REFS — confirmed current line numbers (verified against the working tree, corrections noted)

| Ref | File | What | Confirmed line | Status |
|-----|------|------|----------------|--------|
| Test file 1 | `tests/integrity/test_safe_pickle.py` | HMAC/SHA-256 round-trip, tamper, replace-both, backward-compat | 260 lines total; helper imports at **21–29**; `TestTamperingDetection` **116–199**; `TestBackwardCompatibility` **202–255** | ✅ exists |
| Test file 2 | `tests/integrity/test_l11_safe_pickle_fixes.py` | BUG-L11-2..6 (atomic write, versioning, audit log, perms, magic bytes) | 313 lines; imports **24–35**; `TestPickleMagicValidation` **263–313** | ✅ exists |
| Roadmap | `ROADMAP_V2.md:1180` | "Stress cases for `safe_pickle` — truncated mid-opcode, >1 GB, concurrent dump" P2/Low | confirmed at **1180** (under "Test coverage gaps") | ✅ ref accurate |
| Source under test | `src/na0s/integrity/safe_pickle.py` | canonical module | `safe_load` **295–363**; bare `pickle.load(f)` at **363**; `_resolve_expected_hash` **214–244**; `_validate_pickle_magic` **128–157** | ✅ |
| Shim | `src/na0s/safe_pickle.py` | deprecation shim → `na0s.integrity.safe_pickle` | 1–14 | tests import via this shim today |
| KNOWN_HASHES | `src/na0s/models/__init__.py:26` | hardcoded SHA-256 dict (4 entries: model/structural_scaler/model_embedding/tfidf) | **26–31** | ✅ |
| Pipeline wiring | `src/na0s/predict.py:83` | `from .safe_pickle import safe_load`; call sites **306, 307, 371, 403** | confirmed | ✅ (Q8) |

**Corrections vs. the brief:** none on line numbers. Two facts the brief implies but the tree contradicts — call out in the plan:
- **No `_NumpyCompatUnpickler` on this branch.** Project memory mentions it, but it lives on `ci/test-optional-dep-guards` (commit `91944d6`), not here. `safe_load` here ends in plain `pickle.load`. Do not assume a custom unpickler exists.
- **No `find_class` / `Unpickler` / `RestrictedUnpickler` symbol anywhere in `src/`** (grep-verified, returns empty). The `__reduce__`-rejection behavior this item tests **does not exist yet** — it is delivered by item #4. Item #8's `__reduce__` tests are therefore RED until #4 merges (see Execution preconditions).

---

## 2. Gap vs. ideal (what L11 tests should cover vs. what they do)

| Behavior | Today | Ideal (this item) |
|----------|-------|-------------------|
| Integrity tamper detection | ✅ covered (`test_tampered_pickle_detected_*`, `test_replace_both_attack_blocked`) | keep; add the missing layer below |
| **Adversarial `__reduce__` rejection** | ❌ **no test** — and the code path does not reject it (bare `pickle.load`). A pickle whose `__reduce__` runs `os.system(...)` but whose digest is VALID (attacker-controlled sidecar / known-key / stale KNOWN_HASHES) executes on load | A crafted malicious pickle with a `__reduce__` that would run code must be **refused before instantiating the dangerous object** (item #4's `find_class` allowlist raises `pickle.UnpicklingError`/`ValueError`), and a module-level **sentinel must prove the side-effect never ran** |
| **`KNOWN_HASHES` precedence** | ❌ **no test** — `_resolve_expected_hash` returns `hardcoded` over `.hmac` over `.sha256` (214–244) but precedence is unproven | Tests that (i) hardcoded hash is used even when a (forged) `.hmac`/`.sha256` sidecar also exists; (ii) HMAC chosen over SHA-256 when both sidecars present + no hardcoded entry; (iii) a basename in KNOWN_HASHES whose bytes don't match the hardcoded digest ⇒ `ValueError` even if a matching sidecar exists |
| **Stress / robustness** (ROADMAP:1180) | ❌ **no test** | (i) truncated-mid-opcode pickle ⇒ `ValueError` (magic-byte or unpickle path), no traceback leak, no partial object; (ii) large-file path exercises chunked `_sha256`/`_hmac_sha256` (1<<16 chunking, lines 60/92) — use a few MB, NOT 1 GB (see threshold note); (iii) concurrent `safe_dump` to the same path leaves a consistent file + sidecar (atomic `os.replace`, lines 160–175) and no `.tmp` residue |

### Threshold / magic-number discipline (na0s-review-checklist)
- The roadmap says ">1 GB". A 1 GB temp file is an unacceptable CI cost and would make the suite flaky/slow. **Justified substitution:** use a modest size (e.g. **8 MB**, ≈ 128 × the 65 536-byte chunk) — large enough to force **multiple `iter(lambda: f.read(1<<16), b"")` iterations** in `_sha256`/`_hmac_sha256`, which is the only behavior the chunking guards. Document in the test docstring that the size is a chunking-coverage proxy, not a DoS limit. No new production threshold is introduced.
- "Concurrent" = a bounded set of processes/threads (e.g. **8**) racing one path; assert the *final* state is internally consistent (digest of file == sidecar digest) and no leftover `.tmp` — not a specific winner. No timing-based assertion (avoid flakiness).

---

## 3. Root-cause implementation plan (numbered, item-specific)

> This is a TEST item. Production behavior (the `find_class` allowlist) is **item #4**. Step 3 here is the test design; do not add the unpickler in this PR unless #4 is being co-landed.

1. **New isolated test file** `tests/integrity/test_l11_adversarial_stress.py` (mirrors source pkg per CLAUDE.md test-org rule; matches existing `tests/integrity/test_l11_*` naming). Import from the canonical module `na0s.integrity.safe_pickle` (not the shim) for new code — but add **one** smoke test asserting the shim re-exports the same `safe_load` object the pipeline uses (`predict.py:83` imports via the shim), so wiring stays honest.
2. **Crafted-malicious-pickle "dataset" (authored, in-test).** Define a `class _Exploit` with a `__reduce__` returning `(os.system, ("<sentinel side-effect>",))` and, separately, one returning `(builtins.eval, ("__import__('os')...",))`. Pickle each into a tmp file, then write a **valid** integrity sidecar for it (via `safe_dump` with a test key, OR by writing a matching `.sha256` — both make the digest pass) so the test exercises the case where **integrity passes but the payload is still hostile**. Use a module-level mutable sentinel (e.g. a list the fake callable appends to via `monkeypatch` on a stub, NOT a real `os.system`) so no real command runs — assert the sentinel stays empty.
   - **`TestAdversarialReduceRejection`**: with #4's allowlist in place, `safe_load(path)` raises (`pickle.UnpicklingError`/`ValueError`, message contains the disallowed global) AND the sentinel is never touched (proves rejection is PRE-execution).
   - Negative control: a benign pickle of a dict/list still `safe_load`s successfully under the same allowlist (FP-safe — the allowlist must not break legitimate model loads; cross-check against the 4 real bundled artifacts' types).
3. **`TestKnownHashesPrecedence`** (drives `_resolve_expected_hash`, 214–244):
   - Patch `KNOWN_HASHES` (via `unittest.mock.patch.dict("na0s.integrity.safe_pickle.KNOWN_HASHES", {basename: <real digest>}, clear=False)`) for a tmp file whose basename matches; create ALSO a forged `.hmac` and `.sha256` sidecar → assert `_resolve_expected_hash` returns `("...","hardcoded")` and `safe_load` succeeds (hardcoded wins).
   - No hardcoded entry, BOTH `.hmac` + `.sha256` present (HMAC valid) → source is `sidecar_hmac` (HMAC preferred, lines 224–229).
   - Hardcoded entry present but file bytes mutated so they don't match the hardcoded digest, while a *matching* sidecar is written → `safe_load` still raises `ValueError("Integrity check failed")` (hardcoded is authoritative; sidecar cannot rescue). This is the security-critical precedence assertion.
4. **`TestStressRobustness`**:
   - Truncated-mid-opcode: dump a valid pickle, truncate to N bytes mid-stream, re-sign so the digest matches the truncated bytes → `safe_load` raises `ValueError` (either magic-byte path 128–157 if the head is mangled, or the `pickle.load` path) without leaking a stack trace into the return value and without returning a partial object.
   - Chunking coverage: `safe_dump` an ~8 MB object, `safe_load` round-trips equal; additionally call `_sha256`/`_hmac_sha256` directly on the file to assert the chunked loop produces the same digest as `hashlib.sha256(open(...).read())` (proves the `1<<16` loop is correct, not just exercised).
   - Concurrency: `multiprocessing`/`ThreadPool` of 8 `safe_dump(obj, same_path)`; after join, assert (a) file unpickles to `obj`, (b) sidecar digest == digest of the file on disk, (c) no `*.tmp` left in the dir (atomic-write invariant, lines 160–175). Skip-guard on platforms where multiprocessing start-method is problematic; prefer threads to stay deterministic in CI.
5. **Assertion-richness (anti-hollow-test, na0s-review-checklist §hollow tests):** every test asserts a concrete observable — exception type + message substring, the sentinel-untouched invariant, the `(digest, source)` tuple, byte-equality of digests, or filesystem state. No bare `safe_load(...)` calls without an assertion; no `assert True`.
6. **CLI / suite smoke (na0s-review-checklist §mocked-CLI gap):** after authoring, run `python3 -m pytest tests/integrity/test_l11_adversarial_stress.py -v`, then the integrity dir `tests/integrity/ -q`, then the full suite `python3 -m pytest tests/ -q --tb=line` for zero regressions. Add a one-line import smoke (`python3 -c "from na0s.integrity.safe_pickle import safe_load, safe_dump"`).

---

## 4. Exact files / functions to change

| Action | Path | Detail |
|--------|------|--------|
| **CREATE** | `tests/integrity/test_l11_adversarial_stress.py` | the 4 test classes above (`TestAdversarialReduceRejection`, `TestKnownHashesPrecedence`, `TestStressRobustness`, plus a `TestShimParity` smoke) |
| READ-only (assert against) | `src/na0s/integrity/safe_pickle.py` | `safe_load` 295–363, `_resolve_expected_hash` 214–244, `_validate_pickle_magic` 128–157, `_sha256`/`_hmac_sha256` 57–94, atomic writes 160–180 |
| READ-only (patch target) | `src/na0s/models/__init__.py:26` | `KNOWN_HASHES` — patched via `patch.dict` on the *imported* name in `safe_pickle`, not the source dict |
| **NO SOURCE EDIT** | — | the `find_class` allowlist is item #4; this PR adds tests only. If #4 already merged, `safe_pickle.py`'s new `RestrictedUnpickler` symbol must be verified to exist before asserting on it (no hallucinated API). |

---

## 5. HARVESTER AUDIT
**N/A (reframed) — the "dataset" is CRAFTED, not HARVESTED.** The adversarial pickles are authored inline in the test file (a `__reduce__`-bearing class + a benign control). There is no external corpus to harvest, no taxonomy code to tag, and no decontamination step — a malicious pickle is a binary artifact, not a prompt-injection text sample the F14 harvester ingests. **Do NOT** route this through `data-harvesting` / `eval-scenario-curation`. The only "fine-tune" needed is keeping the crafted payloads in lockstep with item #4's allowlist (covered in Step 6 test plan, not a harvester change).

---

## 6. Test plan — Code + Use-case (Step 6 / Q4)

**Code-level (does each helper behave correctly):**
- `_resolve_expected_hash` precedence tuple under all three source branches (Step 3.3).
- `_sha256`/`_hmac_sha256` chunked-loop correctness vs. one-shot digest (Step 4 chunking).
- atomic-write invariant under concurrency (Step 4).
- (post-#4) `RestrictedUnpickler.find_class` rejects disallowed globals, allows the model classes.

**Use-case / behavioral (end-to-end of the integrity loader — Step 6 reframe):**
- A tampered-but-digest-valid malicious pickle is **refused before its `__reduce__` runs** (sentinel proves no side-effect) — the real RCE scenario.
- A legitimate keyless `.sha256` model still loads (FP-safe; does not break the live `predict.py` load path at 306/307/371/403).
- A stale-KNOWN_HASHES file (bytes ≠ hardcoded digest) is refused even with a forged matching sidecar.
- Full `pytest tests/ -q --tb=line` stays green (Q1: the suite stays green; the new tests pass once #4 lands).

---

## 7. Cleanup / refactor (Step 7 / Q2)
- New file lands in `tests/integrity/` (correct mirror) — no top-level test dump. ✅ convention-compliant.
- Reuse the existing fixture style from `test_l11_safe_pickle_fixes.py` (`tmp_dir`, `pkl_path`, `sample_obj` pytest fixtures, lines 42–55) rather than re-inventing `setUp/tearDown` — keep the dir consistent.
- Do NOT add code to the shim `src/na0s/safe_pickle.py` (CLAUDE.md). New imports use `na0s.integrity.safe_pickle`; one parity test asserts the shim still re-exports the same object.
- If item #4 left any dead helper or TODO in `safe_pickle.py`, flag it for #4's PR — do not refactor production code in this test-only PR.

---

## 8. Roadmap / README / Benchmark updates (Step 8 / Step 9)
- **ROADMAP_V2.md:1180** — check off "Stress cases for `safe_pickle`" once merged; cite the commit SHA. Append a one-line note that `__reduce__`-rejection coverage + KNOWN_HASHES-precedence coverage were added alongside the stress cases, and that the rejection *mechanism* shipped in item #4 (#4's SHA).
- Update the "Completed (24 items)" prose only if the count changes; otherwise leave (test-only).
- **README / Benchmark: N/A** — no metric, no public API, no detector recall changes (Step 9 not triggered).

---

## 9. Q&A self-check

- **Q1 — Can Na0S handle this?** Two parts: (a) *bug/threat handling* — only after #4 adds the allowlist; this item proves it with tests and proves the precedence + stress invariants the current code already provides. (b) *suite green* — full `pytest tests/` must stay green (the only acceptable diff is the new tests passing).
- **Q2 — Cleanup done?** Yes — single new file in the correct mirror dir, fixtures reused, no shim edits, no source refactor in a test PR.
- **Q3 — Pipeline wiring done correctly?** Verified by the `TestShimParity` smoke: the object `predict.py:83` imports (`na0s.safe_pickle.safe_load`) is the same `na0s.integrity.safe_pickle.safe_load` the tests drive — so a passing test reflects the live load path.
- **Q4 — Tested for code AND use-case?** Yes — Step 6 splits helper-level from end-to-end RCE-refusal behavior.
- **Q5 — HARVESTER AUDIT.** N/A — crafted-in-test malicious pickles, not a harvestable text corpus (see Step 5).
- **Q6 — Taxonomy + Coverage Matrix.** N/A — integrity/RCE load-time gate, not a taxonomy-coded prompt-injection attack class; no COVERAGE_MATRIX row.
- **Q7 — Scorer.** N/A — `safe_load` is binary refuse/allow; there is no per-attack score or threshold to tune (the only number, ~8 MB, is a chunking-coverage proxy, justified in §2).
- **Q8 — predict.py / cascade.py references?** **predict.py: YES** — imports `safe_load` at line 83, calls it at 306/307/371/403; the parity smoke covers it. **cascade.py: NO** direct reference (grep-empty); the ML load path reaches cascade only transitively via predict, so no cascade-side test is needed. No source change to either file in this PR.
- **Q9 — Harvester agent harvest this type?** N/A — same as Q5; not a harvestable intel type.
- **Q10 — Other correctness checks.** (i) No hallucinated API — `RestrictedUnpickler`/`find_class` must be confirmed present (from #4) before any test imports it; if #4 is not yet merged, those tests are `@pytest.mark.skipif`-guarded on the symbol's existence so this PR never lands red. (ii) Keyless host: SHA-256-sidecar tests must run without `NA0S_PICKLE_KEY` (no key dependency — project has no API key, and integrity must not require the HMAC key). (iii) FP-safe: the allowlist negative-control proves benign model loads still succeed. (iv) No real `os.system`/`eval` ever executes — payload callables are stubbed/monkeypatched to a sentinel. (v) Concurrency test uses threads + skip-guards to avoid CI flakiness.

---

## 10. Agent / skill assignment (inject `na0s-review-checklist` into every subagent prompt)

| Step | Owner | Why |
|------|-------|-----|
| 1–2 explore + roadmap | `layer-9-11-auditor` + skill `na0s-debugging` | integrity = L11; map `safe_pickle` against MAIN (`PYTHONPATH=<worktree>/src`), confirm #4's symbol surface |
| 3.2 craft `__reduce__` payloads + rejection design | `security-research-auditor` + skill `security-review` | adversarial pickle authoring, RCE-refusal semantics, sentinel-untouched proof |
| 3.3 KNOWN_HASHES precedence | `layer-9-11-auditor` | `_resolve_expected_hash` trust-hierarchy assertions |
| 3.4 stress/concurrency/chunking | `l3-l5-code-auditor` | chunked-hash + atomic-write invariants live in the ML-load fast path |
| anti-hollow / silent-pass audit | `silent-failure-hunter` | ensure no `safe_load` call lacks an assertion; ensure rejection isn't silently swallowed |
| 6 test authoring | `l3-l5-code-auditor` (assertion-rich) | mirror `tests/integrity` fixtures; no hollow tests |
| 8 roadmap | `Plan` | check off ROADMAP:1180 + cite SHA (and #4's SHA for the mechanism) |
| 11 PR | `github-pr-prep` → `pr-review-toolkit:review-pr` / `github-pr-review` + skill `github-ci-fix` | prep, review, drive CI green; merge gated on full suite |

**N/A skills:** `data-harvesting`, `eval-harness`, `eval-scenario-curation`, `incident-to-scenario`, `intel-harvest`, `cron-scheduling`, `detector-authoring`, `llm-judge`, `detector-failure-analysis` — no harvest/eval/cron/detector/judge surface in a load-time integrity test.

---

## 11. Execution preconditions / dependencies
- **Depends-on: item #4** (the `find_class`/`RestrictedUnpickler` allowlist that gives `safe_load` its `__reduce__`-rejection behavior). **Without #4, the `TestAdversarialReduceRejection` class has nothing to assert against** — it must be `skipif`-guarded on the symbol's existence so this PR is never red, and flipped to a hard assertion once #4 merges. The KNOWN_HASHES-precedence and stress classes are **independent of #4** and can land first.
- Work in a dedicated git worktree on branch `hardening/l11-adversarial-stress-tests` off `main` (multi-agent worktree discipline; never branch-switch the primary checkout).
- Verify every symbol against MAIN, not the stale editable install (`PYTHONPATH=<worktree>/src`); `_NumpyCompatUnpickler` is NOT on this branch — do not import it.
- Keyless: SHA-256-sidecar tests must pass with `NA0S_PICKLE_KEY` unset; never add a test that REQUIRES the HMAC key.
- No network, no real subprocess: all payloads stubbed to a sentinel; CI-safe sizes only (~8 MB max, no 1 GB file).

## 12. Definition of done
- [ ] `tests/integrity/test_l11_adversarial_stress.py` created with `TestAdversarialReduceRejection`, `TestKnownHashesPrecedence`, `TestStressRobustness`, `TestShimParity`.
- [ ] `__reduce__`-RCE pickle is refused PRE-execution; module-level sentinel proves no side-effect ran (guarded on #4's symbol; hard-asserted once #4 merges).
- [ ] Benign-pickle negative control loads successfully under the allowlist (FP-safe; does not break the 4 bundled-model types).
- [ ] KNOWN_HASHES precedence proven: hardcoded > HMAC > SHA-256, and stale-hardcoded-vs-matching-sidecar still raises `ValueError`.
- [ ] Stress: truncated-mid-opcode raises cleanly (no partial object, no traceback leak); ~8 MB round-trip exercises chunked digest with byte-equality assertion; 8-way concurrent `safe_dump` leaves a consistent file+sidecar and zero `.tmp` residue.
- [ ] Every test has a concrete assertion (no hollow tests); no real `os.system`/`eval` runs; no magic threshold beyond the justified ~8 MB chunking proxy.
- [ ] Shim parity smoke confirms `predict.py`'s `safe_load` is the canonical object.
- [ ] `pytest tests/integrity/test_l11_adversarial_stress.py -v` green; `pytest tests/integrity -q` green; full `pytest tests/ -q --tb=line` zero regressions.
- [ ] ROADMAP_V2.md:1180 checked off with commit SHA (+ #4's SHA noted for the rejection mechanism).
- [ ] PR opened; merge gated on green full suite; main-merge confirmed with user.
