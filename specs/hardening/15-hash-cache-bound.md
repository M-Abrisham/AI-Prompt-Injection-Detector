---
item: 15
title: "I-cache: bound/evict the mtime-gated hash cache + richer cache key (st_size/ino/mtime_ns)"
priority_tier: P3 (hygiene / integrity-adjacent — NOT a live security fix)
depends_on: [7]          # SOFT: item 7 (sidecar-resolution rework) rewrites _resolve_expected_hash and touches the SAME file's load path that calls _cached_sha256/_cached_hmac_sha256. No hard logic dependency; land 7 first to avoid a merge collision in safe_pickle.py. If 15 lands first, 7 rebases cleanly.
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]
na_steps: [5, 9, 10]
applicable_qs: [Q1, Q2, Q3, Q4, Q8, Q10]
na_qs: [Q5, Q6, Q7, Q9]
touches: ["src/na0s/integrity/safe_pickle.py"]
flag: none               # No new env var. The bound is a justified compile-time constant; eviction is unconditional. (Optional override discussed in §3 but defaulted off-by-omission.)
---

# Item 15 — Bound/evict the mtime hash cache with a richer key

> **Scope discipline (read first).** This is a **HYGIENE** cleanup, not a
> security feature. The mtime-gated digest cache (`safe_pickle.py:46-49`,
> `_cached_sha256` `:65-73`, `_cached_hmac_sha256` `:97-105`) is an *optimization*
> that sits **after** the trust decision — it only caches a recomputed digest of
> the *same* file; it does not change which hash is *expected* or whether a load
> is *accepted*. The integrity decision is `hmac.compare_digest(actual, expected)`
> at `safe_pickle.py:335`, unaffected by this item. The only file edited is
> `src/na0s/integrity/safe_pickle.py`. `na0s.safe_pickle` is a SHIM
> (`src/na0s/safe_pickle.py:1` `# SHIM -- do not add new code here`) → do **not**
> edit it.

## 1. Root cause (confirmed against source, 2026-06-22, HEAD `hardening/rag-poison-wiring`)

Two module-global dicts cache file digests, gated only by mtime, with **no size
bound and no eviction**:

```python
# safe_pickle.py:46-49
_sha256_cache: dict = {}  # path -> (mtime, hex_digest)
_hmac_cache: dict = {}    # path -> (mtime, hex_digest)
```

```python
# safe_pickle.py:65-73
def _cached_sha256(path):
    mtime = os.path.getmtime(path)
    cached = _sha256_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    digest = _sha256(path)
    _sha256_cache[path] = (mtime, digest)   # <-- never evicted, never capped
    return digest
```

`_cached_hmac_sha256` (`safe_pickle.py:97-105`) is the structurally identical
HMAC twin. Both are read in `safe_load` at `safe_pickle.py:310` (hardcoded
branch), `:318` (hmac branch), `:333` (sha256 branch). KEY REFS cited
`65-73,97-105` — **verified exact, no drift**; the backing dicts are at `:46-49`
(KEY REFS omitted these — note the correction).

### Two distinct defects

**Defect A — unbounded growth (the headline hygiene gap).**
The dict key is the raw `path` string and entries are inserted on every cache
miss with **no cap and no eviction**. The *production* path universe is small and
fixed — `predict.py` loads exactly four pkls (`MODEL_PATH`, `VECTORIZER_PATH`,
`CHAR_VECTORIZER_PATH`, `SCALER_PATH`, `predict.py:227-230`) and
`predict_embedding.py` two more (`:60-61`), each loaded **once** behind
`predict.py`'s own `_model_cache_lock` double-checked cache
(`predict.py:281-322`). So in the steady-state SDK, the cache holds ≤ ~6 entries
and the bug is *latent*. It becomes real when `safe_load`/`safe_dump` are called
over **varying paths**: batch retraining, `model_rollback.py:81` iterating
timestamped backups (`for p in self.backup_dir.iterdir()`), `sbom.py:73`
iterating `models_dir.glob("*.pkl")`, and long-lived test processes that
`safe_dump` to fresh `tempfile` paths thousands of times — each unique path is a
permanent entry. Unbounded process-lifetime growth = a slow memory leak. **This
is what "bound/evict" in the title targets.**

**Defect B — coarse cache key can return a stale digest (the "richer key"
half).**
The freshness gate is `cached[0] == mtime` where `mtime = os.path.getmtime(path)`
(`safe_pickle.py:67`). `os.path.getmtime` returns a float in **seconds**; on many
filesystems mtime resolution is 1 s (some legacy FS, 2 s). Two consequences:
- A file *rewritten within the same mtime tick* (rapid `safe_dump`→`safe_load` in
  a tight loop or test) yields the **old cached digest** for the new bytes.
- `os.utime` can *reset* mtime to a prior value, again returning a stale digest.

This is mostly a *correctness*/test-flakiness concern, with a *minor* integrity
flavor: an attacker with write access who swaps the file **and** restores the
original mtime (`os.utime`) within a live process that already cached the old
digest would get the cached digest re-used. **Important caveat (do not
over-state this):** that attacker already has the write access required to
rewrite the on-disk sidecar too, so this is **not** a primary integrity boundary
— it is a defense-in-depth + correctness improvement. The honest framing is
"richer key removes a stale-cache footgun," not "fixes an exploit." (`size` +
`inode` + nanosecond mtime are all available for free from a single `os.stat`,
so the fix is near-zero-cost.)

**Defect C (secondary) — no thread lock on the cache dicts.**
The model-cache callers serialize through `predict.py`'s `_model_cache_lock`
(`predict.py:289`), but `safe_load` itself is documented as a general primitive
("Used by 20+ call sites", `ROADMAP_V2.md:1107`) and the cache dicts are mutated
without a lock (`safe_pickle.py:72,104`). Concurrent `safe_load` of two different
paths is a benign last-writer-wins on a `dict`, but combining an eviction
(`popitem`) with a concurrent insert on CPython can raise `RuntimeError:
dictionary changed size during iteration` only if we *iterate*; a plain
`popitem(last=False)` on an `OrderedDict` does not iterate, so a lock is
*advisable but not strictly required*. We add a single module-level
`threading.Lock` around the read-modify-write to make the bounded-LRU correct
under concurrency (mirrors the proven pattern in `llm_judge.py`, see §3). Keep
this minimal — do not gold-plate into a full RWLock.

## 2. Gap vs. ideal

| Aspect | Current (`safe_pickle.py`) | Ideal |
|---|---|---|
| Size bound | none — grows with distinct paths | bounded LRU, justified cap (§3) |
| Eviction | none | evict least-recently-used on overflow |
| Freshness key | `mtime` seconds only (`:67`) | `(st_mtime_ns, st_size, st_ino)` from one `os.stat` |
| Stale-on-same-tick rewrite | returns old digest (Defect B) | size/inode/ns change → cache miss → re-hash |
| Thread safety | unlocked dict mutation (`:72,104`) | single `Lock` around RMW (Defect C) |
| Two parallel caches | `_sha256_cache` + `_hmac_cache` (`:48-49`) | keep two (different digest fns) but share one bounded-LRU helper to avoid drift |

**Ideal invariant.** The cache is a *transparent* optimization: for a fixed
`(path, file-identity)` it returns the same digest `_sha256()`/`_hmac_sha256()`
would compute; for *any* observable change to the file it misses and recomputes;
and its memory is **O(cap)**, not O(distinct paths seen). No behavior visible to
`safe_load`'s accept/reject decision changes.

### Edge cases to cover
1. Cap+1 distinct paths loaded → cache holds exactly `cap` entries, the LRU
   (oldest-touched) evicted (Defect A core).
2. A re-load of a still-cached, unchanged file → cache HIT, no re-hash, and it
   is moved to MRU (so it survives the next eviction).
3. Same path, file rewritten with **identical mtime-seconds but different size**
   → cache MISS (Defect B; `st_size` discriminates).
4. Same path, `os.utime` resets mtime to an older value, content changed →
   cache MISS (`st_mtime_ns` or `st_size`/`st_ino` discriminates).
5. Same path, genuinely unchanged across two loads → cache HIT (no spurious
   miss / no perf regression — the optimization still works).
6. `_sha256_cache` and `_hmac_cache` are independent: a path cached for SHA must
   not be returned for an HMAC request (already true — different functions; add
   a regression guard so a future shared-helper refactor can't conflate them).
7. Concurrency: two threads loading two different paths around the cap boundary
   don't corrupt the LRU / don't raise (Defect C).
8. Production steady-state (≤6 fixed model paths) → all HITs after warm-up, cap
   never reached, zero behavior change (proves no perf regression for the real
   SDK workload).
9. File deleted between `os.stat` and `open` (TOCTOU on the stat) — out of scope
   for *this* item (the existing code already `open()`s inside `_sha256`; an
   `OSError` propagates as today; do not change error semantics).

## 3. Bound + key choice (justify — `na0s-review-checklist` "no arbitrary threshold")

**Cap value.** Set `_CACHE_MAXSIZE = 64`. Justification (not arbitrary):
- The *real* production universe is ≤ 6 distinct pkl paths (four in
  `predict.py:227-230`, two in `predict_embedding.py:60-61`). 64 is ~10× that,
  so the steady-state SDK **never evicts** (edge case 8) — the bound only kicks
  in for the pathological batch/rollback/test workloads (Defect A).
- 64 `(path_str, (int, int, int, 64-char-hex))` entries ≈ a few KB — negligible.
- It mirrors the documented in-repo precedent: the LLM-judge response cache is a
  bounded `OrderedDict` LRU (`llm_judge.py:277-279`, evicted via
  `popitem(last=False)` at `:561-562`) with `cache_size` defaulting to a small
  power-of-two-ish constant; we reuse the *exact same eviction idiom* for
  consistency. Cite this precedent in the code comment so the number is
  traceable, not magic.
- **No env flag** (CLAUDE.md "don't gold-plate"; this is hygiene). If a future
  need arises, an `int(os.getenv("NA0S_HASH_CACHE_MAX", "64"))` read at *helper
  init* is the documented extension point — but ship the constant, not the flag.

**Key choice.** Replace the `mtime`-seconds gate with an identity tuple derived
from a **single** `os.stat(path)` call:
`key_id = (st.st_mtime_ns, st.st_size, st.st_ino)`.
- `st_mtime_ns` (nanosecond mtime) — finer than `getmtime()`'s float seconds,
  catches most rapid-rewrite cases at no cost.
- `st_size` — catches same-tick rewrites that change length (edge case 3) and is
  cheap, robust across platforms.
- `st_ino` — catches atomic-replace (`safe_dump` uses `os.replace`,
  `safe_pickle.py:160-175`, which changes the inode) even when size+mtime
  coincide (edge case 4). On Windows `st_ino` may be 0 for some FS — that simply
  makes the key fall back to `(mtime_ns, size)`, still strictly better than
  today; document this, don't branch on platform.

This is the minimum richer key that closes Defects B without adding a content
re-hash (which would defeat the cache's purpose). One `os.stat` replaces the one
`os.path.getmtime` (which is itself a `stat` under the hood) — **net-zero syscall
cost.**

## 4. Implementation plan (root-cause, numbered)

**All edits in `src/na0s/integrity/safe_pickle.py` only.**

1. **Add a bounded-LRU primitive** near the cache definitions
   (`safe_pickle.py:46-49`). Replace the two plain `dict`s with two
   `collections.OrderedDict`s and a single module `threading.Lock`, plus the cap
   constant:
   ```python
   from collections import OrderedDict
   import threading
   _CACHE_MAXSIZE = 64   # ~10x the ~6 real model paths (predict.py:227-230,
                         # predict_embedding.py:60-61); mirrors the bounded LRU
                         # idiom in judge/llm_judge.py:277-279,561-562.
   _sha256_cache: "OrderedDict[str, tuple]" = OrderedDict()  # path -> (key_id, hex)
   _hmac_cache: "OrderedDict[str, tuple]" = OrderedDict()
   _cache_lock = threading.Lock()
   ```
   (`os` already imported `:31`; add `threading` + `from collections import
   OrderedDict` to the import block `:27-36`.)

2. **Add a `_file_identity(path)` helper** (the richer key) near `_sha256`
   (`:57`):
   ```python
   def _file_identity(path):
       """Cheap change-detection key: (mtime_ns, size, inode) from one stat."""
       st = os.stat(path)
       return (st.st_mtime_ns, st.st_size, st.st_ino)
   ```
   Single `os.stat` (replaces `os.path.getmtime`, which was itself a stat).

3. **Add a shared `_cache_get_or_compute(cache, path, compute)` helper** so the
   SHA and HMAC paths can't drift (DRY; `na0s-review-checklist` "silent refactor
   destruction" — one helper, two callers, identical semantics):
   ```python
   def _cache_get_or_compute(cache, path, compute):
       key_id = _file_identity(path)
       with _cache_lock:
           cached = cache.get(path)
           if cached is not None and cached[0] == key_id:
               cache.move_to_end(path)          # mark MRU
               return cached[1]
       digest = compute()                        # hash OUTSIDE the lock (slow I/O)
       with _cache_lock:
           cache[path] = (key_id, digest)
           cache.move_to_end(path)
           while len(cache) > _CACHE_MAXSIZE:
               cache.popitem(last=False)         # evict LRU (oldest-touched)
       return digest
   ```
   - **Hash computed OUTSIDE the lock** (the `compute()` read/hash is the slow
     part — holding the lock across 64 KB-chunked I/O would serialize all loads).
     Two threads racing the same cold path may both hash it once — a benign,
     idempotent double-compute, not a correctness bug (last write wins, same
     value). This matches the perf intent of the original mtime cache.
   - `move_to_end` + `popitem(last=False)` is the exact LRU idiom from
     `llm_judge.py:559-562`.

4. **Rewrite `_cached_sha256` (`:65-73`)** to delegate:
   ```python
   def _cached_sha256(path):
       return _cache_get_or_compute(_sha256_cache, path, lambda: _sha256(path))
   ```

5. **Rewrite `_cached_hmac_sha256` (`:97-105`)** to delegate (note: `key` is
   captured by the closure, NOT part of the cache key — `_hmac_cache` is only
   ever consulted on the `sidecar_hmac` branch where the key is fixed for the
   process; document this so a future multi-key caller knows to flush):
   ```python
   def _cached_hmac_sha256(path, key):
       return _cache_get_or_compute(
           _hmac_cache, path, lambda: _hmac_sha256(path, key)
       )
   ```
   **Edge note to call out in review:** the HMAC cache key does *not* include the
   signing key. In the current pipeline a process has at most one
   `NA0S_PICKLE_KEY`, so this is safe; but add a one-line comment + a test
   asserting the documented assumption, so item-7's key-aware selection (depends-
   on) doesn't silently introduce a multi-key hazard.

6. **Add a `_reset_caches()` test seam** (module-private) that clears both
   `OrderedDict`s under the lock — so the new tests and any cache-sensitive
   existing test can reset deterministically without reaching into module
   internals byte-by-byte:
   ```python
   def _reset_caches():
       with _cache_lock:
           _sha256_cache.clear()
           _hmac_cache.clear()
   ```
   (Public-ish only to tests; underscore-prefixed. Not exported in any `__all__`.)

7. **No change** to `safe_load`'s three call sites (`:310,:318,:333`) — they call
   `_cached_sha256` / `_cached_hmac_sha256` unchanged; the signatures are
   preserved (Defect A/B/C fixed entirely behind the helpers). This is the key
   "no caller churn" property.

8. **Docstrings.** Update the cache comment block (`:46-49`) to state: bounded
   LRU (cap 64), richer `(mtime_ns,size,inode)` key, thread-locked. Keep it
   honest — note it is an optimization layer, not an integrity boundary.

### Exact files / functions to change
- `src/na0s/integrity/safe_pickle.py`:
  - imports: add `threading`, `from collections import OrderedDict` (block `:27-36`).
  - `:46-49` cache dicts → two `OrderedDict` + `_cache_lock` + `_CACHE_MAXSIZE`.
  - **new** `_file_identity()` (near `:57`).
  - **new** `_cache_get_or_compute()` (near `:65`).
  - **rewrite** `_cached_sha256` (`:65-73`) and `_cached_hmac_sha256`
    (`:97-105`) to delegate.
  - **new** `_reset_caches()` test seam.
  - comment/docstring update `:46-49`.
- **No new module** (CLAUDE.md: integrity primitive stays in `integrity/`; this
  is an edit to the canonical file).
- **No shim edit** (`src/na0s/safe_pickle.py` is a SHIM).
- **No predict.py / cascade.py edit** (see Q8 / Step 4).

## Step-by-step orchestration (template steps 1-11)

- **Step 1 — Explore current rules around target.** DONE (§1-2): unbounded
  growth (Defect A), coarse stale-able key (Defect B), unlocked mutation
  (Defect C).
- **Step 2 — Roadmap / taxonomy / README / coverage for the picture.** Roadmap
  home = **Layer 11: Supply Chain Integrity** (`ROADMAP_V2.md:1104-1185`). The
  layer is "24/24 COMPLETE" (`:124,:1104`); this is a *hardening follow-up*, not a
  new task — frame it as such. The existing open L11 P2 item "Stress cases for
  `safe_pickle` … very-large files … concurrent `safe_dump`"
  (`ROADMAP_V2.md:1180`) is adjacent (concurrency overlaps Defect C) but distinct
  — cross-reference, do not fold. **Taxonomy/coverage/scorer = N/A** (Step 10):
  this is an integrity-optimization control, not an attack class.
- **Step 3 — Root-cause plan.** §4 above.
- **Step 4 — Implement + WIRE (predict.py + cascade.py parity).** **Wiring is
  automatic; no predict/cascade edit.** The cache lives entirely inside
  `_cached_sha256`/`_cached_hmac_sha256`, reached only via `safe_load`
  (`safe_pickle.py:310,318,333`). Every model load — `predict.py:306-307,371,403`,
  `predict_embedding.py:122-124,195`, `dataset/hard_negatives.py:517-518` — goes
  through `safe_load`. `cascade.py` has **no direct `safe_load` import** (grep: 0
  hits); it consumes models via `predict.py`'s cached loaders, so the one edit
  serves both pipelines. **Parity is preserved by construction** — there is no
  second copy of this cache to keep in sync.
- **Step 5 — HARVESTER AUDIT.** **N/A — no dataset.** The "input" here is a binary
  pkl on disk plus its on-disk identity (`stat`); there is no threat-intel or
  attack-string corpus to harvest, decontaminate, or tag. (Per scope rules, Step
  5/Q5/Q9 are N/A for integrity-hygiene items.)
- **Step 6 — Tests (Code + use-case).** §"Test plan" below.
- **Step 7 — Cleanup / refactor.** The file is already in its canonical
  `integrity/` home. The refactor *is* the de-duplication into
  `_cache_get_or_compute` (removes the two near-identical cache bodies at
  `:65-73`/`:97-105`). De-clutter scope: the stray top-level files
  (`_skeptic_test_out.txt`, `pyt_out.txt`, `_xfail_run.txt`, `logs/`) are **out of
  scope** for this item — leave them.
- **Step 8 — Roadmap update.** Under **L11** (`ROADMAP_V2.md:1180` area), add a
  checked hardening follow-up: "`safe_pickle` digest cache is now a bounded LRU
  (cap 64) with a `(mtime_ns,size,inode)` key + a module lock — fixes
  unbounded-growth and same-mtime-tick stale-digest footguns." Cite the merge SHA
  (per `feedback_roadmap_sync`). Note this *partially addresses* the adjacent
  concurrency bullet at `:1180` (the cache RMW is now locked) — annotate, don't
  close that bullet (its large-file/multi-process `safe_dump` parts remain).
- **Step 9 — README / Benchmark.** **README: N/A** (no new env var, no public API
  change — `safe_load`/`safe_dump` signatures unchanged). **Benchmark: N/A** — no
  recall/FPR/latency-table change for the real ≤6-path SDK workload (cache still
  HITs; edge case 8 proves it). A micro-benchmark of the bounded path is *test*,
  not *published benchmark*, scope.
- **Step 10 — Taxonomy / Coverage / thresholds.** **N/A — integrity-optimization
  hygiene; maps to no `data/taxonomy.yaml` leaf, no COVERAGE_MATRIX row, no scorer
  threshold.** (The only number introduced, the cap 64, is justified in §3, not a
  detection threshold.)
- **Step 11 — PR + held-out gate.** §"PR / test-gate" below.

## Test plan (Code + Use-case) — Step 6 / Q4

New isolated test file: **`tests/integrity/test_hash_cache.py`** (mirrors source
per CLAUDE.md test org; alongside `tests/integrity/test_safe_pickle.py`). Use
`tempfile.TemporaryDirectory` + real `safe_dump`/`_sha256` — never the bundled
models, never a network. Call `_reset_caches()` in `setUp` so every case starts
cold. Import the privates explicitly
(`from na0s.integrity.safe_pickle import _cached_sha256, _sha256_cache, …`).

**Defect A — bounded / evicting (headline):**
1. `test_cache_bounded_to_maxsize` — `safe_dump` / `_cached_sha256` over
   `_CACHE_MAXSIZE + 10` distinct temp paths; assert
   `len(_sha256_cache) == _CACHE_MAXSIZE` (exact, not "<=") — proves the bound.
2. `test_lru_evicts_oldest_not_newest` — fill to cap; touch entry #0 again
   (re-`_cached_sha256` it → MRU); insert one more; assert entry #0 is **still
   present** and the *second-oldest* was evicted (proves LRU recency, not FIFO).
3. `test_evicted_entry_recomputes_on_next_access` — after eviction, re-access an
   evicted path; assert it returns the **correct** digest (equal to `_sha256()`
   of the file) and is re-inserted — eviction is not data loss.

**Defect B — richer key catches stale content:**
4. `test_same_mtime_different_size_is_cache_miss` — write file, `_cached_sha256`,
   then rewrite with different-length content but force the **same mtime-seconds**
   via `os.utime(path, (t, t))`; assert the second `_cached_sha256` returns the
   **new** file's digest (not the stale cached one). Concrete: compare against
   `_sha256(path)` recomputed.
5. `test_utime_reset_with_changed_content_is_cache_miss` — change content, reset
   mtime to the original via `os.utime`; assert MISS + correct new digest
   (`st_size`/`st_ino` discriminate).
6. `test_unchanged_file_is_cache_hit` — two `_cached_sha256` of an untouched
   file; assert the second is a HIT (patch `na0s.integrity.safe_pickle._sha256`
   with a call-counting wrapper and assert it was called **exactly once**) — the
   optimization still works (no spurious miss / no perf regression).

**Defect C + isolation:**
7. `test_sha256_and_hmac_caches_are_independent` — `_cached_sha256` then
   `_cached_hmac_sha256(path, key)` on the same path; assert the two caches hold
   distinct digests and neither returns the other's value (regression guard for a
   future shared-helper refactor; edge case 6).
8. `test_hmac_cache_key_excludes_signing_key_documented` — assert the documented
   assumption: a second `_cached_hmac_sha256(path, key2)` with a *different* key
   on an *unchanged* file returns the **stale** key1 digest (this is the
   documented single-key-per-process limitation). The test pins the behavior so
   item-7's key-aware work can't silently regress into a multi-key hazard
   undetected — it asserts current contract + carries a `# DOCUMENTED LIMITATION`
   comment, not an aspiration.
9. `test_concurrent_loads_around_cap_do_not_raise` — spawn N threads
   (`threading`) each `_cached_sha256`-ing one of `_CACHE_MAXSIZE * 2` distinct
   temp files; join; assert no exception raised and `len(_sha256_cache) ==
   _CACHE_MAXSIZE` (the lock + non-iterating `popitem` hold under contention).

**Use-case / behavior (end-to-end through a real loader):**
10. `test_safe_load_roundtrip_unchanged_with_bounded_cache` — full
    `safe_dump(obj, path)` then `safe_load(path)` (keyless SHA path) returns the
    original object, and a second `safe_load` is a cache HIT (patch `_sha256` to
    count) — proves the bounded cache is transparent to the public API.
11. `test_scan_steady_state_no_eviction` — set `na0s.predict.SCALER_PATH` (or
    reuse the bundled fixed-path loaders), warm the model cache, run
    `na0s.scan("ignore previous instructions")` twice; assert the digest cache
    size stayed ≤ 6 and never evicted (edge case 8 — proves zero behavior/perf
    change for the real SDK workload). Keep it a thin assertion on cache size, not
    a flaky latency assertion.

No assertion-light tests: each asserts an exact `len`, a recomputed-digest
equality, a call-count, a returned object, or a no-raise under threads. No mocked
CLI substituting for the real path — tests use real `safe_dump`/`_sha256` and a
real `scan()` (`na0s-review-checklist` "no hollow tests", "mocked-CLI smoke
gap").

## Smoke step (CLI / suite — required)

1. Targeted first: `python3 -m pytest tests/integrity/ -v` — proves the new
   `test_hash_cache.py` **and** zero regression in `test_safe_pickle.py` /
   `test_l11_safe_pickle_fixes.py` (these import `_cached_sha256`/cache internals
   only via the public `safe_load` round-trips, so the signature-preserving
   refactor should leave them green — confirm explicitly).
2. CLI smoke (real, not mocked): a 3-line
   `PYTHONPATH=<worktree>/src python3 -c "import na0s; print(na0s.scan('ignore all previous instructions'))"`
   run **twice in the same process via a tiny loop** — confirm it returns a normal
   `ScanResult` both times and does not error (the model load path exercises the
   bounded cache for real). Optionally print
   `len(na0s.integrity.safe_pickle._sha256_cache)` and assert it is small/bounded.
3. Full suite last (CLAUDE.md mandate): `python3 -m pytest tests/ -q --tb=line` —
   zero net regressions before reporting done (~15 min). Verify against MAIN env
   (`PYTHONPATH=<worktree>/src`, per `na0s-debugging` — `na0s.integrity.safe_pickle`
   exists on main) to dodge the stale editable-install trap.

## Q&A self-check

- **Q1 — Can Na0S handle the target (bug + suite green)?** Pre-fix: the cache
  grows unbounded over varying paths (Defect A) and can return a stale digest on
  a same-tick rewrite (Defect B), unlocked (Defect C). Post-§4: bounded LRU (cap
  64), `(mtime_ns,size,inode)` key, locked RMW. Full suite must stay green
  (signatures unchanged → no caller churn expected).
- **Q2 — Cleanup done?** Step 7: file already canonical; the dedup into
  `_cache_get_or_compute` removes the two duplicated cache bodies. Stray
  `*_out.txt` / `logs/` out of scope.
- **Q3 — Pipeline wiring correct?** Yes — single chokepoint inside the two
  `_cached_*` helpers, reached only through `safe_load`. predict + cascade both
  route through it; no second cache copy to keep in parity. No predict/cascade
  edit.
- **Q4 — Tested for code AND use-case?** Yes — 9 cache-level tests (bound,
  LRU-recency, stale-key, isolation, concurrency) + 2 end-to-end
  `safe_load`/`scan()` behavior tests.
- **Q5 — Harvester audit.** **N/A — no harvestable dataset; the "input" is a pkl
  + its filesystem `stat`, not threat intel.**
- **Q6 — Taxonomy / Coverage.** **N/A — integrity-optimization hygiene, maps to no
  attack-class taxonomy leaf or COVERAGE_MATRIX row.**
- **Q7 — Scorer.** **N/A — no per-attack score; the only number (cap 64) is a
  cache bound justified in §3, not a detection threshold.**
- **Q8 — predict.py / cascade.py refs?** Indirect only: both consume models via
  `safe_load`-backed loaders (`predict.py:306-307,371,403`,
  `predict_embedding.py:122-124,195`); `cascade.py` has no direct `safe_load`
  import (grep: 0). One edit covers both — no predict/cascade source change.
- **Q9 — Harvester agent harvests this type?** **N/A — not harvestable intel.**
- **Q10 — Other checks.** (a) **Perf invariant:** hash computed *outside* the
  lock so loads don't serialize on I/O (§4 step 3); steady-state ≤6 paths never
  evict (test 11). (b) **Constant-time compare** at `safe_pickle.py:335` is
  untouched — the cache returns a hex string that still flows into
  `hmac.compare_digest`. (c) **No new error semantics:** `os.stat` raising for a
  vanished file propagates exactly as `os.path.getmtime` did before. (d) **HMAC
  single-key assumption** is documented + pinned by test 8 so item-7's key-aware
  selection can't silently break it. (e) **Windows `st_ino == 0`** falls back to
  `(mtime_ns, size)` — still strictly better than today; documented, not
  branched.

## Agent / skill team (inject `na0s-review-checklist` into every subagent prompt)

| Step / concern | Agent / skill |
|---|---|
| Lead plan + decomposition | `Plan` |
| Integrity-layer correctness — confirm the cache is *post-decision* and the bound/key change does not touch the accept/reject path (`compare_digest` at `:335`) | `layer-9-11-auditor` + skill `security-review` |
| Supply-chain framing — keep the "this is hygiene, NOT an exploit fix" honesty; sanity-check the Defect-B integrity caveat is not over-stated | `security-research-auditor` |
| Hunt any *other* unbounded module-global cache / unlocked RMW in `integrity/` and the loader paths (predict/predict_embedding) | `silent-failure-hunter` |
| L4/L5 loader-consumer review (scaler/char-vec/embedding-scaler call `safe_load`; confirm no caller relied on the old unbounded-cache identity) | `l3-l5-code-auditor` |
| Test authoring (LRU-recency + same-mtime-tick + concurrency fixtures), full-suite green, env-trap avoidance | skills `na0s-debugging`, `eval-harness` |
| PR prep + self-review + CI gate | `pr-review-toolkit:review-pr`, skills `github-pr-prep`, `github-ci-fix` |
| Checklist enforcement on the diff | skill `na0s-review-checklist` |

`cron-scheduling` / `data-harvesting` skills: **N/A** (no scheduled job, no
harvest).

## Execution preconditions / dependencies

- **Depends-on: item 7 (SOFT).** Item 7 (sidecar-resolution rework) rewrites
  `_resolve_expected_hash` and edits `safe_load`'s branch bodies — the same file
  and the same load path that calls `_cached_sha256`/`_cached_hmac_sha256`. No
  logic dependency (this item lives entirely in the two `_cached_*` helpers, which
  item 7 does not touch), but landing 7 first avoids a `safe_pickle.py` merge
  collision and lets test 8's HMAC single-key assumption be validated against
  item-7's final key-aware selection. If item 15 lands first, item 7 rebases
  cleanly (disjoint functions).
- **No dependency** on items 1-6 / 8-14 / 16-17 (different surfaces).
- **Env:** verify against MAIN, not the d8 editable install
  (`PYTHONPATH=<worktree>/src`) — `na0s.integrity.safe_pickle` exists on main.
- **Worktree:** isolated git worktree on `hardening/hash-cache-bound` off `main`
  (per `project_multi_agent_worktree`); never branch-switch the primary checkout,
  never `git stash`.

## Definition of done

- [ ] `_sha256_cache` / `_hmac_cache` are bounded `OrderedDict` LRUs capped at
      `_CACHE_MAXSIZE = 64` (justified, comment-cited to `llm_judge.py` precedent
      and the ≤6 real model paths), evicting LRU via `popitem(last=False)`.
- [ ] Cache key is `(st_mtime_ns, st_size, st_ino)` from one `os.stat`
      (`_file_identity`) — same-mtime-tick rewrite and `os.utime` reset both miss.
- [ ] Read-modify-write of both caches is under a single module
      `threading.Lock`; the hash itself is computed **outside** the lock (no I/O
      serialization).
- [ ] `_cached_sha256` / `_cached_hmac_sha256` signatures unchanged; `safe_load`
      call sites (`:310,:318,:333`) untouched. Two callers share one
      `_cache_get_or_compute` helper (dedup, no drift).
- [ ] `tests/integrity/test_hash_cache.py` — 9 cache-level + 2 end-to-end tests;
      all non-hollow (exact `len`, recomputed-digest equality, call-counts,
      no-raise-under-threads).
- [ ] HMAC single-key-per-process assumption documented in code + pinned by a
      test (so item-7 can't silently break it).
- [ ] `python3 -m pytest tests/integrity/ -v` green; 2× in-process `scan()` CLI
      smoke returns normal `ScanResult` with a bounded cache; full `tests/` suite
      green, zero net regressions.
- [ ] Cache comment/docstring (`:46-49`) updated to state bounded-LRU + richer key
      + lock, honestly framed as an optimization (not an integrity boundary).
- [ ] ROADMAP_V2 L11 hardening follow-up checked with merge SHA; the adjacent
      concurrency bullet (`:1180`) annotated as partially addressed (not closed).
- [ ] PR opened; full-suite / held-out gate passes before merge; merge-to-main
      confirmed with the user (per memory `feedback_no_git_commit`).
