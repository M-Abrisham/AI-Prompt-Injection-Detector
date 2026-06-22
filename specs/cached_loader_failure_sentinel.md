# Spec: cached-loader failure sentinel poisoning (P1)

## Finding
`_get_cached_scaler` (predict.py ~397-405) and `_get_cached_char_vectorizer`
(predict.py ~429-437) both cache `False` (→ treated as `None`) for BOTH:
- "file does not exist" — a **legitimate** pre-L3/L4 backward-compat absence, AND
- "file exists but `safe_load` raised" — e.g. a SHA-256/HMAC **integrity mismatch**, or a
  **partial read during a concurrent deploy swap**.

**Root cause:** one overloaded `False` sentinel for two semantically different states; the
load-failure branch only warns, then caches `False` for the process lifetime.

**Impact (MEDIUM):** a single transient `safe_load` failure **permanently poisons the
process** to word-only features (the scaler/char artifact is never retried), silently
degrading every subsequent scan. Worse, it **bypasses the F-AR8 fail-loud contract**:
because the cache returns `None`, `_transform`'s None-skip path runs and the missing
features are silently dropped — exactly the silently-wrong-scores case F-AR8 was written
to prevent, but upstream of it.

## Applicable general-prompt steps
1. Explore both cached loaders + their double-checked-locking; confirm the sentinel
   overload and the poisoning path.
2. Full picture: read `safe_pickle.safe_load` (what it raises: integrity ValueError vs
   FileNotFound vs partial-read), `_transform`'s None-skip, and F-AR8's docstring contract.
3. Root-cause plan (below).
4. Implement from root cause (predict.py loaders; cascade uses the SAME loaders so it
   inherits the fix — confirm parity).
6. Test the two cases distinctly.
7. Cleanup; 8. Roadmap; 11. PR.

## N/A steps
5, 10, Q1/Q5/Q6/Q9/Q10/Q12 (not an attack type / no harvester/taxonomy/scorer).

## Implementation (root-cause fix) — DESIGN DECISION to make explicit
Distinguish the two states:
- `os.path.isfile(path) == False` → cache `False` (genuine backward-compat absence; keep
  current graceful None-skip behavior).
- File EXISTS but `safe_load` raises → **do NOT cache the failure**, log at `error`, and
  re-raise (fail-loud, F-AR8/integrity-consistent). Rationale: a present-but-unloadable
  artifact is a real bundle/integrity problem, not graceful degradation; a transient
  concurrent-deploy partial read should be retried on the next call (no permanent poison),
  and a genuine tamper trips the safe_load integrity gate loudly rather than silently
  running on word-only features.
- If hard-raising every present-file failure is judged too aggressive for runtime
  availability, the fallback is: do-not-cache + log error (so a retry recovers) WITHOUT
  raising — but a TAMPER must still fail loud. State which contract you implement and why.
Keep the existing double-checked locking intact (it is correct).

## Test
- File absent → loader returns `None`, cached, graceful skip preserved (backward compat).
- File present but `safe_load` raises (patch `safe_load` to raise for a tmp path) → loader
  fails loud per the chosen contract AND does not permanently cache a poison sentinel
  (a subsequent successful load recovers).

## Acceptance criteria
- A transient load failure no longer permanently degrades the process to word-only features.
- A genuine integrity/tamper failure surfaces loudly (does not silently drop features).
- Backward-compat (artifact legitimately absent) unchanged.
- No regression in predict/cascade/integrity suites; verified in shared venv (numpy 2.4).

## Q&A checks
- Q2 (cleanup): no dead sentinel branches left.
- Q3 (wiring): cascade inherits the fix via the shared loaders — confirm.
- Q4: both states tested.
- Q13: confirm no other cached loader (`_get_cached_models`, vectorizer cache) has the same
  sentinel overload; if so, note it.
