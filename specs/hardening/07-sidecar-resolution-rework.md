---
item: 7
title: "Sidecar resolution: plain-SHA256 downgrade fail-closed + L1 .hmac DoS (+L4/L5 defense-in-depth)"
priority_tier: P0 (supply-chain / integrity)
depends_on: [6]          # SOFT: item 6 (fail-closed optional loaders) touches the SAME loaders; land 6 first to avoid a merge collision. No hard logic dependency.
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]
na_steps: [5, 9, 10]
applicable_qs: [Q1, Q2, Q3, Q4, Q8, Q10]
na_qs: [Q5, Q6, Q7, Q9]
touches: ["src/na0s/integrity/safe_pickle.py"]
flag: NA0S_PICKLE_KEY            # existing env var; this item changes how its presence/absence drives sidecar selection. NO new flag is introduced.
---

# Item 7 — Sidecar resolution rework: downgrade fail-closed + keyless `.hmac` DoS

> **Naming disambiguation (read first).** This item is *core* `safe_pickle`
> (L11 supply-chain). It is independent of the *attack-class* taxonomy. The
> "L4/L5 defense-in-depth" in the title refers to the fact that the affected
> artifacts (`structural_scaler.pkl`, `char_tfidf_vectorizer.pkl`,
> `embedding_structural_scaler.pkl`) are consumed by the L4/L5 loaders — it
> does **not** mean an L4/L5 detector rule change. The only source file this
> item edits is `src/na0s/integrity/safe_pickle.py`.

## 1. Root cause (confirmed against source, 2026-06-22, HEAD `hardening/rag-poison-wiring`)

`na0s.integrity.safe_pickle` runs a 3-tier trust hierarchy resolved in
`_resolve_expected_hash()` (`safe_pickle.py:214-244`): hardcoded
`KNOWN_HASHES` → `.hmac` sidecar → `.sha256` sidecar. The chosen source then
drives verification in `safe_load()` (`safe_pickle.py:295-363`).
(`na0s.safe_pickle` is a SHIM → `na0s.integrity.safe_pickle`,
`src/na0s/safe_pickle.py:1,13` — do **not** edit the shim.)

Two distinct flaws, both **on the keyless / user-trained artifact path**
(bundled files in `KNOWN_HASHES`, `src/na0s/models/__init__.py:26-30`:
`model.pkl`, `structural_scaler.pkl`, `model_embedding.pkl`,
`tfidf_vectorizer.pkl`, always resolve `hardcoded` first at
`safe_pickle.py:221-222`, so they are NOT exposed — the exposed artifacts are
the *optional / user-trained* ones with only on-disk sidecars):

### Threat A — plain-SHA256 downgrade is fail-OPEN when a key IS set
- `_resolve_expected_hash` accepts a `.sha256` sidecar at
  `safe_pickle.py:232-236` **without consulting `NA0S_PICKLE_KEY`**.
- In `safe_load` the `sidecar_sha256` branch (`safe_pickle.py:319-333`), when a
  key IS set, only emits an audit warning + a logger warning
  (`safe_pickle.py:320-332`) and then verifies against the plain SHA
  (`safe_pickle.py:333`).
- Plain SHA-256 of a file is **attacker-recomputable**: anyone with write
  access can replace `<file>.pkl`, delete `<file>.hmac`, and rewrite
  `<file>.sha256` to the SHA of the malicious pickle. The load then **succeeds**
  — a silent downgrade from HMAC-authenticated to attacker-forgeable. The
  module docstring (`safe_pickle.py:18-21`) calls this fallback "weaker" but the
  code treats it as a *warn-and-accept*, not a *fail-closed*, downgrade. This is
  the classic signature-stripping / algorithm-downgrade attack.

### Threat B — `.hmac` sidecar bricks the load in KEYLESS mode (DoS)
- `_resolve_expected_hash` prefers `.hmac` whenever the file merely **exists**
  on disk (`safe_pickle.py:224-226`), regardless of whether a key is set.
- In `safe_load`, the `sidecar_hmac` branch with **no key** raises
  unconditionally (`safe_pickle.py:312-317`:
  `ValueError("HMAC sidecar exists … but NA0S_PICKLE_KEY is not set")`).
- So an attacker (or a botched deploy) who merely **drops an attacker-authored
  `<file>.hmac`** beside a legitimate `<file>.pkl` + valid `<file>.sha256` in a
  keyless deployment **bricks the model load** — even though a perfectly valid
  `.sha256` is sitting right there and would verify fine. This is a
  denial-of-service / availability bug: a deployment that was working keyless
  is taken offline by a single dropped file. **This is the item's headline
  test.**

**Net effect.** The sidecar-selection logic conflates "which sidecar files
exist on disk" with "which trust tier the operator is actually configured for".
The trust tier should be driven by *operator configuration* (`NA0S_PICKLE_KEY`
presence), not by *which file an attacker chose to drop*. Threat A = fail-open
(integrity), Threat B = fail-closed-too-hard (availability). Both stem from the
same selection bug.

### Line-ref drift note (KEY REFS reconciliation)
KEY REFS cited `safe_pickle.py:113-125,183-210,214-244,295-363`. Verified
current as of HEAD:
- `113-125` = `_parse_sidecar` (still exact). Relevant because a `v1:hmac-sha256:`
  *typed* sidecar lets us tell HMAC content apart from SHA content even before
  selection — see plan step 2.
- `183-210` = `_check_permissions` (still exact). Tangential; a world-writable
  sidecar is the *enabler* for both threats. We will surface (not silently
  swallow) the permission warning in the resolution path.
- `214-244` = `_resolve_expected_hash` (still exact) — **primary fix site.**
- `295-363` = `safe_load` (still exact) — **secondary fix site** (the two
  branch bodies at `:312-317` and `:319-333`).

No material drift; all four windows are correct.

## 2. Gap vs. ideal

| Scenario (file present unless noted) | Current behavior | Ideal behavior |
|---|---|---|
| Key set, only `.hmac` | verify HMAC (correct) | unchanged |
| Key set, only `.sha256` | warn + accept plain SHA (**fail-open downgrade, Threat A**) | **fail-closed**: refuse unless explicitly allowed; never auto-accept attacker-forgeable SHA when the operator opted into HMAC |
| Key set, `.hmac` + `.sha256` both present | prefer `.hmac`, verify HMAC (correct) | unchanged (prefer strongest) |
| **Keyless**, only `.sha256` | verify plain SHA (correct, legacy) | unchanged |
| **Keyless**, `.hmac` + valid `.sha256` | **raise — DoS (Threat B)** | **fall back to the `.sha256`** and verify it; an unverifiable `.hmac` we cannot check must not veto a verifiable `.sha256` |
| **Keyless**, only `.hmac` | raise (`safe_pickle.py:314-317`) | unchanged — genuinely unverifiable, correct to refuse (but message should say "no `.sha256` fallback present") |
| In `KNOWN_HASHES` (bundled) | hardcoded, both threats N/A | unchanged |

**Ideal invariant.** *Trust tier is chosen by operator configuration, not by
attacker-controlled file presence.* Concretely:
- **Key set** → require an HMAC-authenticated sidecar. A plain `.sha256` is a
  *downgrade* and must fail closed by default.
- **Keyless** → the verifiable artifact is the `.sha256`. An `.hmac` we cannot
  key-check is *advisory at best*; its mere presence must never *remove* the
  ability to verify a valid `.sha256`.

### Edge cases to cover
1. Keyless, `.hmac` + valid `.sha256` → verify the `.sha256`, load OK (Threat B fix; headline).
2. Keyless, `.hmac` + *tampered* `.sha256` → still raise `Integrity check failed` (the `.sha256` fallback must itself verify; we don't blindly accept it).
3. Keyless, only `.hmac` (no `.sha256`) → raise, message names the missing fallback (genuinely unverifiable).
4. Key set, only `.sha256`, default → **fail closed** with a downgrade-specific message (Threat A fix).
5. Key set, only `.sha256`, explicit allow (`NA0S_ALLOW_SHA256_DOWNGRADE=1`) → log auditable warning + verify SHA (air-gapped recovery / migration window only).
6. Key set, `.hmac` + `.sha256` → prefer + verify `.hmac`, ignore `.sha256` (unchanged; already correct, regression-guard it).
7. Bundled `KNOWN_HASHES` file → unaffected by either change (hardcoded wins; regression-guard it).
8. Tampered `.hmac` *with* a valid key (Threat A's mirror) → still `Integrity check failed` (unchanged).
9. Sidecar **typing**: a `v1:hmac-sha256:` payload sitting in a `.sha256`-named file (or vice-versa) must not let an attacker smuggle the wrong algorithm; selection must key off the **filename extension** (`.hmac` vs `.sha256`), which is what drives the verifier, AND the parsed `v1:<algo>:` tag should be cross-checked against the expected algorithm (mismatch → refuse). See plan step 2.
10. `_check_permissions` already warns on world-readable / group-writable sidecars (`safe_pickle.py:183-210`) — ensure the resolution path still calls it for the *selected* sidecar (currently only `safe_dump` calls it, `:290-291`; `safe_load` does not). Defense-in-depth, low priority — gate behind "don't gold-plate".

## 3. Default decision (justify — no arbitrary numeric threshold)

This item is **boolean gates only**; the `na0s-review-checklist`
"arbitrary threshold" rule is satisfied by construction (no numbers introduced).
Two decisions, both justified by the threat model and `project_positioning`
(Na0S is a *security* SDK → secure default):

- **`NA0S_ALLOW_SHA256_DOWNGRADE` defaults to `0` (disabled).** When a key is
  set, a plain `.sha256` fails closed. Rationale: opting into HMAC is an
  explicit statement that the operator wants forgery-resistant integrity;
  silently accepting a forgeable SHA defeats that. Opt-out exists only for a
  migration window (a fleet mid-upgrade from SHA to HMAC sidecars).
- **Threat B fix is unconditional** (no flag): falling back from an
  unverifiable `.hmac` to a verifiable `.sha256` in keyless mode is *strictly
  safer* than the status quo (it still *verifies* the `.sha256`; it just stops
  letting a dropped file veto a good one). No opt-out needed because there is no
  security regression — the `.sha256` is still cryptographically checked.

Read both flags at **call time** via a tiny helper (mirrors item 6's
`_fail_closed()` pattern) so tests can `monkeypatch.setenv` without re-import
(`na0s-review-checklist` "env blind spot").

## 4. Implementation plan (root-cause, numbered)

**All edits in `src/na0s/integrity/safe_pickle.py` only.**

1. **Add a call-time helper** near `_get_signing_key` (`safe_pickle.py:82-85`):
   ```python
   def _allow_sha256_downgrade():
       """True if operator explicitly permits plain-SHA256 sidecar despite a key."""
       return os.getenv("NA0S_ALLOW_SHA256_DOWNGRADE", "0") not in ("0", "false", "False", "")
   ```
   (`os` already imported, `safe_pickle.py:31`.) Default = refuse.

2. **Make `_resolve_expected_hash` key-aware (`safe_pickle.py:214-244`).** The
   selection must consult `_get_signing_key()` so it returns the sidecar that
   matches the operator's configured trust tier, with a safe fallback:
   - Hardcoded first — unchanged (`:220-222`).
   - Compute `key = _get_signing_key()`, `hmac_file`, `hash_file` existence.
   - **Key set:** prefer `.hmac` (return `"sidecar_hmac"`). Else if `.sha256`
     exists: this is the *downgrade* case — return `"sidecar_sha256"` **only if**
     `_allow_sha256_downgrade()`, otherwise raise a *new, downgrade-specific*
     `ValueError` ("NA0S_PICKLE_KEY is set but only a plain SHA-256 sidecar
     exists for {path}; refusing to downgrade. Re-run safe_dump to write an HMAC
     sidecar, or set NA0S_ALLOW_SHA256_DOWNGRADE=1 for a migration window.").
     This converts Threat A from fail-open to fail-closed.
   - **Keyless:** if `.sha256` exists, return `"sidecar_sha256"` (verifiable).
     Else if `.hmac` exists, raise the *existing* unverifiable-HMAC error but with
     an improved message noting no `.sha256` fallback is present. **Key change
     for Threat B:** when keyless, a present-but-unverifiable `.hmac` must **not
     be selected over** a present, verifiable `.sha256`. (i.e. in keyless mode
     the `.sha256` wins; the `.hmac` is ignored because we cannot check it.)
   - The current code's bug is that the `.hmac`-preference (`:224-226`) is
     evaluated *before* the key is consulted and *unconditionally*; we move the
     key check ahead of it.
   - Preserve the existing `FileNotFoundError` for "no source at all"
     (`:239-244`).

3. **Cross-check the parsed sidecar tag against the expected algorithm
   (edge case 9).** `_parse_sidecar` (`safe_pickle.py:113-125`) currently
   discards the `<algo>` field. Add a sibling `_parse_sidecar_typed(raw)` that
   returns `(algo_or_None, digest)`, and in `_resolve_expected_hash` after
   reading a sidecar, if the file declares a `v1:<algo>:` tag that contradicts
   the extension it was read from (e.g. a `v1:hmac-sha256:` payload in a
   `.sha256`-named file), raise `ValueError` ("sidecar algorithm tag does not
   match its filename"). Keep legacy bare-hex acceptance (no tag → trust the
   extension; `na0s-review-checklist` backward-compat). This closes the
   smuggling vector without breaking the legacy bare-hex tests
   (`test_l11_safe_pickle_fixes.py:115-117,147-160`).

4. **Simplify `safe_load`'s two branches (`safe_pickle.py:312-333`).** Because
   selection is now key-aware, the `sidecar_hmac` branch's `if not key: raise`
   guard (`:313-317`) becomes a defensive invariant (selection only returns
   `sidecar_hmac` when a key is present) — keep it as a belt-and-suspenders
   assertion with a comment, do not delete (avoid silent-refactor destruction).
   The `sidecar_sha256` branch's "key set but SHA sidecar" warning (`:320-332`)
   now only fires on the **explicit opt-out** path (`_allow_sha256_downgrade()`
   true), so the warning text stays accurate; keep it (it is the audit trail for
   the migration-window opt-out).

5. **(Defense-in-depth, optional — only if cheap, do not gold-plate.)** Call
   `_check_permissions(selected_sidecar, label="sidecar")` once in
   `_resolve_expected_hash` after selection so a world-writable sidecar is logged
   at *load* time, not just *dump* time. Skip if it complicates the diff.

6. **Docstrings.** Update the module trust-hierarchy docstring
   (`safe_pickle.py:7-24`) and `safe_load`'s (`:295-302`) to state the new
   key-aware selection rule and the downgrade-refusal default. The current
   docstring (`:18-21`) describes the *old* warn-and-accept behavior and would
   be stale — fixing it is part of the change, not gold-plating.

### Exact files / functions to change
- `src/na0s/integrity/safe_pickle.py`:
  - **new** `_allow_sha256_downgrade()` (near `:82`).
  - **new** `_parse_sidecar_typed()` (near `:113`); `_parse_sidecar` kept for
    backward compat / existing imports in `test_l11_safe_pickle_fixes.py:31`.
  - **rewrite** `_resolve_expected_hash` (`:214-244`) — key-aware selection.
  - **light edit** `safe_load` branch comments (`:312-333`).
  - **docstring** updates (`:7-24`, `:295-302`).
- **No new module** (CLAUDE.md: integrity primitives stay in `integrity/`; this
  is an edit to an existing canonical file, not a new top-level dump).
- **No shim edit** (`src/na0s/safe_pickle.py` is a SHIM — do not touch).

## Step-by-step orchestration (template steps 1-11)

- **Step 1 — Explore current rules around target.** DONE (§1-2): two flaws in
  sidecar selection (`_resolve_expected_hash` + the two `safe_load` branches).
- **Step 2 — Roadmap / taxonomy / README / coverage for the picture.** Roadmap
  home = **Layer 11: Supply Chain Integrity** section (`ROADMAP_V2.md:1104-1107`),
  which already documents the "3-tier trust hierarchy … HMAC-SHA256 sidecar …
  plain SHA-256 sidecar (backward-compatible)". Add the two fixes as checked
  items there (Step 8). No taxonomy/coverage row applies (Step 10 N/A — integrity
  control, not an attack class).
- **Step 3 — Root-cause plan.** §4 above.
- **Step 4 — Implement + WIRE (predict.py + cascade.py parity).** **Wiring is
  automatic and requires no predict/cascade edit.** Both pipelines reach this
  code only through `safe_load()`; callers are
  `predict.py:306-307,371,403`, `predict_embedding.py:122-124,195`,
  `dataset/hard_negatives.py:517-518` (grep-confirmed). `cascade.py` has **no
  direct `safe_load` import** (grep: 0 hits) — it consumes models via the
  `predict.py`-cached loaders, so fixing `safe_pickle` fixes both paths with one
  edit. This is the key wiring fact for Q3/Q8. **Do not** add a parallel guard in
  predict/cascade (avoid drift). **Interaction with item 6:** item 6 wraps the
  *optional* loaders to re-raise integrity `ValueError` under `NA0S_FAIL_CLOSED`;
  this item makes `safe_load` raise in *more* cases (Threat A) and *fewer* cases
  (Threat B). The two compose cleanly — item 6's re-raise simply surfaces this
  item's new downgrade `ValueError`. Land item 6 first to keep the loader diffs
  from colliding (depends-on: 6, soft).
- **Step 5 — HARVESTER AUDIT.** **N/A — the "dataset" is a tampered binary
  artifact + a dropped sidecar, not harvested threat intel; nothing for the
  data-harvesting pipeline to ingest.**
- **Step 6 — Tests (Code + use-case).** §"Test plan" below.
- **Step 7 — Cleanup / refactor.** The touched file is already in its canonical
  `integrity/` home (no move needed). Keep `_parse_sidecar` *and* add
  `_parse_sidecar_typed` rather than changing the existing signature (existing
  tests import `_parse_sidecar`, `test_l11_safe_pickle_fixes.py:31`). De-clutter:
  the stray top-level files (`_skeptic_test_out.txt`, `pyt_out.txt`,
  `_xfail_run.txt`, `logs/`) are **out of scope** for this item — do not touch.
- **Step 8 — Roadmap update.** Under **L11 Supply Chain Integrity**
  (`ROADMAP_V2.md:1104+`), add two checked items: (a) "sidecar selection is
  key-aware: plain-SHA256 sidecar fails closed when `NA0S_PICKLE_KEY` is set
  (downgrade refusal), `NA0S_ALLOW_SHA256_DOWNGRADE=1` migration opt-out"; (b)
  "keyless `.hmac`-DoS fixed: an unverifiable `.hmac` no longer vetoes a valid
  `.sha256` fallback". Cite the merge SHA when landed (per
  `feedback_roadmap_sync`). Note L11 is marked "24/24 COMPLETE" — these are two
  *hardening follow-ups* to a completed layer; reflect that framing.
- **Step 9 — README / Benchmark.** README: add `NA0S_ALLOW_SHA256_DOWNGRADE` to
  the env-var table (alongside `NA0S_PICKLE_KEY`). **Benchmark: N/A** — no
  recall/FPR change; the *success* path for legitimately-keyed and legitimately-
  keyless loads is unchanged. Only attacker / misconfig paths change behavior.
- **Step 10 — Taxonomy / Coverage / thresholds.** **N/A — supply-chain
  integrity control; maps to no `data/taxonomy.yaml` leaf, no COVERAGE_MATRIX
  row, and introduces no scorer threshold.**
- **Step 11 — PR + held-out gate.** §"PR / test-gate" below.

## Test plan (Code + Use-case) — Step 6 / Q4

New isolated test file: **`tests/integrity/test_sidecar_resolution.py`**
(mirrors source per CLAUDE.md test org; reuses the `safe_dump`-then-mutate
tamper idiom proven in `tests/integrity/test_l11_safe_pickle_fixes.py:196-203`
and `tests/integrity/test_safe_pickle.py:166-184`). All tests `patch.dict`
`NA0S_PICKLE_KEY` / `NA0S_ALLOW_SHA256_DOWNGRADE` per case and use
`tempfile.TemporaryDirectory` fixtures — never the real bundled models, never a
network. Reset the `_sha256_cache` / `_hmac_cache` module dicts
(`safe_pickle.py:48-49`) between cache-sensitive cases.

**Threat B (headline — keyless `.hmac` DoS):**
1. `test_keyless_dropped_hmac_does_not_brick_valid_sha256` — keyless
   `safe_dump` writes pkl + valid `.sha256`; attacker drops an arbitrary
   `<file>.hmac` (e.g. `_format_sidecar("hmac-sha256", "0"*64)`); `safe_load`
   keyless **returns the original object** (no raise). This is the exact
   item-scope assertion: "attacker-dropped .hmac beside valid .sha256 in keyless
   mode does not brick load."
2. `test_keyless_dropped_hmac_plus_tampered_sha256_still_raises` — same setup
   but ALSO tamper the `.pkl` so the `.sha256` no longer matches → still
   `pytest.raises(ValueError, match="Integrity check failed")` (the `.sha256`
   fallback is *verified*, not blindly accepted; edge case 2).
3. `test_keyless_only_hmac_no_sha256_fallback_raises` — keyless, only `.hmac`,
   no `.sha256` → raises; assert the message names the missing `.sha256`
   fallback (edge case 3).

**Threat A (downgrade fail-closed):**
4. `test_key_set_sha256_only_fails_closed_by_default` — write a keyless
   `.sha256`-sidecar artifact, then load with `NA0S_PICKLE_KEY` set and
   `NA0S_ALLOW_SHA256_DOWNGRADE` unset → `pytest.raises(ValueError,
   match="refusing to downgrade")` (Threat A core; edge case 4).
5. `test_key_set_sha256_downgrade_allowed_with_optout` — same, but
   `monkeypatch.setenv("NA0S_ALLOW_SHA256_DOWNGRADE","1")` → loads the object
   **and** emits the audit/logger downgrade warning (`caplog` on
   `na0s.safe_pickle`/`na0s.integrity_audit`); edge case 5.
6. `test_key_set_forged_sha256_swap_still_refused` — the actual attack:
   keyless-dump, then with key set, attacker replaces `.pkl` + recomputes a
   *valid* `.sha256` of the malicious pickle, deletes any `.hmac`. Default
   (no opt-out) → raises "refusing to downgrade" (the forgery never even gets to
   the compare). With opt-out=1 the forged SHA *would* verify — assert that too,
   documenting the residual risk the opt-out accepts (proves the flag is the
   only thing standing between safe and forgeable; justifies the `=0` default).

**Regression guards (must-not-break):**
7. `test_key_set_prefers_hmac_over_sha256` — both sidecars present, key set →
   HMAC verified, load OK (edge case 6).
8. `test_keyless_sha256_roundtrip_unchanged` — pure legacy keyless path still
   round-trips (mirrors `test_safe_pickle.py:97-104`).
9. `test_key_set_hmac_roundtrip_unchanged` — pure HMAC path still round-trips
   (mirrors `test_safe_pickle.py:78-82`).
10. `test_bundled_known_hash_unaffected` — a file whose basename is in
    `KNOWN_HASHES` resolves `hardcoded` regardless of sidecars present (edge
    case 7); construct by monkeypatching a temp file's basename into a copied
    `KNOWN_HASHES` entry, or assert the resolution `source == "hardcoded"`.
11. `test_typed_sidecar_algo_mismatch_refused` — write a `v1:hmac-sha256:…`
    payload into a `.sha256`-named file → `safe_load` raises on the algorithm/
    extension mismatch (edge case 9); plus
    `test_legacy_bare_hex_sidecar_still_loads` re-confirming
    `test_l11_safe_pickle_fixes.py:147-160` still passes under the new
    `_parse_sidecar_typed` path.

**Use-case / behavior (end-to-end through a real loader):**
12. `test_scan_loads_keyless_model_with_stray_hmac` — set the loader's `*_PATH`
    global (monkeypatch `na0s.predict.SCALER_PATH` to a keyless
    `safe_dump`'d-then-stray-`.hmac` temp file), reset the scaler cache, run the
    real `scan("ignore previous instructions")` keyless → it completes and
    returns a normal `ScanResult` (Threat B fix reaches the pipeline; before
    the fix this would raise/degrade). Pair with
    `test_scan_refuses_downgraded_model_with_key_set` — key set + only a
    `.sha256` sidecar on a monkeypatched mandatory `MODEL_PATH` → `scan()`
    propagates the downgrade `ValueError` (mandatory loaders already
    fail-closed; combined with item 6 the optional ones do too).

No assertion-light tests: each asserts a raised type+message **or** a concrete
returned object / `ScanResult` field, plus a `caplog` assertion on the
opt-out/downgrade warning. Tamper recipes are concrete (real `safe_dump` +
byte/sidecar mutation), not mocked — satisfies `na0s-review-checklist` "no
hollow tests".

## Smoke step (CLI / suite — required)

1. Targeted first: `python3 -m pytest tests/integrity/ -v` — proves the new file
   + **no regression** in the existing `test_safe_pickle.py` /
   `test_l11_safe_pickle_fixes.py` (especially `test_replace_both_attack_blocked`
   `:166-184`, `test_key_set_but_sha256_sidecar_warns` `:240-255` — note this
   *last* test asserts the OLD warn-and-accept behavior with a key set, so it
   **will need updating** to either set `NA0S_ALLOW_SHA256_DOWNGRADE=1` or assert
   the new refusal; flag this explicitly — it is a *behavior change*, not a
   weakened assertion, per `feedback_never_delete_to_fix`).
2. CLI smoke (real, not mocked): in a tmp dir, keyless-`safe_dump` a scaler,
   drop a junk `.hmac`, point `NA0S` at it, and run the package CLI / a 3-line
   `python -c "import na0s; print(na0s.scan('hi'))"` with the monkeypatched path
   — confirm it does **not** error out (Threat B). Then set `NA0S_PICKLE_KEY`,
   delete the `.hmac`, and confirm the same load now **exits non-zero / surfaces
   the downgrade error** (Threat A). Proves the behavior reaches top level.
3. Full suite last (CLAUDE.md mandate): `python3 -m pytest tests/ -q --tb=line`
   — zero net regressions before reporting done (~15 min). Verify against MAIN
   env (`PYTHONPATH=<worktree>/src`) per `na0s-debugging` to dodge the stale
   editable-install trap; `na0s.integrity.safe_pickle` exists on main.

## Q&A self-check

- **Q1 — Can Na0S handle the target?** Not yet: Threat A is fail-open, Threat B
  is a DoS. After §4, selection is key-aware → downgrade fails closed, stray
  `.hmac` no longer bricks keyless loads. Full suite must stay green.
- **Q2 — Cleanup done?** Step 7: file already canonical; keep `_parse_sidecar`
  for back-compat; stray `*_out.txt` / `logs/` out of scope.
- **Q3 — Pipeline wiring correct?** Yes — single `safe_load` chokepoint; predict
  + cascade both route through it (`predict.py:306-307,371,403`;
  `predict_embedding.py:122-124,195`; cascade has no direct import). One edit
  fixes both. No duplicate guard added.
- **Q4 — Tested for code AND use-case?** Yes — 11 code-level resolution tests +
  2 end-to-end `scan()` behavior tests.
- **Q5 — Harvester audit.** **N/A — artifact/sidecar tamper, nothing to harvest.**
- **Q6 — Taxonomy / Coverage.** **N/A — integrity control, no attack-class
  taxonomy or COVERAGE_MATRIX row.**
- **Q7 — Scorer.** **N/A — no per-attack score; boolean integrity/availability
  gate.**
- **Q8 — predict.py / cascade.py refs?** Indirect: both consume models via
  `safe_load`-backed loaders (`predict.py:306-307,371,403`); cascade reuses the
  predict-cached loaders (no direct `safe_load`). Covered by the single edit; no
  predict/cascade source change.
- **Q9 — Harvester agent harvests this type?** **N/A — not harvestable intel.**
- **Q10 — Other checks.** (a) Cache coherence: `_sha256_cache` / `_hmac_cache`
  (`safe_pickle.py:48-49`) are keyed by path+mtime, not by selected algorithm —
  confirm a path that flips selection (e.g. key toggled between loads) doesn't
  return a stale cross-algorithm digest; the caches are per-digest-fn so this is
  safe, but add a test note. (b) Constant-time compare on the SHA fallback is
  preserved (`hmac.compare_digest`, `safe_pickle.py:335`). (c) Confirm no other
  caller depends on the old `.hmac`-always-preferred behavior in keyless mode
  (grep callers — none set a key conditionally per-load).

## Agent / skill team (inject `na0s-review-checklist` into every subagent prompt)

| Step / concern | Agent / skill |
|---|---|
| Lead plan + decomposition | `Plan` |
| Integrity / supply-chain correctness of key-aware selection + downgrade refusal + DoS fix | `security-research-auditor` + skill `security-review` |
| L9-L11 integrity-layer cross-check (safe_pickle 3-tier contract, sidecar parsing) | `layer-9-11-auditor` |
| Find any *other* fail-open / fail-closed-too-hard sidecar/selection site; confirm no swallowed selection error elsewhere | `silent-failure-hunter` |
| L4/L5 loader-consumer review (scaler/char-vec/embedding-scaler paths that call `safe_load`) | `l3-l5-code-auditor` |
| Test authoring + tamper-fixture correctness, full-suite green, env-trap avoidance, updating the now-stale `test_key_set_but_sha256_sidecar_warns` | skills `eval-harness`, `na0s-debugging` |
| PR prep + self-review + CI gate | `pr-review-toolkit:review-pr`, skills `github-pr-prep`, `github-ci-fix` |
| Checklist enforcement on the diff | skill `na0s-review-checklist` |

`cron-scheduling` / `data-harvesting` skills: **N/A** for this item (no
scheduled job, no harvest).

## Execution preconditions / dependencies

- **Depends-on: item 6 (SOFT).** Item 6 (fail-closed optional loaders) edits the
  same `predict.py` / `predict_embedding.py` *loaders* this item's `safe_load`
  feeds. No logic dependency, but landing 6 first avoids a merge collision and
  lets 6's re-raise cleanly surface this item's new downgrade `ValueError`. If
  this item lands first, item 6 still applies on top without conflict.
- **No dependency** on items 1-5 / 8 / 13 / 17 (different surfaces).
- **Env:** verify against MAIN, not the d8 editable install
  (`PYTHONPATH=<worktree>/src`) — `na0s.integrity.safe_pickle` exists on main.
- **Worktree:** isolated git worktree on `hardening/sidecar-resolution-rework`
  off `main` (per `project_multi_agent_worktree`); never branch-switch the
  primary checkout, never `git stash`.

## Definition of done

- [ ] `_allow_sha256_downgrade()` call-time helper added (default `0`/refuse),
      justified boolean default (no numeric threshold).
- [ ] `_resolve_expected_hash` is **key-aware**: key set + only `.sha256` →
      fail closed with a downgrade-specific `ValueError` (unless explicit
      opt-out); keyless + `.hmac` + valid `.sha256` → selects & verifies the
      `.sha256` (DoS fix); keyless + only `.hmac` → raises with an improved
      "no `.sha256` fallback" message.
- [ ] Sidecar algorithm-tag/extension mismatch refused (`_parse_sidecar_typed`),
      legacy bare-hex sidecars still load.
- [ ] `safe_load` HMAC-no-key guard retained as a defensive invariant (commented);
      downgrade warning fires only on the opt-out path.
- [ ] Bundled `KNOWN_HASHES` files unaffected (hardcoded wins) — regression-guarded.
- [ ] `tests/integrity/test_sidecar_resolution.py` — 11 resolution tests + 2
      end-to-end `scan()` behavior tests; all non-hollow.
- [ ] Stale `test_key_set_but_sha256_sidecar_warns`
      (`test_safe_pickle.py:240-255`) updated to the new behavior (refuse, or
      assert under opt-out) — fixed, not weakened.
- [ ] `python3 -m pytest tests/integrity/ -v` green; CLI smoke shows keyless
      load survives a stray `.hmac` and key-set load refuses a SHA downgrade;
      full `tests/` suite green, zero net regressions.
- [ ] Module + `safe_load` docstrings updated to the key-aware rule.
- [ ] README env note (`NA0S_ALLOW_SHA256_DOWNGRADE`) + ROADMAP_V2 L11 items
      checked with merge SHA.
- [ ] PR opened; full-suite / held-out gate passes before merge; merge-to-main
      confirmed with the user (per memory `feedback_no_git_commit`).
