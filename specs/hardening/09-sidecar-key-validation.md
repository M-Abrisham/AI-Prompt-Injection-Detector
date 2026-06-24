---
item: 9
title: "L3 digest 64-hex validation in _parse_sidecar + L2 reject weak NA0S_PICKLE_KEY"
priority_tier: H2 (high — supply-chain / integrity input-validation)
category: supply-chain / model-integrity hardening (input validation on trust tier-2/3)
depends_on: []           # self-contained; na0s.integrity.safe_pickle already on main
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]
na_steps: [5, 9, 10]
applicable_qs: [Q1, Q2, Q3, Q4, Q8, Q10]
na_qs: [Q5, Q6, Q7, Q9]
files_touched:
  - src/na0s/integrity/safe_pickle.py            # _parse_sidecar 64-hex guard + _get_signing_key min-len reject
  - tests/integrity/test_safe_pickle.py          # extend (or add a focused class)
  - tests/integrity/test_l11_safe_pickle_fixes.py # extend _parse_sidecar parse-class
roadmap_refs:
  - ROADMAP_V2.md:1107    # L11 prose: 3-tier trust, versioned sidecar parsing
  - ROADMAP_V2.md:1172    # L11 completed-summary: versioned sidecar format claim
  - ROADMAP_V2.md:1177    # "Externalize integrity knobs" P3 (NA0S_PICKLE_KEY env-var name)
  - ROADMAP_V2.md:1180    # "Stress cases for safe_pickle" P2 test-coverage gap (sibling)
---

# H2 — L3 digest 64-hex validation in `_parse_sidecar` + L2 reject weak `NA0S_PICKLE_KEY`

## 0. Confirmed root cause (refs verified against live file)

KEY REFS named `src/na0s/integrity/safe_pickle.py:82-85,113-125`. Opened the
file — **both refs are accurate, no drift.**

| Concern | KEY REF | Actual line(s) | Status |
|---|---|---|---|
| `_get_signing_key()` — weak/short key accepted | 82-85 | **82-85** | accurate — THE L2 BUG |
| `_parse_sidecar()` — no digest shape validation | 113-125 | **113-125** | accurate — THE L3 BUG |

**Defect 1 — `_get_signing_key()` (lines 82-85):**

```python
def _get_signing_key():
    """Return the HMAC signing key from NA0S_PICKLE_KEY env var, or None."""
    key_str = os.getenv("NA0S_PICKLE_KEY", "")
    return key_str.encode() if key_str else None
```

Any non-empty string is accepted as an HMAC-SHA256 key. A one-character key
(`NA0S_PICKLE_KEY="x"`), a whitespace-only-then-stripped value, or a trivially
guessable key all pass. HMAC with a low-entropy key is brute-forceable, so the
"attacker cannot forge the HMAC without the secret key" guarantee documented in
the module header (lines 14-16) and in ROADMAP_V2.md:1107 silently degrades to
near-zero. The function does **no length / emptiness-after-strip check** — the
only gate is "non-empty raw string".

**Defect 2 — `_parse_sidecar()` (lines 113-125):**

```python
def _parse_sidecar(raw):
    raw = raw.strip()
    if raw.startswith("v1:"):
        parts = raw.split(":", 2)
        if len(parts) == 3:
            return parts[2]
    # Legacy bare hex digest
    return raw
```

Whatever bytes are in the sidecar after the `v1:algo:` prefix (or the whole
stripped blob, in legacy mode) are returned **verbatim as the "expected
digest"** with no validation that it is a 64-char lowercase-hex SHA-256 /
HMAC-SHA256 string. Failure modes this admits:

- A malformed/truncated sidecar (`v1:sha256:` with empty digest, or
  `v1:sha256:zzzz`) is accepted as an "expected" value and only fails *later* at
  the `hmac.compare_digest(actual, expected)` mismatch (line 335) — a confusing
  "Integrity check failed" error that masks the real cause (corrupt sidecar, not
  a tampered pickle).
- A `v1:`-prefixed line with **fewer than 3 colon-parts** (e.g. `v1:sha256`
  missing the digest) silently falls through the `if` and returns the **entire
  `"v1:sha256"` string** as the digest (the `return raw` legacy branch), which is
  even more misleading.
- There is no place that ever asserts the parsed value matches
  `^[0-9a-f]{64}$`. Confirmed: grep for `fullmatch` / `isalnum` / a `64` literal
  in the file finds only `len(parts) == 3` (line 122), `len(header) < 2`
  (line 137) — **no digest-shape check anywhere** (verified).

Both `_sha256()` (line 62) and `_hmac_sha256()` (line 94) return
`h.hexdigest()`, which is **exactly 64 lowercase-hex chars** for SHA-256 and
HMAC-SHA256 (verified empirically:
`len(hashlib.sha256(b'x').hexdigest()) == 64`,
`len(hmac.new(b'k', b'x', hashlib.sha256).hexdigest()) == 64`). So a legitimate
expected digest is **always** 64 lowercase-hex chars — the validation is exact,
not a heuristic, and introduces **no false-positive risk** for genuine sidecars.

**Why this is in scope as supply-chain hardening (not gold-plating):** these are
the inputs to the tier-2 (HMAC) and tier-3 (SHA-256 sidecar) trust legs of the
3-tier hierarchy (header lines 7-22). `_parse_sidecar` parses an **on-disk file
an attacker with write access can edit**; `_get_signing_key` reads an
**operator-supplied secret** whose whole job is to be unforgeable. Validating
both is fail-fast input hygiene on the trust boundary, turning a deferred,
misleading "Integrity check failed" into an immediate, accurate
"malformed sidecar" / "weak key" error.

**Ideal state:**
- `_parse_sidecar` returns the digest only when it matches `^[0-9a-f]{64}$`
  (case-insensitive accept, normalize to lowercase since `hexdigest()` is
  lowercase); otherwise raises `ValueError("malformed sidecar ...")` so the
  caller fails fast with an accurate message. (Decision point in Step 3 on
  raise-vs-return; raise is correct — see 3.)
- `_get_signing_key` rejects a key whose stripped length is below a justified
  minimum, raising `ValueError` with actionable guidance; keeps returning `None`
  when the var is unset (so the keyless SHA-256 path is unaffected).

---

## 1. Explore: current vs ideal, gaps & edge cases  — APPLICABLE

**Current (gaps):**
- G1: `_parse_sidecar` accepts any string as a digest (no `^[0-9a-f]{64}$` shape
  check). Malformed sidecars surface as a misleading downstream compare-mismatch.
- G2: `_parse_sidecar` `v1:`-branch with `< 3` parts falls through to
  `return raw`, returning the prefix itself as a "digest".
- G3: `_get_signing_key` accepts arbitrarily weak keys (length 1, whitespace-y).

**Ideal:** both inputs validated at the boundary, fail-fast with accurate
messages; legitimate sidecars/keys unaffected (zero FP).

**Edge cases the plan/tests must cover:**
- E1: legit `v1:sha256:<64hex>` and `v1:hmac-sha256:<64hex>` → parse OK
  (happy-path, both algorithms).
- E2: legit **legacy bare** `<64hex>` (no `v1:` prefix) → parse OK (backward
  compat — `tests/integrity/test_l11_safe_pickle_fixes.py:115-117` and the
  round-trip at `tests/integrity/test_safe_pickle.py:147-160` MUST still pass).
- E3: uppercase / mixed-case 64-hex digest → accept, normalized to lowercase
  (some external tooling emits uppercase; `compare_digest` is byte-exact so
  normalization is required for the accept to be meaningful — but only accept,
  never silently "fix" a wrong-length value).
- E4: empty digest after prefix (`v1:sha256:`) → reject (`ValueError`).
- E5: short/long hex (`abcd`, 63 chars, 65 chars) → reject.
- E6: non-hex chars (`z`*64, embedded space, `v1:sha256:dead beef...`) → reject.
- E7: `v1:`-prefixed but only 2 parts (`v1:sha256`) → reject (currently silently
  returns `"v1:sha256"`).
- E8: whitespace around a valid digest (`  <64hex>\n`) → still accepted (the
  existing `raw.strip()` at line 119 + the parse path); regression-guard the
  `test_parse_versioned_with_whitespace` expectation
  (`test_l11_safe_pickle_fixes.py:119-120`).
- K1: unset `NA0S_PICKLE_KEY` → `None` (keyless SHA-256 path unchanged — must NOT
  raise; this is the documented backward-compatible fallback).
- K2: whitespace-only key (`"   "`) → reject (strips to empty → effectively no
  key but currently `.encode()`d as 3 bytes).
- K3: too-short key (below the justified minimum) → reject with guidance.
- K4: adequate-length key → accepted (round-trip dump/load still works — guards
  every `@patch.dict(os.environ, {"NA0S_PICKLE_KEY": "..."})` test in the suite).

**FP-safety note:** the suite's existing HMAC tests use short keys like
`"testsecret"` (10), `"atomickey"` (9), `"verkey"` (6), `"magickey"` (8),
`"auditkey"` (8), `"my_secret"` (9), `"newsecret"` (9). The chosen minimum MUST
NOT break these — see Step 3 threshold justification (the floor is set so all
existing legitimate test keys pass; only degenerate keys are rejected).

---

## 2. Roadmap / taxonomy / README / coverage cross-read  — APPLICABLE (partial)

- **ROADMAP_V2.md:1107** describes the 3-tier trust + "versioned sidecar format
  with backward-compatible parsing" — this item *hardens the parser* that line
  references; it strengthens the documented invariant rather than changing it.
- **ROADMAP_V2.md:1172** asserts the versioned sidecar format in the completed
  summary; adding shape-validation makes "the parser only ever yields a real
  digest" actually TRUE.
- **ROADMAP_V2.md:1177** ("Externalize integrity knobs into `config.py`",
  P3) explicitly names `NA0S_PICKLE_KEY` as a hardcoded env-var name and lists
  knob externalization. The min-key-length constant added here SHOULD be a
  module constant now and is a candidate for that future externalization — note
  the linkage but do NOT do the config.py move in this item (keep diff atomic;
  P3 is deferred).
- **ROADMAP_V2.md:1180** ("Stress cases for safe_pickle", P2) is a sibling
  test-coverage gap (truncated/large/concurrent). This item adds the
  *malformed-sidecar / weak-key* slice of that surface; cross-reference it but it
  is a distinct gap (do not claim it closes :1180 wholesale).
- Taxonomy / Coverage Matrix / per-attack scorer: see Step 10 (N/A) — loader
  input-validation, not a detection class with a TPR/FPR row.
- README: see Step 9 — `NA0S_PICKLE_KEY` is operator-facing config; if README/
  docs document a minimum, add one line. (Grep found NO README/docs mention of
  `NA0S_PICKLE_KEY` outside the module itself — so no doc contradiction to fix.)

---

## 3. Root-cause implementation plan (numbered)  — APPLICABLE

All edits in `src/na0s/integrity/safe_pickle.py`. Do NOT edit the shim
`src/na0s/safe_pickle.py` (CLAUDE.md: never add code to shim files).

**3a. Add a module-level digest validator + constant (near line 44, by
`_PROTO0_OPCODES`):**

```python
import re  # add to the stdlib import block (lines 27-34); currently absent
_HEX64_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")   # SHA-256 / HMAC-SHA256 = 64 hex
```

Justification for `64`: it is **not** an arbitrary threshold — it is the exact,
invariant hexdigest length of SHA-256 and HMAC-SHA256 (`h.hexdigest()` at lines
62 and 94), verified empirically. Accept `A-F` so externally-generated uppercase
digests parse; normalize to lowercase on return because `compare_digest`
(line 335) is byte-exact against `hexdigest()` output, which is lowercase.

**3b. Harden `_parse_sidecar` (lines 113-125)** to extract the candidate then
validate-or-raise:

```python
def _parse_sidecar(raw):
    """Parse a sidecar value, returning the validated lowercase hex digest.

    Accepts versioned (``v1:algo:digest``) and legacy bare-hex formats. Raises
    ValueError if the extracted value is not a 64-char hex SHA-256/HMAC digest.
    """
    raw = raw.strip()
    if raw.startswith("v1:"):
        parts = raw.split(":", 2)
        if len(parts) == 3:
            candidate = parts[2]
        else:
            raise ValueError(
                "Malformed sidecar: 'v1:' header without algo:digest body"
            )
    else:
        candidate = raw  # legacy bare hex
    if not _HEX64_RE.match(candidate):
        raise ValueError(
            "Malformed sidecar: expected a 64-char hex digest, got "
            "{!r} (len {})".format(candidate[:80], len(candidate))
        )
    return candidate.lower()
```

**Raise vs. return decision:** raise `ValueError`. Returning a sentinel would
push the failure to the `compare_digest` mismatch (line 335) which raises a
*different, misleading* "Integrity check failed ... File may be tampered"
message — wrong root cause for a corrupt-sidecar case. Raising here is also
**safe**: it happens strictly *before* `pickle.load` (the load path is
`_validate_pickle_magic` → `_resolve_expected_hash`(calls `_parse_sidecar`) →
compare → `pickle.load`, lines 304-363), so a malformed sidecar **fails closed**
(never unpickles). `ValueError` matches the family already raised on integrity
failure (line 346), so callers catching `ValueError` keep working.

Note the `[:80]` truncation in the message prevents a huge sidecar blob from
flooding logs — not a security threshold, just message hygiene.

**3c. Harden `_get_signing_key` (lines 82-85):**

```python
_MIN_PICKLE_KEY_LEN = 16  # bytes; justified below

def _get_signing_key():
    """Return the HMAC signing key from NA0S_PICKLE_KEY, or None if unset.

    Raises ValueError if the key is set but too short / whitespace-only.
    """
    key_str = os.getenv("NA0S_PICKLE_KEY")
    if key_str is None:
        return None
    stripped = key_str.strip()
    if not stripped:
        raise ValueError(
            "NA0S_PICKLE_KEY is set but empty/whitespace-only. Unset it to use "
            "the SHA-256 fallback, or set a key of at least {} chars.".format(
                _MIN_PICKLE_KEY_LEN
            )
        )
    if len(stripped) < _MIN_PICKLE_KEY_LEN:
        raise ValueError(
            "NA0S_PICKLE_KEY too weak ({} chars); require >= {} for HMAC "
            "strength.".format(len(stripped), _MIN_PICKLE_KEY_LEN)
        )
    return stripped.encode()
```

**Behavior-change note (deliberate):** the original returned `key_str.encode()`
(raw, un-stripped). The hardened version encodes the **stripped** value. This is
correct (a trailing newline in an env var should not be part of the secret) but
it is a **breaking change for any sidecar dumped with a trailing-whitespace
key**. Mitigation: this is acceptable because (a) no such artefact exists in the
repo/tests (all test keys are clean literals), (b) it only affects HMAC sidecars
re-loaded with a now-stripped key, and (c) the alternative (encode raw, but
length-check stripped) leaves the whitespace-in-key footgun. **Decision: strip
then encode.** Flag this explicitly in the PR body so a reviewer signs off.

**Threshold justification for `_MIN_PICKLE_KEY_LEN = 16`:** NOT arbitrary.
16 chars ≈ 96 bits for a random alphanumeric secret (log2(62)*16 ≈ 95 bits),
comfortably above the ~80-bit brute-force floor and aligned with common
"minimum secret length" guidance (e.g. 16-char minimum for HMAC keys). **BUT**
the existing suite uses shorter literal keys: `"verkey"` (6), `"magickey"`/
`"auditkey"` (8), `"atomickey"` (9), `"testsecret"`/`"my_secret"`/`"newsecret"`
(9-10). Setting the floor to 16 **would break ~15 existing tests**. Two options,
choose in implementation:
  - **Option A (preferred):** set `_MIN_PICKLE_KEY_LEN = 8` (≈48 bits — rejects
    only the truly degenerate 1-7 char keys and whitespace-only), which passes
    `"magickey"`/`"auditkey"`/`"atomickey"`/`"testsecret"` etc. but **still
    breaks** `"verkey"` (6). So the test using `"verkey"`
    (`test_l11_safe_pickle_fixes.py:122` `test_safe_dump_writes_versioned_hmac_sidecar`)
    must be updated to a >=8 literal. This is a legitimate test-fixture update
    (NOT weakening an assertion — the assertion is unchanged; only the fixture
    key is lengthened), allowed under CLAUDE.md.
  - **Option B:** keep 16 and bulk-update every short test key to a 16+ literal.
    More churn; same assertion semantics.
  **Recommendation: Option A with `_MIN_PICKLE_KEY_LEN = 8`**, and update the
  handful of <8 test keys (only `"verkey"`) to a clean >=8 literal. Document the
  8-char rationale in a code comment (rejects degenerate keys; a stronger 16+
  policy is deferred to the config.py externalization at ROADMAP_V2.md:1177).
  Whichever is chosen, the number MUST be a named constant with a comment, never
  a bare literal in the conditional (na0s-review-checklist: no magic thresholds).

**3d. Audit-log the rejection paths (optional, recommended):** emit a
`na0s.integrity_audit` warning JSON on weak-key / malformed-sidecar rejection,
mirroring the existing audit events (lines 280-287, 336-345), so operators see
*why* a load failed. Keep it lightweight; the raised `ValueError` is the primary
signal. Decide in review whether to include — not strictly required for
correctness.

**3e. Verify call-site compatibility:** `_parse_sidecar` is called at
`_resolve_expected_hash` lines 229 and 236; `_get_signing_key` at lines 259
(safe_dump), 307 (safe_load). A raised `ValueError` from either now propagates
out of `safe_dump`/`safe_load` — confirm no caller swallows it incorrectly
(callers in `predict.py:306-307,371,403`, `ml/predict_embedding.py:122,195`,
`dataset/hard_negatives.py:517-518` already let `safe_load` raise on integrity
failure; a weak-key/malformed-sidecar `ValueError` is the same failure family).
No call-site signature change.

---

## 4. Implement + wire (predict.py / cascade.py parity)  — APPLICABLE (no new pipeline wiring)

This is **input validation inside an existing, already-wired loader**, not a new
detector. `safe_load` is already wired across the runtime
(`predict.py:83,306-307,371,403`; `ml/predict_embedding.py:46,122,195`;
`dataset/hard_negatives.py:24,517-518`). No `_HAS_*` flag, no new
`predict.py`/`cascade.py` registration. The "wiring that matters" is that the
hardened `_parse_sidecar`/`_get_signing_key` sit on the **same code path** those
loaders already use — so the end-to-end assertion (Step 6 / Q4) is: a real model
with a valid sidecar still loads through `predict.py`'s `safe_load` calls, and a
malformed-sidecar / weak-key now fails fast there too. No cascade parity edit is
required because cascade reuses the same `safe_load` (no separate pickle path).

Agent: **layer-9-11-auditor** owns the edit (integrity/L11 territory), with
**silent-failure-hunter** verifying the new `raise` paths actually fail loud and
the removed permissiveness didn't mask anything. Inject `na0s-review-checklist`
(hallucinated APIs, import blindness — confirm `re` import added; arbitrary
thresholds — the `64` and key-min justification; FP-safety) into both prompts.

---

## 5. Harvester audit / harvested datasets  — N/A

N/A — this is on-disk sidecar / operator-secret input validation; there is no
external attack-string corpus to harvest. Test fixtures are synthesized inline
(crafted malformed sidecar strings + short/whitespace key env values in a tmp
dir). Crafted-malicious-pickle *datasets* are scoped to item 8, not item 9.

---

## 6. Tests: Code + Use-Case  — APPLICABLE

Land tests in the existing integrity test files (CLAUDE.md: tests mirror source
sub-package; these already exist under `tests/integrity/`):
- `_parse_sidecar` unit cases → extend the `TestSidecarVersioning` class in
  `tests/integrity/test_l11_safe_pickle_fixes.py` (it already imports
  `_parse_sidecar` at line 31 and tests parse/versioning at lines 104-160).
- key + end-to-end cases → extend `tests/integrity/test_safe_pickle.py`
  (`TestHelpers` already imports `_get_signing_key` at line 22; add a
  `TestKeyStrength` class and reuse the `TestTamperingDetection` /
  `TestBackwardCompatibility` fixture style). Use `unittest` + `unittest.mock`,
  no network.

**A. Code-level (`_parse_sidecar`):**
- `test_parse_valid_versioned_sha256` (E1): `_parse_sidecar("v1:sha256:" + "a"*64)
  == "a"*64`.
- `test_parse_valid_versioned_hmac` (E1): `v1:hmac-sha256:<64hex>` → the 64-hex.
- `test_parse_valid_legacy_bare` (E2): `_parse_sidecar("b"*64) == "b"*64`
  (backward compat — must NOT raise).
- `test_parse_uppercase_normalized` (E3): `_parse_sidecar(("A"*64)) == "a"*64`
  (accept + lowercase).
- `test_parse_empty_digest_raises` (E4): `v1:sha256:` →
  `pytest.raises(ValueError, match="64-char hex")`.
- `test_parse_short_hex_raises` (E5): `"abcd"` and `"a"*63` → ValueError.
- `test_parse_long_hex_raises` (E5): `"a"*65` → ValueError.
- `test_parse_nonhex_raises` (E6): `"z"*64` and `"a"*63 + " "` → ValueError.
- `test_parse_v1_two_parts_raises` (E7): `"v1:sha256"` →
  `pytest.raises(ValueError, match="without algo:digest")` (regression for the
  old silent `return "v1:sha256"`).
- `test_parse_whitespace_preserved` (E8): `"  " + "a"*64 + "\n"` → `"a"*64`
  (keeps the existing strip behavior;
  `test_l11_safe_pickle_fixes.py:119` analog must still hold).
- **Regression guard:** the existing `test_parse_legacy_bare_hex` (line 115, key
  `"a"*64`) and `test_parse_versioned_sidecar` (line 112) MUST still pass
  unchanged — note that the existing `test_parse_versioned_sidecar` uses
  `"v1:sha256:abcdef0123456789"` (16 hex, NOT 64) → **this test will now
  correctly fail** because the digest is too short. Update its fixture to a full
  64-hex string (assertion semantics unchanged — it asserts the digest is
  returned; only the fixture length is corrected to be a *valid* digest). This
  is a legitimate fixture fix, not assertion weakening; flag it in the PR.

**B. Code-level (`_get_signing_key`):**
- `test_key_unset_returns_none` (K1): `NA0S_PICKLE_KEY` absent → `None`
  (re-asserts existing `test_get_signing_key_none_without_env`, line 45).
- `test_key_whitespace_only_raises` (K2): `"   "` →
  `pytest.raises(ValueError, match="empty/whitespace-only")`.
- `test_key_too_short_raises` (K3): `"x" * (MIN-1)` →
  `pytest.raises(ValueError, match="too weak")`.
- `test_key_adequate_accepted` (K4): `"x" * MIN` → returns `b"x"*MIN`
  (and a stripped key `"  abcdefgh  "` returns `b"abcdefgh"` per the strip
  decision in 3c).
- `test_key_min_boundary`: exactly `MIN` chars accepted, `MIN-1` rejected
  (boundary, not magic — pins the documented floor).

**C. Use-Case / end-to-end (loader contract — Q4):**
- `test_load_malformed_hmac_sidecar_fails_fast` (E4-E7 via real load): dump with
  a valid key, then overwrite the `.hmac` sidecar with `"v1:hmac-sha256:zzzz"`;
  `safe_load` raises `ValueError` whose message names the **malformed sidecar**
  (proves it fails at parse, not at the later compare → distinct from the
  existing `test_tampered_hmac_sidecar_detected` at line 154 which writes a
  *valid-shape* `"0"*64` and correctly hits the compare-mismatch path). Keep BOTH
  tests — they exercise the two different failure points.
- `test_load_legit_still_roundtrips` (regression, K4/E1): dump+load with an
  adequate key returns the original object (guards the happy path through the
  real loader; mirrors `test_hmac_load_succeeds_with_correct_key`, line 77).
- `test_dump_with_weak_key_raises` (K3 on the dump side): `NA0S_PICKLE_KEY="x"`
  then `safe_dump(obj, path)` → `ValueError` from `_get_signing_key` at line 259
  (proves the weak key is rejected at write time too, not just load).
- `test_keyless_dump_still_works` (K1 regression): unset key → SHA-256 sidecar
  written, no raise (the documented fallback is untouched; mirrors
  `test_sha256_dump_creates_sha256_sidecar`, line 86).

**Assertion discipline (anti-hollow):** every test asserts a concrete value,
exact exception type, AND a message substring (`match=`). No `assertTrue(True)`,
no smoke-only cases. The only numbers are the digest length 64 (an invariant,
not a tunable) and `_MIN_PICKLE_KEY_LEN` (a named, comment-justified constant
referenced via import in the boundary test — not a hardcoded magic literal in
the test).

**Smoke step (CLI / import):** run the whole-package import smoke
`PYTHONPATH=<worktree>/src python3 -c "import na0s; from na0s.integrity.safe_pickle import _parse_sidecar, _get_signing_key; print('ok')"`
(proves the new `re` import + module loads), then the targeted suite
`python3 -m pytest tests/integrity/ -v`. (The checklist's mocked-CLI-gap guard:
since safe_pickle has no standalone CLI, the import smoke is the equivalent
load-time proof that the edit doesn't break module import.)

Agents: **layer-9-11-auditor** authors the integrity tests;
**silent-failure-hunter** reviews that K2/K3/E4-E7 fail loud (no swallowed
raise). Skill: **na0s-debugging** for any import-cache / `patch.dict` env-leak
trap under the full ~8000-test run.

---

## 7. Cleanup / refactor per conventions  — APPLICABLE (light)

- Add `import re` to the existing stdlib import block (lines 27-34), alphabetical
  position (after `pickle`, before `stat`). Confirm it's actually used (it is, in
  `_HEX64_RE`) — no unused-import smell.
- Keep the two new constants (`_HEX64_RE`, `_MIN_PICKLE_KEY_LEN`) module-level
  with comments; they are the future targets of the ROADMAP_V2.md:1177 config.py
  externalization — leave a `# TODO(P3): externalize via config.py (ROADMAP L11)`
  next to `_MIN_PICKLE_KEY_LEN` so the linkage is discoverable but DON'T do the
  move here (atomic diff).
- No file moves, no new module (edit-in-place in the canonical
  `integrity/safe_pickle.py`). Do NOT touch the shim.
- Tests go in the two existing `tests/integrity/` files (no new flat test file
  needed; the parse/key surfaces already have homes there).

---

## 8. Roadmap update (cite SHA on completion)  — APPLICABLE

- Add a checked entry under the L11 TODO section of ROADMAP_V2.md (near :1177-
  1180): "H2 safe_pickle — `_parse_sidecar` now validates `^[0-9a-f]{64}$`
  (raises on malformed); `_get_signing_key` rejects empty/whitespace/too-short
  `NA0S_PICKLE_KEY` (min `_MIN_PICKLE_KEY_LEN`). (SHA: <fill at commit>)".
- Note in the same line that this is a *partial* down-payment on the :1180 stress-
  case gap (malformed-sidecar slice) and links to the :1177 config externalization
  (the new min-len constant is a future externalization target) — do not mark
  either parent item fully done.
- Per the Roadmap-Todo Sync memory: the todo + its check-off both live in
  ROADMAP_V2.md; cite the commit SHA when pushed.

---

## 9. README / Benchmark updates  — APPLICABLE (minimal, conditional)

- BENCHMARK: N/A — no detection metric / benchmark number changes (input
  validation on a loader path; zero impact on TPR/FPR/recall).
- README/docs: grep found **no** existing mention of `NA0S_PICKLE_KEY` outside
  the module, so there is no doc to contradict. IF a security/configuration doc
  enumerates `NA0S_PICKLE_KEY`, add a one-line "minimum N characters; unset to
  use the SHA-256 fallback" note. If no such doc exists, skip (do not create a
  doc file just for this — CLAUDE.md: don't proactively create docs). The module
  docstring (header lines 14-16) already describes the key; optionally append the
  min-length requirement there for in-source discoverability.

---

## 10. Taxonomy + Coverage Matrix + per-feature thresholds  — N/A

N/A — `_parse_sidecar`/`_get_signing_key` are loader input-validation helpers,
not a detector with a taxonomy code or a COVERAGE_MATRIX TPR/FPR row, and there
is no per-attack scorer threshold. The only number introduced is the 64-hex
digest length (a cryptographic invariant of SHA-256/HMAC-SHA256, not a tunable
detection threshold) and `_MIN_PICKLE_KEY_LEN` (a security-policy constant,
justified in Step 3, not a scorer threshold). No taxonomy/coverage row to add or
reconcile.

---

## 11. PR + held-out test gate  — APPLICABLE

- Branch: `hardening/sidecar-key-validation` off `main` (the rename is DONE on
  main; `na0s.integrity.safe_pickle` exists there — verified). Work in a git
  worktree per multi-agent discipline; do NOT branch-switch the primary checkout.
- PR body MUST call out the two **deliberate behavior changes** for reviewer
  sign-off: (a) `_parse_sidecar` now RAISES on malformed input (was silent
  return); (b) `_get_signing_key` strips-then-encodes and rejects short/empty
  keys (was raw-encode of any non-empty value), plus the two fixture corrections
  (`test_parse_versioned_sidecar` 16→64 hex; `verkey` 6→8 chars). Frame both as
  fixture/length corrections, NOT assertion weakening.
- **Gate:** targeted first — `python3 -m pytest tests/integrity/ -v` (green incl.
  the extended classes) — then the FULL suite
  `python3 -m pytest tests/ -q --tb=line` with zero net regressions (CLAUDE.md;
  ~8000 tests, ~15 min). Pay special attention to the ~15 existing tests that
  set short `NA0S_PICKLE_KEY` literals across the suite — confirm the chosen
  `_MIN_PICKLE_KEY_LEN` (8) passes all of them except the single `verkey` fixture
  that gets lengthened. Verify against MAIN-equivalent env
  (`PYTHONPATH=<worktree>/src`) per the na0s-debugging stale-editable-install trap.
- Skill **github-pr-prep** to assemble the PR; **github-ci-fix** only if CI goes
  red. Review via **pr-review-toolkit:review-pr** / **github-pr-review**; inject
  `na0s-review-checklist` into the reviewer prompt.

---

## Q&A self-check

- **Q1 — Can Na0S handle the target (bug) + suite green?**
  After fix: yes. Malformed sidecars and weak keys fail fast at the trust
  boundary with accurate `ValueError`s (fail-closed, before `pickle.load`);
  legitimate sidecars/keys are unaffected. Suite must stay green (Step 11 gate),
  with the two documented fixture corrections.
- **Q2 — Cleanup done?** Yes — edit-in-place in canonical module, `re` import
  added cleanly, constants commented + linked to the deferred config.py
  externalization; no new files, shim untouched, tests in correct sub-dir.
- **Q3 — Pipeline wiring correct?** No new wiring needed — the hardened helpers
  sit on the already-wired `safe_load`/`safe_dump` path used by `predict.py` and
  the embedding loader. End-to-end load behavior is asserted in Step 6.C.
- **Q4 — Tested for code AND use-case?** Yes — code-level (`_parse_sidecar` 10
  cases incl. boundary + the v1-two-parts regression; `_get_signing_key` 5 cases
  incl. boundary) + use-case (malformed-sidecar fail-fast through real
  `safe_load`, weak-key reject on dump, legit roundtrip, keyless fallback intact)
  + import smoke.
- **Q5 — Harvester audit?** N/A — loader input validation; no harvested dataset.
- **Q6 — Taxonomy + Coverage Matrix?** N/A — not a detector row.
- **Q7 — Scorer thresholds?** N/A — the 64-hex length is a crypto invariant and
  `_MIN_PICKLE_KEY_LEN` is a justified security constant, not a scorer threshold.
- **Q8 — predict.py / cascade.py references to target?**
  APPLICABLE: `predict.py` calls `safe_load` at lines 306-307, 371, 403 (import
  at line 83) — these are on the hardened code path, so the change DOES affect
  predict.py behavior (a malformed sidecar / weak key now fails there). No
  separate cascade pickle path exists (cascade reuses `safe_load`), so no parity
  edit is required — but Step 6.C asserts the end-to-end load contract through
  this surface.
- **Q9 — Harvester harvests this type?** N/A — not an intel/attack type.
- **Q10 — Other correctness checks:**
  (a) confirm `re` import added and used (no unused-import / no hallucinated
  stdlib); (b) confirm both new `raise`s land BEFORE `pickle.load` (fail-closed —
  load path lines 304-363); (c) confirm `_MIN_PICKLE_KEY_LEN` is a named constant
  with a comment, never a bare literal (no-magic-threshold rule); (d) audit the
  ~15 existing short-key tests so the floor doesn't silently break them — only
  `verkey` (6) needs lengthening; (e) confirm `patch.dict(os.environ, ...)` in
  the new key tests fully restores env so it can't poison the full-suite run
  (env-leak trap); (f) confirm the strip-then-encode change has no
  trailing-whitespace-keyed artefact in the repo (none — all test keys clean).

---

## Agent / skill team (per step)

| Step / area | Owner agent | Reviewer / support | Skills (inject na0s-review-checklist into every prompt) |
|---|---|---|---|
| 0-2 root-cause confirm | layer-9-11-auditor | security-research-auditor | security-review |
| 3-4 implement | layer-9-11-auditor | silent-failure-hunter | na0s-debugging |
| 6 tests | layer-9-11-auditor | silent-failure-hunter | na0s-debugging, eval-harness (suite gate only) |
| 7 cleanup | layer-9-11-auditor | l3-l5-code-auditor (loader-adjacency) | — |
| 8 roadmap | (author) | — | — |
| 9 README (conditional) | (author) | — | — |
| 11 PR + CI | github-pr-prep | github-pr-review / pr-review-toolkit:review-pr | github-ci-fix |

`layer-9-11-auditor` owns this because `safe_pickle` lives in the integrity
(L10/L11) supply-chain subsystem. `silent-failure-hunter` is paired throughout
because the core of this item is converting silent permissiveness into loud,
accurate failures — exactly its remit.

---

## Execution preconditions / dependencies

- **Depends-on: none.** Self-contained. `na0s.integrity.safe_pickle` already
  exists on `main` (verified — `predict.py:83` and 5 other call sites import it),
  so the target module is editable immediately with no prerequisite item.
- **Not blocked by item 8** (crafted-malicious-pickle datasets): item 8 is a
  *detection* dataset concern; item 9 is loader *input validation*. They touch
  the same subsystem but are independent — this item's tests synthesize their own
  malformed-sidecar / weak-key fixtures. Either order is fine.
- **Coordinate (not block) with items 1/2/5/6** (the sibling supply-chain
  hardening specs in this dir, all touching `safe_pickle`/loaders): if any lands
  first and shifts line numbers in `integrity/safe_pickle.py`, re-confirm the
  82-85 / 113-125 ranges before editing. No logical dependency, only line-drift
  awareness — they edit different functions.
- Must be done in a git worktree off `main`; verify env with
  `PYTHONPATH=<worktree>/src` (editable install may be stale per memory).

---

## Definition of done

- [ ] `import re` added to `integrity/safe_pickle.py` stdlib import block (used by
      `_HEX64_RE`).
- [ ] `_HEX64_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")` added with a comment
      explaining 64 = SHA-256/HMAC-SHA256 hexdigest invariant.
- [ ] `_parse_sidecar` validates the extracted candidate against `_HEX64_RE`,
      raises `ValueError("Malformed sidecar ...")` on failure (incl. the
      `v1:`-with-<3-parts case), returns lowercase digest on success.
- [ ] `_MIN_PICKLE_KEY_LEN` named constant (= 8, comment-justified; with a
      `TODO(P3)` linking ROADMAP_V2.md:1177 config externalization).
- [ ] `_get_signing_key` returns `None` when unset, raises `ValueError` on
      empty/whitespace-only and on `< _MIN_PICKLE_KEY_LEN`, strips-then-encodes
      an adequate key.
- [ ] Both new `raise`s verified to execute BEFORE `pickle.load` (fail-closed).
- [ ] `tests/integrity/test_l11_safe_pickle_fixes.py` extended: `_parse_sidecar`
      valid (versioned sha256/hmac, legacy bare, uppercase-normalized) + reject
      (empty, short, long, non-hex, v1-two-parts) + whitespace-preserved; the
      existing `test_parse_versioned_sidecar` fixture corrected 16→64 hex.
- [ ] `tests/integrity/test_safe_pickle.py` extended: `_get_signing_key` unset/
      whitespace/too-short/adequate/boundary + end-to-end (malformed-sidecar
      fail-fast through `safe_load`, weak-key reject on `safe_dump`, legit
      roundtrip, keyless fallback intact).
- [ ] The single short test key `verkey` (6) lengthened to >=8 (fixture fix, not
      assertion weakening); all other existing short keys (>=8) confirmed passing.
- [ ] All new tests assert concrete value + exact exception type + message
      substring (no hollow assertions); env fully restored after each `patch.dict`.
- [ ] Import smoke green: `python3 -c "from na0s.integrity.safe_pickle import
      _parse_sidecar, _get_signing_key"`.
- [ ] `python3 -m pytest tests/integrity/ -v` green.
- [ ] Full suite `python3 -m pytest tests/ -q --tb=line` — zero net regressions
      vs main-equivalent baseline.
- [ ] ROADMAP_V2.md L11 H2 line added + checked off with commit SHA; noted as
      partial down-payment on :1180 and linked to :1177.
- [ ] PR opened off `main` via worktree; the two deliberate behavior changes +
      two fixture corrections called out for reviewer sign-off; CI green before
      merge.
