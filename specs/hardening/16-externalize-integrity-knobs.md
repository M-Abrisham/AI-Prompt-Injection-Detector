---
item: 16
title: "L11 — externalize integrity knobs (chunk size / YAML max / backup retention / HMAC key env name) → config.py"
priority_tier: P3
effort: Trivial
depends_on:
  - "soft: roadmap ROADMAP_V2.md:1184 — move scripts/safe_yaml.py → src/na0s/integrity/safe_yaml.py (P2). Required ONLY for the YAML-max knob to be importable from na0s.config; the other three knobs have no dependency."
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 9]
na_steps: [5, 10, 11_dataset_parts]
applicable_qa: [Q1, Q2, Q3, Q4, Q8, Q10]
na_qa: [Q5, Q6, Q7, Q9]
status: PLAN-ONLY
---

# Item 16 — Externalize L11 Integrity Knobs into `config.py`

> **Plan-only spec.** The only write produced by executing this plan is source/test
> edits at implementation time; this document itself only *describes* them. No source
> or test files are modified while authoring this spec.

## 0. Root cause (confirmed against live source, 2026-06-22)

Four Layer-11 (supply-chain / integrity) tuning knobs are hardcoded as module-local
literals instead of living in the central `src/na0s/config.py`, which the v1.0.0 tree
plan (ROADMAP_V2.md:163-167) designates as the single home for "externalized thresholds
pulled in from every layer … after P3 polish tasks consolidate hardcoded values from
L4/L5/L6/L7/L8/L10/**L11**". The roadmap TODO is ROADMAP_V2.md:1177.

Confirmed locations (line numbers verified, not guessed):

| # | Knob | Current value | Hardcoded at | Notes |
|---|------|---------------|--------------|-------|
| K1 | SHA-256 / HMAC read chunk size | `1 << 16` (64 KB) | `src/na0s/integrity/safe_pickle.py:60` (`_sha256`) and `:92` (`_hmac_sha256`) | Two duplicated literals in one file. |
| K2 | YAML max file size (billion-laughs DoS cap) | `10 * 1024 * 1024` (10 MB) | `scripts/safe_yaml.py:26` (`_DEFAULT_MAX_SIZE`) | **Lives in `scripts/`, NOT in the `na0s` package** — see dependency wrinkle below. |
| K3 | Backup retention default | `keep: int = 3` | `src/na0s/integrity/model_rollback.py:137` (`ModelRollback.cleanup`) | Default arg value, no env override today. |
| K4 | HMAC signing-key env-var NAME | `"NA0S_PICKLE_KEY"` | `src/na0s/integrity/safe_pickle.py:84` (`_get_signing_key`), plus message string literals at `:269-270`, `:315`, `:330`, and docstring `:15` | The *name of the env var* is hardcoded; the value is correctly read from the environment already. |

Root-cause statement: **these four L11 integrity tuning values are scattered module-level
literals rather than centralized in `config.py`, violating the v1.0.0 "one config home"
convention (ROADMAP_V2.md:163-167); the YAML-cap knob additionally sits in non-package
`scripts/safe_yaml.py`, so it cannot be imported from `na0s.config` until that file is
promoted into `na0s.integrity` (parallel P2 task, ROADMAP_V2.md:1184).**

### Adjacent inconsistency surfaced (in-scope to note, optional to fix)

Four *other* integrity hashers use an **8192-byte** chunk, not 64 KB:
`fingerprint.py:22`, `sbom.py:20`, `model_provenance.py:26`, `req.py:22`. If we are
centralizing a chunk-size constant, these are the natural second consumer. Decision in
§7: introduce ONE `INTEGRITY_HASH_CHUNK_BYTES` constant and migrate `safe_pickle.py`
(the roadmap-named target) now; sweeping the four 8192 sites is a clearly-labelled
optional follow-up so we don't silently change their I/O behavior under a P3 ticket.

### Why this is low-risk (and where the trap is)

`safe_pickle._sha256` / `_hmac_sha256` feed `hashlib`/`hmac` incrementally, so the chunk
size is **purely an I/O batching choice — it does NOT change the resulting digest**. K1 is
behavior-preserving by construction. K3 only changes a *default* (every real caller in the
repo passes `keep=` explicitly — `tests/integrity/test_l11_encryption_rollback.py:312,320`).
K4 is a pure rename-with-indirection. K2 is the only one with a structural wrinkle (package
boundary). The trap to avoid: do **not** turn a behavior-preserving cleanup into a behavior
change by also altering the 8192 sites or the 10 MB value without justification.

---

## 1. Step 1 — Explore current system vs. ideal; gaps & edge cases

**Current (verified):**
- `config.py` (66 lines) already externalizes L0/L4/L6/L7/L9/cascade values and has a
  precedent for an env-backed scalar: `MAX_INPUT_LENGTH = int(os.getenv("NA0S_MAX_INPUT_LENGTH", 50_000))`
  (`src/na0s/config.py:11`).
- A crash-proof env helper already exists: `src/na0s/_env.py` → `safe_int_env(name, default, lo, hi)`
  and `safe_float_env(...)`. This is the *correct* idiom to use (clamps + falls back on
  garbage), strictly better than the bare `int(os.getenv(...))` at config.py:11.
- The reload-driven test pattern is established in `tests/test_l0_config.py:1-40`
  (`importlib.reload` to re-execute module-level env reads, default/override/invalid cases).

**Ideal:** All four knobs are named constants in `config.py` (env-overridable via `_env`
helpers where an override makes sense), and every L11 site imports them instead of inlining
literals — matching how L0/L4/L6 already consume `config.py`.

**Gaps:** K1 (2 literals), K2 (1 literal, wrong package), K3 (1 default arg), K4 (1 env
name + 4 message strings). No detection-behavior gap — this is hygiene/consistency.

**Edge cases to honor in the plan:**
- Reload-safety: config constants are read at import; tests must `importlib.reload` (per
  test_l0_config) rather than assume late binding.
- K2 package boundary: `na0s.config` cannot be imported by `scripts/safe_yaml.py` without
  creating a `scripts → na0s` dependency that doesn't exist today, and `na0s.config` cannot
  import from `scripts/`. Resolution options in §3.
- K1 chunk size must stay > 0 and reasonably bounded (clamp `lo=1`); a 0/negative value
  would make `iter(lambda: f.read(n), b"")` spin forever — so use `safe_int_env` with
  `lo=4096` to keep performance sane and reject 0.
- K3 `keep` must be `>= 0`; `keep=0` legitimately means "delete all backups" — clamp `lo=0`.

## 2. Step 2 — Roadmap / taxonomy / README / coverage cross-read

- ROADMAP_V2.md:1177 is the literal TODO for this item; ROADMAP_V2.md:163-167 is the
  config-consolidation design intent; ROADMAP_V2.md:1184 is the dependent P2 `safe_yaml`
  move. ROADMAP_V2.md:1172 ("Completed (24 items)") documents the 64 KB / 10 MB values as
  shipped behavior — the spec must preserve these defaults verbatim so that line stays true.
- README / Coverage Matrix: N/A — see Step 10. No attack-class row changes.
- Taxonomy: N/A — no attack code involved.

## 3. Step 3 — Root-cause implementation plan (numbered)

1. **Add an L11 config block to `src/na0s/config.py`** (after the existing L9 block, ~line 65),
   importing the existing helper: `from na0s._env import safe_int_env`. Define:
   - `INTEGRITY_HASH_CHUNK_BYTES: int = safe_int_env("NA0S_INTEGRITY_HASH_CHUNK_BYTES", 1 << 16, lo=4096, hi=1 << 24)` — default **65536 (64 KB)**, unchanged.
   - `INTEGRITY_YAML_MAX_BYTES: int = safe_int_env("NA0S_INTEGRITY_YAML_MAX_BYTES", 10 * 1024 * 1024, lo=0, hi=1 << 31)` — default **10 MB**, unchanged; `lo=0` preserves the documented "0 disables the check" contract (`scripts/safe_yaml.py:35`).
   - `INTEGRITY_BACKUP_RETENTION: int = safe_int_env("NA0S_INTEGRITY_BACKUP_RETENTION", 3, lo=0, hi=10_000)` — default **3**, unchanged.
   - `PICKLE_SIGNING_KEY_ENV: str = "NA0S_PICKLE_KEY"` — plain string constant (the *name*, not the secret; not env-overridable — renaming the var that holds the renamed var is circular and pointless).
   - Justify-every-number note for the reviewer: 64 KB and 10 MB and 3 are **carried over verbatim** from the existing code, not invented. The clamp bounds (`lo=4096`, `hi=1<<24` for chunk; `hi=1<<31` for YAML; `hi=10_000` for retention) are guardrails, not thresholds, and are documented inline as "reject pathological env values."

2. **K1 — wire `safe_pickle.py`:** add `from na0s.config import INTEGRITY_HASH_CHUNK_BYTES`
   at the top, replace `1 << 16` at `:60` and `:92` with `INTEGRITY_HASH_CHUNK_BYTES`.
   Behavior-preserving (digest identical regardless of chunk size).

3. **K4 — wire `safe_pickle.py`:** add `from na0s.config import PICKLE_SIGNING_KEY_ENV`;
   change `_get_signing_key` `:84` to `os.getenv(PICKLE_SIGNING_KEY_ENV, "")`. The four
   human-readable message/docstring strings (`:15`, `:269-270`, `:315`, `:330`) should
   interpolate `PICKLE_SIGNING_KEY_ENV` (e.g. `f"{PICKLE_SIGNING_KEY_ENV} is not set..."`)
   so the env-var name appears in exactly one place. Keep wording otherwise identical so
   `tests/integrity/test_safe_pickle.py:113` (`assertIn("NA0S_PICKLE_KEY is not set", ...)`)
   still passes — the interpolated value IS `"NA0S_PICKLE_KEY"`, so the substring assertion
   holds. Verify this explicitly (it's a real regression risk if wording drifts).

4. **K3 — wire `model_rollback.py`:** import `INTEGRITY_BACKUP_RETENTION`; change
   `cleanup(self, model_name, keep: int = 3)` `:137` to `keep: int | None = None`, and at
   the top of the body do `if keep is None: keep = INTEGRITY_BACKUP_RETENTION`. Using a
   `None` sentinel (not `keep=INTEGRITY_BACKUP_RETENTION` as the default) keeps the default
   late-bound so a reloaded/overridden config takes effect, matching the reload-test idiom.
   Existing explicit callers (`keep=2`, `keep=5`) are unaffected.

5. **K2 — wire `safe_yaml`:** two sub-options; the spec RECOMMENDS option (b) and lists (a)
   as the no-dependency fallback:
   - **(a) No-dependency fallback (if the P2 move has NOT landed):** keep
     `scripts/safe_yaml.py` as-is but change `_DEFAULT_MAX_SIZE` to read the env var directly
     using the same name: `_DEFAULT_MAX_SIZE = int(os.getenv("NA0S_INTEGRITY_YAML_MAX_BYTES", 10 * 1024 * 1024))`.
     This externalizes the knob to the *same env var* as config.py without crossing the
     package boundary. Document in a comment that the canonical constant is
     `na0s.config.INTEGRITY_YAML_MAX_BYTES` and they must stay in sync.
   - **(b) Preferred (after / together with ROADMAP_V2.md:1184):** move the file to
     `src/na0s/integrity/safe_yaml.py`, leave a thin re-export shim at `scripts/safe_yaml.py`
     (`from na0s.integrity.safe_yaml import safe_load_yaml`), and have the moved module do
     `from na0s.config import INTEGRITY_YAML_MAX_BYTES` for its default. This is the clean
     end state and removes the duplication from (a). All five current importers
     (`scripts/sync_datasets.py:24`, `scripts/quarantine.py:46`, `scripts/license_checker.py:20`,
     `scripts/generate_taxonomy_samples.py:31`, plus `data/_base.py` per roadmap) keep working
     through the shim.
   - **Execution rule:** if executing item 16 standalone, ship (a) and leave a roadmap note;
     if the P2 `safe_yaml` move is being done in the same sprint, do (b) and skip (a).

6. **Optional, clearly-flagged follow-up (do NOT bundle silently):** migrate the four
   8192-byte hashers (`fingerprint.py:22`, `sbom.py:20`, `model_provenance.py:26`,
   `req.py:22`) to `INTEGRITY_HASH_CHUNK_BYTES`. This *changes* their chunk from 8 KB → 64 KB
   (still digest-identical, only I/O batching). Gate behind reviewer approval; if approved,
   one extra commit, otherwise leave them and note the residual inconsistency in the roadmap.

## 4. Step 4 — Pipeline wiring (predict.py / cascade.py parity)

**Applicable but narrow.** predict.py and cascade.py already `from na0s import config`
(verified: both appear in the config-importer grep). They do **not** reference any of the four
L11 knobs (these are model-load / YAML-load / backup utilities, not part of the per-text scan
hot path). So:
- No new wiring into the scan pipeline is required.
- Parity check = confirm that adding constants to `config.py` does not shadow or collide with
  names predict.py/cascade.py already import from config (grep the new constant names against
  both files; they're new names, so no collision — verify at implementation time).
- `safe_load` is exercised indirectly when the ML classifier model is loaded during a real
  `scan()`; the smoke test in §6 covers that path end-to-end.

## 5. Step 5 — Harvester / dataset audit

**N/A — this is supply-chain/integrity hygiene, not an attack class; there is no dataset to
harvest, decontaminate, or train on for "externalizing constants."**

## 6. Step 6 — Tests: Code + behavior (use-case reframed as integrity/loader behavior)

New test module: `tests/integrity/test_config_knobs.py` (mirrors source package; joins the
existing `tests/integrity/` dir). Pattern follows `tests/test_l0_config.py` (importlib.reload).

**A. Config-constant tests (code):**
- `INTEGRITY_HASH_CHUNK_BYTES` default == 65536; env override `NA0S_INTEGRITY_HASH_CHUNK_BYTES=131072` takes effect after reload; garbage (`"abc"`) and `0` fall back to default (proves the `safe_int_env` clamp, not a hollow `==` of two literals).
- `INTEGRITY_YAML_MAX_BYTES` default == 10*1024*1024; env override + invalid fallback.
- `INTEGRITY_BACKUP_RETENTION` default == 3; env override `=5`; negative `=-1` falls back to 3.
- `PICKLE_SIGNING_KEY_ENV` == "NA0S_PICKLE_KEY" (guards against accidental rename breaking downstream docs/tests).

**B. Behavior / use-case tests (the load actually works + tamper still rejected):**
- *Legit round-trip still loads* (K1 wired): `safe_dump(obj, p)` then `safe_load(p)` returns
  the object byte-for-byte, with `NA0S_INTEGRITY_HASH_CHUNK_BYTES` set to a small value
  (e.g. 4096) AND a large value (e.g. 1<<20) — proves the digest is independent of chunk size
  (the core correctness claim of K1). This is the anti-hollow assertion: same digest under
  different chunk sizes.
- *Tampered file still rejected* (K1 path intact): flip a byte after dump, assert
  `safe_load` raises `ValueError` with "Integrity check failed" — under both a non-default
  chunk size and the default, so we know the rename didn't break verification.
- *K4 indirection*: with `PICKLE_SIGNING_KEY_ENV`'s value set in the env, `safe_dump` writes
  a `.hmac` sidecar (not `.sha256`); the "is not set" `UserWarning` message still contains the
  literal `"NA0S_PICKLE_KEY"` (guards the wording-drift regression flagged in §3.3).
- *K3 default sourced from config*: build a `ModelRollback`, create N>3 backups, call
  `cleanup(model_name)` with no `keep` arg, assert it keeps `INTEGRITY_BACKUP_RETENTION` (3);
  then reload config with `NA0S_INTEGRITY_BACKUP_RETENTION=1` and assert it keeps 1 — proving
  the default is late-bound to config, not frozen at import.
- *K2 cap honored*: `safe_load_yaml` on a file just over the (overridden, small) cap raises
  `ValueError("too large")`; on a file under it, parses normally; `max_size_bytes=0` disables
  the check (preserves the documented contract).

**C. CLI / suite smoke (mandatory per checklist):**
- Targeted first: `python3 -m pytest tests/integrity/ -q --tb=line`.
- Real loader smoke (no mocks): run a one-line scan that forces model load, e.g.
  `python3 -c "import na0s; print(na0s.scan('ignore previous instructions').label)"`
  (confirm the exact public entrypoint name at implementation time via `na0s/__init__.py`
  `__all__`) — proves `safe_load` still loads the shipped model after the K1/K4 edits.
- Full suite last: `python3 -m pytest tests/ -q --tb=line`, expect zero net regressions
  (baseline noise per MEMORY: ~15 env-only failures that reproduce on `main`; compare, don't
  assume green).

## 7. Step 7 — Cleanup / refactor per conventions

- New constants land in the **existing** `config.py` (no new top-level file) — conforms to
  CLAUDE.md "core pipeline files stay top level."
- New test goes in **`tests/integrity/`** (mirrors `src/na0s/integrity/`), not a flat root
  file — conforms to test-org rule.
- Collapse K1's two duplicated `1 << 16` literals into one named constant — net de-duplication.
- Collapse K4's five copies of the literal `"NA0S_PICKLE_KEY"` into one constant.
- Do NOT add code to shim files (`src/na0s/model_rollback.py` is a shim — edits go to
  `src/na0s/integrity/model_rollback.py` only).
- Branch name: `hardening/externalize-integrity-knobs` (per CLAUDE.md branch convention).
- Decision recorded: the 8192-site sweep (§3.6) is an *optional, separately-committed*
  refactor, not folded into the P3 cleanup, to avoid an unreviewed I/O-behavior change.

## 8. Step 8 — Roadmap update

- Check off ROADMAP_V2.md:1177 ("Externalize integrity knobs into config.py"), citing the
  implementing commit SHA, and note which K2 option (a/b) shipped.
- If the 8192 follow-up or the `safe_yaml` move (ROADMAP_V2.md:1184) were done together,
  cross-check those lines too. If K2 used option (a), explicitly leave ROADMAP_V2.md:1184
  open with a note that the env-var is already shared so the eventual move is mechanical.
- Update the config.py LOC estimate in ROADMAP_V2.md:165 ("65 LOC → ~150 LOC") if it drifts.
- Per MEMORY "Roadmap-Todo Sync": the checkmark + SHA is mandatory, not optional.

## 9. Step 9 — README / Benchmark

- README: minor — if the README documents tunable env vars, add the three new
  `NA0S_INTEGRITY_*` names; otherwise N/A. Confirm at implementation time (don't invent a
  section). No quickstart or behavior change to document.
- Benchmark: **N/A** — no detector recall/FPR change; this item cannot move any benchmark
  number (digest-identical, defaults unchanged). Note this in the PR so a reviewer doesn't
  expect a benchmark delta.

## 10. Step 10 — Taxonomy / Coverage Matrix / per-feature thresholds

**N/A — no attack-class taxonomy code, no Coverage Matrix row, and no detection scorer is
involved. These knobs are I/O/retention/env-name plumbing for the integrity layer, not a
detector with a recall/FPR threshold.** The clamp bounds added in §3 are guardrails on env
parsing, not security thresholds, so the "no arbitrary magic threshold" rule is satisfied by
(i) keeping all three defaults byte-identical to the shipped values and (ii) documenting the
clamps inline as pathological-input rejection.

## 11. Step 11 — PR & held-out test gate

- Open PR via `github-pr-prep` then `github-pr-review` skills (per MEMORY "Use GitHub Skills").
- Gate: targeted `tests/integrity/` green → full `tests/` shows zero NET regressions vs the
  `main` baseline (re-run baseline if unsure; MEMORY documents ~15 env-only failures that are
  NOT regressions). CI recall gate is irrelevant here (no detector change) but must still pass.
- Do not merge to `main` without explicit user confirmation (per MEMORY git policy).

---

## Q&A self-check

- **Q1 — Can Na0S handle the target (correct knob handling + suite green)?** Today the knobs
  work but are scattered. After the plan: constants centralized, env-overridable with clamped
  fallbacks, digest/behavior preserved, full suite green. Covered by §6 A/B/C.
- **Q2 — Cleanup done?** Yes: two K1 literals and five K4 literals de-duplicated to single
  constants; new code in `config.py`/`tests/integrity/`; shim files untouched. §7.
- **Q3 — Pipeline wiring correct?** Constants imported at each L11 site; predict/cascade
  already import `config` and need no new wiring (no L11 knob is on the scan hot path). §4.
- **Q4 — Tested for code AND use-case?** Yes: code tests (defaults/override/fallback) +
  behavior tests (round-trip load, tamper-reject, hmac sidecar, retention, YAML cap) +
  loader smoke. §6.
- **Q5 — Harvester audit?** N/A — no dataset/harvest dimension to this item.
- **Q6 — Taxonomy + Coverage Matrix?** N/A — no attack code/row. §10.
- **Q7 — Scorer scores it correctly?** N/A — no detection scorer involved. §10.
- **Q8 — predict.py / cascade.py references?** Both import `config` already; neither
  references the four knobs. Parity = confirm no name collision with the new constants. §4.
- **Q9 — Harvester agent harvests this type?** N/A — not a harvestable intel/attack type.
- **Q10 — Other correctness checks?** (a) Reload-safety in tests (importlib.reload). (b)
  K4 wording-drift regression against `test_safe_pickle.py:113`. (c) Chunk clamp `lo>=4096`
  to prevent a 0-chunk infinite loop. (d) K2 package-boundary decision (option a vs b). (e)
  K3 `None`-sentinel for late binding. (f) Verify no shim gets new code. All in §3/§6.

---

## Execution preconditions / dependencies

- **Soft dependency on the P2 `safe_yaml` move (ROADMAP_V2.md:1184)** — only the K2 (YAML-max)
  knob. If that move has not landed, ship K2 via option §3.5(a) (shared env var, no package
  cross-dependency); the other three knobs (K1, K3, K4) have **no** dependency and can land
  immediately.
- No dependency on any other hardening item in this batch (this is isolated P3 plumbing).
- Requires the existing `src/na0s/_env.py` helpers (present, verified) — no new infra.
- Run in a git worktree off `main` (per MEMORY multi-agent worktree discipline); verify
  imports against `main` with `PYTHONPATH=<worktree>/src`, not the stale editable install.

## Definition of Done

- [ ] `config.py` has `INTEGRITY_HASH_CHUNK_BYTES`, `INTEGRITY_YAML_MAX_BYTES`,
      `INTEGRITY_BACKUP_RETENTION`, `PICKLE_SIGNING_KEY_ENV`, all defaulting to the
      byte-identical shipped values, env-overridable via `_env` helpers (except the key-name).
- [ ] `safe_pickle.py:60,92` use `INTEGRITY_HASH_CHUNK_BYTES`; `:84` and the four message/
      docstring strings use `PICKLE_SIGNING_KEY_ENV`; "is not set" wording unchanged.
- [ ] `model_rollback.py:137` `cleanup` defaults `keep` from `INTEGRITY_BACKUP_RETENTION` via
      a `None` sentinel; explicit-`keep` callers unaffected.
- [ ] K2 wired via option (a) or (b); choice + rationale recorded in PR and roadmap.
- [ ] `tests/integrity/test_config_knobs.py` added: default/override/invalid for each knob +
      round-trip-loads + tamper-rejects + hmac-sidecar + retention + YAML-cap behavior tests.
- [ ] `python3 -m pytest tests/integrity/ -q` green; full `tests/` shows zero NET regressions
      vs `main` baseline; loader smoke (`scan()` forcing model load) passes.
- [ ] ROADMAP_V2.md:1177 checked off with commit SHA; LOC note (line 165) refreshed if drifted.
- [ ] No shim file edited; new test in mirrored dir; branch named `hardening/...`.
- [ ] na0s-review-checklist applied: no hallucinated APIs (every symbol verified above against
      file:line), imports/wiring confirmed, tests assert behavior not literals, no arbitrary
      thresholds (defaults carried over; clamps are guardrails), CLI/suite smoke included,
      no destructive autonomous action, env/reload blind spot covered.

---

## Agent / skill team assignment (per step)

Inject `na0s-review-checklist` into every spawned agent's prompt.

| Step | Owner agent / skill | Why |
|------|---------------------|-----|
| 1–2 (explore + roadmap) | `Plan` + `layer-9-11-auditor` | L11 ownership; confirm line refs and config-home intent. |
| 3 (root-cause plan) | `layer-9-11-auditor` + skill `security-review` | integrity-layer expertise; verify no security regression in K4/K1. |
| 4 (wiring/parity) | `silent-failure-hunter` | catch the K3 late-binding / sentinel and config name-collision traps. |
| 5 | — | N/A. |
| 6 (tests) | `l3-l5-code-auditor` + skill `na0s-debugging` + skill `eval-harness` (smoke only) | author non-hollow code+behavior tests; reload-pattern; loader smoke. |
| 7 (cleanup) | `layer-9-11-auditor` | de-dup literals, enforce dir/shim conventions. |
| 8 (roadmap) | `Plan` | check-off + SHA + LOC note per Roadmap-Todo Sync. |
| 9 (README) | (only if README has an env-var table) `security-research-auditor` | minimal doc add. |
| 10 | — | N/A. |
| 11 (PR + gate) | skills `github-pr-prep` → `github-pr-review` → `pr-review-toolkit:review-pr`; `github-ci-fix` if CI red | least-privilege PR flow; close to green. |
| cross-cutting | `security-research-auditor` | final adversarial read of K1/K4 (does the rename ever weaken tamper detection? — answer: no, digest-identical). |

> N/A skills for this item: `cron-scheduling`, `data-harvesting` (no scheduled job or
> harvest dimension).
