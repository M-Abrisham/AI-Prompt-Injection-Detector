---
item: 10
title: "M3 — allowlist/validate NA0S_PROMPTGUARD_MODEL env model id (off-by-default supply-chain hardening)"
priority_tier: P1 (supply-chain / model-source-integrity; only live when N5 PromptGuard is opt-in enabled)
depends_on: [3]          # inherits the #3 revision-pin + use_safetensors helper / loader hardening
applicable_steps: [1, 2, 3, 4, 6, 7, 8, "9-partial", "10-partial", 11]
na_steps:
  - "5  — HARVESTER AUDIT: N/A, no harvested attack corpus for an env-value validation change"
  - "Q5 — N/A, same reason"
  - "Q6 — Taxonomy/Coverage Matrix: N/A, no per-attack code; integrity guard only (label-collision check only)"
  - "Q7 — per-attack SCORER threshold: N/A, no detector score is produced; this validates a config string"
  - "Q9 — harvester agent harvest type: N/A, not an attack class"
classification: supply-chain / model-source-integrity (NOT a prompt-injection attack class)
also_in_scope_note: "NB late_chunking model id is NOT in scope (hardcoded default, no env override). Only NA0S_PROMPTGUARD_MODEL is attacker-influenceable."
---

# Item 10 — Allowlist / validate the `NA0S_PROMPTGUARD_MODEL` env model id

## 0. TL;DR root cause

`PromptGuardClassifier.__init__` reads the **untrusted** environment variable
`NA0S_PROMPTGUARD_MODEL` and feeds its value **verbatim** into
`AutoTokenizer.from_pretrained()` / `AutoModelForSequenceClassification.from_pretrained()`
with **no allowlist and no format validation**. Whoever can set that env var (a compromised
deploy config, a CI secret tamper, a co-tenant in a shared process) can repoint the N5 loader
at an arbitrary Hugging Face repo or a local filesystem path, pulling a model the operator
never vetted — i.e. a model-source-injection / arbitrary-artifact-download primitive that also
re-opens the pickle-RCE surface item #3 closes via `use_safetensors`. The hardening: validate
the env value against a small operator-controlled allowlist (with a syntactic sanity gate) and,
on rejection, fail safe to the pinned default rather than load the attacker-chosen id.

This is **off-by-default** in effect: N5 only runs when `NA0S_ENABLE_PROMPTGUARD=1`
(`promptguard_classifier.py:181-183`), so the exposure exists only on deployments that have
opted into PromptGuard. The guard must not regress that opt-in path.

---

## 1. KEY REFS — verified line numbers (corrections noted)

Opened `src/na0s/ml/promptguard_classifier.py` end-to-end. All three refs **confirmed exact**:

| Ref (as given) | Verified | Note |
|---|---|---|
| `promptguard_classifier.py:154` | ✅ `model_name = os.environ.get(_ENV_MODEL, DEFAULT_MODEL_NAME)` | The single point where the untrusted env value enters; `_ENV_MODEL = "NA0S_PROMPTGUARD_MODEL"` (`:76`), `DEFAULT_MODEL_NAME = "meta-llama/Prompt-Guard-2-22M"` (`:68`). |
| `promptguard_classifier.py:181-183` | ✅ `is_enabled()` → `os.environ.get(_ENV_ENABLE, "0").lower() in ("1","true","yes")` | The opt-in gate that makes this off-by-default. |
| `promptguard_classifier.py:385-387` | ✅ `AutoTokenizer.from_pretrained(self._model_name)` / `AutoModelForSequenceClassification.from_pretrained(self._model_name,)` | The sink: `self._model_name` (set from the env at `:158`) flows unvalidated into HF loaders inside the `_ensure_loaded()` `try/except` (`:380-399`). |

### Corrections / scope clarifications discovered while grounding
- **Canonical module is `na0s.ml.promptguard_classifier`.** `src/na0s/promptguard_classifier.py`
  is a shim (`:1-22`) that aliases the canonical module into `sys.modules`. Do NOT add code to
  the shim (CLAUDE.md). All edits land in `src/na0s/ml/promptguard_classifier.py`.
- **The sibling `na0s.ml.promptguard.py` does NOT read the env var.** Its
  `PromptGuardClassifier.__init__` takes `model_name: str = DEFAULT_MODEL_NAME` as a ctor arg
  only (`promptguard.py:74,77`); `grep` confirms **no `os.environ` read anywhere** in that file.
  So the attacker-influenceable surface is exactly ONE module — do not over-scope into
  `promptguard.py` (its `from_pretrained` at `:184-186` is hardened by item #3, not item #10).
- **`promptguard_signal.py`** reads `NA0S_PROMPTGUARD_ENABLED` (`:42`), a different env var that
  gates the *signal*, not the model id. Out of scope.
- **`late_chunking.py`** model id is hardcoded (item-specific scope note confirms): no env
  override, so not attacker-influenceable → **NOT in scope** for this item.

Net: exactly **one** untrusted-env → loader path to harden: `:154` (entry) → `:158` (store) →
`:385-387` (sink).

---

## 2. Gap vs ideal

### Threat model
- **Untrusted env value → arbitrary artifact source.** `from_pretrained(<env string>)` will
  (a) download from any HF repo id `org/name`, (b) load from any local path if the string is a
  path, or (c) hit a configured HF endpoint/mirror. None of these is operator-vetted. Even with
  item #3's `use_safetensors=True`, a malicious-but-safetensors model is still a model the
  operator did not choose (data-poisoned / backdoored classifier that silently passes attacks).
- **Env vars are a realistic tamper channel** in container/CI deployments — they are not a
  trusted boundary the way source code is. The defensive posture (project positioning: an SDK
  *embedded in someone else's AI*) means we must not blindly trust process-environment config.

### Ideal end state
1. The env value is accepted **only** if it is on an operator-controlled allowlist OR passes a
   conservative syntactic validator; otherwise the loader **falls back to the pinned default**
   (`DEFAULT_MODEL_NAME`) and emits exactly one `logger.warning` naming the rejected id.
2. The default model is always implicitly allowlisted (zero-config deployments keep working).
3. Operators can extend the allowlist via a second env var (air-gap / private-mirror escape
   hatch) — mirroring item #3's `NA0S_HF_REVISION*` override pattern — without code changes.
4. Validation is **fail-safe, not fail-closed-crash**: a rejected id must never raise; it
   degrades to default (or, if even the default is unavailable, the existing `_init_failed`
   path keeps `scan()`/`predict()` returning a normal `ScanResult`).
5. The validator is FP-safe: a legitimate custom org/model id an operator deliberately set is
   not silently downgraded *unless* they failed to allowlist it (and then it's a loud warning,
   not a silent swap).

### Current gap
None of (1)-(5) exists. Line 154 is a bare `os.environ.get` with the env value used as-is.

---

## 3. Root-cause implementation plan (numbered, by file/function)

> Conventions (CLAUDE.md): new shared logic goes in a sub-package, not a top-level file. The
> natural home is `src/na0s/integrity/` (model-supply integrity already lives there:
> `model_provenance.py`, `safe_pickle.py`, `dep_scanner.py`). **Reuse item #3's
> `integrity/hf_loading.py` if it exists by the time this lands** (depends-on #3) — add the
> allowlist there so all model-source policy is in one module. Do NOT add code to shim files.

1. **Add `validate_model_id()` to `src/na0s/integrity/hf_loading.py`** (created by item #3; if
   #3 has not landed, create the module per #3's Step 3 and put this function there). Signature
   and contract:
   - `def resolve_model_id(requested: str, *, default: str) -> str:` — returns `requested` if it
     is allowed, else logs one `warning` and returns `default`.
   - Allow rules (in order):
     1. `requested == default` → always allowed (zero-config path).
     2. `requested` ∈ `ALLOWED_MODEL_IDS` (a frozen set seeded with the known-good Na0S models:
        `meta-llama/Prompt-Guard-2-22M`, and the embedding defaults already enumerated in #3's
        `PINNED_REVISIONS` keys — reuse that table's keys as the canonical allowlist so the two
        never drift).
     3. `requested` ∈ operator extension via env `NA0S_MODEL_ALLOWLIST` (comma-separated ids).
     4. Otherwise rejected.
   - **Syntactic gate (defense-in-depth, applies before download even for allowlisted-by-pattern
     ids if the operator opts into pattern mode):** reject ids containing `..`, leading `/` or
     `~`, NUL/whitespace/control chars, or a `file://`/scheme prefix — i.e. block path-traversal
     and local-FS escapes. Keep this a **documented, justified** rule, not a magic regex: the
     allowed shape is the HF `org/name` form `[A-Za-z0-9._-]+/[A-Za-z0-9._-]+` (cite HF repo-id
     grammar in a comment). No arbitrary thresholds — this is a grammar, not a tuned number.
   - Returns the *string* only; revision/safetensors kwargs stay item #3's job. This function is
     purely the **source-allowlist** layer; #3 is the **fetch-hardening** layer. They compose.

2. **`src/na0s/ml/promptguard_classifier.py:153-154`** — wrap the env read:
   ```
   if model_name is None:
       requested = os.environ.get(_ENV_MODEL, DEFAULT_MODEL_NAME)
       model_name = resolve_model_id(requested, default=DEFAULT_MODEL_NAME)
   ```
   Import `resolve_model_id` lazily / behind a `try/except ImportError` so the pure-python
   integrity helper never forces a heavy import on the core path (matches the `_HAS_TRANSFORMERS`
   guard style at `:52-57`). If the import fails, fall back to the **previous** behavior of using
   the raw env value? **NO** — fall back to `DEFAULT_MODEL_NAME` (fail-safe). Document the choice.
   Note: an **explicit** `model_name=` ctor arg (`:153` else-branch) is operator code, not env —
   leave it un-validated by default, OR validate it too but allow-by-default; decide and document
   (recommend: validate the ctor arg as well but never downgrade a non-None explicit arg silently
   — instead warn-and-honor, because an explicit ctor arg is in-process trusted code). The
   **env** path is the one that MUST be hardened.

3. **Allowlist source-of-truth:** `ALLOWED_MODEL_IDS` derives from item #3's `PINNED_REVISIONS`
   keys (one table to rotate). If #3 is not yet merged, seed it from the verified defaults:
   `meta-llama/Prompt-Guard-2-22M` (`promptguard_classifier.py:68`, `promptguard.py:58`) and the
   ST defaults from #3's ref list. Add a code comment that the two must stay in sync (and a test
   asserts it — see C4).

4. **No change to `is_enabled()` (`:181-183`)** — the opt-in gate is correct; this item only
   constrains *which* model loads once enabled.

5. **Import discipline:** the new helper is pure-python (no transformers). It must be importable
   even when `_HAS_TRANSFORMERS is False`, so the validation runs regardless of whether the model
   ultimately loads.

> Centralize, don't sprinkle: the allowlist lives next to the revision pins (item #3), so a
> single `integrity/hf_loading.py` owns "which models, which revisions, safetensors-only" — the
> whole model-source policy. Review-checklist "no copy-paste magic numbers / one source of truth".

---

## 4. Pipeline wiring (Q8 / Q3) — APPLICABLE

`predict.py` and `cascade.py` both reach the hardened constructor **indirectly** through the
singleton:
- `predict.py:183-184` imports `get_promptguard_score` from `na0s.promptguard_classifier`
  (shim → canonical); called in the N5 block at `predict.py:1160-1181`.
- `cascade.py:57-58` imports the same `get_promptguard_score`; called at `cascade.py:536-562`,
  with the auto-disable counter `_pg_failure_state` (`cascade.py:64`, `:556-562`, `:1407-1410`).
- Both routes call `get_promptguard_score` → `get_promptguard_classifier()`
  (`promptguard_classifier.py:409`) → `PromptGuardClassifier()` (`:431`) → the env read at `:154`.

**Wiring change is indirect and parity-free:** because both entrypoints construct the classifier
through the same singleton/ctor, fixing `:154` hardens **both paths automatically** — no new flag
or signal is added to `predict.py`/`cascade.py`. Parity check = *confirm both routes hit the same
ctor* (they do) **and** that a rejected-id-downgraded-to-default load still feeds the existing
auto-disable counter correctly if the default itself fails to load (`cascade.py:556-562`). No
COVERAGE_MATRIX row changes (no detector score is produced).

---

## 5. HARVESTER AUDIT (Step 5 / Q5 / Q9) — **N/A**
N/A — this is an env-config validation / model-source-integrity change. There is no
prompt-injection attack corpus to harvest, decontaminate, tag, or train on. The "inputs" here
are config strings (the env value), not attack payloads.

---

## 6. Test plan — Code + Use-Case (APPLICABLE; reframed per scope)

New tests in **`tests/integrity/test_hf_loading.py`** (co-located with item #3's helper tests;
the dir already exists with `__init__.py`) + targeted additions to the existing
**`tests/ml/test_promptguard_classifier.py`** (which already has a `TestEnvVarConfiguration`
class at `:573` and mocks `from_pretrained`). Tests **must not hit the network or real HF**
(CLAUDE.md / no-API-key memory) — `transformers`/`sentence-transformers` are **not installed in
the dev env** (verified: both imports fail), so all model paths are mocked / `_HAS_TRANSFORMERS`
is already False. No hollow asserts — check returned strings, log records, and call kwargs
explicitly.

### Code-level (the validator does what it claims)
1. **C1 — default always allowed:** `resolve_model_id(DEFAULT_MODEL_NAME, default=DEFAULT_MODEL_NAME)`
   returns the default unchanged, emits no warning (`caplog` empty at WARNING).
2. **C2 — allowlisted id passes:** an id in `ALLOWED_MODEL_IDS` (and one added via
   `NA0S_MODEL_ALLOWLIST`) is returned unchanged, no warning.
3. **C3 — rejected id downgrades + warns:** a not-allowlisted id (`"evil/backdoor"`) returns the
   default AND emits exactly one WARNING naming the rejected id (assert message contains the id).
4. **C4 — traversal / scheme / control-char ids rejected:** parametrized over
   `"../etc/passwd"`, `"/abs/path"`, `"~/x"`, `"file:///x"`, `"a/b\x00"`, `"a b/c"` →
   each returns the default + warns. Assert the syntactic gate, not just the set membership.
5. **C5 — allowlist ⊇ pinned-revision keys (anti-drift):** assert every key of item #3's
   `PINNED_REVISIONS` is in `ALLOWED_MODEL_IDS` (guards the "two tables drift" regression). If #3
   not yet merged, assert against the seeded constant + add a TODO to wire to `PINNED_REVISIONS`.

### Constructor-integration (env value is actually validated at :154)
6. **C6 — env rejected → ctor uses default:** with
   `mock.patch.dict(os.environ, {"NA0S_PROMPTGUARD_MODEL": "evil/backdoor"})`,
   `PromptGuardClassifier().model_name == DEFAULT_MODEL_NAME` (NOT `"evil/backdoor"`). This
   *changes the expected behavior of the existing test* `test_model_name_from_env`
   (`tests/ml/test_promptguard_classifier.py:576-579`, currently asserts `"custom/model"` is
   accepted verbatim) — that test must be **updated, not deleted**, to use an *allowlisted*
   custom id (e.g. add it via `NA0S_MODEL_ALLOWLIST`) so it still proves env override works for
   *legitimate* ids. (CLAUDE.md / memory: never weaken a test to pass — here we are tightening it
   to match the new, stricter contract and keeping the legitimate-override coverage.)
7. **C7 — env allowlisted → ctor honors it:** with both `NA0S_PROMPTGUARD_MODEL=custom/model`
   and `NA0S_MODEL_ALLOWLIST=custom/model`, `model_name == "custom/model"`.
8. **C8 — explicit ctor arg honored:** `PromptGuardClassifier(model_name="explicit/model")`
   keeps the existing pass-through semantics (in-process trusted) — preserves
   `test_model_name_explicit_overrides_env` (`:588-591`).
9. **C9 — import-failure fail-safe:** simulate `resolve_model_id` import failing (monkeypatch the
   import) and assert the ctor falls back to `DEFAULT_MODEL_NAME`, never the raw env value, and
   never raises.

### Use-Case / behavior (Step 6 reframed: tampered config rejected; legit still works; scan OK)
10. **U1 — tampered env, scan still works:** with `NA0S_ENABLE_PROMPTGUARD=1` and a malicious
    `NA0S_PROMPTGUARD_MODEL`, and the loaders mocked, assert `get_promptguard_classifier()`
    yields a classifier whose `model_name` is the default, and that `predict()`/`scan()` on one
    benign + one injection sample returns a normal `ScanResult` (no exception). This is the
    "tampered file/config rejected, scan still works" use-case.
11. **U2 — legit allowlisted env still loads:** allowlisted custom id + mocked loader → the
    singleton uses that id; the layer reports available; `get_promptguard_score` returns a float.
12. **U3 — FP-safety / score invariance:** the downgrade-to-default must not change benign
    verdicts vs. the no-env baseline (since the *default* model is what would run anyway when the
    env is unset). Run `predict()` on a small benign batch with env unset vs. env=malicious;
    assert identical verdicts (zero FP/score drift). This proves the guard is purely a
    source-policy gate, not a scoring change.
13. **U4 — CLI smoke (review checklist):** run the real CLI on one benign + one injection string
    with `NA0S_PROMPTGUARD_MODEL` set to a junk value and `NA0S_ENABLE_PROMPTGUARD` unset
    (transformers absent → degraded path); confirm exit 0 + a verdict, proving the import-guarded
    validator doesn't break the keyless core.

### Suite gate
14. **S1 — full suite:** `python3 -m pytest tests/ -q --tb=line` → zero net regressions vs.
    baseline (CLAUDE.md; ~15 min). Run targeted
    `python3 -m pytest tests/integrity/ tests/ml/test_promptguard_classifier.py -v` first.

---

## 7. Cleanup / refactor (Q2) — APPLICABLE (light)

- The change is additive + one ctor edit; no dead code. Co-locating the allowlist with #3's
  revision table is the only refactor (one model-source-policy module).
- **Update, don't bypass, the now-stricter existing test** `test_model_name_from_env`
  (`tests/ml/test_promptguard_classifier.py:576-579`) — see C6. Document the contract change in
  the test docstring.
- Repo hygiene: working tree has stray scratch artifacts (`_skeptic_test_out.txt`,
  `_xfail_run.txt`, `pyt_out.txt`, `logs/`) — out of scope; do NOT `git add` them. Scope every
  `git add`.
- Land on a dedicated branch `hardening/env-model-id-allowlist` (branch-naming convention), off
  `main` (or off #3's branch if sequencing onto #3) — NOT the current `hardening/rag-poison-wiring`.

---

## 8. Roadmap / README / Benchmark updates (Steps 8-9)

- **ROADMAP_V2.md:** add a checked item under the supply-chain / Layer-11 integrity section:
  `[ ] SUP-M3 — allowlist + validate NA0S_PROMPTGUARD_MODEL env model id (fail-safe to pinned
  default)`. **Label-collision warning:** the token `M3` already appears in ROADMAP_V2.md for
  (a) the held-out adversarial-canary item (`:1492`, `:1628`) and (b) taxonomy `M3 = Document`
  (`:1275`). Use a distinct label (`SUP-M3` / `INTEG-MODEL-ALLOWLIST`) to avoid the duplicate-ID
  smell. Check the box + cite the commit SHA once landed (Roadmap-Todo Sync memory rule).
- **README / SECURITY.md (Step 9, PARTIAL):** add one line under supply-chain hardening that the
  PromptGuard model source is allowlisted and falls back to the pinned default on a rejected env
  value. Only if a relevant section exists; do not create docs.
- **Benchmark:** no metric change expected (same default model, same scores when env unset or
  allowlisted). Record in the PR that benchmark numbers are unchanged.

---

## 9. Taxonomy / Coverage Matrix / Scorer (Step 10 / Q6 / Q7) — N/A (label-check only)

- **Q6 Taxonomy + Coverage Matrix:** N/A — this is model-source integrity, not a
  prompt-injection technique; no `data/taxonomy.yaml` code, no COVERAGE_MATRIX recall row. Only
  Step-10 action: avoid the `M3` label collision (Step 8).
- **Q7 Scorer thresholds:** N/A — no detector score is produced. The validator is a grammar +
  set-membership check, not a tuned threshold; the only "numbers" (HF repo-id charset) are a
  documented grammar, not a magic value.

---

## 10. Q&A self-check (instantiated)

- **Q1 — Can Na0S handle the threat + suite green?** Not yet: `:154` uses the untrusted env value
  verbatim. Fix per Step 3 (validate → allowlist → fail-safe default), then full suite (S1).
- **Q2 — Cleanup?** Light: co-locate with #3's table; update the now-stricter env test; don't
  commit scratch files (Step 7).
- **Q3 — Pipeline wiring correct?** Yes — predict.py + cascade.py inherit via the shared
  singleton/ctor; verify the auto-disable counter still trips if the default fails (Step 4).
- **Q4 — Code AND use-case tested?** Yes — C1-C9 (code) + U1-U4 (behavior) + S1 (suite).
- **Q5 — Harvester audit?** N/A (Step 5).
- **Q6 — Taxonomy/Coverage?** N/A — label-collision check only (Step 9).
- **Q7 — Scorer?** N/A (Step 9).
- **Q8 — predict/cascade references?** YES — `predict.py:183-184` + N5 block `:1160-1181`;
  `cascade.py:57-58` + N5 block `:536-562` + `_pg_failure_state` `:64,1407-1410`. Both route
  through the hardened ctor (Step 4).
- **Q9 — Harvester agent harvests this type?** N/A — not an attack class.
- **Q10 — Other checks:**
  - **Sibling-module scope:** confirm `na0s.ml.promptguard.py` (no env read; ctor-only model_name,
    `:74,77`) and `promptguard_signal.py` (reads `NA0S_PROMPTGUARD_ENABLED`, `:42`) are correctly
    **out of scope** — done above.
  - **Env-override naming:** pick the extension env var name (`NA0S_MODEL_ALLOWLIST`) consistent
    with item #3's `NA0S_HF_REVISION*` so the two compose; document both in the module docstring
    (`promptguard_classifier.py:18-24` lists env vars — add the new one there).
  - **Explicit-ctor-arg policy:** decide and document whether a non-None `model_name=` arg is
    validated (recommend warn-and-honor; it's trusted in-process code) — see Step 3.2.
  - **Sequencing on #3:** if #3's `integrity/hf_loading.py` does not yet exist, create it here and
    leave a note in #3's spec; the allowlist and the revision table belong in the same module.

---

## 11. Agent / skill team (inject `na0s-review-checklist` into every prompt)

| Step | Owner agent / skill | Mandate |
|---|---|---|
| 1-2 explore + threat-model | **security-research-auditor** + skill **security-review** | Confirm the env→loader path is the only attacker-influenceable model-source surface; document the model-source-injection threat. |
| 3 implementation (validator + ctor edit) | **l3-l5-code-auditor** (owns ml/N5 loaders) + skill **detector-authoring** (wiring discipline only) | Add `resolve_model_id` to `integrity/hf_loading.py`; wrap `:153-154`; fail-safe to default. |
| 3 integrity-module placement | **layer-9-11-auditor** (L11 supply-chain owner) | Confirm the allowlist belongs in `integrity/hf_loading.py` alongside #3's revision table; no top-level dump. |
| 4 predict/cascade parity | **l3-l5-code-auditor** | Verify both entrypoints inherit via the singleton; auto-disable counter intact. |
| graceful-failure audit | **silent-failure-hunter** | A rejected id must downgrade-and-warn (one warning, never silent, never crash); import-failure falls back to default not raw env (C9). |
| 6 tests | **l3-l5-code-auditor** + skill **na0s-debugging** (mock / `_HAS_TRANSFORMERS`-False patterns) | C1-C9 + U1-U4; update (not delete) the stricter env test; no hollow asserts; no network. |
| 6 suite gate / CI | skill **eval-harness** + **github-ci-fix** | Full suite green; targeted ml/integrity first. |
| 8 roadmap/readme | **Plan** (+ creative-writer only if README prose) | Add `SUP-M3` label (avoid the `M3` collision); cite SHA. |
| 11 PR | skill **github-pr-prep** then **pr-review-toolkit:review-pr** / **github-pr-review** | Held-out tests must pass before merge. |

---

## Execution preconditions / dependencies

- **Depends-on: item #3** (HF revision pin). #3 introduces `integrity/hf_loading.py` and the
  `PINNED_REVISIONS` table that this item's `ALLOWED_MODEL_IDS` should derive from. If #3 lands
  first, this item adds `resolve_model_id` to the same module and sources the allowlist from
  `PINNED_REVISIONS.keys()`. If they are developed in parallel, this item may create the module
  and #3 extends it — coordinate so the two tables stay one source of truth (test C5).
- **Off-by-default:** the exposure only exists when `NA0S_ENABLE_PROMPTGUARD=1`; the guard must
  not regress that opt-in path (U2).
- **Environment:** `transformers`/`sentence-transformers` are NOT installed in the dev env, so
  all model-load tests mock and the validator (pure-python) runs regardless. Verify any
  rename/import against MAIN with `PYTHONPATH=<worktree>/src` (the editable install points at a
  stale checkout — env memory).
- **Branch:** `hardening/env-model-id-allowlist`, off `main` (or off #3's branch if sequencing).

## Definition of done

- [ ] `resolve_model_id(requested, *, default)` added to `src/na0s/integrity/hf_loading.py` with
      allowlist + HF-repo-id grammar + traversal/scheme/control-char rejection + one-warning-on-reject.
- [ ] `ALLOWED_MODEL_IDS` sourced from (or asserted ⊇) item #3's `PINNED_REVISIONS` keys (C5).
- [ ] Operator extension via `NA0S_MODEL_ALLOWLIST` env var; documented in
      `promptguard_classifier.py` env-var docstring (`:18-24`).
- [ ] `promptguard_classifier.py:153-154` validates the env value → uses it only if allowed, else
      fail-safe to `DEFAULT_MODEL_NAME`; import-failure also falls back to default (never raw env).
- [ ] Explicit `model_name=` ctor-arg policy decided + documented (warn-and-honor, in-process trusted).
- [ ] Sibling modules confirmed out of scope (`promptguard.py` no env read; `promptguard_signal.py`
      different env var; `late_chunking.py` hardcoded).
- [ ] Graceful failure verified: rejected id never raises; `scan()`/`predict()` returns a normal
      `ScanResult` on benign + malicious input under a tampered env (U1).
- [ ] Tests added: `tests/integrity/test_hf_loading.py` (C1-C5, C9) + edits to
      `tests/ml/test_promptguard_classifier.py` (C6-C8, U1-U4) — the existing `test_model_name_from_env`
      is **tightened, not deleted**. All mock; none hit the network.
- [ ] CLI smoke passes with a junk env model id and transformers absent (degraded keyless path).
- [ ] `python3 -m pytest tests/ -q --tb=line` → zero net regressions vs. baseline.
- [ ] ROADMAP_V2.md item added with distinct `SUP-M3` label (no `M3` collision) + commit SHA cited.
- [ ] PR opened via github-pr-prep; held-out tests green before merge; benchmark numbers confirmed unchanged.
