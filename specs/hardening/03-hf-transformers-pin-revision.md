---
item: 3
title: "M2 — pin transformers>=5.3.0 floor + revision= + use_safetensors on HF loads (CVE-2026-4372)"
priority_tier: P0 (supply-chain / RCE-class, blocks any deploy that enables L5/N5/embedding/worm)
depends_on: []          # self-contained; no other hardening item must land first
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 9, 10-partial, 11]
na_steps:
  - "5  — HARVESTER AUDIT: N/A, no harvested attack dataset for a dependency-pin change"
  - "Q5 — N/A, same reason"
  - "Q6 — Taxonomy/Coverage Matrix: PARTIAL/N/A, no per-attack code; integrity note only"
  - "Q7 — per-attack SCORER threshold: N/A, no scorer; this is a loader/dep guard"
  - "Q9 — harvester agent harvest type: N/A, not an attack class"
classification: supply-chain / model-supply-integrity (NOT a prompt-injection attack class)
---

# Item 3 — Pin `transformers>=5.3.0` floor + `revision=` + `use_safetensors=True` on every HF load

## 0. TL;DR root cause

Na0S loads Hugging Face artifacts (transformer models, tokenizers, sentence-transformer
encoders) **without a pinned dependency floor, without a pinned model `revision`, and
without forcing `use_safetensors=True`**. `transformers` is not pinned *anywhere* — it
enters only transitively through `sentence-transformers>=2.2,<4`
(`pyproject.toml:61`) and is entirely absent from `requirements-benchmark.txt`. That
leaves three live supply-chain exposures: (a) a vulnerable `transformers` resolving below
the CVE-2026-4372 fixed floor, (b) a moving HF model tag (`main`) that can be silently
re-pointed to a malicious snapshot, and (c) a fallback to pickle-backed `pytorch_model.bin`
deserialization (arbitrary code execution at load time) when no safetensors file is forced.

---

## 1. KEY REFS — verified line numbers (corrections noted)

Opened every ref. All confirmed; two refs are **incomplete** and one **drifted**:

| Ref (as given) | Verified | Note |
|---|---|---|
| `pyproject.toml:61` | ✅ `"sentence-transformers>=2.2,<4"` | **`transformers` itself is NEVER pinned in pyproject** — only pulled transitively. There is no `transformers` extra. |
| `requirements-benchmark.txt:71` | ⚠️ line 71 = `torch==2.8.0`; **`transformers` is absent from the whole file** | Benchmark lockfile has no transformers entry at all → unpinned in CI. Correction: the gap is *absence*, not a wrong pin. |
| `src/na0s/ml/promptguard.py:184-187` | ✅ `AutoTokenizer.from_pretrained(self._model_name)` / `AutoModelForSequenceClassification.from_pretrained(self._model_name).to(...)` | default model `meta-llama/Prompt-Guard-2-22M` (`promptguard.py:58`); no `revision`, no `use_safetensors`. |
| `promptguard_classifier.py:385-387` | ✅ identical pattern | default `meta-llama/Prompt-Guard-2-22M` (`:68`); env override `NA0S_PROMPTGUARD_MODEL` (`:76`). No `revision`/`use_safetensors`. |
| `late_chunking.py:333-334` | ✅ `AutoTokenizer.from_pretrained(model_name)` / `AutoModel.from_pretrained(model_name)` | default `sentence-transformers/all-MiniLM-L6-v2` (`:56-57`). No `revision`/`use_safetensors`. |
| `embedding_classifier.py:356` | ✅ `SentenceTransformer(self._model_name)` | default `all-MiniLM-L6-v2` (`:59`). `SentenceTransformer` ctor → no `revision`. |
| `embedding_adapter.py:320` | ✅ `SentenceTransformer(self._model_name)` | default `all-MiniLM-L6-v2` (`:291`). No `revision`. |
| `worm/detector.py:417` | ✅ `SentenceTransformer(model_name)` | default `all-MiniLM-L6-v2` (`:397`); already gated by `_is_model_cached()` (`:406`) so network-load is opt-in. No `revision`. |
| `layer16/detectors/embedding_drift.py:152` | ✅ `SentenceTransformer(_MODEL_NAME)` | default `all-MiniLM-L6-v2` (`:55`). No `revision`. |

### Correction — one load site is MISSING from the KEY REFS
- **`src/na0s/ml/predict_embedding.py:192`** — `embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)`. This is the loader that `cascade.py:806` actually calls via `_load_embedding_models` (imported at `cascade.py:100` as `load_models`). **It must be in scope** or the cascade path stays unhardened. (It already uses `safe_load` for the *pickle* classifier at `:195` — that is the separate item-8 / safe_pickle defense, leave it.)

Total in-scope load sites: **9** (8 from refs + predict_embedding.py).

---

## 2. Gap vs ideal

### Threat model (CVE-2026-4372 + HF default behavior)
- **`transformers` floor (CVE-2026-4372):** an unpinned `transformers` can resolve to a
  version below the security-fixed release. Verify the exact fixed version against the
  upstream advisory (see Step "Resources" — `WebSearch`/`WebFetch` the NVD/GHSA entry)
  and set the floor to the smallest fully-patched release. The spec title proposes
  `>=5.3.0`; **the implementing agent MUST confirm the real fixed version** and correct the
  floor if the advisory says otherwise — do not ship an unverified magic version (review
  checklist: "no arbitrary thresholds").
- **Moving `revision`:** `from_pretrained(name)` and `SentenceTransformer(name)` default to
  the `main` branch / latest tag of the repo. A compromised or re-pointed HF repo serves a
  different model at the same name. Pinning `revision=<commit-sha-or-immutable-tag>` makes the
  artifact content-addressed and tamper-evident.
- **`use_safetensors`:** by default `transformers` will load `pytorch_model.bin` (a pickle)
  if no `*.safetensors` is present — arbitrary code execution at deserialization. Forcing
  `use_safetensors=True` on `AutoModel*.from_pretrained` rejects the pickle path. Confirm
  current HF default (the default flipped across versions) via `WebFetch` of the
  `from_pretrained` docs; if the floor we pin already defaults safetensors-only, keep the
  explicit `use_safetensors=True` anyway (defense-in-depth + version-independent).

### Ideal end state
1. `transformers` is an **explicit, pinned, floored** dependency in the `embedding` extra
   of `pyproject.toml` and in `requirements-benchmark.txt`.
2. Every `from_pretrained` call passes `use_safetensors=True` **and** `revision=<pinned>`.
3. Every `SentenceTransformer(...)` call passes `revision=<pinned>` (ST forwards `revision`
   to its underlying transformer download; confirm the ST≥? signature accepts it — see Q10).
4. Pins live in ONE place (a constants module), not copy-pasted across 9 sites.
5. Failure is graceful: a load that violates the pin (e.g. safetensors missing) is caught by
   the existing `try/except` and disables the optional layer — it must never crash `scan()`.

---

## 3. Root-cause implementation plan (numbered, by file/function)

> Conventions reminder (CLAUDE.md): no new top-level modules — put shared pin constants in a
> sub-package. The natural home is `src/na0s/integrity/` (supply-chain integrity already lives
> there: `model_provenance.py`, `safe_pickle.py`, `fingerprint.py`). Do **not** add code to
> shim files.

1. **Create `src/na0s/integrity/hf_loading.py`** — single source of truth for pinned revisions
   and a tiny helper. Contents:
   - A frozen dict mapping each model name → its pinned `revision` (immutable commit SHA or
     signed tag), e.g. `PINNED_REVISIONS = {"meta-llama/Prompt-Guard-2-22M": "<sha>",
     "sentence-transformers/all-MiniLM-L6-v2": "<sha>", ...}`. The agent must fetch the real
     current commit SHAs from each HF repo (`WebFetch`/`huggingface-hub` `model_info`) and
     record provenance in a comment. **No placeholder SHAs may ship.**
   - `def hf_from_pretrained_kwargs(model_name) -> dict` → returns
     `{"use_safetensors": True, "revision": PINNED_REVISIONS[model_name]}` (revision omitted
     only if the model isn't in the map, with a `logger.warning`, so an env-overridden custom
     model still loads but is flagged).
   - `def st_kwargs(model_name) -> dict` → returns `{"revision": ...}` (no `use_safetensors`
     — `SentenceTransformer` ctor doesn't take it; revision only). Confirm ST signature first.
   - An optional `NA0S_HF_REVISION_<...>` env override hook so air-gapped/local mirrors aren't
     bricked, mirroring the existing `NA0S_PROMPTGUARD_MODEL` env pattern
     (`promptguard_classifier.py:76`).

2. **`pyproject.toml`** — add an explicit pinned `transformers` line to the `embedding`
   extra (it's the extra that pulls sentence-transformers):
   `"transformers>=<FIXED_VERSION>,<6"` (upper bound mirrors the existing `<4`/`<3` style on
   sibling deps; pick the next-major ceiling). Justify the floor with the CVE advisory in a
   comment.

3. **`requirements-benchmark.txt`** — add `transformers==<exact-fixed-pin>` so CI is fully
   locked (this file uses `==` exact pins throughout, e.g. `torch==2.8.0:71`).

4. **`src/na0s/ml/promptguard.py:184-187`** — pass
   `**hf_from_pretrained_kwargs(self._model_name)` to both `AutoTokenizer.from_pretrained`
   and `AutoModelForSequenceClassification.from_pretrained`. Keep inside the existing
   `try/except` (`:179-198`) so a pin/safetensors miss disables the layer, never crashes.

5. **`src/na0s/ml/promptguard_classifier.py:385-387`** — identical change. Note the env model
   override at `:154` → if a non-mapped model is used, the helper warns and skips `revision`
   (still forces `use_safetensors=True`).

6. **`src/na0s/ml/late_chunking.py:333-334`** — pass `**hf_from_pretrained_kwargs(model_name)`
   to `AutoTokenizer.from_pretrained` and `AutoModel.from_pretrained`. This function is NOT
   inside a try/except (it's a module-level loader at `:311`) — wrap the additions so a
   missing revision degrades to the existing behavior rather than raising (the caller already
   tolerates `None`; confirm and preserve the `return model, tokenizer` contract).

7. **`src/na0s/ml/embedding_classifier.py:356`** — `SentenceTransformer(self._model_name, **st_kwargs(self._model_name))`. Inside existing `try/except` (`:351-…`).

8. **`src/na0s/ml/embedding_adapter.py:320`** — `SentenceTransformer(self._model_name, **st_kwargs(self._model_name))` in `_ensure_encoder`.

9. **`src/na0s/worm/detector.py:417`** — `SentenceTransformer(model_name, **st_kwargs(model_name))`; already inside `try/except (OSError, RuntimeError, ImportError, ValueError, TypeError)` (`:426`) and gated by `_is_model_cached` (`:406`).

10. **`src/na0s/layer16/detectors/embedding_drift.py:152`** — `SentenceTransformer(_MODEL_NAME, **st_kwargs(_MODEL_NAME))` inside `_load_model`.

11. **`src/na0s/ml/predict_embedding.py:192`** (the MISSING ref, cascade-reachable) —
    `SentenceTransformer(DEFAULT_EMBEDDING_MODEL, **st_kwargs(DEFAULT_EMBEDDING_MODEL))`.

12. **Import discipline:** each file imports the helper lazily / behind the existing
    `_HAS_TRANSFORMERS` / `_HAS_SENTENCE_TRANSFORMERS` guards so `integrity.hf_loading`
    (pure-python, no heavy deps) never forces a transformers import on the core path.

> **Centralize, don't sprinkle:** all 9 sites call the helper. This is the review-checklist
> "no copy-paste magic numbers" rule — one pinned-revision table, one place to rotate SHAs.

---

## 4. Pipeline wiring (Q8 / Q3) — APPLICABLE

`predict.py` and `cascade.py` **do** reference the affected loaders, so wiring parity matters:
- `cascade.py:57-58` imports `get_promptguard_score`; `:124` imports
  `get_embedding_classifier`; `:100` imports `predict_embedding.load_models` as
  `_load_embedding_models`, called at `:806`.
- `predict.py:183-184` imports `get_promptguard_score`; `:194` imports
  `get_embedding_classifier` (called `:899`).

**The wiring change is indirect:** these entrypoints call the loaders we are hardening, so by
fixing the 9 leaf load sites both `predict.py` and `cascade.py` inherit the hardened behavior
automatically — **no new flags or signals are added to predict/cascade**. The parity check
is therefore: *confirm both paths route through the same hardened loaders* (they do — same
modules) and that the existing PromptGuard auto-disable counter (`cascade.py:556-562`,
`_pg_failure_state`) still trips correctly if a pinned load now fails. No COVERAGE_MATRIX
score row changes (this isn't a detector signal).

---

## 5. HARVESTER AUDIT — **N/A**
N/A — this is a dependency-pin / loader-kwargs supply-chain change; there is no attack
corpus to harvest, decontaminate, or train on.

---

## 6. Test plan — Code + Use-Case (APPLICABLE, reframed per scope)

New test file: **`tests/integrity/test_hf_loading.py`** (mirrors `src/na0s/integrity/`; create
`tests/integrity/__init__.py` if absent). Plus targeted edits to existing
`tests/ml/test_promptguard.py`, `tests/ml/test_promptguard_classifier.py`,
`tests/ml/test_late_chunking.py`. **Tests must NOT hit the network or real HF** (CLAUDE.md /
no-API-key memory) — mock `from_pretrained` / `SentenceTransformer` with `unittest.mock`.

### Code-level (the kwargs are actually passed)
1. **C1 — from_pretrained kwargs:** patch `AutoModelForSequenceClassification.from_pretrained`
   with a `MagicMock`, trigger `_ensure_loaded()`, assert the call kwargs contain
   `use_safetensors=True` and `revision=<pinned>` for the default model. Repeat for the
   tokenizer and for `promptguard_classifier`, `late_chunking`.
2. **C2 — ST kwargs:** patch `SentenceTransformer`, trigger each ST loader
   (`embedding_classifier`, `embedding_adapter`, `worm/detector`, `embedding_drift`,
   `predict_embedding`), assert `revision=<pinned>` is forwarded.
3. **C3 — helper contract:** `hf_from_pretrained_kwargs` returns `use_safetensors=True` for a
   known model; for an UNKNOWN/env-overridden model it still returns
   `use_safetensors=True` but omits `revision` and emits exactly one `logger.warning`
   (assert via `caplog`). No hollow asserts — check the dict keys/values explicitly.
4. **C4 — pin integrity:** assert no `PINNED_REVISIONS` value is a placeholder
   (`!= "main"`, matches a 40-hex-char SHA or a documented immutable tag) — guards against a
   future regression that re-points to a moving ref.

### Use-Case / behavior (Step 6 reframed: integrity change is FP-safe + non-crashing)
5. **U1 — legit load still works:** with mocked loaders returning a fake model, the full
   `_ensure_loaded()` / classifier path returns a usable object and the layer reports
   available. Proves the pin doesn't break the happy path.
6. **U2 — tampered/unsafetensors load is rejected gracefully:** make the mocked
   `from_pretrained` raise the error `transformers` raises when `use_safetensors=True` but no
   safetensors exists (e.g. `OSError`/`ValueError`); assert the layer sets `_init_failed`
   / `_available=False` and that `scan()` / `predict()` on a benign + a malicious sample
   **still returns a normal `ScanResult`** (no exception propagates). This is the
   "tampered file rejected, scan still works" use-case.
7. **U3 — full scan FP-safety:** run `predict()` on a small benign batch with the optional
   layers force-disabled vs. mock-enabled; assert benign verdicts are unchanged (pin change
   must not alter scores → zero FP/score drift, since we only changed *how* the model is
   fetched, not the model).
8. **U4 — CLI smoke (review checklist):** run the real CLI on one benign + one injection
   string (`python3 -m na0s.cli` or `na0s` entrypoint) and confirm it exits 0 / produces a
   verdict with transformers absent (degraded path) — proves the import-guarded helper
   doesn't break the keyless core.

### Suite gate
9. **S1 — full suite:** `python3 -m pytest tests/ -q --tb=line` → zero net regressions vs.
   the pre-change baseline (CLAUDE.md mandates a full run; ~15 min). Run targeted
   `python3 -m pytest tests/integrity/ tests/ml/ -v` first.

> Note: transformers / sentence-transformers are **not installed in the dev env** (verified —
> both imports fail). So the load bodies are dormant locally; tests rely on mocking the
> import-guarded symbols. This matches existing `tests/ml/` patterns (importorskip / mock).

---

## 7. Cleanup / refactor (Q2) — APPLICABLE (light)

- Centralizing the 9 call sites onto one helper **is** the refactor — removes future copy-paste
  drift. No dead code introduced.
- Repo hygiene: the working tree has stray scratch artifacts (`_skeptic_test_out.txt`,
  `_xfail_run.txt`, `pyt_out.txt`, `logs/`) from prior runs — out of scope for this item but
  note them; do not commit them with this change (scope every `git add`).
- New code lands on a dedicated branch `hardening/hf-transformers-pin` (per branch-naming
  convention), NOT the current `hardening/rag-poison-wiring` branch.

---

## 8. Roadmap / README / Benchmark updates (Steps 8-9)

- **ROADMAP_V2.md:** add (or locate the integrity/supply-chain section) a checked item:
  `[ ] M2-supply — pin transformers floor (CVE-2026-4372) + revision= + use_safetensors on
  all 9 HF loads`. Note: the existing `M2` token at `ROADMAP_V2.md:1474` is the unrelated
  "4-layer funnel" judge item and at `:1275` Category M2=Audio taxonomy — **do not collide**;
  use a distinct label (e.g. `SUP-M2` / `INTEG-HF-PIN`) to avoid the duplicate-ID smell.
  Check the box + cite the commit SHA once landed (Roadmap-Todo Sync memory rule).
- **README / SECURITY.md:** add one line under supply-chain hardening that HF artifacts are
  revision-pinned + safetensors-only. Only if a relevant section exists; do not create docs.
- **Benchmark:** no metric change expected (same model, same scores). Record in the PR that
  benchmark numbers are unchanged (the pin only affects *fetch*, not inference).

---

## 9. Taxonomy / Coverage Matrix / Scorer (Step 10, Q6, Q7) — PARTIAL / N/A

- **Q6 Taxonomy + Coverage Matrix:** PARTIAL/N/A — this is supply-chain integrity, not a
  prompt-injection technique, so it has no `data/taxonomy.yaml` code and no COVERAGE_MATRIX
  recall row. The only Step-10 action: avoid the `M2` label collision called out in Step 8.
- **Q7 Scorer thresholds:** N/A — no detector score is produced; the one number we introduce
  (the `transformers` version floor) is justified by the CVE advisory, not a tuned threshold.

---

## 10. Q&A self-check (instantiated)

- **Q1 — Can Na0S handle the threat + suite green?** Not yet: floor unpinned, no revision, no
  safetensors → all three CVE-class exposures open. Fix per Step 3, then full suite (S1).
- **Q2 — Cleanup?** Light refactor (centralize loader kwargs); see Step 7.
- **Q3 — Pipeline wiring correct?** Yes — predict.py + cascade.py inherit via the shared
  loaders; verify auto-disable counter still trips (Step 4).
- **Q4 — Code AND use-case tested?** Yes — C1-C4 (code) + U1-U4 (behavior) + S1 (suite).
- **Q5 — Harvester audit?** N/A (Step 5).
- **Q6 — Taxonomy/Coverage?** Partial/N/A — label-collision check only (Step 9).
- **Q7 — Scorer?** N/A (Step 9).
- **Q8 — predict/cascade references?** YES — `predict.py:183-184,194,899`,
  `cascade.py:57-58,100,124,806`; route through hardened loaders (Step 4).
- **Q9 — Harvester agent harvests this type?** N/A — not an attack class.
- **Q10 — Other checks:**
  - Confirm `SentenceTransformer.__init__` actually accepts `revision=` at the pinned ST floor
    (`>=2.2`). If an early ST version doesn't, bump the ST floor or branch the helper. **Do not
    pass an unsupported kwarg** (review checklist: no hallucinated API).
  - Confirm the real CVE-2026-4372 fixed version via NVD/GHSA before writing the floor.
  - Confirm current HF `use_safetensors` default for the pinned transformers version (the
    default flipped historically); keep it explicit regardless.
  - Air-gap escape hatch: `NA0S_HF_REVISION*` env override so local mirrors / pre-downloaded
    caches (e.g. worm detector's `_is_model_cached` path) aren't bricked by a pinned SHA.

---

## 11. Agent / skill team (inject `na0s-review-checklist` into every prompt)

| Step | Owner agent / skill | Mandate |
|---|---|---|
| 1-2 explore + threat-model + CVE/HF-default verification | **security-research-auditor** + skill **security-review**; `WebSearch`/`WebFetch` NVD+GHSA+HF docs | Confirm fixed `transformers` version, HF `use_safetensors` default, ST `revision` support. Record provenance. |
| 3 implementation (helper + 9 sites + pyproject + reqs) | **l3-l5-code-auditor** (owns ml/L5 loaders) + skill **detector-authoring** (wiring discipline only) | Centralize on `integrity/hf_loading.py`; pass kwargs at all 9 sites behind feature flags. |
| 4 predict/cascade parity | **l3-l5-code-auditor** | Verify both entrypoints inherit hardening; auto-disable counter intact. |
| graceful-failure audit | **silent-failure-hunter** | Ensure a pin/safetensors miss disables the layer and never crashes `scan()`/`predict()` (U2). |
| 6 tests | **l3-l5-code-auditor** + skill **na0s-debugging** (mock/importorskip patterns) | C1-C4 + U1-U4, no hollow asserts, no network. |
| 6 suite gate / CI | skill **eval-harness** + **github-ci-fix** | Full suite green; lock `requirements-benchmark.txt`. |
| 8 roadmap/readme | **Plan** + creative-writer (only if README prose needed) | Avoid `M2` label collision; cite SHA. |
| 11 PR | skill **github-pr-prep** then **pr-review-toolkit:review-pr** / **github-pr-review** | Held-out tests must pass before merge. |
| layer16 site sanity | **layer-9-11-auditor** (adjacent layer owner) — light review of `embedding_drift.py` change | Confirm conversation-layer load path unaffected. |

---

## Execution preconditions / dependencies

- **Depends-on: none.** This item is self-contained — it touches dependency metadata and
  loader kwargs only, with no reliance on other hardening items.
- **Adjacent / non-blocking:** Item 8 (malicious-pickle / `safe_pickle`) is conceptually
  related (both are model-supply integrity) and touches `predict_embedding.py:195`'s
  `safe_load`, but the two changes are on different lines and do not conflict; they can land
  in either order.
- **Environment:** transformers + sentence-transformers are NOT installed in the dev env, so
  all tests must mock. Verify any rename/import against MAIN with
  `PYTHONPATH=<worktree>/src` (the editable install points at a stale checkout — env memory).
- **Before writing the version floor:** the CVE-2026-4372 fixed `transformers` version MUST be
  confirmed from the upstream advisory; `>=5.3.0` in the title is provisional.

## Definition of done

- [ ] CVE-2026-4372 fixed `transformers` version confirmed from NVD/GHSA; floor matches it.
- [ ] `transformers>=<FIXED>,<6` pinned in `pyproject.toml` `embedding` extra; `transformers==<FIXED>` added to `requirements-benchmark.txt`.
- [ ] `src/na0s/integrity/hf_loading.py` created with real (non-placeholder) pinned revision SHAs + provenance comments + env-override hook.
- [ ] All 9 load sites pass `use_safetensors=True` (transformers) / `revision=` (transformers + ST): promptguard.py:184-187, promptguard_classifier.py:385-387, late_chunking.py:333-334, embedding_classifier.py:356, embedding_adapter.py:320, worm/detector.py:417, layer16/detectors/embedding_drift.py:152, predict_embedding.py:192.
- [ ] `SentenceTransformer(revision=)` support confirmed at the pinned ST floor; no unsupported kwarg passed.
- [ ] Graceful failure verified: a pinned/safetensors-violating load disables the optional layer; `scan()`/`predict()` never raises (U2).
- [ ] Tests added: `tests/integrity/test_hf_loading.py` (C1-C4, U1-U4) + edits to `tests/ml/` loaders; all mock, none hit network.
- [ ] CLI smoke passes with transformers absent (degraded keyless path).
- [ ] `python3 -m pytest tests/ -q --tb=line` → zero net regressions vs. baseline.
- [ ] ROADMAP_V2.md item added with distinct label (no `M2` collision) + commit SHA cited.
- [ ] PR opened via github-pr-prep; held-out tests green before merge; benchmark numbers confirmed unchanged.
