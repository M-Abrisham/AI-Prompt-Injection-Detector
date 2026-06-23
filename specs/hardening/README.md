# Na0S Pickle / Supply-Chain Integrity Hardening — Spec Index

18 grounded implementation specs, one per todo item, authored from the general orchestration
prompt. Each spec instantiates the 11-step pipeline + Q&A, marking non-applicable steps
`N/A — <reason>` (these are integrity-hardening tasks, not prompt-injection attack classes,
so harvester/taxonomy/coverage/scorer steps are mostly N/A — except #17).

> Status: **specs authored (plan-only)**. No source modified yet. Execution is batched into
> PRs by dependency wave (below).

## Items

| # | Spec | Tier | Depends on | Root cause (one line) |
|---|------|------|-----------|------------------------|
| 1 | [deploy-model-known-hashes-drop](01-deploy-model-known-hashes-drop.md) | NOW | — | `deploy()` rebuilds `KNOWN_HASHES` from copied files only → erases `model_embedding.pkl` (its sole integrity source) via the `re.sub` at deploy_model.py:165-170. |
| 2 | [distill-model-bare-pickle-fallback](02-distill-model-bare-pickle-fallback.md) | NOW | — | `distill_model.py:254-267` falls back to bare `pickle.load`/`dump` (no integrity, no sidecar); also dead code since the import always resolves. |
| 3 | [hf-transformers-pin-revision](03-hf-transformers-pin-revision.md) | NOW | — | 9 HF load sites with no `transformers` floor (unpinned/transitive), no `revision=`, no `use_safetensors=True` → CVE-2026-4372 + moving-tag + pickle-RCE. |
| 4 | [toctou-allowlist-unpickler](04-toctou-allowlist-unpickler.md) | CORE | — | `safe_load` opens the file 3× (hashed bytes ≠ executed bytes, CWE-367) and ends in unrestricted `pickle.load` (no `find_class` allowlist). |
| 5 | [route-raw-loaders-safeload](05-route-raw-loaders-safeload.md) | CORE | — | 3 loaders (faiss:219, stacking:130, embedding_adapter:437 torch.load) deserialize artifacts with zero integrity check. |
| 6 | [fail-closed-optional-loaders](06-fail-closed-optional-loaders.md) | CORE | — | 3 optional loaders wrap `safe_load` in bare `except Exception` → swallow integrity `ValueError` → tampered optional `.pkl` fails OPEN. |
| 7 | [sidecar-resolution-rework](07-sidecar-resolution-rework.md) | CORE | 6 (soft) | `_resolve_expected_hash` picks tier by file-present, not config → keyed load accepts forgeable `.sha256` (downgrade); keyless load lets dropped `.hmac` veto valid `.sha256` (DoS). |
| 8 | [l11-adversarial-stress-tests](08-l11-adversarial-stress-tests.md) | TEST | **4** | No `__reduce__`-rejection mechanism exists yet, no precedence/stress tests. |
| 9 | [sidecar-key-validation](09-sidecar-key-validation.md) | HARDEN | — | `_parse_sidecar` accepts any digest (no 64-hex check); `_get_signing_key` accepts 1-char/whitespace keys. |
| 10 | [env-model-id-allowlist](10-env-model-id-allowlist.md) | HARDEN | **3** | `NA0S_PROMPTGUARD_MODEL` env value flows verbatim into `from_pretrained` (model-source injection). |
| 11 | [ci-security-gate](11-ci-security-gate.md) | CI | **5, 6** | SAST tools present but none gate the build (bandit pre-commit-only/empty config, ruff S off, CodeQL default suite, no fickling/modelscan, security-review needs absent key). |
| 12 | [worm-3tier-integrity](12-worm-3tier-integrity.md) | HARDEN | — | Worm corpus model uses a hand-rolled single-tier `.sha256` joblib gate instead of the canonical 3-tier `safe_load`. |
| 13 | [format-migration-skops-safetensors](13-format-migration-skops-safetensors.md) | BIG | **5** | Every persisted artifact is pickle → digest gate is the only RCE barrier; migrate to skops/.npz/safetensors (`allow_pickle=False`). |
| 14 | [deploy-model-ast-rewrite](14-deploy-model-ast-rewrite.md) | CLEANUP | **1** | Brace-fragile `re.sub` (158-170) emits invalid Python on any nested `}`; no parse-verify. |
| 15 | [hash-cache-bound](15-hash-cache-bound.md) | CLEANUP | **7** | Module-global digest caches: unbounded, coarse mtime-second key, no lock (hygiene, not security). |
| 16 | [externalize-integrity-knobs](16-externalize-integrity-knobs.md) | CLEANUP | soft | 4 integrity knobs are scattered module literals, not centralized in config.py. |
| 17 | [di-decontam-retrain](17-di-decontam-retrain.md) | DATA | **1, 14** | Bundled weights were fit on a corpus that included eval sets (train-on-test); DI-1/2/5 done on main, only **DI-3 retrain** open. |
| 18 | [roadmap-doc-updates](18-roadmap-doc-updates.md) | DOC | soft (all) | L11 mislabeled "24/24 COMPLETE" despite 12 open specs; stale rag_poison + specs/01:16 + memory refs. |

## Execution waves (dependency-ordered)

- **Wave 1 — no deps (parallel):** #1, #2, #3, #4, #5, #6, #9, #12
- **Wave 2 — deps on Wave 1:** #7 (←6), #8 (←4), #10 (←3), #11 (←5,6), #14 (←1)
- **Wave 3:** #13 (←5), #15 (←7), #16
- **Wave 4:** #17 (←1,14; **needs main + DVC corpus**), #18 (docs last, cites commit SHAs)

## Cross-cutting blockers found during spec authoring (READ BEFORE EXECUTING)

1. **#4 — the unpickler hook does not exist on this branch.** `_NumpyCompatUnpickler`/`find_class`
   live only on unpushed branch `ci/test-optional-dep-guards @ 91944d6`, **not** on
   `hardening/rag-poison-wiring`. The restricted unpickler must be built from scratch here (or
   that branch merged first). #8 hard-depends on #4 and must `skipif`-guard until #4 lands.
2. **#5 — `safe_load` is pickle-specific** (rejects torch zip files). `embedding_adapter.py:437`
   (`torch.load`) needs a new torch-aware `verify_file_digest` helper + `weights_only=True`, not a
   direct `safe_load` call.
3. **#12 — `safe_load` cannot read a `joblib.dump` file** (empirically `UnpicklingError`); the worm
   model must migrate joblib → plain-pickle `.pkl`.
4. **Missing deps:** `skops` + `safetensors` (#13) and `libcst` (#14) are **not installed**. #14
   falls back to stdlib `ast` (fine); #13 must add the two libs to `pyproject.toml` behind import
   guards.
5. **Behavior-changing test updates (need explicit sign-off — these are corrections, not
   weakening):** #7 updates `test_key_set_but_sha256_sidecar_warns` (warn→fail-closed); #9 corrects
   a 16-hex fixture to 64-hex and sets `_MIN_PICKLE_KEY_LEN=8` (not 16) to avoid breaking ~15 tests;
   #10 tightens a test that currently accepts an arbitrary env id.
6. **#17 cannot fully run locally:** the training corpus is DVC-tracked and not on disk; DI-3 needs
   `dvc pull` or the `auto-retrain.yml` CI. Must base off **main** (DI-1/2/5 already landed there),
   not the contaminated primary checkout.
7. **Branch base:** several specs require basing off `main`. Recommend a fresh worktree off `main`
   for the whole effort (per multi-agent worktree discipline — never branch-switch the primary
   checkout).
